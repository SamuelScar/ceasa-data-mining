#!/usr/bin/env python3
"""Gera base de modelagem quando o pré-processado contextual foi descartado."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from modelagem.gerar_base_modelagem import (
    CONTEXTOS,
    JANELAS_HISTORICAS,
    MAX_LAG_RETORNO,
    ContextoTemporal,
    assinatura_metodologia,
    carregar_eventos,
    carregar_execucao,
    carregar_series,
    construir_catalogo,
    construir_eventos_modelagem,
    construir_particoes,
    gerar_resumo,
    gravar_banco,
    preparar_atributos_series,
    preparar_auditoria,
    sha256_arquivo,
    validar_configuracao_temporal,
    validar_periodos_eventos,
    validar_tabelas,
)


CHAVES_PROVENIENCIA_RELACOES = {
    "banco_relacoes",
    "banco_relacoes_sha256",
    "execucao_assimetrica_id",
}


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifesto_ffill",
        type=Path,
        help="manifesto que registra o pré-processado contextual descartado",
    )
    parser.add_argument("granger_db", type=Path)
    parser.add_argument("domino_db", type=Path)
    parser.add_argument("--contexto", choices=tuple(CONTEXTOS), required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def abrir_somente_leitura(caminho: Path) -> sqlite3.Connection:
    conexao = sqlite3.connect(
        f"{caminho.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA query_only = ON")
    return conexao


def normalizar_parametros(
    parametros_granger: dict[str, Any],
    parametros_domino: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    granger = dict(parametros_granger)
    domino = dict(parametros_domino)
    for chave in ("data_inicio", "data_corte_treino", "data_fim_teste"):
        granger.pop(chave, None)
    for chave in CHAVES_PROVENIENCIA_RELACOES:
        domino.pop(chave, None)
    return granger, domino


def construir_metodologia(
    execucao_granger: dict[str, Any],
    execucao_domino: dict[str, Any],
) -> dict[str, Any]:
    granger, domino = normalizar_parametros(
        json.loads(str(execucao_granger["parametros_json"])),
        json.loads(str(execucao_domino["parametros_json"])),
    )
    return {
        "schemas_fontes": {
            "preprocessamento": int(execucao_granger["schema_preprocessamento"]),
            "granger": int(execucao_granger["schema_version"]),
            "efeito_domino": int(execucao_domino["schema_version"]),
        },
        "preprocessamento": {
            "proveniencia": "manifesto do experimento de sensibilidade ao ffill",
            "banco_descartado_apos_granger": True,
            "uso_na_modelagem": (
                "fonte indireta; os atributos diarios usam as series "
                "materializadas e auditadas no banco de Granger"
            ),
        },
        "granger": granger,
        "efeito_domino": domino,
        "modelagem": {
            "janelas_historicas_datas_segunda_sexta": list(JANELAS_HISTORICAS),
            "max_lag_retorno": MAX_LAG_RETORNO,
            "unidade_analise": "produto, origem, destino e data do choque",
            "purga_fronteiras": "janela do rotulo deve terminar dentro do periodo",
        },
    }


def validar_argumentos(args: argparse.Namespace) -> None:
    for caminho in (args.manifesto_ffill, args.granger_db, args.domino_db):
        if not caminho.is_file():
            raise FileNotFoundError(f"entrada não encontrada: {caminho}")
    if args.output_db.exists() and not args.overwrite:
        raise FileExistsError(
            f"a saída já existe: {args.output_db}; use --overwrite para substituí-la"
        )
    if args.output_db.resolve() in {
        args.granger_db.resolve(),
        args.domino_db.resolve(),
    }:
        raise ValueError("a saída deve ser diferente dos bancos de entrada")


def carregar_proveniencia_preprocessamento(
    manifesto_path: Path,
    contexto: str,
) -> tuple[Path, dict[str, Any]]:
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    if manifesto.get("tipo") != "sensibilidade_ffill_granger":
        raise ValueError("o manifesto não pertence ao experimento de ffill")
    if manifesto.get("status") != "concluida":
        raise ValueError("o experimento de ffill não está concluído")
    if manifesto.get("inclui_teste_final") is not False:
        raise ValueError("o manifesto não protege o teste final")
    try:
        registro = manifesto["preprocessamentos_descartados"][contexto]
    except (KeyError, TypeError) as erro:
        raise ValueError(
            f"o manifesto não registra o pré-processamento de {contexto}"
        ) from erro
    if registro.get("descartado_apos_granger") is not True:
        raise ValueError("o pré-processamento não está marcado como descartado")
    hash_registrado = str(registro.get("sha256", ""))
    if len(hash_registrado) != 64:
        raise ValueError("hash inválido do pré-processamento descartado")
    raiz = manifesto_path.parent
    caminho = raiz / str(registro["caminho"])
    return caminho, dict(registro)


def validar_fontes_experimentais(
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[str, Path],
    ContextoTemporal,
    dict[str, Any],
]:
    preprocessamento_path, registro_preprocessamento = (
        carregar_proveniencia_preprocessamento(
            args.manifesto_ffill,
            args.contexto,
        )
    )

    granger = abrir_somente_leitura(args.granger_db)
    try:
        validar_tabelas(
            granger,
            {"execucoes_granger", "series_temporais", "resultados_granger"},
            "de Granger",
        )
        execucao_granger = carregar_execucao(
            granger,
            "execucoes_granger",
            "de Granger",
        )
    finally:
        granger.close()

    domino = abrir_somente_leitura(args.domino_db)
    try:
        validar_tabelas(
            domino,
            {
                "execucoes_efeito_domino",
                "eventos_efeito_domino",
                "relacoes_analisadas",
            },
            "do Efeito Dominó",
        )
        execucao_domino = carregar_execucao(
            domino,
            "execucoes_efeito_domino",
            "do Efeito Dominó",
        )
    finally:
        domino.close()

    hashes = {
        "preprocessamento": str(registro_preprocessamento["sha256"]),
        "granger": sha256_arquivo(args.granger_db),
        "domino": sha256_arquivo(args.domino_db),
    }
    if execucao_granger["banco_origem_sha256"] != hashes["preprocessamento"]:
        raise ValueError(
            "o hash do pré-processamento no Granger diverge do manifesto"
        )
    if int(execucao_granger["banco_origem_tamanho"]) != int(
        registro_preprocessamento["tamanho_bytes"]
    ):
        raise ValueError(
            "o tamanho do pré-processamento no Granger diverge do manifesto"
        )
    if execucao_domino["banco_origem_sha256"] != hashes["granger"]:
        raise ValueError(
            "o banco do Efeito Dominó não foi produzido pelo Granger informado"
        )
    if int(execucao_domino["execucao_granger_id"]) != int(execucao_granger["id"]):
        raise ValueError("o Efeito Dominó referencia outra execução de Granger")

    parametros_granger = json.loads(str(execucao_granger["parametros_json"]))
    parametros_domino = json.loads(str(execucao_domino["parametros_json"]))
    contexto = ContextoTemporal(
        inicio_series=CONTEXTOS[args.contexto].inicio_series,
        corte_treino=CONTEXTOS[args.contexto].corte_treino,
        fim_teste=CONTEXTOS[args.contexto].fim_avaliacao,
        janela_rotulo_dias=int(parametros_domino["janela_dias_uteis"]),
    )
    observado = (
        parametros_granger.get("data_inicio"),
        parametros_granger.get("data_corte_treino"),
        parametros_granger.get("data_fim_teste"),
    )
    esperado = (
        contexto.inicio_series.isoformat(),
        contexto.corte_treino.isoformat(),
        contexto.fim_teste.isoformat(),
    )
    if observado != esperado:
        raise ValueError(
            f"o banco de Granger registra o recorte {observado}, esperado {esperado}"
        )

    execucao_preprocessamento = {
        "id": int(execucao_granger["execucao_preprocessamento_id"]),
        "schema_version": int(execucao_granger["schema_preprocessamento"]),
    }
    execucoes = {
        "preprocessamento": execucao_preprocessamento,
        "granger": execucao_granger,
        "domino": execucao_domino,
    }
    entradas = {
        "preprocessamento": preprocessamento_path,
        "granger": args.granger_db,
        "domino": args.domino_db,
    }
    metodologia = construir_metodologia(execucao_granger, execucao_domino)
    return execucoes, hashes, entradas, contexto, metodologia


def executar(args: argparse.Namespace) -> Path:
    validar_argumentos(args)
    config = CONTEXTOS[args.contexto]
    output = args.output_db.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print("[1/6] Validando bancos e manifesto de proveniência...", flush=True)
    execucoes, hashes, entradas, contexto, metodologia = (
        validar_fontes_experimentais(args)
    )
    validar_configuracao_temporal(config, contexto)

    print("[2/6] Carregando eventos e séries temporais...", flush=True)
    eventos, auditoria = carregar_eventos(args.domino_db)
    validar_periodos_eventos(eventos, contexto)
    series = carregar_series(args.granger_db)

    print("[3/6] Construindo atributos disponíveis até cada choque...", flush=True)
    series = preparar_atributos_series(series)
    eventos_modelagem = construir_eventos_modelagem(eventos, series)
    auditoria = preparar_auditoria(
        auditoria,
        eventos_modelagem,
        contexto.janela_rotulo_dias,
    )

    print("[4/6] Construindo as divisões temporais com purga...", flush=True)
    particoes = construir_particoes(eventos_modelagem, config, contexto)
    catalogo = construir_catalogo()
    metodologia_sha256 = assinatura_metodologia(metodologia, catalogo)
    resumo = gerar_resumo(eventos_modelagem, particoes, config)

    print("[5/6] Gravando o banco temporário...", flush=True)
    descritor, nome_temporario = tempfile.mkstemp(
        prefix=f".{output.stem}-",
        suffix=".tmp.sqlite",
        dir=output.parent,
    )
    os.close(descritor)
    temporario = Path(nome_temporario)
    try:
        gravar_banco(
            temporario,
            entradas,
            execucoes,
            hashes,
            config,
            contexto,
            metodologia,
            metodologia_sha256,
            eventos_modelagem,
            auditoria,
            particoes,
            catalogo,
            resumo,
        )
        print("[6/6] Validando e publicando o banco final...", flush=True)
        temporario.replace(output)
    except Exception:
        temporario.unlink(missing_ok=True)
        raise

    print(f"Base experimental criada em: {output}", flush=True)
    for chave, valor in sorted(resumo.items()):
        print(f"  {chave}: {valor:,}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    executar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
