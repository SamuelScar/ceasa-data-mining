#!/usr/bin/env python3
"""Executa o Efeito Dominó com relações do Granger assimétrico."""

from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from efeitos.domino.executar_efeito_domino import (
    DEFAULT_ALTA_MINIMA,
    DEFAULT_JANELA_DIAS,
    DEFAULT_PERCENTIL,
    Configuracao,
    abrir_somente_leitura,
    calcular_limites,
    carregar_series,
    caracterizar_eventos,
    gerar_resumo,
    gravar_banco,
    sha256_arquivo,
    validar_banco_granger,
    valor_ambiente,
)


CRITERIO_RELACOES = (
    "status=testado, significativo_fdr=1 e resposta dinâmica positiva; "
    "métricas do período de avaliação não participam do filtro"
)


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "banco_granger",
        type=Path,
        help="banco bivariado que fornece as séries temporais",
    )
    parser.add_argument(
        "banco_assimetrico",
        type=Path,
        help="banco que fornece as relações assimétricas",
    )
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument(
        "--alta-minima",
        type=float,
        default=float(
            valor_ambiente("DOMINO_ALTA_MINIMA", str(DEFAULT_ALTA_MINIMA))
        ),
    )
    parser.add_argument(
        "--percentil",
        type=float,
        default=float(valor_ambiente("DOMINO_PERCENTIL", str(DEFAULT_PERCENTIL))),
    )
    parser.add_argument(
        "--janela-dias",
        type=int,
        default=int(
            valor_ambiente("DOMINO_JANELA_DIAS", str(DEFAULT_JANELA_DIAS))
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validar_argumentos(args: argparse.Namespace) -> Configuracao:
    for nome in ("banco_granger", "banco_assimetrico"):
        caminho = getattr(args, nome)
        if not caminho.is_file():
            raise FileNotFoundError(f"banco não encontrado: {caminho}")
    if args.output_db.exists() and not args.overwrite:
        raise FileExistsError(
            f"a saída já existe: {args.output_db}; use --overwrite para substituí-la"
        )
    if args.output_db.resolve() in {
        args.banco_granger.resolve(),
        args.banco_assimetrico.resolve(),
    }:
        raise ValueError("a saída deve ser diferente dos bancos de entrada")
    if not 0 < args.alta_minima < 1:
        raise ValueError("--alta-minima deve estar entre 0 e 1")
    if not 0 < args.percentil < 1:
        raise ValueError("--percentil deve estar entre 0 e 1")
    if args.janela_dias < 1:
        raise ValueError("--janela-dias deve ser positivo")
    return Configuracao(
        alta_minima=args.alta_minima,
        percentil=args.percentil,
        janela_dias=args.janela_dias,
    )


def validar_banco_assimetrico(
    conexao: sqlite3.Connection,
    hash_granger: str,
) -> sqlite3.Row:
    if conexao.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise RuntimeError("falha de integridade no banco assimétrico")
    execucao = conexao.execute(
        """
        SELECT id, schema_version, banco_origem_sha256, status
        FROM execucoes_assimetricas
        ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    if execucao is None or execucao["status"] != "concluida":
        raise ValueError("o banco assimétrico não possui execução concluída")
    if execucao["banco_origem_sha256"] != hash_granger:
        raise ValueError(
            "o banco assimétrico não foi produzido a partir do Granger informado"
        )
    return execucao


def carregar_relacoes_assimetricas(
    conexao: sqlite3.Connection,
) -> pd.DataFrame:
    relacoes = pd.read_sql_query(
        """
        SELECT resultado_granger_id, produto, origem, destino,
               lag_selecionado, estatistica_f, p_valor, q_valor_fdr
        FROM resultados_assimetricos
        WHERE status = 'testado'
          AND significativo_fdr = 1
          AND efeito_positivo = 1
        ORDER BY produto, origem, destino
        """,
        conexao,
    )
    if relacoes.empty:
        raise ValueError("não existem relações assimétricas positivas após FDR")
    relacoes.insert(0, "id", np.arange(1, len(relacoes) + 1, dtype=int))
    return relacoes


def executar(args: argparse.Namespace) -> Path:
    config = validar_argumentos(args)
    output = args.output_db.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    hash_granger = sha256_arquivo(args.banco_granger)

    print("[1/4] Validando os bancos de entrada...", flush=True)
    origem = abrir_somente_leitura(args.banco_granger)
    relacoes_origem = abrir_somente_leitura(args.banco_assimetrico)
    try:
        execucao_granger = validar_banco_granger(origem)
        execucao_assimetrica = validar_banco_assimetrico(
            relacoes_origem,
            hash_granger,
        )
        print("[2/4] Carregando relações assimétricas e séries...", flush=True)
        relacoes = carregar_relacoes_assimetricas(relacoes_origem)
        series = carregar_series(origem)
    finally:
        relacoes_origem.close()
        origem.close()

    print("[3/4] Calculando limites e caracterizando os eventos...", flush=True)
    limites = calcular_limites(series, config)
    eventos = caracterizar_eventos(series, relacoes, limites, config)
    resumo = gerar_resumo(relacoes, eventos)

    print("[4/4] Gravando e validando o banco de saída...", flush=True)
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
            args.banco_granger,
            execucao_granger,
            config,
            relacoes,
            limites,
            eventos,
            resumo,
            metadados_relacoes={
                "relacoes": CRITERIO_RELACOES,
                "tipo_relacoes": "granger_assimetrico",
                "banco_relacoes": str(args.banco_assimetrico),
                "banco_relacoes_sha256": sha256_arquivo(
                    args.banco_assimetrico
                ),
                "execucao_assimetrica_id": int(execucao_assimetrica["id"]),
            },
        )
        temporario.replace(output)
    except Exception:
        temporario.unlink(missing_ok=True)
        raise

    print(f"Banco assimétrico do Efeito Dominó criado em: {output}", flush=True)
    for chave, valor in sorted(resumo.items()):
        print(f"  {chave}: {valor:,}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    executar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
