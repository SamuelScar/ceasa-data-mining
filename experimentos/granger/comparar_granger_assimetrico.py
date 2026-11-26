#!/usr/bin/env python3
"""Compara o Granger assimétrico nas duas validações temporais."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ChaveRelacao = tuple[str, str, str]


@dataclass(frozen=True)
class Resultado:
    lag: int
    significativo_fdr: bool
    efeito_positivo: bool
    residuos_adequados: bool
    melhora_rmse: bool
    melhora_mae: bool
    melhora_rmse_altas: bool
    melhora_mae_altas: bool
    relacao_robusta: bool
    relacao_alta_robusta: bool


@dataclass(frozen=True)
class Execucao:
    contexto: str
    caminho: Path
    assinatura_metodo: str
    banco_bivariado: Path
    resumo: dict[str, int]
    resultados: dict[ChaveRelacao, Resultado]
    significativas_bivariadas: set[ChaveRelacao]


def abrir_somente_leitura(caminho: Path) -> sqlite3.Connection:
    if not caminho.is_file():
        raise FileNotFoundError(f"banco não encontrado: {caminho}")
    conexao = sqlite3.connect(f"{caminho.resolve().as_uri()}?mode=ro", uri=True)
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA query_only = ON")
    return conexao


def resolver_banco_origem(registrado: str) -> Path:
    caminho = Path(registrado)
    if caminho.is_file():
        return caminho
    candidato = Path.cwd() / caminho
    if candidato.is_file():
        return candidato
    raise FileNotFoundError(f"banco bivariado de origem não encontrado: {registrado}")


def carregar_significativas_bivariadas(caminho: Path) -> set[ChaveRelacao]:
    with abrir_somente_leitura(caminho) as conexao:
        return {
            (linha["produto"], linha["origem"], linha["destino"])
            for linha in conexao.execute(
                """
                SELECT produto, origem, destino
                FROM resultados_granger
                WHERE status = 'testado' AND significativo_fdr = 1
                """
            )
        }


def assinatura_metodo(parametros: dict[str, Any]) -> str:
    chaves = (
        "alpha",
        "ffill_limite",
        "max_lag",
        "max_lag_amostra",
        "selecao_lag",
        "correcao_multiplos_testes",
        "modelo_restrito",
        "modelo_irrestrito",
        "horizonte_resposta",
        "criterio_efeito",
    )
    return json.dumps(
        {chave: parametros[chave] for chave in chaves},
        ensure_ascii=False,
        sort_keys=True,
    )


def carregar_execucao(contexto: str, caminho: Path) -> Execucao:
    with abrir_somente_leitura(caminho) as conexao:
        if conexao.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"{caminho}: falha de integridade")
        execucao = conexao.execute(
            """
            SELECT banco_origem, parametros_json, status
            FROM execucoes_assimetricas
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
        if execucao is None or execucao["status"] != "concluida":
            raise RuntimeError(f"{caminho}: execução assimétrica não concluída")
        parametros = json.loads(execucao["parametros_json"])
        resumo = {
            linha["chave"]: int(linha["valor"])
            for linha in conexao.execute("SELECT chave, valor FROM resumo_assimetria")
        }
        resultados: dict[ChaveRelacao, Resultado] = {}
        for linha in conexao.execute(
            """
            SELECT resultado.produto, resultado.origem, resultado.destino,
                   resultado.lag_selecionado, resultado.significativo_fdr,
                   resultado.efeito_positivo,
                   diagnostico.sem_autocorrelacao_irrestrito,
                   avaliacao.melhora_rmse, avaliacao.melhora_mae,
                   avaliacao.melhora_rmse_altas,
                   avaliacao.melhora_mae_altas,
                   avaliacao.relacao_robusta,
                   avaliacao.relacao_alta_robusta
            FROM resultados_assimetricos AS resultado
            JOIN diagnosticos_residuos AS diagnostico
              ON diagnostico.resultado_id = resultado.id
            JOIN avaliacoes_preditivas AS avaliacao
              ON avaliacao.resultado_id = resultado.id
            WHERE resultado.status = 'testado'
            """
        ):
            chave = (linha["produto"], linha["origem"], linha["destino"])
            resultados[chave] = Resultado(
                lag=int(linha["lag_selecionado"]),
                significativo_fdr=bool(linha["significativo_fdr"]),
                efeito_positivo=bool(linha["efeito_positivo"]),
                residuos_adequados=bool(
                    linha["sem_autocorrelacao_irrestrito"]
                ),
                melhora_rmse=bool(linha["melhora_rmse"]),
                melhora_mae=bool(linha["melhora_mae"]),
                melhora_rmse_altas=bool(linha["melhora_rmse_altas"]),
                melhora_mae_altas=bool(linha["melhora_mae_altas"]),
                relacao_robusta=bool(linha["relacao_robusta"]),
                relacao_alta_robusta=bool(linha["relacao_alta_robusta"]),
            )

    banco_bivariado = resolver_banco_origem(execucao["banco_origem"])
    return Execucao(
        contexto=contexto,
        caminho=caminho,
        assinatura_metodo=assinatura_metodo(parametros),
        banco_bivariado=banco_bivariado,
        resumo=resumo,
        resultados=resultados,
        significativas_bivariadas=carregar_significativas_bivariadas(
            banco_bivariado
        ),
    )


def chaves_com_flag(execucao: Execucao, atributo: str) -> set[ChaveRelacao]:
    return {
        chave
        for chave, resultado in execucao.resultados.items()
        if bool(getattr(resultado, atributo))
    }


def serializar_chaves(chaves: set[ChaveRelacao]) -> list[dict[str, str]]:
    return [
        {"produto": produto, "origem": origem, "destino": destino}
        for produto, origem, destino in sorted(chaves)
    ]


def calcular_jaccard(
    primeiro: set[ChaveRelacao],
    segundo: set[ChaveRelacao],
) -> float:
    uniao = primeiro | segundo
    return len(primeiro & segundo) / len(uniao) if uniao else 1.0


def resumir_contexto(execucao: Execucao) -> dict[str, Any]:
    testadas = set(execucao.resultados)
    bivariadas = execucao.significativas_bivariadas & testadas
    assimetricas = chaves_com_flag(execucao, "significativo_fdr")
    positivas = {
        chave
        for chave in assimetricas
        if execucao.resultados[chave].efeito_positivo
    }
    robustas = chaves_com_flag(execucao, "relacao_robusta")
    altas_robustas = chaves_com_flag(execucao, "relacao_alta_robusta")
    return {
        "resumo": execucao.resumo,
        "comparacao_bivariado": {
            "significativas_bivariadas_testaveis": len(bivariadas),
            "retidas_pelo_assimetrico": len(bivariadas & assimetricas),
            "removidas_pelo_assimetrico": len(bivariadas - assimetricas),
            "novas_no_assimetrico": len(assimetricas - bivariadas),
        },
        "significativas_com_efeito_positivo": serializar_chaves(positivas),
        "relacoes_robustas": serializar_chaves(robustas),
        "relacoes_altas_robustas": serializar_chaves(altas_robustas),
    }


def comparar_execucoes(v1: Execucao, v2: Execucao) -> dict[str, Any]:
    if v1.assinatura_metodo != v2.assinatura_metodo:
        raise ValueError("as validações não usam o mesmo método assimétrico")
    comuns = set(v1.resultados) & set(v2.resultados)
    significativas_v1 = chaves_com_flag(v1, "significativo_fdr")
    significativas_v2 = chaves_com_flag(v2, "significativo_fdr")
    altas_v1 = chaves_com_flag(v1, "relacao_alta_robusta")
    altas_v2 = chaves_com_flag(v2, "relacao_alta_robusta")
    positivas_nas_duas = {
        chave
        for chave in comuns
        if v1.resultados[chave].efeito_positivo
        and v2.resultados[chave].efeito_positivo
    }
    lags_estaveis = {
        chave
        for chave in comuns
        if v1.resultados[chave].lag == v2.resultados[chave].lag
    }
    candidatas = altas_v1 & altas_v2
    return {
        "gerado_em": datetime.now(UTC).isoformat(),
        "metodo": json.loads(v1.assinatura_metodo),
        "validacoes": {
            "validacao_1": resumir_contexto(v1),
            "validacao_2": resumir_contexto(v2),
        },
        "estabilidade": {
            "direcoes_comuns": len(comuns),
            "lags_estaveis": len(lags_estaveis),
            "efeitos_positivos_nas_duas": len(positivas_nas_duas),
            "significativas_nas_duas": len(
                significativas_v1 & significativas_v2
            ),
            "jaccard_significativas": calcular_jaccard(
                significativas_v1,
                significativas_v2,
            ),
            "altas_robustas_nas_duas": len(candidatas),
            "jaccard_altas_robustas": calcular_jaccard(altas_v1, altas_v2),
            "relacoes_significativas_estaveis": serializar_chaves(
                significativas_v1 & significativas_v2
            ),
            "candidatas_efeito_domino": serializar_chaves(candidatas),
        },
    }


def exibir_relatorio(relatorio: dict[str, Any]) -> None:
    print("=== GRANGER ASSIMÉTRICO ===")
    for contexto, dados in relatorio["validacoes"].items():
        resumo = dados["resumo"]
        comparacao = dados["comparacao_bivariado"]
        print(f"\n{contexto}")
        print(f"  direções testadas: {resumo['direcoes_testadas']}")
        print(f"  significativas FDR: {resumo['significativas_fdr']}")
        print(f"  efeitos positivos FDR: {resumo['efeitos_positivos_fdr']}")
        print(
            "  bivariadas retidas/removidas/novas: "
            f"{comparacao['retidas_pelo_assimetrico']}/"
            f"{comparacao['removidas_pelo_assimetrico']}/"
            f"{comparacao['novas_no_assimetrico']}"
        )
        print(
            "  robustas gerais/altas: "
            f"{resumo['relacoes_robustas']}/"
            f"{resumo['relacoes_altas_robustas']}"
        )
    print("\n=== ESTABILIDADE ENTRE VALIDAÇÕES ===")
    for chave, valor in relatorio["estabilidade"].items():
        if not isinstance(valor, list):
            print(f"{chave}: {valor}")


def gravar_relatorio(caminho: Path, relatorio: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validacao-1", type=Path, required=True)
    parser.add_argument("--validacao-2", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    v1 = carregar_execucao("validacao_1", args.validacao_1)
    v2 = carregar_execucao("validacao_2", args.validacao_2)
    relatorio = comparar_execucoes(v1, v2)
    exibir_relatorio(relatorio)
    gravar_relatorio(args.output_json, relatorio)
    print(f"\nRelatório gravado em: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
