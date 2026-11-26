#!/usr/bin/env python3
"""Compara o experimento encerrado de Granger condicional."""

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
    configuracao_sha256: str
    banco_bivariado: Path
    resumo: dict[str, int]
    redes: dict[str, tuple[str, ...]]
    resultados: dict[ChaveRelacao, Resultado]
    significativas_bivariadas_globais: set[ChaveRelacao]
    significativas_bivariadas_familia_equivalente: set[ChaveRelacao]


def abrir_somente_leitura(caminho: Path) -> sqlite3.Connection:
    if not caminho.is_file():
        raise FileNotFoundError(f"banco não encontrado: {caminho}")
    conexao = sqlite3.connect(
        f"{caminho.resolve().as_uri()}?mode=ro",
        uri=True,
    )
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


def corrigir_benjamini_hochberg(
    p_valores: dict[ChaveRelacao, float],
    alpha: float = 0.05,
) -> set[ChaveRelacao]:
    if not p_valores:
        return set()
    ordenados = sorted(p_valores.items(), key=lambda item: item[1])
    n = len(ordenados)
    maior_indice = 0
    for indice, (_, p_valor) in enumerate(ordenados, start=1):
        if p_valor <= alpha * indice / n:
            maior_indice = indice
    return {chave for chave, _ in ordenados[:maior_indice]}


def carregar_resultados_bivariados(
    caminho: Path,
) -> tuple[set[ChaveRelacao], dict[ChaveRelacao, float]]:
    with abrir_somente_leitura(caminho) as conexao:
        linhas = conexao.execute(
            """
            SELECT produto, origem, destino, p_valor, significativo_fdr
            FROM resultados_granger
            WHERE status = 'testado'
            """
        ).fetchall()
        significativas = {
            (linha["produto"], linha["origem"], linha["destino"])
            for linha in linhas
            if bool(linha["significativo_fdr"])
        }
        p_valores = {
            (linha["produto"], linha["origem"], linha["destino"]): float(
                linha["p_valor"]
            )
            for linha in linhas
        }
        return significativas, p_valores


def carregar_execucao(contexto: str, caminho: Path) -> Execucao:
    with abrir_somente_leitura(caminho) as conexao:
        if conexao.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"{caminho}: falha de integridade")
        execucao = conexao.execute(
            """
            SELECT banco_origem, configuracao_redes_sha256, status
            FROM execucoes_condicionais
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
        if execucao is None or execucao["status"] != "concluida":
            raise RuntimeError(f"{caminho}: execução condicional não concluída")

        resumo = {
            linha["chave"]: int(linha["valor"])
            for linha in conexao.execute(
                "SELECT chave, valor FROM resumo_condicional"
            )
        }
        redes = {
            linha["produto"]: tuple(json.loads(linha["mercados_json"]))
            for linha in conexao.execute(
                """
                SELECT produto, mercados_json
                FROM redes_produtos
                WHERE status = 'testada'
                """
            )
        }
        resultados: dict[ChaveRelacao, Resultado] = {}
        for linha in conexao.execute(
            """
            SELECT resultado.produto, resultado.origem, resultado.destino,
                   rede.lag_selecionado, resultado.significativo_fdr,
                   diagnostico.sem_autocorrelacao_irrestrito,
                   avaliacao.melhora_rmse, avaliacao.melhora_mae,
                   avaliacao.melhora_rmse_altas,
                   avaliacao.melhora_mae_altas,
                   avaliacao.relacao_robusta,
                   avaliacao.relacao_alta_robusta
            FROM resultados_condicionais AS resultado
            JOIN redes_produtos AS rede ON rede.id = resultado.rede_id
            JOIN diagnosticos_residuos AS diagnostico
              ON diagnostico.resultado_id = resultado.id
            JOIN avaliacoes_preditivas AS avaliacao
              ON avaliacao.resultado_id = resultado.id
            """
        ):
            chave = (linha["produto"], linha["origem"], linha["destino"])
            resultados[chave] = Resultado(
                lag=int(linha["lag_selecionado"]),
                significativo_fdr=bool(linha["significativo_fdr"]),
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
    significativas_globais, p_valores = carregar_resultados_bivariados(
        banco_bivariado
    )
    p_valores_familia = {
        chave: p_valor
        for chave, p_valor in p_valores.items()
        if chave in resultados
    }
    return Execucao(
        contexto=contexto,
        caminho=caminho,
        configuracao_sha256=execucao["configuracao_redes_sha256"],
        banco_bivariado=banco_bivariado,
        resumo=resumo,
        redes=redes,
        resultados=resultados,
        significativas_bivariadas_globais=significativas_globais,
        significativas_bivariadas_familia_equivalente=(
            corrigir_benjamini_hochberg(p_valores_familia)
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
    bivariadas_globais = execucao.significativas_bivariadas_globais & testadas
    bivariadas = execucao.significativas_bivariadas_familia_equivalente
    condicionais = chaves_com_flag(execucao, "significativo_fdr")
    robustas = chaves_com_flag(execucao, "relacao_robusta")
    altas_robustas = chaves_com_flag(execucao, "relacao_alta_robusta")
    return {
        "resumo": execucao.resumo,
        "redes_testadas": {
            produto: list(mercados)
            for produto, mercados in sorted(execucao.redes.items())
        },
        "comparacao_bivariado": {
            "significativas_bivariadas_globais_testaveis": len(
                bivariadas_globais
            ),
            "significativas_bivariadas_familia_equivalente": len(bivariadas),
            "retidas_apos_condicionamento": len(bivariadas & condicionais),
            "removidas_apos_condicionamento": len(bivariadas - condicionais),
            "novas_apos_condicionamento": len(condicionais - bivariadas),
            "retidas": serializar_chaves(bivariadas & condicionais),
            "removidas": serializar_chaves(bivariadas - condicionais),
            "novas": serializar_chaves(condicionais - bivariadas),
        },
        "relacoes_robustas": serializar_chaves(robustas),
        "relacoes_altas_robustas": serializar_chaves(altas_robustas),
    }


def comparar_execucoes(v1: Execucao, v2: Execucao) -> dict[str, Any]:
    if v1.configuracao_sha256 != v2.configuracao_sha256:
        raise ValueError("as validações não usam a mesma configuração de redes")
    significativas_v1 = chaves_com_flag(v1, "significativo_fdr")
    significativas_v2 = chaves_com_flag(v2, "significativo_fdr")
    altas_v1 = chaves_com_flag(v1, "relacao_alta_robusta")
    altas_v2 = chaves_com_flag(v2, "relacao_alta_robusta")
    comuns = set(v1.resultados) & set(v2.resultados)
    lags_estaveis = {
        chave
        for chave in comuns
        if v1.resultados[chave].lag == v2.resultados[chave].lag
    }
    return {
        "gerado_em": datetime.now(UTC).isoformat(),
        "configuracao_redes_sha256": v1.configuracao_sha256,
        "validacoes": {
            "validacao_1": resumir_contexto(v1),
            "validacao_2": resumir_contexto(v2),
        },
        "estabilidade": {
            "direcoes_comuns": len(comuns),
            "lags_estaveis": len(lags_estaveis),
            "significativas_nas_duas": len(
                significativas_v1 & significativas_v2
            ),
            "jaccard_significativas": calcular_jaccard(
                significativas_v1,
                significativas_v2,
            ),
            "altas_robustas_nas_duas": len(altas_v1 & altas_v2),
            "jaccard_altas_robustas": calcular_jaccard(altas_v1, altas_v2),
            "relacoes_significativas_estaveis": serializar_chaves(
                significativas_v1 & significativas_v2
            ),
            "relacoes_altas_robustas_estaveis": serializar_chaves(
                altas_v1 & altas_v2
            ),
        },
    }


def exibir_relatorio(relatorio: dict[str, Any]) -> None:
    print("=== GRANGER CONDICIONAL ===")
    for contexto, dados in relatorio["validacoes"].items():
        resumo = dados["resumo"]
        comparacao = dados["comparacao_bivariado"]
        print(f"\n{contexto}")
        print(f"  produtos testados: {resumo['produtos_testados']}")
        print(f"  direções testadas: {resumo['direcoes_testadas']}")
        print(f"  significativas FDR: {resumo['significativas_fdr']}")
        print(
            "  bivariadas retidas/removidas/novas: "
            f"{comparacao['retidas_apos_condicionamento']}/"
            f"{comparacao['removidas_apos_condicionamento']}/"
            f"{comparacao['novas_apos_condicionamento']}"
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
