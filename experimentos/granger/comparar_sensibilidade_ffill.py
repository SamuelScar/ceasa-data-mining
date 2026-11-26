#!/usr/bin/env python3
"""Compara o experimento encerrado de limites de preenchimento."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LIMITES_ESPERADOS = (0, 1, 3)
ChaveRelacao = tuple[str, str, str]


@dataclass(frozen=True)
class Resultado:
    status: str
    lag_selecionado: int | None
    significativo_fdr: bool
    residuos_adequados: bool
    melhora_rmse: bool
    melhora_mae: bool
    altas_avaliadas: bool
    melhora_rmse_altas: bool
    melhora_mae_altas: bool
    relacao_robusta: bool


@dataclass(frozen=True)
class Execucao:
    contexto: str
    caminho: Path
    ffill_limite: int
    banco_origem_sha256: str
    parametros: dict[str, Any]
    resultados: dict[ChaveRelacao, Resultado]


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


def carregar_execucao(contexto: str, caminho: Path) -> Execucao:
    with abrir_somente_leitura(caminho) as conexao:
        integridade = conexao.execute("PRAGMA quick_check").fetchone()[0]
        if integridade != "ok":
            raise RuntimeError(f"{caminho}: falha de integridade: {integridade}")

        registro = conexao.execute(
            """
            SELECT schema_version, banco_origem_sha256, parametros_json, status
            FROM execucoes_granger
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if registro is None or registro["status"] != "concluida":
            raise RuntimeError(f"{caminho}: execução de Granger não concluída")
        if int(registro["schema_version"]) < 3:
            raise RuntimeError(
                f"{caminho}: schema sem métricas específicas para altas"
            )

        parametros = json.loads(registro["parametros_json"])
        resultados: dict[ChaveRelacao, Resultado] = {}
        for linha in conexao.execute(
            """
            SELECT resultado.produto, resultado.origem, resultado.destino,
                   resultado.status, resultado.lag_selecionado,
                   resultado.significativo_fdr,
                   COALESCE(diagnostico.sem_autocorrelacao_irrestrito, 0)
                       AS residuos_adequados,
                   COALESCE(avaliacao.melhora_rmse, 0) AS melhora_rmse,
                   COALESCE(avaliacao.melhora_mae, 0) AS melhora_mae,
                   COALESCE(avaliacao.altas_avaliadas, 0) AS altas_avaliadas,
                   COALESCE(avaliacao.melhora_rmse_altas, 0)
                       AS melhora_rmse_altas,
                   COALESCE(avaliacao.melhora_mae_altas, 0)
                       AS melhora_mae_altas,
                   COALESCE(avaliacao.relacao_robusta, 0) AS relacao_robusta
            FROM resultados_granger AS resultado
            LEFT JOIN diagnosticos_residuos AS diagnostico
              ON diagnostico.resultado_id = resultado.id
            LEFT JOIN avaliacoes_preditivas AS avaliacao
              ON avaliacao.resultado_id = resultado.id
            ORDER BY resultado.produto, resultado.origem, resultado.destino
            """
        ):
            chave = (linha["produto"], linha["origem"], linha["destino"])
            resultados[chave] = Resultado(
                status=linha["status"],
                lag_selecionado=linha["lag_selecionado"],
                significativo_fdr=bool(linha["significativo_fdr"]),
                residuos_adequados=bool(linha["residuos_adequados"]),
                melhora_rmse=bool(linha["melhora_rmse"]),
                melhora_mae=bool(linha["melhora_mae"]),
                altas_avaliadas=bool(linha["altas_avaliadas"]),
                melhora_rmse_altas=bool(linha["melhora_rmse_altas"]),
                melhora_mae_altas=bool(linha["melhora_mae_altas"]),
                relacao_robusta=bool(linha["relacao_robusta"]),
            )

    return Execucao(
        contexto=contexto,
        caminho=caminho,
        ffill_limite=int(parametros["ffill_limite"]),
        banco_origem_sha256=registro["banco_origem_sha256"],
        parametros=parametros,
        resultados=resultados,
    )


def parametros_comparaveis(parametros: dict[str, Any]) -> dict[str, Any]:
    return {
        chave: valor
        for chave, valor in parametros.items()
        if chave != "ffill_limite"
    }


def validar_contexto(execucoes: list[Execucao]) -> list[Execucao]:
    ordenadas = sorted(execucoes, key=lambda execucao: execucao.ffill_limite)
    limites = tuple(execucao.ffill_limite for execucao in ordenadas)
    if limites != LIMITES_ESPERADOS:
        raise ValueError(
            f"as execuções devem usar ffill={LIMITES_ESPERADOS}; "
            f"encontrado: {limites}"
        )

    contextos = {execucao.contexto for execucao in ordenadas}
    if len(contextos) != 1:
        raise ValueError("as execuções comparadas devem pertencer ao mesmo contexto")

    referencia = ordenadas[0]
    parametros_referencia = parametros_comparaveis(referencia.parametros)
    chaves_referencia = set(referencia.resultados)
    for execucao in ordenadas[1:]:
        if execucao.banco_origem_sha256 != referencia.banco_origem_sha256:
            raise ValueError(
                f"{referencia.contexto}: bancos pré-processados diferentes"
            )
        if parametros_comparaveis(execucao.parametros) != parametros_referencia:
            raise ValueError(
                f"{referencia.contexto}: parâmetros não comparáveis"
            )
        if set(execucao.resultados) != chaves_referencia:
            raise ValueError(
                f"{referencia.contexto}: direções registradas diferentes"
            )
    return ordenadas


def conjunto_com_flag(
    execucao: Execucao,
    atributo: str,
) -> set[ChaveRelacao]:
    return {
        chave
        for chave, resultado in execucao.resultados.items()
        if bool(getattr(resultado, atributo))
    }


def calcular_jaccard(
    primeiro: set[ChaveRelacao],
    segundo: set[ChaveRelacao],
) -> float:
    uniao = primeiro | segundo
    return len(primeiro & segundo) / len(uniao) if uniao else 1.0


def resumir_execucao(execucao: Execucao) -> dict[str, int]:
    testados = [
        resultado
        for resultado in execucao.resultados.values()
        if resultado.status == "testado"
    ]
    significativos = [
        resultado for resultado in testados if resultado.significativo_fdr
    ]
    altas_avaliadas = [
        resultado for resultado in significativos if resultado.altas_avaliadas
    ]
    return {
        "direcoes_testadas": len(testados),
        "significativas_fdr": len(significativos),
        "residuos_adequados_fdr": sum(
            resultado.residuos_adequados for resultado in significativos
        ),
        "melhora_rmse_fdr": sum(
            resultado.melhora_rmse for resultado in significativos
        ),
        "melhora_mae_fdr": sum(
            resultado.melhora_mae for resultado in significativos
        ),
        "altas_avaliadas_fdr": len(altas_avaliadas),
        "melhora_rmse_altas_fdr": sum(
            resultado.melhora_rmse_altas for resultado in altas_avaliadas
        ),
        "melhora_mae_altas_fdr": sum(
            resultado.melhora_mae_altas for resultado in altas_avaliadas
        ),
        "relacoes_robustas": sum(
            resultado.relacao_robusta for resultado in significativos
        ),
    }


def comparar_contexto(execucoes: list[Execucao]) -> dict[str, Any]:
    ordenadas = validar_contexto(execucoes)
    chaves = sorted(ordenadas[0].resultados)
    significativos = [
        conjunto_com_flag(execucao, "significativo_fdr")
        for execucao in ordenadas
    ]
    robustos = [
        conjunto_com_flag(execucao, "relacao_robusta")
        for execucao in ordenadas
    ]
    alteracoes: list[dict[str, Any]] = []
    for chave in chaves:
        resultados = [execucao.resultados[chave] for execucao in ordenadas]
        assinatura = [
            (
                resultado.status,
                resultado.lag_selecionado,
                resultado.significativo_fdr,
                resultado.residuos_adequados,
                resultado.melhora_rmse,
                resultado.melhora_mae,
                resultado.altas_avaliadas,
                resultado.melhora_rmse_altas,
                resultado.melhora_mae_altas,
                resultado.relacao_robusta,
            )
            for resultado in resultados
        ]
        if len(set(assinatura)) == 1:
            continue
        alteracoes.append(
            {
                "produto": chave[0],
                "origem": chave[1],
                "destino": chave[2],
                "por_ffill": {
                    str(execucao.ffill_limite): {
                        "status": resultado.status,
                        "lag": resultado.lag_selecionado,
                        "significativo_fdr": resultado.significativo_fdr,
                        "residuos_adequados": resultado.residuos_adequados,
                        "melhora_rmse": resultado.melhora_rmse,
                        "melhora_mae": resultado.melhora_mae,
                        "altas_avaliadas": resultado.altas_avaliadas,
                        "melhora_rmse_altas": resultado.melhora_rmse_altas,
                        "melhora_mae_altas": resultado.melhora_mae_altas,
                        "relacao_robusta": resultado.relacao_robusta,
                    }
                    for execucao, resultado in zip(
                        ordenadas,
                        resultados,
                        strict=True,
                    )
                },
            }
        )

    return {
        "banco_origem_sha256": ordenadas[0].banco_origem_sha256,
        "configuracoes": {
            str(execucao.ffill_limite): resumir_execucao(execucao)
            for execucao in ordenadas
        },
        "estabilidade": {
            "direcoes": len(chaves),
            "significativas_em_todos_ffill": len(set.intersection(*significativos)),
            "significativas_em_algum_ffill": len(set.union(*significativos)),
            "robustas_em_todos_ffill": len(set.intersection(*robustos)),
            "robustas_em_algum_ffill": len(set.union(*robustos)),
            "direcoes_com_alguma_alteracao": len(alteracoes),
        },
        "alteracoes": alteracoes,
    }


def comparar_validacoes(
    validacao_1: list[Execucao],
    validacao_2: list[Execucao],
) -> dict[str, Any]:
    v1 = validar_contexto(validacao_1)
    v2 = validar_contexto(validacao_2)
    comparacao: dict[str, Any] = {}
    for execucao_v1, execucao_v2 in zip(v1, v2, strict=True):
        significativas_v1 = conjunto_com_flag(execucao_v1, "significativo_fdr")
        significativas_v2 = conjunto_com_flag(execucao_v2, "significativo_fdr")
        robustas_v1 = conjunto_com_flag(execucao_v1, "relacao_robusta")
        robustas_v2 = conjunto_com_flag(execucao_v2, "relacao_robusta")
        comparacao[str(execucao_v1.ffill_limite)] = {
            "significativas_nas_duas": len(significativas_v1 & significativas_v2),
            "jaccard_significativas": calcular_jaccard(
                significativas_v1,
                significativas_v2,
            ),
            "robustas_nas_duas": len(robustas_v1 & robustas_v2),
            "jaccard_robustas": calcular_jaccard(robustas_v1, robustas_v2),
        }
    return comparacao


def comparar_execucoes(
    validacao_1: list[Execucao],
    validacao_2: list[Execucao],
) -> dict[str, Any]:
    return {
        "gerado_em": datetime.now(UTC).isoformat(),
        "validacoes": {
            "validacao_1": comparar_contexto(validacao_1),
            "validacao_2": comparar_contexto(validacao_2),
        },
        "estabilidade_entre_validacoes": comparar_validacoes(
            validacao_1,
            validacao_2,
        ),
    }


def exibir_relatorio(relatorio: dict[str, Any]) -> None:
    print("=== SENSIBILIDADE AO PREENCHIMENTO ===")
    for contexto, resultado in relatorio["validacoes"].items():
        print(f"\n{contexto}")
        for limite, resumo in resultado["configuracoes"].items():
            print(f"  ffill={limite}: {resumo}")
        print(f"  estabilidade: {resultado['estabilidade']}")

    print("\n=== ESTABILIDADE ENTRE VALIDAÇÕES ===")
    for limite, resumo in relatorio["estabilidade_entre_validacoes"].items():
        print(f"ffill={limite}: {resumo}")


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
    parser.add_argument("--validacao-1", nargs=3, type=Path, required=True)
    parser.add_argument("--validacao-2", nargs=3, type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    validacao_1 = [
        carregar_execucao("validacao_1", caminho)
        for caminho in args.validacao_1
    ]
    validacao_2 = [
        carregar_execucao("validacao_2", caminho)
        for caminho in args.validacao_2
    ]
    relatorio = comparar_execucoes(validacao_1, validacao_2)
    exibir_relatorio(relatorio)
    gravar_relatorio(args.output_json, relatorio)
    print(f"\nRelatório gravado em: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
