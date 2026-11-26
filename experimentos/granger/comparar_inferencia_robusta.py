#!/usr/bin/env python3
"""Compara a inferência clássica, HC3 e HAC nas duas validações."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experimentos.granger.executar_modelo_condicional import abrir_somente_leitura


ChaveRelacao = tuple[str, str, str]


@dataclass(frozen=True)
class Resultado:
    significativo_classico: bool
    significativo_hc3: bool
    significativo_hac: bool
    robusta_hc3: bool
    robusta_hac: bool
    alta_robusta_hc3: bool
    alta_robusta_hac: bool


@dataclass(frozen=True)
class Execucao:
    contexto: str
    caminho: Path
    assinatura: str
    resumo: dict[str, int]
    resultados: dict[ChaveRelacao, Resultado]


def assinatura_metodo(parametros: dict[str, Any]) -> str:
    chaves = (
        "alpha",
        "ffill_limite",
        "max_lag",
        "max_lag_amostra",
        "lag",
        "coeficientes",
        "teste_primario",
        "teste_sensibilidade",
        "correcao_multiplos_testes",
    )
    return json.dumps(
        {chave: parametros[chave] for chave in chaves},
        ensure_ascii=False,
        sort_keys=True,
    )


def carregar_execucao(contexto: str, caminho: Path) -> Execucao:
    with abrir_somente_leitura(caminho) as conexao:
        if conexao.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"falha de integridade: {caminho}")
        registro = conexao.execute(
            """
            SELECT parametros_json, status
            FROM execucoes_inferencia_robusta
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if registro is None or registro["status"] != "concluida":
            raise RuntimeError(f"execução robusta incompleta: {caminho}")
        resumo = {
            linha["chave"]: int(linha["valor"])
            for linha in conexao.execute(
                "SELECT chave, valor FROM resumo_inferencia_robusta"
            )
        }
        resultados = {
            (linha["produto"], linha["origem"], linha["destino"]): Resultado(
                significativo_classico=bool(linha["significativo_classico"]),
                significativo_hc3=bool(linha["significativo_hc3"]),
                significativo_hac=bool(linha["significativo_hac"]),
                robusta_hc3=bool(linha["relacao_hc3_robusta"]),
                robusta_hac=bool(linha["relacao_hac_robusta"]),
                alta_robusta_hc3=bool(linha["relacao_hc3_alta_robusta"]),
                alta_robusta_hac=bool(linha["relacao_hac_alta_robusta"]),
            )
            for linha in conexao.execute(
                """
                SELECT *
                FROM resultados_inferencia_robusta
                WHERE status = 'testado'
                """
            )
        }
    return Execucao(
        contexto=contexto,
        caminho=caminho,
        assinatura=assinatura_metodo(json.loads(registro["parametros_json"])),
        resumo=resumo,
        resultados=resultados,
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
    classicas = chaves_com_flag(execucao, "significativo_classico")
    hc3 = chaves_com_flag(execucao, "significativo_hc3")
    hac = chaves_com_flag(execucao, "significativo_hac")
    return {
        "resumo": execucao.resumo,
        "comparacao_classico_hc3": {
            "retidas": len(classicas & hc3),
            "removidas": len(classicas - hc3),
            "novas": len(hc3 - classicas),
        },
        "comparacao_classico_hac": {
            "retidas": len(classicas & hac),
            "removidas": len(classicas - hac),
            "novas": len(hac - classicas),
        },
    }


def comparar_execucoes(v1: Execucao, v2: Execucao) -> dict[str, Any]:
    if v1.assinatura != v2.assinatura:
        raise ValueError("as validações não usam a mesma inferência robusta")
    comuns = set(v1.resultados) & set(v2.resultados)
    significativas: dict[str, tuple[set[ChaveRelacao], set[ChaveRelacao]]] = {
        metodo: (
            chaves_com_flag(v1, f"significativo_{metodo}"),
            chaves_com_flag(v2, f"significativo_{metodo}"),
        )
        for metodo in ("classico", "hc3", "hac")
    }
    estabilidade = {
        metodo: {
            "significativas_nas_duas": len(primeiro & segundo),
            "jaccard": calcular_jaccard(primeiro, segundo),
            "relacoes": serializar_chaves(primeiro & segundo),
        }
        for metodo, (primeiro, segundo) in significativas.items()
    }
    robustas_hac = (
        chaves_com_flag(v1, "robusta_hac")
        & chaves_com_flag(v2, "robusta_hac")
    )
    altas_hac = (
        chaves_com_flag(v1, "alta_robusta_hac")
        & chaves_com_flag(v2, "alta_robusta_hac")
    )
    hc3_nas_duas = significativas["hc3"][0] & significativas["hc3"][1]
    candidatas = (robustas_hac | altas_hac) & hc3_nas_duas
    return {
        "gerado_em": datetime.now(UTC).isoformat(),
        "inclui_teste_final": False,
        "metodo": json.loads(v1.assinatura),
        "validacoes": {
            "validacao_1": resumir_contexto(v1),
            "validacao_2": resumir_contexto(v2),
        },
        "estabilidade": {
            "direcoes_comuns": len(comuns),
            **estabilidade,
            "robustas_hac_nas_duas": len(robustas_hac),
            "altas_robustas_hac_nas_duas": len(altas_hac),
            "candidatas_hac_confirmadas_hc3": serializar_chaves(candidatas),
        },
        "criterio_avanco": (
            "relação HAC robusta geral ou nas altas nas duas validações e "
            "também significativa por HC3 nos dois períodos"
        ),
        "recomendacao": (
            "avaliar_hac_no_efeito_domino"
            if candidatas
            else "manter_inferencia_classica"
        ),
    }


def exibir_relatorio(relatorio: dict[str, Any]) -> None:
    print("=== INFERÊNCIA ROBUSTA DO GRANGER ===")
    for contexto, dados in relatorio["validacoes"].items():
        print(f"\n{contexto}")
        for chave, valor in dados["resumo"].items():
            print(f"  {chave}: {valor}")
        comparacao = dados["comparacao_classico_hac"]
        print(
            "  HAC retidas/removidas/novas: "
            f"{comparacao['retidas']}/{comparacao['removidas']}/"
            f"{comparacao['novas']}"
        )
    print("\n=== ESTABILIDADE ENTRE VALIDAÇÕES ===")
    estabilidade = relatorio["estabilidade"]
    for metodo in ("classico", "hc3", "hac"):
        print(
            f"{metodo}: {estabilidade[metodo]['significativas_nas_duas']} "
            f"comuns | Jaccard={estabilidade[metodo]['jaccard']:.4f}"
        )
    print(f"robustas_hac_nas_duas: {estabilidade['robustas_hac_nas_duas']}")
    print(
        "altas_robustas_hac_nas_duas: "
        f"{estabilidade['altas_robustas_hac_nas_duas']}"
    )
    print(f"recomendacao: {relatorio['recomendacao']}")


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
    relatorio = comparar_execucoes(
        carregar_execucao("validacao_1", args.validacao_1),
        carregar_execucao("validacao_2", args.validacao_2),
    )
    exibir_relatorio(relatorio)
    gravar_relatorio(args.output_json, relatorio)
    print(f"\nRelatório gravado em: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
