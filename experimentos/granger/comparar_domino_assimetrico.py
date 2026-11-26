#!/usr/bin/env python3
"""Compara eventos do Efeito Dominó bivariado e assimétrico."""

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
class Execucao:
    caminho: Path
    tipo_relacoes: str
    banco_origem_sha256: str
    assinatura_configuracao: str
    resumo: dict[str, int]
    relacoes: set[ChaveRelacao]
    periodos: dict[str, dict[str, int | float | bool]]


def abrir_somente_leitura(caminho: Path) -> sqlite3.Connection:
    if not caminho.is_file():
        raise FileNotFoundError(f"banco não encontrado: {caminho}")
    conexao = sqlite3.connect(f"{caminho.resolve().as_uri()}?mode=ro", uri=True)
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA query_only = ON")
    return conexao


def proporcao(numerador: int, denominador: int) -> float:
    return numerador / denominador if denominador else 0.0


def resumir_periodo(linha: sqlite3.Row | None) -> dict[str, int | float | bool]:
    if linha is None:
        totais = {
            "eventos": 0,
            "classificaveis": 0,
            "confirmados": 0,
            "nao_confirmados": 0,
            "movimentos_simultaneos": 0,
            "sem_base_destino": 0,
            "sem_cobertura_destino": 0,
        }
    else:
        totais = {
            chave: int(linha[chave] or 0)
            for chave in (
                "eventos",
                "classificaveis",
                "confirmados",
                "nao_confirmados",
                "movimentos_simultaneos",
                "sem_base_destino",
                "sem_cobertura_destino",
            )
        }
    return {
        **totais,
        "cobertura_classificavel": proporcao(
            totais["classificaveis"],
            totais["eventos"],
        ),
        "proporcao_confirmada": proporcao(
            totais["confirmados"],
            totais["classificaveis"],
        ),
        "possui_duas_classes": (
            totais["confirmados"] > 0 and totais["nao_confirmados"] > 0
        ),
    }


def carregar_execucao(caminho: Path) -> Execucao:
    with abrir_somente_leitura(caminho) as conexao:
        if conexao.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"{caminho}: falha de integridade")
        execucao = conexao.execute(
            """
            SELECT banco_origem_sha256, parametros_json, status
            FROM execucoes_efeito_domino
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
        if execucao is None or execucao["status"] != "concluida":
            raise RuntimeError(f"{caminho}: execução não concluída")
        parametros = json.loads(execucao["parametros_json"])
        resumo = {
            linha["chave"]: int(linha["valor"])
            for linha in conexao.execute(
                "SELECT chave, valor FROM resumo_efeito_domino"
            )
        }
        relacoes = {
            (linha["produto"], linha["origem"], linha["destino"])
            for linha in conexao.execute(
                "SELECT produto, origem, destino FROM relacoes_analisadas"
            )
        }
        linhas_periodos = {
            linha["periodo"]: linha
            for linha in conexao.execute(
                """
                SELECT periodo,
                       COUNT(*) AS eventos,
                       SUM(efeito_domino IS NOT NULL) AS classificaveis,
                       SUM(efeito_domino = 1) AS confirmados,
                       SUM(efeito_domino = 0) AS nao_confirmados,
                       SUM(status = 'movimento_simultaneo')
                           AS movimentos_simultaneos,
                       SUM(status IN (
                           'sem_base_observada_destino',
                           'sem_preco_base_destino'
                       )) AS sem_base_destino,
                       SUM(status = 'sem_cobertura_destino')
                           AS sem_cobertura_destino
                FROM eventos_efeito_domino
                GROUP BY periodo
                """
            )
        }
    return Execucao(
        caminho=caminho,
        tipo_relacoes=parametros.get("tipo_relacoes", "granger_bivariado"),
        banco_origem_sha256=execucao["banco_origem_sha256"],
        assinatura_configuracao=json.dumps(
            {
                chave: parametros[chave]
                for chave in (
                    "alta_minima",
                    "percentil",
                    "janela_dias_uteis",
                    "regra",
                    "movimentos_simultaneos",
                    "eventos",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        resumo=resumo,
        relacoes=relacoes,
        periodos={
            periodo: resumir_periodo(linhas_periodos.get(periodo))
            for periodo in ("treino", "teste")
        },
    )


def serializar_relacoes(relacoes: set[ChaveRelacao]) -> list[dict[str, str]]:
    return [
        {"produto": produto, "origem": origem, "destino": destino}
        for produto, origem, destino in sorted(relacoes)
    ]


def comparar_periodo(
    referencia: dict[str, int | float | bool],
    assimetrico: dict[str, int | float | bool],
) -> dict[str, Any]:
    contagens = (
        "eventos",
        "classificaveis",
        "confirmados",
        "nao_confirmados",
        "movimentos_simultaneos",
        "sem_base_destino",
        "sem_cobertura_destino",
    )
    comparacao: dict[str, Any] = {
        "bivariado": referencia,
        "assimetrico": assimetrico,
    }
    for chave in contagens:
        valor_referencia = int(referencia[chave])
        valor_assimetrico = int(assimetrico[chave])
        comparacao[f"delta_{chave}"] = valor_assimetrico - valor_referencia
        comparacao[f"proporcao_{chave}_preservada"] = proporcao(
            valor_assimetrico,
            valor_referencia,
        )
    comparacao["delta_cobertura_classificavel"] = float(
        assimetrico["cobertura_classificavel"]
    ) - float(referencia["cobertura_classificavel"])
    comparacao["delta_proporcao_confirmada"] = float(
        assimetrico["proporcao_confirmada"]
    ) - float(referencia["proporcao_confirmada"])
    return comparacao


def comparar_contexto(
    contexto: str,
    bivariado: Execucao,
    assimetrico: Execucao,
) -> dict[str, Any]:
    if bivariado.tipo_relacoes != "granger_bivariado":
        raise ValueError(f"{contexto}: a referência não é bivariada")
    if assimetrico.tipo_relacoes != "granger_assimetrico":
        raise ValueError(f"{contexto}: a alternativa não é assimétrica")
    if bivariado.banco_origem_sha256 != assimetrico.banco_origem_sha256:
        raise ValueError(f"{contexto}: as execuções usam séries diferentes")
    if bivariado.assinatura_configuracao != assimetrico.assinatura_configuracao:
        raise ValueError(f"{contexto}: as execuções usam parâmetros diferentes")
    apta_modelagem = all(
        bool(assimetrico.periodos[periodo]["possui_duas_classes"])
        for periodo in ("treino", "teste")
    )
    return {
        "relacoes": {
            "bivariadas": len(bivariado.relacoes),
            "assimetricas": len(assimetrico.relacoes),
            "comuns": len(bivariado.relacoes & assimetrico.relacoes),
            "somente_bivariadas": serializar_relacoes(
                bivariado.relacoes - assimetrico.relacoes
            ),
            "somente_assimetricas": serializar_relacoes(
                assimetrico.relacoes - bivariado.relacoes
            ),
        },
        "periodos": {
            periodo: comparar_periodo(
                bivariado.periodos[periodo],
                assimetrico.periodos[periodo],
            )
            for periodo in ("treino", "teste")
        },
        "assimetrico_apto_para_base_modelagem": apta_modelagem,
    }


def comparar_execucoes(
    entradas: dict[str, tuple[Execucao, Execucao]],
) -> dict[str, Any]:
    contextos = {
        contexto: comparar_contexto(contexto, *execucoes)
        for contexto, execucoes in entradas.items()
    }
    return {
        "gerado_em": datetime.now(UTC).isoformat(),
        "inclui_teste_final": False,
        "criterio_gate_modelagem": (
            "a alternativa assimétrica deve preservar as duas classes nos "
            "períodos de treino e teste de cada validação"
        ),
        "contextos": contextos,
        "assimetrico_apto_nas_duas_validacoes": all(
            contexto["assimetrico_apto_para_base_modelagem"]
            for contexto in contextos.values()
        ),
    }


def exibir_relatorio(relatorio: dict[str, Any]) -> None:
    print("=== EFEITO DOMINÓ: BIVARIADO X ASSIMÉTRICO ===")
    for nome, contexto in relatorio["contextos"].items():
        relacoes = contexto["relacoes"]
        teste = contexto["periodos"]["teste"]
        print(f"\n{nome}")
        print(
            "  relações bivariadas/assimétricas/comuns: "
            f"{relacoes['bivariadas']}/{relacoes['assimetricas']}/"
            f"{relacoes['comuns']}"
        )
        print(
            "  eventos classificáveis no teste: "
            f"{teste['bivariado']['classificaveis']} -> "
            f"{teste['assimetrico']['classificaveis']}"
        )
        print(
            "  proporção confirmada no teste: "
            f"{teste['bivariado']['proporcao_confirmada']:.4f} -> "
            f"{teste['assimetrico']['proporcao_confirmada']:.4f}"
        )
        print(
            "  apto para base de modelagem: "
            f"{contexto['assimetrico_apto_para_base_modelagem']}"
        )
    print(
        "\nApto nas duas validações: "
        f"{relatorio['assimetrico_apto_nas_duas_validacoes']}"
    )


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
    for contexto in ("validacao-1", "validacao-2"):
        parser.add_argument(f"--{contexto}-bivariado", type=Path, required=True)
        parser.add_argument(f"--{contexto}-assimetrico", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    entradas = {
        "validacao_1": (
            carregar_execucao(args.validacao_1_bivariado),
            carregar_execucao(args.validacao_1_assimetrico),
        ),
        "validacao_2": (
            carregar_execucao(args.validacao_2_bivariado),
            carregar_execucao(args.validacao_2_assimetrico),
        ),
    }
    relatorio = comparar_execucoes(entradas)
    exibir_relatorio(relatorio)
    gravar_relatorio(args.output_json, relatorio)
    print(f"\nRelatório gravado em: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
