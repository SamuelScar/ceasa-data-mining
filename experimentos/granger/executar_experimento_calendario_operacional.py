"""Testa calendários operacionais derivados exclusivamente da base CEASA."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from experimentos.granger.executar_experimento_series_temporais import (
    Contexto,
    abrir_somente_leitura,
    calcular_jaccard,
    carregar_contexto,
    chave_relacao,
    combinar_resultados,
    caminho_contexto,
    relacao_robusta,
    relacoes_significativas,
    resumir_metodo,
    resumir_resultados,
    sha256_arquivo,
)
from granger.executar_granger import (
    MERCADOS_FIXOS,
    PRODUTOS_FIXOS,
    executar_testes_granger,
    identificar_segmentos,
)


FFILL_LIMITE = 3
MINIMO_DIAS_COMUNS = 400
ALPHA = 0.05
MAX_LAG = 10

ChaveRelacao = tuple[str, str, str]


@dataclass(frozen=True)
class Configuracao:
    nome: str
    cobertura_minima: float
    taxa_mudancas_minima: float


@dataclass(frozen=True)
class ResultadoOperacional:
    configuracao: Configuracao
    pares: pd.DataFrame
    resultados: pd.DataFrame


CONFIGURACOES = (
    Configuracao("calendario-operacional", 0.0, 0.0),
    Configuracao("calendario-operacional-com-qualidade", 0.75, 0.15),
)


def carregar_calendarios_operacionais(
    banco_bruto: Path,
    data_inicio: date,
    data_fim: date,
    mercados: tuple[str, ...] = MERCADOS_FIXOS,
) -> dict[str, pd.DatetimeIndex]:
    placeholders = ",".join("?" for _ in mercados)
    consulta = f"""
        SELECT entreposto.slug AS mercado, cotacao.data_cotacao
        FROM cotacoes AS cotacao
        JOIN entrepostos AS entreposto ON entreposto.id = cotacao.entreposto_id
        WHERE entreposto.slug IN ({placeholders})
          AND cotacao.data_cotacao BETWEEN ? AND ?
        GROUP BY entreposto.slug, cotacao.data_cotacao
        ORDER BY entreposto.slug, cotacao.data_cotacao
    """
    parametros = (*mercados, data_inicio.isoformat(), data_fim.isoformat())
    with abrir_somente_leitura(banco_bruto) as conexao:
        linhas = pd.read_sql_query(
            consulta,
            conexao,
            params=parametros,
            parse_dates=["data_cotacao"],
        )

    linhas = linhas[linhas["data_cotacao"].dt.day_of_week.lt(5)]
    calendarios = {
        mercado: pd.DatetimeIndex(
            linhas.loc[linhas["mercado"].eq(mercado), "data_cotacao"]
        ).sort_values()
        for mercado in mercados
    }
    ausentes = [mercado for mercado, calendario in calendarios.items() if calendario.empty]
    if ausentes:
        raise ValueError(
            "mercados sem calendário operacional: " + ", ".join(ausentes)
        )
    return calendarios


def precos_em_sessoes_ativas(
    series: pd.DataFrame,
    produto: str,
    mercado: str,
    calendario: pd.DatetimeIndex,
    limite_ffill: int = FFILL_LIMITE,
) -> pd.DataFrame:
    observadas = series[
        series["produto"].eq(produto)
        & series["mercado"].eq(mercado)
        & series["observado"].eq(1)
    ].set_index("data_cotacao")
    frame = pd.DataFrame(index=calendario)
    frame.index.name = "data_cotacao"
    frame["preco_observado"] = observadas["preco_kg_observado"].reindex(calendario)
    frame["preco"] = frame["preco_observado"].ffill(limit=limite_ffill)
    frame["observado"] = frame["preco_observado"].notna().astype(int)
    frame["preenchido"] = (
        frame["preco"].notna() & frame["preco_observado"].isna()
    ).astype(int)
    return frame


def taxa_mudancas(retornos: pd.Series) -> float:
    validos = retornos.dropna()
    return float(validos.abs().ge(1e-12).mean()) if len(validos) else 0.0


def construir_pares_operacionais(
    series: pd.DataFrame,
    calendarios: dict[str, pd.DatetimeIndex],
    data_corte_treino: date,
    configuracao: Configuracao,
    produtos: tuple[str, ...] = PRODUTOS_FIXOS,
    mercados: tuple[str, ...] = MERCADOS_FIXOS,
    minimo_dias_comuns: int = MINIMO_DIAS_COMUNS,
    limite_ffill: int = FFILL_LIMITE,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    series_operacionais = {
        (produto, mercado): precos_em_sessoes_ativas(
            series,
            produto,
            mercado,
            calendarios[mercado],
            limite_ffill,
        )
        for produto in produtos
        for mercado in mercados
    }
    linhas_pares: list[dict[str, Any]] = []
    frames_pares: dict[int, pd.DataFrame] = {}
    par_id = 0

    for produto in produtos:
        for mercado_a, mercado_b in combinations(sorted(mercados), 2):
            par_id += 1
            calendario_comum = calendarios[mercado_a].intersection(
                calendarios[mercado_b]
            )
            a = series_operacionais[(produto, mercado_a)].reindex(calendario_comum)
            b = series_operacionais[(produto, mercado_b)].reindex(calendario_comum)
            treino = calendario_comum.date <= data_corte_treino
            periodos_treino = int(treino.sum())
            observado_comum = int(
                (a["observado"].eq(1) & b["observado"].eq(1) & treino).sum()
            )
            cobertura = observado_comum / periodos_treino if periodos_treino else 0.0
            disponivel = a["preco"].notna() & b["preco"].notna()
            disponivel_comum = int((disponivel & treino).sum())

            frame: pd.DataFrame | None = None
            segmentos_treino = 0
            maior_segmento_treino = 0
            taxa_a = 0.0
            taxa_b = 0.0
            if observado_comum >= minimo_dias_comuns and cobertura >= configuracao.cobertura_minima:
                frame = pd.DataFrame(index=calendario_comum)
                frame.index.name = "data_cotacao"
                frame["preco_a"] = a["preco"]
                frame["preco_b"] = b["preco"]
                frame["observado_a"] = a["observado"]
                frame["observado_b"] = b["observado"]
                frame["preenchido_a"] = a["preenchido"]
                frame["preenchido_b"] = b["preenchido"]
                frame["segmento_par"] = identificar_segmentos(disponivel)
                mesmo_segmento = frame["segmento_par"].eq(
                    frame["segmento_par"].shift(1)
                )
                frame["retorno_a"] = np.log(frame["preco_a"]).diff().where(
                    disponivel & mesmo_segmento
                )
                frame["retorno_b"] = np.log(frame["preco_b"]).diff().where(
                    disponivel & mesmo_segmento
                )
                frame["periodo"] = np.where(
                    frame.index.date <= data_corte_treino,
                    "treino",
                    "teste",
                )
                frame = frame.loc[disponivel].copy()
                treino_frame = frame[frame["periodo"].eq("treino")]
                tamanhos = treino_frame.groupby("segmento_par").size()
                segmentos_treino = int(len(tamanhos))
                maior_segmento_treino = (
                    int(tamanhos.max()) if not tamanhos.empty else 0
                )
                taxa_a = taxa_mudancas(treino_frame["retorno_a"])
                taxa_b = taxa_mudancas(treino_frame["retorno_b"])

            elegivel = bool(
                frame is not None
                and taxa_a >= configuracao.taxa_mudancas_minima
                and taxa_b >= configuracao.taxa_mudancas_minima
            )
            if elegivel:
                motivo = "elegivel"
                frames_pares[par_id] = frame
            elif observado_comum < minimo_dias_comuns:
                motivo = "dias_observados_insuficientes"
            elif cobertura < configuracao.cobertura_minima:
                motivo = "cobertura_insuficiente"
            else:
                motivo = "mudancas_insuficientes"

            linhas_pares.append(
                {
                    "id": par_id,
                    "produto": produto,
                    "mercado_a": mercado_a,
                    "mercado_b": mercado_b,
                    "dias_observados_comuns_treino": observado_comum,
                    "dias_disponiveis_comuns_treino": disponivel_comum,
                    "periodos_operacionais_comuns_treino": periodos_treino,
                    "cobertura_observada_comum": cobertura,
                    "taxa_mudancas_a": taxa_a,
                    "taxa_mudancas_b": taxa_b,
                    "elegivel": int(elegivel),
                    "motivo": motivo,
                    "segmentos_treino": segmentos_treino if elegivel else 0,
                    "maior_segmento_treino": (
                        maior_segmento_treino if elegivel else 0
                    ),
                }
            )

    return pd.DataFrame(linhas_pares), frames_pares


def executar_configuracao(
    contexto: Contexto,
    calendarios: dict[str, pd.DatetimeIndex],
    configuracao: Configuracao,
) -> ResultadoOperacional:
    inicio = date.fromisoformat(contexto.parametros["data_inicio"])
    corte = date.fromisoformat(contexto.parametros["data_corte_treino"])
    fim = date.fromisoformat(contexto.parametros["data_fim_teste"])
    calendarios_contexto = {
        mercado: calendario[
            (calendario.date >= inicio) & (calendario.date <= fim)
        ]
        for mercado, calendario in calendarios.items()
    }
    pares, frames = construir_pares_operacionais(
        contexto.series,
        calendarios_contexto,
        corte,
        configuracao,
    )
    argumentos = SimpleNamespace(
        alpha=ALPHA,
        max_lag=MAX_LAG,
        max_lag_amostra=MAX_LAG,
    )
    (
        _,
        resultados,
        _,
        _,
        diagnosticos,
        avaliacoes,
    ) = executar_testes_granger(pares, frames, argumentos)
    return ResultadoOperacional(
        configuracao=configuracao,
        pares=pares,
        resultados=combinar_resultados(resultados, diagnosticos, avaliacoes),
    )


def criterio_selecao(resultado: ResultadoOperacional) -> tuple[int, ...]:
    resumo = resumir_resultados(resultado.resultados)
    return (
        int(resumo["relacoes_robustas"]),
        int(resumo["melhora_rmse_mae_fdr"]),
        int(resumo["residuos_adequados_fdr"]),
        int(resumo["significativas_fdr"]),
        round(resultado.configuracao.cobertura_minima * 100),
        round(resultado.configuracao.taxa_mudancas_minima * 100),
    )


def selecionar_configuracao(
    resultados: list[ResultadoOperacional],
) -> ResultadoOperacional:
    if not resultados:
        raise ValueError("nenhuma configuração operacional foi executada")
    return max(resultados, key=criterio_selecao)


def relacoes_robustas(resultados: pd.DataFrame) -> set[ChaveRelacao]:
    return {
        chave_relacao(linha)
        for linha in resultados.itertuples(index=False)
        if relacao_robusta(pd.Series(linha._asdict()))
    }


def resumir_operacional(resultado: ResultadoOperacional) -> dict[str, Any]:
    return {
        "configuracao": resultado.configuracao.nome,
        "cobertura_minima": resultado.configuracao.cobertura_minima,
        "taxa_mudancas_minima": resultado.configuracao.taxa_mudancas_minima,
        **resumir_metodo(
            resultado.resultados,
            int(resultado.pares["elegivel"].sum()),
        ),
    }


def construir_relatorio(
    banco_bruto: Path,
    contextos: dict[str, Contexto],
    candidatos_v1: list[ResultadoOperacional],
    selecionado_v1: ResultadoOperacional,
    resultado_v2: ResultadoOperacional,
    calendarios: dict[str, pd.DatetimeIndex],
) -> dict[str, Any]:
    resultados_operacionais = {
        "validacao_1": selecionado_v1,
        "validacao_2": resultado_v2,
    }
    relatorio: dict[str, Any] = {
        "schema_version": 1,
        "tipo": "calendario_operacional_granger",
        "gerado_em": datetime.now(UTC).isoformat(),
        "inclui_teste_final": False,
        "fonte_calendario": {
            "caminho": str(banco_bruto),
            "tamanho_bytes": banco_bruto.stat().st_size,
            "sha256": sha256_arquivo(banco_bruto),
            "regra": "qualquer cotacao CEASA do mercado em dia de segunda a sexta",
        },
        "parametros_fixos": {
            "ffill_limite_sessoes_ativas": FFILL_LIMITE,
            "minimo_dias_comuns_observados": MINIMO_DIAS_COMUNS,
            "max_lag": MAX_LAG,
            "max_lag_amostra": MAX_LAG,
            "alpha": ALPHA,
            "correcao": "Benjamini-Hochberg global",
        },
        "selecao": {
            "usa_somente_validacao_1": True,
            "configuracao_escolhida": selecionado_v1.configuracao.nome,
            "criterio_ordenado": [
                "relacoes_robustas",
                "significativas_com_melhora_rmse_e_mae",
                "significativas_com_residuos_adequados",
                "significativas_fdr",
                "maior_cobertura",
                "maior_taxa_mudancas",
            ],
        },
        "dias_operacionais": {
            mercado: int(len(calendario))
            for mercado, calendario in calendarios.items()
        },
        "fontes_granger": {
            nome: {
                "caminho": str(contexto.caminho),
                "sha256": sha256_arquivo(contexto.caminho),
            }
            for nome, contexto in contextos.items()
        },
        "candidatos_validacao_1": {
            resultado.configuracao.nome: resumir_operacional(resultado)
            for resultado in candidatos_v1
        },
        "validacoes": {},
    }

    for nome, contexto in contextos.items():
        operacional = resultados_operacionais[nome]
        relatorio["validacoes"][nome] = {
            "baseline_diario": resumir_metodo(
                contexto.resultados,
                int(contexto.pares["elegivel"].sum()),
            ),
            "calendario_operacional": resumir_operacional(operacional),
        }

    baseline_v1 = relacoes_significativas(contextos["validacao_1"].resultados)
    baseline_v2 = relacoes_significativas(contextos["validacao_2"].resultados)
    operacional_v1 = relacoes_significativas(selecionado_v1.resultados)
    operacional_v2 = relacoes_significativas(resultado_v2.resultados)
    robustas_v1 = relacoes_robustas(selecionado_v1.resultados)
    robustas_v2 = relacoes_robustas(resultado_v2.resultados)
    relatorio["estabilidade_entre_validacoes"] = {
        "baseline_diario": {
            "significativas_nas_duas": len(baseline_v1 & baseline_v2),
            "jaccard": calcular_jaccard(baseline_v1, baseline_v2),
        },
        "calendario_operacional": {
            "significativas_nas_duas": len(operacional_v1 & operacional_v2),
            "jaccard": calcular_jaccard(operacional_v1, operacional_v2),
            "robustas_nas_duas": len(robustas_v1 & robustas_v2),
            "relacoes_robustas_nas_duas": [
                {"produto": produto, "origem": origem, "destino": destino}
                for produto, origem, destino in sorted(robustas_v1 & robustas_v2)
            ],
        },
    }
    return relatorio


def imprimir_resumo(relatorio: dict[str, Any], saida: Path) -> None:
    print("\n=== CALENDÁRIO OPERACIONAL CEASA ===")
    print("Configuração escolhida somente na validação 1:", end=" ")
    print(relatorio["selecao"]["configuracao_escolhida"])
    for nome, validacao in relatorio["validacoes"].items():
        print(f"\n{nome}")
        for metodo in ("baseline_diario", "calendario_operacional"):
            resumo = validacao[metodo]
            print(
                f"  {metodo}: pares={resumo['pares_elegiveis']}, "
                f"testadas={resumo['direcoes_testadas']}, "
                f"FDR={resumo['significativas_fdr']}, "
                f"robustas={resumo['relacoes_robustas']}"
            )
    print("\nEstabilidade entre validações")
    for metodo, resumo in relatorio["estabilidade_entre_validacoes"].items():
        texto = (
            f"  {metodo}: comuns={resumo['significativas_nas_duas']}, "
            f"Jaccard={resumo['jaccard']:.4f}"
        )
        if "robustas_nas_duas" in resumo:
            texto += f", robustas nas duas={resumo['robustas_nas_duas']}"
        print(texto)
    print(f"\nRelatório gravado em: {saida}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raiz_experimento",
        type=Path,
        help="diretório da sensibilidade ffill com validacao_1 e validacao_2",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/cotacoes.sqlite"),
        help="SQLite bruto usado somente para identificar dias ativos",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="destino; por padrão usa calendario-operacional/comparacao.json",
    )
    return parser


def executar(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    raiz = args.raiz_experimento
    banco_bruto = args.database
    saida = args.output_json or raiz / "calendario-operacional/comparacao.json"
    if not banco_bruto.is_file():
        print(f"ERRO: banco bruto não encontrado: {banco_bruto}")
        return 2
    if saida.exists():
        print(f"ERRO: relatório já existe: {saida}")
        return 2

    contextos = {
        nome: carregar_contexto(nome, caminho_contexto(raiz, nome))
        for nome in ("validacao_1", "validacao_2")
    }
    calendarios = carregar_calendarios_operacionais(
        banco_bruto,
        min(date.fromisoformat(c.parametros["data_inicio"]) for c in contextos.values()),
        max(date.fromisoformat(c.parametros["data_fim_teste"]) for c in contextos.values()),
    )

    candidatos_v1 = []
    for configuracao in CONFIGURACOES:
        print(
            f"Executando validação 1: {configuracao.nome}...",
            flush=True,
        )
        candidatos_v1.append(
            executar_configuracao(
                contextos["validacao_1"],
                calendarios,
                configuracao,
            )
        )
    selecionado_v1 = selecionar_configuracao(candidatos_v1)
    print(
        f"Confirmando {selecionado_v1.configuracao.nome} na validação 2...",
        flush=True,
    )
    resultado_v2 = executar_configuracao(
        contextos["validacao_2"],
        calendarios,
        selecionado_v1.configuracao,
    )
    relatorio = construir_relatorio(
        banco_bruto,
        contextos,
        candidatos_v1,
        selecionado_v1,
        resultado_v2,
        calendarios,
    )
    saida.parent.mkdir(parents=True, exist_ok=True)
    temporario = saida.with_suffix(".tmp")
    temporario.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(saida)
    imprimir_resumo(relatorio, saida)
    return 0


def main() -> None:
    raise SystemExit(executar())


if __name__ == "__main__":
    main()
