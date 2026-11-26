"""Compara filtros de qualidade diários e confirmação semanal do Granger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from granger.executar_granger import (
    MERCADOS_FIXOS,
    PRODUTOS_FIXOS,
    construir_pares,
    corrigir_benjamini_hochberg,
    executar_testes_granger,
    identificar_segmentos,
)


COBERTURA_DIARIA_MINIMA = 0.75
TAXA_MUDANCAS_MINIMA = 0.15
COBERTURAS_SEMANAIS = (0.75, 0.85)
LAGS_SEMANAIS = (2, 4)
NOME_BANCO_DIARIO = "ffill-3.sqlite"

ChaveRelacao = tuple[str, str, str]


@dataclass(frozen=True)
class Contexto:
    nome: str
    caminho: Path
    parametros: dict[str, Any]
    series: pd.DataFrame
    pares: pd.DataFrame
    resultados: pd.DataFrame


@dataclass(frozen=True)
class ResultadoSemanal:
    cobertura_minima: float
    max_lag: int
    periodos_treino: int
    minimo_periodos_comuns: int
    pares: pd.DataFrame
    resultados: pd.DataFrame

    @property
    def nome(self) -> str:
        cobertura = round(self.cobertura_minima * 100)
        return f"cobertura-{cobertura}-lag-{self.max_lag}"


def sha256_arquivo(caminho: Path, tamanho_bloco: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while bloco := arquivo.read(tamanho_bloco):
            digest.update(bloco)
    return digest.hexdigest()


def abrir_somente_leitura(caminho: Path) -> sqlite3.Connection:
    if not caminho.is_file():
        raise FileNotFoundError(f"banco de Granger não encontrado: {caminho}")
    conexao = sqlite3.connect(f"{caminho.resolve().as_uri()}?mode=ro", uri=True)
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA query_only = ON")
    return conexao


def carregar_contexto(nome: str, caminho: Path) -> Contexto:
    with abrir_somente_leitura(caminho) as conexao:
        integridade = conexao.execute("PRAGMA quick_check").fetchone()[0]
        if integridade != "ok":
            raise RuntimeError(f"{caminho}: falha de integridade: {integridade}")

        execucao = conexao.execute(
            """
            SELECT parametros_json, status
            FROM execucoes_granger
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if execucao is None or execucao["status"] != "concluida":
            raise RuntimeError(f"{caminho}: execução de Granger não concluída")

        series = pd.read_sql_query(
            "SELECT * FROM series_temporais ORDER BY produto, mercado, data_cotacao",
            conexao,
            parse_dates=["data_cotacao"],
        )
        pares = pd.read_sql_query(
            "SELECT * FROM pares_candidatos ORDER BY id",
            conexao,
        )
        resultados = pd.read_sql_query(
            """
            SELECT
                resultado.*,
                diagnostico.status AS status_diagnostico,
                COALESCE(diagnostico.sem_autocorrelacao_irrestrito, 0)
                    AS residuos_adequados,
                avaliacao.status AS status_avaliacao,
                COALESCE(avaliacao.melhora_rmse, 0) AS melhora_rmse,
                COALESCE(avaliacao.melhora_mae, 0) AS melhora_mae,
                COALESCE(avaliacao.altas_avaliadas, 0) AS altas_avaliadas,
                COALESCE(avaliacao.melhora_rmse_altas, 0)
                    AS melhora_rmse_altas,
                COALESCE(avaliacao.melhora_mae_altas, 0)
                    AS melhora_mae_altas
            FROM resultados_granger AS resultado
            LEFT JOIN diagnosticos_residuos AS diagnostico
              ON diagnostico.resultado_id = resultado.id
            LEFT JOIN avaliacoes_preditivas AS avaliacao
              ON avaliacao.resultado_id = resultado.id
            ORDER BY resultado.produto, resultado.origem, resultado.destino
            """,
            conexao,
        )

    return Contexto(
        nome=nome,
        caminho=caminho,
        parametros=json.loads(execucao["parametros_json"]),
        series=series,
        pares=pares,
        resultados=resultados,
    )


def qualificar_pares_diarios(
    contexto: Contexto,
    cobertura_minima: float = COBERTURA_DIARIA_MINIMA,
    taxa_mudancas_minima: float = TAXA_MUDANCAS_MINIMA,
) -> pd.DataFrame:
    treino = contexto.series[contexto.series["periodo"] == "treino"].copy()
    periodos_treino = int(treino["data_cotacao"].nunique())
    retornos = treino[treino["retorno_log"].notna()].copy()
    retornos["mudou"] = retornos["retorno_log"].abs().ge(1e-12).astype(int)
    qualidade = (
        retornos.groupby(["produto", "mercado"], observed=True)
        .agg(retornos=("retorno_log", "size"), mudancas=("mudou", "sum"))
        .reset_index()
    )
    qualidade["taxa_mudancas"] = np.where(
        qualidade["retornos"] > 0,
        qualidade["mudancas"] / qualidade["retornos"],
        0.0,
    )
    taxas = {
        (linha.produto, linha.mercado): float(linha.taxa_mudancas)
        for linha in qualidade.itertuples(index=False)
    }

    pares = contexto.pares.copy()
    pares["periodos_treino"] = periodos_treino
    pares["cobertura_observada"] = (
        pares["dias_observados_comuns_treino"] / periodos_treino
    )
    pares["taxa_mudancas_a"] = [
        taxas.get((linha.produto, linha.mercado_a), 0.0)
        for linha in pares.itertuples(index=False)
    ]
    pares["taxa_mudancas_b"] = [
        taxas.get((linha.produto, linha.mercado_b), 0.0)
        for linha in pares.itertuples(index=False)
    ]
    pares["qualificado_qualidade"] = (
        pares["elegivel"].eq(1)
        & pares["cobertura_observada"].ge(cobertura_minima)
        & pares["taxa_mudancas_a"].ge(taxa_mudancas_minima)
        & pares["taxa_mudancas_b"].ge(taxa_mudancas_minima)
    )
    return pares


def recalcular_fdr(
    resultados: pd.DataFrame,
    pares_aceitos: set[int],
    alpha: float,
) -> pd.DataFrame:
    filtrados = resultados[resultados["par_id"].isin(pares_aceitos)].copy()
    filtrados["q_valor_fdr"] = np.nan
    filtrados["significativo_fdr"] = 0
    testados = filtrados["status"].eq("testado") & filtrados["p_valor"].notna()
    indices = filtrados.index[testados].tolist()
    valores = corrigir_benjamini_hochberg(
        filtrados.loc[indices, "p_valor"].astype(float).tolist()
    )
    filtrados.loc[indices, "q_valor_fdr"] = valores
    filtrados.loc[indices, "significativo_fdr"] = [
        int(valor < alpha) for valor in valores
    ]
    return filtrados.reset_index(drop=True)


def primeira_segunda_completa(inicio: date) -> pd.Timestamp:
    data = pd.Timestamp(inicio)
    deslocamento = (-data.day_of_week) % 7
    segunda = data + pd.Timedelta(days=deslocamento)
    return segunda if segunda.date() >= inicio else segunda + pd.Timedelta(days=7)


def construir_series_semanais(
    series_diarias: pd.DataFrame,
    data_inicio: date,
    data_corte_treino: date,
    data_fim_teste: date,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], int]:
    primeiro = primeira_segunda_completa(data_inicio)
    ultimo = pd.Timestamp(data_fim_teste) - pd.Timedelta(days=4)
    calendario = pd.date_range(primeiro, ultimo, freq="W-MON")
    corte = pd.Timestamp(data_corte_treino)

    observadas = series_diarias[series_diarias["observado"].eq(1)].copy()
    observadas["inicio_semana"] = observadas["data_cotacao"] - pd.to_timedelta(
        observadas["data_cotacao"].dt.day_of_week,
        unit="D",
    )
    observadas["fim_semana"] = observadas["inicio_semana"] + pd.Timedelta(days=4)
    semana_inteira_no_periodo = (
        observadas["fim_semana"].le(corte)
        | observadas["inicio_semana"].gt(corte)
    )
    observadas = observadas[
        semana_inteira_no_periodo
        & observadas["inicio_semana"].ge(pd.Timestamp(data_inicio))
        & observadas["fim_semana"].le(pd.Timestamp(data_fim_teste))
    ]

    chaves = ["produto", "ceasa_slug", "uf", "mercado", "inicio_semana"]
    semanais = (
        observadas.groupby(chaves, as_index=False, observed=True)
        .agg(
            preco_kg_observado=("preco_kg_observado", "median"),
            n_cotacoes=("n_cotacoes", "sum"),
            n_exatas=("n_exatas", "sum"),
            n_calculadas=("n_calculadas", "sum"),
            n_padronizadas=("n_padronizadas", "sum"),
            n_inferidas=("n_inferidas", "sum"),
        )
        .rename(columns={"inicio_semana": "data_cotacao"})
    )

    identidades = (
        series_diarias[["produto", "ceasa_slug", "uf", "mercado"]]
        .drop_duplicates()
        .sort_values(["produto", "mercado"])
    )
    frames: list[pd.DataFrame] = []
    mapa: dict[tuple[str, str], pd.DataFrame] = {}
    for identidade in identidades.itertuples(index=False):
        subconjunto = semanais[
            semanais["produto"].eq(identidade.produto)
            & semanais["mercado"].eq(identidade.mercado)
        ].set_index("data_cotacao")
        frame = pd.DataFrame(index=calendario)
        frame.index.name = "data_cotacao"
        frame["preco_kg_observado"] = subconjunto[
            "preco_kg_observado"
        ].reindex(calendario)
        frame["preco_kg"] = frame["preco_kg_observado"]
        for coluna in (
            "n_cotacoes",
            "n_exatas",
            "n_calculadas",
            "n_padronizadas",
            "n_inferidas",
        ):
            frame[coluna] = (
                subconjunto[coluna].reindex(calendario).fillna(0).astype(int)
            )
        frame["observado"] = frame["preco_kg_observado"].notna().astype(int)
        frame["preenchido"] = 0
        disponivel = frame["preco_kg"].notna()
        frame["segmento_serie"] = identificar_segmentos(disponivel)
        mesmo_segmento = frame["segmento_serie"].eq(
            frame["segmento_serie"].shift(1)
        )
        frame["retorno_log"] = np.log(frame["preco_kg"]).diff().where(
            disponivel & mesmo_segmento
        )
        frame["periodo"] = np.where(
            frame.index.date <= data_corte_treino,
            "treino",
            "teste",
        )
        frame["produto"] = identidade.produto
        frame["mercado"] = identidade.mercado
        frame["ceasa_slug"] = identidade.ceasa_slug
        frame["uf"] = identidade.uf
        mapa[(identidade.produto, identidade.mercado)] = frame.copy()
        frames.append(frame.reset_index())

    resultado = pd.concat(frames, ignore_index=True)
    semana_mista = (calendario <= corte) & (
        calendario + pd.Timedelta(days=4) > corte
    )
    periodos_treino = int(((calendario <= corte) & ~semana_mista).sum())
    return resultado, mapa, periodos_treino


def combinar_resultados(
    resultados: pd.DataFrame,
    diagnosticos: pd.DataFrame,
    avaliacoes: pd.DataFrame,
) -> pd.DataFrame:
    diagnosticos = diagnosticos.rename(
        columns={
            "status": "status_diagnostico",
            "sem_autocorrelacao_irrestrito": "residuos_adequados",
        }
    )
    avaliacoes = avaliacoes.rename(columns={"status": "status_avaliacao"})
    colunas_diagnosticos = [
        "resultado_id",
        "status_diagnostico",
        "residuos_adequados",
    ]
    colunas_avaliacoes = [
        "resultado_id",
        "status_avaliacao",
        "melhora_rmse",
        "melhora_mae",
        "altas_avaliadas",
        "melhora_rmse_altas",
        "melhora_mae_altas",
    ]
    return (
        resultados.merge(
            diagnosticos[colunas_diagnosticos],
            how="left",
            left_on="id",
            right_on="resultado_id",
        )
        .drop(columns=["resultado_id"])
        .merge(
            avaliacoes[colunas_avaliacoes],
            how="left",
            left_on="id",
            right_on="resultado_id",
        )
        .drop(columns=["resultado_id"])
    )


def executar_granger_semanal(
    contexto: Contexto,
    mapa_semanal: dict[tuple[str, str], pd.DataFrame],
    periodos_treino: int,
    cobertura_minima: float,
    max_lag: int,
) -> ResultadoSemanal:
    minimo_periodos = math.ceil(periodos_treino * cobertura_minima)
    argumentos = SimpleNamespace(
        data_corte_treino=date.fromisoformat(
            contexto.parametros["data_corte_treino"]
        ),
        min_dias_comuns=minimo_periodos,
        alpha=float(contexto.parametros["alpha"]),
        max_lag=max_lag,
        max_lag_amostra=max_lag,
    )
    pares, _, frames_pares = construir_pares(mapa_semanal, argumentos)
    (
        _,
        resultados,
        _,
        _,
        diagnosticos,
        avaliacoes,
    ) = executar_testes_granger(pares, frames_pares, argumentos)
    combinados = combinar_resultados(resultados, diagnosticos, avaliacoes)
    return ResultadoSemanal(
        cobertura_minima=cobertura_minima,
        max_lag=max_lag,
        periodos_treino=periodos_treino,
        minimo_periodos_comuns=minimo_periodos,
        pares=pares,
        resultados=combinados,
    )


def relacao_robusta(linha: pd.Series) -> bool:
    return bool(
        linha["significativo_fdr"]
        and linha.get("status_diagnostico") == "avaliado"
        and linha.get("residuos_adequados", 0)
        and linha.get("status_avaliacao") == "avaliado"
        and linha.get("melhora_rmse", 0)
        and linha.get("melhora_mae", 0)
    )


def resumir_resultados(resultados: pd.DataFrame) -> dict[str, int | float]:
    testados = resultados[resultados["status"].eq("testado")]
    significativos = testados[testados["significativo_fdr"].eq(1)].copy()
    if significativos.empty:
        return {
            "direcoes_testadas": int(len(testados)),
            "significativas_fdr": 0,
            "residuos_adequados_fdr": 0,
            "melhora_rmse_fdr": 0,
            "melhora_mae_fdr": 0,
            "melhora_rmse_mae_fdr": 0,
            "relacoes_robustas": 0,
            "taxa_robustas_fdr": 0.0,
        }

    melhora_ambas = significativos["melhora_rmse"].eq(1) & significativos[
        "melhora_mae"
    ].eq(1)
    robustas = significativos.apply(relacao_robusta, axis=1)
    return {
        "direcoes_testadas": int(len(testados)),
        "significativas_fdr": int(len(significativos)),
        "residuos_adequados_fdr": int(
            significativos["residuos_adequados"].fillna(0).sum()
        ),
        "melhora_rmse_fdr": int(significativos["melhora_rmse"].fillna(0).sum()),
        "melhora_mae_fdr": int(significativos["melhora_mae"].fillna(0).sum()),
        "melhora_rmse_mae_fdr": int(melhora_ambas.sum()),
        "relacoes_robustas": int(robustas.sum()),
        "taxa_robustas_fdr": float(robustas.mean()),
    }


def chave_relacao(linha: Any) -> ChaveRelacao:
    return linha.produto, linha.origem, linha.destino


def relacoes_significativas(resultados: pd.DataFrame) -> set[ChaveRelacao]:
    significativos = resultados[
        resultados["status"].eq("testado")
        & resultados["significativo_fdr"].eq(1)
    ]
    return {
        chave_relacao(linha) for linha in significativos.itertuples(index=False)
    }


def calcular_jaccard(
    primeiro: set[ChaveRelacao],
    segundo: set[ChaveRelacao],
) -> float:
    uniao = primeiro | segundo
    return len(primeiro & segundo) / len(uniao) if uniao else 1.0


def criterio_selecao(resultado: ResultadoSemanal) -> tuple[int, ...]:
    resumo = resumir_resultados(resultado.resultados)
    return (
        int(resumo["relacoes_robustas"]),
        int(resumo["melhora_rmse_mae_fdr"]),
        int(resumo["residuos_adequados_fdr"]),
        int(resumo["significativas_fdr"]),
        round(resultado.cobertura_minima * 100),
        -resultado.max_lag,
    )


def selecionar_configuracao(
    resultados: list[ResultadoSemanal],
) -> ResultadoSemanal:
    if not resultados:
        raise ValueError("nenhuma configuração semanal foi executada")
    return max(resultados, key=criterio_selecao)


def serializar_relacoes(resultados: pd.DataFrame) -> list[dict[str, Any]]:
    significativos = resultados[
        resultados["status"].eq("testado")
        & resultados["significativo_fdr"].eq(1)
    ]
    relacoes = []
    for linha in significativos.itertuples(index=False):
        relacoes.append(
            {
                "produto": linha.produto,
                "origem": linha.origem,
                "destino": linha.destino,
                "lag": int(linha.lag_selecionado),
                "p_valor": float(linha.p_valor),
                "q_valor_fdr": float(linha.q_valor_fdr),
                "residuos_adequados": bool(linha.residuos_adequados),
                "melhora_rmse": bool(linha.melhora_rmse),
                "melhora_mae": bool(linha.melhora_mae),
                "relacao_robusta": relacao_robusta(pd.Series(linha._asdict())),
            }
        )
    return relacoes


def resumir_metodo(
    resultados: pd.DataFrame,
    pares_elegiveis: int,
) -> dict[str, Any]:
    return {
        "pares_elegiveis": pares_elegiveis,
        **resumir_resultados(resultados),
        "relacoes_significativas": serializar_relacoes(resultados),
    }


def filtrar_por_relacoes(
    resultados: pd.DataFrame,
    relacoes: set[ChaveRelacao],
) -> pd.DataFrame:
    mascara = [
        chave_relacao(linha) in relacoes
        for linha in resultados.itertuples(index=False)
    ]
    return resultados.loc[mascara].copy()


def construir_relatorio(
    contextos: dict[str, Contexto],
    qualidade: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    candidatos_v1: list[ResultadoSemanal],
    semanal_v2: ResultadoSemanal,
    selecionado_v1: ResultadoSemanal,
) -> dict[str, Any]:
    relatorio: dict[str, Any] = {
        "schema_version": 1,
        "tipo": "qualidade_series_temporais_granger",
        "gerado_em": datetime.now(UTC).isoformat(),
        "inclui_teste_final": False,
        "politica_diaria": {
            "cobertura_minima": COBERTURA_DIARIA_MINIMA,
            "taxa_mudancas_minima": TAXA_MUDANCAS_MINIMA,
            "mantem_minimo_dias_observados_original": True,
            "fdr_recalculado_apos_filtro": True,
        },
        "selecao_semanal": {
            "usa_somente_validacao_1": True,
            "agregacao": "mediana dos precos diarios observados",
            "preenchimento": "nenhum",
            "semanas_que_cruzam_fronteira_temporal": "descartadas",
            "coberturas_avaliadas": list(COBERTURAS_SEMANAIS),
            "max_lags_avaliados": list(LAGS_SEMANAIS),
            "criterio_ordenado": [
                "relacoes_robustas",
                "significativas_com_melhora_rmse_e_mae",
                "significativas_com_residuos_adequados",
                "significativas_fdr",
                "maior_cobertura",
                "menor_max_lag",
            ],
            "configuracao_escolhida": selecionado_v1.nome,
        },
        "fontes": {
            nome: {
                "caminho": str(contexto.caminho),
                "sha256": sha256_arquivo(contexto.caminho),
            }
            for nome, contexto in contextos.items()
        },
        "validacoes": {},
    }

    for nome, contexto in contextos.items():
        pares_qualificados, resultados_qualificados = qualidade[nome]
        semanal = selecionado_v1 if nome == "validacao_1" else semanal_v2
        relacoes_diarias = relacoes_significativas(resultados_qualificados)
        relacoes_semanais = relacoes_significativas(semanal.resultados)
        confirmadas = relacoes_diarias & relacoes_semanais
        resultados_confirmados = filtrar_por_relacoes(
            resultados_qualificados,
            confirmadas,
        )
        relatorio["validacoes"][nome] = {
            "baseline_diario": resumir_metodo(
                contexto.resultados,
                int(contexto.pares["elegivel"].sum()),
            ),
            "diario_com_qualidade": resumir_metodo(
                resultados_qualificados,
                int(pares_qualificados["qualificado_qualidade"].sum()),
            ),
            "semanal": {
                "configuracao": semanal.nome,
                "periodos_treino": semanal.periodos_treino,
                "minimo_periodos_comuns": semanal.minimo_periodos_comuns,
                **resumir_metodo(
                    semanal.resultados,
                    int(semanal.pares["elegivel"].sum()),
                ),
            },
            "diario_confirmado_semanal": {
                **resumir_resultados(resultados_confirmados),
                "relacoes": [
                    {"produto": produto, "origem": origem, "destino": destino}
                    for produto, origem, destino in sorted(confirmadas)
                ],
            },
        }

    relatorio["candidatos_semanais_validacao_1"] = {
        resultado.nome: {
            "periodos_treino": resultado.periodos_treino,
            "minimo_periodos_comuns": resultado.minimo_periodos_comuns,
            **resumir_metodo(
                resultado.resultados,
                int(resultado.pares["elegivel"].sum()),
            ),
        }
        for resultado in candidatos_v1
    }

    estabilidade = {}
    for metodo in (
        "baseline_diario",
        "diario_com_qualidade",
        "semanal",
        "diario_confirmado_semanal",
    ):
        if metodo == "diario_confirmado_semanal":
            v1 = {
                (linha["produto"], linha["origem"], linha["destino"])
                for linha in relatorio["validacoes"]["validacao_1"][metodo][
                    "relacoes"
                ]
            }
            v2 = {
                (linha["produto"], linha["origem"], linha["destino"])
                for linha in relatorio["validacoes"]["validacao_2"][metodo][
                    "relacoes"
                ]
            }
        else:
            v1 = {
                (linha["produto"], linha["origem"], linha["destino"])
                for linha in relatorio["validacoes"]["validacao_1"][metodo][
                    "relacoes_significativas"
                ]
            }
            v2 = {
                (linha["produto"], linha["origem"], linha["destino"])
                for linha in relatorio["validacoes"]["validacao_2"][metodo][
                    "relacoes_significativas"
                ]
            }
        estabilidade[metodo] = {
            "significativas_nas_duas": len(v1 & v2),
            "jaccard": calcular_jaccard(v1, v2),
        }
    relatorio["estabilidade_entre_validacoes"] = estabilidade
    return relatorio


def imprimir_resumo(relatorio: dict[str, Any], saida: Path) -> None:
    print("\n=== QUALIDADE DAS SÉRIES TEMPORAIS ===")
    print(
        "Configuração semanal escolhida somente na validação 1:",
        relatorio["selecao_semanal"]["configuracao_escolhida"],
    )
    for nome, validacao in relatorio["validacoes"].items():
        print(f"\n{nome}")
        for metodo in (
            "baseline_diario",
            "diario_com_qualidade",
            "semanal",
            "diario_confirmado_semanal",
        ):
            resumo = validacao[metodo]
            print(
                f"  {metodo}: testadas={resumo['direcoes_testadas']}, "
                f"FDR={resumo['significativas_fdr']}, "
                f"robustas={resumo['relacoes_robustas']}"
            )
    print("\nEstabilidade entre validações")
    for metodo, estabilidade in relatorio["estabilidade_entre_validacoes"].items():
        print(
            f"  {metodo}: comuns={estabilidade['significativas_nas_duas']}, "
            f"Jaccard={estabilidade['jaccard']:.4f}"
        )
    print(f"\nRelatório gravado em: {saida}")


def caminho_contexto(raiz: Path, nome: str) -> Path:
    return raiz / nome / "granger" / NOME_BANCO_DIARIO


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raiz_experimento",
        type=Path,
        help="diretório da sensibilidade ffill com validacao_1 e validacao_2",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="destino do relatório; por padrão usa qualidade-series/comparacao.json",
    )
    return parser


def executar(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    raiz = args.raiz_experimento
    saida = args.output_json or raiz / "qualidade-series/comparacao.json"
    if saida.exists():
        print(f"ERRO: relatório já existe: {saida}")
        return 2

    contextos = {
        nome: carregar_contexto(nome, caminho_contexto(raiz, nome))
        for nome in ("validacao_1", "validacao_2")
    }
    qualidade: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    mapas_semanais: dict[str, dict[tuple[str, str], pd.DataFrame]] = {}
    periodos_semanais: dict[str, int] = {}

    for nome, contexto in contextos.items():
        print(f"Preparando séries de {nome}...", flush=True)
        pares_qualificados = qualificar_pares_diarios(contexto)
        ids = set(
            pares_qualificados.loc[
                pares_qualificados["qualificado_qualidade"], "id"
            ].astype(int)
        )
        resultados_qualificados = recalcular_fdr(
            contexto.resultados,
            ids,
            float(contexto.parametros["alpha"]),
        )
        qualidade[nome] = pares_qualificados, resultados_qualificados

        _, mapa, periodos = construir_series_semanais(
            contexto.series,
            date.fromisoformat(contexto.parametros["data_inicio"]),
            date.fromisoformat(contexto.parametros["data_corte_treino"]),
            date.fromisoformat(contexto.parametros["data_fim_teste"]),
        )
        mapas_semanais[nome] = mapa
        periodos_semanais[nome] = periodos

    candidatos_v1 = []
    for cobertura in COBERTURAS_SEMANAIS:
        for max_lag in LAGS_SEMANAIS:
            print(
                f"Executando validação 1 semanal: cobertura={cobertura:.0%}, "
                f"max_lag={max_lag}...",
                flush=True,
            )
            candidatos_v1.append(
                executar_granger_semanal(
                    contextos["validacao_1"],
                    mapas_semanais["validacao_1"],
                    periodos_semanais["validacao_1"],
                    cobertura,
                    max_lag,
                )
            )

    selecionado_v1 = selecionar_configuracao(candidatos_v1)
    print(
        f"Confirmando {selecionado_v1.nome} na validação 2...",
        flush=True,
    )
    semanal_v2 = executar_granger_semanal(
        contextos["validacao_2"],
        mapas_semanais["validacao_2"],
        periodos_semanais["validacao_2"],
        selecionado_v1.cobertura_minima,
        selecionado_v1.max_lag,
    )
    relatorio = construir_relatorio(
        contextos,
        qualidade,
        candidatos_v1,
        semanal_v2,
        selecionado_v1,
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
