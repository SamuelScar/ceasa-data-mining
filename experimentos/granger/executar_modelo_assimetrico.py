#!/usr/bin/env python3
"""Executa Granger assimétrico para isolar altas de preço na origem."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as distribuicao_f

from experimentos.granger.executar_modelo_condicional import (
    abrir_somente_leitura,
    carregar_metadados_origem,
    sha256_arquivo,
)
from granger.executar_granger import (
    MINIMO_OBSERVACOES_ALTAS_TESTE,
    MINIMO_OBSERVACOES_TESTE,
    PERCENTIL_ALTA_DESTINO,
    calcular_metricas_erros,
    corrigir_benjamini_hochberg,
    diagnosticar_residuos,
)


SCHEMA_VERSION = 1
HORIZONTE_RESPOSTA = 10


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite do Granger bivariado")
    parser.add_argument("--output-db", type=Path, required=True)
    return parser


def decompor_retorno(retorno: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separa o retorno em parcelas negativa e positiva mutuamente exclusivas."""
    return np.minimum(retorno, 0.0), np.maximum(retorno, 0.0)


def carregar_direcoes(conexao: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT id AS resultado_granger_id, par_id, produto, origem, destino
        FROM resultados_granger
        WHERE status = 'testado'
        ORDER BY produto, par_id, origem, destino
        """,
        conexao,
    )


def carregar_series_pares(conexao: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT par_id, mercado_a, mercado_b, data_cotacao, periodo,
               segmento_par, retorno_a, retorno_b
        FROM series_pares
        WHERE retorno_a IS NOT NULL AND retorno_b IS NOT NULL
        ORDER BY par_id, data_cotacao
        """,
        conexao,
    )


def construir_frame_direcao(
    serie_par: pd.DataFrame,
    origem: str,
    destino: str,
) -> pd.DataFrame:
    mercado_a = str(serie_par["mercado_a"].iloc[0])
    mercado_b = str(serie_par["mercado_b"].iloc[0])
    if {origem, destino} != {mercado_a, mercado_b}:
        raise ValueError("a direção não pertence ao par informado")
    coluna_origem = "retorno_a" if origem == mercado_a else "retorno_b"
    coluna_destino = "retorno_a" if destino == mercado_a else "retorno_b"
    return serie_par[
        ["data_cotacao", "periodo", "segmento_par"]
    ].assign(
        origem=serie_par[coluna_origem].to_numpy(dtype=float),
        destino=serie_par[coluna_destino].to_numpy(dtype=float),
    )


def partes_treino(
    frame: pd.DataFrame,
    lag: int,
    max_lag_amostra: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    partes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    treino = frame[frame["periodo"] == "treino"]
    for _, segmento in treino.groupby("segmento_par", sort=True):
        origem = segmento["origem"].to_numpy(dtype=float)
        destino = segmento["destino"].to_numpy(dtype=float)
        if len(segmento) <= max_lag_amostra:
            continue
        origem_negativa, origem_positiva = decompor_retorno(origem)
        alvo = destino[max_lag_amostra:]
        defasagens_destino = np.column_stack(
            [
                destino[
                    max_lag_amostra - deslocamento : len(destino) - deslocamento
                ]
                for deslocamento in range(1, lag + 1)
            ]
        )
        defasagens_negativas = np.column_stack(
            [
                origem_negativa[
                    max_lag_amostra - deslocamento : len(origem) - deslocamento
                ]
                for deslocamento in range(1, lag + 1)
            ]
        )
        defasagens_positivas = np.column_stack(
            [
                origem_positiva[
                    max_lag_amostra - deslocamento : len(origem) - deslocamento
                ]
                for deslocamento in range(1, lag + 1)
            ]
        )
        restrita = np.column_stack(
            [np.ones(len(alvo)), defasagens_destino, defasagens_negativas]
        )
        irrestrita = np.column_stack([restrita, defasagens_positivas])
        partes.append((alvo, restrita, irrestrita))
    return partes


def matrizes_treino(
    frame: pd.DataFrame,
    lag: int,
    max_lag_amostra: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    partes = partes_treino(frame, lag, max_lag_amostra)
    if not partes:
        raise ValueError("nenhum segmento de treino possui histórico suficiente")
    return (
        np.concatenate([parte[0] for parte in partes]),
        np.vstack([parte[1] for parte in partes]),
        np.vstack([parte[2] for parte in partes]),
        partes,
    )


def matrizes_teste(
    frame: pd.DataFrame,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alvos: list[np.ndarray] = []
    restritas: list[np.ndarray] = []
    irrestritas: list[np.ndarray] = []
    for _, segmento in frame.groupby("segmento_par", sort=True):
        origem = segmento["origem"].to_numpy(dtype=float)
        destino = segmento["destino"].to_numpy(dtype=float)
        periodos = segmento["periodo"].to_numpy()
        posicoes = np.flatnonzero(periodos == "teste")
        posicoes = posicoes[posicoes >= lag]
        if not len(posicoes):
            continue
        origem_negativa, origem_positiva = decompor_retorno(origem)
        defasagens_destino = np.column_stack(
            [
                destino[posicoes - deslocamento]
                for deslocamento in range(1, lag + 1)
            ]
        )
        defasagens_negativas = np.column_stack(
            [
                origem_negativa[posicoes - deslocamento]
                for deslocamento in range(1, lag + 1)
            ]
        )
        defasagens_positivas = np.column_stack(
            [
                origem_positiva[posicoes - deslocamento]
                for deslocamento in range(1, lag + 1)
            ]
        )
        restrita = np.column_stack(
            [np.ones(len(posicoes)), defasagens_destino, defasagens_negativas]
        )
        alvos.append(destino[posicoes])
        restritas.append(restrita)
        irrestritas.append(np.column_stack([restrita, defasagens_positivas]))
    if not alvos:
        raise ValueError("nenhum ponto de teste possui histórico suficiente")
    return np.concatenate(alvos), np.vstack(restritas), np.vstack(irrestritas)


def ajustar_modelos(
    alvo: np.ndarray,
    restrita: np.ndarray,
    irrestrita: np.ndarray,
    lag: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    n = len(alvo)
    graus_den = n - irrestrita.shape[1]
    if graus_den <= 0:
        raise ValueError("amostra insuficiente para o modelo irrestrito")
    if np.linalg.matrix_rank(restrita) < restrita.shape[1]:
        raise ValueError("matriz restrita sem posto completo")
    if np.linalg.matrix_rank(irrestrita) < irrestrita.shape[1]:
        raise ValueError("matriz irrestrita sem posto completo")

    beta_restrito = np.linalg.lstsq(restrita, alvo, rcond=None)[0]
    beta_irrestrito = np.linalg.lstsq(irrestrita, alvo, rcond=None)[0]
    residuo_restrito = alvo - restrita @ beta_restrito
    residuo_irrestrito = alvo - irrestrita @ beta_irrestrito
    rss_restrito = float(residuo_restrito @ residuo_restrito)
    rss_irrestrito = float(residuo_irrestrito @ residuo_irrestrito)
    if rss_irrestrito <= 0:
        raise ValueError("resíduo nulo no modelo irrestrito")
    diferenca = max(0.0, rss_restrito - rss_irrestrito)
    estatistica_f = (diferenca / lag) / (rss_irrestrito / graus_den)
    p_valor = float(distribuicao_f.sf(estatistica_f, lag, graus_den))
    bic = float(
        n * np.log(rss_irrestrito / n) + irrestrita.shape[1] * np.log(n)
    )
    return (
        {
            "lag": lag,
            "n_observacoes": n,
            "bic": bic,
            "rss_restrito": rss_restrito,
            "rss_irrestrito": rss_irrestrito,
            "estatistica_f": estatistica_f,
            "graus_liberdade_num": lag,
            "graus_liberdade_den": graus_den,
            "p_valor": p_valor,
        },
        beta_restrito,
        beta_irrestrito,
    )


def selecionar_lag(
    frame: pd.DataFrame,
    max_lag: int,
    max_lag_amostra: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidatos: list[dict[str, Any]] = []
    for lag in range(1, max_lag + 1):
        alvo, restrita, irrestrita, _ = matrizes_treino(
            frame,
            lag,
            max_lag_amostra,
        )
        resultado, _, _ = ajustar_modelos(alvo, restrita, irrestrita, lag)
        candidatos.append(resultado)
    return candidatos, min(candidatos, key=lambda item: (item["bic"], item["lag"]))


def resposta_dinamica_assimetrica(
    beta_irrestrito: np.ndarray,
    lag: int,
    horizonte: int = HORIZONTE_RESPOSTA,
) -> float:
    coeficientes_destino = beta_irrestrito[1 : 1 + lag]
    inicio_positivas = 1 + 2 * lag
    coeficientes_positivos = beta_irrestrito[
        inicio_positivas : inicio_positivas + lag
    ]
    resposta_destino = [0.0]
    for passo in range(1, horizonte + 1):
        resposta = (
            float(coeficientes_positivos[passo - 1]) if passo <= lag else 0.0
        )
        for defasagem, coeficiente in enumerate(
            coeficientes_destino,
            start=1,
        ):
            if defasagem <= passo:
                resposta += float(coeficiente) * resposta_destino[
                    passo - defasagem
                ]
        resposta_destino.append(resposta)
    return float(sum(resposta_destino[1:]))


def avaliacao_sem_metricas(status: str, observacao: str) -> dict[str, Any]:
    return {
        "status": status,
        "n_observacoes": 0,
        "rmse_restrito": None,
        "rmse_irrestrito": None,
        "ganho_rmse_percentual": None,
        "mae_restrito": None,
        "mae_irrestrito": None,
        "ganho_mae_percentual": None,
        "melhora_rmse": 0,
        "melhora_mae": 0,
        "limite_alta_destino": None,
        "n_observacoes_altas": 0,
        "altas_avaliadas": 0,
        "rmse_restrito_altas": None,
        "rmse_irrestrito_altas": None,
        "ganho_rmse_altas_percentual": None,
        "mae_restrito_altas": None,
        "mae_irrestrito_altas": None,
        "ganho_mae_altas_percentual": None,
        "melhora_rmse_altas": 0,
        "melhora_mae_altas": 0,
        "relacao_robusta": 0,
        "relacao_alta_robusta": 0,
        "observacao": observacao,
    }


def avaliar_previsao(
    frame: pd.DataFrame,
    lag: int,
    alvo_treino: np.ndarray,
    beta_restrito: np.ndarray,
    beta_irrestrito: np.ndarray,
) -> dict[str, Any]:
    try:
        alvo, restrita, irrestrita = matrizes_teste(frame, lag)
    except ValueError as erro:
        return avaliacao_sem_metricas("amostra_insuficiente", str(erro))
    if len(alvo) < MINIMO_OBSERVACOES_TESTE:
        avaliacao = avaliacao_sem_metricas(
            "amostra_insuficiente",
            f"mínimo de {MINIMO_OBSERVACOES_TESTE} observações de teste",
        )
        avaliacao["n_observacoes"] = len(alvo)
        return avaliacao

    erro_restrito = alvo - restrita @ beta_restrito
    erro_irrestrito = alvo - irrestrita @ beta_irrestrito
    metricas = calcular_metricas_erros(erro_restrito, erro_irrestrito)
    limite_alta = max(
        0.0,
        float(np.quantile(alvo_treino, PERCENTIL_ALTA_DESTINO)),
    )
    mascara_altas = alvo > limite_alta
    n_altas = int(mascara_altas.sum())
    altas_avaliadas = n_altas >= MINIMO_OBSERVACOES_ALTAS_TESTE
    metricas_altas = (
        calcular_metricas_erros(
            erro_restrito[mascara_altas],
            erro_irrestrito[mascara_altas],
            "altas",
        )
        if altas_avaliadas
        else {
            "rmse_restrito_altas": None,
            "rmse_irrestrito_altas": None,
            "ganho_rmse_altas_percentual": None,
            "mae_restrito_altas": None,
            "mae_irrestrito_altas": None,
            "ganho_mae_altas_percentual": None,
            "melhora_rmse_altas": 0,
            "melhora_mae_altas": 0,
        }
    )
    return {
        "status": "avaliado",
        "n_observacoes": len(alvo),
        **metricas,
        "limite_alta_destino": limite_alta,
        "n_observacoes_altas": n_altas,
        "altas_avaliadas": int(altas_avaliadas),
        **metricas_altas,
        "relacao_robusta": 0,
        "relacao_alta_robusta": 0,
        "observacao": (
            "previsão um passo à frente; modelo restrito já controla as "
            "quedas da origem"
        ),
    }


def executar_direcoes(
    direcoes: pd.DataFrame,
    series_pares: pd.DataFrame,
    max_lag: int,
    max_lag_amostra: int,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    pares = {
        int(par_id): grupo.copy()
        for par_id, grupo in series_pares.groupby("par_id", sort=False)
    }
    resultados: list[dict[str, Any]] = []
    lags: list[dict[str, Any]] = []
    coeficientes: list[dict[str, Any]] = []
    diagnosticos: list[dict[str, Any]] = []
    avaliacoes: list[dict[str, Any]] = []

    for resultado_id, direcao in enumerate(
        direcoes.itertuples(index=False),
        start=1,
    ):
        base = {
            "id": resultado_id,
            "resultado_granger_id": int(direcao.resultado_granger_id),
            "par_id": int(direcao.par_id),
            "produto": direcao.produto,
            "origem": direcao.origem,
            "destino": direcao.destino,
            "status": "testado",
            "n_observacoes": None,
            "lag_selecionado": None,
            "bic": None,
            "rss_restrito": None,
            "rss_irrestrito": None,
            "estatistica_f": None,
            "graus_liberdade_num": None,
            "graus_liberdade_den": None,
            "p_valor": None,
            "q_valor_fdr": None,
            "significativo_bruto": 0,
            "significativo_fdr": 0,
            "resposta_acumulada": None,
            "efeito_positivo": 0,
            "observacao": None,
        }
        try:
            frame = construir_frame_direcao(
                pares[int(direcao.par_id)],
                direcao.origem,
                direcao.destino,
            )
            candidatos, selecionado = selecionar_lag(
                frame,
                max_lag,
                max_lag_amostra,
            )
            lag = int(selecionado["lag"])
            alvo, restrita, irrestrita, partes = matrizes_treino(
                frame,
                lag,
                max_lag_amostra,
            )
            _, beta_restrito, beta_irrestrito = ajustar_modelos(
                alvo,
                restrita,
                irrestrita,
                lag,
            )
            resposta = resposta_dinamica_assimetrica(beta_irrestrito, lag)
            base.update(
                {
                    **{
                        chave: selecionado[chave]
                        for chave in (
                            "n_observacoes",
                            "bic",
                            "rss_restrito",
                            "rss_irrestrito",
                            "estatistica_f",
                            "graus_liberdade_num",
                            "graus_liberdade_den",
                            "p_valor",
                        )
                    },
                    "lag_selecionado": lag,
                    "significativo_bruto": int(selecionado["p_valor"] < alpha),
                    "resposta_acumulada": resposta,
                    "efeito_positivo": int(resposta > 0),
                }
            )
            lags.extend(
                {"resultado_id": resultado_id, **candidato}
                for candidato in candidatos
            )
            coeficientes.append(
                {
                    "resultado_id": resultado_id,
                    "tipo_termo": "intercepto",
                    "defasagem": 0,
                    "coeficiente": float(beta_irrestrito[0]),
                }
            )
            tipos = ("destino", "origem_negativa", "origem_positiva")
            for indice_tipo, tipo in enumerate(tipos):
                inicio = 1 + indice_tipo * lag
                for deslocamento in range(1, lag + 1):
                    coeficientes.append(
                        {
                            "resultado_id": resultado_id,
                            "tipo_termo": tipo,
                            "defasagem": deslocamento,
                            "coeficiente": float(
                                beta_irrestrito[inicio + deslocamento - 1]
                            ),
                        }
                    )

            maior_alvo, maior_restrita, maior_irrestrita = max(
                partes,
                key=lambda parte: len(parte[0]),
            )
            diagnostico = diagnosticar_residuos(
                maior_alvo - maior_restrita @ beta_restrito,
                maior_alvo - maior_irrestrita @ beta_irrestrito,
                lag,
                alpha,
            )
            diagnosticos.append({"resultado_id": resultado_id, **diagnostico})
            avaliacoes.append(
                {
                    "resultado_id": resultado_id,
                    **avaliar_previsao(
                        frame,
                        lag,
                        alvo,
                        beta_restrito,
                        beta_irrestrito,
                    ),
                }
            )
            resultados.append(base)
        except (KeyError, ValueError, np.linalg.LinAlgError) as erro:
            base["status"] = "erro_modelo"
            base["observacao"] = str(erro)
            resultados.append(base)

    testados = [
        resultado
        for resultado in resultados
        if resultado["status"] == "testado"
    ]
    q_valores = corrigir_benjamini_hochberg(
        [float(resultado["p_valor"]) for resultado in testados]
    )
    for resultado, q_valor in zip(testados, q_valores, strict=True):
        resultado["q_valor_fdr"] = q_valor
        resultado["significativo_fdr"] = int(q_valor < alpha)

    resultados_por_id = {resultado["id"]: resultado for resultado in resultados}
    diagnosticos_por_id = {
        diagnostico["resultado_id"]: diagnostico for diagnostico in diagnosticos
    }
    for avaliacao in avaliacoes:
        resultado = resultados_por_id[avaliacao["resultado_id"]]
        diagnostico = diagnosticos_por_id[avaliacao["resultado_id"]]
        base_robusta = (
            bool(resultado["significativo_fdr"])
            and bool(resultado["efeito_positivo"])
            and diagnostico["status"] == "avaliado"
            and bool(diagnostico["sem_autocorrelacao_irrestrito"])
            and avaliacao["status"] == "avaliado"
        )
        avaliacao["relacao_robusta"] = int(
            base_robusta
            and bool(avaliacao["melhora_rmse"])
            and bool(avaliacao["melhora_mae"])
        )
        avaliacao["relacao_alta_robusta"] = int(
            base_robusta
            and bool(avaliacao["altas_avaliadas"])
            and bool(avaliacao["melhora_rmse_altas"])
            and bool(avaliacao["melhora_mae_altas"])
        )

    return {
        "resultados_assimetricos": pd.DataFrame(resultados),
        "resultados_lags_assimetricos": pd.DataFrame(lags),
        "coeficientes_assimetricos": pd.DataFrame(coeficientes),
        "diagnosticos_residuos": pd.DataFrame(diagnosticos),
        "avaliacoes_preditivas": pd.DataFrame(avaliacoes),
    }


def criar_esquema(conexao: sqlite3.Connection) -> None:
    conexao.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_assimetricas (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT,
            banco_origem TEXT NOT NULL,
            banco_origem_sha256 TEXT NOT NULL,
            execucao_granger_id INTEGER NOT NULL,
            parametros_json TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE resultados_assimetricos (
            id INTEGER PRIMARY KEY,
            resultado_granger_id INTEGER NOT NULL UNIQUE,
            par_id INTEGER NOT NULL,
            produto TEXT NOT NULL,
            origem TEXT NOT NULL,
            destino TEXT NOT NULL,
            status TEXT NOT NULL,
            n_observacoes INTEGER,
            lag_selecionado INTEGER,
            bic REAL,
            rss_restrito REAL,
            rss_irrestrito REAL,
            estatistica_f REAL,
            graus_liberdade_num INTEGER,
            graus_liberdade_den INTEGER,
            p_valor REAL,
            q_valor_fdr REAL,
            significativo_bruto INTEGER NOT NULL CHECK (
                significativo_bruto IN (0, 1)
            ),
            significativo_fdr INTEGER NOT NULL CHECK (
                significativo_fdr IN (0, 1)
            ),
            resposta_acumulada REAL,
            efeito_positivo INTEGER NOT NULL CHECK (efeito_positivo IN (0, 1)),
            observacao TEXT,
            UNIQUE (produto, origem, destino)
        );

        CREATE TABLE resultados_lags_assimetricos (
            resultado_id INTEGER NOT NULL,
            lag INTEGER NOT NULL,
            n_observacoes INTEGER NOT NULL,
            bic REAL NOT NULL,
            rss_restrito REAL NOT NULL,
            rss_irrestrito REAL NOT NULL,
            estatistica_f REAL NOT NULL,
            graus_liberdade_num INTEGER NOT NULL,
            graus_liberdade_den INTEGER NOT NULL,
            p_valor REAL NOT NULL,
            PRIMARY KEY (resultado_id, lag),
            FOREIGN KEY (resultado_id) REFERENCES resultados_assimetricos(id)
        );

        CREATE TABLE coeficientes_assimetricos (
            resultado_id INTEGER NOT NULL,
            tipo_termo TEXT NOT NULL,
            defasagem INTEGER NOT NULL,
            coeficiente REAL NOT NULL,
            PRIMARY KEY (resultado_id, tipo_termo, defasagem),
            FOREIGN KEY (resultado_id) REFERENCES resultados_assimetricos(id)
        );

        CREATE TABLE diagnosticos_residuos (
            resultado_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            n_observacoes INTEGER NOT NULL,
            lag_teste INTEGER,
            estatistica_ljung_box_restrito REAL,
            p_valor_ljung_box_restrito REAL,
            estatistica_ljung_box_irrestrito REAL,
            p_valor_ljung_box_irrestrito REAL,
            sem_autocorrelacao_irrestrito INTEGER NOT NULL CHECK (
                sem_autocorrelacao_irrestrito IN (0, 1)
            ),
            observacao TEXT,
            FOREIGN KEY (resultado_id) REFERENCES resultados_assimetricos(id)
        );

        CREATE TABLE avaliacoes_preditivas (
            resultado_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            n_observacoes INTEGER NOT NULL,
            rmse_restrito REAL,
            rmse_irrestrito REAL,
            ganho_rmse_percentual REAL,
            mae_restrito REAL,
            mae_irrestrito REAL,
            ganho_mae_percentual REAL,
            melhora_rmse INTEGER NOT NULL CHECK (melhora_rmse IN (0, 1)),
            melhora_mae INTEGER NOT NULL CHECK (melhora_mae IN (0, 1)),
            limite_alta_destino REAL,
            n_observacoes_altas INTEGER NOT NULL,
            altas_avaliadas INTEGER NOT NULL CHECK (altas_avaliadas IN (0, 1)),
            rmse_restrito_altas REAL,
            rmse_irrestrito_altas REAL,
            ganho_rmse_altas_percentual REAL,
            mae_restrito_altas REAL,
            mae_irrestrito_altas REAL,
            ganho_mae_altas_percentual REAL,
            melhora_rmse_altas INTEGER NOT NULL CHECK (
                melhora_rmse_altas IN (0, 1)
            ),
            melhora_mae_altas INTEGER NOT NULL CHECK (
                melhora_mae_altas IN (0, 1)
            ),
            relacao_robusta INTEGER NOT NULL CHECK (relacao_robusta IN (0, 1)),
            relacao_alta_robusta INTEGER NOT NULL CHECK (
                relacao_alta_robusta IN (0, 1)
            ),
            observacao TEXT,
            FOREIGN KEY (resultado_id) REFERENCES resultados_assimetricos(id)
        );

        CREATE TABLE resumo_assimetria (
            chave TEXT PRIMARY KEY,
            valor TEXT NOT NULL
        );
        """
    )


def criar_indices_e_views(conexao: sqlite3.Connection) -> None:
    conexao.executescript(
        """
        CREATE INDEX idx_assimetrico_fdr
            ON resultados_assimetricos(significativo_fdr, q_valor_fdr);
        CREATE INDEX idx_assimetrico_relacao
            ON resultados_assimetricos(produto, origem, destino);
        CREATE INDEX idx_assimetrico_robusta
            ON avaliacoes_preditivas(relacao_robusta, relacao_alta_robusta);

        CREATE VIEW relacoes_assimetricas_significativas AS
        SELECT * FROM resultados_assimetricos WHERE significativo_fdr = 1;

        CREATE VIEW relacoes_assimetricas_altas_robustas AS
        SELECT resultado.*
        FROM resultados_assimetricos AS resultado
        JOIN avaliacoes_preditivas AS avaliacao
          ON avaliacao.resultado_id = resultado.id
        WHERE avaliacao.relacao_alta_robusta = 1;
        """
    )


def construir_resumo(tabelas: dict[str, pd.DataFrame]) -> dict[str, int]:
    resultados = tabelas["resultados_assimetricos"]
    diagnosticos = tabelas["diagnosticos_residuos"]
    avaliacoes = tabelas["avaliacoes_preditivas"]
    testados = resultados[resultados["status"] == "testado"]
    significativos = testados[testados["significativo_fdr"] == 1]
    ids_significativos = set(significativos["id"])
    diagnosticos_fdr = (
        diagnosticos[diagnosticos["resultado_id"].isin(ids_significativos)]
        if not diagnosticos.empty
        else diagnosticos
    )
    avaliacoes_fdr = (
        avaliacoes[avaliacoes["resultado_id"].isin(ids_significativos)]
        if not avaliacoes.empty
        else avaliacoes
    )

    def somar(frame: pd.DataFrame, coluna: str) -> int:
        return int(frame[coluna].sum()) if coluna in frame else 0

    return {
        "direcoes_recebidas": int(len(resultados)),
        "direcoes_testadas": int(len(testados)),
        "erros_modelo": int((resultados["status"] == "erro_modelo").sum()),
        "significativas_sem_correcao": somar(testados, "significativo_bruto"),
        "significativas_fdr": int(len(significativos)),
        "efeitos_positivos_fdr": somar(significativos, "efeito_positivo"),
        "residuos_adequados_fdr": somar(
            diagnosticos_fdr,
            "sem_autocorrelacao_irrestrito",
        ),
        "melhora_rmse_fdr": somar(avaliacoes_fdr, "melhora_rmse"),
        "melhora_mae_fdr": somar(avaliacoes_fdr, "melhora_mae"),
        "melhora_rmse_altas_fdr": somar(
            avaliacoes_fdr,
            "melhora_rmse_altas",
        ),
        "melhora_mae_altas_fdr": somar(
            avaliacoes_fdr,
            "melhora_mae_altas",
        ),
        "relacoes_robustas": somar(avaliacoes, "relacao_robusta"),
        "relacoes_altas_robustas": somar(
            avaliacoes,
            "relacao_alta_robusta",
        ),
    }


def gravar_dataframe(
    conexao: sqlite3.Connection,
    tabela: str,
    frame: pd.DataFrame,
) -> None:
    if not frame.empty:
        frame.to_sql(tabela, conexao, if_exists="append", index=False)


def validar_saida(conexao: sqlite3.Connection) -> None:
    if conexao.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("falha de integridade na saída assimétrica")
    if conexao.execute("PRAGMA foreign_key_check").fetchall():
        raise RuntimeError("referências inválidas na saída assimétrica")
    invalidos = conexao.execute(
        """
        SELECT COUNT(*)
        FROM resultados_assimetricos
        WHERE status = 'testado'
          AND (p_valor NOT BETWEEN 0 AND 1 OR q_valor_fdr NOT BETWEEN 0 AND 1)
        """
    ).fetchone()[0]
    if invalidos:
        raise RuntimeError("resultados assimétricos inconsistentes")


def executar(args: argparse.Namespace) -> Path:
    entrada = args.database.resolve()
    saida = args.output_db.resolve()
    if entrada == saida:
        raise ValueError("a saída deve ser diferente da entrada")
    if saida.exists():
        raise FileExistsError(f"a saída já existe: {saida}")
    saida.parent.mkdir(parents=True, exist_ok=True)

    origem = abrir_somente_leitura(entrada)
    destino: sqlite3.Connection | None = None
    temporario: Path | None = None
    iniciada_em = datetime.now(UTC).isoformat()
    try:
        execucao_origem, parametros_origem = carregar_metadados_origem(origem)
        direcoes = carregar_direcoes(origem)
        series_pares = carregar_series_pares(origem)
        if direcoes.empty:
            raise ValueError("o banco de origem não possui direções testadas")
        max_lag = int(parametros_origem["max_lag"])
        max_lag_amostra = int(parametros_origem["max_lag_amostra"])
        alpha = float(parametros_origem["alpha"])

        print("Executando Granger assimétrico por direção...", flush=True)
        tabelas = executar_direcoes(
            direcoes,
            series_pares,
            max_lag,
            max_lag_amostra,
            alpha,
        )
        resumo = construir_resumo(tabelas)
        parametros = {
            "alpha": alpha,
            "ffill_limite": int(parametros_origem["ffill_limite"]),
            "max_lag": max_lag,
            "max_lag_amostra": max_lag_amostra,
            "selecao_lag": "BIC em amostra fixa pelo max_lag",
            "correcao_multiplos_testes": "Benjamini-Hochberg global",
            "modelo_restrito": (
                "intercepto + defasagens do destino + defasagens negativas "
                "da origem"
            ),
            "modelo_irrestrito": (
                "modelo restrito + defasagens positivas da origem"
            ),
            "horizonte_resposta": HORIZONTE_RESPOSTA,
            "criterio_efeito": "resposta dinâmica acumulada positiva",
            "data_inicio": parametros_origem["data_inicio"],
            "data_corte_treino": parametros_origem["data_corte_treino"],
            "data_fim_teste": parametros_origem["data_fim_teste"],
        }

        handle = tempfile.NamedTemporaryFile(
            prefix=f".{saida.stem}-",
            suffix=".tmp.sqlite",
            dir=saida.parent,
            delete=False,
        )
        temporario = Path(handle.name)
        handle.close()
        destino = sqlite3.connect(temporario)
        destino.execute("PRAGMA journal_mode = DELETE")
        destino.execute("PRAGMA synchronous = NORMAL")
        criar_esquema(destino)
        destino.execute(
            """
            INSERT INTO execucoes_assimetricas (
                id, schema_version, iniciada_em, banco_origem,
                banco_origem_sha256, execucao_granger_id, parametros_json,
                status
            ) VALUES (1, ?, ?, ?, ?, ?, ?, 'executando')
            """,
            (
                SCHEMA_VERSION,
                iniciada_em,
                str(args.database),
                sha256_arquivo(entrada),
                int(execucao_origem["id"]),
                json.dumps(parametros, ensure_ascii=False, sort_keys=True),
            ),
        )
        for tabela, frame in tabelas.items():
            gravar_dataframe(destino, tabela, frame)
        destino.executemany(
            "INSERT INTO resumo_assimetria (chave, valor) VALUES (?, ?)",
            [(chave, str(valor)) for chave, valor in sorted(resumo.items())],
        )
        criar_indices_e_views(destino)
        destino.execute(
            """
            UPDATE execucoes_assimetricas
            SET concluida_em = ?, status = 'concluida'
            WHERE id = 1
            """,
            (datetime.now(UTC).isoformat(),),
        )
        destino.commit()
        validar_saida(destino)
        destino.close()
        destino = None
        os.replace(temporario, saida)
        temporario = None

        print(f"Banco assimétrico criado em: {saida}", flush=True)
        for chave, valor in sorted(resumo.items()):
            print(f"  {chave}: {valor:,}", flush=True)
        return saida
    finally:
        origem.close()
        if destino is not None:
            destino.close()
        if temporario is not None and temporario.exists():
            temporario.unlink()


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    executar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
