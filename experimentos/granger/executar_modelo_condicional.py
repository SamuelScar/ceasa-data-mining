#!/usr/bin/env python3
"""Executa o modelo do experimento encerrado de Granger condicional."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as distribuicao_f

from granger.executar_granger import (
    MINIMO_OBSERVACOES_ALTAS_TESTE,
    MINIMO_OBSERVACOES_TESTE,
    PERCENTIL_ALTA_DESTINO,
    calcular_metricas_erros,
    corrigir_benjamini_hochberg,
    diagnosticar_residuos,
    identificar_segmentos,
    testar_estacionariedade,
)


SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1
MINIMO_MERCADOS_REDE = 3
OBSERVACOES_POR_COEFICIENTE = 6


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite do Granger bivariado")
    parser.add_argument("--output-db", type=Path, required=True)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument(
        "--output-config-redes",
        type=Path,
        help="seleciona as redes pela cobertura desta entrada e grava o JSON",
    )
    grupo.add_argument(
        "--config-redes",
        type=Path,
        help="reutiliza redes anteriormente selecionadas",
    )
    parser.add_argument(
        "--data-inicio-treino",
        help="restringe o início do treino sem alterar o banco de origem",
    )
    parser.add_argument(
        "--data-corte-treino",
        help="restringe o fim do treino sem alterar o banco de origem",
    )
    parser.add_argument(
        "--data-fim-teste",
        help="restringe o fim da avaliação sem alterar o banco de origem",
    )
    return parser


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


def sha256_arquivo(caminho: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while bloco := arquivo.read(chunk_size):
            digest.update(bloco)
    return digest.hexdigest()


def json_canonico(valor: object) -> str:
    return json.dumps(
        valor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def hash_configuracao(configuracao: dict[str, Any]) -> str:
    return hashlib.sha256(json_canonico(configuracao).encode()).hexdigest()


def gravar_json_atomico(caminho: Path, valor: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(valor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def carregar_metadados_origem(
    conexao: sqlite3.Connection,
) -> tuple[sqlite3.Row, dict[str, Any]]:
    integridade = conexao.execute("PRAGMA quick_check").fetchone()[0]
    if integridade != "ok":
        raise RuntimeError(
            f"falha de integridade no banco de origem: {integridade}"
        )
    registro = conexao.execute(
        """
        SELECT *
        FROM execucoes_granger
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if registro is None or registro["status"] != "concluida":
        raise RuntimeError("a execução de Granger de origem não está concluída")
    if int(registro["schema_version"]) < 3:
        raise RuntimeError(
            "o banco de origem não possui os diagnósticos necessários"
        )
    parametros = json.loads(registro["parametros_json"])
    if int(parametros["ffill_limite"]) != 3:
        raise ValueError("o experimento condicional exige ffill_limite=3")
    if int(parametros["max_lag"]) != int(parametros["max_lag_amostra"]):
        raise ValueError("max_lag e max_lag_amostra devem ser iguais na origem")
    return registro, parametros


def carregar_series(conexao: sqlite3.Connection) -> pd.DataFrame:
    series = pd.read_sql_query(
        """
        SELECT produto, mercado, data_cotacao, periodo, retorno_log
        FROM series_temporais
        ORDER BY produto, data_cotacao, mercado
        """,
        conexao,
    )
    series["data_cotacao"] = pd.to_datetime(series["data_cotacao"])
    return series


def aplicar_recorte_temporal(
    series: pd.DataFrame,
    parametros_origem: dict[str, Any],
    data_inicio_treino: str | None,
    data_corte_treino: str | None,
    data_fim_teste: str | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    datas = {
        "data_inicio_treino": data_inicio_treino
        or parametros_origem["data_inicio"],
        "data_corte_treino": data_corte_treino
        or parametros_origem["data_corte_treino"],
        "data_fim_teste": data_fim_teste
        or parametros_origem["data_fim_teste"],
    }
    inicio = pd.Timestamp(datas["data_inicio_treino"])
    corte = pd.Timestamp(datas["data_corte_treino"])
    fim = pd.Timestamp(datas["data_fim_teste"])
    inicio_origem = pd.Timestamp(parametros_origem["data_inicio"])
    corte_origem = pd.Timestamp(parametros_origem["data_corte_treino"])
    fim_origem = pd.Timestamp(parametros_origem["data_fim_teste"])
    if not inicio <= corte < fim:
        raise ValueError("o recorte deve respeitar início <= corte < fim")
    if inicio < inicio_origem:
        raise ValueError("o início do recorte antecede a série de origem")
    if corte > corte_origem:
        raise ValueError("o treino do recorte ultrapassa o treino da origem")
    if fim > fim_origem:
        raise ValueError("o teste do recorte ultrapassa a avaliação da origem")

    recorte = series[
        series["data_cotacao"].between(inicio, fim, inclusive="both")
    ].copy()
    if recorte.empty:
        raise ValueError("o recorte temporal não possui observações")
    recorte["periodo"] = np.where(
        recorte["data_cotacao"] <= corte,
        "treino",
        "teste",
    )
    return recorte, {
        chave: pd.Timestamp(valor).date().isoformat()
        for chave, valor in datas.items()
    }


def construir_frame_rede(
    series: pd.DataFrame,
    produto: str,
    mercados: tuple[str, ...],
) -> pd.DataFrame:
    produto_series = series[series["produto"] == produto]
    retornos = produto_series.pivot(
        index="data_cotacao",
        columns="mercado",
        values="retorno_log",
    ).reindex(columns=list(mercados))
    periodos = (
        produto_series[["data_cotacao", "periodo"]]
        .drop_duplicates()
        .set_index("data_cotacao")["periodo"]
        .reindex(retornos.index)
    )
    completo = retornos.notna().all(axis=1)
    frame = retornos.copy()
    frame["periodo"] = periodos
    frame["segmento_rede"] = identificar_segmentos(completo)
    return frame.loc[completo].copy()


def tamanhos_segmentos_treino(frame: pd.DataFrame) -> list[int]:
    treino = frame[frame["periodo"] == "treino"]
    return treino.groupby("segmento_rede", sort=True).size().astype(int).tolist()


def contar_observacoes_utilizaveis(
    frame: pd.DataFrame,
    max_lag: int,
) -> int:
    return sum(
        max(0, tamanho - max_lag)
        for tamanho in tamanhos_segmentos_treino(frame)
    )


def minimo_observacoes_rede(n_mercados: int, max_lag: int) -> int:
    coeficientes_equacao = 1 + n_mercados * max_lag
    return OBSERVACOES_POR_COEFICIENTE * coeficientes_equacao


def selecionar_redes(
    series: pd.DataFrame,
    max_lag: int,
    metadados_origem: dict[str, Any],
) -> dict[str, Any]:
    redes: list[dict[str, Any]] = []
    for produto in sorted(series["produto"].unique()):
        mercados = tuple(
            sorted(series.loc[series["produto"] == produto, "mercado"].unique())
        )
        selecionada: dict[str, Any] | None = None
        for n_mercados in range(len(mercados), MINIMO_MERCADOS_REDE - 1, -1):
            minimo = minimo_observacoes_rede(n_mercados, max_lag)
            candidatos: list[dict[str, Any]] = []
            for conjunto in combinations(mercados, n_mercados):
                frame = construir_frame_rede(series, produto, conjunto)
                tamanhos = tamanhos_segmentos_treino(frame)
                n_observacoes = contar_observacoes_utilizaveis(frame, max_lag)
                if n_observacoes < minimo:
                    continue
                candidatos.append(
                    {
                        "produto": produto,
                        "mercados": list(conjunto),
                        "n_mercados": n_mercados,
                        "n_observacoes_selecao": n_observacoes,
                        "segmentos_treino_selecao": len(tamanhos),
                        "maior_segmento_treino_selecao": max(
                            tamanhos,
                            default=0,
                        ),
                        "minimo_observacoes": minimo,
                    }
                )
            if candidatos:
                selecionada = min(
                    candidatos,
                    key=lambda rede: (
                        -rede["n_observacoes_selecao"],
                        rede["mercados"],
                    ),
                )
                break

        redes.append(
            selecionada
            or {
                "produto": produto,
                "mercados": [],
                "n_mercados": 0,
                "n_observacoes_selecao": 0,
                "segmentos_treino_selecao": 0,
                "maior_segmento_treino_selecao": 0,
                "minimo_observacoes": 0,
                "motivo_exclusao": "cobertura_insuficiente",
            }
        )

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "metodo": "maior rede por cobertura no primeiro treino",
        "max_lag": max_lag,
        "max_lag_amostra": max_lag,
        "minimo_mercados": MINIMO_MERCADOS_REDE,
        "observacoes_por_coeficiente": OBSERVACOES_POR_COEFICIENTE,
        "selecao_origem": metadados_origem,
        "redes": redes,
    }


def carregar_configuracao(caminho: Path) -> dict[str, Any]:
    if not caminho.is_file():
        raise FileNotFoundError(f"configuração de redes não encontrada: {caminho}")
    configuracao = json.loads(caminho.read_text(encoding="utf-8"))
    if int(configuracao.get("schema_version", 0)) != CONFIG_SCHEMA_VERSION:
        raise ValueError("schema da configuração de redes incompatível")
    return configuracao


def validar_configuracao(
    configuracao: dict[str, Any],
    series: pd.DataFrame,
    max_lag: int,
) -> None:
    if int(configuracao["max_lag"]) != max_lag:
        raise ValueError("a configuração de redes usa outro max_lag")
    produtos_disponiveis = set(series["produto"].unique())
    for rede in configuracao["redes"]:
        if rede["produto"] not in produtos_disponiveis:
            raise ValueError(f"produto ausente na entrada: {rede['produto']}")
        mercados_disponiveis = set(
            series.loc[
                series["produto"] == rede["produto"],
                "mercado",
            ].unique()
        )
        if not set(rede["mercados"]).issubset(mercados_disponiveis):
            raise ValueError(
                f"mercado da rede de {rede['produto']} ausente na entrada"
            )


def partes_treino(
    frame: pd.DataFrame,
    mercados: tuple[str, ...],
    lag: int,
    max_lag_amostra: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    partes: list[tuple[np.ndarray, np.ndarray]] = []
    treino = frame[frame["periodo"] == "treino"]
    for _, segmento in treino.groupby("segmento_rede", sort=True):
        valores = segmento[list(mercados)].to_numpy(dtype=float)
        if len(valores) <= max_lag_amostra:
            continue
        alvo = valores[max_lag_amostra:]
        defasagens = np.column_stack(
            [
                valores[
                    max_lag_amostra - deslocamento : len(valores) - deslocamento,
                    indice,
                ]
                for indice in range(len(mercados))
                for deslocamento in range(1, lag + 1)
            ]
        )
        matriz = np.column_stack([np.ones(len(alvo)), defasagens])
        partes.append((alvo, matriz))
    return partes


def matrizes_treino(
    frame: pd.DataFrame,
    mercados: tuple[str, ...],
    lag: int,
    max_lag_amostra: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    partes = partes_treino(frame, mercados, lag, max_lag_amostra)
    if not partes:
        raise ValueError("nenhum segmento possui histórico suficiente")
    return (
        np.vstack([alvo for alvo, _ in partes]),
        np.vstack([matriz for _, matriz in partes]),
        partes,
    )


def selecionar_lag_rede(
    frame: pd.DataFrame,
    mercados: tuple[str, ...],
    max_lag: int,
    max_lag_amostra: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidatos: list[dict[str, Any]] = []
    for lag in range(1, max_lag + 1):
        alvo, matriz, _ = matrizes_treino(
            frame,
            mercados,
            lag,
            max_lag_amostra,
        )
        n_observacoes = len(alvo)
        n_parametros_equacao = matriz.shape[1]
        if n_observacoes <= n_parametros_equacao:
            raise ValueError("amostra insuficiente para o VAR")
        if np.linalg.matrix_rank(matriz) < n_parametros_equacao:
            raise ValueError("matriz do VAR sem posto completo")
        coeficientes = np.linalg.lstsq(matriz, alvo, rcond=None)[0]
        residuos = alvo - matriz @ coeficientes
        covariancia = residuos.T @ residuos / n_observacoes
        sinal, log_determinante = np.linalg.slogdet(covariancia)
        if sinal <= 0 or not np.isfinite(log_determinante):
            raise ValueError("covariância residual singular no VAR")
        n_parametros_sistema = len(mercados) * n_parametros_equacao
        bic = float(
            log_determinante
            + np.log(n_observacoes) * n_parametros_sistema / n_observacoes
        )
        candidatos.append(
            {
                "lag": lag,
                "n_observacoes": n_observacoes,
                "n_parametros_equacao": n_parametros_equacao,
                "n_parametros_sistema": n_parametros_sistema,
                "log_determinante_covariancia": float(log_determinante),
                "bic": bic,
            }
        )
    return candidatos, min(candidatos, key=lambda item: (item["bic"], item["lag"]))


def indices_modelo_restrito(
    mercados: tuple[str, ...],
    origem: str,
    lag: int,
) -> list[int]:
    indice_origem = mercados.index(origem)
    inicio = 1 + indice_origem * lag
    fim = inicio + lag
    return [0, *range(1, inicio), *range(fim, 1 + len(mercados) * lag)]


def testar_direcao_condicional(
    alvo: np.ndarray,
    matriz_irrestrita: np.ndarray,
    mercados: tuple[str, ...],
    origem: str,
    destino: str,
    lag: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = alvo[:, mercados.index(destino)]
    indices_restritos = indices_modelo_restrito(mercados, origem, lag)
    matriz_restrita = matriz_irrestrita[:, indices_restritos]
    beta_restrito = np.linalg.lstsq(matriz_restrita, y, rcond=None)[0]
    beta_irrestrito = np.linalg.lstsq(matriz_irrestrita, y, rcond=None)[0]
    residuo_restrito = y - matriz_restrita @ beta_restrito
    residuo_irrestrito = y - matriz_irrestrita @ beta_irrestrito
    rss_restrito = float(residuo_restrito @ residuo_restrito)
    rss_irrestrito = float(residuo_irrestrito @ residuo_irrestrito)
    graus_den = len(y) - matriz_irrestrita.shape[1]
    if graus_den <= 0 or rss_irrestrito <= 0:
        raise ValueError("graus de liberdade insuficientes no teste condicional")
    diferenca = max(0.0, rss_restrito - rss_irrestrito)
    estatistica_f = (diferenca / lag) / (rss_irrestrito / graus_den)
    p_valor = float(distribuicao_f.sf(estatistica_f, lag, graus_den))
    return (
        {
            "n_observacoes": len(y),
            "rss_restrito": rss_restrito,
            "rss_irrestrito": rss_irrestrito,
            "estatistica_f": estatistica_f,
            "graus_liberdade_num": lag,
            "graus_liberdade_den": graus_den,
            "p_valor": p_valor,
        },
        beta_restrito,
        beta_irrestrito,
        residuo_restrito,
        residuo_irrestrito,
    )


def matrizes_teste(
    frame: pd.DataFrame,
    mercados: tuple[str, ...],
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    alvos: list[np.ndarray] = []
    matrizes: list[np.ndarray] = []
    for _, segmento in frame.groupby("segmento_rede", sort=True):
        valores = segmento[list(mercados)].to_numpy(dtype=float)
        periodos = segmento["periodo"].to_numpy()
        posicoes = np.flatnonzero(periodos == "teste")
        posicoes = posicoes[posicoes >= lag]
        if not len(posicoes):
            continue
        alvo = valores[posicoes]
        defasagens = np.column_stack(
            [
                valores[posicoes - deslocamento, indice]
                for indice in range(len(mercados))
                for deslocamento in range(1, lag + 1)
            ]
        )
        alvos.append(alvo)
        matrizes.append(np.column_stack([np.ones(len(alvo)), defasagens]))
    if not alvos:
        raise ValueError("nenhum ponto de teste possui histórico suficiente")
    return np.vstack(alvos), np.vstack(matrizes)


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
    alvo_teste: np.ndarray,
    matriz_teste: np.ndarray,
    alvo_treino: np.ndarray,
    mercados: tuple[str, ...],
    origem: str,
    destino: str,
    lag: int,
    beta_restrito: np.ndarray,
    beta_irrestrito: np.ndarray,
) -> dict[str, Any]:
    destino_indice = mercados.index(destino)
    y = alvo_teste[:, destino_indice]
    if len(y) < MINIMO_OBSERVACOES_TESTE:
        avaliacao = avaliacao_sem_metricas(
            "amostra_insuficiente",
            f"mínimo de {MINIMO_OBSERVACOES_TESTE} observações de teste",
        )
        avaliacao["n_observacoes"] = len(y)
        return avaliacao
    indices_restritos = indices_modelo_restrito(mercados, origem, lag)
    erro_restrito = y - matriz_teste[:, indices_restritos] @ beta_restrito
    erro_irrestrito = y - matriz_teste @ beta_irrestrito
    metricas = calcular_metricas_erros(erro_restrito, erro_irrestrito)
    limite_alta = max(
        0.0,
        float(np.quantile(alvo_treino[:, destino_indice], PERCENTIL_ALTA_DESTINO)),
    )
    mascara_altas = y > limite_alta
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
        "n_observacoes": len(y),
        **metricas,
        "limite_alta_destino": limite_alta,
        "n_observacoes_altas": n_altas,
        "altas_avaliadas": int(altas_avaliadas),
        **metricas_altas,
        "relacao_robusta": 0,
        "relacao_alta_robusta": 0,
        "observacao": "previsão condicional um passo à frente",
    }


def executar_redes(
    series: pd.DataFrame,
    configuracao: dict[str, Any],
    alpha: float,
) -> dict[str, pd.DataFrame]:
    max_lag = int(configuracao["max_lag"])
    max_lag_amostra = int(configuracao["max_lag_amostra"])
    redes: list[dict[str, Any]] = []
    estacionariedade: list[dict[str, Any]] = []
    lags: list[dict[str, Any]] = []
    resultados: list[dict[str, Any]] = []
    coeficientes: list[dict[str, Any]] = []
    diagnosticos: list[dict[str, Any]] = []
    avaliacoes: list[dict[str, Any]] = []
    resultado_id = 0

    for rede_id, rede_config in enumerate(configuracao["redes"], start=1):
        produto = rede_config["produto"]
        mercados = tuple(rede_config["mercados"])
        rede_base = {
            "id": rede_id,
            "produto": produto,
            "mercados_json": json.dumps(mercados, ensure_ascii=False),
            "n_mercados": len(mercados),
            "n_observacoes_selecao": rede_config["n_observacoes_selecao"],
            "n_observacoes": 0,
            "segmentos_treino": 0,
            "maior_segmento_treino": 0,
            "lag_selecionado": None,
            "bic": None,
            "status": "cobertura_insuficiente" if not mercados else "selecionada",
            "observacao": rede_config.get("motivo_exclusao"),
        }
        if not mercados:
            redes.append(rede_base)
            continue

        frame = construir_frame_rede(series, produto, mercados)
        tamanhos = tamanhos_segmentos_treino(frame)
        rede_base.update(
            {
                "n_observacoes": contar_observacoes_utilizaveis(
                    frame,
                    max_lag_amostra,
                ),
                "segmentos_treino": len(tamanhos),
                "maior_segmento_treino": max(tamanhos, default=0),
            }
        )
        treino = frame[frame["periodo"] == "treino"]
        if treino.empty:
            maior_segmento = treino
        else:
            maior_id = treino.groupby("segmento_rede").size().idxmax()
            maior_segmento = treino[treino["segmento_rede"] == maior_id]
        rede_estacionaria = True
        for mercado in mercados:
            diagnostico = testar_estacionariedade(maior_segmento[mercado], alpha)
            rede_estacionaria = rede_estacionaria and diagnostico.estacionaria
            estacionariedade.append(
                {
                    "rede_id": rede_id,
                    "mercado": mercado,
                    "n_observacoes": diagnostico.n_observacoes,
                    "p_valor_adf": diagnostico.p_valor_adf,
                    "p_valor_kpss": diagnostico.p_valor_kpss,
                    "estacionaria": int(diagnostico.estacionaria),
                    "status": diagnostico.status,
                }
            )

        if not rede_estacionaria:
            rede_base["status"] = "nao_estacionaria"
            rede_base["observacao"] = "ao menos uma série reprovou ADF ou KPSS"
            redes.append(rede_base)
            continue

        try:
            candidatos, selecionado = selecionar_lag_rede(
                frame,
                mercados,
                max_lag,
                max_lag_amostra,
            )
        except (ValueError, np.linalg.LinAlgError) as erro:
            rede_base["status"] = "erro_modelo"
            rede_base["observacao"] = str(erro)
            redes.append(rede_base)
            continue

        lag = int(selecionado["lag"])
        rede_base.update(
            {
                "lag_selecionado": lag,
                "bic": selecionado["bic"],
                "status": "testada",
                "observacao": None,
            }
        )
        redes.append(rede_base)
        lags.extend({"rede_id": rede_id, **candidato} for candidato in candidatos)
        alvo_treino, matriz_treino, partes = matrizes_treino(
            frame,
            mercados,
            lag,
            max_lag_amostra,
        )
        try:
            alvo_teste, matriz_teste = matrizes_teste(frame, mercados, lag)
        except ValueError:
            alvo_teste = np.empty((0, len(mercados)))
            matriz_teste = np.empty((0, matriz_treino.shape[1]))

        for origem in mercados:
            for destino in mercados:
                if origem == destino:
                    continue
                resultado_id += 1
                teste, beta_restrito, beta_irrestrito, _, _ = (
                    testar_direcao_condicional(
                        alvo_treino,
                        matriz_treino,
                        mercados,
                        origem,
                        destino,
                        lag,
                    )
                )
                resultados.append(
                    {
                        "id": resultado_id,
                        "rede_id": rede_id,
                        "produto": produto,
                        "origem": origem,
                        "destino": destino,
                        "status": "testado",
                        **teste,
                        "q_valor_fdr": None,
                        "significativo_bruto": int(teste["p_valor"] < alpha),
                        "significativo_fdr": 0,
                    }
                )
                coeficientes.append(
                    {
                        "resultado_id": resultado_id,
                        "mercado_preditor": "intercepto",
                        "defasagem": 0,
                        "coeficiente": float(beta_irrestrito[0]),
                    }
                )
                for mercado_indice, mercado in enumerate(mercados):
                    inicio = 1 + mercado_indice * lag
                    for deslocamento in range(1, lag + 1):
                        coeficientes.append(
                            {
                                "resultado_id": resultado_id,
                                "mercado_preditor": mercado,
                                "defasagem": deslocamento,
                                "coeficiente": float(
                                    beta_irrestrito[inicio + deslocamento - 1]
                                ),
                            }
                        )

                maior_alvo, maior_matriz = max(
                    partes,
                    key=lambda parte: len(parte[0]),
                )
                destino_indice = mercados.index(destino)
                indices_restritos = indices_modelo_restrito(mercados, origem, lag)
                diagnostico = diagnosticar_residuos(
                    maior_alvo[:, destino_indice]
                    - maior_matriz[:, indices_restritos] @ beta_restrito,
                    maior_alvo[:, destino_indice] - maior_matriz @ beta_irrestrito,
                    lag,
                    alpha,
                )
                diagnosticos.append({"resultado_id": resultado_id, **diagnostico})
                if len(alvo_teste):
                    avaliacao = avaliar_previsao(
                        alvo_teste,
                        matriz_teste,
                        alvo_treino,
                        mercados,
                        origem,
                        destino,
                        lag,
                        beta_restrito,
                        beta_irrestrito,
                    )
                else:
                    avaliacao = avaliacao_sem_metricas(
                        "amostra_insuficiente",
                        "nenhum ponto de teste possui histórico suficiente",
                    )
                avaliacoes.append({"resultado_id": resultado_id, **avaliacao})

    q_valores = corrigir_benjamini_hochberg(
        [float(resultado["p_valor"]) for resultado in resultados]
    )
    resultados_por_id: dict[int, dict[str, Any]] = {}
    for resultado, q_valor in zip(resultados, q_valores, strict=True):
        resultado["q_valor_fdr"] = q_valor
        resultado["significativo_fdr"] = int(q_valor < alpha)
        resultados_por_id[resultado["id"]] = resultado
    diagnosticos_por_id = {
        diagnostico["resultado_id"]: diagnostico
        for diagnostico in diagnosticos
    }
    for avaliacao in avaliacoes:
        resultado = resultados_por_id[avaliacao["resultado_id"]]
        diagnostico = diagnosticos_por_id[avaliacao["resultado_id"]]
        base_robusta = (
            bool(resultado["significativo_fdr"])
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
        "redes_produtos": pd.DataFrame(redes),
        "diagnosticos_estacionariedade": pd.DataFrame(estacionariedade),
        "resultados_lags_redes": pd.DataFrame(lags),
        "resultados_condicionais": pd.DataFrame(resultados),
        "coeficientes_condicionais": pd.DataFrame(coeficientes),
        "diagnosticos_residuos": pd.DataFrame(diagnosticos),
        "avaliacoes_preditivas": pd.DataFrame(avaliacoes),
    }


def criar_esquema(conexao: sqlite3.Connection) -> None:
    conexao.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_condicionais (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT,
            banco_origem TEXT NOT NULL,
            banco_origem_sha256 TEXT NOT NULL,
            configuracao_redes_sha256 TEXT NOT NULL,
            parametros_json TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE redes_produtos (
            id INTEGER PRIMARY KEY,
            produto TEXT NOT NULL UNIQUE,
            mercados_json TEXT NOT NULL,
            n_mercados INTEGER NOT NULL,
            n_observacoes_selecao INTEGER NOT NULL,
            n_observacoes INTEGER NOT NULL,
            segmentos_treino INTEGER NOT NULL,
            maior_segmento_treino INTEGER NOT NULL,
            lag_selecionado INTEGER,
            bic REAL,
            status TEXT NOT NULL,
            observacao TEXT
        );

        CREATE TABLE diagnosticos_estacionariedade (
            rede_id INTEGER NOT NULL,
            mercado TEXT NOT NULL,
            n_observacoes INTEGER NOT NULL,
            p_valor_adf REAL,
            p_valor_kpss REAL,
            estacionaria INTEGER NOT NULL CHECK (estacionaria IN (0, 1)),
            status TEXT NOT NULL,
            PRIMARY KEY (rede_id, mercado),
            FOREIGN KEY (rede_id) REFERENCES redes_produtos(id)
        );

        CREATE TABLE resultados_lags_redes (
            rede_id INTEGER NOT NULL,
            lag INTEGER NOT NULL,
            n_observacoes INTEGER NOT NULL,
            n_parametros_equacao INTEGER NOT NULL,
            n_parametros_sistema INTEGER NOT NULL,
            log_determinante_covariancia REAL NOT NULL,
            bic REAL NOT NULL,
            PRIMARY KEY (rede_id, lag),
            FOREIGN KEY (rede_id) REFERENCES redes_produtos(id)
        );

        CREATE TABLE resultados_condicionais (
            id INTEGER PRIMARY KEY,
            rede_id INTEGER NOT NULL,
            produto TEXT NOT NULL,
            origem TEXT NOT NULL,
            destino TEXT NOT NULL,
            status TEXT NOT NULL,
            n_observacoes INTEGER NOT NULL,
            rss_restrito REAL NOT NULL,
            rss_irrestrito REAL NOT NULL,
            estatistica_f REAL NOT NULL,
            graus_liberdade_num INTEGER NOT NULL,
            graus_liberdade_den INTEGER NOT NULL,
            p_valor REAL NOT NULL,
            q_valor_fdr REAL NOT NULL,
            significativo_bruto INTEGER NOT NULL CHECK (
                significativo_bruto IN (0, 1)
            ),
            significativo_fdr INTEGER NOT NULL CHECK (
                significativo_fdr IN (0, 1)
            ),
            UNIQUE (produto, origem, destino),
            FOREIGN KEY (rede_id) REFERENCES redes_produtos(id)
        );

        CREATE TABLE coeficientes_condicionais (
            resultado_id INTEGER NOT NULL,
            mercado_preditor TEXT NOT NULL,
            defasagem INTEGER NOT NULL,
            coeficiente REAL NOT NULL,
            PRIMARY KEY (resultado_id, mercado_preditor, defasagem),
            FOREIGN KEY (resultado_id) REFERENCES resultados_condicionais(id)
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
            FOREIGN KEY (resultado_id) REFERENCES resultados_condicionais(id)
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
            FOREIGN KEY (resultado_id) REFERENCES resultados_condicionais(id)
        );

        CREATE TABLE resumo_condicional (
            chave TEXT PRIMARY KEY,
            valor TEXT NOT NULL
        );
        """
    )


def criar_indices_e_views(conexao: sqlite3.Connection) -> None:
    conexao.executescript(
        """
        CREATE INDEX idx_condicional_fdr
            ON resultados_condicionais(significativo_fdr, q_valor_fdr);
        CREATE INDEX idx_condicional_produto
            ON resultados_condicionais(produto, origem, destino);
        CREATE INDEX idx_condicional_robusta
            ON avaliacoes_preditivas(relacao_robusta, relacao_alta_robusta);

        CREATE VIEW relacoes_condicionais_significativas AS
        SELECT * FROM resultados_condicionais WHERE significativo_fdr = 1;

        CREATE VIEW relacoes_condicionais_robustas AS
        SELECT resultado.*
        FROM resultados_condicionais AS resultado
        JOIN avaliacoes_preditivas AS avaliacao
          ON avaliacao.resultado_id = resultado.id
        WHERE avaliacao.relacao_robusta = 1;

        CREATE VIEW relacoes_condicionais_altas_robustas AS
        SELECT resultado.*
        FROM resultados_condicionais AS resultado
        JOIN avaliacoes_preditivas AS avaliacao
          ON avaliacao.resultado_id = resultado.id
        WHERE avaliacao.relacao_alta_robusta = 1;
        """
    )


def construir_resumo(tabelas: dict[str, pd.DataFrame]) -> dict[str, int]:
    redes = tabelas["redes_produtos"]
    resultados = tabelas["resultados_condicionais"]
    diagnosticos = tabelas["diagnosticos_residuos"]
    avaliacoes = tabelas["avaliacoes_preditivas"]
    significativos = resultados[resultados["significativo_fdr"] == 1]
    ids_significativos = set(significativos["id"])
    diagnosticos_fdr = diagnosticos[
        diagnosticos["resultado_id"].isin(ids_significativos)
    ]
    avaliacoes_fdr = avaliacoes[
        avaliacoes["resultado_id"].isin(ids_significativos)
    ]
    return {
        "produtos_configurados": int((redes["n_mercados"] > 0).sum()),
        "produtos_testados": int((redes["status"] == "testada").sum()),
        "direcoes_testadas": int(len(resultados)),
        "significativas_sem_correcao": int(resultados["significativo_bruto"].sum()),
        "significativas_fdr": int(len(significativos)),
        "residuos_adequados_fdr": int(
            diagnosticos_fdr["sem_autocorrelacao_irrestrito"].sum()
        ),
        "melhora_rmse_fdr": int(avaliacoes_fdr["melhora_rmse"].sum()),
        "melhora_mae_fdr": int(avaliacoes_fdr["melhora_mae"].sum()),
        "melhora_rmse_altas_fdr": int(
            avaliacoes_fdr["melhora_rmse_altas"].sum()
        ),
        "melhora_mae_altas_fdr": int(
            avaliacoes_fdr["melhora_mae_altas"].sum()
        ),
        "relacoes_robustas": int(avaliacoes["relacao_robusta"].sum()),
        "relacoes_altas_robustas": int(
            avaliacoes["relacao_alta_robusta"].sum()
        ),
    }


def validar_saida(conexao: sqlite3.Connection) -> None:
    if conexao.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("falha de integridade na saída condicional")
    if conexao.execute("PRAGMA foreign_key_check").fetchall():
        raise RuntimeError("referências inválidas na saída condicional")
    invalidos = conexao.execute(
        """
        SELECT COUNT(*)
        FROM resultados_condicionais
        WHERE q_valor_fdr NOT BETWEEN 0 AND 1
           OR p_valor NOT BETWEEN 0 AND 1
           OR rss_irrestrito > rss_restrito + 1e-12
        """
    ).fetchone()[0]
    if invalidos:
        raise RuntimeError("resultados condicionais inconsistentes")


def gravar_dataframe(
    conexao: sqlite3.Connection,
    tabela: str,
    frame: pd.DataFrame,
) -> None:
    if not frame.empty:
        frame.to_sql(tabela, conexao, if_exists="append", index=False)


def executar(args: argparse.Namespace) -> Path:
    entrada = args.database.resolve()
    saida = args.output_db.resolve()
    if entrada == saida:
        raise ValueError("a saída deve ser diferente da entrada")
    if saida.exists():
        raise FileExistsError(f"a saída já existe: {saida}")
    saida.parent.mkdir(parents=True, exist_ok=True)

    origem = abrir_somente_leitura(entrada)
    temporario: Path | None = None
    destino: sqlite3.Connection | None = None
    inicio = datetime.now(UTC).isoformat()
    try:
        _, parametros_origem = carregar_metadados_origem(origem)
        series = carregar_series(origem)
        series, recorte_temporal = aplicar_recorte_temporal(
            series,
            parametros_origem,
            args.data_inicio_treino,
            args.data_corte_treino,
            args.data_fim_teste,
        )
        max_lag = int(parametros_origem["max_lag"])
        if args.output_config_redes is not None:
            configuracao = selecionar_redes(
                series,
                max_lag,
                {
                    "banco": str(args.database),
                    "banco_sha256": sha256_arquivo(entrada),
                    "data_corte_treino": parametros_origem["data_corte_treino"],
                },
            )
        else:
            configuracao = carregar_configuracao(args.config_redes)
        validar_configuracao(configuracao, series, max_lag)

        print("Executando redes condicionais por produto...", flush=True)
        tabelas = executar_redes(
            series,
            configuracao,
            float(parametros_origem["alpha"]),
        )
        resumo = construir_resumo(tabelas)
        parametros = {
            "alpha": parametros_origem["alpha"],
            "ffill_limite": parametros_origem["ffill_limite"],
            "max_lag": max_lag,
            "max_lag_amostra": configuracao["max_lag_amostra"],
            "ordem_var": "BIC comum da rede em amostra fixa",
            "correcao_multiplos_testes": "Benjamini-Hochberg global",
            "criterio_rede": configuracao["metodo"],
            "observacoes_por_coeficiente": configuracao[
                "observacoes_por_coeficiente"
            ],
            **recorte_temporal,
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
            INSERT INTO execucoes_condicionais (
                id, schema_version, iniciada_em, banco_origem,
                banco_origem_sha256, configuracao_redes_sha256,
                parametros_json, status
            ) VALUES (1, ?, ?, ?, ?, ?, ?, 'executando')
            """,
            (
                SCHEMA_VERSION,
                inicio,
                str(args.database),
                sha256_arquivo(entrada),
                hash_configuracao(configuracao),
                json.dumps(parametros, ensure_ascii=False, sort_keys=True),
            ),
        )
        for tabela, frame in tabelas.items():
            gravar_dataframe(destino, tabela, frame)
        destino.executemany(
            "INSERT INTO resumo_condicional (chave, valor) VALUES (?, ?)",
            [(chave, str(valor)) for chave, valor in sorted(resumo.items())],
        )
        criar_indices_e_views(destino)
        destino.execute(
            """
            UPDATE execucoes_condicionais
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
        if args.output_config_redes is not None:
            gravar_json_atomico(args.output_config_redes, configuracao)

        print(f"Banco condicional criado em: {saida}", flush=True)
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
