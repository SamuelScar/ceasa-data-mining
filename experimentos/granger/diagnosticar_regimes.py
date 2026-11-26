#!/usr/bin/env python3
"""Diagnostica se os dados sustentam um Granger com dois regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid, het_arch
from statsmodels.stats.stattools import jarque_bera

from experimentos.granger.executar_modelo_assimetrico import (
    construir_frame_direcao,
)
from experimentos.granger.executar_modelo_condicional import abrir_somente_leitura
from granger.executar_granger import montar_matrizes_segmentos_treino
from pipeline.progresso_terminal import PainelPipeline


SCHEMA_VERSION = 1
ALPHA = 0.05
QUANTIS_LIMIAR = (0.60, 0.70, 0.75)
OBSERVACOES_POR_COEFICIENTE = 6
DELTA_BIC_MINIMO = 6.0
MINIMO_TESTE = 20

ChaveRelacao = tuple[str, str, str]


def sha256_arquivo(caminho: Path, bloco: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while dados := arquivo.read(bloco):
            digest.update(dados)
    return digest.hexdigest()


def gravar_json(caminho: Path, valor: dict[str, Any]) -> None:
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(valor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def validar_entradas(raiz: Path) -> dict[str, Path]:
    caminhos = {
        "manifesto": raiz / "manifesto.json",
        "validacao_1": raiz / "validacao_1/granger/ffill-3.sqlite",
        "validacao_2": raiz / "validacao_2/granger/ffill-3.sqlite",
    }
    ausentes = [caminho for caminho in caminhos.values() if not caminho.is_file()]
    if ausentes:
        nomes = ", ".join(str(caminho) for caminho in ausentes)
        raise FileNotFoundError(f"artefatos obrigatórios ausentes: {nomes}")
    manifesto = json.loads(caminhos["manifesto"].read_text(encoding="utf-8"))
    if manifesto.get("status") != "concluida":
        raise ValueError("o experimento de preenchimento não está concluído")
    if manifesto.get("inclui_teste_final") is not False:
        raise ValueError("a entrada não declara a proteção do teste final")
    return caminhos


def carregar_parametros(conexao: sqlite3.Connection) -> dict[str, Any]:
    linha = conexao.execute(
        """
        SELECT parametros_json, status
        FROM execucoes_granger
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if linha is None or linha["status"] != "concluida":
        raise ValueError("execução de Granger ausente ou incompleta")
    return json.loads(linha["parametros_json"])


def carregar_direcoes(conexao: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT id, par_id, produto, origem, destino, lag_selecionado
        FROM resultados_granger
        WHERE status = 'testado'
        ORDER BY produto, par_id, origem, destino
        """,
        conexao,
    )


def carregar_series(conexao: sqlite3.Connection) -> dict[int, pd.DataFrame]:
    frame = pd.read_sql_query(
        """
        SELECT par_id, mercado_a, mercado_b, data_cotacao, periodo,
               segmento_par, retorno_a, retorno_b
        FROM series_pares
        WHERE retorno_a IS NOT NULL AND retorno_b IS NOT NULL
        ORDER BY par_id, data_cotacao
        """,
        conexao,
    )
    return {
        int(par_id): grupo.copy()
        for par_id, grupo in frame.groupby("par_id", sort=False)
    }


def calcular_bic(residuos: np.ndarray, parametros: int) -> float:
    n = len(residuos)
    rss = float(residuos @ residuos)
    if n <= parametros or rss <= 0:
        raise ValueError("amostra ou resíduo inválido para calcular BIC")
    return float(n * np.log(rss / n) + parametros * np.log(n))


def construir_partes_threshold_treino(
    frame: pd.DataFrame,
    lag: int,
    max_lag_amostra: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    partes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    treino = frame[frame["periodo"] == "treino"]
    for _, segmento in treino.groupby("segmento_par", sort=True):
        valores = segmento[["origem", "destino"]].dropna().to_numpy(dtype=float)
        if len(valores) <= max_lag_amostra:
            continue
        origem = valores[:, 0]
        destino = valores[:, 1]
        alvo = destino[max_lag_amostra:]
        destino_lags = np.column_stack(
            [
                destino[
                    max_lag_amostra - deslocamento : len(destino) - deslocamento
                ]
                for deslocamento in range(1, lag + 1)
            ]
        )
        origem_lags = np.column_stack(
            [
                origem[
                    max_lag_amostra - deslocamento : len(origem) - deslocamento
                ]
                for deslocamento in range(1, lag + 1)
            ]
        )
        base = np.column_stack(
            [np.ones(len(alvo)), destino_lags, origem_lags]
        )
        variavel_regime = origem[max_lag_amostra - 1 : len(origem) - 1]
        partes.append((alvo, base, variavel_regime))
    return partes


def construir_threshold(base: np.ndarray, regime: np.ndarray) -> np.ndarray:
    indicador = regime.astype(float).reshape(-1, 1)
    return np.column_stack([base, indicador, base[:, 1:] * indicador])


def construir_teste_threshold(
    frame: pd.DataFrame,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alvos: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    regimes: list[np.ndarray] = []
    for _, segmento in frame.groupby("segmento_par", sort=True):
        valores = segmento[["origem", "destino", "periodo"]].dropna()
        if len(valores) <= lag:
            continue
        origem = valores["origem"].to_numpy(dtype=float)
        destino = valores["destino"].to_numpy(dtype=float)
        periodos = valores["periodo"].to_numpy()
        posicoes = np.flatnonzero(periodos == "teste")
        posicoes = posicoes[posicoes >= lag]
        if not len(posicoes):
            continue
        destino_lags = np.column_stack(
            [destino[posicoes - deslocamento] for deslocamento in range(1, lag + 1)]
        )
        origem_lags = np.column_stack(
            [origem[posicoes - deslocamento] for deslocamento in range(1, lag + 1)]
        )
        alvos.append(destino[posicoes])
        bases.append(
            np.column_stack(
                [np.ones(len(posicoes)), destino_lags, origem_lags]
            )
        )
        regimes.append(origem[posicoes - 1])
    if not alvos:
        raise ValueError("nenhum ponto de teste possui histórico suficiente")
    return np.concatenate(alvos), np.vstack(bases), np.concatenate(regimes)


def metricas_previsao(
    alvo: np.ndarray,
    base: np.ndarray,
    threshold: np.ndarray,
    beta_base: np.ndarray,
    beta_threshold: np.ndarray,
) -> dict[str, float | bool | int]:
    erro_base = alvo - base @ beta_base
    erro_threshold = alvo - threshold @ beta_threshold
    rmse_base = float(np.sqrt(np.mean(erro_base**2)))
    rmse_threshold = float(np.sqrt(np.mean(erro_threshold**2)))
    mae_base = float(np.mean(np.abs(erro_base)))
    mae_threshold = float(np.mean(np.abs(erro_threshold)))
    return {
        "n_observacoes_teste": len(alvo),
        "rmse_linear": rmse_base,
        "rmse_threshold": rmse_threshold,
        "ganho_rmse_percentual": (
            100.0 * (rmse_base - rmse_threshold) / rmse_base
            if rmse_base
            else 0.0
        ),
        "mae_linear": mae_base,
        "mae_threshold": mae_threshold,
        "ganho_mae_percentual": (
            100.0 * (mae_base - mae_threshold) / mae_base
            if mae_base
            else 0.0
        ),
        "melhora_rmse_e_mae": bool(
            rmse_threshold < rmse_base and mae_threshold < mae_base
        ),
    }


def avaliar_limiares(
    frame: pd.DataFrame,
    lag: int,
    max_lag_amostra: int,
) -> list[dict[str, Any]]:
    partes = construir_partes_threshold_treino(frame, lag, max_lag_amostra)
    if not partes:
        raise ValueError("nenhum segmento de treino disponível")
    alvo = np.concatenate([parte[0] for parte in partes])
    base = np.vstack([parte[1] for parte in partes])
    variavel_regime = np.concatenate([parte[2] for parte in partes])
    if np.linalg.matrix_rank(base) < base.shape[1]:
        raise ValueError("modelo linear sem posto completo")
    beta_base = np.linalg.lstsq(base, alvo, rcond=None)[0]
    bic_base = calcular_bic(alvo - base @ beta_base, base.shape[1])
    try:
        alvo_teste, base_teste, regime_teste = construir_teste_threshold(frame, lag)
    except ValueError:
        alvo_teste = np.array([])
        base_teste = np.empty((0, base.shape[1]))
        regime_teste = np.array([])

    minimo_regime = OBSERVACOES_POR_COEFICIENTE * base.shape[1]
    resultados: list[dict[str, Any]] = []
    for quantil in QUANTIS_LIMIAR:
        limiar = float(np.quantile(variavel_regime, quantil))
        regime = variavel_regime > limiar
        n_alto = int(regime.sum())
        n_normal = len(regime) - n_alto
        resultado: dict[str, Any] = {
            "quantil": quantil,
            "limiar": limiar,
            "n_regime_normal": n_normal,
            "n_regime_alto": n_alto,
            "minimo_por_regime": minimo_regime,
            "viavel": False,
            "motivo": None,
            "bic_linear": bic_base,
            "bic_threshold": None,
            "delta_bic": None,
            "n_observacoes_teste": len(alvo_teste),
            "melhora_rmse_e_mae": False,
        }
        if limiar <= 0:
            resultado["motivo"] = "limiar não representa uma alta da origem"
            resultados.append(resultado)
            continue
        if min(n_normal, n_alto) < minimo_regime:
            resultado["motivo"] = "amostra insuficiente em pelo menos um regime"
            resultados.append(resultado)
            continue
        matriz_threshold = construir_threshold(base, regime)
        if np.linalg.matrix_rank(matriz_threshold) < matriz_threshold.shape[1]:
            resultado["motivo"] = "modelo threshold sem posto completo"
            resultados.append(resultado)
            continue
        beta_threshold = np.linalg.lstsq(matriz_threshold, alvo, rcond=None)[0]
        bic_threshold = calcular_bic(
            alvo - matriz_threshold @ beta_threshold,
            matriz_threshold.shape[1],
        )
        resultado.update(
            {
                "viavel": True,
                "bic_threshold": bic_threshold,
                "delta_bic": bic_base - bic_threshold,
            }
        )
        if len(alvo_teste) >= MINIMO_TESTE:
            matriz_teste = construir_threshold(base_teste, regime_teste > limiar)
            resultado.update(
                metricas_previsao(
                    alvo_teste,
                    base_teste,
                    matriz_teste,
                    beta_base,
                    beta_threshold,
                )
            )
        else:
            resultado["motivo"] = "teste insuficiente para comparar previsão"
        resultados.append(resultado)
    return resultados


def construir_matrizes_var(
    frame: pd.DataFrame,
    lag: int,
    max_lag_amostra: int,
) -> tuple[np.ndarray, np.ndarray]:
    alvos: list[np.ndarray] = []
    regressores: list[np.ndarray] = []
    treino = frame[frame["periodo"] == "treino"]
    for _, segmento in treino.groupby("segmento_par", sort=True):
        valores = segmento[["origem", "destino"]].dropna().to_numpy(dtype=float)
        if len(valores) <= max_lag_amostra:
            continue
        origem = valores[:, 0]
        destino = valores[:, 1]
        alvos.append(valores[max_lag_amostra:])
        origem_lags = np.column_stack(
            [
                origem[
                    max_lag_amostra - deslocamento : len(origem) - deslocamento
                ]
                for deslocamento in range(1, lag + 1)
            ]
        )
        destino_lags = np.column_stack(
            [
                destino[
                    max_lag_amostra - deslocamento : len(destino) - deslocamento
                ]
                for deslocamento in range(1, lag + 1)
            ]
        )
        regressores.append(
            np.column_stack(
                [np.ones(len(origem_lags)), origem_lags, destino_lags]
            )
        )
    if not alvos:
        raise ValueError("nenhum segmento disponível para o VAR")
    return np.vstack(alvos), np.vstack(regressores)


def calcular_raio_companheiro(matrizes: list[np.ndarray]) -> float:
    lag = len(matrizes)
    dimensao = matrizes[0].shape[0]
    companheira = np.zeros((dimensao * lag, dimensao * lag))
    companheira[:dimensao, :] = np.hstack(matrizes)
    if lag > 1:
        companheira[dimensao:, :-dimensao] = np.eye(dimensao * (lag - 1))
    return float(np.max(np.abs(np.linalg.eigvals(companheira))))


def diagnosticar_estabilidade_var(
    frame: pd.DataFrame,
    lag: int,
    max_lag_amostra: int,
) -> dict[str, float | bool]:
    alvo, regressores = construir_matrizes_var(frame, lag, max_lag_amostra)
    if np.linalg.matrix_rank(regressores) < regressores.shape[1]:
        raise ValueError("VAR sem posto completo")
    beta = np.linalg.lstsq(regressores, alvo, rcond=None)[0]
    matrizes = []
    for indice in range(lag):
        matrizes.append(
            np.array(
                [
                    [beta[1 + indice, 0], beta[1 + lag + indice, 0]],
                    [beta[1 + indice, 1], beta[1 + lag + indice, 1]],
                ]
            )
        )
    raio = calcular_raio_companheiro(matrizes)
    return {"raio_companheiro": raio, "var_estavel": bool(raio < 1.0)}


def diagnosticar_residuos_avancados(
    partes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    alvo, _, irrestrita = max(partes, key=lambda parte: len(parte[0]))
    if len(alvo) <= irrestrita.shape[1] + 10:
        raise ValueError("maior segmento insuficiente para os diagnósticos")
    modelo = sm.OLS(alvo, irrestrita).fit()
    residuos = np.asarray(modelo.resid, dtype=float)
    _, p_jb, _, curtose = jarque_bera(residuos)
    defasagens_arch = min(5, max(1, len(residuos) // 20))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="acorr_lm currently returns",
            category=FutureWarning,
        )
        _, p_arch, _, _ = het_arch(residuos, nlags=defasagens_arch)
    estatistica_cusum, p_cusum, _ = breaks_cusumolsresid(
        residuos,
        ddof=irrestrita.shape[1],
    )
    influencia = modelo.get_influence()
    cooks = np.asarray(influencia.cooks_distance[0], dtype=float)
    studentizados = np.asarray(
        influencia.resid_studentized_external,
        dtype=float,
    )
    limite_cook = 4.0 / len(residuos)
    return {
        "n_maior_segmento": len(residuos),
        "p_valor_jarque_bera": float(p_jb),
        "curtose": float(curtose),
        "caudas_pesadas": bool(p_jb < ALPHA and curtose > 3.0),
        "p_valor_arch": float(p_arch),
        "heterocedasticidade": bool(p_arch < ALPHA),
        "estatistica_cusum": float(estatistica_cusum),
        "p_valor_cusum": float(p_cusum),
        "quebra_cusum": bool(p_cusum < ALPHA),
        "limite_cook": limite_cook,
        "n_cook_acima_limite": int(np.sum(cooks > limite_cook)),
        "fracao_cook_acima_limite": float(np.mean(cooks > limite_cook)),
        "n_residuos_studentizados_acima_3": int(
            np.sum(np.abs(studentizados[np.isfinite(studentizados)]) > 3.0)
        ),
    }


def diagnosticar_direcao(
    direcao: Any,
    serie_par: pd.DataFrame,
    max_lag_amostra: int,
) -> dict[str, Any]:
    frame = construir_frame_direcao(
        serie_par,
        str(direcao.origem),
        str(direcao.destino),
    )
    lag = int(direcao.lag_selecionado)
    partes = montar_matrizes_segmentos_treino(
        frame,
        "origem",
        "destino",
        lag,
        max_lag_amostra,
    )
    if not partes:
        raise ValueError("nenhum segmento de treino disponível")
    limiares = avaliar_limiares(frame, lag, max_lag_amostra)
    return {
        "produto": str(direcao.produto),
        "origem": str(direcao.origem),
        "destino": str(direcao.destino),
        "lag": lag,
        "status": "avaliado",
        "residuos": diagnosticar_residuos_avancados(partes),
        "estabilidade_var": diagnosticar_estabilidade_var(
            frame,
            lag,
            max_lag_amostra,
        ),
        "limiares": limiares,
    }


def melhor_limiar(direcao: dict[str, Any]) -> dict[str, Any] | None:
    viaveis = [item for item in direcao.get("limiares", []) if item["viavel"]]
    return max(viaveis, key=lambda item: item["delta_bic"]) if viaveis else None


def resumir_direcoes(direcoes: list[dict[str, Any]]) -> dict[str, int]:
    avaliadas = [item for item in direcoes if item["status"] == "avaliado"]
    melhores = [melhor_limiar(item) for item in avaliadas]
    melhores = [item for item in melhores if item is not None]
    return {
        "direcoes_total": len(direcoes),
        "direcoes_avaliadas": len(avaliadas),
        "erros": len(direcoes) - len(avaliadas),
        "var_instavel": sum(
            not item["estabilidade_var"]["var_estavel"] for item in avaliadas
        ),
        "heterocedasticidade": sum(
            item["residuos"]["heterocedasticidade"] for item in avaliadas
        ),
        "caudas_pesadas": sum(
            item["residuos"]["caudas_pesadas"] for item in avaliadas
        ),
        "quebra_cusum": sum(
            item["residuos"]["quebra_cusum"] for item in avaliadas
        ),
        "threshold_viavel": len(melhores),
        "threshold_delta_bic_maior_igual_6": sum(
            item["delta_bic"] >= DELTA_BIC_MINIMO for item in melhores
        ),
        "threshold_melhora_rmse_e_mae": sum(
            item["melhora_rmse_e_mae"] for item in melhores
        ),
    }


def diagnosticar_banco(caminho: Path) -> dict[str, Any]:
    with abrir_somente_leitura(caminho) as conexao:
        if conexao.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError(f"falha de integridade: {caminho}")
        parametros = carregar_parametros(conexao)
        direcoes = carregar_direcoes(conexao)
        series = carregar_series(conexao)
    max_lag_amostra = int(parametros["max_lag_amostra"])
    resultados: list[dict[str, Any]] = []
    for direcao in direcoes.itertuples(index=False):
        base = {
            "produto": str(direcao.produto),
            "origem": str(direcao.origem),
            "destino": str(direcao.destino),
            "lag": int(direcao.lag_selecionado),
        }
        try:
            resultados.append(
                diagnosticar_direcao(
                    direcao,
                    series[int(direcao.par_id)],
                    max_lag_amostra,
                )
            )
        except (KeyError, ValueError, np.linalg.LinAlgError) as erro:
            resultados.append(
                {**base, "status": "erro", "observacao": str(erro)}
            )
    return {
        "banco": str(caminho),
        "parametros_origem": parametros,
        "resumo": resumir_direcoes(resultados),
        "direcoes": resultados,
    }


def indexar_direcoes(execucao: dict[str, Any]) -> dict[ChaveRelacao, dict[str, Any]]:
    return {
        (item["produto"], item["origem"], item["destino"]): item
        for item in execucao["direcoes"]
        if item["status"] == "avaliado"
    }


def serializar_chaves(chaves: list[ChaveRelacao]) -> list[dict[str, str]]:
    return [
        {"produto": produto, "origem": origem, "destino": destino}
        for produto, origem, destino in sorted(chaves)
    ]


def comparar_validacoes(
    validacao_1: dict[str, Any],
    validacao_2: dict[str, Any],
) -> dict[str, Any]:
    v1 = indexar_direcoes(validacao_1)
    v2 = indexar_direcoes(validacao_2)
    comuns = set(v1) & set(v2)
    evidencias: list[ChaveRelacao] = []
    for chave in comuns:
        por_quantil_v1 = {item["quantil"]: item for item in v1[chave]["limiares"]}
        por_quantil_v2 = {item["quantil"]: item for item in v2[chave]["limiares"]}
        for quantil in QUANTIS_LIMIAR:
            primeiro = por_quantil_v1[quantil]
            segundo = por_quantil_v2[quantil]
            if all(
                item["viavel"]
                and item["delta_bic"] >= DELTA_BIC_MINIMO
                and item["melhora_rmse_e_mae"]
                for item in (primeiro, segundo)
            ):
                evidencias.append(chave)
                break
    quebras_comuns = [
        chave
        for chave in comuns
        if v1[chave]["residuos"]["quebra_cusum"]
        and v2[chave]["residuos"]["quebra_cusum"]
    ]
    return {
        "direcoes_comuns": len(comuns),
        "quebras_cusum_nas_duas": len(quebras_comuns),
        "relacoes_com_quebra_nas_duas": serializar_chaves(quebras_comuns),
        "evidencias_threshold_nas_duas": len(evidencias),
        "relacoes_com_evidencia_threshold": serializar_chaves(evidencias),
        "criterio_evidencia_threshold": (
            "mesmo quantil viável, delta BIC >= 6 e melhora simultânea de "
            "RMSE e MAE fora da amostra nas duas validações"
        ),
        "recomendacao": (
            "prosseguir_com_threshold_var_parcimonioso"
            if evidencias
            else "nao_ha_evidencia_suficiente_para_threshold_var"
        ),
    }


def exibir_resumo(relatorio: dict[str, Any]) -> None:
    print("=== DIAGNÓSTICO DE MUDANÇA DE REGIME ===")
    for contexto, execucao in relatorio["validacoes"].items():
        print(f"\n{contexto}")
        for chave, valor in execucao["resumo"].items():
            print(f"  {chave}: {valor}")
    print("\n=== COMPARAÇÃO ENTRE VALIDAÇÕES ===")
    comparacao = relatorio["comparacao"]
    for chave, valor in comparacao.items():
        if not isinstance(valor, list):
            print(f"{chave}: {valor}")


def executar(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experimento_ffill", type=Path)
    args = parser.parse_args(argv)
    raiz = args.experimento_ffill
    saida = raiz / "diagnostico-regimes"
    try:
        caminhos = validar_entradas(raiz)
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}", flush=True)
        return 2
    if saida.exists():
        print(f"ERRO: a saída já existe: {saida}", flush=True)
        return 2

    resultados: dict[str, dict[str, Any]] = {}
    painel = PainelPipeline("Diagnóstico de mudança de regime", 3)
    with painel:
        for indice, contexto in enumerate(("validacao_1", "validacao_2"), start=1):
            painel.iniciar_etapa(
                indice,
                f"Diagnósticos — {contexto}",
                "avalia estabilidade, resíduos e limiares sem consultar o teste final",
            )
            resultados[contexto] = diagnosticar_banco(caminhos[contexto])
            painel.concluir_etapa(indice, contexto)
        painel.iniciar_etapa(
            3,
            "Comparação entre validações",
            "exige evidência de regime nos dois períodos",
        )
        comparacao = comparar_validacoes(
            resultados["validacao_1"],
            resultados["validacao_2"],
        )
        painel.concluir_etapa(3, "Comparação entre validações")
        painel.iniciar_finalizacao()
        saida.mkdir(parents=True)
        relatorio = {
            "schema_version": SCHEMA_VERSION,
            "gerado_em": datetime.now(UTC).isoformat(),
            "inclui_teste_final": False,
            "objetivo": (
                "diagnosticar mudança de regime antes de implementar inferência "
                "causal threshold"
            ),
            "inferencia_causal_threshold_realizada": False,
            "configuracao": {
                "alpha_diagnosticos": ALPHA,
                "variavel_regime": "retorno da origem defasado em um dia útil",
                "quantis_limiar": list(QUANTIS_LIMIAR),
                "observacoes_por_coeficiente_por_regime": (
                    OBSERVACOES_POR_COEFICIENTE
                ),
                "delta_bic_minimo": DELTA_BIC_MINIMO,
                "minimo_observacoes_teste": MINIMO_TESTE,
            },
            "validacoes": resultados,
            "comparacao": comparacao,
        }
        gravar_json(saida / "comparacao.json", relatorio)
        manifesto = {
            "schema_version": SCHEMA_VERSION,
            "tipo": "diagnostico_mudanca_regime",
            "status": "concluida",
            "concluida_em": datetime.now(UTC).isoformat(),
            "inclui_teste_final": False,
            "experimento_ffill": str(raiz),
            "entradas": {
                contexto: {
                    "caminho": str(caminhos[contexto]),
                    "tamanho_bytes": caminhos[contexto].stat().st_size,
                    "sha256": sha256_arquivo(caminhos[contexto]),
                }
                for contexto in ("validacao_1", "validacao_2")
            },
            "artefatos": {
                "comparacao": {
                    "caminho": "comparacao.json",
                    "tamanho_bytes": (saida / "comparacao.json").stat().st_size,
                    "sha256": sha256_arquivo(saida / "comparacao.json"),
                }
            },
        }
        gravar_json(saida / "manifesto.json", manifesto)
        painel.finalizar()

    exibir_resumo(relatorio)
    print(f"\nResultados preservados em: {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
