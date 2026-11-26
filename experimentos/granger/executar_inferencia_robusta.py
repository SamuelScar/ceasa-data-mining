#!/usr/bin/env python3
"""Recalcula a inferência do Granger bivariado com covariâncias HC3 e HAC."""

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
from scipy.stats import chi2

from experimentos.granger.executar_modelo_assimetrico import (
    carregar_series_pares,
    construir_frame_direcao,
)
from experimentos.granger.executar_modelo_condicional import (
    abrir_somente_leitura,
    carregar_metadados_origem,
    sha256_arquivo,
)
from granger.executar_granger import (
    corrigir_benjamini_hochberg,
    montar_matrizes_segmentos_treino,
)


SCHEMA_VERSION = 1
HORIZONTE_RESPOSTA = 10


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite do Granger bivariado")
    parser.add_argument("--output-db", type=Path, required=True)
    return parser


def carregar_direcoes(conexao: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT id AS resultado_granger_id, par_id, produto, origem, destino,
               lag_selecionado
        FROM resultados_granger
        WHERE status = 'testado'
        ORDER BY produto, par_id, origem, destino
        """,
        conexao,
    )


def covariancia_hc3(
    regressores: np.ndarray,
    residuos: np.ndarray,
) -> np.ndarray:
    xtx_inversa = np.linalg.inv(regressores.T @ regressores)
    alavancagem = np.sum((regressores @ xtx_inversa) * regressores, axis=1)
    denominador = 1.0 - alavancagem
    if np.any(denominador <= np.finfo(float).eps):
        raise ValueError("alavancagem unitária impede a covariância HC3")
    escores = regressores * (residuos / denominador)[:, None]
    return xtx_inversa @ (escores.T @ escores) @ xtx_inversa


def covariancia_hac_segmentada(
    regressores_partes: list[np.ndarray],
    residuos_partes: list[np.ndarray],
    max_lags: int,
) -> np.ndarray:
    regressores = np.vstack(regressores_partes)
    n, k = regressores.shape
    xtx_inversa = np.linalg.inv(regressores.T @ regressores)
    carne = np.zeros((k, k))
    for x_parte, residuo_parte in zip(
        regressores_partes,
        residuos_partes,
        strict=True,
    ):
        escores = x_parte * residuo_parte[:, None]
        carne += escores.T @ escores
        limite = min(max_lags, len(escores) - 1)
        for defasagem in range(1, limite + 1):
            peso = 1.0 - defasagem / (max_lags + 1.0)
            cruzada = escores[defasagem:].T @ escores[:-defasagem]
            carne += peso * (cruzada + cruzada.T)
    correcao = n / (n - k)
    return correcao * xtx_inversa @ carne @ xtx_inversa


def teste_wald_origem(
    coeficientes: np.ndarray,
    covariancia: np.ndarray,
    lag: int,
) -> tuple[float, float]:
    restricao = np.zeros((lag, len(coeficientes)))
    inicio_origem = 1 + lag
    for indice in range(lag):
        restricao[indice, inicio_origem + indice] = 1.0
    efeito = restricao @ coeficientes
    covariancia_efeito = restricao @ covariancia @ restricao.T
    if np.linalg.matrix_rank(covariancia_efeito) < lag:
        raise ValueError("covariância singular no teste Wald")
    estatistica = float(
        efeito.T @ np.linalg.solve(covariancia_efeito, efeito)
    )
    estatistica = max(0.0, estatistica)
    return estatistica, float(chi2.sf(estatistica, lag))


def resposta_dinamica(
    coeficientes: np.ndarray,
    lag: int,
    horizonte: int = HORIZONTE_RESPOSTA,
) -> float:
    destino = coeficientes[1 : 1 + lag]
    origem = coeficientes[1 + lag : 1 + 2 * lag]
    respostas = [0.0]
    for passo in range(1, horizonte + 1):
        resposta = float(origem[passo - 1]) if passo <= lag else 0.0
        for defasagem, coeficiente in enumerate(destino, start=1):
            if defasagem <= passo:
                resposta += float(coeficiente) * respostas[passo - defasagem]
        respostas.append(resposta)
    return float(sum(respostas[1:]))


def ajustar_inferencia_robusta(
    partes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    lag: int,
    coeficientes_esperados: np.ndarray | None = None,
) -> dict[str, float | int]:
    if not partes:
        raise ValueError("nenhum segmento de treino disponível")
    alvos = [parte[0] for parte in partes]
    regressores = [parte[2] for parte in partes]
    alvo = np.concatenate(alvos)
    matriz = np.vstack(regressores)
    if len(alvo) <= matriz.shape[1]:
        raise ValueError("amostra insuficiente para o modelo irrestrito")
    if np.linalg.matrix_rank(matriz) < matriz.shape[1]:
        raise ValueError("modelo irrestrito sem posto completo")
    coeficientes = np.linalg.lstsq(matriz, alvo, rcond=None)[0]
    if coeficientes_esperados is not None and not np.allclose(
        coeficientes,
        coeficientes_esperados,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("coeficientes OLS divergiram do Granger de origem")
    residuos_partes = [
        alvo_parte - matriz_parte @ coeficientes
        for alvo_parte, matriz_parte in zip(alvos, regressores, strict=True)
    ]
    residuos = np.concatenate(residuos_partes)
    hc3 = covariancia_hc3(matriz, residuos)
    hac = covariancia_hac_segmentada(regressores, residuos_partes, lag)
    estatistica_hc3, p_hc3 = teste_wald_origem(coeficientes, hc3, lag)
    estatistica_hac, p_hac = teste_wald_origem(coeficientes, hac, lag)
    return {
        "n_observacoes": len(alvo),
        "estatistica_hc3": estatistica_hc3,
        "p_valor_hc3": p_hc3,
        "estatistica_hac": estatistica_hac,
        "p_valor_hac": p_hac,
        "resposta_acumulada": resposta_dinamica(coeficientes, lag),
    }


def carregar_diagnosticos_origem(
    conexao: sqlite3.Connection,
) -> dict[int, dict[str, Any]]:
    linhas = conexao.execute(
        """
        SELECT resultado.id,
               resultado.n_observacoes AS n_observacoes_classico,
               resultado.p_valor AS p_valor_classico,
               resultado.q_valor_fdr AS q_valor_classico,
               resultado.significativo_fdr AS significativo_classico,
               diagnostico.status AS status_residuos,
               diagnostico.sem_autocorrelacao_irrestrito,
               avaliacao.status AS status_previsao,
               avaliacao.melhora_rmse,
               avaliacao.melhora_mae,
               avaliacao.altas_avaliadas,
               avaliacao.melhora_rmse_altas,
               avaliacao.melhora_mae_altas
        FROM resultados_granger AS resultado
        LEFT JOIN diagnosticos_residuos AS diagnostico
          ON diagnostico.resultado_id = resultado.id
        LEFT JOIN avaliacoes_preditivas AS avaliacao
          ON avaliacao.resultado_id = resultado.id
        WHERE resultado.status = 'testado'
        """
    ).fetchall()
    return {int(linha["id"]): dict(linha) for linha in linhas}


def carregar_coeficientes_origem(
    conexao: sqlite3.Connection,
) -> dict[int, np.ndarray]:
    coeficientes: dict[int, list[float]] = {}
    linhas = conexao.execute(
        """
        SELECT resultado_id, coeficiente
        FROM coeficientes_granger
        ORDER BY resultado_id,
                 CASE tipo_termo
                     WHEN 'intercepto' THEN 0
                     WHEN 'destino' THEN 1
                     WHEN 'origem' THEN 2
                 END,
                 defasagem
        """
    )
    for linha in linhas:
        coeficientes.setdefault(int(linha["resultado_id"]), []).append(
            float(linha["coeficiente"])
        )
    return {
        resultado_id: np.asarray(valores, dtype=float)
        for resultado_id, valores in coeficientes.items()
    }


def executar_direcoes(
    direcoes: pd.DataFrame,
    series_pares: pd.DataFrame,
    diagnosticos: dict[int, dict[str, Any]],
    coeficientes_origem: dict[int, np.ndarray],
    max_lag_amostra: int,
    alpha: float,
) -> pd.DataFrame:
    identificadores = set(direcoes["resultado_granger_id"].astype(int))
    diagnosticos_ausentes = identificadores - set(diagnosticos)
    if diagnosticos_ausentes:
        raise ValueError(
            "diagnósticos ausentes para "
            f"{len(diagnosticos_ausentes)} direções do Granger"
        )
    coeficientes_ausentes = identificadores - set(coeficientes_origem)
    if coeficientes_ausentes:
        raise ValueError(
            "coeficientes ausentes para "
            f"{len(coeficientes_ausentes)} direções do Granger"
        )
    pares = {
        int(par_id): grupo.copy()
        for par_id, grupo in series_pares.groupby("par_id", sort=False)
    }
    resultados: list[dict[str, Any]] = []
    for identificador, direcao in enumerate(
        direcoes.itertuples(index=False),
        start=1,
    ):
        origem = diagnosticos[int(direcao.resultado_granger_id)]
        base: dict[str, Any] = {
            "id": identificador,
            "resultado_granger_id": int(direcao.resultado_granger_id),
            "par_id": int(direcao.par_id),
            "produto": str(direcao.produto),
            "origem": str(direcao.origem),
            "destino": str(direcao.destino),
            "lag": None,
            "status": "testado",
            "n_observacoes": None,
            "p_valor_classico": origem["p_valor_classico"],
            "q_valor_classico": origem["q_valor_classico"],
            "significativo_classico": origem["significativo_classico"],
            "estatistica_hc3": None,
            "p_valor_hc3": None,
            "q_valor_hc3": None,
            "significativo_hc3": 0,
            "estatistica_hac": None,
            "p_valor_hac": None,
            "q_valor_hac": None,
            "significativo_hac": 0,
            "resposta_acumulada": None,
            "efeito_positivo": 0,
            "residuos_adequados": int(
                origem["status_residuos"] == "avaliado"
                and bool(origem["sem_autocorrelacao_irrestrito"])
            ),
            "previsao_avaliada": int(origem["status_previsao"] == "avaliado"),
            "melhora_rmse": int(origem["melhora_rmse"] or 0),
            "melhora_mae": int(origem["melhora_mae"] or 0),
            "altas_avaliadas": int(origem["altas_avaliadas"] or 0),
            "melhora_rmse_altas": int(origem["melhora_rmse_altas"] or 0),
            "melhora_mae_altas": int(origem["melhora_mae_altas"] or 0),
            "relacao_hc3_robusta": 0,
            "relacao_hac_robusta": 0,
            "relacao_hc3_alta_robusta": 0,
            "relacao_hac_alta_robusta": 0,
            "observacao": None,
        }
        try:
            frame = construir_frame_direcao(
                pares[int(direcao.par_id)],
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
            inferencia = ajustar_inferencia_robusta(
                partes,
                lag,
                coeficientes_origem[int(direcao.resultado_granger_id)],
            )
            if int(inferencia["n_observacoes"]) != int(
                origem["n_observacoes_classico"]
            ):
                raise RuntimeError(
                    "amostra divergiu do Granger de origem: "
                    f"{inferencia['n_observacoes']} != "
                    f"{origem['n_observacoes_classico']}"
                )
            base.update({"lag": lag, **inferencia})
            base["efeito_positivo"] = int(
                float(inferencia["resposta_acumulada"]) > 0
            )
        except (KeyError, ValueError, np.linalg.LinAlgError) as erro:
            base["status"] = "erro_modelo"
            base["observacao"] = str(erro)
        resultados.append(base)

    testados = [item for item in resultados if item["status"] == "testado"]
    for metodo in ("hc3", "hac"):
        q_valores = corrigir_benjamini_hochberg(
            [float(item[f"p_valor_{metodo}"]) for item in testados]
        )
        for item, q_valor in zip(testados, q_valores, strict=True):
            item[f"q_valor_{metodo}"] = q_valor
            item[f"significativo_{metodo}"] = int(q_valor < alpha)

    for item in testados:
        base_robusta = (
            bool(item["residuos_adequados"])
            and bool(item["previsao_avaliada"])
        )
        melhora_geral = bool(item["melhora_rmse"] and item["melhora_mae"])
        melhora_altas = bool(
            item["altas_avaliadas"]
            and item["melhora_rmse_altas"]
            and item["melhora_mae_altas"]
        )
        for metodo in ("hc3", "hac"):
            significativo = bool(item[f"significativo_{metodo}"])
            item[f"relacao_{metodo}_robusta"] = int(
                significativo and base_robusta and melhora_geral
            )
            item[f"relacao_{metodo}_alta_robusta"] = int(
                significativo
                and base_robusta
                and bool(item["efeito_positivo"])
                and melhora_altas
            )
    return pd.DataFrame(resultados)


def criar_esquema(conexao: sqlite3.Connection) -> None:
    conexao.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_inferencia_robusta (
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

        CREATE TABLE resultados_inferencia_robusta (
            id INTEGER PRIMARY KEY,
            resultado_granger_id INTEGER NOT NULL,
            par_id INTEGER NOT NULL,
            produto TEXT NOT NULL,
            origem TEXT NOT NULL,
            destino TEXT NOT NULL,
            lag INTEGER,
            status TEXT NOT NULL,
            n_observacoes INTEGER,
            p_valor_classico REAL,
            q_valor_classico REAL,
            significativo_classico INTEGER NOT NULL,
            estatistica_hc3 REAL,
            p_valor_hc3 REAL,
            q_valor_hc3 REAL,
            significativo_hc3 INTEGER NOT NULL,
            estatistica_hac REAL,
            p_valor_hac REAL,
            q_valor_hac REAL,
            significativo_hac INTEGER NOT NULL,
            resposta_acumulada REAL,
            efeito_positivo INTEGER NOT NULL,
            residuos_adequados INTEGER NOT NULL,
            previsao_avaliada INTEGER NOT NULL,
            melhora_rmse INTEGER NOT NULL,
            melhora_mae INTEGER NOT NULL,
            altas_avaliadas INTEGER NOT NULL,
            melhora_rmse_altas INTEGER NOT NULL,
            melhora_mae_altas INTEGER NOT NULL,
            relacao_hc3_robusta INTEGER NOT NULL,
            relacao_hac_robusta INTEGER NOT NULL,
            relacao_hc3_alta_robusta INTEGER NOT NULL,
            relacao_hac_alta_robusta INTEGER NOT NULL,
            observacao TEXT,
            UNIQUE (produto, origem, destino)
        );

        CREATE TABLE resumo_inferencia_robusta (
            chave TEXT PRIMARY KEY,
            valor TEXT NOT NULL
        );
        """
    )


def construir_resumo(resultados: pd.DataFrame) -> dict[str, int]:
    testados = resultados[resultados["status"] == "testado"]
    return {
        "direcoes_registradas": len(resultados),
        "direcoes_testadas": len(testados),
        "erros_modelo": int((resultados["status"] == "erro_modelo").sum()),
        "significativas_classico": int(testados["significativo_classico"].sum()),
        "significativas_hc3": int(testados["significativo_hc3"].sum()),
        "significativas_hac": int(testados["significativo_hac"].sum()),
        "robustas_hc3": int(testados["relacao_hc3_robusta"].sum()),
        "robustas_hac": int(testados["relacao_hac_robusta"].sum()),
        "altas_robustas_hc3": int(testados["relacao_hc3_alta_robusta"].sum()),
        "altas_robustas_hac": int(testados["relacao_hac_alta_robusta"].sum()),
    }


def validar_saida(conexao: sqlite3.Connection) -> None:
    if conexao.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("falha de integridade na inferência robusta")
    invalidos = conexao.execute(
        """
        SELECT COUNT(*)
        FROM resultados_inferencia_robusta
        WHERE status = 'testado'
          AND (
            p_valor_hc3 NOT BETWEEN 0 AND 1
            OR q_valor_hc3 NOT BETWEEN 0 AND 1
            OR p_valor_hac NOT BETWEEN 0 AND 1
            OR q_valor_hac NOT BETWEEN 0 AND 1
          )
        """
    ).fetchone()[0]
    if invalidos:
        raise RuntimeError("p-valores robustos inválidos")


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
        diagnosticos = carregar_diagnosticos_origem(origem)
        coeficientes_origem = carregar_coeficientes_origem(origem)
        alpha = float(parametros_origem["alpha"])
        max_lag_amostra = int(parametros_origem["max_lag_amostra"])
        print("Calculando testes Wald robustos por direção...", flush=True)
        resultados = executar_direcoes(
            direcoes,
            series_pares,
            diagnosticos,
            coeficientes_origem,
            max_lag_amostra,
            alpha,
        )
        resumo = construir_resumo(resultados)
        parametros = {
            "alpha": alpha,
            "ffill_limite": int(parametros_origem["ffill_limite"]),
            "max_lag": int(parametros_origem["max_lag"]),
            "max_lag_amostra": max_lag_amostra,
            "lag": "reutilizado do Granger bivariado",
            "coeficientes": "OLS idêntico ao Granger bivariado",
            "invariantes": (
                "mesmo número de observações e mesmos coeficientes OLS "
                "do banco de origem"
            ),
            "teste_primario": (
                "Wald qui-quadrado com Newey-West segmentado, janela igual ao lag "
                "e correção n/(n-k)"
            ),
            "teste_sensibilidade": "Wald qui-quadrado com covariância HC3",
            "correcao_multiplos_testes": (
                "Benjamini-Hochberg global separada para HAC e HC3"
            ),
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
            INSERT INTO execucoes_inferencia_robusta (
                id, schema_version, iniciada_em, banco_origem,
                banco_origem_sha256, execucao_granger_id,
                parametros_json, status
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
        resultados.to_sql(
            "resultados_inferencia_robusta",
            destino,
            if_exists="append",
            index=False,
        )
        destino.executemany(
            "INSERT INTO resumo_inferencia_robusta (chave, valor) VALUES (?, ?)",
            [(chave, str(valor)) for chave, valor in sorted(resumo.items())],
        )
        destino.executescript(
            """
            CREATE INDEX idx_robusta_hac
                ON resultados_inferencia_robusta(significativo_hac, q_valor_hac);
            CREATE INDEX idx_robusta_hc3
                ON resultados_inferencia_robusta(significativo_hc3, q_valor_hc3);
            CREATE VIEW relacoes_hac_significativas AS
                SELECT * FROM resultados_inferencia_robusta
                WHERE status = 'testado' AND significativo_hac = 1;
            """
        )
        destino.execute(
            """
            UPDATE execucoes_inferencia_robusta
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

        print(f"Banco de inferência robusta criado em: {saida}", flush=True)
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
