from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.granger.comparar_granger_condicional import (
    corrigir_benjamini_hochberg,
)
from experimentos.granger.executar_condicional import construir_etapas
from experimentos.granger.executar_modelo_condicional import (
    construir_frame_rede,
    matrizes_treino,
    selecionar_lag_rede,
    selecionar_redes,
    testar_direcao_condicional,
)


def series_longas_com_cobertura() -> pd.DataFrame:
    datas = pd.bdate_range("2022-01-03", periods=80)
    linhas = []
    for mercado in ("a", "b", "c", "d"):
        for indice, data in enumerate(datas):
            disponivel = mercado != "d" or indice < 20
            linhas.append(
                {
                    "produto": "produto",
                    "mercado": mercado,
                    "data_cotacao": data,
                    "periodo": "treino",
                    "retorno_log": float(indice + 1) if disponivel else np.nan,
                }
            )
    return pd.DataFrame(linhas)


def frame_var_sintetico() -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    n = 320
    origem = gerador.normal(size=n)
    controle = gerador.normal(size=n)
    destino = np.zeros(n)
    ruido = gerador.normal(scale=0.2, size=n)
    for indice in range(1, n):
        destino[indice] = (
            0.35 * destino[indice - 1]
            + 0.65 * origem[indice - 1]
            + 0.45 * controle[indice - 1]
            + ruido[indice]
        )
    return pd.DataFrame(
        {
            "controle": controle,
            "destino": destino,
            "origem": origem,
            "periodo": np.where(np.arange(n) < 240, "treino", "teste"),
            "segmento_rede": 1,
        }
    )


class SelecaoRedesTest(unittest.TestCase):
    def test_escolhe_maior_rede_com_cobertura_suficiente(self) -> None:
        configuracao = selecionar_redes(
            series_longas_com_cobertura(),
            max_lag=2,
            metadados_origem={"banco": "sintetico"},
        )

        rede = configuracao["redes"][0]
        self.assertEqual(rede["mercados"], ["a", "b", "c"])
        self.assertEqual(rede["n_mercados"], 3)

    def test_frame_conjunto_quebra_quando_um_mercado_falta(self) -> None:
        series = series_longas_com_cobertura()

        frame = construir_frame_rede(series, "produto", ("a", "b", "c", "d"))

        self.assertEqual(len(frame), 20)


class GrangerCondicionalTest(unittest.TestCase):
    def test_detecta_origem_apos_controlar_terceiro_mercado(self) -> None:
        frame = frame_var_sintetico()
        mercados = ("controle", "destino", "origem")
        candidatos, selecionado = selecionar_lag_rede(
            frame,
            mercados,
            max_lag=3,
            max_lag_amostra=3,
        )
        lag = int(selecionado["lag"])
        alvo, matriz, _ = matrizes_treino(
            frame,
            mercados,
            lag,
            max_lag_amostra=3,
        )

        resultado, *_ = testar_direcao_condicional(
            alvo,
            matriz,
            mercados,
            origem="origem",
            destino="destino",
            lag=lag,
        )

        self.assertEqual(len(candidatos), 3)
        self.assertLess(resultado["p_valor"], 0.05)


class ExecutorTest(unittest.TestCase):
    def test_congela_configuracao_da_primeira_validacao(self) -> None:
        etapas = construir_etapas(
            Path("resultados/experimento"),
            Path("resultados/experimento/granger-condicional"),
        )

        self.assertIn("--output-config-redes", etapas[0].comando)
        self.assertIn("--config-redes", etapas[1].comando)
        configuracao_v1 = etapas[0].comando[-1]
        configuracao_v2 = etapas[1].comando[-1]
        self.assertEqual(configuracao_v1, configuracao_v2)


class ComparadorTest(unittest.TestCase):
    def test_recalcula_fdr_na_familia_equivalente(self) -> None:
        p_valores = {
            ("a", "x", "y"): 0.01,
            ("a", "y", "x"): 0.02,
            ("b", "x", "y"): 0.20,
        }

        significativas = corrigir_benjamini_hochberg(p_valores)

        self.assertEqual(
            significativas,
            {("a", "x", "y"), ("a", "y", "x")},
        )


if __name__ == "__main__":
    unittest.main()
