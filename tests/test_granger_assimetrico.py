from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.granger.executar_assimetria import construir_etapas
from experimentos.granger.executar_modelo_assimetrico import (
    ajustar_modelos,
    decompor_retorno,
    matrizes_treino,
    resposta_dinamica_assimetrica,
)


def frame_assimetrico_sintetico() -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    n = 420
    origem = gerador.normal(size=n)
    destino = np.zeros(n)
    ruido = gerador.normal(scale=0.15, size=n)
    for indice in range(1, n):
        destino[indice] = (
            0.25 * destino[indice - 1]
            + 0.8 * max(origem[indice - 1], 0.0)
            + ruido[indice]
        )
    return pd.DataFrame(
        {
            "origem": origem,
            "destino": destino,
            "periodo": np.where(np.arange(n) < 320, "treino", "teste"),
            "segmento_par": 1,
        }
    )


class DecomposicaoTest(unittest.TestCase):
    def test_separa_altas_e_quedas_sem_recompor_valor_diferente(self) -> None:
        retorno = np.array([-0.4, 0.0, 0.7])

        negativo, positivo = decompor_retorno(retorno)

        np.testing.assert_array_equal(negativo, [-0.4, 0.0, 0.0])
        np.testing.assert_array_equal(positivo, [0.0, 0.0, 0.7])
        np.testing.assert_array_equal(negativo + positivo, retorno)


class ModeloAssimetricoTest(unittest.TestCase):
    def test_detecta_informacao_adicional_das_altas_da_origem(self) -> None:
        alvo, restrita, irrestrita, _ = matrizes_treino(
            frame_assimetrico_sintetico(),
            lag=1,
            max_lag_amostra=1,
        )

        resultado, _, _ = ajustar_modelos(alvo, restrita, irrestrita, lag=1)

        self.assertEqual(restrita.shape[1], 3)
        self.assertEqual(irrestrita.shape[1], 4)
        self.assertLess(resultado["p_valor"], 0.05)

    def test_resposta_dinamica_preserva_sinal_positivo(self) -> None:
        beta = np.array([0.0, 0.5, -0.2, 0.4])

        resposta = resposta_dinamica_assimetrica(beta, lag=1, horizonte=3)

        self.assertAlmostEqual(resposta, 0.7)


class ExecutorTest(unittest.TestCase):
    def test_usa_ffill_tres_sem_incluir_teste_final(self) -> None:
        etapas = construir_etapas(
            Path("resultados/experimento"),
            Path("resultados/experimento/granger-assimetrico"),
        )

        comandos = [" ".join(etapa.comando) for etapa in etapas]
        self.assertEqual(len(etapas), 3)
        self.assertIn("validacao_1/granger/ffill-3.sqlite", comandos[0])
        self.assertIn("validacao_2/granger/ffill-3.sqlite", comandos[1])
        self.assertNotIn("teste_final", " ".join(comandos))


if __name__ == "__main__":
    unittest.main()
