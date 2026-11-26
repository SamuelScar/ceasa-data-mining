from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experimentos.granger.diagnosticar_regimes import (
    QUANTIS_LIMIAR,
    avaliar_limiares,
    calcular_raio_companheiro,
    comparar_validacoes,
)


def frame_com_regime() -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    n = 1_000
    origem = gerador.normal(size=n)
    destino = np.zeros(n)
    ruido = gerador.normal(scale=0.2, size=n)
    for indice in range(1, n):
        efeito = 1.2 if origem[indice - 1] > 0.25 else 0.05
        destino[indice] = (
            0.25 * destino[indice - 1]
            + efeito * origem[indice - 1]
            + ruido[indice]
        )
    return pd.DataFrame(
        {
            "origem": origem,
            "destino": destino,
            "periodo": np.where(np.arange(n) < 800, "treino", "teste"),
            "segmento_par": 1,
        }
    )


def execucao_sintetica(delta_bic: float, melhora: bool) -> dict[str, object]:
    limiares = [
        {
            "quantil": quantil,
            "viavel": True,
            "delta_bic": delta_bic,
            "melhora_rmse_e_mae": melhora,
        }
        for quantil in QUANTIS_LIMIAR
    ]
    return {
        "direcoes": [
            {
                "produto": "batata",
                "origem": "curitiba",
                "destino": "cascavel",
                "status": "avaliado",
                "residuos": {"quebra_cusum": True},
                "limiares": limiares,
            }
        ]
    }


class ThresholdDiagnosticoTest(unittest.TestCase):
    def test_identifica_regime_com_ganho_de_bic_e_previsao(self) -> None:
        resultados = avaliar_limiares(
            frame_com_regime(),
            lag=1,
            max_lag_amostra=1,
        )

        evidencias = [
            resultado
            for resultado in resultados
            if resultado["viavel"]
            and resultado["delta_bic"] >= 6
            and resultado["melhora_rmse_e_mae"]
        ]

        self.assertTrue(evidencias)

    def test_raio_companheiro_distingue_var_estavel(self) -> None:
        estavel = calcular_raio_companheiro([np.diag([0.5, 0.4])])
        instavel = calcular_raio_companheiro([np.diag([1.1, 0.4])])

        self.assertLess(estavel, 1.0)
        self.assertGreater(instavel, 1.0)

    def test_exige_evidencia_nas_duas_validacoes(self) -> None:
        comparacao = comparar_validacoes(
            execucao_sintetica(delta_bic=10.0, melhora=True),
            execucao_sintetica(delta_bic=8.0, melhora=True),
        )

        self.assertEqual(comparacao["evidencias_threshold_nas_duas"], 1)
        self.assertEqual(
            comparacao["recomendacao"],
            "prosseguir_com_threshold_var_parcimonioso",
        )


if __name__ == "__main__":
    unittest.main()
