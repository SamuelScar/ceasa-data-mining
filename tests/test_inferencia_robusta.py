from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from experimentos.granger.comparar_inferencia_robusta import (
    Execucao,
    Resultado,
    comparar_execucoes,
)
from experimentos.granger.executar_experimento_inferencia_robusta import (
    construir_etapas,
)
from experimentos.granger.executar_inferencia_robusta import (
    ajustar_inferencia_robusta,
    covariancia_hac_segmentada,
)


def partes_sinteticas() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    gerador = np.random.default_rng(42)
    partes = []
    for tamanho in (350, 300):
        origem = gerador.normal(size=tamanho)
        destino = np.zeros(tamanho)
        for indice in range(1, tamanho):
            escala = 0.15 + 0.3 * abs(origem[indice - 1])
            destino[indice] = (
                0.25 * destino[indice - 1]
                + 0.8 * origem[indice - 1]
                + gerador.normal(scale=escala)
            )
        alvo = destino[1:]
        restrita = np.column_stack([np.ones(tamanho - 1), destino[:-1]])
        irrestrita = np.column_stack([restrita, origem[:-1]])
        partes.append((alvo, restrita, irrestrita))
    return partes


def resultado(robusta_hac: bool) -> Resultado:
    return Resultado(
        significativo_classico=True,
        significativo_hc3=True,
        significativo_hac=True,
        robusta_hc3=robusta_hac,
        robusta_hac=robusta_hac,
        alta_robusta_hc3=False,
        alta_robusta_hac=False,
    )


def execucao(contexto: str, robusta_hac: bool) -> Execucao:
    assinatura = json.dumps(
        {
            "alpha": 0.05,
            "coeficientes": "OLS idêntico ao Granger bivariado",
        },
        sort_keys=True,
    )
    return Execucao(
        contexto=contexto,
        caminho=Path(f"{contexto}.sqlite"),
        assinatura=assinatura,
        resumo={},
        resultados={
            ("batata", "curitiba", "cascavel"): resultado(robusta_hac)
        },
    )


class InferenciaRobustaTest(unittest.TestCase):
    def test_detecta_origem_com_erros_heterocedasticos(self) -> None:
        inferencia = ajustar_inferencia_robusta(partes_sinteticas(), lag=1)

        self.assertLess(inferencia["p_valor_hc3"], 0.05)
        self.assertLess(inferencia["p_valor_hac"], 0.05)
        self.assertGreater(inferencia["resposta_acumulada"], 0)

    def test_hac_segmentada_produz_matriz_simetrica(self) -> None:
        partes = partes_sinteticas()
        regressores = [parte[2] for parte in partes]
        alvos = [parte[0] for parte in partes]
        matriz = np.vstack(regressores)
        alvo = np.concatenate(alvos)
        beta = np.linalg.lstsq(matriz, alvo, rcond=None)[0]
        residuos = [
            y - x @ beta for y, x in zip(alvos, regressores, strict=True)
        ]

        covariancia = covariancia_hac_segmentada(
            regressores,
            residuos,
            max_lags=1,
        )

        np.testing.assert_allclose(covariancia, covariancia.T)

    def test_rejeita_coeficientes_diferentes_do_granger_original(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "coeficientes OLS divergiram"):
            ajustar_inferencia_robusta(
                partes_sinteticas(),
                lag=1,
                coeficientes_esperados=np.zeros(3),
            )

    def test_exige_robustez_hac_nas_duas_validacoes(self) -> None:
        relatorio = comparar_execucoes(
            execucao("validacao_1", robusta_hac=True),
            execucao("validacao_2", robusta_hac=True),
        )

        self.assertEqual(
            relatorio["recomendacao"],
            "avaliar_hac_no_efeito_domino",
        )

    def test_executor_usa_as_duas_validacoes_sem_teste_final(self) -> None:
        caminhos = {
            "validacao_1": Path("raiz/validacao_1/granger/ffill-3.sqlite"),
            "validacao_2": Path("raiz/validacao_2/granger/ffill-3.sqlite"),
        }

        etapas = construir_etapas(caminhos, Path("raiz/granger-inferencia-robusta"))
        comandos = " ".join(" ".join(etapa.comando) for etapa in etapas)

        self.assertEqual(len(etapas), 3)
        self.assertIn("validacao_1/granger/ffill-3.sqlite", comandos)
        self.assertIn("validacao_2/granger/ffill-3.sqlite", comandos)
        self.assertNotIn("teste_final", comandos)


if __name__ == "__main__":
    unittest.main()
