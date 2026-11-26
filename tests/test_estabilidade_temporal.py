from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experimentos.granger.analisar_estabilidade_temporal import resposta_dinamica
from experimentos.granger.executar_estabilidade_temporal import (
    construir_janelas_contexto,
)
from experimentos.granger.executar_modelo_condicional import (
    aplicar_recorte_temporal,
)


class JanelasTemporaisTest(unittest.TestCase):
    def test_constroi_tres_janelas_na_primeira_validacao(self) -> None:
        contexto = {
            "nome": "validacao_1",
            "inicio_series": "2022-01-04",
            "corte_treino": "2023-12-31",
            "fim_avaliacao": "2024-12-31",
        }

        janelas = construir_janelas_contexto(contexto)

        self.assertEqual(len(janelas), 3)
        self.assertEqual(janelas[0]["data_corte_treino"], "2023-06-30")
        self.assertEqual(janelas[-1]["data_fim_teste"], "2024-06-30")

    def test_constroi_sete_janelas_na_segunda_validacao(self) -> None:
        contexto = {
            "nome": "validacao_2",
            "inicio_series": "2022-01-04",
            "corte_treino": "2024-12-31",
            "fim_avaliacao": "2025-10-31",
        }

        janelas = construir_janelas_contexto(contexto)

        self.assertEqual(len(janelas), 7)
        self.assertEqual(janelas[-1]["data_corte_treino"], "2024-12-31")
        self.assertEqual(janelas[-1]["data_fim_teste"], "2025-06-30")

    def test_recorte_redefine_treino_e_teste_sem_alterar_origem(self) -> None:
        series = pd.DataFrame(
            {
                "data_cotacao": pd.to_datetime(
                    ["2022-01-04", "2023-06-30", "2023-07-03"]
                ),
                "periodo": ["treino", "treino", "treino"],
            }
        )
        parametros = {
            "data_inicio": "2022-01-04",
            "data_corte_treino": "2023-12-31",
            "data_fim_teste": "2024-12-31",
        }

        recorte, datas = aplicar_recorte_temporal(
            series,
            parametros,
            "2022-01-04",
            "2023-06-30",
            "2023-09-30",
        )

        self.assertEqual(
            recorte["periodo"].tolist(),
            ["treino", "treino", "teste"],
        )
        self.assertEqual(datas["data_corte_treino"], "2023-06-30")


class RespostaDinamicaTest(unittest.TestCase):
    def test_acumula_resposta_positiva_da_origem_no_destino(self) -> None:
        matriz = np.array([[0.0, 0.0], [0.5, 0.0]])

        resposta = resposta_dinamica(
            [matriz],
            indice_origem=0,
            indice_destino=1,
            horizonte=5,
        )

        self.assertEqual(resposta, 0.5)


if __name__ == "__main__":
    unittest.main()
