from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.granger.executar_experimento_calendario_operacional import (
    Configuracao,
    construir_pares_operacionais,
)
from experimentos.granger.executar_experimento_series_temporais import (
    Contexto,
    ResultadoSemanal,
    construir_series_semanais,
    qualificar_pares_diarios,
    recalcular_fdr,
    selecionar_configuracao,
)


def serie_diaria_minima() -> pd.DataFrame:
    registros = []
    precos = {
        "2023-01-02": 10.0,
        "2023-01-03": 14.0,
        "2023-01-09": 30.0,
        "2023-01-12": 40.0,
        "2023-01-16": 20.0,
        "2023-01-17": 24.0,
    }
    for data_cotacao, preco in precos.items():
        registros.append(
            {
                "produto": "batata",
                "ceasa_slug": "ceasa-pr",
                "uf": "PR",
                "mercado": "curitiba",
                "data_cotacao": pd.Timestamp(data_cotacao),
                "periodo": "treino" if data_cotacao <= "2023-01-11" else "teste",
                "preco_kg_observado": preco,
                "preco_kg": preco,
                "n_cotacoes": 1,
                "n_exatas": 1,
                "n_calculadas": 0,
                "n_padronizadas": 0,
                "n_inferidas": 0,
                "observado": 1,
                "preenchido": 0,
                "segmento_serie": 1,
                "retorno_log": np.nan,
            }
        )
    return pd.DataFrame(registros)


class SeriesSemanaisTest(unittest.TestCase):
    def test_agrega_mediana_e_descarta_semana_que_cruza_corte(self) -> None:
        _, mapa, periodos_treino = construir_series_semanais(
            serie_diaria_minima(),
            date(2023, 1, 2),
            date(2023, 1, 11),
            date(2023, 1, 20),
        )

        serie = mapa[("batata", "curitiba")]
        self.assertEqual(periodos_treino, 1)
        self.assertEqual(serie.loc["2023-01-02", "preco_kg_observado"], 12.0)
        self.assertTrue(pd.isna(serie.loc["2023-01-09", "preco_kg_observado"]))
        self.assertEqual(serie.loc["2023-01-16", "preco_kg_observado"], 22.0)
        self.assertEqual(serie.loc["2023-01-02", "segmento_serie"], 1)
        self.assertEqual(serie.loc["2023-01-16", "segmento_serie"], 2)


class QualidadeDiariaTest(unittest.TestCase):
    def setUp(self) -> None:
        datas = pd.bdate_range("2023-01-02", periods=10)
        retornos = [np.nan, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
        series = []
        for mercado in ("curitiba", "maringa"):
            for data_cotacao, retorno in zip(datas, retornos, strict=True):
                series.append(
                    {
                        "produto": "batata",
                        "mercado": mercado,
                        "data_cotacao": data_cotacao,
                        "periodo": "treino",
                        "retorno_log": retorno,
                    }
                )
        pares = pd.DataFrame(
            [
                {
                    "id": 1,
                    "produto": "batata",
                    "mercado_a": "curitiba",
                    "mercado_b": "maringa",
                    "dias_observados_comuns_treino": 8,
                    "elegivel": 1,
                }
            ]
        )
        self.contexto = Contexto(
            nome="validacao_1",
            caminho=Path("granger.sqlite"),
            parametros={},
            series=pd.DataFrame(series),
            pares=pares,
            resultados=pd.DataFrame(),
        )

    def test_exige_cobertura_e_taxa_de_mudancas(self) -> None:
        qualificados = qualificar_pares_diarios(
            self.contexto,
            cobertura_minima=0.75,
            taxa_mudancas_minima=0.15,
        )
        rigorosos = qualificar_pares_diarios(
            self.contexto,
            cobertura_minima=0.75,
            taxa_mudancas_minima=0.25,
        )

        self.assertTrue(bool(qualificados.iloc[0]["qualificado_qualidade"]))
        self.assertFalse(bool(rigorosos.iloc[0]["qualificado_qualidade"]))

    def test_recalcula_fdr_somente_na_familia_filtrada(self) -> None:
        resultados = pd.DataFrame(
            [
                {"par_id": 1, "status": "testado", "p_valor": 0.04},
                {"par_id": 2, "status": "testado", "p_valor": 0.001},
            ]
        )

        filtrados = recalcular_fdr(resultados, {1}, alpha=0.05)

        self.assertEqual(len(filtrados), 1)
        self.assertAlmostEqual(filtrados.iloc[0]["q_valor_fdr"], 0.04)
        self.assertEqual(filtrados.iloc[0]["significativo_fdr"], 1)


def resultados_semanais(robusto: bool, quantidade: int) -> pd.DataFrame:
    linhas = []
    for indice in range(quantidade):
        linhas.append(
            {
                "status": "testado",
                "significativo_fdr": 1,
                "status_diagnostico": "avaliado",
                "residuos_adequados": int(robusto),
                "status_avaliacao": "avaliado",
                "melhora_rmse": int(robusto),
                "melhora_mae": int(robusto),
                "produto": "batata",
                "origem": f"origem-{indice}",
                "destino": "destino",
                "lag_selecionado": 1,
                "p_valor": 0.01,
                "q_valor_fdr": 0.02,
            }
        )
    return pd.DataFrame(linhas)


class SelecaoSemanalTest(unittest.TestCase):
    def test_prioriza_robustez_obtida_somente_na_validacao_um(self) -> None:
        robusto = ResultadoSemanal(
            cobertura_minima=0.85,
            max_lag=2,
            periodos_treino=100,
            minimo_periodos_comuns=85,
            pares=pd.DataFrame(),
            resultados=resultados_semanais(True, 1),
        )
        numeroso = ResultadoSemanal(
            cobertura_minima=0.75,
            max_lag=4,
            periodos_treino=100,
            minimo_periodos_comuns=75,
            pares=pd.DataFrame(),
            resultados=resultados_semanais(False, 3),
        )

        selecionado = selecionar_configuracao([numeroso, robusto])

        self.assertEqual(selecionado.nome, robusto.nome)


class CalendarioOperacionalTest(unittest.TestCase):
    def test_remove_fechamento_e_conta_preenchimento_em_sessoes_ativas(self) -> None:
        registros = []
        for mercado in ("curitiba", "maringa"):
            for data_cotacao, preco in (
                ("2023-01-02", 10.0),
                ("2023-01-06", 12.0),
            ):
                registros.append(
                    {
                        "produto": "batata",
                        "mercado": mercado,
                        "data_cotacao": pd.Timestamp(data_cotacao),
                        "preco_kg_observado": preco,
                        "observado": 1,
                    }
                )
        calendarios = {
            "curitiba": pd.DatetimeIndex(
                pd.to_datetime(
                    ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
                )
            ),
            "maringa": pd.DatetimeIndex(
                pd.to_datetime(
                    ["2023-01-02", "2023-01-04", "2023-01-05", "2023-01-06"]
                )
            ),
        }
        configuracao = Configuracao("teste", cobertura_minima=0.3, taxa_mudancas_minima=0.0)

        pares, frames = construir_pares_operacionais(
            pd.DataFrame(registros),
            calendarios,
            date(2023, 1, 5),
            configuracao,
            produtos=("batata",),
            mercados=("curitiba", "maringa"),
            minimo_dias_comuns=1,
            limite_ffill=3,
        )

        frame = frames[1]
        self.assertEqual(pares.iloc[0]["periodos_operacionais_comuns_treino"], 3)
        self.assertNotIn(pd.Timestamp("2023-01-03"), frame.index)
        self.assertEqual(frame.loc["2023-01-04", "preenchido_a"], 1)
        self.assertEqual(frame.loc["2023-01-04", "preenchido_b"], 1)
        self.assertEqual(frame.loc["2023-01-04", "segmento_par"], 1)


if __name__ == "__main__":
    unittest.main()
