from __future__ import annotations

import argparse
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.granger.comparar_robustez_lags import (
    Execucao,
    Resultado,
    comparar_execucoes,
)
from granger.executar_granger import (
    avaliar_diagnosticos_e_previsao,
    avaliar_direcao_granger,
    construir_parser,
    criar_esquema,
    criar_indices_e_views,
    montar_matrizes_granger,
    preencher_lacunas,
    relacao_eh_robusta,
    validar_argumentos,
)


def criar_par(n: int = 80, n_treino: int | None = None) -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    origem = gerador.normal(size=n)
    destino = np.empty(n)
    destino[0] = gerador.normal()
    ruido = gerador.normal(scale=0.2, size=n)
    for indice in range(1, n):
        destino[indice] = (
            0.35 * destino[indice - 1]
            + 0.55 * origem[indice - 1]
            + ruido[indice]
        )
    treino = n if n_treino is None else n_treino
    return pd.DataFrame(
        {
            "retorno_a": origem,
            "retorno_b": destino,
            "segmento_par": 1,
            "periodo": np.where(
                np.arange(n) < treino,
                "treino",
                "teste",
            ),
        }
    )


class AmostraComumTest(unittest.TestCase):
    def test_padrao_reproduz_amostra_definida_pelo_max_lag(self) -> None:
        par = criar_par()

        alvo, restrita, irrestrita = montar_matrizes_granger(
            par,
            "retorno_a",
            "retorno_b",
            lag=10,
            max_lag_amostra=10,
        )

        np.testing.assert_allclose(alvo, par["retorno_b"].to_numpy()[10:])
        self.assertEqual(restrita.shape, (70, 11))
        self.assertEqual(irrestrita.shape, (70, 21))

    def test_limites_diferentes_usam_mesma_quantidade_de_observacoes(self) -> None:
        par = criar_par()

        candidatos_5, _, _ = avaliar_direcao_granger(
            par,
            "retorno_a",
            "retorno_b",
            max_lag=5,
            max_lag_amostra=20,
        )
        candidatos_10, _, _ = avaliar_direcao_granger(
            par,
            "retorno_a",
            "retorno_b",
            max_lag=10,
            max_lag_amostra=20,
        )

        self.assertEqual(
            {candidato["n_observacoes"] for candidato in candidatos_5},
            {60},
        )
        self.assertEqual(
            {candidato["n_observacoes"] for candidato in candidatos_10},
            {60},
        )
        for candidato_5, candidato_10 in zip(
            candidatos_5,
            candidatos_10[:5],
            strict=True,
        ):
            self.assertEqual(candidato_5["lag"], candidato_10["lag"])
            self.assertAlmostEqual(candidato_5["bic"], candidato_10["bic"])


class ArgumentosTest(unittest.TestCase):
    def criar_argumentos(
        self,
        diretorio: str,
        *extras: str,
    ) -> argparse.Namespace:
        entrada = Path(diretorio) / "entrada.sqlite"
        saida = Path(diretorio) / "saida.sqlite"
        entrada.touch()
        return construir_parser().parse_args(
            [str(entrada), "--output-db", str(saida), *extras]
        )

    def test_amostra_usa_max_lag_por_padrao(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            argumentos = self.criar_argumentos(diretorio, "--max-lag", "15")

            validar_argumentos(argumentos)

            self.assertEqual(argumentos.max_lag_amostra, 15)

    def test_amostra_nao_pode_ser_menor_que_lag_testado(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            argumentos = self.criar_argumentos(
                diretorio,
                "--max-lag",
                "10",
                "--max-lag-amostra",
                "5",
            )

            with self.assertRaisesRegex(ValueError, "maior ou igual"):
                validar_argumentos(argumentos)

    def test_ffill_zero_preserva_somente_observacoes(self) -> None:
        serie = pd.Series([10.0, np.nan, 12.0])

        preenchida = preencher_lacunas(serie, limite=0)

        pd.testing.assert_series_equal(preenchida, serie)


class DiagnosticosTest(unittest.TestCase):
    def test_modelo_irrestrito_melhora_previsao_fora_amostra(self) -> None:
        par = criar_par(n=160, n_treino=110)

        diagnostico, avaliacao = avaliar_diagnosticos_e_previsao(
            par,
            "retorno_a",
            "retorno_b",
            lag=1,
            max_lag_amostra=10,
            alpha=0.05,
        )

        self.assertEqual(diagnostico["status"], "avaliado")
        self.assertEqual(avaliacao["status"], "avaliado")
        self.assertEqual(avaliacao["n_observacoes"], 50)
        self.assertEqual(avaliacao["melhora_rmse"], 1)
        self.assertEqual(avaliacao["melhora_mae"], 1)
        self.assertGreater(avaliacao["ganho_rmse_percentual"], 0)
        self.assertGreater(avaliacao["ganho_mae_percentual"], 0)
        self.assertGreaterEqual(avaliacao["limite_alta_destino"], 0)

    def test_avalia_separadamente_as_altas_do_destino(self) -> None:
        par = criar_par(n=500, n_treino=200)

        _, avaliacao = avaliar_diagnosticos_e_previsao(
            par,
            "retorno_a",
            "retorno_b",
            lag=1,
            max_lag_amostra=10,
            alpha=0.05,
        )

        self.assertEqual(avaliacao["altas_avaliadas"], 1)
        self.assertGreaterEqual(avaliacao["n_observacoes_altas"], 10)
        self.assertIsNotNone(avaliacao["rmse_restrito_altas"])
        self.assertIsNotNone(avaliacao["mae_irrestrito_altas"])

    def test_relacao_robusta_exige_todos_os_criterios(self) -> None:
        diagnostico = {
            "status": "avaliado",
            "sem_autocorrelacao_irrestrito": 1,
        }
        avaliacao = {
            "status": "avaliado",
            "melhora_rmse": 1,
            "melhora_mae": 1,
        }

        self.assertTrue(relacao_eh_robusta(True, diagnostico, avaliacao))
        self.assertFalse(relacao_eh_robusta(False, diagnostico, avaliacao))
        self.assertFalse(
            relacao_eh_robusta(
                True,
                diagnostico,
                {**avaliacao, "melhora_mae": 0},
            )
        )


class EsquemaTest(unittest.TestCase):
    def test_cria_estruturas_de_diagnostico_e_robustez(self) -> None:
        with sqlite3.connect(":memory:") as conexao:
            criar_esquema(conexao)
            criar_indices_e_views(conexao)
            estruturas = {
                linha[0]
                for linha in conexao.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }

        self.assertIn("diagnosticos_residuos", estruturas)
        self.assertIn("avaliacoes_preditivas", estruturas)
        self.assertIn("relacoes_granger_robustas", estruturas)

        with sqlite3.connect(":memory:") as conexao:
            criar_esquema(conexao)
            colunas = {
                linha[1]
                for linha in conexao.execute(
                    "PRAGMA table_info(avaliacoes_preditivas)"
                )
            }

        self.assertIn("limite_alta_destino", colunas)
        self.assertIn("ganho_rmse_altas_percentual", colunas)


class ComparacaoTest(unittest.TestCase):
    def test_identifica_significancia_instavel(self) -> None:
        chave = ("tomate", "curitiba", "maringa")
        execucoes = []
        for limite in (5, 10, 15, 20):
            parametros = {
                "alpha": 0.05,
                "max_lag": limite,
                "max_lag_amostra": 20,
                "selecao_lag": "BIC em amostra fixa pelo max_lag_amostra",
            }
            execucoes.append(
                Execucao(
                    caminho=Path(f"lag-{limite}.sqlite"),
                    max_lag=limite,
                    max_lag_amostra=20,
                    banco_origem_sha256="hash-igual",
                    parametros=parametros,
                    resultados={
                        chave: Resultado(
                            status="testado",
                            n_observacoes=100,
                            lag_selecionado=1,
                            bic=1.0,
                            p_valor=0.01,
                            q_valor_fdr=0.01,
                            significativo_fdr=limite >= 15,
                        )
                    },
                    bics={
                        chave: {
                            lag: float(lag)
                            for lag in range(1, limite + 1)
                        }
                    },
                )
            )

        relatorio = comparar_execucoes(execucoes)

        self.assertEqual(relatorio["comparacao"]["significancia_instavel"], 1)
        self.assertEqual(relatorio["comparacao"]["lag_instavel"], 0)


if __name__ == "__main__":
    unittest.main()
