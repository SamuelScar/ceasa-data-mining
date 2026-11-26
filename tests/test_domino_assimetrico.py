from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from experimentos.granger.comparar_domino_assimetrico import (
    Execucao,
    comparar_contexto,
)
from experimentos.granger.executar_domino_assimetrico import (
    carregar_relacoes_assimetricas,
)
from experimentos.granger.executar_experimento_domino_assimetrico import (
    construir_etapas,
)


def periodo(
    positivos: int,
    negativos: int,
) -> dict[str, int | float | bool]:
    classificaveis = positivos + negativos
    return {
        "eventos": classificaveis,
        "classificaveis": classificaveis,
        "confirmados": positivos,
        "nao_confirmados": negativos,
        "movimentos_simultaneos": 0,
        "sem_base_destino": 0,
        "sem_cobertura_destino": 0,
        "cobertura_classificavel": 1.0,
        "proporcao_confirmada": positivos / classificaveis,
        "possui_duas_classes": positivos > 0 and negativos > 0,
    }


class RelacoesAssimetricasTest(unittest.TestCase):
    def test_filtra_sem_usar_metricas_do_periodo_de_avaliacao(self) -> None:
        conexao = sqlite3.connect(":memory:")
        conexao.execute(
            """
            CREATE TABLE resultados_assimetricos (
                resultado_granger_id INTEGER,
                produto TEXT,
                origem TEXT,
                destino TEXT,
                lag_selecionado INTEGER,
                estatistica_f REAL,
                p_valor REAL,
                q_valor_fdr REAL,
                status TEXT,
                significativo_fdr INTEGER,
                efeito_positivo INTEGER
            )
            """
        )
        conexao.executemany(
            """
            INSERT INTO resultados_assimetricos
            VALUES (?, 'batata', ?, 'cascavel', 1, 10, 0.001, 0.01,
                    'testado', ?, ?)
            """,
            [
                (1, "curitiba", 1, 1),
                (2, "londrina", 1, 0),
                (3, "maringa", 0, 1),
            ],
        )

        relacoes = carregar_relacoes_assimetricas(conexao)

        self.assertEqual(len(relacoes), 1)
        self.assertEqual(relacoes.iloc[0]["origem"], "curitiba")
        conexao.close()


class ComparadorTest(unittest.TestCase):
    def test_gate_exige_duas_classes_em_treino_e_teste(self) -> None:
        chave = {("batata", "curitiba", "cascavel")}
        bivariado = Execucao(
            caminho=Path("bivariado.sqlite"),
            tipo_relacoes="granger_bivariado",
            banco_origem_sha256="hash",
            assinatura_configuracao="config",
            resumo={},
            relacoes=chave,
            periodos={"treino": periodo(5, 5), "teste": periodo(4, 4)},
        )
        assimetrico = Execucao(
            caminho=Path("assimetrico.sqlite"),
            tipo_relacoes="granger_assimetrico",
            banco_origem_sha256="hash",
            assinatura_configuracao="config",
            resumo={},
            relacoes=chave,
            periodos={"treino": periodo(3, 3), "teste": periodo(2, 0)},
        )

        comparacao = comparar_contexto(
            "validacao_1",
            bivariado,
            assimetrico,
        )

        self.assertFalse(comparacao["assimetrico_apto_para_base_modelagem"])


class ExecutorTest(unittest.TestCase):
    def test_constroi_quatro_bancos_e_uma_comparacao(self) -> None:
        raiz = Path("resultados/experimento")
        entradas = {
            "validacao_1_granger": raiz / "v1-granger.sqlite",
            "validacao_2_granger": raiz / "v2-granger.sqlite",
            "validacao_1_assimetrico": raiz / "v1-assimetrico.sqlite",
            "validacao_2_assimetrico": raiz / "v2-assimetrico.sqlite",
        }

        etapas = construir_etapas(entradas, raiz / "efeito-domino")

        self.assertEqual(len(etapas), 5)
        self.assertIn(
            "efeitos/domino/executar_efeito_domino.py",
            etapas[0].comando,
        )
        self.assertIn(
            "experimentos.granger.executar_domino_assimetrico",
            etapas[1].comando,
        )
        self.assertIn(
            "experimentos.granger.comparar_domino_assimetrico",
            etapas[-1].comando,
        )


if __name__ == "__main__":
    unittest.main()
