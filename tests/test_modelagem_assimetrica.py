from __future__ import annotations

import unittest
from pathlib import Path

from experimentos.granger.comparar_modelagem_assimetrica import (
    ResultadoModelagem,
    comparar_resultados,
)
from experimentos.granger.executar_experimento_modelagem_assimetrica import (
    construir_etapas,
)
from experimentos.granger.gerar_base_modelagem_experimental import (
    normalizar_parametros,
)


def resultado(
    tipo: str,
    deltas: dict[str, float],
) -> ResultadoModelagem:
    metricas = {
        divisao: {
            "f1": 0.60 + deltas.get("f1", 0.0),
            "pr_auc": 0.55 + deltas.get("pr_auc", 0.0),
            "mcc": 0.30 + deltas.get("mcc", 0.0),
            "precisao": 0.58,
            "revocacao": 0.62,
            "acuracia_balanceada": 0.61,
            "brier": 0.22,
        }
        for divisao in ("validacao_1", "validacao_2")
    }
    return ResultadoModelagem(
        tipo=tipo,
        banco=Path(f"{tipo}.sqlite"),
        selecao=Path(f"{tipo}.json"),
        semente=42,
        catalogo_configuracoes_sha256="catalogo",
        atributos_sha256="atributos",
        modelo={
            "codigo": "logistica_c1_normal",
            "familia": "regressao_logistica",
            "parametros": {"C": 1.0, "class_weight": None},
            "limiar": 0.5,
        },
        resumo={},
        metricas=metricas,
        amostras={
            "validacao_1": {"linhas": 100, "positivos": 40, "negativos": 60},
            "validacao_2": {"linhas": 80, "positivos": 30, "negativos": 50},
        },
    )


def comparacao_domino() -> dict[str, object]:
    periodo = {
        "bivariado": {"classificaveis": 100},
        "assimetrico": {"classificaveis": 80},
    }
    return {
        "inclui_teste_final": False,
        "contextos": {
            "validacao_1": {"periodos": {"teste": periodo}},
            "validacao_2": {"periodos": {"teste": periodo}},
        },
    }


class ProvenienciaTest(unittest.TestCase):
    def test_remove_datas_e_identificadores_sem_remover_metodo(self) -> None:
        granger, domino = normalizar_parametros(
            {
                "data_inicio": "2022-01-04",
                "data_corte_treino": "2023-12-31",
                "data_fim_teste": "2024-12-31",
                "alpha": 0.05,
            },
            {
                "tipo_relacoes": "granger_assimetrico",
                "relacoes": "FDR e resposta positiva",
                "banco_relacoes": "validacao_1.sqlite",
                "banco_relacoes_sha256": "hash",
                "execucao_assimetrica_id": 1,
            },
        )

        self.assertEqual(granger, {"alpha": 0.05})
        self.assertEqual(
            domino,
            {
                "tipo_relacoes": "granger_assimetrico",
                "relacoes": "FDR e resposta positiva",
            },
        )


class ComparadorTest(unittest.TestCase):
    def test_exige_melhora_das_tres_metricas_nas_duas_validacoes(self) -> None:
        bivariado = resultado("bivariado", {})
        assimetrico = resultado(
            "assimetrico",
            {"f1": 0.02, "pr_auc": 0.01, "mcc": -0.01},
        )

        relatorio = comparar_resultados(
            bivariado,
            assimetrico,
            comparacao_domino(),
        )

        self.assertFalse(relatorio["melhora_consistente_nas_duas_validacoes"])
        self.assertEqual(relatorio["decisao"], "manter_granger_bivariado")


class ExecutorTest(unittest.TestCase):
    def test_constroi_quatro_bases_duas_validacoes_e_comparacao(self) -> None:
        raiz = Path("resultados/experimento")
        entradas = {
            "manifesto_ffill": raiz / "manifesto.json",
            "comparacao_domino": raiz / "domino/comparacao.json",
            "validacao_1_granger": raiz / "v1-granger.sqlite",
            "validacao_2_granger": raiz / "v2-granger.sqlite",
            "validacao_1_domino_bivariado": raiz / "v1-bi.sqlite",
            "validacao_1_domino_assimetrico": raiz / "v1-as.sqlite",
            "validacao_2_domino_bivariado": raiz / "v2-bi.sqlite",
            "validacao_2_domino_assimetrico": raiz / "v2-as.sqlite",
        }

        etapas = construir_etapas(entradas, raiz / "modelagem")

        self.assertEqual(len(etapas), 7)
        self.assertIn(
            "experimentos.granger.gerar_base_modelagem_experimental",
            etapas[0].comando,
        )
        self.assertIn("treinamento.executar_validacao", etapas[4].comando)
        self.assertIn(
            "experimentos.granger.comparar_modelagem_assimetrica",
            etapas[-1].comando,
        )


if __name__ == "__main__":
    unittest.main()
