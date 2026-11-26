from __future__ import annotations

import json
import unittest

from treinamento.executar_avaliacao_final import migracao_diagnostica_compativel


def metodologia(schema_granger: int) -> dict[str, object]:
    granger: dict[str, object] = {
        "alpha": 0.05,
        "ffill_limite": 3,
        "max_lag": 10,
    }
    if schema_granger == 3:
        granger.update(
            {
                "altas_destino": (
                    "retorno acima do maior entre zero e o percentil 0.90 "
                    "do destino no treino"
                ),
                "min_observacoes_altas_teste": 10,
            }
        )
    return {
        "schemas_fontes": {
            "preprocessamento": 9,
            "granger": schema_granger,
            "efeito_domino": 1,
        },
        "preprocessamento": {"unidade_mercado": "entreposto"},
        "granger": granger,
        "efeito_domino": {"janela_dias_uteis": 5},
        "modelagem": {"max_lag_retorno": 10},
    }


def documento_selecao(metodo: dict[str, object]) -> dict[str, object]:
    parametros = {"metodologia": metodo}
    return {
        "bases_modelagem": {
            "validacao_1": {"parametros_json": parametros},
            "validacao_2": {"parametros_json": parametros},
        }
    }


class CompatibilidadeMetodologicaTest(unittest.TestCase):
    def test_aceita_apenas_diagnosticos_adicionados_no_schema_3(self) -> None:
        documento = documento_selecao(metodologia(2))
        execucao_final = {
            "parametros_json": json.dumps(
                {"metodologia": metodologia(3)},
                ensure_ascii=False,
            )
        }

        self.assertTrue(
            migracao_diagnostica_compativel(documento, execucao_final)
        )

    def test_rejeita_mudanca_em_parametro_do_granger(self) -> None:
        documento = documento_selecao(metodologia(2))
        metodo_final = metodologia(3)
        metodo_final["granger"]["alpha"] = 0.01
        execucao_final = {
            "parametros_json": json.dumps({"metodologia": metodo_final})
        }

        self.assertFalse(
            migracao_diagnostica_compativel(documento, execucao_final)
        )

    def test_rejeita_novo_campo_nao_autorizado(self) -> None:
        documento = documento_selecao(metodologia(2))
        metodo_final = metodologia(3)
        metodo_final["granger"]["novo_criterio"] = True
        execucao_final = {
            "parametros_json": json.dumps({"metodologia": metodo_final})
        }

        self.assertFalse(
            migracao_diagnostica_compativel(documento, execucao_final)
        )


if __name__ == "__main__":
    unittest.main()
