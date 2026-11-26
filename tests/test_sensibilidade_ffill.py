from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experimentos.granger.comparar_sensibilidade_ffill import (
    Execucao,
    Resultado,
    comparar_execucoes,
    validar_contexto,
)
from experimentos.granger.executar_sensibilidade_ffill import (
    CONTEXTOS,
    LIMITES_FFILL,
    caminho_granger,
    construir_etapa_comparacao,
    construir_etapas_contexto,
    descartar_grangers_alternativos,
)


CHAVE = ("tomate", "curitiba", "maringa")


def criar_resultado(
    *,
    significativo: bool = True,
    robusto: bool = False,
) -> Resultado:
    return Resultado(
        status="testado",
        lag_selecionado=2,
        significativo_fdr=significativo,
        residuos_adequados=True,
        melhora_rmse=True,
        melhora_mae=robusto,
        altas_avaliadas=True,
        melhora_rmse_altas=True,
        melhora_mae_altas=robusto,
        relacao_robusta=robusto,
    )


def criar_execucoes(contexto: str, robusto_no_ffill_1: bool) -> list[Execucao]:
    execucoes = []
    for limite in LIMITES_FFILL:
        parametros = {
            "alpha": 0.05,
            "ffill_limite": limite,
            "max_lag": 10,
        }
        execucoes.append(
            Execucao(
                contexto=contexto,
                caminho=Path(f"{contexto}-ffill-{limite}.sqlite"),
                ffill_limite=limite,
                banco_origem_sha256=f"hash-{contexto}",
                parametros=parametros,
                resultados={
                    CHAVE: criar_resultado(
                        robusto=robusto_no_ffill_1 and limite == 1
                    )
                },
            )
        )
    return execucoes


class ComparadorTest(unittest.TestCase):
    def test_resume_estabilidade_por_contexto_e_entre_validacoes(self) -> None:
        validacao_1 = criar_execucoes("validacao_1", robusto_no_ffill_1=True)
        validacao_2 = criar_execucoes("validacao_2", robusto_no_ffill_1=True)

        relatorio = comparar_execucoes(validacao_1, validacao_2)

        estabilidade = relatorio["validacoes"]["validacao_1"]["estabilidade"]
        self.assertEqual(estabilidade["significativas_em_todos_ffill"], 1)
        self.assertEqual(estabilidade["robustas_em_algum_ffill"], 1)
        self.assertEqual(
            relatorio["estabilidade_entre_validacoes"]["1"][
                "robustas_nas_duas"
            ],
            1,
        )

    def test_rejeita_bancos_preprocessados_diferentes_no_mesmo_contexto(self) -> None:
        execucoes = criar_execucoes("validacao_1", robusto_no_ffill_1=False)
        divergente = execucoes[2]
        execucoes[2] = Execucao(
            contexto=divergente.contexto,
            caminho=divergente.caminho,
            ffill_limite=divergente.ffill_limite,
            banco_origem_sha256="outro-hash",
            parametros=divergente.parametros,
            resultados=divergente.resultados,
        )

        with self.assertRaisesRegex(ValueError, "pré-processados diferentes"):
            validar_contexto(execucoes)


class ExecutorTest(unittest.TestCase):
    def test_constroi_tres_grangers_sobre_o_mesmo_preprocessado(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio) / "execucao"
            etapas = construir_etapas_contexto(
                raiz,
                Path("data/cotacoes.sqlite"),
                CONTEXTOS[0],
            )

        self.assertEqual(len(etapas), 4)
        preprocessados = {etapa.comando[2] for etapa in etapas[1:]}
        self.assertEqual(len(preprocessados), 1)
        limites = {
            etapa.comando[etapa.comando.index("--ffill-limite") + 1]
            for etapa in etapas[1:]
        }
        self.assertEqual(limites, {"0", "1", "3"})

    def test_comparacao_recebe_as_seis_execucoes(self) -> None:
        etapa = construir_etapa_comparacao(Path("resultados/experimento"))

        bancos = [valor for valor in etapa.comando if valor.endswith(".sqlite")]
        self.assertEqual(len(bancos), 6)

    def test_descarta_somente_bancos_de_limites_alternativos(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio)
            for contexto in CONTEXTOS:
                for limite in LIMITES_FFILL:
                    caminho = caminho_granger(raiz, contexto, limite)
                    caminho.parent.mkdir(parents=True, exist_ok=True)
                    caminho.write_bytes(f"ffill={limite}".encode())

            descartados = descartar_grangers_alternativos(raiz)

            self.assertEqual(len(descartados), 4)
            for contexto in CONTEXTOS:
                self.assertTrue(caminho_granger(raiz, contexto, 3).is_file())
                self.assertFalse(caminho_granger(raiz, contexto, 0).exists())
                self.assertFalse(caminho_granger(raiz, contexto, 1).exists())


if __name__ == "__main__":
    unittest.main()
