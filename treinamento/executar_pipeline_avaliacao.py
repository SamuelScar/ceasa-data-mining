"""Executa a validação dos modelos e a avaliação final em sequência."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


BASE_MODELAGEM = Path("resultados/modelagem/base_modelagem.sqlite")
BANCO_VALIDACAO = Path("resultados/treinamento/validacao.sqlite")
CONFIGURACAO_SELECIONADA = Path(
    "resultados/treinamento/configuracao_selecionada.json"
)
BANCO_AVALIACAO_FINAL = Path("resultados/treinamento/avaliacao_final.sqlite")
MODELO_FINAL = Path("resultados/treinamento/modelo_final.pkl")


@dataclass(frozen=True)
class Etapa:
    nome: str
    descricao: str
    comando: list[str]


ETAPAS = [
    Etapa(
        nome="Validação e seleção",
        descricao="comparação dos modelos somente nas divisões de desenvolvimento",
        comando=[
            sys.executable,
            "-m",
            "treinamento.executar_validacao",
            str(BASE_MODELAGEM),
            "--output-db",
            str(BANCO_VALIDACAO),
            "--selection-json",
            str(CONFIGURACAO_SELECIONADA),
            "--overwrite",
        ],
    ),
    Etapa(
        nome="Avaliação final",
        descricao="ajuste final e avaliação única no período oficial de teste",
        comando=[
            sys.executable,
            "-m",
            "treinamento.executar_avaliacao_final",
            str(BASE_MODELAGEM),
            str(CONFIGURACAO_SELECIONADA),
            "--output-db",
            str(BANCO_AVALIACAO_FINAL),
            "--model-output",
            str(MODELO_FINAL),
        ],
    ),
]


def duracao_legivel(segundos: float) -> str:
    total = round(segundos)
    horas, resto = divmod(total, 3600)
    minutos, segundos_restantes = divmod(resto, 60)
    if horas:
        return f"{horas}h {minutos:02d}min {segundos_restantes:02d}s"
    if minutos:
        return f"{minutos}min {segundos_restantes:02d}s"
    return f"{segundos_restantes}s"


def barra_progresso(concluidas: int, total: int) -> str:
    largura = 36
    preenchido = round(largura * concluidas / total)
    barra = "#" * preenchido + "-" * (largura - preenchido)
    percentual = 100 * concluidas / total
    return f"[{barra}] {percentual:6.2f}%"


def exibir_estado(concluidas: int, inicio: float) -> None:
    total = len(ETAPAS)
    restantes = total - concluidas
    print("\n" + "=" * 72, flush=True)
    print(f"Progresso geral: {barra_progresso(concluidas, total)}", flush=True)
    print(
        f"Etapas concluídas: {concluidas}/{total} | "
        f"Restantes: {restantes} | Tempo: {duracao_legivel(perf_counter() - inicio)}",
        flush=True,
    )
    print("=" * 72, flush=True)


def validar_precondicoes() -> bool:
    if not BASE_MODELAGEM.is_file():
        print(
            "ERRO: a base de modelagem não foi encontrada em "
            f"{BASE_MODELAGEM}.",
            flush=True,
        )
        print(
            "Execute primeiro o serviço base-modelagem ou o pipeline-completo.",
            flush=True,
        )
        return False

    artefatos_finais_existentes = [
        caminho
        for caminho in (BANCO_AVALIACAO_FINAL, MODELO_FINAL)
        if caminho.exists()
    ]
    if artefatos_finais_existentes:
        print(
            "ERRO: a avaliação final não será iniciada porque já existem "
            "artefatos oficiais:",
            flush=True,
        )
        for caminho in artefatos_finais_existentes:
            print(f"- {caminho}", flush=True)
        print(
            "O pipeline não sobrescreve resultados oficiais. Para uma nova execução, "
            "preserve e remova conscientemente esses artefatos antes de continuar.",
            flush=True,
        )
        return False

    return True


def executar() -> int:
    print("Pipeline de treinamento e avaliação", flush=True)
    print("Ordem: validação e seleção -> avaliação final oficial", flush=True)

    if not validar_precondicoes():
        return 1

    inicio_total = perf_counter()
    exibir_estado(0, inicio_total)

    for indice, etapa in enumerate(ETAPAS, start=1):
        inicio_etapa = perf_counter()
        print(
            f"\n>>> Etapa {indice}/{len(ETAPAS)}: {etapa.nome}\n"
            f"    {etapa.descricao}\n",
            flush=True,
        )
        resultado = subprocess.run(etapa.comando, check=False)
        if resultado.returncode != 0:
            print(
                f"\nFALHA na etapa {indice}/{len(ETAPAS)}: {etapa.nome}",
                flush=True,
            )
            print(
                "As etapas seguintes não foram executadas. "
                f"Código de saída: {resultado.returncode}",
                flush=True,
            )
            exibir_estado(indice - 1, inicio_total)
            return resultado.returncode

        print(
            f"\nOK: {etapa.nome} concluída em "
            f"{duracao_legivel(perf_counter() - inicio_etapa)}.",
            flush=True,
        )
        exibir_estado(indice, inicio_total)

    print(
        "\nPipeline concluído com sucesso em "
        f"{duracao_legivel(perf_counter() - inicio_total)}.",
        flush=True,
    )
    print("Artefatos gerados:", flush=True)
    for caminho in (
        BANCO_VALIDACAO,
        CONFIGURACAO_SELECIONADA,
        BANCO_AVALIACAO_FINAL,
        MODELO_FINAL,
    ):
        print(f"- {caminho}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
