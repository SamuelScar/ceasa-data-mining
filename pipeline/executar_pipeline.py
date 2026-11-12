"""Executa sequencialmente as etapas de preparação e caracterização dos dados."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class Etapa:
    nome: str
    descricao: str
    comando: list[str]


ETAPAS = [
    Etapa(
        nome="Pré-processamento",
        descricao="normalização, conversões e indicadores de qualidade",
        comando=[
            sys.executable,
            "preprocessamento/tratar_ceasa.py",
            "data/cotacoes.sqlite",
            "--output-db",
            "resultados/preprocessamento/cotacoes_preprocessadas.sqlite",
            "--overwrite",
        ],
    ),
    Etapa(
        nome="Séries e Granger",
        descricao="construção das séries e relações de influência temporal",
        comando=[
            sys.executable,
            "granger/executar_granger.py",
            "resultados/preprocessamento/cotacoes_preprocessadas.sqlite",
            "--output-db",
            "resultados/granger/granger.sqlite",
            "--overwrite",
        ],
    ),
    Etapa(
        nome="Efeito Dominó",
        descricao="localização e classificação das propagações de preço",
        comando=[
            sys.executable,
            "efeitos/domino/executar_efeito_domino.py",
            "resultados/granger/granger.sqlite",
            "--output-db",
            "resultados/efeitos/domino/efeito_domino.sqlite",
            "--overwrite",
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


def executar() -> int:
    inicio_total = perf_counter()
    print("Pipeline completo das cotações CEASA", flush=True)
    print("Ordem: pré-processamento -> Granger -> Efeito Dominó", flush=True)
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
            f"\nOK: {etapa.nome} concluído em "
            f"{duracao_legivel(perf_counter() - inicio_etapa)}.",
            flush=True,
        )
        exibir_estado(indice, inicio_total)

    print(
        "\nPipeline concluído com sucesso em "
        f"{duracao_legivel(perf_counter() - inicio_total)}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
