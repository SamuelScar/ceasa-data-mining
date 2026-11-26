#!/usr/bin/env python3
"""Orquestra a comparação do Efeito Dominó bivariado e assimétrico."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pipeline.progresso_terminal import PainelPipeline, executar_comando


@dataclass(frozen=True)
class Etapa:
    nome: str
    descricao: str
    comando: list[str]


def agora_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_arquivo(caminho: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while bloco := arquivo.read(chunk_size):
            digest.update(bloco)
    return digest.hexdigest()


def gravar_manifesto(caminho: Path, manifesto: dict[str, object]) -> None:
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def caminhos_entrada(raiz: Path) -> dict[str, Path]:
    return {
        "manifesto_ffill": raiz / "manifesto.json",
        "validacao_1_granger": raiz / "validacao_1/granger/ffill-3.sqlite",
        "validacao_2_granger": raiz / "validacao_2/granger/ffill-3.sqlite",
        "manifesto_assimetrico": raiz / "granger-assimetrico/manifesto.json",
        "validacao_1_assimetrico": (
            raiz / "granger-assimetrico/validacao_1.sqlite"
        ),
        "validacao_2_assimetrico": (
            raiz / "granger-assimetrico/validacao_2.sqlite"
        ),
    }


def validar_manifesto(caminho: Path, tipo: str | None = None) -> None:
    manifesto = json.loads(caminho.read_text(encoding="utf-8"))
    if manifesto.get("status") != "concluida":
        raise ValueError(f"execução não concluída: {caminho}")
    if manifesto.get("inclui_teste_final") is not False:
        raise ValueError(f"o manifesto não protege o teste final: {caminho}")
    if tipo is not None and manifesto.get("tipo") != tipo:
        raise ValueError(f"tipo de manifesto inesperado em: {caminho}")


def validar_entradas(raiz: Path) -> dict[str, Path]:
    entradas = caminhos_entrada(raiz)
    ausentes = [caminho for caminho in entradas.values() if not caminho.is_file()]
    if ausentes:
        nomes = ", ".join(str(caminho) for caminho in ausentes)
        raise FileNotFoundError(f"artefatos obrigatórios ausentes: {nomes}")
    validar_manifesto(entradas["manifesto_ffill"])
    validar_manifesto(
        entradas["manifesto_assimetrico"],
        "granger_assimetrico",
    )
    return entradas


def construir_etapas(
    entradas: dict[str, Path],
    saida: Path,
) -> list[Etapa]:
    bancos: dict[str, dict[str, Path]] = {}
    etapas: list[Etapa] = []
    for contexto in ("validacao_1", "validacao_2"):
        diretorio = saida / contexto
        bivariado = diretorio / "bivariado.sqlite"
        assimetrico = diretorio / "assimetrico.sqlite"
        bancos[contexto] = {
            "bivariado": bivariado,
            "assimetrico": assimetrico,
        }
        granger = entradas[f"{contexto}_granger"]
        granger_assimetrico = entradas[f"{contexto}_assimetrico"]
        etapas.extend(
            [
                Etapa(
                    nome=f"Efeito Dominó bivariado — {contexto}",
                    descricao="constrói a referência com o método atual",
                    comando=[
                        sys.executable,
                        "efeitos/domino/executar_efeito_domino.py",
                        str(granger),
                        "--output-db",
                        str(bivariado),
                    ],
                ),
                Etapa(
                    nome=f"Efeito Dominó assimétrico — {contexto}",
                    descricao=(
                        "usa somente FDR e resposta positiva aprendidos no treino"
                    ),
                    comando=[
                        sys.executable,
                        "-m",
                        "experimentos.granger.executar_domino_assimetrico",
                        str(granger),
                        str(granger_assimetrico),
                        "--output-db",
                        str(assimetrico),
                    ],
                ),
            ]
        )

    etapas.append(
        Etapa(
            nome="Comparação dos eventos do Efeito Dominó",
            descricao="mede cobertura e preservação das duas classes",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.comparar_domino_assimetrico",
                "--validacao-1-bivariado",
                str(bancos["validacao_1"]["bivariado"]),
                "--validacao-1-assimetrico",
                str(bancos["validacao_1"]["assimetrico"]),
                "--validacao-2-bivariado",
                str(bancos["validacao_2"]["bivariado"]),
                "--validacao-2-assimetrico",
                str(bancos["validacao_2"]["assimetrico"]),
                "--output-json",
                str(saida / "comparacao.json"),
            ],
        )
    )
    return etapas


def executar(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experimento_ffill", type=Path)
    args = parser.parse_args(argv)
    raiz = args.experimento_ffill
    saida = raiz / "granger-assimetrico/efeito-domino"
    try:
        entradas = validar_entradas(raiz)
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}", flush=True)
        return 2
    if saida.exists():
        print(f"ERRO: a saída já existe: {saida}", flush=True)
        return 2

    saida.mkdir(parents=True)
    etapas = construir_etapas(entradas, saida)
    inicio = perf_counter()
    manifesto: dict[str, object] = {
        "schema_version": 1,
        "tipo": "comparacao_domino_assimetrico",
        "status": "em_andamento",
        "iniciada_em": agora_iso(),
        "concluida_em": None,
        "duracao_segundos": None,
        "inclui_teste_final": False,
        "experimento_ffill": str(raiz),
        "entradas": {
            nome: {
                "caminho": str(caminho),
                "tamanho_bytes": caminho.stat().st_size,
                "sha256": sha256_arquivo(caminho),
            }
            for nome, caminho in entradas.items()
        },
        "etapas_concluidas": [],
        "artefatos": {},
    }
    caminho_manifesto = saida / "manifesto.json"
    gravar_manifesto(caminho_manifesto, manifesto)

    print(f"Experimento do Efeito Dominó: {saida}", flush=True)
    print("As métricas de avaliação do Granger não serão usadas como filtro.")
    print("O teste final não será construído nem consultado.", flush=True)
    painel = PainelPipeline("Efeito Dominó assimétrico", len(etapas))
    with painel:
        for indice, etapa in enumerate(etapas, start=1):
            inicio_etapa = perf_counter()
            painel.iniciar_etapa(indice, etapa.nome, etapa.descricao)
            codigo_saida = executar_comando(etapa.comando, painel)
            if codigo_saida != 0:
                manifesto["status"] = "falha"
                manifesto["concluida_em"] = agora_iso()
                manifesto["duracao_segundos"] = round(
                    perf_counter() - inicio,
                    3,
                )
                manifesto["falha"] = {
                    "etapa": etapa.nome,
                    "codigo_saida": codigo_saida,
                }
                gravar_manifesto(caminho_manifesto, manifesto)
                painel.falhar_etapa(etapa.nome, codigo_saida)
                return codigo_saida
            concluidas = manifesto["etapas_concluidas"]
            assert isinstance(concluidas, list)
            concluidas.append(
                {
                    "nome": etapa.nome,
                    "duracao_segundos": round(
                        perf_counter() - inicio_etapa,
                        3,
                    ),
                }
            )
            gravar_manifesto(caminho_manifesto, manifesto)
            painel.concluir_etapa(indice, etapa.nome)

        painel.iniciar_finalizacao()
        artefatos = {
            "validacao_1_bivariado": saida / "validacao_1/bivariado.sqlite",
            "validacao_1_assimetrico": saida / "validacao_1/assimetrico.sqlite",
            "validacao_2_bivariado": saida / "validacao_2/bivariado.sqlite",
            "validacao_2_assimetrico": saida / "validacao_2/assimetrico.sqlite",
            "comparacao": saida / "comparacao.json",
        }
        manifesto["artefatos"] = {
            nome: {
                "caminho": str(caminho.relative_to(saida)),
                "tamanho_bytes": caminho.stat().st_size,
                "sha256": sha256_arquivo(caminho),
            }
            for nome, caminho in artefatos.items()
        }
        manifesto["status"] = "concluida"
        manifesto["concluida_em"] = agora_iso()
        manifesto["duracao_segundos"] = round(perf_counter() - inicio, 3)
        gravar_manifesto(caminho_manifesto, manifesto)
        painel.finalizar()

    print(f"\nResultados preservados em: {saida}", flush=True)
    print(f"Comparação: {saida / 'comparacao.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
