#!/usr/bin/env python3
"""Orquestra a inferência robusta do Granger nas duas validações."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from pipeline.progresso_terminal import PainelPipeline, executar_comando


@dataclass(frozen=True)
class Etapa:
    nome: str
    descricao: str
    comando: list[str]


def agora_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_arquivo(caminho: Path, bloco: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while dados := arquivo.read(bloco):
            digest.update(dados)
    return digest.hexdigest()


def gravar_json(caminho: Path, valor: dict[str, Any]) -> None:
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(valor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def validar_entrada(raiz: Path) -> dict[str, Path]:
    caminhos = {
        "manifesto": raiz / "manifesto.json",
        "validacao_1": raiz / "validacao_1/granger/ffill-3.sqlite",
        "validacao_2": raiz / "validacao_2/granger/ffill-3.sqlite",
    }
    ausentes = [caminho for caminho in caminhos.values() if not caminho.is_file()]
    if ausentes:
        nomes = ", ".join(str(caminho) for caminho in ausentes)
        raise FileNotFoundError(f"artefatos obrigatórios ausentes: {nomes}")
    manifesto = json.loads(caminhos["manifesto"].read_text(encoding="utf-8"))
    if manifesto.get("status") != "concluida":
        raise ValueError("o experimento de preenchimento não está concluído")
    if manifesto.get("inclui_teste_final") is not False:
        raise ValueError("a entrada não protege o teste final")
    return caminhos


def construir_etapas(caminhos: dict[str, Path], saida: Path) -> list[Etapa]:
    v1 = saida / "validacao_1.sqlite"
    v2 = saida / "validacao_2.sqlite"
    return [
        Etapa(
            nome="Inferência robusta — validação 1",
            descricao="recalcula os p-valores por HC3 e HAC em 2022–2023",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.executar_inferencia_robusta",
                str(caminhos["validacao_1"]),
                "--output-db",
                str(v1),
            ],
        ),
        Etapa(
            nome="Inferência robusta — validação 2",
            descricao="repete o mesmo protocolo em 2022–2024",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.executar_inferencia_robusta",
                str(caminhos["validacao_2"]),
                "--output-db",
                str(v2),
            ],
        ),
        Etapa(
            nome="Comparação da inferência robusta",
            descricao="mede estabilidade e ganho preditivo nas duas validações",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.comparar_inferencia_robusta",
                "--validacao-1",
                str(v1),
                "--validacao-2",
                str(v2),
                "--output-json",
                str(saida / "comparacao.json"),
            ],
        ),
    ]


def executar(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experimento_ffill", type=Path)
    args = parser.parse_args(argv)
    raiz = args.experimento_ffill
    saida = raiz / "granger-inferencia-robusta"
    try:
        caminhos = validar_entrada(raiz)
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}", flush=True)
        return 2
    if saida.exists():
        print(f"ERRO: a saída já existe: {saida}", flush=True)
        return 2

    saida.mkdir(parents=True)
    etapas = construir_etapas(caminhos, saida)
    inicio = perf_counter()
    manifesto: dict[str, Any] = {
        "schema_version": 1,
        "tipo": "granger_inferencia_robusta",
        "status": "em_andamento",
        "iniciada_em": agora_iso(),
        "concluida_em": None,
        "duracao_segundos": None,
        "inclui_teste_final": False,
        "experimento_ffill": str(raiz),
        "metodo_primario": "HAC Newey-West segmentado",
        "metodo_sensibilidade": "HC3",
        "entradas": {
            contexto: {
                "caminho": str(caminhos[contexto]),
                "tamanho_bytes": caminhos[contexto].stat().st_size,
                "sha256": sha256_arquivo(caminhos[contexto]),
            }
            for contexto in ("validacao_1", "validacao_2")
        },
        "etapas_concluidas": [],
        "artefatos": {},
    }
    caminho_manifesto = saida / "manifesto.json"
    gravar_json(caminho_manifesto, manifesto)

    print(f"Experimento de inferência robusta: {saida}", flush=True)
    print("HAC será o método principal e HC3 a análise de sensibilidade.")
    print("O teste final não será construído nem consultado.", flush=True)
    painel = PainelPipeline("Inferência robusta do Granger", len(etapas))
    with painel:
        for indice, etapa in enumerate(etapas, start=1):
            inicio_etapa = perf_counter()
            painel.iniciar_etapa(indice, etapa.nome, etapa.descricao)
            codigo = executar_comando(etapa.comando, painel)
            if codigo != 0:
                manifesto["status"] = "falha"
                manifesto["concluida_em"] = agora_iso()
                manifesto["duracao_segundos"] = round(
                    perf_counter() - inicio,
                    3,
                )
                manifesto["falha"] = {
                    "etapa": etapa.nome,
                    "codigo_saida": codigo,
                }
                gravar_json(caminho_manifesto, manifesto)
                painel.falhar_etapa(etapa.nome, codigo)
                return codigo
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
            gravar_json(caminho_manifesto, manifesto)
            painel.concluir_etapa(indice, etapa.nome)

        painel.iniciar_finalizacao()
        artefatos = {
            "validacao_1": saida / "validacao_1.sqlite",
            "validacao_2": saida / "validacao_2.sqlite",
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
        gravar_json(caminho_manifesto, manifesto)
        painel.finalizar()

    print(f"\nResultados preservados em: {saida}")
    print(f"Comparação: {saida / 'comparacao.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
