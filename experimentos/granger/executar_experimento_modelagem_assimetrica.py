#!/usr/bin/env python3
"""Orquestra a comparação de modelos bivariados e assimétricos."""

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
    domino = raiz / "granger-assimetrico/efeito-domino"
    return {
        "manifesto_ffill": raiz / "manifesto.json",
        "manifesto_domino": domino / "manifesto.json",
        "comparacao_domino": domino / "comparacao.json",
        "validacao_1_granger": raiz / "validacao_1/granger/ffill-3.sqlite",
        "validacao_2_granger": raiz / "validacao_2/granger/ffill-3.sqlite",
        "validacao_1_domino_bivariado": (
            domino / "validacao_1/bivariado.sqlite"
        ),
        "validacao_1_domino_assimetrico": (
            domino / "validacao_1/assimetrico.sqlite"
        ),
        "validacao_2_domino_bivariado": (
            domino / "validacao_2/bivariado.sqlite"
        ),
        "validacao_2_domino_assimetrico": (
            domino / "validacao_2/assimetrico.sqlite"
        ),
    }


def validar_manifesto(
    caminho: Path,
    tipo: str,
) -> None:
    manifesto = json.loads(caminho.read_text(encoding="utf-8"))
    if manifesto.get("tipo") != tipo:
        raise ValueError(f"tipo de manifesto inesperado em: {caminho}")
    if manifesto.get("status") != "concluida":
        raise ValueError(f"execução não concluída: {caminho}")
    if manifesto.get("inclui_teste_final") is not False:
        raise ValueError(f"o manifesto não protege o teste final: {caminho}")


def validar_entradas(raiz: Path) -> dict[str, Path]:
    entradas = caminhos_entrada(raiz)
    ausentes = [caminho for caminho in entradas.values() if not caminho.is_file()]
    if ausentes:
        nomes = ", ".join(str(caminho) for caminho in ausentes)
        raise FileNotFoundError(f"artefatos obrigatórios ausentes: {nomes}")
    validar_manifesto(
        entradas["manifesto_ffill"],
        "sensibilidade_ffill_granger",
    )
    validar_manifesto(
        entradas["manifesto_domino"],
        "comparacao_domino_assimetrico",
    )
    comparacao = json.loads(
        entradas["comparacao_domino"].read_text(encoding="utf-8")
    )
    if comparacao.get("assimetrico_apto_nas_duas_validacoes") is not True:
        raise ValueError(
            "o Efeito Dominó assimétrico não preservou as duas classes"
        )
    if comparacao.get("inclui_teste_final") is not False:
        raise ValueError("a comparação do Efeito Dominó consultou o teste final")
    return entradas


def construir_etapas(
    entradas: dict[str, Path],
    saida: Path,
) -> list[Etapa]:
    manifesto_ffill = entradas["manifesto_ffill"]
    bases: dict[str, dict[str, Path]] = {
        "bivariado": {},
        "assimetrico": {},
    }
    etapas: list[Etapa] = []
    for contexto in ("validacao_1", "validacao_2"):
        for variante in ("bivariado", "assimetrico"):
            base = saida / variante / f"{contexto}.sqlite"
            bases[variante][contexto] = base
            etapas.append(
                Etapa(
                    nome=f"Base {variante} — {contexto}",
                    descricao=(
                        "constrói atributos e partições com a mesma metodologia"
                    ),
                    comando=[
                        sys.executable,
                        "-m",
                        (
                            "experimentos.granger."
                            "gerar_base_modelagem_experimental"
                        ),
                        str(manifesto_ffill),
                        str(entradas[f"{contexto}_granger"]),
                        str(entradas[f"{contexto}_domino_{variante}"]),
                        "--contexto",
                        contexto,
                        "--output-db",
                        str(base),
                    ],
                )
            )

    validacoes: dict[str, dict[str, Path]] = {}
    for variante in ("bivariado", "assimetrico"):
        diretorio = saida / variante
        banco = diretorio / "validacao.sqlite"
        selecao = diretorio / "configuracao_selecionada.json"
        validacoes[variante] = {"banco": banco, "selecao": selecao}
        etapas.append(
            Etapa(
                nome=f"Treinamento e validação — {variante}",
                descricao="avalia o mesmo catálogo de modelos e limiares",
                comando=[
                    sys.executable,
                    "-m",
                    "treinamento.executar_validacao",
                    str(bases[variante]["validacao_1"]),
                    str(bases[variante]["validacao_2"]),
                    "--output-db",
                    str(banco),
                    "--selection-json",
                    str(selecao),
                ],
            )
        )

    etapas.append(
        Etapa(
            nome="Comparação da modelagem",
            descricao="compara F1, PR-AUC e MCC nas duas validações",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.comparar_modelagem_assimetrica",
                "--bivariado-db",
                str(validacoes["bivariado"]["banco"]),
                "--bivariado-json",
                str(validacoes["bivariado"]["selecao"]),
                "--assimetrico-db",
                str(validacoes["assimetrico"]["banco"]),
                "--assimetrico-json",
                str(validacoes["assimetrico"]["selecao"]),
                "--comparacao-domino",
                str(entradas["comparacao_domino"]),
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
    saida = raiz / "granger-assimetrico/modelagem"
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
        "tipo": "comparacao_modelagem_assimetrica",
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

    print(f"Experimento de modelagem: {saida}", flush=True)
    print("As duas alternativas usarão o mesmo catálogo de modelos e atributos.")
    print("O teste final não será construído nem consultado.", flush=True)
    painel = PainelPipeline("Modelagem assimétrica", len(etapas))
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
            "bivariado_validacao_1": saida / "bivariado/validacao_1.sqlite",
            "bivariado_validacao_2": saida / "bivariado/validacao_2.sqlite",
            "bivariado_resultados": saida / "bivariado/validacao.sqlite",
            "bivariado_selecao": (
                saida / "bivariado/configuracao_selecionada.json"
            ),
            "assimetrico_validacao_1": (
                saida / "assimetrico/validacao_1.sqlite"
            ),
            "assimetrico_validacao_2": (
                saida / "assimetrico/validacao_2.sqlite"
            ),
            "assimetrico_resultados": saida / "assimetrico/validacao.sqlite",
            "assimetrico_selecao": (
                saida / "assimetrico/configuracao_selecionada.json"
            ),
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
