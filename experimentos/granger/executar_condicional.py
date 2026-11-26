"""Executa o experimento encerrado de Granger condicional."""

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


def construir_etapas(raiz_ffill: Path, saida: Path) -> list[Etapa]:
    v1_origem = raiz_ffill / "validacao_1/granger/ffill-3.sqlite"
    v2_origem = raiz_ffill / "validacao_2/granger/ffill-3.sqlite"
    configuracao = saida / "configuracao_redes.json"
    v1_saida = saida / "validacao_1.sqlite"
    v2_saida = saida / "validacao_2.sqlite"
    comparacao = saida / "comparacao.json"
    return [
        Etapa(
            nome="Granger condicional — validação 1",
            descricao="seleciona redes somente pela cobertura de 2022–2023",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.executar_modelo_condicional",
                str(v1_origem),
                "--output-db",
                str(v1_saida),
                "--output-config-redes",
                str(configuracao),
            ],
        ),
        Etapa(
            nome="Granger condicional — validação 2",
            descricao="reutiliza exatamente as redes congeladas na validação 1",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.executar_modelo_condicional",
                str(v2_origem),
                "--output-db",
                str(v2_saida),
                "--config-redes",
                str(configuracao),
            ],
        ),
        Etapa(
            nome="Comparação do Granger condicional",
            descricao="mede retenção bivariada e estabilidade entre validações",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.comparar_granger_condicional",
                "--validacao-1",
                str(v1_saida),
                "--validacao-2",
                str(v2_saida),
                "--output-json",
                str(comparacao),
            ],
        ),
    ]


def validar_entrada(raiz: Path) -> list[Path]:
    caminhos = [
        raiz / "manifesto.json",
        raiz / "validacao_1/granger/ffill-3.sqlite",
        raiz / "validacao_2/granger/ffill-3.sqlite",
    ]
    ausentes = [caminho for caminho in caminhos if not caminho.is_file()]
    if ausentes:
        nomes = ", ".join(str(caminho) for caminho in ausentes)
        raise FileNotFoundError(f"artefatos obrigatórios ausentes: {nomes}")
    manifesto = json.loads(caminhos[0].read_text(encoding="utf-8"))
    if manifesto.get("status") != "concluida":
        raise ValueError("o experimento de ffill não está concluído")
    if manifesto.get("inclui_teste_final") is not False:
        raise ValueError("a entrada não declara a proteção do teste final")
    return caminhos[1:]


def executar(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experimento_ffill", type=Path)
    args = parser.parse_args(argv)
    raiz = args.experimento_ffill
    saida = raiz / "granger-condicional"
    try:
        entradas = validar_entrada(raiz)
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}", flush=True)
        return 2
    if saida.exists():
        print(f"ERRO: a saída já existe: {saida}", flush=True)
        return 2

    saida.mkdir(parents=True)
    etapas = construir_etapas(raiz, saida)
    inicio = perf_counter()
    manifesto: dict[str, object] = {
        "schema_version": 1,
        "tipo": "granger_condicional",
        "status": "em_andamento",
        "iniciada_em": agora_iso(),
        "concluida_em": None,
        "duracao_segundos": None,
        "inclui_teste_final": False,
        "experimento_ffill": str(raiz),
        "entradas": {
            contexto: {
                "caminho": str(caminho),
                "tamanho_bytes": caminho.stat().st_size,
                "sha256": sha256_arquivo(caminho),
            }
            for contexto, caminho in zip(
                ("validacao_1", "validacao_2"),
                entradas,
                strict=True,
            )
        },
        "etapas_concluidas": [],
        "artefatos": {},
    }
    caminho_manifesto = saida / "manifesto.json"
    gravar_manifesto(caminho_manifesto, manifesto)

    print(f"Experimento condicional: {saida}", flush=True)
    print("As redes serão selecionadas somente pela primeira janela de treino.")
    print("O teste final não será construído nem consultado.", flush=True)
    painel = PainelPipeline("Granger condicional", len(etapas))
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
            nome: saida / arquivo
            for nome, arquivo in {
                "configuracao_redes": "configuracao_redes.json",
                "validacao_1": "validacao_1.sqlite",
                "validacao_2": "validacao_2.sqlite",
                "comparacao": "comparacao.json",
            }.items()
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
