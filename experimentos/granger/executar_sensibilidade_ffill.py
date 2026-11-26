"""Executa o experimento encerrado de sensibilidade ao preenchimento."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from pipeline.progresso_terminal import PainelPipeline, executar_comando


RAIZ_RESULTADOS = Path("resultados/experimentos/sensibilidade-ffill")
ENTRADA_BRUTA = Path("data/cotacoes.sqlite")
LIMITES_FFILL = (0, 1, 3)
FFILL_MANTIDO = 3
FUSO_HORARIO = ZoneInfo("America/Sao_Paulo")


@dataclass(frozen=True)
class Contexto:
    nome: str
    inicio_series: str
    corte_treino: str
    fim_avaliacao: str


CONTEXTOS = (
    Contexto("validacao_1", "2022-01-04", "2023-12-31", "2024-12-31"),
    Contexto("validacao_2", "2022-01-04", "2024-12-31", "2025-10-31"),
)


@dataclass(frozen=True)
class Etapa:
    nome: str
    descricao: str
    comando: list[str]


def agora_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_arquivo(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as arquivo:
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


def gerar_diretorio_execucao() -> Path:
    base = datetime.now(FUSO_HORARIO).strftime("execucao-%Y%m%d-%H%M%S")
    caminho = RAIZ_RESULTADOS / base
    sufixo = 2
    while caminho.exists():
        caminho = RAIZ_RESULTADOS / f"{base}-{sufixo:02d}"
        sufixo += 1
    return caminho


def caminho_preprocessamento(raiz: Path, contexto: Contexto) -> Path:
    return raiz / contexto.nome / "preprocessamento/cotacoes_preprocessadas.sqlite"


def caminho_granger(
    raiz: Path,
    contexto: Contexto,
    ffill_limite: int,
) -> Path:
    return raiz / contexto.nome / "granger" / f"ffill-{ffill_limite}.sqlite"


def construir_etapas_contexto(
    raiz: Path,
    entrada: Path,
    contexto: Contexto,
) -> list[Etapa]:
    preprocessado = caminho_preprocessamento(raiz, contexto)
    etapas = [
        Etapa(
            nome=f"Pré-processamento — {contexto.nome}",
            descricao="gera uma única entrada comum para os três limites",
            comando=[
                sys.executable,
                "preprocessamento/tratar_ceasa.py",
                str(entrada),
                "--output-db",
                str(preprocessado),
                "--data-corte-treino",
                contexto.corte_treino,
            ],
        )
    ]
    for limite in LIMITES_FFILL:
        etapas.append(
            Etapa(
                nome=f"Granger — {contexto.nome} — ffill={limite}",
                descricao="altera somente o limite de preenchimento",
                comando=[
                    sys.executable,
                    "granger/executar_granger.py",
                    str(preprocessado),
                    "--output-db",
                    str(caminho_granger(raiz, contexto, limite)),
                    "--data-inicio",
                    contexto.inicio_series,
                    "--data-corte-treino",
                    contexto.corte_treino,
                    "--data-fim-teste",
                    contexto.fim_avaliacao,
                    "--ffill-limite",
                    str(limite),
                ],
            )
        )
    return etapas


def construir_etapa_comparacao(raiz: Path) -> Etapa:
    return Etapa(
        nome="Comparação dos limites de preenchimento",
        descricao="valida a comparabilidade e resume as duas validações",
        comando=[
            sys.executable,
            "-m",
            "experimentos.granger.comparar_sensibilidade_ffill",
            "--validacao-1",
            *[
                str(caminho_granger(raiz, CONTEXTOS[0], limite))
                for limite in LIMITES_FFILL
            ],
            "--validacao-2",
            *[
                str(caminho_granger(raiz, CONTEXTOS[1], limite))
                for limite in LIMITES_FFILL
            ],
            "--output-json",
            str(raiz / "comparacao.json"),
        ],
    )


def registrar_artefato(raiz: Path, caminho: Path) -> dict[str, object]:
    return {
        "caminho": str(caminho.relative_to(raiz)),
        "tamanho_bytes": caminho.stat().st_size,
        "sha256": sha256_arquivo(caminho),
    }


def descartar_preprocessamento(
    raiz: Path,
    contexto: Contexto,
) -> dict[str, object]:
    caminho = caminho_preprocessamento(raiz, contexto)
    registro = registrar_artefato(raiz, caminho)
    caminho.unlink()
    registro["descartado_apos_granger"] = True
    return registro


def descartar_grangers_alternativos(raiz: Path) -> dict[str, dict[str, object]]:
    descartados = {}
    for contexto in CONTEXTOS:
        for limite in LIMITES_FFILL:
            if limite == FFILL_MANTIDO:
                continue

            caminho = caminho_granger(raiz, contexto, limite)
            registro = registrar_artefato(raiz, caminho)
            caminho.unlink()
            registro["descartado_apos_comparacao"] = True
            descartados[f"{contexto.nome}_ffill_{limite}"] = registro

    return descartados


def registrar_falha(
    manifesto: dict[str, object],
    caminho_manifesto: Path,
    inicio_total: float,
    etapa: Etapa,
    codigo_saida: int,
) -> None:
    manifesto["status"] = "falha"
    manifesto["concluida_em"] = agora_iso()
    manifesto["duracao_segundos"] = round(perf_counter() - inicio_total, 3)
    manifesto["falha"] = {
        "etapa": etapa.nome,
        "codigo_saida": codigo_saida,
        "artefatos_parciais_preservados": True,
    }
    gravar_manifesto(caminho_manifesto, manifesto)


def executar_etapa(
    etapa: Etapa,
    indice: int,
    painel: PainelPipeline,
    manifesto: dict[str, object],
    caminho_manifesto: Path,
) -> int:
    inicio = perf_counter()
    painel.iniciar_etapa(indice, etapa.nome, etapa.descricao)
    codigo_saida = executar_comando(etapa.comando, painel)
    if codigo_saida != 0:
        painel.falhar_etapa(etapa.nome, codigo_saida)
        return codigo_saida

    etapas_concluidas = manifesto["etapas_concluidas"]
    assert isinstance(etapas_concluidas, list)
    etapas_concluidas.append(
        {
            "nome": etapa.nome,
            "duracao_segundos": round(perf_counter() - inicio, 3),
        }
    )
    gravar_manifesto(caminho_manifesto, manifesto)
    painel.concluir_etapa(indice, etapa.nome)
    return 0


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ENTRADA_BRUTA)
    parser.add_argument("--output-dir", type=Path)
    return parser


def executar(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    entrada = args.database
    raiz = args.output_dir or gerar_diretorio_execucao()
    if not entrada.is_file():
        print(f"ERRO: banco bruto não encontrado: {entrada}", flush=True)
        return 2
    if raiz.exists():
        print(f"ERRO: diretório de saída já existe: {raiz}", flush=True)
        return 2

    raiz.mkdir(parents=True)
    etapas_contextos = [
        (contexto, construir_etapas_contexto(raiz, entrada, contexto))
        for contexto in CONTEXTOS
    ]
    etapa_comparacao = construir_etapa_comparacao(raiz)
    total_etapas = sum(len(etapas) for _, etapas in etapas_contextos) + 1
    inicio_total = perf_counter()
    manifesto: dict[str, object] = {
        "schema_version": 1,
        "tipo": "sensibilidade_ffill_granger",
        "status": "em_andamento",
        "iniciada_em": agora_iso(),
        "concluida_em": None,
        "duracao_segundos": None,
        "inclui_teste_final": False,
        "limites_ffill": list(LIMITES_FFILL),
        "contextos": [asdict(contexto) for contexto in CONTEXTOS],
        "entrada": {
            "caminho": str(entrada),
            "tamanho_bytes": entrada.stat().st_size,
            "sha256": sha256_arquivo(entrada),
        },
        "etapas_concluidas": [],
        "preprocessamentos_descartados": {},
        "grangers_descartados": {},
        "artefatos": {},
    }
    caminho_manifesto = raiz / "manifesto.json"
    gravar_manifesto(caminho_manifesto, manifesto)

    print(f"Experimento de sensibilidade: {raiz.name}", flush=True)
    print("Limites: ffill 0, 1 e 3 nas duas validações.", flush=True)
    print("O teste final não será construído nem consultado.", flush=True)

    indice = 0
    painel = PainelPipeline("Sensibilidade do Granger ao ffill", total_etapas)
    with painel:
        for contexto, etapas in etapas_contextos:
            for etapa in etapas:
                indice += 1
                codigo_saida = executar_etapa(
                    etapa,
                    indice,
                    painel,
                    manifesto,
                    caminho_manifesto,
                )
                if codigo_saida != 0:
                    registrar_falha(
                        manifesto,
                        caminho_manifesto,
                        inicio_total,
                        etapa,
                        codigo_saida,
                    )
                    painel.exibir_log(
                        "Artefatos parciais preservados para diagnóstico."
                    )
                    return codigo_saida

            descartados = manifesto["preprocessamentos_descartados"]
            assert isinstance(descartados, dict)
            descartados[contexto.nome] = descartar_preprocessamento(
                raiz,
                contexto,
            )
            gravar_manifesto(caminho_manifesto, manifesto)
            painel.exibir_log(
                f"Pré-processado temporário descartado: {contexto.nome}"
            )

        indice += 1
        codigo_saida = executar_etapa(
            etapa_comparacao,
            indice,
            painel,
            manifesto,
            caminho_manifesto,
        )
        if codigo_saida != 0:
            registrar_falha(
                manifesto,
                caminho_manifesto,
                inicio_total,
                etapa_comparacao,
                codigo_saida,
            )
            return codigo_saida

        painel.iniciar_finalizacao()
        artefatos = {
            f"{contexto.nome}_ffill_{limite}": registrar_artefato(
                raiz,
                caminho_granger(raiz, contexto, limite),
            )
            for contexto in CONTEXTOS
            for limite in LIMITES_FFILL
            if limite == FFILL_MANTIDO
        }
        artefatos["comparacao"] = registrar_artefato(
            raiz,
            raiz / "comparacao.json",
        )
        manifesto["artefatos"] = artefatos
        manifesto["grangers_descartados"] = descartar_grangers_alternativos(raiz)
        manifesto["status"] = "concluida"
        manifesto["concluida_em"] = agora_iso()
        manifesto["duracao_segundos"] = round(perf_counter() - inicio_total, 3)
        gravar_manifesto(caminho_manifesto, manifesto)
        painel.finalizar()

    print(f"\nResultados preservados em: {raiz}", flush=True)
    print(f"Bancos mantidos: ffill={FFILL_MANTIDO}", flush=True)
    print(f"Comparação: {raiz / 'comparacao.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
