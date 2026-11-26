"""Executa e preserva o pipeline das duas validações de desenvolvimento."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from progresso_terminal import PainelPipeline, executar_comando


RAIZ_RESULTADOS = Path("resultados/versoes")
ENTRADA_BRUTA = Path("data/cotacoes.sqlite")
PADRAO_VERSAO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
FUSO_HORARIO_VERSAO = ZoneInfo("America/Sao_Paulo")

PARAMETROS_PIPELINE = (
    "DOMINO_ALTA_MINIMA",
    "DOMINO_PERCENTIL",
    "DOMINO_JANELA_DIAS",
    "TREINAMENTO_SEMENTE",
)

GRANGER_METODO_SELECIONADO = {
    "metodo": "bivariado_classico",
    "min_dias_comuns_observados": 400,
    "ffill_limite": 3,
    "max_lag": 10,
    "max_lag_amostra": 10,
    "selecao_lag": "BIC em amostra fixa",
    "alpha": 0.05,
    "correcao_multiplos_testes": "Benjamini-Hochberg global",
    "criterio_relacoes": "significativo_fdr",
}

DIRETORIOS_CODIGO = (
    Path("preprocessamento"),
    Path("granger"),
    Path("efeitos"),
    Path("modelagem"),
    Path("treinamento"),
    Path("pipeline"),
)

ARQUIVOS_CONFIGURACAO = (
    Path("compose.yaml"),
    Path("Dockerfile"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


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


def argumentos_granger_selecionado() -> list[str]:
    return [
        "--min-dias-comuns",
        str(GRANGER_METODO_SELECIONADO["min_dias_comuns_observados"]),
        "--ffill-limite",
        str(GRANGER_METODO_SELECIONADO["ffill_limite"]),
        "--max-lag",
        str(GRANGER_METODO_SELECIONADO["max_lag"]),
        "--max-lag-amostra",
        str(GRANGER_METODO_SELECIONADO["max_lag_amostra"]),
        "--alpha",
        str(GRANGER_METODO_SELECIONADO["alpha"]),
    ]


def agora_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_arquivo(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def executar_git(*args: str) -> str | None:
    try:
        resultado = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if resultado.returncode != 0:
        return None
    return resultado.stdout.strip()


def arquivos_codigo() -> list[Path]:
    arquivos = {path for path in ARQUIVOS_CONFIGURACAO if path.is_file()}
    for diretorio in DIRETORIOS_CODIGO:
        if diretorio.is_dir():
            arquivos.update(diretorio.rglob("*.py"))
    return sorted(arquivos)


def registrar_codigo() -> dict[str, object]:
    git_disponivel = shutil.which("git") is not None
    status = executar_git("status", "--short") if git_disponivel else None
    return {
        "git_disponivel": git_disponivel,
        "git_commit": executar_git("rev-parse", "HEAD") if git_disponivel else None,
        "git_status": status.splitlines() if status else [],
        "arquivos_sha256": {
            str(path): sha256_arquivo(path) for path in arquivos_codigo()
        },
    }


def gravar_manifesto(path: Path, manifesto: dict[str, object]) -> None:
    temporario = path.with_suffix(".tmp")
    temporario.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(path)


def artefatos_contexto(raiz: Path, contexto: Contexto) -> dict[str, Path]:
    base = raiz / "contextos" / contexto.nome
    return {
        f"{contexto.nome}_preprocessamento": (
            base / "preprocessamento/cotacoes_preprocessadas.sqlite"
        ),
        f"{contexto.nome}_granger": base / "granger/resultados_granger.sqlite",
        f"{contexto.nome}_efeito_domino": (
            base / "efeitos/domino/resultados_efeito_domino.sqlite"
        ),
        f"{contexto.nome}_base_modelagem": (
            base / "modelagem/base_modelagem.sqlite"
        ),
    }


def construir_etapas(raiz: Path) -> tuple[list[Etapa], dict[str, Path]]:
    artefatos: dict[str, Path] = {}
    etapas: list[Etapa] = []

    for contexto in CONTEXTOS:
        atuais = artefatos_contexto(raiz, contexto)
        artefatos.update(atuais)
        pre = atuais[f"{contexto.nome}_preprocessamento"]
        granger = atuais[f"{contexto.nome}_granger"]
        domino = atuais[f"{contexto.nome}_efeito_domino"]
        base_modelagem = atuais[f"{contexto.nome}_base_modelagem"]

        etapas.extend(
            [
                Etapa(
                    nome=f"Pré-processamento — {contexto.nome}",
                    descricao="referências aprendidas somente no treino do contexto",
                    comando=[
                        sys.executable,
                        "preprocessamento/tratar_ceasa.py",
                        str(ENTRADA_BRUTA),
                        "--output-db",
                        str(pre),
                        "--data-corte-treino",
                        contexto.corte_treino,
                    ],
                ),
                Etapa(
                    nome=f"Granger — {contexto.nome}",
                    descricao="relações selecionadas apenas com o treino do contexto",
                    comando=[
                        sys.executable,
                        "granger/executar_granger.py",
                        str(pre),
                        "--output-db",
                        str(granger),
                        "--data-inicio",
                        contexto.inicio_series,
                        "--data-corte-treino",
                        contexto.corte_treino,
                        "--data-fim-teste",
                        contexto.fim_avaliacao,
                        *argumentos_granger_selecionado(),
                    ],
                ),
                Etapa(
                    nome=f"Efeito Dominó — {contexto.nome}",
                    descricao="limites aprendidos apenas com o treino do contexto",
                    comando=[
                        sys.executable,
                        "efeitos/domino/executar_efeito_domino.py",
                        str(granger),
                        "--output-db",
                        str(domino),
                    ],
                ),
                Etapa(
                    nome=f"Base de modelagem — {contexto.nome}",
                    descricao="atributos e partição exclusivos do contexto",
                    comando=[
                        sys.executable,
                        "modelagem/gerar_base_modelagem.py",
                        str(pre),
                        str(granger),
                        str(domino),
                        "--contexto",
                        contexto.nome,
                        "--output-db",
                        str(base_modelagem),
                    ],
                ),
            ]
        )

    base_1 = artefatos["validacao_1_base_modelagem"]
    base_2 = artefatos["validacao_2_base_modelagem"]
    artefatos["validacao"] = raiz / "treinamento/validacao.sqlite"
    artefatos["configuracao_selecionada"] = (
        raiz / "treinamento/configuracao_selecionada.json"
    )
    etapas.append(
        Etapa(
            nome="Validação e seleção dos modelos",
            descricao="compara os modelos sem construir nem consultar o teste final",
            comando=[
                sys.executable,
                "-m",
                "treinamento.executar_validacao",
                str(base_1),
                str(base_2),
                "--output-db",
                str(artefatos["validacao"]),
                "--selection-json",
                str(artefatos["configuracao_selecionada"]),
            ],
        )
    )
    return etapas, artefatos


def gerar_versao_automatica() -> str:
    base = datetime.now(FUSO_HORARIO_VERSAO).strftime("execucao-%Y%m%d-%H%M%S")
    versao = base
    numero = 2
    while (RAIZ_RESULTADOS / versao).exists():
        versao = f"{base}-{numero:02d}"
        numero += 1
    return versao


def obter_versao() -> tuple[str, bool]:
    versao = os.getenv("PIPELINE_VERSAO", "").strip()
    if not versao:
        return gerar_versao_automatica(), True
    if PADRAO_VERSAO.fullmatch(versao) is None:
        raise ValueError(
            "PIPELINE_VERSAO deve ter de 1 a 80 caracteres e usar apenas "
            "letras, números, ponto, hífen ou sublinhado"
        )
    return versao, False


def metadados_artefatos(
    raiz: Path,
    artefatos: dict[str, Path],
) -> dict[str, dict[str, object]]:
    return {
        nome: {
            "caminho": str(path.relative_to(raiz)),
            "tamanho_bytes": path.stat().st_size,
            "sha256": sha256_arquivo(path),
        }
        for nome, path in artefatos.items()
    }


def executar() -> int:
    try:
        versao, versao_automatica = obter_versao()
    except ValueError as erro:
        print(f"ERRO: {erro}", flush=True)
        return 2

    if not ENTRADA_BRUTA.is_file():
        print(f"ERRO: banco bruto não encontrado: {ENTRADA_BRUTA}", flush=True)
        return 2

    raiz = RAIZ_RESULTADOS / versao
    if raiz.exists():
        print(f"ERRO: a versão {versao!r} já existe em {raiz}.", flush=True)
        return 2

    raiz.mkdir(parents=True)
    etapas, artefatos = construir_etapas(raiz)
    inicio_total = perf_counter()
    manifesto: dict[str, object] = {
        "schema_version": 2,
        "tipo": "desenvolvimento",
        "versao": versao,
        "versao_automatica": versao_automatica,
        "status": "em_andamento",
        "iniciada_em": agora_iso(),
        "concluida_em": None,
        "duracao_segundos": None,
        "inclui_teste_final": False,
        "contextos": [asdict(contexto) for contexto in CONTEXTOS],
        "parametros": {nome: os.getenv(nome) for nome in PARAMETROS_PIPELINE},
        "metodologia_granger": GRANGER_METODO_SELECIONADO,
        "codigo": registrar_codigo(),
        "entrada": {
            "caminho": str(ENTRADA_BRUTA),
            "tamanho_bytes": ENTRADA_BRUTA.stat().st_size,
            "sha256": sha256_arquivo(ENTRADA_BRUTA),
        },
        "etapas_concluidas": [],
        "artefatos": {},
    }
    caminho_manifesto = raiz / "manifesto.json"
    gravar_manifesto(caminho_manifesto, manifesto)

    print(f"Pipeline de desenvolvimento versionado: {versao}", flush=True)
    print("Contextos: validação 1 -> validação 2 -> seleção", flush=True)
    print("O teste final não será construído nem consultado.", flush=True)

    painel = PainelPipeline(
        f"Pipeline de desenvolvimento — {versao}",
        len(etapas),
    )
    with painel:
        for indice, etapa in enumerate(etapas, start=1):
            inicio_etapa = perf_counter()
            painel.iniciar_etapa(indice, etapa.nome, etapa.descricao)
            codigo_saida = executar_comando(etapa.comando, painel)
            if codigo_saida != 0:
                manifesto["status"] = "falha"
                manifesto["concluida_em"] = agora_iso()
                manifesto["duracao_segundos"] = round(
                    perf_counter() - inicio_total,
                    3,
                )
                manifesto["falha"] = {
                    "etapa": etapa.nome,
                    "codigo_saida": codigo_saida,
                }
                gravar_manifesto(caminho_manifesto, manifesto)
                painel.falhar_etapa(etapa.nome, codigo_saida)
                painel.exibir_log()
                painel.exibir_log(
                    f"FALHA em {etapa.nome}. Artefatos parciais preservados."
                )
                return codigo_saida

            concluidas = manifesto["etapas_concluidas"]
            assert isinstance(concluidas, list)
            concluidas.append(
                {
                    "nome": etapa.nome,
                    "duracao_segundos": round(perf_counter() - inicio_etapa, 3),
                }
            )
            gravar_manifesto(caminho_manifesto, manifesto)
            painel.concluir_etapa(indice, etapa.nome)

        painel.iniciar_finalizacao()
        manifesto["artefatos"] = metadados_artefatos(raiz, artefatos)
        manifesto["status"] = "concluida"
        manifesto["concluida_em"] = agora_iso()
        manifesto["duracao_segundos"] = round(perf_counter() - inicio_total, 3)
        gravar_manifesto(caminho_manifesto, manifesto)
        painel.finalizar()

    print(f"\nResultados preservados em: {raiz}", flush=True)
    print(f"Manifesto: {caminho_manifesto}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
