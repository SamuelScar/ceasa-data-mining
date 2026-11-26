"""Executa o contexto final somente com uma seleção previamente congelada."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from executar_pipeline_versionado import (
    ENTRADA_BRUTA,
    FUSO_HORARIO_VERSAO,
    GRANGER_METODO_SELECIONADO,
    Contexto,
    Etapa,
    argumentos_granger_selecionado,
    agora_iso,
    gravar_manifesto,
    metadados_artefatos,
    registrar_codigo,
    sha256_arquivo,
)
from progresso_terminal import PainelPipeline, executar_comando


RAIZ_RESULTADOS = Path("resultados/avaliacoes-finais")
PADRAO_VERSAO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
CONTEXTO_FINAL = Contexto(
    "final",
    "2022-01-04",
    "2025-10-31",
    "2026-10-31",
)


def obter_selecao() -> Path:
    valor = os.getenv("PIPELINE_SELECAO_JSON", "").strip()
    if not valor:
        raise ValueError(
            "PIPELINE_SELECAO_JSON deve apontar para uma configuração "
            "selecionada e congelada"
        )
    path = Path(valor)
    if not path.is_file():
        raise ValueError(f"configuração selecionada não encontrada: {path}")
    try:
        documento = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ValueError("não foi possível ler a configuração selecionada") from erro
    if int(documento.get("schema_version", 0)) != 2:
        raise ValueError("a configuração selecionada não usa o protocolo contextual")
    if documento.get("teste_final_consultado") is not False:
        raise ValueError("a configuração não comprova o isolamento do teste final")
    if set(documento.get("bases_modelagem", {})) != {
        "validacao_1",
        "validacao_2",
    }:
        raise ValueError("a configuração não registra as duas validações")
    if len(str(documento.get("metodologia_sha256", ""))) != 64:
        raise ValueError("a configuração não possui assinatura metodológica válida")
    politica = documento.get("analise_sensibilidade_cobertura")
    if not isinstance(politica, dict) or (
        politica.get("regra_principal_cobertura_minima_negativo") != 1
        or politica.get("coberturas_minimas_negativo") != [1, 2, 3, 4, 5]
        or politica.get("positivos_confirmados_mantidos") is not True
        or politica.get("participa_selecao") is not False
    ):
        raise ValueError(
            "a configuração não registra a política esperada de sensibilidade "
            "da cobertura"
        )
    return path.resolve()


def gerar_versao_automatica() -> str:
    base = datetime.now(FUSO_HORARIO_VERSAO).strftime(
        "avaliacao-final-%Y%m%d-%H%M%S"
    )
    versao = base
    numero = 2
    while (RAIZ_RESULTADOS / versao).exists():
        versao = f"{base}-{numero:02d}"
        numero += 1
    return versao


def obter_versao() -> tuple[str, bool]:
    valor = os.getenv("PIPELINE_AVALIACAO_FINAL_VERSAO", "").strip()
    if not valor:
        return gerar_versao_automatica(), True
    if PADRAO_VERSAO.fullmatch(valor) is None:
        raise ValueError(
            "PIPELINE_AVALIACAO_FINAL_VERSAO deve usar apenas letras, números, "
            "ponto, hífen ou sublinhado"
        )
    return valor, False


def retomada_solicitada() -> bool:
    valor = os.getenv("PIPELINE_AVALIACAO_FINAL_RETOMAR", "0").strip().lower()
    if valor in {"", "0", "false", "nao", "não"}:
        return False
    if valor in {"1", "true", "sim"}:
        return True
    raise ValueError(
        "PIPELINE_AVALIACAO_FINAL_RETOMAR deve ser 0 ou 1"
    )


def construir_etapas(
    raiz: Path,
    selecao: Path,
) -> tuple[list[Etapa], dict[str, Path]]:
    base_contexto = raiz / "contexto-final"
    artefatos = {
        "selecao": selecao,
        "preprocessamento": (
            base_contexto / "preprocessamento/cotacoes_preprocessadas.sqlite"
        ),
        "granger": base_contexto / "granger/resultados_granger.sqlite",
        "efeito_domino": (
            base_contexto / "efeitos/domino/resultados_efeito_domino.sqlite"
        ),
        "base_modelagem": base_contexto / "modelagem/base_modelagem.sqlite",
        "avaliacao_final": raiz / "treinamento/avaliacao_final.sqlite",
        "modelo_final": raiz / "treinamento/modelo_final.pkl",
    }
    etapas = [
        Etapa(
            "Pré-processamento — final",
            "referências aprendidas somente até outubro de 2025",
            [
                sys.executable,
                "preprocessamento/tratar_ceasa.py",
                str(ENTRADA_BRUTA),
                "--output-db",
                str(artefatos["preprocessamento"]),
                "--data-corte-treino",
                CONTEXTO_FINAL.corte_treino,
            ],
        ),
        Etapa(
            "Granger — final",
            "relações congeladas antes do teste oficial",
            [
                sys.executable,
                "granger/executar_granger.py",
                str(artefatos["preprocessamento"]),
                "--output-db",
                str(artefatos["granger"]),
                "--data-inicio",
                CONTEXTO_FINAL.inicio_series,
                "--data-corte-treino",
                CONTEXTO_FINAL.corte_treino,
                "--data-fim-teste",
                CONTEXTO_FINAL.fim_avaliacao,
                *argumentos_granger_selecionado(),
            ],
        ),
        Etapa(
            "Efeito Dominó — final",
            "caracterização do período oficial com limites congelados",
            [
                sys.executable,
                "efeitos/domino/executar_efeito_domino.py",
                str(artefatos["granger"]),
                "--output-db",
                str(artefatos["efeito_domino"]),
            ],
        ),
        Etapa(
            "Base de modelagem — final",
            "constrói somente treino final e teste oficial",
            [
                sys.executable,
                "modelagem/gerar_base_modelagem.py",
                str(artefatos["preprocessamento"]),
                str(artefatos["granger"]),
                str(artefatos["efeito_domino"]),
                "--contexto",
                "final",
                "--output-db",
                str(artefatos["base_modelagem"]),
            ],
        ),
        Etapa(
            "Avaliação final",
            "ajusta a seleção congelada antes de consultar o teste",
            [
                sys.executable,
                "-m",
                "treinamento.executar_avaliacao_final",
                str(artefatos["base_modelagem"]),
                str(artefatos["selecao"]),
                "--output-db",
                str(artefatos["avaliacao_final"]),
                "--model-output",
                str(artefatos["modelo_final"]),
            ],
        ),
    ]
    return etapas, artefatos


def carregar_manifesto_falho(path: Path, versao: str) -> dict[str, object]:
    try:
        manifesto = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ValueError("não foi possível ler o manifesto da retomada") from erro
    if manifesto.get("tipo") != "avaliacao_final":
        raise ValueError("o manifesto não representa uma avaliação final")
    if manifesto.get("versao") != versao:
        raise ValueError("a versão informada diverge do manifesto")
    if manifesto.get("status") != "falha":
        raise ValueError("somente uma avaliação com falha pode ser retomada")
    falha = manifesto.get("falha")
    if not isinstance(falha, dict) or falha.get("etapa") != "Avaliação final":
        raise ValueError("a retomada só é permitida para falha na etapa final")
    return manifesto


def retomar_avaliacao(
    raiz: Path,
    versao: str,
    selecao_origem: Path,
) -> int:
    caminho_manifesto = raiz / "manifesto.json"
    manifesto = carregar_manifesto_falho(caminho_manifesto, versao)
    selecao_copia = raiz / "treinamento/configuracao_selecionada.json"
    selecao_registrada = manifesto.get("selecao_origem")
    if not isinstance(selecao_registrada, dict):
        raise ValueError("o manifesto não registra a seleção original")
    hash_selecao = sha256_arquivo(selecao_origem)
    if (
        not selecao_copia.is_file()
        or hash_selecao != selecao_registrada.get("sha256")
        or hash_selecao != sha256_arquivo(selecao_copia)
    ):
        raise ValueError("a seleção da retomada diverge da execução original")

    etapas, artefatos = construir_etapas(raiz, selecao_copia)
    entradas_necessarias = (
        artefatos["preprocessamento"],
        artefatos["granger"],
        artefatos["efeito_domino"],
        artefatos["base_modelagem"],
    )
    ausentes = [str(path) for path in entradas_necessarias if not path.is_file()]
    if ausentes:
        raise ValueError(
            "a retomada não encontrou os artefatos concluídos: "
            + ", ".join(ausentes)
        )
    saidas = (artefatos["avaliacao_final"], artefatos["modelo_final"])
    existentes = [str(path) for path in saidas if path.exists()]
    if existentes:
        raise ValueError(
            "a retomada não sobrescreve saídas existentes: "
            + ", ".join(existentes)
        )

    falhas_anteriores = manifesto.setdefault("falhas_anteriores", [])
    if not isinstance(falhas_anteriores, list):
        raise ValueError("o histórico de falhas do manifesto é inválido")
    falhas_anteriores.append(manifesto.pop("falha"))
    retomadas = manifesto.setdefault("retomadas", [])
    if not isinstance(retomadas, list):
        raise ValueError("o histórico de retomadas do manifesto é inválido")
    retomada: dict[str, object] = {
        "iniciada_em": agora_iso(),
        "concluida_em": None,
        "status": "em_andamento",
        "codigo": registrar_codigo(),
    }
    retomadas.append(retomada)
    duracao_anterior = float(manifesto.get("duracao_segundos") or 0.0)
    manifesto["status"] = "em_andamento"
    manifesto["concluida_em"] = None
    gravar_manifesto(caminho_manifesto, manifesto)

    etapa = etapas[-1]
    inicio = perf_counter()
    painel = PainelPipeline(f"Retomada — {versao}", 1)
    with painel:
        painel.iniciar_etapa(1, etapa.nome, etapa.descricao)
        codigo_saida = executar_comando(etapa.comando, painel)
        duracao_retomada = round(perf_counter() - inicio, 3)
        if codigo_saida != 0:
            retomada.update(
                {
                    "concluida_em": agora_iso(),
                    "status": "falha",
                    "duracao_segundos": duracao_retomada,
                }
            )
            manifesto["status"] = "falha"
            manifesto["concluida_em"] = agora_iso()
            manifesto["duracao_segundos"] = round(
                duracao_anterior + duracao_retomada,
                3,
            )
            manifesto["falha"] = {
                "etapa": etapa.nome,
                "codigo_saida": codigo_saida,
            }
            gravar_manifesto(caminho_manifesto, manifesto)
            painel.falhar_etapa(etapa.nome, codigo_saida)
            painel.exibir_log()
            return codigo_saida

        concluidas = manifesto.get("etapas_concluidas")
        if not isinstance(concluidas, list):
            raise ValueError("o manifesto não registra as etapas concluídas")
        concluidas.append(
            {
                "nome": etapa.nome,
                "duracao_segundos": duracao_retomada,
                "retomada": True,
            }
        )
        retomada.update(
            {
                "concluida_em": agora_iso(),
                "status": "concluida",
                "duracao_segundos": duracao_retomada,
            }
        )
        manifesto["artefatos"] = metadados_artefatos(raiz, artefatos)
        manifesto["status"] = "concluida"
        manifesto["concluida_em"] = agora_iso()
        manifesto["duracao_segundos"] = round(
            duracao_anterior + duracao_retomada,
            3,
        )
        gravar_manifesto(caminho_manifesto, manifesto)
        painel.concluir_etapa(1, etapa.nome)
        painel.finalizar()

    print(f"\nAvaliação final retomada e preservada em: {raiz}", flush=True)
    return 0


def executar() -> int:
    try:
        selecao_origem = obter_selecao()
        versao, versao_automatica = obter_versao()
        retomar = retomada_solicitada()
    except ValueError as erro:
        print(f"ERRO: {erro}", flush=True)
        return 2

    if not ENTRADA_BRUTA.is_file():
        print(f"ERRO: banco bruto não encontrado: {ENTRADA_BRUTA}", flush=True)
        return 2

    raiz = RAIZ_RESULTADOS / versao
    if retomar:
        if versao_automatica:
            print(
                "ERRO: a retomada exige PIPELINE_AVALIACAO_FINAL_VERSAO.",
                flush=True,
            )
            return 2
        try:
            return retomar_avaliacao(raiz, versao, selecao_origem)
        except ValueError as erro:
            print(f"ERRO: {erro}", flush=True)
            return 2
    if raiz.exists():
        print(f"ERRO: a avaliação {versao!r} já existe em {raiz}.", flush=True)
        return 2

    selecao_copia = raiz / "treinamento/configuracao_selecionada.json"
    selecao_copia.parent.mkdir(parents=True)
    shutil.copy2(selecao_origem, selecao_copia)
    if sha256_arquivo(selecao_origem) != sha256_arquivo(selecao_copia):
        raise RuntimeError("a cópia da configuração selecionada diverge da origem")

    etapas, artefatos = construir_etapas(raiz, selecao_copia)
    inicio_total = perf_counter()
    manifesto: dict[str, object] = {
        "schema_version": 2,
        "tipo": "avaliacao_final",
        "versao": versao,
        "versao_automatica": versao_automatica,
        "status": "em_andamento",
        "iniciada_em": agora_iso(),
        "concluida_em": None,
        "duracao_segundos": None,
        "inclui_teste_final": True,
        "selecao_origem": {
            "caminho": str(selecao_origem),
            "sha256": sha256_arquivo(selecao_origem),
        },
        "contexto": {
            "inicio_series": CONTEXTO_FINAL.inicio_series,
            "corte_treino": CONTEXTO_FINAL.corte_treino,
            "fim_avaliacao": CONTEXTO_FINAL.fim_avaliacao,
        },
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

    print("AVALIAÇÃO FINAL OFICIAL", flush=True)
    print(f"Seleção congelada: {selecao_origem}", flush=True)
    print("Esta execução consultará o período oficial de teste.", flush=True)

    painel = PainelPipeline(
        f"Avaliação final — {versao}",
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

    print(f"\nAvaliação final preservada em: {raiz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
