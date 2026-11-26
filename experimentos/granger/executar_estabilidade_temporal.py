"""Executa o experimento encerrado de estabilidade temporal."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

from pipeline.progresso_terminal import PainelPipeline, executar_comando


SCHEMA_VERSION = 1
MESES_MINIMOS_TREINO = 18
PASSO_MESES = 3
MESES_AVALIACAO = 6
HORIZONTE_RESPOSTA = 5
LIMITES = {
    "minimo_janelas_testadas": 3,
    "frequencia_minima_significancia": 0.6,
    "frequencia_minima_residuos": 0.6,
    "frequencia_minima_resposta_positiva": 0.6,
}


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


def sha256_json_canonico(caminho: Path) -> str:
    valor = json.loads(caminho.read_text(encoding="utf-8"))
    conteudo = json.dumps(
        valor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(conteudo.encode()).hexdigest()


def adicionar_meses(valor: date, meses: int) -> date:
    indice = valor.year * 12 + valor.month - 1 + meses
    ano, mes_zero = divmod(indice, 12)
    mes = mes_zero + 1
    dia = min(valor.day, calendar.monthrange(ano, mes)[1])
    return date(ano, mes, dia)


def fim_mes_apos(inicio: date, meses: int) -> date:
    primeiro_mes = date(inicio.year, inicio.month, 1)
    return adicionar_meses(primeiro_mes, meses) - timedelta(days=1)


def construir_janelas_contexto(
    contexto: dict[str, str],
    meses_minimos: int = MESES_MINIMOS_TREINO,
    passo_meses: int = PASSO_MESES,
    meses_avaliacao: int = MESES_AVALIACAO,
) -> list[dict[str, str]]:
    nome = contexto["nome"]
    inicio = date.fromisoformat(contexto["inicio_series"])
    corte_maximo = date.fromisoformat(contexto["corte_treino"])
    fim_disponivel = date.fromisoformat(contexto["fim_avaliacao"])
    corte = fim_mes_apos(inicio, meses_minimos)
    janelas: list[dict[str, str]] = []
    while corte <= corte_maximo:
        fim_teste = fim_mes_apos(corte + timedelta(days=1), meses_avaliacao)
        if fim_teste > fim_disponivel:
            break
        janelas.append(
            {
                "id": f"{nome}_{corte:%Y%m%d}",
                "contexto": nome,
                "data_inicio_treino": inicio.isoformat(),
                "data_corte_treino": corte.isoformat(),
                "data_fim_teste": fim_teste.isoformat(),
            }
        )
        corte = fim_mes_apos(corte + timedelta(days=1), passo_meses)
    return janelas


def gravar_json(caminho: Path, valor: dict[str, Any]) -> None:
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(valor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def validar_entrada(raiz: Path) -> dict[str, Any]:
    caminhos = {
        "manifesto_ffill": raiz / "manifesto.json",
        "validacao_1_origem": raiz / "validacao_1/granger/ffill-3.sqlite",
        "validacao_2_origem": raiz / "validacao_2/granger/ffill-3.sqlite",
        "manifesto_condicional": raiz / "granger-condicional/manifesto.json",
        "configuracao_redes": (
            raiz / "granger-condicional/configuracao_redes.json"
        ),
        "validacao_1_final": raiz / "granger-condicional/validacao_1.sqlite",
        "validacao_2_final": raiz / "granger-condicional/validacao_2.sqlite",
    }
    ausentes = [
        caminho for caminho in caminhos.values() if not caminho.is_file()
    ]
    if ausentes:
        nomes = ", ".join(str(caminho) for caminho in ausentes)
        raise FileNotFoundError(f"artefatos obrigatórios ausentes: {nomes}")
    manifesto_ffill = json.loads(
        caminhos["manifesto_ffill"].read_text(encoding="utf-8")
    )
    manifesto_condicional = json.loads(
        caminhos["manifesto_condicional"].read_text(encoding="utf-8")
    )
    for nome, manifesto in (
        ("ffill", manifesto_ffill),
        ("condicional", manifesto_condicional),
    ):
        if manifesto.get("status") != "concluida":
            raise ValueError(f"o experimento {nome} não está concluído")
        if manifesto.get("inclui_teste_final") is not False:
            raise ValueError(f"o experimento {nome} não protege o teste final")
    if len(manifesto_ffill.get("contextos", [])) != 2:
        raise ValueError("o manifesto ffill deve declarar duas validações")
    return {"caminhos": caminhos, "manifesto_ffill": manifesto_ffill}


def construir_configuracao(
    raiz: Path,
    manifesto_ffill: dict[str, Any],
    caminhos: dict[str, Path],
) -> dict[str, Any]:
    janelas = [
        janela
        for contexto in manifesto_ffill["contextos"]
        for janela in construir_janelas_contexto(contexto)
    ]
    quantidades = Counter(janela["contexto"] for janela in janelas)
    if any(
        quantidades[contexto] < LIMITES["minimo_janelas_testadas"]
        for contexto in ("validacao_1", "validacao_2")
    ):
        raise ValueError("cada validação deve possuir pelo menos três janelas")
    origens = {
        "validacao_1": caminhos["validacao_1_origem"],
        "validacao_2": caminhos["validacao_2_origem"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "metodo": "janelas expansivas trimestrais com redes congeladas",
        "inclui_teste_final": False,
        "experimento_ffill": str(raiz),
        "meses_minimos_treino": MESES_MINIMOS_TREINO,
        "passo_meses": PASSO_MESES,
        "meses_avaliacao_por_janela": MESES_AVALIACAO,
        "horizonte_resposta_dias_uteis": HORIZONTE_RESPOSTA,
        "limites": LIMITES,
        "origens": {
            contexto: {
                "caminho": str(caminho),
                "tamanho_bytes": caminho.stat().st_size,
                "sha256": sha256_arquivo(caminho),
            }
            for contexto, caminho in origens.items()
        },
        "configuracao_redes_sha256": sha256_json_canonico(
            caminhos["configuracao_redes"]
        ),
        "janelas": janelas,
    }


def construir_etapas(
    configuracao: dict[str, Any],
    caminhos: dict[str, Path],
    diretorio_janelas: Path,
    saida: Path,
) -> list[Etapa]:
    etapas: list[Etapa] = []
    for janela in configuracao["janelas"]:
        contexto = janela["contexto"]
        origem = caminhos[f"{contexto}_origem"]
        etapas.append(
            Etapa(
                nome=f"Janela {contexto} até {janela['data_corte_treino']}",
                descricao=(
                    "treino expansivo e avaliação nos seis meses seguintes"
                ),
                comando=[
                    sys.executable,
                    "-m",
                    "experimentos.granger.executar_modelo_condicional",
                    str(origem),
                    "--output-db",
                    str(diretorio_janelas / f"{janela['id']}.sqlite"),
                    "--config-redes",
                    str(caminhos["configuracao_redes"]),
                    "--data-inicio-treino",
                    janela["data_inicio_treino"],
                    "--data-corte-treino",
                    janela["data_corte_treino"],
                    "--data-fim-teste",
                    janela["data_fim_teste"],
                ],
            )
        )
    etapas.append(
        Etapa(
            nome="Agregação da estabilidade temporal",
            descricao="aplica a mesma regra às duas validações e compara redes",
            comando=[
                sys.executable,
                "-m",
                "experimentos.granger.analisar_estabilidade_temporal",
                "--configuracao",
                str(saida / "configuracao.json"),
                "--diretorio-janelas",
                str(diretorio_janelas),
                "--validacao-1",
                str(caminhos["validacao_1_final"]),
                "--validacao-2",
                str(caminhos["validacao_2_final"]),
                "--output-db",
                str(saida / "estabilidade.sqlite"),
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
    saida = raiz / "granger-condicional/estabilidade-temporal"
    try:
        entrada = validar_entrada(raiz)
        configuracao = construir_configuracao(
            raiz,
            entrada["manifesto_ffill"],
            entrada["caminhos"],
        )
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}", flush=True)
        return 2
    if saida.exists():
        print(f"ERRO: a saída já existe: {saida}", flush=True)
        return 2

    saida.mkdir(parents=True)
    caminho_configuracao = saida / "configuracao.json"
    gravar_json(caminho_configuracao, configuracao)
    manifesto: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tipo": "estabilidade_temporal_granger_condicional",
        "status": "em_andamento",
        "iniciada_em": agora_iso(),
        "concluida_em": None,
        "duracao_segundos": None,
        "inclui_teste_final": False,
        "experimento_ffill": str(raiz),
        "janelas_planejadas": len(configuracao["janelas"]),
        "etapas_concluidas": [],
        "artefatos": {},
    }
    caminho_manifesto = saida / "manifesto.json"
    gravar_json(caminho_manifesto, manifesto)
    inicio = perf_counter()

    print(f"Experimento de estabilidade temporal: {saida}", flush=True)
    print("As redes e os limites serão idênticos nas duas validações.")
    print("Os bancos das janelas serão descartados após a agregação.")
    print("O teste final não será construído nem consultado.", flush=True)

    with tempfile.TemporaryDirectory(
        prefix=".janelas-",
        dir=saida,
    ) as temporario:
        diretorio_janelas = Path(temporario)
        etapas = construir_etapas(
            configuracao,
            entrada["caminhos"],
            diretorio_janelas,
            saida,
        )
        painel = PainelPipeline(
            "Estabilidade temporal do Granger",
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
                        perf_counter() - inicio,
                        3,
                    )
                    manifesto["falha"] = {
                        "etapa": etapa.nome,
                        "codigo_saida": codigo_saida,
                    }
                    gravar_json(caminho_manifesto, manifesto)
                    painel.falhar_etapa(etapa.nome, codigo_saida)
                    return codigo_saida
                manifesto["etapas_concluidas"].append(
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
                "configuracao": caminho_configuracao,
                "estabilidade": saida / "estabilidade.sqlite",
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

    print(f"\nResultados preservados em: {saida}", flush=True)
    print(f"Comparação: {saida / 'comparacao.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(executar())
