"""Gera a base supervisionada para modelagem do Efeito Dominó."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
JANELAS_HISTORICAS = (5, 10, 20)
MAX_LAG_RETORNO = 10
DEFAULT_INICIO_VALIDACAO_1 = "2024-01-01"
DEFAULT_INICIO_VALIDACAO_2 = "2025-01-01"

COLUNAS_CATEGORICAS = [
    "produto",
    "origem",
    "destino",
    "uf_origem",
    "uf_destino",
]

COLUNAS_INTEIRAS = [
    "mesmo_estado",
    "ano",
    "mes",
    "trimestre",
    "dia_semana",
    "lag_granger",
]

COLUNAS_REAIS = [
    "granger_estatistica_f",
    "granger_p_valor",
    "granger_q_valor_fdr",
    "limite_origem_percentual",
    "limite_destino_percentual",
    "alta_origem_log",
    "alta_origem_percentual",
    "excesso_limite_origem_log",
    "razao_alta_limite_origem",
    "origem_preco_kg",
    "destino_preco_kg",
    "origem_log_preco_kg",
    "destino_log_preco_kg",
    "diferenca_log_precos",
    "origem_retorno_atual",
    "destino_retorno_atual",
]

for lado in ("origem", "destino"):
    COLUNAS_INTEIRAS.extend(
        [
            f"{lado}_n_cotacoes",
            f"{lado}_n_exatas",
            f"{lado}_n_calculadas",
            f"{lado}_n_padronizadas",
            f"{lado}_n_inferidas",
        ]
    )
    COLUNAS_REAIS.extend(
        f"{lado}_retorno_lag_{lag}" for lag in range(1, MAX_LAG_RETORNO + 1)
    )
    for janela in JANELAS_HISTORICAS:
        COLUNAS_REAIS.extend(
            [
                f"{lado}_retorno_media_{janela}d",
                f"{lado}_retorno_desvio_{janela}d",
                f"{lado}_retorno_acumulado_{janela}d",
                f"{lado}_cobertura_observada_{janela}d",
            ]
        )


@dataclass(frozen=True)
class Configuracao:
    inicio_validacao_1: date
    inicio_validacao_2: date


@dataclass(frozen=True)
class ContextoTemporal:
    inicio_series: date
    corte_treino: date
    fim_teste: date
    janela_rotulo_dias: int


def agora_iso() -> str:
    return datetime.now(UTC).isoformat()


def valor_ambiente(nome: str, padrao: str) -> str:
    valor = os.getenv(nome)
    return valor if valor not in (None, "") else padrao


def data_iso(valor: str) -> date:
    try:
        return date.fromisoformat(valor)
    except ValueError as erro:
        raise argparse.ArgumentTypeError(
            f"data inválida: {valor!r}; use o formato AAAA-MM-DD"
        ) from erro


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combina os bancos pré-processado, de Granger e do Efeito Dominó "
            "em uma base supervisionada com divisões temporais auditáveis."
        )
    )
    parser.add_argument(
        "preprocessamento_db",
        type=Path,
        help="Banco gerado pelo pré-processamento",
    )
    parser.add_argument(
        "granger_db",
        type=Path,
        help="Banco de séries temporais e resultados de Granger",
    )
    parser.add_argument(
        "domino_db",
        type=Path,
        help="Banco de eventos caracterizados do Efeito Dominó",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("resultados/modelagem/base_modelagem.sqlite"),
        help="SQLite que receberá a base supervisionada",
    )
    parser.add_argument(
        "--inicio-validacao-1",
        type=data_iso,
        default=data_iso(
            valor_ambiente(
                "MODELAGEM_INICIO_VALIDACAO_1",
                DEFAULT_INICIO_VALIDACAO_1,
            )
        ),
        help="Início da primeira validação temporal",
    )
    parser.add_argument(
        "--inicio-validacao-2",
        type=data_iso,
        default=data_iso(
            valor_ambiente(
                "MODELAGEM_INICIO_VALIDACAO_2",
                DEFAULT_INICIO_VALIDACAO_2,
            )
        ),
        help="Início da segunda validação temporal",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Substitui a saída somente depois de uma execução válida",
    )
    return parser


def validar_argumentos(args: argparse.Namespace) -> Configuracao:
    entradas = [
        args.preprocessamento_db,
        args.granger_db,
        args.domino_db,
    ]
    for entrada in entradas:
        if not entrada.is_file():
            raise FileNotFoundError(f"banco de entrada não encontrado: {entrada}")

    caminhos = [entrada.resolve() for entrada in entradas]
    if len(set(caminhos)) != len(caminhos):
        raise ValueError("os três bancos de entrada devem ser arquivos diferentes")
    if args.output_db.resolve() in caminhos:
        raise ValueError("o banco de saída deve ser diferente dos bancos de entrada")
    if args.output_db.exists() and not args.overwrite:
        raise FileExistsError(
            f"a saída já existe: {args.output_db}; use --overwrite para substituí-la"
        )
    if args.inicio_validacao_1 >= args.inicio_validacao_2:
        raise ValueError(
            "--inicio-validacao-1 deve ser anterior a --inicio-validacao-2"
        )
    return Configuracao(
        inicio_validacao_1=args.inicio_validacao_1,
        inicio_validacao_2=args.inicio_validacao_2,
    )


def sha256_arquivo(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def abrir_somente_leitura(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def validar_tabelas(
    connection: sqlite3.Connection,
    obrigatorias: set[str],
    descricao: str,
) -> None:
    tabelas = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    ausentes = obrigatorias - tabelas
    if ausentes:
        raise ValueError(
            f"banco {descricao} incompleto; estruturas ausentes: "
            + ", ".join(sorted(ausentes))
        )


def carregar_execucao(
    connection: sqlite3.Connection,
    tabela: str,
    descricao: str,
) -> dict[str, object]:
    execucao = connection.execute(
        f"SELECT * FROM {tabela} ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if execucao is None or execucao["status"] != "concluida":
        raise ValueError(f"o banco {descricao} não possui execução concluída")
    return dict(execucao)


def validar_fontes(
    preprocessamento_db: Path,
    granger_db: Path,
    domino_db: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, str],
    ContextoTemporal,
]:
    preprocessamento = abrir_somente_leitura(preprocessamento_db)
    try:
        validar_tabelas(
            preprocessamento,
            {"execucoes_preprocessamento", "cotacoes_preprocessadas"},
            "pré-processado",
        )
        execucao_preprocessamento = carregar_execucao(
            preprocessamento,
            "execucoes_preprocessamento",
            "pré-processado",
        )
    finally:
        preprocessamento.close()

    granger = abrir_somente_leitura(granger_db)
    try:
        validar_tabelas(
            granger,
            {
                "execucoes_granger",
                "series_temporais",
                "resultados_granger",
            },
            "de Granger",
        )
        execucao_granger = carregar_execucao(
            granger,
            "execucoes_granger",
            "de Granger",
        )
    finally:
        granger.close()

    domino = abrir_somente_leitura(domino_db)
    try:
        validar_tabelas(
            domino,
            {
                "execucoes_efeito_domino",
                "eventos_efeito_domino",
                "relacoes_analisadas",
            },
            "do Efeito Dominó",
        )
        execucao_domino = carregar_execucao(
            domino,
            "execucoes_efeito_domino",
            "do Efeito Dominó",
        )
    finally:
        domino.close()

    hashes = {
        "preprocessamento": sha256_arquivo(preprocessamento_db),
        "granger": sha256_arquivo(granger_db),
        "domino": sha256_arquivo(domino_db),
    }
    if execucao_granger["banco_origem_sha256"] != hashes["preprocessamento"]:
        raise ValueError(
            "o banco de Granger não foi produzido a partir do banco "
            "pré-processado informado"
        )
    if execucao_domino["banco_origem_sha256"] != hashes["granger"]:
        raise ValueError(
            "o banco do Efeito Dominó não foi produzido a partir do banco "
            "de Granger informado"
        )
    if int(execucao_granger["execucao_preprocessamento_id"]) != int(
        execucao_preprocessamento["id"]
    ):
        raise ValueError("a execução de Granger referencia outro pré-processamento")
    if int(execucao_domino["execucao_granger_id"]) != int(execucao_granger["id"]):
        raise ValueError("a execução do Efeito Dominó referencia outro Granger")

    parametros_granger = json.loads(str(execucao_granger["parametros_json"]))
    parametros_domino = json.loads(str(execucao_domino["parametros_json"]))
    contexto = ContextoTemporal(
        inicio_series=date.fromisoformat(parametros_granger["data_inicio"]),
        corte_treino=date.fromisoformat(
            parametros_granger["data_corte_treino"]
        ),
        fim_teste=date.fromisoformat(parametros_granger["data_fim_teste"]),
        janela_rotulo_dias=int(parametros_domino["janela_dias_uteis"]),
    )
    return (
        execucao_preprocessamento,
        execucao_granger,
        execucao_domino,
        hashes,
        contexto,
    )


def validar_configuracao_temporal(
    config: Configuracao,
    contexto: ContextoTemporal,
) -> None:
    if not (
        contexto.inicio_series
        < config.inicio_validacao_1
        < config.inicio_validacao_2
        <= contexto.corte_treino
        < contexto.fim_teste
    ):
        raise ValueError(
            "as datas devem respeitar: início das séries < validação 1 < "
            "validação 2 <= corte do treino < fim do teste"
        )
    if contexto.janela_rotulo_dias < 1:
        raise ValueError("a janela do rótulo do Efeito Dominó deve ser positiva")


def carregar_eventos(domino_db: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    connection = abrir_somente_leitura(domino_db)
    try:
        eventos = pd.read_sql_query(
            """
            SELECT
                e.id AS evento_domino_id,
                e.relacao_id,
                e.produto,
                e.origem,
                e.destino,
                e.data_choque,
                e.periodo AS periodo_domino,
                e.preco_origem AS preco_origem_evento,
                e.alta_origem_log,
                e.alta_origem_percentual,
                e.limite_origem_percentual,
                e.preco_destino_inicial AS preco_destino_evento,
                e.limite_destino_percentual,
                e.dias_observados_destino,
                e.maior_alta_destino_percentual,
                e.data_maior_alta_destino,
                e.data_resposta,
                e.dias_ate_resposta,
                e.efeito_domino,
                e.status,
                r.lag_selecionado AS lag_granger,
                r.estatistica_f AS granger_estatistica_f,
                r.p_valor AS granger_p_valor,
                r.q_valor_fdr AS granger_q_valor_fdr
            FROM eventos_efeito_domino e
            JOIN relacoes_analisadas r ON r.id = e.relacao_id
            ORDER BY e.id
            """,
            connection,
        )
    finally:
        connection.close()

    if eventos.empty:
        raise ValueError("o banco do Efeito Dominó não contém eventos")
    eventos["data_choque"] = pd.to_datetime(
        eventos["data_choque"],
        errors="raise",
    )
    auditoria = eventos[
        [
            "evento_domino_id",
            "periodo_domino",
            "status",
            "preco_origem_evento",
            "preco_destino_evento",
            "dias_observados_destino",
            "maior_alta_destino_percentual",
            "data_maior_alta_destino",
            "data_resposta",
            "dias_ate_resposta",
            "efeito_domino",
        ]
    ].copy()
    return eventos, auditoria


def carregar_series(granger_db: Path) -> pd.DataFrame:
    connection = abrir_somente_leitura(granger_db)
    try:
        series = pd.read_sql_query(
            """
            SELECT
                produto,
                mercado,
                uf,
                data_cotacao,
                preco_kg,
                n_cotacoes,
                n_exatas,
                n_calculadas,
                n_padronizadas,
                n_inferidas,
                observado,
                preenchido,
                segmento_serie,
                retorno_log
            FROM series_temporais
            ORDER BY produto, mercado, data_cotacao
            """,
            connection,
        )
    finally:
        connection.close()

    if series.empty:
        raise ValueError("o banco de Granger não contém séries temporais")
    series["data_cotacao"] = pd.to_datetime(
        series["data_cotacao"],
        errors="raise",
    )
    chave = ["produto", "mercado", "data_cotacao"]
    if series.duplicated(chave).any():
        raise ValueError("as séries de Granger possuem chaves temporais duplicadas")
    return series


def preparar_atributos_series(series: pd.DataFrame) -> pd.DataFrame:
    series = series.copy()
    series["log_preco_kg"] = np.where(
        series["preco_kg"].notna() & (series["preco_kg"] > 0),
        np.log(series["preco_kg"]),
        np.nan,
    )
    chave_serie = ["produto", "mercado"]
    grupos_segmento = series.groupby(
        [*chave_serie, "segmento_serie"],
        sort=False,
    )
    for lag in range(1, MAX_LAG_RETORNO + 1):
        series[f"retorno_lag_{lag}"] = grupos_segmento["retorno_log"].shift(lag)

    for janela in JANELAS_HISTORICAS:
        series[f"retorno_media_{janela}d"] = grupos_segmento[
            "retorno_log"
        ].transform(
            lambda valores: valores.rolling(
                janela,
                min_periods=1,
            ).mean()
        )
        series[f"retorno_desvio_{janela}d"] = grupos_segmento[
            "retorno_log"
        ].transform(
            lambda valores: valores.rolling(
                janela,
                min_periods=1,
            ).std(ddof=0)
        )
        series[f"retorno_acumulado_{janela}d"] = grupos_segmento[
            "retorno_log"
        ].transform(
            lambda valores: valores.rolling(
                janela,
                min_periods=1,
            ).sum()
        )
        series[f"cobertura_observada_{janela}d"] = grupos_segmento[
            "observado"
        ].transform(
            lambda valores: valores.rolling(
                janela,
                min_periods=1,
            ).mean()
        )
    return series


def colunas_series() -> list[str]:
    colunas = [
        "produto",
        "mercado",
        "data_cotacao",
        "uf",
        "preco_kg",
        "log_preco_kg",
        "retorno_log",
        "n_cotacoes",
        "n_exatas",
        "n_calculadas",
        "n_padronizadas",
        "n_inferidas",
        "observado",
        "preenchido",
    ]
    colunas.extend(f"retorno_lag_{lag}" for lag in range(1, MAX_LAG_RETORNO + 1))
    for janela in JANELAS_HISTORICAS:
        colunas.extend(
            [
                f"retorno_media_{janela}d",
                f"retorno_desvio_{janela}d",
                f"retorno_acumulado_{janela}d",
                f"cobertura_observada_{janela}d",
            ]
        )
    return colunas


def renomear_lado(series: pd.DataFrame, lado: str) -> pd.DataFrame:
    frame = series[colunas_series()].copy()
    frame = frame.rename(
        columns={
            "mercado": lado,
            "data_cotacao": "data_choque",
            **{
                coluna: f"{lado}_{coluna}"
                for coluna in frame.columns
                if coluna not in {"produto", "mercado", "data_cotacao"}
            },
        }
    )
    return frame


def validar_juncoes(eventos: pd.DataFrame) -> None:
    if eventos["origem_preco_kg"].isna().any():
        raise ValueError("existem choques sem a série da origem na mesma data")

    classificaveis = eventos["efeito_domino"].notna()
    destino_invalido = classificaveis & (
        eventos["destino_preco_kg"].isna()
        | (eventos["destino_observado"] != 1)
    )
    if destino_invalido.any():
        raise ValueError(
            "existem eventos classificáveis sem preço observado no destino"
        )

    origem_invalida = eventos["origem_observado"] != 1
    if origem_invalida.any():
        raise ValueError("existem choques baseados em preço não observado na origem")

    diferenca_origem = (
        eventos["origem_preco_kg"] - eventos["preco_origem_evento"]
    ).abs()
    if (diferenca_origem > 1e-9).any():
        raise ValueError("o preço da origem diverge entre Granger e Efeito Dominó")

    diferenca_destino = (
        eventos.loc[classificaveis, "destino_preco_kg"]
        - eventos.loc[classificaveis, "preco_destino_evento"]
    ).abs()
    if (diferenca_destino > 1e-9).any():
        raise ValueError("o preço do destino diverge entre Granger e Efeito Dominó")


def construir_eventos_modelagem(
    eventos: pd.DataFrame,
    series: pd.DataFrame,
) -> pd.DataFrame:
    origem = renomear_lado(series, "origem")
    destino = renomear_lado(series, "destino")
    frame = eventos.merge(
        origem,
        on=["produto", "origem", "data_choque"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        destino,
        on=["produto", "destino", "data_choque"],
        how="left",
        validate="many_to_one",
    )
    validar_juncoes(frame)

    frame["mesmo_estado"] = (
        frame["origem_uf"].notna()
        & frame["destino_uf"].notna()
        & (frame["origem_uf"] == frame["destino_uf"])
    ).astype(int)
    frame["ano"] = frame["data_choque"].dt.year
    frame["mes"] = frame["data_choque"].dt.month
    frame["trimestre"] = frame["data_choque"].dt.quarter
    frame["dia_semana"] = frame["data_choque"].dt.weekday
    frame["excesso_limite_origem_log"] = frame["alta_origem_log"] - np.log1p(
        frame["limite_origem_percentual"]
    )
    frame["razao_alta_limite_origem"] = (
        frame["alta_origem_percentual"]
        / frame["limite_origem_percentual"]
    )
    frame["diferenca_log_precos"] = (
        frame["origem_log_preco_kg"] - frame["destino_log_preco_kg"]
    )
    frame["grupo_choque"] = (
        frame["produto"]
        + "|"
        + frame["origem"]
        + "|"
        + frame["data_choque"].dt.strftime("%Y-%m-%d")
    )
    frame["elegivel_modelagem"] = frame["efeito_domino"].notna().astype(int)
    frame["efeito_domino"] = frame["efeito_domino"].astype("Int64")

    frame = frame.rename(
        columns={
            "origem_uf": "uf_origem",
            "destino_uf": "uf_destino",
            "origem_retorno_log": "origem_retorno_atual",
            "destino_retorno_log": "destino_retorno_atual",
        }
    )
    frame.insert(0, "id", np.arange(1, len(frame) + 1, dtype=int))
    frame["data_choque"] = frame["data_choque"].dt.strftime("%Y-%m-%d")

    colunas = [
        "id",
        "evento_domino_id",
        "relacao_id",
        "data_choque",
        "grupo_choque",
        *COLUNAS_CATEGORICAS,
        *COLUNAS_INTEIRAS,
        *COLUNAS_REAIS,
        "efeito_domino",
        "elegivel_modelagem",
    ]
    ausentes = set(colunas) - set(frame.columns)
    if ausentes:
        raise RuntimeError(
            "atributos de modelagem ausentes: " + ", ".join(sorted(ausentes))
        )
    return frame[colunas].copy()


def preparar_auditoria(
    auditoria: pd.DataFrame,
    eventos_modelagem: pd.DataFrame,
    janela_rotulo_dias: int,
) -> pd.DataFrame:
    ids = eventos_modelagem[["id", "evento_domino_id"]]
    frame = ids.merge(
        auditoria,
        on="evento_domino_id",
        how="left",
        validate="one_to_one",
    )
    datas = pd.to_datetime(
        eventos_modelagem.set_index("evento_domino_id")["data_choque"],
        errors="raise",
    )
    disponibilidade = datas + pd.offsets.BDay(janela_rotulo_dias)
    frame["data_disponibilidade_rotulo"] = frame["evento_domino_id"].map(
        disponibilidade.dt.strftime("%Y-%m-%d")
    )
    frame["efeito_domino"] = frame["efeito_domino"].astype("Int64")
    return frame


def adicionar_divisao(
    linhas: list[dict[str, object]],
    eventos: pd.DataFrame,
    nome: str,
    fim_treino: date,
    inicio_avaliacao: date,
    fim_avaliacao: date,
    papel_avaliacao: str,
    janela_rotulo_dias: int,
) -> None:
    for evento in eventos.itertuples(index=False):
        data_choque = evento.data_choque.date()
        data_disponibilidade = (
            evento.data_choque + pd.offsets.BDay(janela_rotulo_dias)
        ).date()
        papel = None
        motivo = None

        if data_choque <= fim_treino:
            if data_disponibilidade <= fim_treino:
                papel = "treino"
            else:
                papel = "excluido"
                motivo = "janela_rotulo_ultrapassa_fronteira"
        elif inicio_avaliacao <= data_choque <= fim_avaliacao:
            if data_disponibilidade <= fim_avaliacao:
                papel = papel_avaliacao
            else:
                papel = "excluido"
                motivo = "janela_rotulo_ultrapassa_fim_avaliacao"

        if papel is not None:
            linhas.append(
                {
                    "evento_id": int(evento.id),
                    "divisao": nome,
                    "papel": papel,
                    "motivo_exclusao": motivo,
                }
            )


def construir_particoes(
    eventos_modelagem: pd.DataFrame,
    config: Configuracao,
    contexto: ContextoTemporal,
) -> pd.DataFrame:
    eventos = eventos_modelagem.loc[
        eventos_modelagem["elegivel_modelagem"] == 1,
        ["id", "data_choque"],
    ].copy()
    eventos["data_choque"] = pd.to_datetime(
        eventos["data_choque"],
        errors="raise",
    )
    linhas: list[dict[str, object]] = []

    fim_treino_1 = config.inicio_validacao_1 - timedelta(days=1)
    fim_validacao_1 = config.inicio_validacao_2 - timedelta(days=1)
    adicionar_divisao(
        linhas,
        eventos,
        "validacao_1",
        fim_treino_1,
        config.inicio_validacao_1,
        fim_validacao_1,
        "validacao",
        contexto.janela_rotulo_dias,
    )

    fim_treino_2 = config.inicio_validacao_2 - timedelta(days=1)
    adicionar_divisao(
        linhas,
        eventos,
        "validacao_2",
        fim_treino_2,
        config.inicio_validacao_2,
        contexto.corte_treino,
        "validacao",
        contexto.janela_rotulo_dias,
    )

    adicionar_divisao(
        linhas,
        eventos,
        "final",
        contexto.corte_treino,
        contexto.corte_treino + timedelta(days=1),
        contexto.fim_teste,
        "teste",
        contexto.janela_rotulo_dias,
    )
    particoes = pd.DataFrame(linhas)
    if particoes.empty:
        raise ValueError("nenhuma partição temporal pôde ser construída")
    particoes.insert(0, "id", np.arange(1, len(particoes) + 1, dtype=int))
    return particoes


def construir_catalogo() -> pd.DataFrame:
    linhas: list[dict[str, object]] = []

    def registrar(
        nome: str,
        papel: str,
        tipo: str,
        origem: str,
        regra: str,
        usar_modelo: int,
    ) -> None:
        linhas.append(
            {
                "nome": nome,
                "papel": papel,
                "tipo": tipo,
                "origem": origem,
                "regra": regra,
                "usar_modelo": usar_modelo,
            }
        )

    for nome in (
        "id",
        "evento_domino_id",
        "relacao_id",
        "data_choque",
        "grupo_choque",
    ):
        registrar(
            nome,
            "identificador",
            "texto" if nome in {"data_choque", "grupo_choque"} else "inteiro",
            "controle",
            "usado para rastreabilidade e separação; não entra no modelo",
            0,
        )
    for nome in COLUNAS_CATEGORICAS:
        registrar(
            nome,
            "preditora_categorica",
            "texto",
            "Granger",
            "informação disponível na data do choque",
            1,
        )
    for nome in COLUNAS_INTEIRAS:
        registrar(
            nome,
            "preditora_numerica",
            "inteiro",
            "séries históricas ou calendário",
            "calculada sem usar datas posteriores ao choque",
            1,
        )
    for nome in COLUNAS_REAIS:
        registrar(
            nome,
            "preditora_numerica",
            "real",
            "Granger, séries históricas ou Efeito Dominó",
            "calculada com informações disponíveis até a data do choque",
            1,
        )
    registrar(
        "efeito_domino",
        "alvo",
        "inteiro_nulo",
        "Efeito Dominó",
        "1 confirmado; 0 não confirmado; nulo não classificável",
        0,
    )
    registrar(
        "elegivel_modelagem",
        "controle",
        "inteiro",
        "Efeito Dominó",
        "1 somente quando o alvo é 0 ou 1",
        0,
    )
    catalogo = pd.DataFrame(linhas)
    catalogo.insert(0, "id", np.arange(1, len(catalogo) + 1, dtype=int))
    return catalogo


def criar_schema(connection: sqlite3.Connection) -> None:
    categoricas_sql = ",\n".join(
        f"{nome} TEXT" for nome in COLUNAS_CATEGORICAS
    )
    inteiras_sql = ",\n".join(
        f"{nome} INTEGER" for nome in COLUNAS_INTEIRAS
    )
    reais_sql = ",\n".join(f"{nome} REAL" for nome in COLUNAS_REAIS)
    connection.executescript(
        f"""
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_modelagem (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT,
            parametros_json TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE fontes_dados (
            nome TEXT PRIMARY KEY,
            caminho TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            execucao_id INTEGER NOT NULL,
            schema_version INTEGER NOT NULL
        );

        CREATE TABLE eventos_modelagem (
            id INTEGER PRIMARY KEY,
            evento_domino_id INTEGER NOT NULL UNIQUE,
            relacao_id INTEGER NOT NULL,
            data_choque TEXT NOT NULL,
            grupo_choque TEXT NOT NULL,
            {categoricas_sql},
            {inteiras_sql},
            {reais_sql},
            efeito_domino INTEGER CHECK (efeito_domino IN (0, 1)),
            elegivel_modelagem INTEGER NOT NULL
                CHECK (elegivel_modelagem IN (0, 1))
        );

        CREATE TABLE auditoria_eventos (
            id INTEGER PRIMARY KEY,
            evento_domino_id INTEGER NOT NULL UNIQUE,
            periodo_domino TEXT NOT NULL,
            status TEXT NOT NULL,
            preco_origem_evento REAL NOT NULL,
            preco_destino_evento REAL,
            dias_observados_destino INTEGER NOT NULL,
            maior_alta_destino_percentual REAL,
            data_maior_alta_destino TEXT,
            data_resposta TEXT,
            dias_ate_resposta INTEGER,
            efeito_domino INTEGER CHECK (efeito_domino IN (0, 1)),
            data_disponibilidade_rotulo TEXT NOT NULL,
            FOREIGN KEY (id) REFERENCES eventos_modelagem(id)
        );

        CREATE TABLE particoes_modelagem (
            id INTEGER PRIMARY KEY,
            evento_id INTEGER NOT NULL,
            divisao TEXT NOT NULL,
            papel TEXT NOT NULL
                CHECK (papel IN ('treino', 'validacao', 'teste', 'excluido')),
            motivo_exclusao TEXT,
            FOREIGN KEY (evento_id) REFERENCES eventos_modelagem(id),
            UNIQUE (evento_id, divisao)
        );

        CREATE TABLE catalogo_atributos (
            id INTEGER PRIMARY KEY,
            nome TEXT NOT NULL UNIQUE,
            papel TEXT NOT NULL,
            tipo TEXT NOT NULL,
            origem TEXT NOT NULL,
            regra TEXT NOT NULL,
            usar_modelo INTEGER NOT NULL CHECK (usar_modelo IN (0, 1))
        );

        CREATE TABLE resumo_modelagem (
            chave TEXT PRIMARY KEY,
            valor TEXT NOT NULL
        );

        CREATE INDEX idx_eventos_modelagem_data
            ON eventos_modelagem(data_choque);
        CREATE INDEX idx_eventos_modelagem_relacao
            ON eventos_modelagem(relacao_id, data_choque);
        CREATE INDEX idx_eventos_modelagem_grupo
            ON eventos_modelagem(grupo_choque);
        CREATE INDEX idx_eventos_modelagem_alvo
            ON eventos_modelagem(elegivel_modelagem, efeito_domino);
        CREATE INDEX idx_particoes_divisao_papel
            ON particoes_modelagem(divisao, papel);

        CREATE VIEW treino_validacao_1 AS
        SELECT e.*
        FROM eventos_modelagem e
        JOIN particoes_modelagem p ON p.evento_id = e.id
        WHERE p.divisao = 'validacao_1' AND p.papel = 'treino';

        CREATE VIEW validacao_1 AS
        SELECT e.*
        FROM eventos_modelagem e
        JOIN particoes_modelagem p ON p.evento_id = e.id
        WHERE p.divisao = 'validacao_1' AND p.papel = 'validacao';

        CREATE VIEW treino_validacao_2 AS
        SELECT e.*
        FROM eventos_modelagem e
        JOIN particoes_modelagem p ON p.evento_id = e.id
        WHERE p.divisao = 'validacao_2' AND p.papel = 'treino';

        CREATE VIEW validacao_2 AS
        SELECT e.*
        FROM eventos_modelagem e
        JOIN particoes_modelagem p ON p.evento_id = e.id
        WHERE p.divisao = 'validacao_2' AND p.papel = 'validacao';

        CREATE VIEW treino_final AS
        SELECT e.*
        FROM eventos_modelagem e
        JOIN particoes_modelagem p ON p.evento_id = e.id
        WHERE p.divisao = 'final' AND p.papel = 'treino';

        CREATE VIEW teste_final AS
        SELECT e.*
        FROM eventos_modelagem e
        JOIN particoes_modelagem p ON p.evento_id = e.id
        WHERE p.divisao = 'final' AND p.papel = 'teste';
        """
    )


def gerar_resumo(
    eventos: pd.DataFrame,
    particoes: pd.DataFrame,
) -> dict[str, int]:
    resumo = {
        "eventos_totais": int(len(eventos)),
        "eventos_elegiveis": int(eventos["elegivel_modelagem"].sum()),
        "eventos_nao_classificaveis": int(
            (eventos["elegivel_modelagem"] == 0).sum()
        ),
        "eventos_positivos": int((eventos["efeito_domino"] == 1).sum()),
        "eventos_negativos": int((eventos["efeito_domino"] == 0).sum()),
        "atributos_preditores": int(
            len(COLUNAS_CATEGORICAS) + len(COLUNAS_INTEIRAS) + len(COLUNAS_REAIS)
        ),
    }
    particoes_com_alvo = particoes.merge(
        eventos[["id", "efeito_domino"]],
        left_on="evento_id",
        right_on="id",
        how="left",
        validate="many_to_one",
    )
    for divisao in ("validacao_1", "validacao_2", "final"):
        for papel in ("treino", "validacao", "teste", "excluido"):
            filtro = (
                (particoes_com_alvo["divisao"] == divisao)
                & (particoes_com_alvo["papel"] == papel)
            )
            quantidade = int(filtro.sum())
            if quantidade:
                prefixo = f"{divisao}_{papel}"
                resumo[prefixo] = quantidade
                resumo[f"{prefixo}_positivos"] = int(
                    (filtro & (particoes_com_alvo["efeito_domino"] == 1)).sum()
                )
                resumo[f"{prefixo}_negativos"] = int(
                    (filtro & (particoes_com_alvo["efeito_domino"] == 0)).sum()
                )
    return resumo


def validar_saida(
    connection: sqlite3.Connection,
    resumo: dict[str, int],
) -> None:
    integridade = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integridade != "ok":
        raise RuntimeError(f"falha de integridade no SQLite: {integridade}")

    total, elegiveis, positivos, negativos = connection.execute(
        """
        SELECT
            COUNT(*),
            SUM(elegivel_modelagem),
            SUM(CASE WHEN efeito_domino = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN efeito_domino = 0 THEN 1 ELSE 0 END)
        FROM eventos_modelagem
        """
    ).fetchone()
    observadas = tuple(
        int(valor or 0) for valor in (total, elegiveis, positivos, negativos)
    )
    esperadas = (
        resumo["eventos_totais"],
        resumo["eventos_elegiveis"],
        resumo["eventos_positivos"],
        resumo["eventos_negativos"],
    )
    if observadas != esperadas:
        raise RuntimeError("as contagens da base não coincidem com o resumo")

    auditoria = connection.execute(
        "SELECT COUNT(*) FROM auditoria_eventos"
    ).fetchone()[0]
    if int(auditoria) != resumo["eventos_totais"]:
        raise RuntimeError("a auditoria não contém todos os eventos")

    inconsistentes = connection.execute(
        """
        SELECT COUNT(*)
        FROM eventos_modelagem
        WHERE (elegivel_modelagem = 1 AND efeito_domino IS NULL)
           OR (elegivel_modelagem = 0 AND efeito_domino IS NOT NULL)
        """
    ).fetchone()[0]
    if int(inconsistentes):
        raise RuntimeError("existem eventos com elegibilidade e alvo incompatíveis")

    particoes_invalidas = connection.execute(
        """
        SELECT COUNT(*)
        FROM particoes_modelagem p
        JOIN eventos_modelagem e ON e.id = p.evento_id
        WHERE e.elegivel_modelagem <> 1 OR e.efeito_domino IS NULL
        """
    ).fetchone()[0]
    if int(particoes_invalidas):
        raise RuntimeError("existem partições associadas a eventos não classificáveis")

    for view, chave in (
        ("treino_validacao_1", "validacao_1_treino"),
        ("validacao_1", "validacao_1_validacao"),
        ("treino_validacao_2", "validacao_2_treino"),
        ("validacao_2", "validacao_2_validacao"),
        ("treino_final", "final_treino"),
        ("teste_final", "final_teste"),
    ):
        quantidade = connection.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        if int(quantidade) != resumo.get(chave, 0):
            raise RuntimeError(f"a view {view} não coincide com as partições")


def gravar_banco(
    temporario: Path,
    entradas: dict[str, Path],
    execucoes: dict[str, dict[str, object]],
    hashes: dict[str, str],
    config: Configuracao,
    contexto: ContextoTemporal,
    eventos: pd.DataFrame,
    auditoria: pd.DataFrame,
    particoes: pd.DataFrame,
    catalogo: pd.DataFrame,
    resumo: dict[str, int],
) -> None:
    connection = sqlite3.connect(temporario)
    try:
        criar_schema(connection)
        parametros = {
            "inicio_series": contexto.inicio_series.isoformat(),
            "inicio_validacao_1": config.inicio_validacao_1.isoformat(),
            "inicio_validacao_2": config.inicio_validacao_2.isoformat(),
            "corte_treino_final": contexto.corte_treino.isoformat(),
            "fim_teste_final": contexto.fim_teste.isoformat(),
            "janela_rotulo_datas_segunda_sexta": contexto.janela_rotulo_dias,
            "janelas_historicas_datas_segunda_sexta": list(JANELAS_HISTORICAS),
            "max_lag_retorno": MAX_LAG_RETORNO,
            "unidade_analise": (
                "produto, origem, destino e data do choque"
            ),
            "alvo": (
                "1 confirmado; 0 nao confirmado; nulo somente para auditoria"
            ),
            "campos_futuros": (
                "armazenados somente em auditoria_eventos"
            ),
            "purga_fronteiras": (
                "remove da divisao eventos cuja janela do rotulo ultrapassa "
                "o fim do respectivo periodo"
            ),
            "uso_preprocessamento": (
                "fonte indireta validada por hash; atributos diarios usam "
                "as agregacoes reproduzidas no banco de Granger"
            ),
        }
        connection.execute(
            """
            INSERT INTO execucoes_modelagem (
                id, schema_version, iniciada_em, parametros_json, status
            ) VALUES (1, ?, ?, ?, 'em_andamento')
            """,
            (
                SCHEMA_VERSION,
                agora_iso(),
                json.dumps(parametros, ensure_ascii=False, sort_keys=True),
            ),
        )
        fontes = []
        for nome in ("preprocessamento", "granger", "domino"):
            fontes.append(
                (
                    nome,
                    str(entradas[nome].resolve()),
                    hashes[nome],
                    int(execucoes[nome]["id"]),
                    int(execucoes[nome]["schema_version"]),
                )
            )
        connection.executemany(
            """
            INSERT INTO fontes_dados (
                nome, caminho, sha256, execucao_id, schema_version
            ) VALUES (?, ?, ?, ?, ?)
            """,
            fontes,
        )
        eventos.to_sql(
            "eventos_modelagem",
            connection,
            if_exists="append",
            index=False,
        )
        auditoria.to_sql(
            "auditoria_eventos",
            connection,
            if_exists="append",
            index=False,
        )
        particoes.to_sql(
            "particoes_modelagem",
            connection,
            if_exists="append",
            index=False,
        )
        catalogo.to_sql(
            "catalogo_atributos",
            connection,
            if_exists="append",
            index=False,
        )
        connection.executemany(
            "INSERT INTO resumo_modelagem (chave, valor) VALUES (?, ?)",
            [(chave, str(valor)) for chave, valor in sorted(resumo.items())],
        )
        connection.execute(
            """
            UPDATE execucoes_modelagem
            SET concluida_em = ?, status = 'concluida'
            WHERE id = 1
            """,
            (agora_iso(),),
        )
        connection.commit()
        validar_saida(connection, resumo)
        connection.commit()
    finally:
        connection.close()


def executar(args: argparse.Namespace) -> Path:
    config = validar_argumentos(args)
    output = args.output_db.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print("[1/6] Validando a cadeia dos três bancos de entrada...", flush=True)
    (
        execucao_preprocessamento,
        execucao_granger,
        execucao_domino,
        hashes,
        contexto,
    ) = validar_fontes(
        args.preprocessamento_db,
        args.granger_db,
        args.domino_db,
    )
    validar_configuracao_temporal(config, contexto)

    print("[2/6] Carregando eventos e séries temporais...", flush=True)
    eventos, auditoria = carregar_eventos(args.domino_db)
    series = carregar_series(args.granger_db)

    print("[3/6] Construindo atributos disponíveis até cada choque...", flush=True)
    series = preparar_atributos_series(series)
    eventos_modelagem = construir_eventos_modelagem(eventos, series)
    auditoria = preparar_auditoria(
        auditoria,
        eventos_modelagem,
        contexto.janela_rotulo_dias,
    )

    print("[4/6] Construindo as divisões temporais com purga...", flush=True)
    particoes = construir_particoes(eventos_modelagem, config, contexto)
    catalogo = construir_catalogo()
    resumo = gerar_resumo(eventos_modelagem, particoes)

    print("[5/6] Gravando o banco temporário...", flush=True)
    descritor, nome_temporario = tempfile.mkstemp(
        prefix=f".{output.stem}-",
        suffix=".tmp.sqlite",
        dir=output.parent,
    )
    os.close(descritor)
    temporario = Path(nome_temporario)
    try:
        gravar_banco(
            temporario,
            {
                "preprocessamento": args.preprocessamento_db,
                "granger": args.granger_db,
                "domino": args.domino_db,
            },
            {
                "preprocessamento": execucao_preprocessamento,
                "granger": execucao_granger,
                "domino": execucao_domino,
            },
            hashes,
            config,
            contexto,
            eventos_modelagem,
            auditoria,
            particoes,
            catalogo,
            resumo,
        )
        print("[6/6] Validando e publicando o banco final...", flush=True)
        temporario.replace(output)
    except Exception:
        temporario.unlink(missing_ok=True)
        raise

    print(f"Base de modelagem criada em: {output}", flush=True)
    for chave, valor in sorted(resumo.items()):
        print(f"  {chave}: {valor:,}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    executar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
