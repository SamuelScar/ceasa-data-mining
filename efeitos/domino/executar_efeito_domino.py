"""Caracteriza ocorrências do Efeito Dominó a partir das relações de Granger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
DEFAULT_ALTA_MINIMA = 0.05
DEFAULT_PERCENTIL = 0.90
DEFAULT_JANELA_DIAS = 5


@dataclass(frozen=True)
class Configuracao:
    alta_minima: float
    percentil: float
    janela_dias: int


def agora_iso() -> str:
    return datetime.now(UTC).isoformat()


def valor_ambiente(nome: str, padrao: str) -> str:
    valor = os.getenv(nome)
    return valor if valor not in (None, "") else padrao


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Localiza choques de preço nas relações significativas de Granger e "
            "classifica se houve propagação para o mercado de destino."
        )
    )
    parser.add_argument("input_db", type=Path, help="Banco gerado pela etapa de Granger")
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("resultados/efeitos/domino/efeito_domino.sqlite"),
        help="SQLite que receberá os eventos caracterizados",
    )
    parser.add_argument(
        "--alta-minima",
        type=float,
        default=float(
            valor_ambiente("DOMINO_ALTA_MINIMA", str(DEFAULT_ALTA_MINIMA))
        ),
        help="Alta percentual mínima para um movimento ser considerado forte",
    )
    parser.add_argument(
        "--percentil",
        type=float,
        default=float(valor_ambiente("DOMINO_PERCENTIL", str(DEFAULT_PERCENTIL))),
        help="Percentil histórico mínimo da alta, calculado somente no treino",
    )
    parser.add_argument(
        "--janela-dias",
        type=int,
        default=int(
            valor_ambiente("DOMINO_JANELA_DIAS", str(DEFAULT_JANELA_DIAS))
        ),
        help="Quantidade de dias úteis para observar a propagação",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Substitui a saída somente depois de uma execução válida",
    )
    return parser


def validar_argumentos(args: argparse.Namespace) -> Configuracao:
    if not args.input_db.is_file():
        raise FileNotFoundError(f"banco de Granger não encontrado: {args.input_db}")
    if args.input_db.resolve() == args.output_db.resolve():
        raise ValueError("o banco de entrada e o banco de saída devem ser diferentes")
    if not 0 < args.alta_minima < 1:
        raise ValueError("--alta-minima deve estar entre 0 e 1")
    if not 0 < args.percentil < 1:
        raise ValueError("--percentil deve estar entre 0 e 1")
    if args.janela_dias < 1:
        raise ValueError("--janela-dias deve ser positivo")
    if args.output_db.exists() and not args.overwrite:
        raise FileExistsError(
            f"a saída já existe: {args.output_db}; use --overwrite para substituí-la"
        )
    return Configuracao(
        alta_minima=args.alta_minima,
        percentil=args.percentil,
        janela_dias=args.janela_dias,
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


def validar_banco_granger(connection: sqlite3.Connection) -> sqlite3.Row:
    tabelas = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    obrigatorias = {
        "execucoes_granger",
        "series_temporais",
        "resultados_granger",
    }
    ausentes = obrigatorias - tabelas
    if ausentes:
        raise ValueError(
            "banco de Granger incompleto; estruturas ausentes: "
            + ", ".join(sorted(ausentes))
        )

    execucao = connection.execute(
        """
        SELECT id, schema_version, status, parametros_json
        FROM execucoes_granger
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if execucao is None or execucao["status"] != "concluida":
        raise ValueError("o banco de Granger não possui execução concluída")
    return execucao


def carregar_relacoes(connection: sqlite3.Connection) -> pd.DataFrame:
    relacoes = pd.read_sql_query(
        """
        SELECT
            id AS resultado_granger_id,
            produto,
            origem,
            destino,
            lag_selecionado,
            estatistica_f,
            p_valor,
            q_valor_fdr
        FROM resultados_granger
        WHERE status = 'testado' AND significativo_fdr = 1
        ORDER BY produto, origem, destino
        """,
        connection,
    )
    if relacoes.empty:
        raise ValueError("não existem relações significativas após a correção FDR")
    relacoes.insert(0, "id", np.arange(1, len(relacoes) + 1, dtype=int))
    return relacoes


def carregar_series(connection: sqlite3.Connection) -> pd.DataFrame:
    series = pd.read_sql_query(
        """
        SELECT
            produto,
            mercado,
            data_cotacao,
            periodo,
            preco_kg,
            observado,
            preenchido,
            retorno_log
        FROM series_temporais
        ORDER BY produto, mercado, data_cotacao
        """,
        connection,
    )
    if series.empty:
        raise ValueError("o banco de Granger não contém séries temporais")
    series["data_cotacao"] = pd.to_datetime(series["data_cotacao"], errors="raise")
    return series


def calcular_limites(series: pd.DataFrame, config: Configuracao) -> pd.DataFrame:
    treino = series.loc[
        (series["periodo"] == "treino")
        & (series["observado"] == 1)
        & series["retorno_log"].notna(),
        ["produto", "mercado", "retorno_log"],
    ]
    minimo_log = math.log1p(config.alta_minima)
    linhas: list[dict[str, object]] = []
    for (produto, mercado), grupo in treino.groupby(
        ["produto", "mercado"], sort=True
    ):
        percentil_log = float(grupo["retorno_log"].quantile(config.percentil))
        limite_log = max(minimo_log, percentil_log)
        linhas.append(
            {
                "produto": produto,
                "mercado": mercado,
                "observacoes_treino": int(len(grupo)),
                "percentil_configurado": config.percentil,
                "percentil_log": percentil_log,
                "percentil_percentual": math.expm1(percentil_log),
                "alta_minima_percentual": config.alta_minima,
                "limite_log": limite_log,
                "limite_percentual": math.expm1(limite_log),
            }
        )
    limites = pd.DataFrame(linhas)
    if limites.empty:
        raise ValueError("não foi possível calcular limites usando o período de treino")
    limites.insert(0, "id", np.arange(1, len(limites) + 1, dtype=int))
    return limites


def exibir_progresso(atual: int, total: int, descricao: str) -> None:
    largura = 30
    preenchido = round(largura * atual / total) if total else largura
    barra = "#" * preenchido + "-" * (largura - preenchido)
    percentual = 100 * atual / total if total else 100
    print(
        f"[{barra}] {percentual:6.2f}% | {descricao}",
        flush=True,
    )


def caracterizar_eventos(
    series: pd.DataFrame,
    relacoes: pd.DataFrame,
    limites: pd.DataFrame,
    config: Configuracao,
) -> pd.DataFrame:
    mapa_series = {
        chave: grupo.sort_values("data_cotacao").reset_index(drop=True)
        for chave, grupo in series.groupby(["produto", "mercado"], sort=False)
    }
    mapa_limites = {
        (row.produto, row.mercado): float(row.limite_log)
        for row in limites.itertuples(index=False)
    }
    eventos: list[dict[str, object]] = []
    total = len(relacoes)

    for posicao, relacao in enumerate(relacoes.itertuples(index=False), start=1):
        chave_origem = (relacao.produto, relacao.origem)
        chave_destino = (relacao.produto, relacao.destino)
        origem = mapa_series.get(chave_origem)
        destino = mapa_series.get(chave_destino)
        limite_origem = mapa_limites.get(chave_origem)
        limite_destino = mapa_limites.get(chave_destino)
        if (
            origem is None
            or destino is None
            or limite_origem is None
            or limite_destino is None
        ):
            raise ValueError(
                "série ou limite ausente para a relação "
                f"{relacao.produto}: {relacao.origem} -> {relacao.destino}"
            )

        destino_por_data = destino.set_index("data_cotacao", drop=False)
        choques = origem.loc[
            (origem["observado"] == 1)
            & origem["retorno_log"].notna()
            & (origem["retorno_log"] >= limite_origem)
        ]

        for choque in choques.itertuples(index=False):
            data_choque = choque.data_cotacao
            base = destino_por_data.loc[data_choque] if data_choque in destino_por_data.index else None
            registro: dict[str, object] = {
                "relacao_id": int(relacao.id),
                "produto": relacao.produto,
                "origem": relacao.origem,
                "destino": relacao.destino,
                "data_choque": data_choque.strftime("%Y-%m-%d"),
                "periodo": choque.periodo,
                "preco_origem": float(choque.preco_kg),
                "alta_origem_log": float(choque.retorno_log),
                "alta_origem_percentual": math.expm1(float(choque.retorno_log)),
                "limite_origem_percentual": math.expm1(limite_origem),
                "preco_destino_inicial": None,
                "limite_destino_percentual": math.expm1(limite_destino),
                "dias_observados_destino": 0,
                "maior_alta_destino_percentual": None,
                "data_maior_alta_destino": None,
                "data_resposta": None,
                "dias_ate_resposta": None,
                "efeito_domino": None,
                "status": None,
            }

            if isinstance(base, pd.DataFrame):
                raise ValueError("a série de destino possui datas duplicadas")
            if base is None or int(base["observado"]) != 1:
                registro["status"] = "sem_base_observada_destino"
                eventos.append(registro)
                continue
            preco_destino_base = base["preco_kg"]
            if (
                pd.isna(preco_destino_base)
                or not np.isfinite(float(preco_destino_base))
                or float(preco_destino_base) <= 0
            ):
                registro["status"] = "sem_preco_base_destino"
                eventos.append(registro)
                continue

            preco_base = float(preco_destino_base)
            registro["preco_destino_inicial"] = preco_base
            retorno_destino_atual = base["retorno_log"]
            if (
                pd.notna(retorno_destino_atual)
                and float(retorno_destino_atual) >= limite_destino
            ):
                registro["status"] = "movimento_simultaneo"
                eventos.append(registro)
                continue

            fim_janela = data_choque + pd.offsets.BDay(config.janela_dias)
            futuras = destino.loc[
                (destino["data_cotacao"] > data_choque)
                & (destino["data_cotacao"] <= fim_janela)
                & (destino["periodo"] == choque.periodo)
                & (destino["observado"] == 1)
                & destino["preco_kg"].notna()
                & (destino["preco_kg"] > 0)
            ].copy()
            registro["dias_observados_destino"] = int(len(futuras))
            if futuras.empty:
                registro["status"] = "sem_cobertura_destino"
                eventos.append(registro)
                continue

            futuras["alta_acumulada_log"] = np.log(futuras["preco_kg"] / preco_base)
            indice_maximo = futuras["alta_acumulada_log"].idxmax()
            maior = futuras.loc[indice_maximo]
            maior_log = float(maior["alta_acumulada_log"])
            registro["maior_alta_destino_percentual"] = math.expm1(maior_log)
            registro["data_maior_alta_destino"] = maior["data_cotacao"].strftime(
                "%Y-%m-%d"
            )

            respostas = futuras.loc[futuras["alta_acumulada_log"] >= limite_destino]
            if respostas.empty:
                registro["efeito_domino"] = 0
                registro["status"] = "nao_confirmado"
            else:
                resposta = respostas.iloc[0]
                data_resposta = resposta["data_cotacao"]
                registro["data_resposta"] = data_resposta.strftime("%Y-%m-%d")
                registro["dias_ate_resposta"] = int(
                    len(pd.bdate_range(data_choque + pd.offsets.BDay(1), data_resposta))
                )
                registro["efeito_domino"] = 1
                registro["status"] = "confirmado"
            eventos.append(registro)

        exibir_progresso(
            posicao,
            total,
            f"relação {posicao}/{total}: {relacao.produto} | "
            f"{relacao.origem} -> {relacao.destino}",
        )

    frame = pd.DataFrame(eventos)
    if frame.empty:
        raise ValueError("nenhum choque de origem foi encontrado com os limites atuais")
    frame.insert(0, "id", np.arange(1, len(frame) + 1, dtype=int))
    return frame


def criar_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_efeito_domino (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT,
            banco_origem TEXT NOT NULL,
            banco_origem_sha256 TEXT NOT NULL,
            execucao_granger_id INTEGER NOT NULL,
            parametros_json TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE limites_altas (
            id INTEGER PRIMARY KEY,
            produto TEXT NOT NULL,
            mercado TEXT NOT NULL,
            observacoes_treino INTEGER NOT NULL,
            percentil_configurado REAL NOT NULL,
            percentil_log REAL NOT NULL,
            percentil_percentual REAL NOT NULL,
            alta_minima_percentual REAL NOT NULL,
            limite_log REAL NOT NULL,
            limite_percentual REAL NOT NULL,
            UNIQUE (produto, mercado)
        );

        CREATE TABLE relacoes_analisadas (
            id INTEGER PRIMARY KEY,
            resultado_granger_id INTEGER NOT NULL,
            produto TEXT NOT NULL,
            origem TEXT NOT NULL,
            destino TEXT NOT NULL,
            lag_selecionado INTEGER NOT NULL,
            estatistica_f REAL NOT NULL,
            p_valor REAL NOT NULL,
            q_valor_fdr REAL NOT NULL,
            UNIQUE (produto, origem, destino)
        );

        CREATE TABLE eventos_efeito_domino (
            id INTEGER PRIMARY KEY,
            relacao_id INTEGER NOT NULL,
            produto TEXT NOT NULL,
            origem TEXT NOT NULL,
            destino TEXT NOT NULL,
            data_choque TEXT NOT NULL,
            periodo TEXT NOT NULL,
            preco_origem REAL NOT NULL,
            alta_origem_log REAL NOT NULL,
            alta_origem_percentual REAL NOT NULL,
            limite_origem_percentual REAL NOT NULL,
            preco_destino_inicial REAL,
            limite_destino_percentual REAL NOT NULL,
            dias_observados_destino INTEGER NOT NULL,
            maior_alta_destino_percentual REAL,
            data_maior_alta_destino TEXT,
            data_resposta TEXT,
            dias_ate_resposta INTEGER,
            efeito_domino INTEGER CHECK (efeito_domino IN (0, 1)),
            status TEXT NOT NULL,
            FOREIGN KEY (relacao_id) REFERENCES relacoes_analisadas(id)
        );

        CREATE TABLE resumo_efeito_domino (
            chave TEXT PRIMARY KEY,
            valor TEXT NOT NULL
        );

        CREATE INDEX idx_eventos_relacao_data
            ON eventos_efeito_domino(relacao_id, data_choque);
        CREATE INDEX idx_eventos_periodo_resultado
            ON eventos_efeito_domino(periodo, efeito_domino);
        CREATE INDEX idx_eventos_status
            ON eventos_efeito_domino(status);

        CREATE VIEW eventos_classificaveis AS
        SELECT *
        FROM eventos_efeito_domino
        WHERE efeito_domino IS NOT NULL;

        CREATE VIEW efeitos_domino_confirmados AS
        SELECT *
        FROM eventos_efeito_domino
        WHERE efeito_domino = 1;
        """
    )


def gerar_resumo(relacoes: pd.DataFrame, eventos: pd.DataFrame) -> dict[str, int]:
    classificaveis = eventos["efeito_domino"].notna()
    resumo = {
        "relacoes_granger_analisadas": int(len(relacoes)),
        "choques_origem_encontrados": int(len(eventos)),
        "eventos_classificaveis": int(classificaveis.sum()),
        "efeitos_domino_confirmados": int((eventos["efeito_domino"] == 1).sum()),
        "efeitos_domino_nao_confirmados": int((eventos["efeito_domino"] == 0).sum()),
        "movimentos_simultaneos": int((eventos["status"] == "movimento_simultaneo").sum()),
        "eventos_sem_base_destino": int(
            eventos["status"].isin(
                ["sem_base_observada_destino", "sem_preco_base_destino"]
            ).sum()
        ),
        "eventos_sem_cobertura_destino": int(
            (eventos["status"] == "sem_cobertura_destino").sum()
        ),
    }
    for periodo in ("treino", "teste"):
        filtro = eventos["periodo"] == periodo
        resumo[f"eventos_classificaveis_{periodo}"] = int(
            (filtro & classificaveis).sum()
        )
        resumo[f"efeitos_confirmados_{periodo}"] = int(
            (filtro & (eventos["efeito_domino"] == 1)).sum()
        )
    return resumo


def validar_saida(connection: sqlite3.Connection, resumo: dict[str, int]) -> None:
    integridade = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integridade != "ok":
        raise RuntimeError(f"falha de integridade no SQLite: {integridade}")

    contagens = connection.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN efeito_domino IS NOT NULL THEN 1 ELSE 0 END) AS classificaveis,
            SUM(CASE WHEN efeito_domino = 1 THEN 1 ELSE 0 END) AS confirmados,
            SUM(CASE WHEN efeito_domino = 0 THEN 1 ELSE 0 END) AS nao_confirmados,
            SUM(CASE WHEN efeito_domino = 1 AND status <> 'confirmado' THEN 1 ELSE 0 END)
                AS confirmados_inconsistentes,
            SUM(CASE WHEN efeito_domino = 0 AND status <> 'nao_confirmado' THEN 1 ELSE 0 END)
                AS negativos_inconsistentes
        FROM eventos_efeito_domino
        """
    ).fetchone()
    esperadas = (
        resumo["choques_origem_encontrados"],
        resumo["eventos_classificaveis"],
        resumo["efeitos_domino_confirmados"],
        resumo["efeitos_domino_nao_confirmados"],
    )
    observadas = tuple(int(contagens[i] or 0) for i in range(4))
    if observadas != esperadas:
        raise RuntimeError("as contagens gravadas não coincidem com o resumo")
    if int(contagens[4] or 0) or int(contagens[5] or 0):
        raise RuntimeError("existem eventos com classificação e status incompatíveis")


def gravar_banco(
    temporario: Path,
    origem: Path,
    execucao_granger: sqlite3.Row,
    config: Configuracao,
    relacoes: pd.DataFrame,
    limites: pd.DataFrame,
    eventos: pd.DataFrame,
    resumo: dict[str, int],
) -> None:
    connection = sqlite3.connect(temporario)
    try:
        criar_schema(connection)
        parametros = {
            "alta_minima": config.alta_minima,
            "percentil": config.percentil,
            "janela_dias_uteis": config.janela_dias,
            "regra": "maximo entre alta minima e percentil calculado no treino",
            "relacoes": "somente significativas apos FDR no Granger",
            "movimentos_simultaneos": "desconsiderados",
            "eventos": "somente cotacoes observadas",
        }
        connection.execute(
            """
            INSERT INTO execucoes_efeito_domino (
                id, schema_version, iniciada_em, banco_origem,
                banco_origem_sha256, execucao_granger_id, parametros_json, status
            ) VALUES (1, ?, ?, ?, ?, ?, ?, 'em_andamento')
            """,
            (
                SCHEMA_VERSION,
                agora_iso(),
                str(origem),
                sha256_arquivo(origem),
                int(execucao_granger["id"]),
                json.dumps(parametros, ensure_ascii=False, sort_keys=True),
            ),
        )
        limites.to_sql("limites_altas", connection, if_exists="append", index=False)
        relacoes.to_sql(
            "relacoes_analisadas", connection, if_exists="append", index=False
        )
        eventos.to_sql(
            "eventos_efeito_domino", connection, if_exists="append", index=False
        )
        connection.executemany(
            "INSERT INTO resumo_efeito_domino (chave, valor) VALUES (?, ?)",
            [(chave, str(valor)) for chave, valor in sorted(resumo.items())],
        )
        connection.execute(
            """
            UPDATE execucoes_efeito_domino
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

    print("[1/4] Validando o banco de Granger...", flush=True)
    source = abrir_somente_leitura(args.input_db)
    try:
        execucao_granger = validar_banco_granger(source)
        print("[2/4] Carregando relações e séries temporais...", flush=True)
        relacoes = carregar_relacoes(source)
        series = carregar_series(source)
    finally:
        source.close()

    print("[3/4] Calculando limites e caracterizando os eventos...", flush=True)
    limites = calcular_limites(series, config)
    eventos = caracterizar_eventos(series, relacoes, limites, config)
    resumo = gerar_resumo(relacoes, eventos)

    print("[4/4] Gravando e validando o banco de saída...", flush=True)
    descritor, nome_temporario = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".tmp.sqlite", dir=output.parent
    )
    os.close(descritor)
    temporario = Path(nome_temporario)
    try:
        gravar_banco(
            temporario,
            args.input_db,
            execucao_granger,
            config,
            relacoes,
            limites,
            eventos,
            resumo,
        )
        temporario.replace(output)
    except Exception:
        temporario.unlink(missing_ok=True)
        raise

    print(f"Banco do Efeito Dominó criado em: {output}", flush=True)
    for chave, valor in sorted(resumo.items()):
        print(f"  {chave}: {valor:,}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = criar_parser().parse_args(argv)
    executar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
