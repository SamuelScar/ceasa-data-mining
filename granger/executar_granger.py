"""Constrói séries temporais e executa testes bivariados de Granger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as distribuicao_f
from statsmodels.tsa.stattools import adfuller, kpss


SCHEMA_VERSION = 1

MERCADOS_FIXOS = (
    "recife",
    "ceasa-grande-vitoria",
    "curitiba",
    "maringa",
    "cascavel",
    "foz-do-iguacu",
    "londrina",
)

PRODUTOS_FIXOS = (
    "batata",
    "tomate",
    "cebola",
    "cenoura",
    "morango",
    "beterraba",
    "melao",
    "pimentao verde",
    "pimentao",
    "batata doce",
    "banana prata",
    "quiabo",
)

TIPOS_CONVERSAO_ACEITOS = (
    "exata",
    "calculada",
    "padronizada",
    "inferida_alta_confianca",
)

MINIMO_RETORNOS_ESTACIONARIEDADE = 30


@dataclass(frozen=True)
class MetadadosPreprocessamento:
    execucao_id: int
    schema_version: int
    banco_origem: str
    parametros: dict[str, Any]


@dataclass(frozen=True)
class ResultadoEstacionariedade:
    n_observacoes: int
    estatistica_adf: float | None
    p_valor_adf: float | None
    defasagens_adf: int | None
    estatistica_kpss: float | None
    p_valor_kpss: float | None
    defasagens_kpss: int | None
    estacionaria: bool
    status: str


def data_iso(value: str) -> date:
    """Converte uma data ISO informada pela linha de comando."""
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"data inválida: {value!r}; use o formato AAAA-MM-DD"
        ) from error


def inteiro_ambiente(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value is not None else default
    except ValueError as error:
        raise ValueError(f"{name} deve ser um número inteiro") from error


def real_ambiente(name: str, default: float) -> float:
    value = os.getenv(name)
    try:
        return float(value) if value is not None else default
    except ValueError as error:
        raise ValueError(f"{name} deve ser um número real") from error


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Constrói séries temporais e executa o teste de Granger."
    )
    parser.add_argument("database", type=Path, help="SQLite pré-processado")
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("resultados/granger/granger.sqlite"),
        help="SQLite que receberá as séries e os resultados",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="autoriza substituir atomicamente uma saída existente",
    )
    parser.add_argument(
        "--data-inicio",
        type=data_iso,
        default=data_iso(os.getenv("GRANGER_DATA_INICIO", "2022-01-04")),
    )
    parser.add_argument(
        "--data-corte-treino",
        type=data_iso,
        default=data_iso(
            os.getenv("GRANGER_DATA_CORTE_TREINO", "2025-10-31")
        ),
    )
    parser.add_argument(
        "--data-fim-teste",
        type=data_iso,
        default=data_iso(os.getenv("GRANGER_DATA_FIM_TESTE", "2026-10-31")),
    )
    parser.add_argument(
        "--min-dias-comuns",
        type=int,
        default=inteiro_ambiente("GRANGER_MIN_DIAS_COMUNS", 400),
    )
    parser.add_argument(
        "--ffill-limite",
        type=int,
        default=inteiro_ambiente("GRANGER_FFILL_LIMITE", 3),
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=inteiro_ambiente("GRANGER_MAX_LAG", 10),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=real_ambiente("GRANGER_ALPHA", 0.05),
    )
    return parser


def validar_argumentos(args: argparse.Namespace) -> None:
    if not args.data_inicio <= args.data_corte_treino < args.data_fim_teste:
        raise ValueError(
            "as datas devem respeitar início <= corte de treino < fim do teste"
        )
    if args.min_dias_comuns < 1:
        raise ValueError("--min-dias-comuns deve ser pelo menos um")
    if args.ffill_limite < 0:
        raise ValueError("--ffill-limite não pode ser negativo")
    if args.max_lag < 1:
        raise ValueError("--max-lag deve ser pelo menos um")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha deve estar entre zero e um")

    database = args.database.resolve()
    output = args.output_db.resolve()
    if database == output:
        raise ValueError("o banco de saída deve ser diferente do pré-processado")
    if not database.is_file():
        raise FileNotFoundError(f"banco pré-processado não encontrado: {database}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"o banco de saída já existe: {output}; use --overwrite"
        )


def abrir_somente_leitura(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def sha256_arquivo(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validar_banco_preprocessado(
    connection: sqlite3.Connection,
    database: Path,
    data_corte_treino: date,
) -> MetadadosPreprocessamento:
    objetos = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    obrigatorios = {
        "cotacoes_convertidas_kg",
        "execucoes_preprocessamento",
    }
    ausentes = sorted(obrigatorios - objetos)
    if ausentes:
        raise ValueError(
            "banco pré-processado incompatível; estruturas ausentes: "
            + ", ".join(ausentes)
        )

    row = connection.execute(
        """
        SELECT id, schema_version, banco_origem, parametros_json, status
        FROM execucoes_preprocessamento
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None or row["status"] != "concluida":
        raise ValueError("o pré-processamento não possui execução concluída")

    parametros = json.loads(row["parametros_json"])
    limite = parametros.get("data_limite_inferencia")
    esperado = data_corte_treino.isoformat()
    if limite != esperado:
        raise ValueError(
            "o limite de inferência do pré-processamento deve coincidir com "
            f"o corte de treino: esperado {esperado}, encontrado {limite!r}"
        )

    banco_origem = Path(row["banco_origem"])
    if not banco_origem.is_absolute():
        banco_origem = Path.cwd() / banco_origem
    if (
        banco_origem.is_file()
        and banco_origem.stat().st_mtime_ns > database.stat().st_mtime_ns
    ):
        raise ValueError(
            "o banco bruto é mais recente que o banco pré-processado; "
            "refaça o pré-processamento antes do Granger"
        )

    return MetadadosPreprocessamento(
        execucao_id=int(row["id"]),
        schema_version=int(row["schema_version"]),
        banco_origem=row["banco_origem"],
        parametros=parametros,
    )


def carregar_metadados_mercados(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in MERCADOS_FIXOS)
    query = f"""
        SELECT DISTINCT entreposto_slug, ceasa_slug, uf
        FROM cotacoes_convertidas_kg
        WHERE entreposto_slug IN ({placeholders})
        ORDER BY entreposto_slug
    """
    metadata = pd.read_sql_query(query, connection, params=MERCADOS_FIXOS)
    if metadata.empty:
        raise ValueError("nenhum entreposto fixado foi encontrado")

    conflicts = metadata.groupby("entreposto_slug").size()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            "entrepostos associados a mais de uma CEASA ou UF: "
            + ", ".join(conflicts.index)
        )

    missing = sorted(set(MERCADOS_FIXOS) - set(metadata["entreposto_slug"]))
    if missing:
        raise ValueError(
            "entrepostos fixados ausentes do banco: " + ", ".join(missing)
        )
    return metadata


def carregar_cotacoes(
    connection: sqlite3.Connection,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, int]:
    markets = ",".join("?" for _ in MERCADOS_FIXOS)
    products = ",".join("?" for _ in PRODUTOS_FIXOS)
    conversions = ",".join("?" for _ in TIPOS_CONVERSAO_ACEITOS)
    query = f"""
        SELECT
            produto_chave_textual AS produto,
            ceasa_slug,
            uf,
            entreposto_slug AS mercado,
            data_cotacao,
            preco_comum_kg,
            tipo_conversao
        FROM cotacoes_convertidas_kg
        WHERE outlier_suspeito = 0
          AND preco_comum_kg IS NOT NULL
          AND preco_comum_kg > 0
          AND data_cotacao BETWEEN ? AND ?
          AND entreposto_slug IN ({markets})
          AND produto_chave_textual IN ({products})
          AND tipo_conversao IN ({conversions})
        ORDER BY produto_chave_textual, entreposto_slug, data_cotacao
    """
    parameters = (
        args.data_inicio.isoformat(),
        args.data_fim_teste.isoformat(),
        *MERCADOS_FIXOS,
        *PRODUTOS_FIXOS,
        *TIPOS_CONVERSAO_ACEITOS,
    )
    quotes = pd.read_sql_query(
        query,
        connection,
        params=parameters,
        parse_dates=["data_cotacao"],
    )
    if quotes.empty:
        raise ValueError("o recorte fixado não encontrou cotações utilizáveis")

    weekend = quotes["data_cotacao"].dt.weekday >= 5
    weekend_count = int(weekend.sum())
    quotes = quotes.loc[~weekend].copy()
    return quotes, weekend_count


def consolidar_mediana_diaria(quotes: pd.DataFrame) -> pd.DataFrame:
    keys = ["produto", "ceasa_slug", "uf", "mercado", "data_cotacao"]
    daily = (
        quotes.groupby(keys, as_index=False, observed=True)
        .agg(
            preco_kg_observado=("preco_comum_kg", "median"),
            n_cotacoes=("preco_comum_kg", "size"),
            n_exatas=("tipo_conversao", lambda s: int((s == "exata").sum())),
            n_calculadas=(
                "tipo_conversao",
                lambda s: int((s == "calculada").sum()),
            ),
            n_padronizadas=(
                "tipo_conversao",
                lambda s: int((s == "padronizada").sum()),
            ),
            n_inferidas=(
                "tipo_conversao",
                lambda s: int((s == "inferida_alta_confianca").sum()),
            ),
        )
        .sort_values(keys)
    )
    return daily


def identificar_segmentos(disponivel: pd.Series) -> pd.Series:
    inicio = disponivel & ~disponivel.shift(1, fill_value=False)
    segmentos = inicio.cumsum().where(disponivel)
    return segmentos.astype("Int64")


def construir_series_temporais(
    daily: pd.DataFrame,
    market_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    calendar = pd.bdate_range(args.data_inicio, args.data_fim_teste)
    metadata = market_metadata.set_index("entreposto_slug").to_dict("index")
    frames: list[pd.DataFrame] = []
    series_map: dict[tuple[str, str], pd.DataFrame] = {}

    for product in PRODUTOS_FIXOS:
        for market in MERCADOS_FIXOS:
            subset = daily[
                (daily["produto"] == product) & (daily["mercado"] == market)
            ].set_index("data_cotacao")

            frame = pd.DataFrame(index=calendar)
            frame.index.name = "data_cotacao"
            frame["preco_kg_observado"] = subset[
                "preco_kg_observado"
            ].reindex(calendar)
            frame["preco_kg"] = frame["preco_kg_observado"].ffill(
                limit=args.ffill_limite
            )

            for column in (
                "n_cotacoes",
                "n_exatas",
                "n_calculadas",
                "n_padronizadas",
                "n_inferidas",
            ):
                frame[column] = (
                    subset[column].reindex(calendar).fillna(0).astype(int)
                )

            frame["observado"] = frame["preco_kg_observado"].notna().astype(int)
            frame["preenchido"] = (
                frame["preco_kg"].notna()
                & frame["preco_kg_observado"].isna()
            ).astype(int)
            available = frame["preco_kg"].notna()
            frame["segmento_serie"] = identificar_segmentos(available)
            returns = np.log(frame["preco_kg"]).diff()
            same_segment = frame["segmento_serie"].eq(
                frame["segmento_serie"].shift(1)
            )
            frame["retorno_log"] = returns.where(available & same_segment)
            frame["periodo"] = np.where(
                frame.index.date <= args.data_corte_treino,
                "treino",
                "teste",
            )
            frame["produto"] = product
            frame["mercado"] = market
            frame["ceasa_slug"] = metadata[market]["ceasa_slug"]
            frame["uf"] = metadata[market]["uf"]

            series_map[(product, market)] = frame.copy()
            frames.append(frame.reset_index())

    result = pd.concat(frames, ignore_index=True)
    result["data_cotacao"] = result["data_cotacao"].dt.strftime("%Y-%m-%d")
    columns = [
        "produto",
        "ceasa_slug",
        "uf",
        "mercado",
        "data_cotacao",
        "periodo",
        "preco_kg_observado",
        "preco_kg",
        "n_cotacoes",
        "n_exatas",
        "n_calculadas",
        "n_padronizadas",
        "n_inferidas",
        "observado",
        "preenchido",
        "segmento_serie",
        "retorno_log",
    ]
    return result[columns], series_map


def construir_pares(
    series_map: dict[tuple[str, str], pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    pair_rows: list[dict[str, Any]] = []
    pair_series_rows: list[pd.DataFrame] = []
    pair_frames: dict[int, pd.DataFrame] = {}
    pair_id = 0

    for product in PRODUTOS_FIXOS:
        for market_a, market_b in combinations(sorted(MERCADOS_FIXOS), 2):
            pair_id += 1
            a = series_map[(product, market_a)]
            b = series_map[(product, market_b)]
            train = a.index.date <= args.data_corte_treino
            observed_common = int(
                (a["observado"].eq(1) & b["observado"].eq(1) & train).sum()
            )
            available_common = int(
                (
                    a["preco_kg"].notna()
                    & b["preco_kg"].notna()
                    & train
                ).sum()
            )
            eligible = observed_common >= args.min_dias_comuns
            if eligible:
                reason = "elegivel"
            elif not a["observado"].any() or not b["observado"].any():
                reason = "serie_ausente"
            else:
                reason = "dias_observados_insuficientes"

            n_segments_train = 0
            longest_segment_train = 0
            if eligible:
                pair = pd.DataFrame(index=a.index)
                pair.index.name = "data_cotacao"
                pair["preco_a"] = a["preco_kg"]
                pair["preco_b"] = b["preco_kg"]
                pair["observado_a"] = a["observado"]
                pair["observado_b"] = b["observado"]
                pair["preenchido_a"] = a["preenchido"]
                pair["preenchido_b"] = b["preenchido"]
                common = pair["preco_a"].notna() & pair["preco_b"].notna()
                pair["segmento_par"] = identificar_segmentos(common)
                same_segment = pair["segmento_par"].eq(
                    pair["segmento_par"].shift(1)
                )
                pair["retorno_a"] = np.log(pair["preco_a"]).diff().where(
                    common & same_segment
                )
                pair["retorno_b"] = np.log(pair["preco_b"]).diff().where(
                    common & same_segment
                )
                pair["periodo"] = np.where(
                    pair.index.date <= args.data_corte_treino,
                    "treino",
                    "teste",
                )
                pair = pair.loc[common].copy()

                train_pair = pair[pair["periodo"] == "treino"]
                segment_sizes = train_pair.groupby("segmento_par").size()
                n_segments_train = int(len(segment_sizes))
                longest_segment_train = (
                    int(segment_sizes.max()) if not segment_sizes.empty else 0
                )

                stored = pair.reset_index()
                stored.insert(0, "par_id", pair_id)
                stored.insert(1, "produto", product)
                stored.insert(2, "mercado_a", market_a)
                stored.insert(3, "mercado_b", market_b)
                stored["data_cotacao"] = stored["data_cotacao"].dt.strftime(
                    "%Y-%m-%d"
                )
                pair_series_rows.append(stored)
                pair_frames[pair_id] = pair

            pair_rows.append(
                {
                    "id": pair_id,
                    "produto": product,
                    "mercado_a": market_a,
                    "mercado_b": market_b,
                    "dias_observados_comuns_treino": observed_common,
                    "dias_disponiveis_comuns_treino": available_common,
                    "elegivel": int(eligible),
                    "motivo": reason,
                    "segmentos_treino": n_segments_train,
                    "maior_segmento_treino": longest_segment_train,
                }
            )

    pair_columns = [
        "par_id",
        "produto",
        "mercado_a",
        "mercado_b",
        "data_cotacao",
        "periodo",
        "preco_a",
        "preco_b",
        "observado_a",
        "observado_b",
        "preenchido_a",
        "preenchido_b",
        "segmento_par",
        "retorno_a",
        "retorno_b",
    ]
    if pair_series_rows:
        pair_series = pd.concat(pair_series_rows, ignore_index=True)[pair_columns]
    else:
        pair_series = pd.DataFrame(columns=pair_columns)
    return pd.DataFrame(pair_rows), pair_series, pair_frames


def testar_estacionariedade(
    values: pd.Series,
    alpha: float,
) -> ResultadoEstacionariedade:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    n = int(len(clean))
    if n < MINIMO_RETORNOS_ESTACIONARIEDADE:
        return ResultadoEstacionariedade(
            n, None, None, None, None, None, None, False, "amostra_insuficiente"
        )
    if np.isclose(float(clean.std(ddof=0)), 0.0):
        return ResultadoEstacionariedade(
            n, None, None, None, None, None, None, False, "serie_constante"
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_result = adfuller(clean.to_numpy(), regression="c", autolag="BIC")
            kpss_result = kpss(clean.to_numpy(), regression="c", nlags="auto")
    except (ValueError, np.linalg.LinAlgError) as error:
        return ResultadoEstacionariedade(
            n,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            f"erro: {error}",
        )

    adf_stat, adf_p, adf_lags = (
        float(adf_result[0]),
        float(adf_result[1]),
        int(adf_result[2]),
    )
    kpss_stat, kpss_p, kpss_lags = (
        float(kpss_result[0]),
        float(kpss_result[1]),
        int(kpss_result[2]),
    )
    stationary = adf_p < alpha and kpss_p > alpha
    status = "estacionaria" if stationary else "nao_estacionaria"
    return ResultadoEstacionariedade(
        n,
        adf_stat,
        adf_p,
        adf_lags,
        kpss_stat,
        kpss_p,
        kpss_lags,
        stationary,
        status,
    )


def diagnosticar_par(
    pair_id: int,
    pair: pd.DataFrame,
    market_a: str,
    market_b: str,
    alpha: float,
) -> tuple[list[dict[str, Any]], bool]:
    train = pair[pair["periodo"] == "treino"].dropna(
        subset=["retorno_a", "retorno_b"]
    )
    if train.empty:
        longest = train
    else:
        sizes = train.groupby("segmento_par").size()
        longest_id = sizes.idxmax()
        longest = train[train["segmento_par"] == longest_id]

    rows: list[dict[str, Any]] = []
    passed = True
    for market, column in ((market_a, "retorno_a"), (market_b, "retorno_b")):
        result = testar_estacionariedade(longest[column], alpha)
        passed = passed and result.estacionaria
        rows.append(
            {
                "par_id": pair_id,
                "mercado": market,
                "segmento_avaliado": (
                    int(longest["segmento_par"].iloc[0])
                    if not longest.empty
                    else None
                ),
                "n_observacoes": result.n_observacoes,
                "estatistica_adf": result.estatistica_adf,
                "p_valor_adf": result.p_valor_adf,
                "defasagens_adf": result.defasagens_adf,
                "estatistica_kpss": result.estatistica_kpss,
                "p_valor_kpss": result.p_valor_kpss,
                "defasagens_kpss": result.defasagens_kpss,
                "estacionaria": int(result.estacionaria),
                "status": result.status,
            }
        )
    return rows, passed


def montar_matrizes_granger(
    pair: pd.DataFrame,
    origin_column: str,
    destination_column: str,
    lag: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    restricted_parts: list[np.ndarray] = []
    unrestricted_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    train = pair[pair["periodo"] == "treino"]
    for _, segment in train.groupby("segmento_par", sort=True):
        values = segment[[origin_column, destination_column]].dropna().to_numpy()
        if len(values) <= max_lag:
            continue
        origin = values[:, 0]
        destination = values[:, 1]
        target = destination[max_lag:]
        destination_lags = np.column_stack(
            [
                destination[max_lag - offset : len(destination) - offset]
                for offset in range(1, lag + 1)
            ]
        )
        origin_lags = np.column_stack(
            [
                origin[max_lag - offset : len(origin) - offset]
                for offset in range(1, lag + 1)
            ]
        )
        intercept = np.ones((len(target), 1))
        restricted_parts.append(np.column_stack([intercept, destination_lags]))
        unrestricted_parts.append(
            np.column_stack([intercept, destination_lags, origin_lags])
        )
        target_parts.append(target)

    if not target_parts:
        raise ValueError("nenhum segmento possui histórico suficiente para os lags")
    return (
        np.concatenate(target_parts),
        np.vstack(restricted_parts),
        np.vstack(unrestricted_parts),
    )


def avaliar_direcao_granger(
    pair: pd.DataFrame,
    origin_column: str,
    destination_column: str,
    max_lag: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    candidates: list[dict[str, Any]] = []
    coefficients: dict[int, np.ndarray] = {}

    for lag in range(1, max_lag + 1):
        target, restricted, unrestricted = montar_matrizes_granger(
            pair,
            origin_column,
            destination_column,
            lag,
            max_lag,
        )
        n = len(target)
        k_unrestricted = unrestricted.shape[1]
        df_denominator = n - k_unrestricted
        if df_denominator <= 0:
            raise ValueError("amostra insuficiente para o modelo irrestrito")
        if np.linalg.matrix_rank(restricted) < restricted.shape[1]:
            raise ValueError("matriz restrita sem posto completo")
        if np.linalg.matrix_rank(unrestricted) < k_unrestricted:
            raise ValueError("matriz irrestrita sem posto completo")

        beta_restricted = np.linalg.lstsq(restricted, target, rcond=None)[0]
        beta_unrestricted = np.linalg.lstsq(unrestricted, target, rcond=None)[0]
        residual_restricted = target - restricted @ beta_restricted
        residual_unrestricted = target - unrestricted @ beta_unrestricted
        rss_restricted = float(residual_restricted @ residual_restricted)
        rss_unrestricted = float(residual_unrestricted @ residual_unrestricted)
        if rss_unrestricted <= 0:
            raise ValueError("resíduo nulo no modelo irrestrito")

        difference = max(0.0, rss_restricted - rss_unrestricted)
        f_statistic = (difference / lag) / (
            rss_unrestricted / df_denominator
        )
        p_value = float(distribuicao_f.sf(f_statistic, lag, df_denominator))
        bic = float(
            n * np.log(rss_unrestricted / n)
            + k_unrestricted * np.log(n)
        )
        candidates.append(
            {
                "lag": lag,
                "n_observacoes": n,
                "bic": bic,
                "rss_restrito": rss_restricted,
                "rss_irrestrito": rss_unrestricted,
                "estatistica_f": f_statistic,
                "graus_liberdade_num": lag,
                "graus_liberdade_den": df_denominator,
                "p_valor": p_value,
            }
        )
        coefficients[lag] = beta_unrestricted

    selected = min(candidates, key=lambda row: (row["bic"], row["lag"]))
    return candidates, selected, coefficients[selected["lag"]]


def corrigir_benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = adjusted
    return result.tolist()


def executar_testes_granger(
    pairs: pd.DataFrame,
    pair_frames: dict[int, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostic_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    result_id = 0

    eligible = pairs[pairs["elegivel"] == 1]
    for pair_row in eligible.itertuples(index=False):
        pair = pair_frames[pair_row.id]
        diagnostics, stationary = diagnosticar_par(
            pair_row.id,
            pair,
            pair_row.mercado_a,
            pair_row.mercado_b,
            args.alpha,
        )
        diagnostic_rows.extend(diagnostics)

        directions = (
            (
                pair_row.mercado_a,
                pair_row.mercado_b,
                "retorno_a",
                "retorno_b",
            ),
            (
                pair_row.mercado_b,
                pair_row.mercado_a,
                "retorno_b",
                "retorno_a",
            ),
        )
        for origin, destination, origin_column, destination_column in directions:
            result_id += 1
            base = {
                "id": result_id,
                "par_id": pair_row.id,
                "produto": pair_row.produto,
                "origem": origin,
                "destino": destination,
                "status": "nao_estacionario" if not stationary else "testado",
                "n_observacoes": None,
                "lag_selecionado": None,
                "bic": None,
                "estatistica_f": None,
                "graus_liberdade_num": None,
                "graus_liberdade_den": None,
                "p_valor": None,
                "q_valor_fdr": None,
                "significativo_bruto": 0,
                "significativo_fdr": 0,
                "observacao": (
                    "ADF e KPSS não aprovaram as duas séries"
                    if not stationary
                    else None
                ),
            }
            if not stationary:
                result_rows.append(base)
                continue

            try:
                candidates, selected, coefficients = avaliar_direcao_granger(
                    pair,
                    origin_column,
                    destination_column,
                    args.max_lag,
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                base["status"] = "erro_modelo"
                base["observacao"] = str(error)
                result_rows.append(base)
                continue

            base.update(
                {
                    "n_observacoes": selected["n_observacoes"],
                    "lag_selecionado": selected["lag"],
                    "bic": selected["bic"],
                    "estatistica_f": selected["estatistica_f"],
                    "graus_liberdade_num": selected["graus_liberdade_num"],
                    "graus_liberdade_den": selected["graus_liberdade_den"],
                    "p_valor": selected["p_valor"],
                    "significativo_bruto": int(selected["p_valor"] < args.alpha),
                }
            )
            result_rows.append(base)

            for candidate in candidates:
                lag_rows.append({"resultado_id": result_id, **candidate})

            selected_lag = int(selected["lag"])
            coefficient_rows.append(
                {
                    "resultado_id": result_id,
                    "tipo_termo": "intercepto",
                    "defasagem": 0,
                    "coeficiente": float(coefficients[0]),
                }
            )
            for offset in range(1, selected_lag + 1):
                coefficient_rows.append(
                    {
                        "resultado_id": result_id,
                        "tipo_termo": "destino",
                        "defasagem": offset,
                        "coeficiente": float(coefficients[offset]),
                    }
                )
                coefficient_rows.append(
                    {
                        "resultado_id": result_id,
                        "tipo_termo": "origem",
                        "defasagem": offset,
                        "coeficiente": float(
                            coefficients[selected_lag + offset]
                        ),
                    }
                )

    tested_indices = [
        index
        for index, row in enumerate(result_rows)
        if row["status"] == "testado"
    ]
    q_values = corrigir_benjamini_hochberg(
        [float(result_rows[index]["p_valor"]) for index in tested_indices]
    )
    for index, q_value in zip(tested_indices, q_values, strict=True):
        result_rows[index]["q_valor_fdr"] = q_value
        result_rows[index]["significativo_fdr"] = int(q_value < args.alpha)

    return (
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(result_rows),
        pd.DataFrame(lag_rows),
        pd.DataFrame(coefficient_rows),
    )


def criar_esquema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_granger (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT,
            banco_origem TEXT NOT NULL,
            banco_origem_sha256 TEXT NOT NULL,
            banco_origem_tamanho INTEGER NOT NULL,
            execucao_preprocessamento_id INTEGER NOT NULL,
            schema_preprocessamento INTEGER NOT NULL,
            parametros_json TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE mercados_recorte (
            mercado TEXT PRIMARY KEY,
            ceasa_slug TEXT NOT NULL,
            uf TEXT NOT NULL
        );

        CREATE TABLE produtos_recorte (
            produto TEXT PRIMARY KEY
        );

        CREATE TABLE series_temporais (
            produto TEXT NOT NULL,
            ceasa_slug TEXT NOT NULL,
            uf TEXT NOT NULL,
            mercado TEXT NOT NULL,
            data_cotacao TEXT NOT NULL,
            periodo TEXT NOT NULL CHECK (periodo IN ('treino', 'teste')),
            preco_kg_observado REAL,
            preco_kg REAL,
            n_cotacoes INTEGER NOT NULL,
            n_exatas INTEGER NOT NULL,
            n_calculadas INTEGER NOT NULL,
            n_padronizadas INTEGER NOT NULL,
            n_inferidas INTEGER NOT NULL,
            observado INTEGER NOT NULL CHECK (observado IN (0, 1)),
            preenchido INTEGER NOT NULL CHECK (preenchido IN (0, 1)),
            segmento_serie INTEGER,
            retorno_log REAL,
            PRIMARY KEY (produto, mercado, data_cotacao),
            FOREIGN KEY (produto) REFERENCES produtos_recorte(produto),
            FOREIGN KEY (mercado) REFERENCES mercados_recorte(mercado)
        );

        CREATE TABLE pares_candidatos (
            id INTEGER PRIMARY KEY,
            produto TEXT NOT NULL,
            mercado_a TEXT NOT NULL,
            mercado_b TEXT NOT NULL,
            dias_observados_comuns_treino INTEGER NOT NULL,
            dias_disponiveis_comuns_treino INTEGER NOT NULL,
            elegivel INTEGER NOT NULL CHECK (elegivel IN (0, 1)),
            motivo TEXT NOT NULL,
            segmentos_treino INTEGER NOT NULL,
            maior_segmento_treino INTEGER NOT NULL,
            UNIQUE (produto, mercado_a, mercado_b),
            FOREIGN KEY (produto) REFERENCES produtos_recorte(produto),
            FOREIGN KEY (mercado_a) REFERENCES mercados_recorte(mercado),
            FOREIGN KEY (mercado_b) REFERENCES mercados_recorte(mercado)
        );

        CREATE TABLE series_pares (
            par_id INTEGER NOT NULL,
            produto TEXT NOT NULL,
            mercado_a TEXT NOT NULL,
            mercado_b TEXT NOT NULL,
            data_cotacao TEXT NOT NULL,
            periodo TEXT NOT NULL CHECK (periodo IN ('treino', 'teste')),
            preco_a REAL NOT NULL,
            preco_b REAL NOT NULL,
            observado_a INTEGER NOT NULL CHECK (observado_a IN (0, 1)),
            observado_b INTEGER NOT NULL CHECK (observado_b IN (0, 1)),
            preenchido_a INTEGER NOT NULL CHECK (preenchido_a IN (0, 1)),
            preenchido_b INTEGER NOT NULL CHECK (preenchido_b IN (0, 1)),
            segmento_par INTEGER NOT NULL,
            retorno_a REAL,
            retorno_b REAL,
            PRIMARY KEY (par_id, data_cotacao),
            FOREIGN KEY (par_id) REFERENCES pares_candidatos(id)
        );

        CREATE TABLE diagnosticos_estacionariedade (
            par_id INTEGER NOT NULL,
            mercado TEXT NOT NULL,
            segmento_avaliado INTEGER,
            n_observacoes INTEGER NOT NULL,
            estatistica_adf REAL,
            p_valor_adf REAL,
            defasagens_adf INTEGER,
            estatistica_kpss REAL,
            p_valor_kpss REAL,
            defasagens_kpss INTEGER,
            estacionaria INTEGER NOT NULL CHECK (estacionaria IN (0, 1)),
            status TEXT NOT NULL,
            PRIMARY KEY (par_id, mercado),
            FOREIGN KEY (par_id) REFERENCES pares_candidatos(id),
            FOREIGN KEY (mercado) REFERENCES mercados_recorte(mercado)
        );

        CREATE TABLE resultados_granger (
            id INTEGER PRIMARY KEY,
            par_id INTEGER NOT NULL,
            produto TEXT NOT NULL,
            origem TEXT NOT NULL,
            destino TEXT NOT NULL,
            status TEXT NOT NULL,
            n_observacoes INTEGER,
            lag_selecionado INTEGER,
            bic REAL,
            estatistica_f REAL,
            graus_liberdade_num INTEGER,
            graus_liberdade_den INTEGER,
            p_valor REAL,
            q_valor_fdr REAL,
            significativo_bruto INTEGER NOT NULL CHECK (significativo_bruto IN (0, 1)),
            significativo_fdr INTEGER NOT NULL CHECK (significativo_fdr IN (0, 1)),
            observacao TEXT,
            UNIQUE (par_id, origem, destino),
            FOREIGN KEY (par_id) REFERENCES pares_candidatos(id),
            FOREIGN KEY (produto) REFERENCES produtos_recorte(produto),
            FOREIGN KEY (origem) REFERENCES mercados_recorte(mercado),
            FOREIGN KEY (destino) REFERENCES mercados_recorte(mercado)
        );

        CREATE TABLE resultados_lags (
            resultado_id INTEGER NOT NULL,
            lag INTEGER NOT NULL,
            n_observacoes INTEGER NOT NULL,
            bic REAL NOT NULL,
            rss_restrito REAL NOT NULL,
            rss_irrestrito REAL NOT NULL,
            estatistica_f REAL NOT NULL,
            graus_liberdade_num INTEGER NOT NULL,
            graus_liberdade_den INTEGER NOT NULL,
            p_valor REAL NOT NULL,
            PRIMARY KEY (resultado_id, lag),
            FOREIGN KEY (resultado_id) REFERENCES resultados_granger(id)
        );

        CREATE TABLE coeficientes_granger (
            resultado_id INTEGER NOT NULL,
            tipo_termo TEXT NOT NULL CHECK (
                tipo_termo IN ('intercepto', 'destino', 'origem')
            ),
            defasagem INTEGER NOT NULL,
            coeficiente REAL NOT NULL,
            PRIMARY KEY (resultado_id, tipo_termo, defasagem),
            FOREIGN KEY (resultado_id) REFERENCES resultados_granger(id)
        );

        CREATE TABLE resumo_granger (
            chave TEXT PRIMARY KEY,
            valor TEXT NOT NULL
        );
        """
    )


def criar_indices_e_views(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX idx_series_produto_mercado_data
            ON series_temporais(produto, mercado, data_cotacao);
        CREATE INDEX idx_series_periodo
            ON series_temporais(periodo);
        CREATE INDEX idx_pares_elegivel
            ON pares_candidatos(elegivel, produto);
        CREATE INDEX idx_series_pares_periodo
            ON series_pares(par_id, periodo, data_cotacao);
        CREATE INDEX idx_resultados_status
            ON resultados_granger(status);
        CREATE INDEX idx_resultados_fdr
            ON resultados_granger(significativo_fdr, q_valor_fdr);

        CREATE VIEW series_treino AS
        SELECT * FROM series_temporais WHERE periodo = 'treino';

        CREATE VIEW series_teste AS
        SELECT * FROM series_temporais WHERE periodo = 'teste';

        CREATE VIEW pares_elegiveis AS
        SELECT * FROM pares_candidatos WHERE elegivel = 1;

        CREATE VIEW resultados_granger_testados AS
        SELECT * FROM resultados_granger WHERE status = 'testado';

        CREATE VIEW relacoes_granger_significativas AS
        SELECT *
        FROM resultados_granger
        WHERE status = 'testado' AND significativo_fdr = 1;
        """
    )


def gravar_dataframe(
    connection: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return
    frame.to_sql(
        table,
        connection,
        if_exists="append",
        index=False,
        chunksize=2_000,
    )


def construir_resumo(
    quotes: pd.DataFrame,
    weekend_count: int,
    daily: pd.DataFrame,
    series: pd.DataFrame,
    pairs: pd.DataFrame,
    pair_series: pd.DataFrame,
    diagnostics: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, int]:
    tested = results[results["status"] == "testado"]
    stationary_pairs = (
        int(diagnostics.groupby("par_id")["estacionaria"].all().sum())
        if not diagnostics.empty
        else 0
    )
    return {
        "cotacoes_carregadas": int(len(quotes)),
        "cotacoes_inferidas_carregadas": int(
            (quotes["tipo_conversao"] == "inferida_alta_confianca").sum()
        ),
        "cotacoes_fim_de_semana_desconsideradas": weekend_count,
        "pontos_diarios_observados": int(len(daily)),
        "pontos_diarios_com_inferencia": int((daily["n_inferidas"] > 0).sum()),
        "linhas_series_temporais": int(len(series)),
        "dias_observados_series": int(series["observado"].sum()),
        "dias_preenchidos_series": int(series["preenchido"].sum()),
        "pares_candidatos": int(len(pairs)),
        "pares_elegiveis": int(pairs["elegivel"].sum()),
        "linhas_series_pares": int(len(pair_series)),
        "pares_estacionarios": stationary_pairs,
        "direcoes_registradas": int(len(results)),
        "direcoes_testadas": int(len(tested)),
        "significativas_sem_correcao": int(tested["significativo_bruto"].sum()),
        "significativas_fdr": int(tested["significativo_fdr"].sum()),
    }


def validar_saida(
    connection: sqlite3.Connection,
    min_days: int,
    max_lag: int,
) -> None:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"falha de integridade no SQLite de saída: {integrity}")
    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_keys:
        raise RuntimeError("foram encontradas referências inválidas na saída")

    bad_eligibility = connection.execute(
        """
        SELECT COUNT(*)
        FROM pares_candidatos
        WHERE elegivel <> CASE
            WHEN dias_observados_comuns_treino >= ? THEN 1 ELSE 0
        END
        """,
        (min_days,),
    ).fetchone()[0]
    if bad_eligibility:
        raise RuntimeError("a qualificação dos pares ficou inconsistente")

    missing_directions = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT pair.id
            FROM pares_candidatos pair
            LEFT JOIN resultados_granger result ON result.par_id = pair.id
            WHERE pair.elegivel = 1
            GROUP BY pair.id
            HAVING COUNT(result.id) <> 2
        )
        """
    ).fetchone()[0]
    if missing_directions:
        raise RuntimeError("pares elegíveis sem as duas direções registradas")

    bad_lags = connection.execute(
        """
        SELECT COUNT(*)
        FROM resultados_granger result
        WHERE result.status = 'testado'
          AND (
              result.lag_selecionado NOT BETWEEN 1 AND ?
              OR (SELECT COUNT(*) FROM resultados_lags lag
                  WHERE lag.resultado_id = result.id) <> ?
              OR result.q_valor_fdr IS NULL
              OR result.q_valor_fdr < 0
              OR result.q_valor_fdr > 1
          )
        """,
        (max_lag, max_lag),
    ).fetchone()[0]
    if bad_lags:
        raise RuntimeError("resultados de lag ou FDR inconsistentes")


def executar(args: argparse.Namespace) -> Path:
    database = args.database.resolve()
    output = args.output_db.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    source = abrir_somente_leitura(database)
    target: sqlite3.Connection | None = None
    temporary_output: Path | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        preprocessing = validar_banco_preprocessado(
            source,
            database,
            args.data_corte_treino,
        )
        source_stat_before = database.stat()
        source_hash = sha256_arquivo(database)
        source_stat_after = database.stat()
        if (
            source_stat_before.st_size != source_stat_after.st_size
            or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
        ):
            raise RuntimeError("o banco pré-processado mudou durante a leitura")

        market_metadata = carregar_metadados_mercados(source)
        print("Carregando o recorte fixado...", flush=True)
        quotes, weekend_count = carregar_cotacoes(source, args)
        daily = consolidar_mediana_diaria(quotes)
        print("Construindo as séries temporais...", flush=True)
        series, series_map = construir_series_temporais(
            daily,
            market_metadata,
            args,
        )
        pairs, pair_series, pair_frames = construir_pares(series_map, args)
        print(
            f"Pares elegíveis: {int(pairs['elegivel'].sum()):,} de {len(pairs):,}",
            flush=True,
        )
        print("Executando diagnósticos e testes de Granger...", flush=True)
        diagnostics, results, lags, coefficients = executar_testes_granger(
            pairs,
            pair_frames,
            args,
        )

        parameters = {
            "alpha": args.alpha,
            "correcao_multiplos_testes": "Benjamini-Hochberg global",
            "data_corte_treino": args.data_corte_treino.isoformat(),
            "data_fim_teste": args.data_fim_teste.isoformat(),
            "data_inicio": args.data_inicio.isoformat(),
            "estacionariedade": "ADF p<alpha e KPSS p>alpha no maior segmento contínuo",
            "ffill_limite": args.ffill_limite,
            "max_lag": args.max_lag,
            "mercados": list(MERCADOS_FIXOS),
            "min_dias_comuns_observados": args.min_dias_comuns,
            "produtos": list(PRODUTOS_FIXOS),
            "selecao_lag": "BIC em amostra fixa pelo max_lag",
            "tipos_conversao": list(TIPOS_CONVERSAO_ACEITOS),
        }
        summary = construir_resumo(
            quotes,
            weekend_count,
            daily,
            series,
            pairs,
            pair_series,
            diagnostics,
            results,
        )

        temporary_handle = tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}-",
            suffix=".tmp.sqlite",
            dir=output.parent,
            delete=False,
        )
        temporary_output = Path(temporary_handle.name)
        temporary_handle.close()

        target = sqlite3.connect(temporary_output)
        target.execute("PRAGMA journal_mode = DELETE")
        target.execute("PRAGMA synchronous = NORMAL")
        criar_esquema(target)
        target.execute(
            """
            INSERT INTO execucoes_granger (
                id, schema_version, iniciada_em, banco_origem,
                banco_origem_sha256, banco_origem_tamanho,
                execucao_preprocessamento_id, schema_preprocessamento,
                parametros_json, status
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 'executando')
            """,
            (
                SCHEMA_VERSION,
                started_at,
                str(args.database),
                source_hash,
                source_stat_after.st_size,
                preprocessing.execucao_id,
                preprocessing.schema_version,
                json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            ),
        )

        markets_to_store = market_metadata.rename(
            columns={"entreposto_slug": "mercado"}
        )[["mercado", "ceasa_slug", "uf"]]
        products_to_store = pd.DataFrame({"produto": PRODUTOS_FIXOS})
        gravar_dataframe(target, "mercados_recorte", markets_to_store)
        gravar_dataframe(target, "produtos_recorte", products_to_store)
        gravar_dataframe(target, "series_temporais", series)
        gravar_dataframe(target, "pares_candidatos", pairs)
        gravar_dataframe(target, "series_pares", pair_series)
        gravar_dataframe(target, "diagnosticos_estacionariedade", diagnostics)
        gravar_dataframe(target, "resultados_granger", results)
        gravar_dataframe(target, "resultados_lags", lags)
        gravar_dataframe(target, "coeficientes_granger", coefficients)
        target.executemany(
            "INSERT INTO resumo_granger (chave, valor) VALUES (?, ?)",
            [(key, str(value)) for key, value in sorted(summary.items())],
        )
        criar_indices_e_views(target)
        target.execute(
            """
            UPDATE execucoes_granger
            SET concluida_em = ?, status = 'concluida'
            WHERE id = 1
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        target.commit()
        validar_saida(target, args.min_dias_comuns, args.max_lag)
        target.close()
        target = None
        os.replace(temporary_output, output)
        temporary_output = None

        print(f"Banco de Granger criado em: {output}", flush=True)
        for key, value in sorted(summary.items()):
            print(f"  {key}: {value:,}", flush=True)
        return output
    finally:
        source.close()
        if target is not None:
            target.close()
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    validar_argumentos(args)
    executar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
