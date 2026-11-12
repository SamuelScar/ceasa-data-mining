#!/usr/bin/env python3
"""Cria um banco SQLite derivado para o pré-processamento das cotações CEASA.

Esta etapa é deliberadamente anterior à seleção de produtos e à construção
das séries temporais. Ela consolida as versões de cada cotação, registra as
transformações aplicadas e não altera o banco de origem.

Exemplo:
    python tratar_ceasa.py data/cotacoes.sqlite \
        --output-db resultados/preprocessamento/cotacoes_preprocessadas.sqlite
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


SCHEMA_VERSION = 8
MARKET_UNIT = "entreposto"
MASS_UNITS = {"kg": 1.0, "g": 0.001}
VOLUME_UNITS = {"l": 1.0, "ml": 0.001}
ITEM_UNIT_FACTORS = {
    "un": 1.0,
    "dz": 12.0,
    "cento": 100.0,
    "espiga": 1.0,
    "pe": 1.0,
}
MASS_TOKENS = {
    "kg",
    "quilo",
    "quilos",
    "quilograma",
    "quilogramas",
    "g",
    "gr",
    "grs",
    "grama",
    "gramas",
}
DEFAULT_OUTLIER_LOW = 0.1
DEFAULT_OUTLIER_HIGH = 10.0
DEFAULT_OUTLIER_MIN_GROUP = 5
DEFAULT_OUTLIER_PROBABLE_LOW = 0.01
DEFAULT_OUTLIER_PROBABLE_HIGH = 20.0
DEFAULT_OUTLIER_RECURRENT_MIN = 5
DEFAULT_INFERENCE_MIN_SUPPORT = 5
DEFAULT_INFERENCE_MIN_DATES = 3
DEFAULT_CHUNK_SIZE = 100_000
SQLITE_WRITE_BATCH_SIZE = 10_000
EXCLUDED_MAPPING_STATUSES = frozenset(
    {"ambigua", "sem_quantidade_exata", "sem_unidade"}
)

WEIGHT_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(kg|quilo(?:s)?|quilograma(?:s)?|g|gr|grs|grama(?:s)?)\b",
    flags=re.IGNORECASE,
)
WEIGHT_INTERVAL_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:-|\u2013|\u2014|a|ate)\s*"
    r"(\d+(?:[.,]\d+)?)\s*(?:-|\u2013|\u2014)?\s*"
    r"(kg|quilo(?:s)?|quilograma(?:s)?|g|gr|grs|grama(?:s)?)\b",
    flags=re.IGNORECASE,
)
INTERVAL_PATTERN = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:-|\u2013|\u2014|a|ate)\s*\d+(?:[.,]\d+)?",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
MALFORMED_KG_PE_PATTERN = re.compile(r"kg(?=pe\b)", flags=re.IGNORECASE)
EXPLICIT_ITEM_COUNT_PATTERN = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(?:un(?:i|id(?:ade)?s?)?|vasos?)\b",
    flags=re.IGNORECASE,
)


CATALOG_COLUMNS = [
    "apresentacao_unidade_id",
    "unidade_sigla",
    "unidade_original",
    "unidade_normalizada",
    "embalagem",
    "quantidade_minima_original",
    "quantidade_maxima_original",
    "detalhe_unidade",
    "assinatura_inferencia",
    "dimensao_mapeada",
    "unidade_base",
    "quantidade_unidades_base",
    "fator_itens_por_unidade_base",
    "quantidade_itens",
    "peso_minimo_kg",
    "peso_maximo_kg",
    "peso_total_kg",
    "volume_total_l",
    "tipo_conversao",
    "status_mapeamento",
    "formula_conversao",
    "evidencia_mapeamento",
    "observacao_conversao",
    "nivel_confianca",
    "origem_regra",
    "cotacoes_associadas",
    "cotacoes_excluidas_associadas",
    "produtos_associados",
    "ceasas_associadas",
    "entrepostos_associados",
]

INFERENCE_RULE_COLUMNS = [
    "id",
    "produto_chave_textual",
    "apresentacao_unidade_id",
    "origem_dado",
    "assinatura_inferencia",
    "peso_total_kg_inferido",
    "suporte_candidato",
    "datas_candidato",
    "ceasas_candidato",
    "entrepostos_candidato",
    "suporte_ancora",
    "datas_ancora",
    "ceasas_ancora",
    "entrepostos_ancora",
    "pesos_ancora_distintos",
    "dominancia_peso",
    "data_limite_inferencia",
    "status_regra",
    "motivo_status",
]

FACT_RULE_COLUMNS = [
    "apresentacao_unidade_id",
    "dimensao_mapeada",
    "unidade_base",
    "quantidade_unidades_base",
    "fator_itens_por_unidade_base",
    "quantidade_itens",
    "peso_minimo_kg",
    "peso_maximo_kg",
    "peso_total_kg",
    "volume_total_l",
    "tipo_conversao",
    "status_mapeamento",
    "formula_conversao",
    "evidencia_mapeamento",
    "observacao_conversao",
    "nivel_confianca",
    "origem_regra",
]


FACT_COLUMNS = [
    "cotacao_id",
    "chave_identidade",
    "coleta_id",
    "coleta_baixada_em",
    "coleta_processada_em",
    "data_cotacao",
    "ceasa_id",
    "ceasa_slug",
    "uf",
    "entreposto_id",
    "entreposto_slug",
    "entreposto_nome",
    "categoria_id",
    "categoria_slug",
    "produto_id",
    "produto_alias_id",
    "produto_original",
    "produto_normalizado_banco",
    "produto_chave_textual",
    "produto_valido",
    "apresentacao_unidade_id",
    "unidade_sigla",
    "unidade_original",
    "unidade_normalizada",
    "embalagem",
    "quantidade_minima",
    "quantidade_maxima",
    "detalhe_unidade",
    "procedencia",
    "classificacao",
    "situacao_mercado",
    "preco_minimo_consolidado",
    "preco_comum_consolidado",
    "preco_maximo_consolidado",
    "preco_minimo_tratado",
    "preco_comum_tratado",
    "preco_maximo_tratado",
    "preco_comum_valido",
    "ordem_precos_inconsistente",
    "dimensoes_validas",
    "dimensao_mapeada",
    "unidade_base",
    "quantidade_unidades_base",
    "fator_itens_por_unidade_base",
    "quantidade_itens",
    "peso_minimo_kg",
    "peso_maximo_kg",
    "peso_total_kg",
    "volume_total_l",
    "preco_minimo_unidade_base",
    "preco_comum_unidade_base",
    "preco_maximo_unidade_base",
    "preco_minimo_item",
    "preco_comum_item",
    "preco_maximo_item",
    "preco_minimo_kg",
    "preco_comum_kg",
    "preco_maximo_kg",
    "tipo_conversao",
    "status_mapeamento",
    "formula_conversao",
    "evidencia_mapeamento",
    "observacao_conversao",
    "nivel_confianca",
    "origem_regra",
    "regra_inferencia_id",
    "conversao_kg_disponivel",
    "origem_dado",
    "fonte_complemento",
    "url_complemento",
    "data_complemento",
    "registro_status",
    "versoes_identidade",
    "houve_conflito_versoes",
    "outlier_suspeito",
    "outlier_status",
    "outlier_razao_mediana",
]


EXCLUDED_COLUMNS = [
    "cotacao_id",
    "chave_identidade",
    "coleta_id",
    "data_cotacao",
    "ceasa_id",
    "ceasa_slug",
    "entreposto_id",
    "entreposto_slug",
    "entreposto_nome",
    "produto_id",
    "produto_alias_id",
    "produto_original",
    "produto_normalizado_banco",
    "produto_chave_textual",
    "apresentacao_unidade_id",
    "unidade_original",
    "unidade_normalizada",
    "embalagem",
    "quantidade_minima",
    "quantidade_maxima",
    "detalhe_unidade",
    "preco_minimo_consolidado",
    "preco_comum_consolidado",
    "preco_maximo_consolidado",
    "status_mapeamento",
    "observacao_conversao",
    "origem_dado",
    "versoes_identidade",
    "houve_conflito_versoes",
]


def normalizar_texto(value: object) -> str | None:
    """Cria uma chave textual sem alterar o valor original."""
    if value is None or pd.isna(value):
        return None
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())
    return text or None


def assinatura_apresentacao_para_inferencia(value: object) -> str | None:
    """Remove apenas a unidade de massa para aproximar formas explícitas e ambíguas."""
    normalized = normalizar_texto(value)
    if normalized is None:
        return None
    tokens = [token for token in normalized.split() if token not in MASS_TOKENS]
    signature = " ".join(tokens)
    return signature or None


def numero_positivo(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def quantidade_exata(qmin: object, qmax: object) -> float | None:
    minimum = numero_positivo(qmin)
    maximum = numero_positivo(qmax)
    if minimum is None:
        return None
    if maximum is None or math.isclose(minimum, maximum, rel_tol=0, abs_tol=1e-9):
        return minimum
    return None


def apresentacao_tem_intervalo(*texts: object) -> bool:
    combined = " ".join(
        str(text) for text in texts if text is not None and not pd.isna(text)
    )
    combined = unicodedata.normalize("NFKD", combined)
    combined = "".join(
        char for char in combined if not unicodedata.combining(char)
    )
    return INTERVAL_PATTERN.search(combined) is not None


def fator_unidade_peso(unit_value: object) -> float:
    unit = normalizar_texto(unit_value)
    return (
        1.0
        if unit in {"kg", "quilo", "quilos", "quilograma", "quilogramas"}
        else 0.001
    )


def peso_encontrado_kg(match: re.Match[str]) -> float:
    value = float(match.group(1).replace(",", "."))
    return value * fator_unidade_peso(match.group(2))


def texto_para_leitura_peso(texto_original: object) -> str | None:
    """Separa o sufixo PE quando ele foi concatenado à unidade kg."""
    if texto_original is None or pd.isna(texto_original):
        return None
    return MALFORMED_KG_PE_PATTERN.sub("kg ", str(texto_original))


def intervalo_peso_kg(texto_original: object) -> tuple[float, float] | None:
    text = texto_para_leitura_peso(texto_original)
    if text is None:
        return None
    match = WEIGHT_INTERVAL_PATTERN.search(text)
    if match is None:
        return None
    factor = fator_unidade_peso(match.group(3))
    minimum = float(match.group(1).replace(",", ".")) * factor
    maximum = float(match.group(2).replace(",", ".")) * factor
    return min(minimum, maximum), max(minimum, maximum)


def quantidade_itens_explicita(texto_original: object) -> tuple[float, str] | None:
    """Extrai uma contagem inequívoca de unidades ou vasos do texto original."""
    if texto_original is None or pd.isna(texto_original):
        return None
    matches = list(EXPLICIT_ITEM_COUNT_PATTERN.finditer(str(texto_original)))
    if not matches:
        return None
    quantities = {
        float(match.group(1).replace(",", "."))
        for match in matches
    }
    if len(quantities) != 1:
        return None
    quantity = quantities.pop()
    if not math.isfinite(quantity) or quantity <= 0:
        return None
    return quantity, matches[0].group(0)


def interpretar_peso_na_apresentacao(
    texto_original: object,
    quantidade: float,
    unidade_base: str,
) -> tuple[float | None, str, str | None]:
    """Distingue peso individual de peso total pela ordem do texto original."""
    if texto_original is None or pd.isna(texto_original):
        return None, "ausente", None

    text = texto_para_leitura_peso(texto_original)
    if text is None:
        return None, "ausente", None
    weight_matches = list(WEIGHT_PATTERN.finditer(text))
    if not weight_matches:
        return None, "ausente", None
    if len(weight_matches) > 1:
        return None, "ambigua", None

    weight_match = weight_matches[0]
    quantity_matches = []
    for match in NUMBER_PATTERN.finditer(text):
        overlaps_weight = not (
            match.end() <= weight_match.start()
            or match.start() >= weight_match.end()
        )
        if overlaps_weight:
            continue
        value = float(match.group(0).replace(",", "."))
        if math.isclose(value, quantidade, rel_tol=0, abs_tol=1e-9):
            quantity_matches.append(match)

    evidence = weight_match.group(0)
    if len(quantity_matches) != 1:
        return None, "ambigua", evidence
    if unidade_base in VOLUME_UNITS:
        return None, "ambigua", evidence

    weight = peso_encontrado_kg(weight_match)
    if quantity_matches[0].start() < weight_match.start():
        return weight, "individual", evidence
    return weight, "total", evidence


def abrir_origem(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Banco de origem não encontrado: {path}")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def registrar_tempo_etapa(nome: str, inicio: float) -> float:
    """Exibe a duração de uma etapa e inicia a medição da próxima."""
    duracao = perf_counter() - inicio
    print(f"Etapa '{nome}' concluída em {duracao:.2f} s.", flush=True)
    return perf_counter()


def preparar_origem(path: Path) -> tuple[Path, Path | None]:
    """Retorna um SQLite legível e, quando necessário, o temporário criado."""
    if path.suffix != ".xz":
        return path, None
    temporary = tempfile.NamedTemporaryFile(
        prefix="ceasa-origem-", suffix=".sqlite", delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with lzma.open(path, "rb") as source, temporary_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1 << 20)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path, temporary_path


def construir_catalogo_apresentacoes(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    presentations = pd.read_sql_query(
        """
        SELECT
            a.id AS apresentacao_unidade_id,
            u.sigla AS unidade_sigla,
            a.unidade_original,
            a.unidade_normalizada,
            a.embalagem,
            a.quantidade_minima,
            a.quantidade_maxima,
            a.detalhe_unidade
        FROM apresentacoes_unidade a
        LEFT JOIN unidades u ON u.id = a.unidade_id
        """,
        connection,
    )

    catalog: list[dict[str, object]] = []
    for row in presentations.itertuples(index=False):
        symbol = normalizar_texto(row.unidade_sigla)
        stored_exact_quantity = quantidade_exata(
            row.quantidade_minima,
            row.quantidade_maxima,
        )
        has_interval = apresentacao_tem_intervalo(
            row.unidade_original,
            row.unidade_normalizada,
        )
        exact_quantity = (
            None
            if has_interval
            else stored_exact_quantity
        )
        unit_base = symbol
        base_quantity: float | None = exact_quantity
        item_factor = ITEM_UNIT_FACTORS.get(symbol)
        item_quantity = (
            exact_quantity * item_factor
            if exact_quantity is not None and item_factor is not None
            else None
        )
        weight_minimum: float | None = None
        weight_maximum: float | None = None
        total_weight: float | None = None
        total_volume: float | None = None
        conversion_type = "incerta"
        mapping_status = "pendente"
        dimension = "desconhecida"
        formula: str | None = None
        evidence: str | None = None
        observation: str | None = None
        confidence = "pendente"
        rule_origin = "sem_regra"

        if symbol in MASS_UNITS:
            dimension = "massa"
            unit_base = "kg"
            item_factor = None
            item_quantity = None
            weight_interval = intervalo_peso_kg(row.unidade_original)
            if has_interval and weight_interval is not None:
                weight_minimum, weight_maximum = weight_interval
                base_quantity = None
                mapping_status = "intervalo_massa"
                evidence = str(row.unidade_original)
                confidence = "alta"
                rule_origin = "parser_deterministico"
                observation = (
                    "Intervalo de peso preservado; conversao para um unico "
                    "preco por kg nao aplicada."
                )
            elif stored_exact_quantity is not None:
                total_weight = stored_exact_quantity * MASS_UNITS[symbol]
                weight_minimum = total_weight
                weight_maximum = total_weight
                base_quantity = total_weight
                conversion_type = "exata"
                mapping_status = "convertida_exata"
                formula = (
                    f"peso_total_kg = {stored_exact_quantity:g} {symbol} "
                    f"* {MASS_UNITS[symbol]:g}"
                )
                evidence = str(row.unidade_original)
                confidence = "alta"
                rule_origin = "parser_deterministico"
            else:
                mapping_status = "sem_quantidade_exata"
                observation = "Peso ausente ou sem quantidade exata."
        elif symbol:
            if symbol in VOLUME_UNITS:
                dimension = "volume"
            elif symbol in ITEM_UNIT_FACTORS:
                dimension = "contagem"
            else:
                dimension = "unidade_nativa"

            if exact_quantity is None:
                base_quantity = None
                item_quantity = None
                mapping_status = (
                    "intervalo_quantidade" if has_interval else "sem_quantidade_exata"
                )
                confidence = "alta" if has_interval else "pendente"
                rule_origin = (
                    "parser_deterministico" if has_interval else "sem_regra"
                )
                evidence = str(row.unidade_original) if has_interval else None
                observation = (
                    "Quantidade da unidade-base informada como intervalo; "
                    "conversao nao aplicada."
                    if has_interval
                    else "Quantidade da unidade-base ausente ou sem valor exato."
                )
            else:
                explicit_weight, weight_semantics, evidence = (
                    interpretar_peso_na_apresentacao(
                        row.unidade_original,
                        exact_quantity,
                        symbol,
                    )
                )
                if weight_semantics == "individual" and explicit_weight is not None:
                    dimension = "massa_e_contagem"
                    total_weight = exact_quantity * explicit_weight
                    weight_minimum = total_weight
                    weight_maximum = total_weight
                    conversion_type = "calculada"
                    mapping_status = "convertida_calculada"
                    formula = (
                        f"peso_total_kg = {exact_quantity:g} {symbol} "
                        f"* {explicit_weight:g} kg/{symbol}"
                    )
                    confidence = "alta"
                    rule_origin = "parser_deterministico"
                    observation = (
                        f"Peso individual identificado pela ordem do texto: "
                        f"{evidence}."
                    )
                elif weight_semantics == "total" and explicit_weight is not None:
                    dimension = "massa_e_contagem"
                    total_weight = explicit_weight
                    weight_minimum = total_weight
                    weight_maximum = total_weight
                    conversion_type = "exata"
                    mapping_status = "convertida_exata"
                    formula = f"peso_total_kg = {explicit_weight:g} kg"
                    confidence = "alta"
                    rule_origin = "parser_deterministico"
                    observation = (
                        f"Peso total identificado antes da quantidade: {evidence}."
                    )
                elif weight_semantics == "ambigua":
                    dimension = "mista" if symbol in VOLUME_UNITS else "ambigua"
                    base_quantity = None
                    item_quantity = None
                    mapping_status = "ambigua"
                    evidence = evidence or str(row.unidade_original)
                    observation = (
                        "Apresentacao combina quantidade e peso de forma ambigua; "
                        "conversao nao aplicada."
                    )
                else:
                    mapping_status = (
                        "volume" if symbol in VOLUME_UNITS else "unidade_nativa"
                    )
                    confidence = "alta"
                    rule_origin = "parser_deterministico"
                    evidence = symbol
                    if symbol in VOLUME_UNITS:
                        total_volume = exact_quantity * VOLUME_UNITS[symbol]
                        formula = (
                            f"volume_total_l = {exact_quantity:g} {symbol} "
                            f"* {VOLUME_UNITS[symbol]:g}"
                        )
                        observation = (
                            "Volume preservado em litros; conversao para kg nao "
                            "aplicada."
                        )
                    elif item_factor is not None:
                        formula = (
                            f"quantidade_itens = {exact_quantity:g} {symbol} "
                            f"* {item_factor:g}"
                        )
                        observation = (
                            "Quantidade de itens normalizada; falta peso por item "
                            "para conversao em kg."
                        )
                    else:
                        observation = (
                            "Unidade nativa preservada; falta peso por "
                            "unidade-base."
                        )
        else:
            base_quantity = None
            item_factor = None
            item_quantity = None
            weight_interval = intervalo_peso_kg(row.unidade_original)
            explicit_item_count = quantidade_itens_explicita(row.unidade_original)
            weight_text = texto_para_leitura_peso(row.unidade_original)
            weight_matches = (
                [] if weight_text is None else list(WEIGHT_PATTERN.finditer(weight_text))
            )
            if weight_interval is not None:
                dimension = "massa"
                weight_minimum, weight_maximum = weight_interval
                unit_base = "kg"
                mapping_status = "intervalo_massa"
                evidence = str(row.unidade_original)
                confidence = "alta"
                rule_origin = "parser_deterministico"
                observation = (
                    "Intervalo de peso preservado; conversao para um unico preco "
                    "por kg nao aplicada."
                )
            elif len(weight_matches) == 1 and not has_interval:
                dimension = "massa"
                total_weight = peso_encontrado_kg(weight_matches[0])
                weight_minimum = total_weight
                weight_maximum = total_weight
                unit_base = "kg"
                base_quantity = total_weight
                conversion_type = "exata"
                mapping_status = "convertida_exata"
                formula = f"peso_total_kg = {total_weight:g} kg"
                evidence = weight_matches[0].group(0)
                confidence = "alta"
                rule_origin = "parser_deterministico"
                observation = "Peso total explícito identificado no texto original."
            elif explicit_item_count is not None and weight_interval is None:
                item_quantity, evidence = explicit_item_count
                dimension = "contagem"
                unit_base = "un"
                base_quantity = item_quantity
                item_factor = 1.0
                conversion_type = "incerta"
                mapping_status = "unidade_nativa"
                formula = f"quantidade_itens = {item_quantity:g} un"
                confidence = "alta"
                rule_origin = "parser_deterministico"
                observation = (
                    "Contagem explícita preservada; falta peso por item para "
                    "conversão em kg."
                )
            else:
                mapping_status = "sem_unidade"
                observation = "Apresentação sem unidade identificada."

        catalog.append(
            {
                "apresentacao_unidade_id": row.apresentacao_unidade_id,
                "unidade_sigla": row.unidade_sigla,
                "unidade_original": row.unidade_original,
                "unidade_normalizada": row.unidade_normalizada,
                "embalagem": row.embalagem,
                "quantidade_minima_original": row.quantidade_minima,
                "quantidade_maxima_original": row.quantidade_maxima,
                "detalhe_unidade": row.detalhe_unidade,
                "assinatura_inferencia": assinatura_apresentacao_para_inferencia(
                    row.unidade_normalizada or row.unidade_original
                ),
                "dimensao_mapeada": dimension,
                "unidade_base": unit_base,
                "quantidade_unidades_base": base_quantity,
                "fator_itens_por_unidade_base": item_factor,
                "quantidade_itens": item_quantity,
                "peso_minimo_kg": weight_minimum,
                "peso_maximo_kg": weight_maximum,
                "peso_total_kg": total_weight,
                "volume_total_l": total_volume,
                "tipo_conversao": conversion_type,
                "status_mapeamento": mapping_status,
                "formula_conversao": formula,
                "evidencia_mapeamento": evidence,
                "observacao_conversao": observation,
                "nivel_confianca": confidence,
                "origem_regra": rule_origin,
                "cotacoes_associadas": 0,
                "cotacoes_excluidas_associadas": 0,
                "produtos_associados": 0,
                "ceasas_associadas": 0,
                "entrepostos_associados": 0,
            }
        )

    return pd.DataFrame(catalog).reindex(columns=CATALOG_COLUMNS)


def criar_esquema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_preprocessamento (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT,
            banco_origem TEXT NOT NULL,
            parametros_json TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE catalogo_apresentacoes (
            apresentacao_unidade_id INTEGER PRIMARY KEY,
            unidade_sigla TEXT,
            unidade_original TEXT,
            unidade_normalizada TEXT,
            embalagem TEXT,
            quantidade_minima_original REAL,
            quantidade_maxima_original REAL,
            detalhe_unidade TEXT,
            assinatura_inferencia TEXT,
            dimensao_mapeada TEXT NOT NULL,
            unidade_base TEXT,
            quantidade_unidades_base REAL,
            fator_itens_por_unidade_base REAL,
            quantidade_itens REAL,
            peso_minimo_kg REAL,
            peso_maximo_kg REAL,
            peso_total_kg REAL,
            volume_total_l REAL,
            tipo_conversao TEXT NOT NULL CHECK (
                tipo_conversao IN (
                    'exata', 'calculada', 'padronizada',
                    'inferida_alta_confianca', 'incerta'
                )
            ),
            status_mapeamento TEXT NOT NULL,
            formula_conversao TEXT,
            evidencia_mapeamento TEXT,
            observacao_conversao TEXT,
            nivel_confianca TEXT NOT NULL,
            origem_regra TEXT NOT NULL,
            cotacoes_associadas INTEGER NOT NULL DEFAULT 0,
            cotacoes_excluidas_associadas INTEGER NOT NULL DEFAULT 0,
            produtos_associados INTEGER NOT NULL DEFAULT 0,
            ceasas_associadas INTEGER NOT NULL DEFAULT 0,
            entrepostos_associados INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE regras_conversao_inferida (
            id INTEGER PRIMARY KEY,
            produto_chave_textual TEXT NOT NULL,
            apresentacao_unidade_id INTEGER NOT NULL,
            origem_dado TEXT NOT NULL CHECK (origem_dado IN ('crawler', 'prohort')),
            assinatura_inferencia TEXT NOT NULL,
            peso_total_kg_inferido REAL NOT NULL CHECK (peso_total_kg_inferido > 0),
            suporte_candidato INTEGER NOT NULL,
            datas_candidato INTEGER NOT NULL,
            ceasas_candidato INTEGER NOT NULL,
            entrepostos_candidato INTEGER NOT NULL,
            suporte_ancora INTEGER NOT NULL,
            datas_ancora INTEGER NOT NULL,
            ceasas_ancora INTEGER NOT NULL,
            entrepostos_ancora INTEGER NOT NULL,
            pesos_ancora_distintos INTEGER NOT NULL,
            dominancia_peso REAL NOT NULL,
            data_limite_inferencia TEXT NOT NULL,
            status_regra TEXT NOT NULL CHECK (
                status_regra IN ('aplicada', 'candidata')
            ),
            motivo_status TEXT NOT NULL,
            UNIQUE (
                produto_chave_textual,
                apresentacao_unidade_id,
                origem_dado
            )
        );

        CREATE TABLE cotacoes_preprocessadas (
            cotacao_id INTEGER PRIMARY KEY,
            chave_identidade TEXT NOT NULL UNIQUE,
            coleta_id INTEGER NOT NULL,
            coleta_baixada_em TEXT,
            coleta_processada_em TEXT,
            data_cotacao TEXT NOT NULL,
            ceasa_id INTEGER,
            ceasa_slug TEXT,
            uf TEXT,
            entreposto_id INTEGER,
            entreposto_slug TEXT,
            entreposto_nome TEXT,
            categoria_id INTEGER,
            categoria_slug TEXT,
            produto_id INTEGER,
            produto_alias_id INTEGER,
            produto_original TEXT,
            produto_normalizado_banco TEXT,
            produto_chave_textual TEXT,
            produto_valido INTEGER NOT NULL,
            apresentacao_unidade_id INTEGER,
            unidade_sigla TEXT,
            unidade_original TEXT,
            unidade_normalizada TEXT,
            embalagem TEXT,
            quantidade_minima REAL,
            quantidade_maxima REAL,
            detalhe_unidade TEXT,
            procedencia TEXT,
            classificacao TEXT,
            situacao_mercado TEXT,
            preco_minimo_consolidado REAL,
            preco_comum_consolidado REAL,
            preco_maximo_consolidado REAL,
            preco_minimo_tratado REAL,
            preco_comum_tratado REAL,
            preco_maximo_tratado REAL,
            preco_comum_valido INTEGER NOT NULL,
            ordem_precos_inconsistente INTEGER NOT NULL,
            dimensoes_validas INTEGER NOT NULL,
            dimensao_mapeada TEXT NOT NULL,
            unidade_base TEXT,
            quantidade_unidades_base REAL,
            fator_itens_por_unidade_base REAL,
            quantidade_itens REAL,
            peso_minimo_kg REAL,
            peso_maximo_kg REAL,
            peso_total_kg REAL,
            volume_total_l REAL,
            preco_minimo_unidade_base REAL,
            preco_comum_unidade_base REAL,
            preco_maximo_unidade_base REAL,
            preco_minimo_item REAL,
            preco_comum_item REAL,
            preco_maximo_item REAL,
            preco_minimo_kg REAL,
            preco_comum_kg REAL,
            preco_maximo_kg REAL,
            tipo_conversao TEXT NOT NULL CHECK (
                tipo_conversao IN (
                    'exata', 'calculada', 'padronizada',
                    'inferida_alta_confianca', 'incerta'
                )
            ),
            status_mapeamento TEXT NOT NULL,
            formula_conversao TEXT,
            evidencia_mapeamento TEXT,
            observacao_conversao TEXT,
            nivel_confianca TEXT NOT NULL,
            origem_regra TEXT NOT NULL,
            regra_inferencia_id INTEGER,
            conversao_kg_disponivel INTEGER NOT NULL,
            origem_dado TEXT NOT NULL CHECK (origem_dado IN ('crawler', 'prohort')),
            fonte_complemento TEXT,
            url_complemento TEXT,
            data_complemento TEXT,
            registro_status TEXT NOT NULL,
            versoes_identidade INTEGER NOT NULL DEFAULT 1,
            houve_conflito_versoes INTEGER NOT NULL DEFAULT 0,
            outlier_suspeito INTEGER NOT NULL DEFAULT 0,
            outlier_status TEXT NOT NULL DEFAULT 'nao_suspeito',
            outlier_razao_mediana REAL,
            FOREIGN KEY (apresentacao_unidade_id)
                REFERENCES catalogo_apresentacoes(apresentacao_unidade_id),
            FOREIGN KEY (regra_inferencia_id)
                REFERENCES regras_conversao_inferida(id)
        );

        CREATE TABLE regras_conversao_padronizada (
            id INTEGER PRIMARY KEY,
            produto_id INTEGER,
            apresentacao_unidade_id INTEGER,
            peso_total_kg REAL NOT NULL CHECK (peso_total_kg > 0),
            formula TEXT NOT NULL,
            fonte_regra TEXT NOT NULL,
            ativa INTEGER NOT NULL DEFAULT 1,
            UNIQUE (produto_id, apresentacao_unidade_id)
        );

        CREATE TABLE cotacoes_excluidas_mapeamento (
            cotacao_id INTEGER PRIMARY KEY,
            chave_identidade TEXT NOT NULL UNIQUE,
            coleta_id INTEGER NOT NULL,
            data_cotacao TEXT NOT NULL,
            ceasa_id INTEGER,
            ceasa_slug TEXT,
            entreposto_id INTEGER,
            entreposto_slug TEXT,
            entreposto_nome TEXT,
            produto_id INTEGER,
            produto_alias_id INTEGER,
            produto_original TEXT,
            produto_normalizado_banco TEXT,
            produto_chave_textual TEXT,
            apresentacao_unidade_id INTEGER,
            unidade_original TEXT,
            unidade_normalizada TEXT,
            embalagem TEXT,
            quantidade_minima REAL,
            quantidade_maxima REAL,
            detalhe_unidade TEXT,
            preco_minimo_consolidado REAL,
            preco_comum_consolidado REAL,
            preco_maximo_consolidado REAL,
            status_mapeamento TEXT NOT NULL CHECK (
                status_mapeamento IN (
                    'ambigua', 'sem_quantidade_exata', 'sem_unidade'
                )
            ),
            observacao_conversao TEXT,
            origem_dado TEXT NOT NULL CHECK (origem_dado IN ('crawler', 'prohort')),
            versoes_identidade INTEGER NOT NULL DEFAULT 1,
            houve_conflito_versoes INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (apresentacao_unidade_id)
                REFERENCES catalogo_apresentacoes(apresentacao_unidade_id)
        );

        CREATE TABLE outliers_suspeitos (
            cotacao_id INTEGER PRIMARY KEY,
            metodo TEXT NOT NULL,
            tamanho_grupo INTEGER NOT NULL,
            outliers_no_grupo INTEGER NOT NULL,
            mediana_grupo REAL NOT NULL,
            razao_mediana REAL NOT NULL,
            classificacao TEXT NOT NULL CHECK (
                classificacao IN (
                    'erro_provavel',
                    'inconsistencia_recorrente',
                    'extremo_isolado'
                )
            ),
            observacao TEXT,
            FOREIGN KEY (cotacao_id) REFERENCES cotacoes_preprocessadas(cotacao_id)
        );

        CREATE TABLE resumo_preprocessamento (
            indicador TEXT PRIMARY KEY,
            valor TEXT NOT NULL
        );
        """
    )


def inserir_catalogo_apresentacoes(
    connection: sqlite3.Connection,
    catalog: pd.DataFrame,
) -> None:
    catalog.to_sql(
        "catalogo_apresentacoes",
        connection,
        if_exists="append",
        index=False,
        chunksize=SQLITE_WRITE_BATCH_SIZE,
    )


def atualizar_cobertura_catalogo(connection: sqlite3.Connection) -> None:
    coverage = connection.execute(
        """
        SELECT
            apresentacao_unidade_id,
            COUNT(*) AS cotacoes_associadas,
            COUNT(DISTINCT produto_id) AS produtos_associados,
            COUNT(DISTINCT ceasa_id) AS ceasas_associadas,
            COUNT(DISTINCT entreposto_id) AS entrepostos_associados
        FROM cotacoes_preprocessadas
        WHERE apresentacao_unidade_id IS NOT NULL
        GROUP BY apresentacao_unidade_id
        """
    ).fetchall()
    connection.executemany(
        """
        UPDATE catalogo_apresentacoes
        SET
            cotacoes_associadas = ?,
            produtos_associados = ?,
            ceasas_associadas = ?,
            entrepostos_associados = ?
        WHERE apresentacao_unidade_id = ?
        """,
        [
            (
                quote_count,
                product_count,
                ceasa_count,
                warehouse_count,
                presentation_id,
            )
            for (
                presentation_id,
                quote_count,
                product_count,
                ceasa_count,
                warehouse_count,
            ) in coverage
        ],
    )
    excluded_coverage = connection.execute(
        """
        SELECT apresentacao_unidade_id, COUNT(*)
        FROM cotacoes_excluidas_mapeamento
        WHERE apresentacao_unidade_id IS NOT NULL
        GROUP BY apresentacao_unidade_id
        """
    ).fetchall()
    connection.executemany(
        """
        UPDATE catalogo_apresentacoes
        SET cotacoes_excluidas_associadas = ?
        WHERE apresentacao_unidade_id = ?
        """,
        [
            (excluded_count, presentation_id)
            for presentation_id, excluded_count in excluded_coverage
        ],
    )


ORIGEM_DADO_SQL = """
CASE
    WHEN c.fonte_complemento IS NOT NULL
     AND TRIM(c.fonte_complemento) <> ''
    THEN 'prohort'
    ELSE 'crawler'
END
""".strip()


def construir_chaves_produtos(
    source: sqlite3.Connection,
) -> pd.DataFrame:
    """Normaliza cada alias de produto uma única vez por execução."""
    product_aliases = pd.read_sql_query(
        """
        SELECT
            pa.id AS produto_alias_id,
            pa.nome_original AS produto_original,
            p.nome_normalizado AS produto_normalizado_banco
        FROM produto_aliases pa
        JOIN produtos p ON p.id = pa.produto_id
        """,
        source,
    )
    product_aliases["produto_chave_textual"] = product_aliases[
        "produto_normalizado_banco"
    ].map(normalizar_texto)
    missing_product_key = product_aliases["produto_chave_textual"].isna()
    product_aliases.loc[
        missing_product_key, "produto_chave_textual"
    ] = product_aliases.loc[missing_product_key, "produto_original"].map(
        normalizar_texto
    )
    return product_aliases


def construir_regras_inferencia(
    source: sqlite3.Connection,
    catalog: pd.DataFrame,
    product_aliases: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Descobre regras por produto e apresentação usando somente o treino."""
    limite_informado = args.data_limite_inferencia
    limite_analise = limite_informado or source.execute(
        "SELECT MAX(data_cotacao) FROM cotacoes"
    ).fetchone()[0]
    if limite_analise is None:
        return pd.DataFrame(columns=INFERENCE_RULE_COLUMNS)

    source.execute("DROP TABLE IF EXISTS temp.produtos_inferencia")
    source.execute(
        """
        CREATE TEMP TABLE produtos_inferencia (
            produto_alias_id INTEGER PRIMARY KEY,
            produto_chave_textual TEXT NOT NULL
        )
        """
    )
    source.executemany(
        """
        INSERT INTO produtos_inferencia (
            produto_alias_id,
            produto_chave_textual
        ) VALUES (?, ?)
        """,
        product_aliases.loc[
            product_aliases["produto_chave_textual"].notna(),
            ["produto_alias_id", "produto_chave_textual"],
        ].itertuples(index=False, name=None),
    )

    source.execute("DROP TABLE IF EXISTS temp.apresentacoes_inferencia")
    source.execute(
        """
        CREATE TEMP TABLE apresentacoes_inferencia (
            apresentacao_unidade_id INTEGER PRIMARY KEY,
            assinatura_inferencia TEXT NOT NULL,
            papel TEXT NOT NULL,
            peso_total_kg REAL
        )
        """
    )
    presentations: list[tuple[object, ...]] = []
    candidate_statuses = {"sem_unidade", "unidade_nativa"}
    for row in catalog.itertuples(index=False):
        signature = row.assinatura_inferencia
        weight = numero_positivo(row.peso_total_kg)
        if signature is None or pd.isna(signature):
            continue
        if weight is not None:
            role = "ancora"
        elif row.status_mapeamento in candidate_statuses:
            role = "candidata"
        else:
            continue
        presentations.append(
            (row.apresentacao_unidade_id, signature, role, weight)
        )
    source.executemany(
        """
        INSERT INTO apresentacoes_inferencia (
            apresentacao_unidade_id,
            assinatura_inferencia,
            papel,
            peso_total_kg
        ) VALUES (?, ?, ?, ?)
        """,
        presentations,
    )

    statistics = pd.read_sql_query(
        f"""
        SELECT
            pi.produto_chave_textual,
            {ORIGEM_DADO_SQL} AS origem_dado,
            ai.assinatura_inferencia,
            CASE
                WHEN ai.papel = 'candidata'
                THEN c.apresentacao_unidade_id
            END AS apresentacao_candidata_id,
            ai.papel,
            ai.peso_total_kg,
            COUNT(DISTINCT c.chave_identidade) AS suporte,
            COUNT(DISTINCT c.data_cotacao) AS quantidade_datas,
            COUNT(DISTINCT e.ceasa_id) AS quantidade_ceasas,
            COUNT(DISTINCT c.entreposto_id) AS quantidade_entrepostos
        FROM cotacoes c
        JOIN produtos_inferencia pi
          ON pi.produto_alias_id = c.produto_alias_id
        JOIN entrepostos e ON e.id = c.entreposto_id
        JOIN apresentacoes_inferencia ai
          ON ai.apresentacao_unidade_id = c.apresentacao_unidade_id
        WHERE c.data_cotacao <= ?
          AND c.entreposto_id IS NOT NULL
          AND CAST(c.preco_comum AS REAL) > 0
        GROUP BY
            pi.produto_chave_textual,
            origem_dado,
            ai.assinatura_inferencia,
            apresentacao_candidata_id,
            ai.papel,
            ai.peso_total_kg
        """,
        source,
        params=(limite_analise,),
    )
    if statistics.empty:
        return pd.DataFrame(columns=INFERENCE_RULE_COLUMNS)

    context = [
        "produto_chave_textual",
        "origem_dado",
        "assinatura_inferencia",
    ]
    candidates = statistics.loc[statistics["papel"].eq("candidata")].copy()
    anchors = statistics.loc[statistics["papel"].eq("ancora")].copy()
    if candidates.empty or anchors.empty:
        return pd.DataFrame(columns=INFERENCE_RULE_COLUMNS)

    anchor_weights = anchors.rename(
        columns={
            "suporte": "suporte_ancora",
            "quantidade_datas": "datas_ancora",
            "quantidade_ceasas": "ceasas_ancora",
            "quantidade_entrepostos": "entrepostos_ancora",
        }
    )
    anchor_totals = (
        anchor_weights.groupby(context, as_index=False)["suporte_ancora"]
        .sum()
        .rename(columns={"suporte_ancora": "suporte_ancora_total"})
    )
    anchor_weights = anchor_weights.merge(anchor_totals, on=context, how="left")
    anchor_weights["dominancia_peso"] = (
        anchor_weights["suporte_ancora"]
        / anchor_weights["suporte_ancora_total"]
    )
    anchor_weights["pesos_ancora_distintos"] = anchor_weights.groupby(
        context
    )["peso_total_kg"].transform("nunique")
    anchor_weights = anchor_weights.sort_values(
        context + ["suporte_ancora", "peso_total_kg"],
        ascending=[True, True, True, False, True],
    )
    best_anchors = anchor_weights.drop_duplicates(context, keep="first").rename(
        columns={"peso_total_kg": "peso_total_kg_inferido"}
    )

    candidates = candidates.rename(
        columns={
            "apresentacao_candidata_id": "apresentacao_unidade_id",
            "suporte": "suporte_candidato",
            "quantidade_datas": "datas_candidato",
            "quantidade_ceasas": "ceasas_candidato",
            "quantidade_entrepostos": "entrepostos_candidato",
        }
    )
    rules = candidates.merge(best_anchors, on=context, how="inner")
    if rules.empty:
        return pd.DataFrame(columns=INFERENCE_RULE_COLUMNS)

    records: list[dict[str, object]] = []
    for rule_id, row in enumerate(rules.itertuples(index=False), start=1):
        reasons: list[str] = []
        if row.suporte_candidato < args.inferencia_min_suporte:
            reasons.append("suporte_candidato_insuficiente")
        if row.suporte_ancora < args.inferencia_min_suporte:
            reasons.append("suporte_ancora_insuficiente")
        if row.datas_candidato < args.inferencia_min_datas:
            reasons.append("datas_candidato_insuficientes")
        if row.datas_ancora < args.inferencia_min_datas:
            reasons.append("datas_ancora_insuficientes")
        if row.pesos_ancora_distintos != 1:
            reasons.append("pesos_ancora_conflitantes")
        if limite_informado is None:
            reasons.append("data_limite_treino_nao_definida")
        records.append(
            {
                "id": rule_id,
                "produto_chave_textual": row.produto_chave_textual,
                "apresentacao_unidade_id": int(row.apresentacao_unidade_id),
                "origem_dado": row.origem_dado,
                "assinatura_inferencia": row.assinatura_inferencia,
                "peso_total_kg_inferido": float(row.peso_total_kg_inferido),
                "suporte_candidato": int(row.suporte_candidato),
                "datas_candidato": int(row.datas_candidato),
                "ceasas_candidato": int(row.ceasas_candidato),
                "entrepostos_candidato": int(row.entrepostos_candidato),
                "suporte_ancora": int(row.suporte_ancora),
                "datas_ancora": int(row.datas_ancora),
                "ceasas_ancora": int(row.ceasas_ancora),
                "entrepostos_ancora": int(row.entrepostos_ancora),
                "pesos_ancora_distintos": int(row.pesos_ancora_distintos),
                "dominancia_peso": float(row.dominancia_peso),
                "data_limite_inferencia": limite_analise,
                "status_regra": "candidata" if reasons else "aplicada",
                "motivo_status": (
                    ";".join(reasons) if reasons else "criterios_atendidos"
                ),
            }
        )
    return pd.DataFrame(records).reindex(columns=INFERENCE_RULE_COLUMNS)


def inserir_regras_inferencia(
    connection: sqlite3.Connection,
    rules: pd.DataFrame,
) -> None:
    if rules.empty:
        return
    rules.to_sql(
        "regras_conversao_inferida",
        connection,
        if_exists="append",
        index=False,
        chunksize=SQLITE_WRITE_BATCH_SIZE,
    )


def preparar_ultimas_coletas(source: sqlite3.Connection) -> None:
    """Materializa em TEMP a coleta mais recente de cada identidade lógica."""
    source.execute("DROP TABLE IF EXISTS temp.ultimas_coletas_identidade")
    source.execute(
        """
        CREATE TEMP TABLE ultimas_coletas_identidade AS
        WITH versoes_identidade AS (
            SELECT
                chave_identidade,
                COUNT(*) AS quantidade_versoes
            FROM cotacoes
            GROUP BY chave_identidade
        ),
        identidade_coleta AS (
            SELECT
                c.chave_identidade,
                p.coleta_id,
                co.baixado_em,
                co.processado_em
            FROM cotacao_proveniencias p
            JOIN cotacoes c ON c.id = p.cotacao_id
            JOIN coletas co ON co.id = p.coleta_id
            GROUP BY c.chave_identidade, p.coleta_id
        ),
        coletas_ordenadas AS (
            SELECT
                identidade_coleta.*,
                ROW_NUMBER() OVER (
                    PARTITION BY chave_identidade
                    ORDER BY
                        COALESCE(NULLIF(baixado_em, ''), processado_em) DESC,
                        processado_em DESC,
                        coleta_id DESC
                ) AS ordem
            FROM identidade_coleta
        )
        SELECT
            ranked.chave_identidade,
            ranked.coleta_id,
            ranked.baixado_em,
            ranked.processado_em,
            versions.quantidade_versoes
        FROM coletas_ordenadas ranked
        JOIN versoes_identidade versions
          ON versions.chave_identidade = ranked.chave_identidade
        WHERE ranked.ordem = 1
        """
    )
    source.execute(
        """
        CREATE UNIQUE INDEX temp.idx_ultimas_coletas_identidade
            ON ultimas_coletas_identidade(chave_identidade)
        """
    )


SOURCE_QUERY = """
SELECT
    c.id AS cotacao_id,
    c.chave_identidade,
    latest.coleta_id,
    latest.baixado_em AS coleta_baixada_em,
    latest.processado_em AS coleta_processada_em,
    c.data_cotacao,
    ce.id AS ceasa_id,
    ce.slug AS ceasa_slug,
    es.uf,
    ep.id AS entreposto_id,
    ep.slug AS entreposto_slug,
    ep.nome AS entreposto_nome,
    c.categoria_id,
    cat.slug AS categoria_slug,
    p.id AS produto_id,
    pa.id AS produto_alias_id,
    pa.nome_original AS produto_original,
    p.nome_normalizado AS produto_normalizado_banco,
    c.apresentacao_unidade_id,
    u.sigla AS unidade_sigla,
    au.unidade_original,
    au.unidade_normalizada,
    au.embalagem,
    au.quantidade_minima,
    au.quantidade_maxima,
    au.detalhe_unidade,
    c.procedencia,
    c.classificacao,
    c.situacao_mercado,
    c.preco_minimo AS preco_minimo_original,
    c.preco_comum AS preco_comum_original,
    c.preco_maximo AS preco_maximo_original,
    c.fonte_complemento,
    c.url_complemento,
    c.data_complemento,
    latest.quantidade_versoes AS versoes_identidade
FROM cotacoes c
JOIN ultimas_coletas_identidade latest
  ON latest.chave_identidade = c.chave_identidade
LEFT JOIN entrepostos ep ON ep.id = c.entreposto_id
LEFT JOIN ceasas ce ON ce.id = ep.ceasa_id
LEFT JOIN estados es ON es.id = ce.estado_id
LEFT JOIN categorias cat ON cat.id = c.categoria_id
LEFT JOIN produto_aliases pa ON pa.id = c.produto_alias_id
LEFT JOIN produtos p ON p.id = pa.produto_id
LEFT JOIN apresentacoes_unidade au ON au.id = c.apresentacao_unidade_id
LEFT JOIN unidades u ON u.id = au.unidade_id
WHERE EXISTS (
    SELECT 1
    FROM cotacao_proveniencias provenance
    WHERE provenance.cotacao_id = c.id
      AND provenance.coleta_id = latest.coleta_id
)
ORDER BY c.chave_identidade, c.id
"""


def processar_chunk(
    chunk: pd.DataFrame,
    catalog: pd.DataFrame,
    product_keys: dict[object, object],
    active_rules: pd.DataFrame,
) -> pd.DataFrame:
    numeric_prices = (
        "preco_minimo_original",
        "preco_comum_original",
        "preco_maximo_original",
    )
    for column in numeric_prices:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    chunk["produto_chave_textual"] = chunk["produto_alias_id"].map(product_keys)
    missing_key = chunk["produto_chave_textual"].isna()
    chunk.loc[missing_key, "produto_chave_textual"] = chunk.loc[
        missing_key, "produto_normalizado_banco"
    ].map(normalizar_texto)
    missing_key = chunk["produto_chave_textual"].isna()
    chunk.loc[missing_key, "produto_chave_textual"] = chunk.loc[
        missing_key, "produto_original"
    ].map(normalizar_texto)
    chunk["produto_valido"] = chunk["produto_chave_textual"].notna().astype(int)
    chunk["origem_dado"] = np.where(
        chunk["fonte_complemento"].notna()
        & chunk["fonte_complemento"].astype(str).str.strip().ne(""),
        "prohort",
        "crawler",
    )

    chunk["preco_comum_tratado"] = chunk["preco_comum_original"].where(
        chunk["preco_comum_original"] > 0
    )
    chunk["preco_minimo_tratado"] = chunk["preco_minimo_original"].where(
        chunk["preco_minimo_original"] > 0
    )
    chunk["preco_maximo_tratado"] = chunk["preco_maximo_original"].where(
        chunk["preco_maximo_original"] > 0
    )

    invalid_minimum = (
        chunk["preco_minimo_tratado"].notna()
        & chunk["preco_comum_tratado"].notna()
        & (chunk["preco_minimo_tratado"] > chunk["preco_comum_tratado"])
    )
    invalid_maximum = (
        chunk["preco_maximo_tratado"].notna()
        & chunk["preco_comum_tratado"].notna()
        & (chunk["preco_comum_tratado"] > chunk["preco_maximo_tratado"])
    )
    chunk["preco_comum_valido"] = chunk["preco_comum_tratado"].notna().astype(int)
    chunk["ordem_precos_inconsistente"] = (
        invalid_minimum | invalid_maximum
    ).astype(int)
    inconsistent_order = chunk["ordem_precos_inconsistente"].eq(1)
    chunk.loc[
        inconsistent_order,
        ["preco_minimo_tratado", "preco_maximo_tratado"],
    ] = np.nan

    chunk = chunk.merge(
        catalog[FACT_RULE_COLUMNS],
        on="apresentacao_unidade_id",
        how="left",
    )
    chunk["tipo_conversao"] = chunk["tipo_conversao"].fillna("incerta")
    chunk["dimensao_mapeada"] = chunk["dimensao_mapeada"].fillna(
        "desconhecida"
    )
    chunk["status_mapeamento"] = chunk["status_mapeamento"].fillna(
        "sem_apresentacao"
    )
    chunk["observacao_conversao"] = chunk["observacao_conversao"].fillna(
        "Cotacao sem apresentacao de unidade associada."
    )
    chunk["nivel_confianca"] = chunk["nivel_confianca"].fillna("pendente")
    chunk["origem_regra"] = chunk["origem_regra"].fillna("sem_regra")

    if active_rules.empty:
        chunk["regra_inferencia_id"] = pd.NA
    else:
        chunk = chunk.merge(
            active_rules,
            on=[
                "produto_chave_textual",
                "apresentacao_unidade_id",
                "origem_dado",
            ],
            how="left",
        )
        inferred = chunk["regra_inferencia_id"].notna() & chunk[
            "peso_total_kg"
        ].isna()
        inferred_weight = chunk.loc[inferred, "peso_total_kg_inferido"]
        chunk.loc[inferred, "dimensao_mapeada"] = "massa_inferida"
        chunk.loc[inferred, "unidade_base"] = "kg"
        chunk.loc[inferred, "quantidade_unidades_base"] = inferred_weight
        chunk.loc[inferred, "peso_minimo_kg"] = inferred_weight
        chunk.loc[inferred, "peso_maximo_kg"] = inferred_weight
        chunk.loc[inferred, "peso_total_kg"] = inferred_weight
        chunk.loc[inferred, "tipo_conversao"] = "inferida_alta_confianca"
        chunk.loc[inferred, "status_mapeamento"] = "convertida_inferida"
        chunk.loc[inferred, "nivel_confianca"] = "alta"
        chunk.loc[inferred, "origem_regra"] = "inferencia_base_treino"
        inferred_rows = chunk.loc[
            inferred,
            [
                "peso_total_kg_inferido",
                "regra_inferencia_id",
                "suporte_ancora",
                "datas_ancora",
                "ceasas_ancora",
                "data_limite_inferencia",
            ],
        ]
        chunk.loc[inferred, "formula_conversao"] = [
            f"peso_total_kg = {weight:g} kg (regra inferida)"
            for weight in inferred_rows["peso_total_kg_inferido"]
        ]
        chunk.loc[inferred, "evidencia_mapeamento"] = [
            f"regra_inferencia_id={int(rule_id)}"
            for rule_id in inferred_rows["regra_inferencia_id"]
        ]
        chunk.loc[inferred, "observacao_conversao"] = [
            (
                "Regra inferida com "
                f"{int(support)} âncoras e "
                f"{int(dates)} datas em "
                f"{int(ceasas)} CEASAs até {limit}."
            )
            for support, dates, ceasas, limit in zip(
                inferred_rows["suporte_ancora"],
                inferred_rows["datas_ancora"],
                inferred_rows["ceasas_ancora"],
                inferred_rows["data_limite_inferencia"],
                strict=True,
            )
        ]
        chunk = chunk.drop(
            columns=[
                "peso_total_kg_inferido",
                "suporte_ancora",
                "datas_ancora",
                "ceasas_ancora",
                "data_limite_inferencia",
            ]
        )

    valid_base_quantity = chunk["quantidade_unidades_base"].notna() & (
        chunk["quantidade_unidades_base"] > 0
    )
    valid_item_quantity = chunk["quantidade_itens"].notna() & (
        chunk["quantidade_itens"] > 0
    )
    valid_weight = chunk["peso_total_kg"].notna() & (chunk["peso_total_kg"] > 0)
    for price_name in ("minimo", "comum", "maximo"):
        treated_column = f"preco_{price_name}_tratado"
        chunk[f"preco_{price_name}_unidade_base"] = np.where(
            valid_base_quantity & chunk[treated_column].notna(),
            chunk[treated_column] / chunk["quantidade_unidades_base"],
            np.nan,
        )
        chunk[f"preco_{price_name}_item"] = np.where(
            valid_item_quantity & chunk[treated_column].notna(),
            chunk[treated_column] / chunk["quantidade_itens"],
            np.nan,
        )
        chunk[f"preco_{price_name}_kg"] = np.where(
            valid_weight & chunk[treated_column].notna(),
            chunk[treated_column] / chunk["peso_total_kg"],
            np.nan,
        )
    chunk["conversao_kg_disponivel"] = valid_weight.astype(int)

    chunk["dimensoes_validas"] = (
        chunk[["ceasa_id", "entreposto_id", "produto_id"]]
        .notna()
        .all(axis=1)
        .astype(int)
    )
    chunk["registro_status"] = np.select(
        [
            chunk["produto_valido"].eq(0),
            chunk["preco_comum_valido"].eq(0),
            chunk["dimensoes_validas"].eq(0),
            chunk["conversao_kg_disponivel"].eq(1),
            chunk["preco_comum_unidade_base"].notna(),
        ],
        [
            "produto_invalido",
            "preco_comum_invalido",
            "dimensoes_incompletas",
            "convertido_kg",
            "preco_unidade_nativa",
        ],
        default="conversao_indisponivel",
    )

    chunk["outlier_suspeito"] = 0
    chunk["outlier_status"] = "nao_suspeito"
    chunk["outlier_razao_mediana"] = np.nan

    for column in (
        "ceasa_id",
        "entreposto_id",
        "categoria_id",
        "produto_id",
        "produto_alias_id",
        "apresentacao_unidade_id",
        "regra_inferencia_id",
    ):
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Int64")

    chunk = chunk.rename(
        columns={
            "preco_minimo_original": "preco_minimo_consolidado",
            "preco_comum_original": "preco_comum_consolidado",
            "preco_maximo_original": "preco_maximo_consolidado",
        }
    )
    return chunk.reindex(columns=FACT_COLUMNS)


def consolidar_cotacoes_brutas(chunk: pd.DataFrame) -> pd.DataFrame:
    """Gera uma linha por identidade usando a última coleta e mediana dos preços."""
    if chunk.empty:
        return chunk.copy()

    order_column = "__ordem_consolidacao"
    working = chunk.copy()
    working[order_column] = np.arange(len(working))
    repeated = working.duplicated("chave_identidade", keep=False)

    singletons = working.loc[~repeated].copy()
    singletons["versoes_identidade"] = pd.to_numeric(
        singletons["versoes_identidade"], errors="raise"
    ).astype(int)
    singletons["houve_conflito_versoes"] = singletons[
        "versoes_identidade"
    ].gt(1).astype(int)

    price_columns = (
        "preco_minimo_original",
        "preco_comum_original",
        "preco_maximo_original",
    )
    repeated_rows = working.loc[repeated]
    consolidated_repeated = repeated_rows.drop_duplicates(
        "chave_identidade",
        keep="first",
    ).copy()
    for column in price_columns:
        numeric_values = pd.to_numeric(repeated_rows[column], errors="coerce")
        medians = numeric_values.groupby(
            repeated_rows["chave_identidade"],
            sort=False,
            dropna=False,
        ).median()
        consolidated_repeated[column] = consolidated_repeated[
            "chave_identidade"
        ].map(medians)

    market_situations: dict[object, object] = {}
    for identity, versions in repeated_rows.groupby(
        "chave_identidade",
        sort=False,
        dropna=False,
    ):
        situations = [
            None if pd.isna(value) else str(value)
            for value in versions["situacao_mercado"]
        ]
        market_situations[identity] = (
            situations[0] if len(set(situations)) == 1 else None
        )
    consolidated_repeated["situacao_mercado"] = consolidated_repeated[
        "chave_identidade"
    ].map(market_situations)
    consolidated_repeated["versoes_identidade"] = pd.to_numeric(
        consolidated_repeated["versoes_identidade"],
        errors="raise",
    ).astype(int)
    consolidated_repeated["houve_conflito_versoes"] = consolidated_repeated[
        "versoes_identidade"
    ].gt(1).astype(int)

    parts = [singletons]
    if not consolidated_repeated.empty:
        parts.append(consolidated_repeated)

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(order_column, kind="stable")
        .drop(columns=order_column)
        .reset_index(drop=True)
    )


def inserir_cotacoes(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    catalog: pd.DataFrame,
    product_aliases: pd.DataFrame,
    inference_rules: pd.DataFrame,
    chunk_size: int,
) -> dict[str, int]:
    total_selected = 0
    total_consolidated = 0
    total_persisted = 0
    total_without_unit = 0
    total_without_exact_quantity = 0
    total_ambiguous = 0
    conflicts = 0
    pending = pd.DataFrame()
    product_keys = dict(
        product_aliases.loc[
            product_aliases["produto_chave_textual"].notna(),
            ["produto_alias_id", "produto_chave_textual"],
        ].itertuples(index=False, name=None)
    )
    rule_columns = [
        "id",
        "produto_chave_textual",
        "apresentacao_unidade_id",
        "origem_dado",
        "peso_total_kg_inferido",
        "suporte_ancora",
        "datas_ancora",
        "ceasas_ancora",
        "data_limite_inferencia",
    ]
    active_rules = (
        inference_rules.loc[
            inference_rules["status_regra"].eq("aplicada"),
            rule_columns,
        ]
        .rename(columns={"id": "regra_inferencia_id"})
        .copy()
    )

    def persist(raw: pd.DataFrame) -> None:
        nonlocal total_consolidated
        nonlocal total_persisted
        nonlocal total_without_unit
        nonlocal total_without_exact_quantity
        nonlocal total_ambiguous
        nonlocal conflicts
        if raw.empty:
            return
        consolidated = consolidar_cotacoes_brutas(raw)
        processed = processar_chunk(
            consolidated,
            catalog,
            product_keys,
            active_rules,
        )
        total_consolidated += len(processed)
        mapping_status = processed["status_mapeamento"]
        without_unit = mapping_status.eq("sem_unidade")
        without_exact_quantity = mapping_status.eq("sem_quantidade_exata")
        ambiguous = mapping_status.eq("ambigua")
        excluded_mapping = mapping_status.isin(EXCLUDED_MAPPING_STATUSES)
        total_without_unit += int(without_unit.sum())
        total_without_exact_quantity += int(without_exact_quantity.sum())
        total_ambiguous += int(ambiguous.sum())
        excluded = processed.loc[excluded_mapping, EXCLUDED_COLUMNS].copy()
        if not excluded.empty:
            excluded.to_sql(
                "cotacoes_excluidas_mapeamento",
                target,
                if_exists="append",
                index=False,
                chunksize=SQLITE_WRITE_BATCH_SIZE,
            )
        processed = processed.loc[~excluded_mapping].copy()
        total_persisted += len(processed)
        conflicts += int(processed["houve_conflito_versoes"].sum())
        if processed.empty:
            return
        processed.to_sql(
            "cotacoes_preprocessadas",
            target,
            if_exists="append",
            index=False,
            chunksize=SQLITE_WRITE_BATCH_SIZE,
        )

    for chunk in pd.read_sql_query(SOURCE_QUERY, source, chunksize=chunk_size):
        total_selected += len(chunk)
        if not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        last_identity = chunk["chave_identidade"].iloc[-1]
        last_group = chunk["chave_identidade"].eq(last_identity)
        pending = chunk.loc[last_group].copy()
        persist(chunk.loc[~last_group].copy())
        print(
            f"Cotações selecionadas: {total_selected:,}; "
            f"identidades consolidadas: {total_consolidated:,}; "
            f"cotações gravadas: {total_persisted:,}",
            flush=True,
        )

    persist(pending)
    return {
        "cotacoes_selecionadas_ultima_coleta": total_selected,
        "cotacoes_consolidadas": total_consolidated,
        "cotacoes_gravadas": total_persisted,
        "cotacoes_excluidas_mapeamento": (
            total_without_unit + total_without_exact_quantity + total_ambiguous
        ),
        "cotacoes_excluidas_sem_unidade": total_without_unit,
        "cotacoes_excluidas_sem_quantidade_exata": total_without_exact_quantity,
        "cotacoes_excluidas_ambiguidade": total_ambiguous,
        "identidades_com_multiplas_versoes": conflicts,
    }


def marcar_outliers(
    connection: sqlite3.Connection,
    low: float,
    high: float,
    minimum_group: int,
    probable_low: float,
    probable_high: float,
    recurrent_minimum: int,
) -> int:
    comparable = pd.read_sql_query(
        """
        SELECT
            cotacao_id,
            produto_id,
            entreposto_id,
            apresentacao_unidade_id,
            preco_comum_consolidado,
            preco_comum_kg
        FROM cotacoes_preprocessadas
        WHERE preco_comum_kg IS NOT NULL AND preco_comum_kg > 0
        """,
        connection,
    )
    if comparable.empty:
        return 0

    keys = ["produto_id", "entreposto_id", "apresentacao_unidade_id"]
    grouped = comparable.groupby(keys, dropna=False)["preco_comum_kg"]
    comparable["mediana_grupo"] = grouped.transform("median")
    comparable["tamanho_grupo"] = grouped.transform("size")
    comparable["razao_mediana"] = (
        comparable["preco_comum_kg"]
        / comparable["mediana_grupo"].replace(0, np.nan)
    )
    flagged = comparable[
        comparable["tamanho_grupo"].ge(minimum_group)
        & (
            comparable["razao_mediana"].lt(low)
            | comparable["razao_mediana"].gt(high)
        )
    ].copy()
    if flagged.empty:
        return 0

    flagged["outliers_no_grupo"] = flagged.groupby(keys, dropna=False)[
        "cotacao_id"
    ].transform("size")
    price_is_cent = np.isclose(
        flagged["preco_comum_consolidado"],
        0.01,
        rtol=0,
        atol=1e-9,
    )
    extreme_ratio = flagged["razao_mediana"].lt(probable_low) | flagged[
        "razao_mediana"
    ].gt(probable_high)
    recurrent_group = flagged["outliers_no_grupo"].ge(recurrent_minimum)
    flagged["metodo"] = "razao_mediana_produto_entreposto_apresentacao"
    flagged["classificacao"] = np.select(
        [price_is_cent | extreme_ratio, recurrent_group],
        ["erro_provavel", "inconsistencia_recorrente"],
        default="extremo_isolado",
    )
    flagged["observacao"] = np.select(
        [
            price_is_cent,
            extreme_ratio,
            recurrent_group,
        ],
        [
            "Preço comum consolidado igual a R$ 0,01.",
            "Razão em relação à mediana ultrapassa o limite de erro provável.",
            "Ocorrência recorrente no mesmo produto, entreposto e apresentação; possível inconsistência no denominador do preço.",
        ],
        default="Ocorrência extrema isolada ou pouco recorrente, desconsiderada da análise principal.",
    )
    flagged[
        [
            "cotacao_id",
            "metodo",
            "tamanho_grupo",
            "outliers_no_grupo",
            "mediana_grupo",
            "razao_mediana",
            "classificacao",
            "observacao",
        ]
    ].to_sql(
        "outliers_suspeitos",
        connection,
        if_exists="append",
        index=False,
        chunksize=SQLITE_WRITE_BATCH_SIZE,
    )
    connection.execute(
        """
        UPDATE cotacoes_preprocessadas
        SET
            outlier_suspeito = 1,
            outlier_status = (
                SELECT outlier.classificacao
                FROM outliers_suspeitos outlier
                WHERE outlier.cotacao_id = cotacoes_preprocessadas.cotacao_id
            ),
            outlier_razao_mediana = (
                SELECT outlier.razao_mediana
                FROM outliers_suspeitos outlier
                WHERE outlier.cotacao_id = cotacoes_preprocessadas.cotacao_id
            )
        WHERE cotacao_id IN (SELECT cotacao_id FROM outliers_suspeitos)
        """
    )
    return len(flagged)


def criar_indices_e_views(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX idx_cotacoes_preprocessadas_produto_data
            ON cotacoes_preprocessadas(produto_id, data_cotacao);
        CREATE INDEX idx_cotacoes_preprocessadas_entreposto_produto_data
            ON cotacoes_preprocessadas(entreposto_id, produto_id, data_cotacao);
        CREATE INDEX idx_cotacoes_preprocessadas_ceasa_produto_data
            ON cotacoes_preprocessadas(ceasa_id, produto_id, data_cotacao);
        CREATE INDEX idx_cotacoes_preprocessadas_conversao
            ON cotacoes_preprocessadas(tipo_conversao, conversao_kg_disponivel);
        CREATE INDEX idx_cotacoes_preprocessadas_apresentacao
            ON cotacoes_preprocessadas(apresentacao_unidade_id);
        CREATE INDEX idx_cotacoes_preprocessadas_outlier
            ON cotacoes_preprocessadas(outlier_suspeito);
        CREATE INDEX idx_cotacoes_preprocessadas_outlier_status
            ON cotacoes_preprocessadas(outlier_status);
        CREATE INDEX idx_cotacoes_preprocessadas_produto_valido
            ON cotacoes_preprocessadas(produto_valido);
        CREATE INDEX idx_cotacoes_preprocessadas_status_mapeamento
            ON cotacoes_preprocessadas(status_mapeamento);
        CREATE INDEX idx_catalogo_apresentacoes_status
            ON catalogo_apresentacoes(status_mapeamento);
        CREATE INDEX idx_catalogo_apresentacoes_dimensao
            ON catalogo_apresentacoes(dimensao_mapeada);
        CREATE INDEX idx_regras_inferencia_status
            ON regras_conversao_inferida(status_regra);
        CREATE INDEX idx_regras_inferencia_contexto
            ON regras_conversao_inferida(
                produto_chave_textual, apresentacao_unidade_id, origem_dado
            );
        CREATE INDEX idx_cotacoes_excluidas_status
            ON cotacoes_excluidas_mapeamento(status_mapeamento);
        CREATE INDEX idx_cotacoes_excluidas_apresentacao
            ON cotacoes_excluidas_mapeamento(apresentacao_unidade_id);
        CREATE INDEX idx_outliers_classificacao
            ON outliers_suspeitos(classificacao);

        CREATE VIEW cotacoes_analiticas AS
        SELECT *
        FROM cotacoes_preprocessadas
        WHERE produto_valido = 1
          AND preco_comum_valido = 1
          AND dimensoes_validas = 1;

        CREATE VIEW cotacoes_analiticas_ordem_consistente AS
        SELECT *
        FROM cotacoes_analiticas
        WHERE ordem_precos_inconsistente = 0;

        CREATE VIEW cotacoes_analiticas_principais AS
        SELECT *
        FROM cotacoes_analiticas
        WHERE outlier_suspeito = 0;

        CREATE VIEW cotacoes_convertidas_kg AS
        SELECT *
        FROM cotacoes_analiticas
        WHERE peso_total_kg IS NOT NULL;

        CREATE VIEW cotacoes_com_unidade_base AS
        SELECT *
        FROM cotacoes_analiticas
        WHERE quantidade_unidades_base IS NOT NULL;

        CREATE VIEW cotacoes_conversao_pendente AS
        SELECT *
        FROM cotacoes_analiticas
        WHERE tipo_conversao = 'incerta';

        CREATE VIEW catalogo_apresentacoes_revisao_pendente AS
        SELECT *
        FROM catalogo_apresentacoes
        WHERE status_mapeamento IN (
            'ambigua',
            'sem_unidade',
            'sem_quantidade_exata'
        );

        CREATE VIEW catalogo_apresentacoes_intervalos AS
        SELECT *
        FROM catalogo_apresentacoes
        WHERE status_mapeamento IN ('intervalo_massa', 'intervalo_quantidade');

        CREATE VIEW outliers_desconsiderados_analise AS
        SELECT
            quote.*,
            outlier.metodo,
            outlier.tamanho_grupo,
            outlier.outliers_no_grupo,
            outlier.mediana_grupo,
            outlier.classificacao,
            outlier.observacao
        FROM cotacoes_analiticas quote
        JOIN outliers_suspeitos outlier ON outlier.cotacao_id = quote.cotacao_id
        WHERE quote.outlier_suspeito = 1;

        CREATE VIEW outliers_erros_provaveis AS
        SELECT
            quote.*,
            outlier.metodo,
            outlier.tamanho_grupo,
            outlier.outliers_no_grupo,
            outlier.mediana_grupo,
            outlier.classificacao,
            outlier.observacao
        FROM cotacoes_analiticas quote
        JOIN outliers_suspeitos outlier ON outlier.cotacao_id = quote.cotacao_id
        WHERE outlier.classificacao = 'erro_provavel';
        """
    )


def consultar_metricas(
    connection: sqlite3.Connection,
    query: str,
) -> dict[str, int]:
    cursor = connection.execute(query)
    row = cursor.fetchone()
    if row is None:
        return {}
    return {
        description[0]: int(value or 0)
        for description, value in zip(cursor.description, row, strict=True)
    }


def gerar_resumo(connection: sqlite3.Connection) -> dict[str, int]:
    summary = consultar_metricas(
        connection,
        """
        SELECT
            COUNT(*) AS cotacoes_total,
            SUM(CASE WHEN preco_comum_valido = 1 THEN 1 ELSE 0 END)
                AS preco_comum_valido,
            SUM(CASE WHEN produto_valido = 0 THEN 1 ELSE 0 END)
                AS produtos_invalidos,
            SUM(CASE WHEN ordem_precos_inconsistente = 1 THEN 1 ELSE 0 END)
                AS ordem_precos_inconsistente,
            SUM(CASE
                WHEN produto_valido = 1
                 AND preco_comum_valido = 1
                 AND dimensoes_validas = 1
                THEN 1 ELSE 0 END) AS cotacoes_analiticas,
            SUM(CASE
                WHEN produto_valido = 1
                 AND preco_comum_valido = 1
                 AND dimensoes_validas = 1
                 AND outlier_suspeito = 0
                THEN 1 ELSE 0 END) AS cotacoes_analiticas_principais,
            SUM(CASE WHEN tipo_conversao = 'exata' THEN 1 ELSE 0 END)
                AS conversao_exata,
            SUM(CASE WHEN tipo_conversao = 'calculada' THEN 1 ELSE 0 END)
                AS conversao_calculada,
            SUM(CASE WHEN tipo_conversao = 'padronizada' THEN 1 ELSE 0 END)
                AS conversao_padronizada,
            SUM(CASE WHEN tipo_conversao = 'inferida_alta_confianca'
                THEN 1 ELSE 0 END) AS conversao_inferida_alta_confianca,
            SUM(CASE WHEN tipo_conversao = 'incerta' THEN 1 ELSE 0 END)
                AS conversao_incerta,
            SUM(CASE WHEN preco_comum_kg IS NOT NULL THEN 1 ELSE 0 END)
                AS preco_comum_kg_disponivel,
            SUM(CASE WHEN preco_comum_unidade_base IS NOT NULL THEN 1 ELSE 0 END)
                AS preco_comum_unidade_base_disponivel,
            SUM(CASE WHEN quantidade_itens IS NOT NULL THEN 1 ELSE 0 END)
                AS quantidade_itens_disponivel,
            SUM(CASE WHEN preco_comum_item IS NOT NULL THEN 1 ELSE 0 END)
                AS preco_comum_item_disponivel,
            SUM(CASE WHEN origem_dado = 'prohort' THEN 1 ELSE 0 END)
                AS origem_prohort,
            SUM(CASE WHEN houve_conflito_versoes = 1 THEN 1 ELSE 0 END)
                AS identidades_multiplas_versoes,
            COALESCE(SUM(versoes_identidade - 1), 0)
                AS versoes_removidas_na_consolidacao
        FROM cotacoes_preprocessadas
        """,
    )
    summary.update(
        consultar_metricas(
            connection,
            """
            SELECT COUNT(*) AS cotacoes_excluidas_auditaveis
            FROM cotacoes_excluidas_mapeamento
            """,
        )
    )
    summary.update(
        consultar_metricas(
            connection,
            """
            SELECT
                COUNT(*) AS apresentacoes_catalogadas,
                SUM(CASE WHEN status_mapeamento = 'unidade_nativa'
                    THEN 1 ELSE 0 END) AS apresentacoes_unidade_nativa,
                SUM(CASE WHEN status_mapeamento IN (
                    'intervalo_massa', 'intervalo_quantidade'
                ) THEN 1 ELSE 0 END) AS apresentacoes_intervalo
            FROM catalogo_apresentacoes
            """,
        )
    )
    summary.update(
        consultar_metricas(
            connection,
            """
            SELECT
                COUNT(*) AS outliers_suspeitos,
                SUM(CASE WHEN classificacao = 'erro_provavel'
                    THEN 1 ELSE 0 END) AS outliers_erro_provavel,
                SUM(CASE WHEN classificacao = 'inconsistencia_recorrente'
                    THEN 1 ELSE 0 END) AS outliers_inconsistencia_recorrente,
                SUM(CASE WHEN classificacao = 'extremo_isolado'
                    THEN 1 ELSE 0 END) AS outliers_extremo_isolado
            FROM outliers_suspeitos
            """,
        )
    )
    summary.update(
        consultar_metricas(
            connection,
            """
            SELECT
                COUNT(*) AS regras_inferencia_total,
                SUM(CASE WHEN status_regra = 'candidata'
                    THEN 1 ELSE 0 END) AS regras_inferencia_candidatas,
                SUM(CASE WHEN status_regra = 'aplicada'
                    THEN 1 ELSE 0 END) AS regras_inferencia_aplicadas
            FROM regras_conversao_inferida
            """,
        )
    )
    connection.executemany(
        "INSERT INTO resumo_preprocessamento(indicador, valor) VALUES (?, ?)",
        [(name, str(value)) for name, value in summary.items()],
    )
    return summary


def validar_saida(
    connection: sqlite3.Connection,
    total_esperado: int,
    apresentacoes_esperadas: int,
    excluidas_esperadas: int,
) -> None:
    problemas: list[str] = []

    integridade = connection.execute("PRAGMA quick_check").fetchall()
    if integridade != [("ok",)]:
        problemas.append(f"PRAGMA quick_check retornou: {integridade!r}")

    fact_metrics = consultar_metricas(
        connection,
        """
        SELECT
            COUNT(*) AS total_gravado,
            SUM(CASE
                WHEN quote.conversao_kg_disponivel <> CASE
                    WHEN quote.peso_total_kg IS NOT NULL
                     AND quote.peso_total_kg > 0 THEN 1
                    ELSE 0
                END
                THEN 1 ELSE 0 END) AS conversoes_inconsistentes,
            SUM(CASE
                WHEN (quote.produto_valido = 0
                      AND quote.registro_status <> 'produto_invalido')
                  OR (quote.produto_valido = 1
                      AND quote.registro_status = 'produto_invalido')
                THEN 1 ELSE 0 END) AS produtos_inconsistentes,
            SUM(CASE
                WHEN quote.ordem_precos_inconsistente = 1
                 AND (
                    quote.preco_minimo_tratado IS NOT NULL
                    OR quote.preco_maximo_tratado IS NOT NULL
                    OR quote.preco_minimo_unidade_base IS NOT NULL
                    OR quote.preco_maximo_unidade_base IS NOT NULL
                    OR quote.preco_minimo_item IS NOT NULL
                    OR quote.preco_maximo_item IS NOT NULL
                    OR quote.preco_minimo_kg IS NOT NULL
                    OR quote.preco_maximo_kg IS NOT NULL
                 )
                THEN 1 ELSE 0 END) AS precos_inconsistentes_expostos,
            SUM(CASE
                WHEN quote.outlier_suspeito <> CASE
                    WHEN outlier.cotacao_id IS NULL THEN 0 ELSE 1 END
                THEN 1 ELSE 0 END) AS outliers_inconsistentes,
            SUM(CASE
                WHEN quote.outlier_status <> CASE
                    WHEN outlier.cotacao_id IS NULL THEN 'nao_suspeito'
                    ELSE outlier.classificacao
                END
                THEN 1 ELSE 0 END) AS status_outliers_inconsistentes,
            SUM(CASE
                WHEN quote.versoes_identidade < 1
                  OR quote.houve_conflito_versoes <> CASE
                    WHEN quote.versoes_identidade > 1 THEN 1 ELSE 0 END
                THEN 1 ELSE 0 END) AS versoes_inconsistentes,
            SUM(CASE
                WHEN (quote.tipo_conversao = 'inferida_alta_confianca'
                      AND (rule.id IS NULL OR rule.status_regra <> 'aplicada'))
                  OR (quote.regra_inferencia_id IS NOT NULL
                      AND quote.tipo_conversao <> 'inferida_alta_confianca')
                THEN 1 ELSE 0 END) AS inferencias_inconsistentes,
            SUM(CASE
                WHEN quote.status_mapeamento IN (
                    'ambigua', 'sem_quantidade_exata', 'sem_unidade'
                ) THEN 1 ELSE 0 END) AS cotacoes_mapeamento_excluido
        FROM cotacoes_preprocessadas quote
        LEFT JOIN outliers_suspeitos outlier
          ON outlier.cotacao_id = quote.cotacao_id
        LEFT JOIN regras_conversao_inferida rule
          ON rule.id = quote.regra_inferencia_id
        """,
    )
    total_gravado = fact_metrics["total_gravado"]
    if total_gravado != total_esperado:
        problemas.append(
            f"esperadas {total_esperado} cotacoes, mas foram gravadas {total_gravado}"
        )

    total_excluido = int(
        connection.execute(
            "SELECT COUNT(*) FROM cotacoes_excluidas_mapeamento"
        ).fetchone()[0]
    )
    if total_excluido != excluidas_esperadas:
        problemas.append(
            f"esperadas {excluidas_esperadas} cotacoes excluidas auditaveis, "
            f"mas foram gravadas {total_excluido}"
        )

    total_catalogado = int(
        connection.execute("SELECT COUNT(*) FROM catalogo_apresentacoes").fetchone()[0]
    )
    if total_catalogado != apresentacoes_esperadas:
        problemas.append(
            f"esperadas {apresentacoes_esperadas} apresentacoes no catalogo, "
            f"mas foram gravadas {total_catalogado}"
        )

    cobertura_catalogo = connection.execute(
        """
        SELECT
            (SELECT COALESCE(SUM(cotacoes_associadas), 0)
             FROM catalogo_apresentacoes),
            (SELECT COUNT(*)
             FROM cotacoes_preprocessadas
             WHERE apresentacao_unidade_id IS NOT NULL)
        """
    ).fetchone()
    if cobertura_catalogo[0] != cobertura_catalogo[1]:
        problemas.append(
            "a cobertura registrada no catalogo difere das cotacoes com apresentacao"
        )

    cobertura_exclusoes = connection.execute(
        """
        SELECT
            (SELECT COALESCE(SUM(cotacoes_excluidas_associadas), 0)
             FROM catalogo_apresentacoes),
            (SELECT COUNT(*)
             FROM cotacoes_excluidas_mapeamento
             WHERE apresentacao_unidade_id IS NOT NULL)
        """
    ).fetchone()
    if cobertura_exclusoes[0] != cobertura_exclusoes[1]:
        problemas.append(
            "a cobertura de exclusoes no catalogo difere da tabela de auditoria"
        )

    unidades_nativas_incorretas = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM catalogo_apresentacoes
            WHERE UPPER(TRIM(unidade_original)) IN ('UN', 'UNID.', 'DZ', 'CENTO', 'MOLHO')
              AND status_mapeamento NOT IN ('unidade_nativa', 'intervalo_quantidade')
            """
        ).fetchone()[0]
    )
    if unidades_nativas_incorretas:
        problemas.append(
            f"{unidades_nativas_incorretas} unidades nativas basicas mal classificadas"
        )

    conversoes_inconsistentes = fact_metrics["conversoes_inconsistentes"]
    if conversoes_inconsistentes:
        problemas.append(
            f"{conversoes_inconsistentes} marcadores de conversao inconsistentes"
        )

    produtos_inconsistentes = fact_metrics["produtos_inconsistentes"]
    if produtos_inconsistentes:
        problemas.append(
            f"{produtos_inconsistentes} marcadores de produto inconsistentes"
        )

    precos_inconsistentes_expostos = fact_metrics[
        "precos_inconsistentes_expostos"
    ]
    if precos_inconsistentes_expostos:
        problemas.append(
            f"{precos_inconsistentes_expostos} limites de preco inconsistentes expostos"
        )

    produtos_invalidos_nas_views = int(
        connection.execute(
            "SELECT COUNT(*) FROM cotacoes_analiticas WHERE produto_valido = 0"
        ).fetchone()[0]
    )
    if produtos_invalidos_nas_views:
        problemas.append(
            f"{produtos_invalidos_nas_views} produtos invalidos na visao analitica"
        )

    outliers_inconsistentes = fact_metrics["outliers_inconsistentes"]
    if outliers_inconsistentes:
        problemas.append(
            f"{outliers_inconsistentes} marcadores de outlier inconsistentes"
        )

    status_outliers_inconsistentes = fact_metrics[
        "status_outliers_inconsistentes"
    ]
    if status_outliers_inconsistentes:
        problemas.append(
            f"{status_outliers_inconsistentes} classificacoes de outlier inconsistentes"
        )

    outliers_na_view_principal = int(
        connection.execute(
            "SELECT COUNT(*) FROM cotacoes_analiticas_principais "
            "WHERE outlier_suspeito = 1"
        ).fetchone()[0]
    )
    if outliers_na_view_principal:
        problemas.append(
            f"{outliers_na_view_principal} outliers na view principal"
        )

    identidades_duplicadas = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT chave_identidade
                FROM cotacoes_preprocessadas
                GROUP BY chave_identidade
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )
    if identidades_duplicadas:
        problemas.append(
            f"{identidades_duplicadas} identidades permaneceram duplicadas"
        )

    versoes_inconsistentes = fact_metrics["versoes_inconsistentes"]
    if versoes_inconsistentes:
        problemas.append(
            f"{versoes_inconsistentes} marcadores de consolidacao inconsistentes"
        )

    inferencias_inconsistentes = fact_metrics["inferencias_inconsistentes"]
    if inferencias_inconsistentes:
        problemas.append(
            f"{inferencias_inconsistentes} conversoes inferidas inconsistentes"
        )

    cotacoes_mapeamento_excluido = fact_metrics[
        "cotacoes_mapeamento_excluido"
    ]
    if cotacoes_mapeamento_excluido:
        problemas.append(
            f"{cotacoes_mapeamento_excluido} cotacoes com mapeamento excluido "
            "permaneceram no banco"
        )

    if problemas:
        raise RuntimeError(
            "Falha na validacao do banco gerado: " + "; ".join(problemas)
        )

    print(
        "Validacao concluida: integridade, contagem e marcadores consistentes.",
        flush=True,
    )


def construir_banco(args: argparse.Namespace) -> Path:
    requested_source = Path(args.database)
    source_path, temporary_source = preparar_origem(requested_source)
    output = Path(args.output_db)
    output.parent.mkdir(parents=True, exist_ok=True)

    if source_path.resolve() == output.resolve():
        raise ValueError("O banco de saída deve ser diferente do banco bruto.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"O banco de saída já existe: {output}. Use --overwrite para substituí-lo."
        )

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}-",
        suffix=".tmp.sqlite",
        dir=output.parent,
        delete=False,
    )
    temporary_output = Path(temporary_handle.name)
    temporary_handle.close()

    started_at = datetime.now(timezone.utc).isoformat()
    parameters = {
        "chunk_size": args.chunk_size,
        "sqlite_write_batch_size": SQLITE_WRITE_BATCH_SIZE,
        "unidade_mercado": MARKET_UNIT,
        "outlier_low": args.outlier_low,
        "outlier_high": args.outlier_high,
        "outlier_min_group": args.outlier_min_group,
        "outlier_probable_low": args.outlier_probable_low,
        "outlier_probable_high": args.outlier_probable_high,
        "outlier_recurrent_min": args.outlier_recurrent_min,
        "data_limite_inferencia": args.data_limite_inferencia,
        "inferencia_min_suporte": args.inferencia_min_suporte,
        "inferencia_min_datas": args.inferencia_min_datas,
    }

    source: sqlite3.Connection | None = None
    target: sqlite3.Connection | None = None
    inicio_total = perf_counter()
    try:
        inicio_etapa = perf_counter()
        source = abrir_origem(source_path)
        catalog = construir_catalogo_apresentacoes(source)
        product_aliases = construir_chaves_produtos(source)
        inicio_etapa = registrar_tempo_etapa(
            "catálogos e chaves de produtos",
            inicio_etapa,
        )
        inference_rules = construir_regras_inferencia(
            source,
            catalog,
            product_aliases,
            args,
        )
        inicio_etapa = registrar_tempo_etapa(
            "regras de inferência",
            inicio_etapa,
        )
        preparar_ultimas_coletas(source)
        total_source = int(source.execute("SELECT COUNT(*) FROM cotacoes").fetchone()[0])
        inicio_etapa = registrar_tempo_etapa(
            "seleção das últimas coletas",
            inicio_etapa,
        )

        target = sqlite3.connect(temporary_output)
        # A saída só é publicada após a validação e pode ser reconstruída em
        # caso de interrupção; o banco bruto e o banco final não são alterados.
        target.execute("PRAGMA journal_mode = OFF")
        target.execute("PRAGMA synchronous = OFF")
        target.execute("PRAGMA locking_mode = EXCLUSIVE")
        criar_esquema(target)
        target.execute(
            """
            INSERT INTO execucoes_preprocessamento (
                schema_version,
                iniciada_em,
                banco_origem,
                parametros_json,
                status
            ) VALUES (?, ?, ?, ?, 'executando')
            """,
            (
                SCHEMA_VERSION,
                started_at,
                str(requested_source),
                json.dumps(parameters, ensure_ascii=False, sort_keys=True),
            ),
        )
        inserir_catalogo_apresentacoes(target, catalog)
        inserir_regras_inferencia(target, inference_rules)
        target.commit()
        inicio_etapa = registrar_tempo_etapa(
            "preparação do banco temporário",
            inicio_etapa,
        )

        processing_stats = inserir_cotacoes(
            source,
            target,
            catalog,
            product_aliases,
            inference_rules,
            args.chunk_size,
        )
        target.commit()
        inicio_etapa = registrar_tempo_etapa(
            "transformação e gravação das cotações",
            inicio_etapa,
        )

        if (
            processing_stats["cotacoes_consolidadas"]
            != processing_stats["cotacoes_gravadas"]
            + processing_stats["cotacoes_excluidas_mapeamento"]
        ):
            raise RuntimeError(
                "Contagem inconsistente entre cotacoes consolidadas, gravadas "
                "e excluidas por mapeamento insuficiente."
            )

        print(
            "Identidades com múltiplas versões consolidadas: "
            f"{processing_stats['identidades_com_multiplas_versoes']:,}",
            flush=True,
        )

        outliers = marcar_outliers(
            target,
            low=args.outlier_low,
            high=args.outlier_high,
            minimum_group=args.outlier_min_group,
            probable_low=args.outlier_probable_low,
            probable_high=args.outlier_probable_high,
            recurrent_minimum=args.outlier_recurrent_min,
        )
        print(f"Outliers apenas sinalizados: {outliers:,}", flush=True)
        target.commit()
        inicio_etapa = registrar_tempo_etapa(
            "detecção e marcação de outliers",
            inicio_etapa,
        )

        criar_indices_e_views(target)
        inicio_etapa = registrar_tempo_etapa(
            "criação de índices e views",
            inicio_etapa,
        )
        atualizar_cobertura_catalogo(target)
        summary = gerar_resumo(target)
        execution_counts = {
            "cotacoes_origem": total_source,
            **processing_stats,
        }
        target.executemany(
            "INSERT OR REPLACE INTO resumo_preprocessamento(indicador, valor) "
            "VALUES (?, ?)",
            [(name, str(value)) for name, value in execution_counts.items()],
        )
        summary.update(execution_counts)
        inicio_etapa = registrar_tempo_etapa(
            "cobertura e resumo",
            inicio_etapa,
        )
        validar_saida(
            target,
            processing_stats["cotacoes_gravadas"],
            len(catalog),
            processing_stats["cotacoes_excluidas_mapeamento"],
        )
        registrar_tempo_etapa("validação final", inicio_etapa)
        target.execute(
            """
            UPDATE execucoes_preprocessamento
            SET concluida_em = ?, status = 'concluida'
            WHERE id = 1
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        target.commit()
        target.close()
        target = None
        source.close()
        source = None

        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"O banco de saída surgiu durante a execução: {output}"
            )
        os.replace(temporary_output, output)

        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Banco pré-processado criado em: {output.resolve()}", flush=True)
        print(
            f"Tempo total do pré-processamento: "
            f"{perf_counter() - inicio_total:.2f} s.",
            flush=True,
        )
        return output
    except Exception:
        if target is not None:
            target.close()
        if source is not None:
            source.close()
        temporary_output.unlink(missing_ok=True)
        raise
    finally:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)


def validar_argumentos(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size deve ser maior que zero.")
    if not 0 < args.outlier_low < 1:
        raise ValueError("--outlier-low deve estar entre zero e um.")
    if args.outlier_high <= 1:
        raise ValueError("--outlier-high deve ser maior que um.")
    if args.outlier_min_group < 2:
        raise ValueError("--outlier-min-group deve ser pelo menos dois.")
    if not 0 < args.outlier_probable_low < args.outlier_low:
        raise ValueError(
            "--outlier-probable-low deve estar entre zero e --outlier-low."
        )
    if args.outlier_probable_high <= args.outlier_high:
        raise ValueError(
            "--outlier-probable-high deve ser maior que --outlier-high."
        )
    if args.outlier_recurrent_min < 2:
        raise ValueError("--outlier-recurrent-min deve ser pelo menos dois.")
    if args.data_limite_inferencia:
        try:
            datetime.strptime(args.data_limite_inferencia, "%Y-%m-%d")
        except ValueError as error:
            raise ValueError(
                "--data-limite-inferencia deve usar o formato AAAA-MM-DD."
            ) from error
    if args.inferencia_min_suporte < 1:
        raise ValueError("--inferencia-min-suporte deve ser pelo menos um.")
    if args.inferencia_min_datas < 1:
        raise ValueError("--inferencia-min-datas deve ser pelo menos um.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", help="SQLite bruto ou arquivo .sqlite.xz")
    parser.add_argument(
        "--output-db",
        default="resultados/preprocessamento/cotacoes_preprocessadas.sqlite",
        help="novo SQLite gerado pelo pré-processamento",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="substitui atomicamente um banco pré-processado existente",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--outlier-low", type=float, default=DEFAULT_OUTLIER_LOW)
    parser.add_argument("--outlier-high", type=float, default=DEFAULT_OUTLIER_HIGH)
    parser.add_argument(
        "--outlier-min-group",
        type=int,
        default=DEFAULT_OUTLIER_MIN_GROUP,
    )
    parser.add_argument(
        "--outlier-probable-low",
        type=float,
        default=DEFAULT_OUTLIER_PROBABLE_LOW,
    )
    parser.add_argument(
        "--outlier-probable-high",
        type=float,
        default=DEFAULT_OUTLIER_PROBABLE_HIGH,
    )
    parser.add_argument(
        "--outlier-recurrent-min",
        type=int,
        default=DEFAULT_OUTLIER_RECURRENT_MIN,
    )
    parser.add_argument(
        "--data-limite-inferencia",
        default=os.getenv("CEASA_DATA_LIMITE_INFERENCIA") or None,
        help=(
            "último dia do período de treino usado para inferir conversões; "
            "sem ele as regras ficam apenas como candidatas"
        ),
    )
    parser.add_argument(
        "--inferencia-min-suporte",
        type=int,
        default=DEFAULT_INFERENCE_MIN_SUPPORT,
    )
    parser.add_argument(
        "--inferencia-min-datas",
        type=int,
        default=DEFAULT_INFERENCE_MIN_DATES,
    )
    args = parser.parse_args(argv)
    validar_argumentos(args)
    construir_banco(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
