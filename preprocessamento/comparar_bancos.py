#!/usr/bin/env python3
"""Compara logicamente dois bancos gerados pelo pré-processamento."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


IGNORED_EXECUTION_COLUMNS = {"iniciada_em", "concluida_em"}


def quote_identifier(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def open_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Banco não encontrado: {path}")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def schema_objects(connection: sqlite3.Connection) -> list[tuple[str, str, str]]:
    return connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
          AND type IN ('table', 'index', 'view')
        ORDER BY type, name
        """
    ).fetchall()


def table_names(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
    ]


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    query = f"PRAGMA table_info({quote_identifier(table)})"
    return [row[1] for row in connection.execute(query)]


def normalize_execution_row(
    columns: list[str],
    row: tuple[object, ...],
) -> tuple[object, ...]:
    normalized: list[object] = []
    for column, value in zip(columns, row, strict=True):
        if column in IGNORED_EXECUTION_COLUMNS:
            continue
        if column == "parametros_json" and isinstance(value, str):
            parameters = json.loads(value)
            parameters.pop("workers", None)
            value = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
        normalized.append(value)
    return tuple(normalized)


def compare_execution_table(
    reference: sqlite3.Connection,
    candidate: sqlite3.Connection,
) -> None:
    table = "execucoes_preprocessamento"
    columns = table_columns(reference, table)
    selected = ", ".join(quote_identifier(column) for column in columns)
    query = f"SELECT {selected} FROM {quote_identifier(table)} ORDER BY rowid"
    reference_rows = [
        normalize_execution_row(columns, row) for row in reference.execute(query)
    ]
    candidate_rows = [
        normalize_execution_row(columns, row) for row in candidate.execute(query)
    ]
    if reference_rows != candidate_rows:
        raise RuntimeError("Metadados das execuções são diferentes.")


def compare_table(
    reference: sqlite3.Connection,
    candidate: sqlite3.Connection,
    table: str,
    batch_size: int,
) -> None:
    reference_count = reference.execute(
        f"SELECT COUNT(*) FROM {quote_identifier(table)}"
    ).fetchone()[0]
    candidate_count = candidate.execute(
        f"SELECT COUNT(*) FROM {quote_identifier(table)}"
    ).fetchone()[0]
    if reference_count != candidate_count:
        raise RuntimeError(
            f"Tabela {table}: {reference_count} linhas na referência e "
            f"{candidate_count} no candidato."
        )

    columns = table_columns(reference, table)
    selected = ", ".join(quote_identifier(column) for column in columns)
    query = f"SELECT {selected} FROM {quote_identifier(table)} ORDER BY rowid"
    reference_cursor = reference.execute(query)
    candidate_cursor = candidate.execute(query)
    offset = 0

    while True:
        reference_rows = reference_cursor.fetchmany(batch_size)
        candidate_rows = candidate_cursor.fetchmany(batch_size)
        if reference_rows != candidate_rows:
            common = min(len(reference_rows), len(candidate_rows))
            difference = next(
                (
                    index
                    for index in range(common)
                    if reference_rows[index] != candidate_rows[index]
                ),
                common,
            )
            raise RuntimeError(
                f"Tabela {table}: diferença encontrada na linha "
                f"{offset + difference + 1}."
            )
        if not reference_rows:
            break
        offset += len(reference_rows)
    print(f"Tabela {table}: {reference_count:,} linhas idênticas.", flush=True)


def compare_databases(reference_path: Path, candidate_path: Path) -> None:
    with open_read_only(reference_path) as reference, open_read_only(
        candidate_path
    ) as candidate:
        if schema_objects(reference) != schema_objects(candidate):
            raise RuntimeError("Schemas, índices ou views são diferentes.")

        tables = table_names(reference)
        if tables != table_names(candidate):
            raise RuntimeError("A relação de tabelas é diferente.")

        compare_execution_table(reference, candidate)
        print("Tabela execucoes_preprocessamento: metadados equivalentes.")
        for table in tables:
            if table == "execucoes_preprocessamento":
                continue
            compare_table(reference, candidate, table, batch_size=2_000)

    print("Comparação concluída: bancos logicamente idênticos.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("referencia", type=Path)
    parser.add_argument("candidato", type=Path)
    args = parser.parse_args()
    compare_databases(args.referencia, args.candidato)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
