from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from preprocessamento.comparar_bancos import compare_databases


def criar_banco(path: Path, workers: int | None, valor: str = "igual") -> None:
    parameters = {"chunk_size": 100_000}
    if workers is not None:
        parameters["workers"] = workers

    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE execucoes_preprocessamento (
                id INTEGER PRIMARY KEY,
                iniciada_em TEXT NOT NULL,
                concluida_em TEXT,
                parametros_json TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE dados (
                id INTEGER PRIMARY KEY,
                valor TEXT NOT NULL
            );
            CREATE INDEX idx_dados_valor ON dados(valor);
            CREATE VIEW dados_ativos AS SELECT * FROM dados;
            """
        )
        connection.execute(
            "INSERT INTO execucoes_preprocessamento VALUES (?, ?, ?, ?, ?)",
            (
                1,
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:01:00Z",
                json.dumps(parameters, sort_keys=True),
                "concluida",
            ),
        )
        connection.execute("INSERT INTO dados VALUES (1, ?)", (valor,))


class CompararBancosTest(unittest.TestCase):
    def test_ignora_tempo_e_numero_de_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.sqlite"
            candidate = Path(directory) / "candidate.sqlite"
            criar_banco(reference, workers=None)
            criar_banco(candidate, workers=2)

            compare_databases(reference, candidate)

    def test_detecta_diferenca_nos_dados(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.sqlite"
            candidate = Path(directory) / "candidate.sqlite"
            criar_banco(reference, workers=1)
            criar_banco(candidate, workers=2, valor="diferente")

            with self.assertRaisesRegex(RuntimeError, "diferença encontrada"):
                compare_databases(reference, candidate)


if __name__ == "__main__":
    unittest.main()
