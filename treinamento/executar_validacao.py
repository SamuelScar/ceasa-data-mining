"""Compara modelos apenas nas duas validações temporais de desenvolvimento."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from treinamento.comum import (
    ALVO,
    COLUNA_GRUPO,
    SCHEMA_VERSION,
    VISOES_VALIDACAO,
    abrir_base_somente_leitura,
    assinatura_atributos,
    calcular_metricas,
    carregar_atributos,
    carregar_visao,
    configuracoes_candidatas,
    construir_estimador,
    parametros_json,
    preparar_matriz,
    probabilidades_positivas,
    selecionar_limiar,
    sha256_arquivo,
    validar_base,
    validar_separacao_temporal,
)


DEFAULT_SEMENTE = 42


def agora_iso() -> str:
    return datetime.now(UTC).isoformat()


def inteiro_ambiente(nome: str, padrao: int) -> int:
    valor = os.getenv(nome)
    if valor in (None, ""):
        return padrao
    try:
        return int(valor)
    except ValueError as erro:
        raise ValueError(f"{nome} deve ser um número inteiro") from erro


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seleciona modelo, hiperparâmetros e limiar usando exclusivamente "
            "as validações temporais de desenvolvimento."
        )
    )
    parser.add_argument("base_db", type=Path, help="SQLite da base de modelagem")
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("resultados/treinamento/validacao.sqlite"),
        help="SQLite com métricas e previsões das validações",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=Path("resultados/treinamento/configuracao_selecionada.json"),
        help="Configuração congelada para a avaliação final",
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=inteiro_ambiente("TREINAMENTO_SEMENTE", DEFAULT_SEMENTE),
        help="Semente usada pelos modelos estocásticos",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Substitui as saídas somente depois de uma execução válida",
    )
    return parser


def validar_argumentos(args: argparse.Namespace) -> None:
    if not args.base_db.is_file():
        raise FileNotFoundError(f"base de modelagem não encontrada: {args.base_db}")
    destinos = [args.output_db.resolve(), args.selection_json.resolve()]
    if len(set(destinos)) != len(destinos):
        raise ValueError("o SQLite e o JSON de seleção devem ser arquivos distintos")
    if args.base_db.resolve() in destinos:
        raise ValueError("as saídas devem ser diferentes da base de modelagem")
    existentes = [path for path in (args.output_db, args.selection_json) if path.exists()]
    if existentes and not args.overwrite:
        raise FileExistsError(
            "saída já existente; use --overwrite para substituí-la: "
            + ", ".join(str(path) for path in existentes)
        )


def caminho_temporario(destino: Path, sufixo: str) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    descritor, nome = tempfile.mkstemp(
        prefix=f".{destino.name}.",
        suffix=sufixo,
        dir=destino.parent,
    )
    os.close(descritor)
    path = Path(nome)
    path.unlink()
    return path


def criar_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_validacao (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT NOT NULL,
            base_modelagem_caminho TEXT NOT NULL,
            base_modelagem_sha256 TEXT NOT NULL,
            execucao_modelagem_id INTEGER NOT NULL,
            semente INTEGER NOT NULL,
            criterio_selecao TEXT NOT NULL,
            teste_final_consultado INTEGER NOT NULL CHECK (teste_final_consultado = 0),
            status TEXT NOT NULL
        );

        CREATE TABLE configuracoes_modelos (
            id INTEGER PRIMARY KEY,
            codigo TEXT NOT NULL UNIQUE,
            familia TEXT NOT NULL,
            parametros_json TEXT NOT NULL,
            selecionada INTEGER NOT NULL CHECK (selecionada IN (0, 1))
        );

        CREATE TABLE resumo_configuracoes (
            configuracao_id INTEGER PRIMARY KEY,
            limiar REAL NOT NULL,
            f1_medio REAL NOT NULL,
            pr_auc_media REAL NOT NULL,
            mcc_medio REAL NOT NULL,
            precisao_media REAL NOT NULL,
            revocacao_media REAL NOT NULL,
            FOREIGN KEY (configuracao_id) REFERENCES configuracoes_modelos(id)
        );

        CREATE TABLE metricas_validacao (
            id INTEGER PRIMARY KEY,
            configuracao_id INTEGER NOT NULL,
            divisao TEXT NOT NULL,
            limiar REAL NOT NULL,
            metrica TEXT NOT NULL,
            valor REAL NOT NULL,
            FOREIGN KEY (configuracao_id) REFERENCES configuracoes_modelos(id),
            UNIQUE (configuracao_id, divisao, metrica)
        );

        CREATE TABLE previsoes_validacao (
            id INTEGER PRIMARY KEY,
            configuracao_id INTEGER NOT NULL,
            divisao TEXT NOT NULL,
            evento_id INTEGER NOT NULL,
            data_choque TEXT NOT NULL,
            grupo_choque TEXT NOT NULL,
            alvo INTEGER NOT NULL CHECK (alvo IN (0, 1)),
            probabilidade REAL NOT NULL CHECK (probabilidade BETWEEN 0 AND 1),
            previsao INTEGER NOT NULL CHECK (previsao IN (0, 1)),
            FOREIGN KEY (configuracao_id) REFERENCES configuracoes_modelos(id),
            UNIQUE (configuracao_id, divisao, evento_id)
        );

        CREATE INDEX idx_metricas_validacao_configuracao
            ON metricas_validacao(configuracao_id, divisao);
        CREATE INDEX idx_previsoes_validacao_evento
            ON previsoes_validacao(divisao, evento_id);
        """
    )


def avaliar_configuracoes(
    frames: dict[str, pd.DataFrame],
    atributos: Any,
    semente: int,
) -> list[dict[str, Any]]:
    divisoes = (
        ("validacao_1", "treino_validacao_1", "validacao_1"),
        ("validacao_2", "treino_validacao_2", "validacao_2"),
    )
    resultados: list[dict[str, Any]] = []
    for ordem, configuracao in enumerate(configuracoes_candidatas(), start=1):
        previsoes_por_divisao: dict[str, pd.DataFrame] = {}
        for divisao, nome_treino, nome_validacao in divisoes:
            treino = frames[nome_treino]
            validacao = frames[nome_validacao]
            X_treino, y_treino = preparar_matriz(treino, atributos)
            X_validacao, _ = preparar_matriz(validacao, atributos)
            modelo = construir_estimador(configuracao, atributos, semente)
            modelo.fit(X_treino, y_treino)
            probabilidades = probabilidades_positivas(modelo, X_validacao)
            previsoes_por_divisao[divisao] = pd.DataFrame(
                {
                    "evento_id": validacao["id"].astype(int).to_numpy(),
                    "data_choque": validacao["data_choque"].astype(str).to_numpy(),
                    COLUNA_GRUPO: validacao[COLUNA_GRUPO].astype(str).to_numpy(),
                    ALVO: validacao[ALVO].astype(int).to_numpy(),
                    "probabilidade": probabilidades,
                }
            )

        limiar, _ = selecionar_limiar(previsoes_por_divisao)
        metricas_por_divisao: dict[str, dict[str, float]] = {}
        for divisao, previsoes in previsoes_por_divisao.items():
            metricas_por_divisao[divisao] = calcular_metricas(
                previsoes[ALVO],
                previsoes["probabilidade"].to_numpy(),
                limiar,
                previsoes[COLUNA_GRUPO],
            )
            previsoes["previsao"] = (
                previsoes["probabilidade"] >= limiar
            ).astype(int)

        def media(nome: str) -> float:
            return float(
                np.mean(
                    [metricas[nome] for metricas in metricas_por_divisao.values()]
                )
            )

        resultados.append(
            {
                "ordem": ordem,
                "configuracao": configuracao,
                "limiar": limiar,
                "f1_medio": media("f1"),
                "pr_auc_media": media("pr_auc"),
                "mcc_medio": media("mcc"),
                "precisao_media": media("precisao"),
                "revocacao_media": media("revocacao"),
                "metricas": metricas_por_divisao,
                "previsoes": previsoes_por_divisao,
            }
        )
    return resultados


def escolher_resultado(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        resultados,
        key=lambda resultado: (
            resultado["f1_medio"],
            resultado["pr_auc_media"],
            resultado["mcc_medio"],
            -resultado["ordem"],
        ),
    )


def escrever_resultados(
    path: Path,
    args: argparse.Namespace,
    execucao_base: dict[str, Any],
    hash_base: str,
    resultados: list[dict[str, Any]],
    selecionado: dict[str, Any],
    iniciada_em: str,
) -> None:
    connection = sqlite3.connect(path)
    try:
        criar_schema(connection)
        with connection:
            connection.execute(
                """
                INSERT INTO execucoes_validacao (
                    id, schema_version, iniciada_em, concluida_em,
                    base_modelagem_caminho, base_modelagem_sha256,
                    execucao_modelagem_id, semente, criterio_selecao,
                    teste_final_consultado, status
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'concluida')
                """,
                (
                    SCHEMA_VERSION,
                    iniciada_em,
                    agora_iso(),
                    str(args.base_db.resolve()),
                    hash_base,
                    int(execucao_base["id"]),
                    args.semente,
                    "maior F1 médio; desempate por PR-AUC, MCC e ordem fixa",
                ),
            )
            for resultado in resultados:
                configuracao = resultado["configuracao"]
                configuracao_id = int(resultado["ordem"])
                connection.execute(
                    """
                    INSERT INTO configuracoes_modelos (
                        id, codigo, familia, parametros_json, selecionada
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        configuracao_id,
                        configuracao.codigo,
                        configuracao.familia,
                        parametros_json(configuracao.parametros),
                        int(resultado is selecionado),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO resumo_configuracoes (
                        configuracao_id, limiar, f1_medio, pr_auc_media,
                        mcc_medio, precisao_media, revocacao_media
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        configuracao_id,
                        resultado["limiar"],
                        resultado["f1_medio"],
                        resultado["pr_auc_media"],
                        resultado["mcc_medio"],
                        resultado["precisao_media"],
                        resultado["revocacao_media"],
                    ),
                )
                for divisao, metricas in resultado["metricas"].items():
                    connection.executemany(
                        """
                        INSERT INTO metricas_validacao (
                            configuracao_id, divisao, limiar, metrica, valor
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                configuracao_id,
                                divisao,
                                resultado["limiar"],
                                nome,
                                valor,
                            )
                            for nome, valor in metricas.items()
                        ],
                    )
                for divisao, previsoes in resultado["previsoes"].items():
                    connection.executemany(
                        """
                        INSERT INTO previsoes_validacao (
                            configuracao_id, divisao, evento_id, data_choque,
                            grupo_choque, alvo, probabilidade, previsao
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                configuracao_id,
                                divisao,
                                int(linha.evento_id),
                                str(linha.data_choque),
                                str(linha.grupo_choque),
                                int(linha.efeito_domino),
                                float(linha.probabilidade),
                                int(linha.previsao),
                            )
                            for linha in previsoes.itertuples(index=False)
                        ],
                    )

        integridade = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integridade != "ok":
            raise RuntimeError(f"falha de integridade no SQLite: {integridade}")
        quantidade = connection.execute(
            "SELECT COUNT(*) FROM configuracoes_modelos WHERE selecionada = 1"
        ).fetchone()[0]
        if int(quantidade) != 1:
            raise RuntimeError("a saída não contém exatamente uma configuração selecionada")
    finally:
        connection.close()


def documento_selecao(
    args: argparse.Namespace,
    execucao_base: dict[str, Any],
    hash_base: str,
    atributos: Any,
    selecionado: dict[str, Any],
) -> dict[str, Any]:
    configuracao = selecionado["configuracao"]
    return {
        "schema_version": SCHEMA_VERSION,
        "gerado_em": agora_iso(),
        "base_modelagem": {
            "caminho": str(args.base_db.resolve()),
            "sha256": hash_base,
            "execucao_id": int(execucao_base["id"]),
            "parametros_json": json.loads(str(execucao_base["parametros_json"])),
        },
        "atributos": atributos.como_dict(),
        "atributos_sha256": assinatura_atributos(atributos),
        "semente": args.semente,
        "criterio_selecao": "maior F1 médio; desempate por PR-AUC, MCC e ordem fixa",
        "modelo": {
            "codigo": configuracao.codigo,
            "familia": configuracao.familia,
            "parametros": configuracao.parametros,
            "limiar": selecionado["limiar"],
        },
        "resumo_validacao": {
            "f1_medio": selecionado["f1_medio"],
            "pr_auc_media": selecionado["pr_auc_media"],
            "mcc_medio": selecionado["mcc_medio"],
            "precisao_media": selecionado["precisao_media"],
            "revocacao_media": selecionado["revocacao_media"],
        },
        "divisoes_consultadas": list(VISOES_VALIDACAO),
        "teste_final_consultado": False,
    }


def main() -> None:
    args = criar_parser().parse_args()
    validar_argumentos(args)
    iniciada_em = agora_iso()
    hash_base = sha256_arquivo(args.base_db)

    base = abrir_base_somente_leitura(args.base_db)
    try:
        execucao_base = validar_base(base, VISOES_VALIDACAO)
        atributos = carregar_atributos(base)
        frames = {
            nome: carregar_visao(base, nome, atributos) for nome in VISOES_VALIDACAO
        }
        validar_separacao_temporal(
            frames["treino_validacao_1"],
            frames["validacao_1"],
            "treino_validacao_1",
            "validacao_1",
        )
        validar_separacao_temporal(
            frames["treino_validacao_2"],
            frames["validacao_2"],
            "treino_validacao_2",
            "validacao_2",
        )
    finally:
        base.close()

    resultados = avaliar_configuracoes(frames, atributos, args.semente)
    selecionado = escolher_resultado(resultados)
    documento = documento_selecao(
        args,
        execucao_base,
        hash_base,
        atributos,
        selecionado,
    )

    temporario_db = caminho_temporario(args.output_db, ".sqlite")
    temporario_json = caminho_temporario(args.selection_json, ".json")
    try:
        escrever_resultados(
            temporario_db,
            args,
            execucao_base,
            hash_base,
            resultados,
            selecionado,
            iniciada_em,
        )
        temporario_json.write_text(
            json.dumps(documento, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporario_db, args.output_db)
        os.replace(temporario_json, args.selection_json)
    finally:
        temporario_db.unlink(missing_ok=True)
        temporario_json.unlink(missing_ok=True)

    configuracao = selecionado["configuracao"]
    print(
        "Validação concluída: "
        f"modelo={configuracao.codigo}, limiar={selecionado['limiar']:.2f}, "
        f"F1 médio={selecionado['f1_medio']:.4f}."
    )


if __name__ == "__main__":
    main()
