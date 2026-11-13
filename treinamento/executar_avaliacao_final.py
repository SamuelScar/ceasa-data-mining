"""Treina a configuração congelada e avalia uma única vez no teste oficial."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from treinamento.comum import (
    ALVO,
    COLUNA_GRUPO,
    SCHEMA_VERSION,
    VISOES_FINAIS,
    VISOES_VALIDACAO,
    AtributosModelo,
    ConfiguracaoModelo,
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
    sha256_arquivo,
    validar_base,
    validar_separacao_temporal,
)


def agora_iso() -> str:
    return datetime.now(UTC).isoformat()


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Treina a configuração escolhida em todo o desenvolvimento e a "
            "avalia no teste temporal oficial."
        )
    )
    parser.add_argument("base_db", type=Path, help="SQLite da base de modelagem")
    parser.add_argument(
        "selection_json",
        type=Path,
        help="Configuração produzida pela validação temporal",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("resultados/treinamento/avaliacao_final.sqlite"),
        help="SQLite com as métricas e previsões do teste oficial",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("resultados/treinamento/modelo_final.pkl"),
        help="Artefato serializado do modelo final",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir conscientemente uma avaliação final existente",
    )
    return parser


def validar_argumentos(args: argparse.Namespace) -> None:
    for entrada in (args.base_db, args.selection_json):
        if not entrada.is_file():
            raise FileNotFoundError(f"arquivo de entrada não encontrado: {entrada}")
    caminhos = {
        args.base_db.resolve(),
        args.selection_json.resolve(),
        args.output_db.resolve(),
        args.model_output.resolve(),
    }
    if len(caminhos) != 4:
        raise ValueError("entradas e saídas devem apontar para arquivos distintos")
    existentes = [path for path in (args.output_db, args.model_output) if path.exists()]
    if existentes and not args.overwrite:
        raise FileExistsError(
            "a avaliação final já possui saída; não será sobrescrita: "
            + ", ".join(str(path) for path in existentes)
        )


def carregar_selecao(path: Path) -> dict[str, Any]:
    try:
        documento = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ValueError(f"não foi possível ler a configuração: {path}") from erro
    campos = {
        "schema_version",
        "base_modelagem",
        "atributos",
        "atributos_sha256",
        "semente",
        "modelo",
        "teste_final_consultado",
    }
    ausentes = campos - set(documento)
    if ausentes:
        raise ValueError(
            "configuração incompleta; campos ausentes: " + ", ".join(sorted(ausentes))
        )
    if int(documento["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("versão incompatível do documento de seleção")
    if documento["teste_final_consultado"] is not False:
        raise ValueError("a configuração não comprova o isolamento do teste final")
    if tuple(documento.get("divisoes_consultadas", ())) != VISOES_VALIDACAO:
        raise ValueError("a configuração não corresponde às validações previstas")
    return documento


def carregar_configuracao_modelo(documento: dict[str, Any]) -> ConfiguracaoModelo:
    modelo = documento["modelo"]
    codigo = str(modelo.get("codigo"))
    candidatas = {
        configuracao.codigo: configuracao
        for configuracao in configuracoes_candidatas()
    }
    if codigo not in candidatas:
        raise ValueError(f"modelo selecionado não reconhecido: {codigo}")
    configuracao = candidatas[codigo]
    if (
        str(modelo.get("familia")) != configuracao.familia
        or modelo.get("parametros") != configuracao.parametros
    ):
        raise ValueError("a configuração selecionada não coincide com o catálogo")
    return configuracao


def validar_selecao_com_base(
    documento: dict[str, Any],
    hash_base: str,
    execucao_base: dict[str, Any],
    atributos: AtributosModelo,
) -> None:
    origem = documento["base_modelagem"]
    if str(origem.get("sha256")) != hash_base:
        raise ValueError("a base atual difere da base usada na seleção do modelo")
    if int(origem.get("execucao_id")) != int(execucao_base["id"]):
        raise ValueError("a execução de modelagem difere da usada na seleção")
    if documento["atributos"] != atributos.como_dict():
        raise ValueError("o catálogo de atributos mudou após a seleção")
    if str(documento["atributos_sha256"]) != assinatura_atributos(atributos):
        raise ValueError("a assinatura dos atributos não corresponde ao catálogo")


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


def extrair_importancias(modelo: Any) -> list[tuple[str, float]]:
    if not hasattr(modelo, "named_steps"):
        return []
    preprocessador = modelo.named_steps.get("preprocessador")
    estimador = modelo.named_steps.get("modelo")
    if preprocessador is None or estimador is None:
        return []
    nomes = [str(nome) for nome in preprocessador.get_feature_names_out()]
    if hasattr(estimador, "feature_importances_"):
        valores = np.asarray(estimador.feature_importances_, dtype=float)
    elif hasattr(estimador, "coef_"):
        coeficientes = np.asarray(estimador.coef_, dtype=float)
        valores = np.abs(coeficientes[0] if coeficientes.ndim == 2 else coeficientes)
    else:
        return []
    if len(nomes) != len(valores):
        raise RuntimeError("os nomes e as importâncias dos atributos não coincidem")
    pares = list(zip(nomes, valores, strict=True))
    return sorted(pares, key=lambda item: (-item[1], item[0]))


def criar_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE execucoes_avaliacao_final (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT NOT NULL,
            base_modelagem_caminho TEXT NOT NULL,
            base_modelagem_sha256 TEXT NOT NULL,
            execucao_modelagem_id INTEGER NOT NULL,
            selecao_caminho TEXT NOT NULL,
            selecao_sha256 TEXT NOT NULL,
            codigo_modelo TEXT NOT NULL,
            familia_modelo TEXT NOT NULL,
            parametros_json TEXT NOT NULL,
            semente INTEGER NOT NULL,
            limiar REAL NOT NULL,
            linhas_treino INTEGER NOT NULL,
            linhas_teste INTEGER NOT NULL,
            teste_final_consultado INTEGER NOT NULL CHECK (teste_final_consultado = 1),
            status TEXT NOT NULL
        );

        CREATE TABLE metricas_teste_final (
            metrica TEXT PRIMARY KEY,
            valor REAL NOT NULL
        );

        CREATE TABLE previsoes_teste_final (
            id INTEGER PRIMARY KEY,
            evento_id INTEGER NOT NULL UNIQUE,
            data_choque TEXT NOT NULL,
            grupo_choque TEXT NOT NULL,
            alvo INTEGER NOT NULL CHECK (alvo IN (0, 1)),
            probabilidade REAL NOT NULL CHECK (probabilidade BETWEEN 0 AND 1),
            previsao INTEGER NOT NULL CHECK (previsao IN (0, 1))
        );

        CREATE TABLE importancias_atributos (
            posicao INTEGER PRIMARY KEY,
            atributo_transformado TEXT NOT NULL UNIQUE,
            importancia_absoluta REAL NOT NULL
        );
        """
    )


def escrever_resultados(
    path: Path,
    args: argparse.Namespace,
    execucao_base: dict[str, Any],
    hash_base: str,
    hash_selecao: str,
    configuracao: ConfiguracaoModelo,
    semente: int,
    limiar: float,
    treino: pd.DataFrame,
    teste: pd.DataFrame,
    probabilidades: np.ndarray,
    metricas: dict[str, float],
    importancias: list[tuple[str, float]],
    iniciada_em: str,
) -> None:
    previsoes = (probabilidades >= limiar).astype(int)
    connection = sqlite3.connect(path)
    try:
        criar_schema(connection)
        with connection:
            connection.execute(
                """
                INSERT INTO execucoes_avaliacao_final (
                    id, schema_version, iniciada_em, concluida_em,
                    base_modelagem_caminho, base_modelagem_sha256,
                    execucao_modelagem_id, selecao_caminho, selecao_sha256,
                    codigo_modelo, familia_modelo, parametros_json, semente,
                    limiar, linhas_treino, linhas_teste,
                    teste_final_consultado, status
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'concluida')
                """,
                (
                    SCHEMA_VERSION,
                    iniciada_em,
                    agora_iso(),
                    str(args.base_db.resolve()),
                    hash_base,
                    int(execucao_base["id"]),
                    str(args.selection_json.resolve()),
                    hash_selecao,
                    configuracao.codigo,
                    configuracao.familia,
                    parametros_json(configuracao.parametros),
                    semente,
                    limiar,
                    len(treino),
                    len(teste),
                ),
            )
            connection.executemany(
                "INSERT INTO metricas_teste_final (metrica, valor) VALUES (?, ?)",
                sorted(metricas.items()),
            )
            connection.executemany(
                """
                INSERT INTO previsoes_teste_final (
                    evento_id, data_choque, grupo_choque, alvo,
                    probabilidade, previsao
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(evento_id),
                        str(data_choque),
                        str(grupo),
                        int(alvo),
                        float(probabilidade),
                        int(previsao),
                    )
                    for evento_id, data_choque, grupo, alvo, probabilidade, previsao in zip(
                        teste["id"],
                        teste["data_choque"],
                        teste[COLUNA_GRUPO],
                        teste[ALVO],
                        probabilidades,
                        previsoes,
                        strict=True,
                    )
                ],
            )
            connection.executemany(
                """
                INSERT INTO importancias_atributos (
                    posicao, atributo_transformado, importancia_absoluta
                ) VALUES (?, ?, ?)
                """,
                [
                    (posicao, nome, float(valor))
                    for posicao, (nome, valor) in enumerate(importancias, start=1)
                ],
            )

        integridade = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integridade != "ok":
            raise RuntimeError(f"falha de integridade no SQLite: {integridade}")
        quantidade = connection.execute(
            "SELECT COUNT(*) FROM previsoes_teste_final"
        ).fetchone()[0]
        if int(quantidade) != len(teste):
            raise RuntimeError("a saída não contém todas as previsões do teste")
    finally:
        connection.close()


def main() -> None:
    args = criar_parser().parse_args()
    validar_argumentos(args)
    iniciada_em = agora_iso()
    documento = carregar_selecao(args.selection_json)
    hash_base = sha256_arquivo(args.base_db)
    hash_selecao = sha256_arquivo(args.selection_json)

    base = abrir_base_somente_leitura(args.base_db)
    try:
        execucao_base = validar_base(base, VISOES_FINAIS)
        atributos = carregar_atributos(base)
        validar_selecao_com_base(documento, hash_base, execucao_base, atributos)
        treino = carregar_visao(base, "treino_final", atributos)
        X_treino, y_treino = preparar_matriz(treino, atributos)

        modelo_documento = documento["modelo"]
        configuracao = carregar_configuracao_modelo(documento)
        semente = int(documento["semente"])
        limiar = float(modelo_documento["limiar"])
        if not 0 < limiar < 1:
            raise ValueError("o limiar selecionado deve estar entre zero e um")
        modelo = construir_estimador(configuracao, atributos, semente)
        modelo.fit(X_treino, y_treino)

        # O teste só é lido depois que configuração e ajuste estão congelados.
        teste = carregar_visao(base, "teste_final", atributos)
        validar_separacao_temporal(
            treino,
            teste,
            "treino_final",
            "teste_final",
        )
        X_teste, _ = preparar_matriz(teste, atributos)
        probabilidades = probabilidades_positivas(modelo, X_teste)
    finally:
        base.close()

    metricas = calcular_metricas(
        teste[ALVO],
        probabilidades,
        limiar,
        teste[COLUNA_GRUPO],
    )
    importancias = extrair_importancias(modelo)
    artefato = {
        "schema_version": SCHEMA_VERSION,
        "gerado_em": agora_iso(),
        "base_modelagem_sha256": hash_base,
        "selecao_sha256": hash_selecao,
        "sklearn_version": sklearn.__version__,
        "atributos": atributos.como_dict(),
        "configuracao": {
            "codigo": configuracao.codigo,
            "familia": configuracao.familia,
            "parametros": configuracao.parametros,
            "semente": semente,
            "limiar": limiar,
        },
        "modelo": modelo,
    }

    temporario_db = caminho_temporario(args.output_db, ".sqlite")
    temporario_modelo = caminho_temporario(args.model_output, ".pkl")
    try:
        escrever_resultados(
            temporario_db,
            args,
            execucao_base,
            hash_base,
            hash_selecao,
            configuracao,
            semente,
            limiar,
            treino,
            teste,
            probabilidades,
            metricas,
            importancias,
            iniciada_em,
        )
        with temporario_modelo.open("wb") as arquivo:
            pickle.dump(artefato, arquivo, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporario_db, args.output_db)
        os.replace(temporario_modelo, args.model_output)
    finally:
        temporario_db.unlink(missing_ok=True)
        temporario_modelo.unlink(missing_ok=True)

    print(
        "Avaliação final concluída: "
        f"modelo={configuracao.codigo}, F1={metricas['f1']:.4f}, "
        f"PR-AUC={metricas['pr_auc']:.4f}."
    )


if __name__ == "__main__":
    main()
