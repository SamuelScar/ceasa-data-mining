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
    COBERTURAS_MINIMAS_NEGATIVO,
    COLUNA_COBERTURA,
    COLUNA_GRUPO,
    SCHEMA_VERSION,
    VISOES_VALIDACAO,
    abrir_base_somente_leitura,
    assinatura_atributos,
    calcular_metricas,
    calcular_sensibilidade_cobertura,
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
    parser.add_argument(
        "base_validacao_1",
        type=Path,
        help="SQLite contextual com treino até 2023 e validação em 2024",
    )
    parser.add_argument(
        "base_validacao_2",
        type=Path,
        help="SQLite contextual com treino até 2024 e validação em 2025",
    )
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
    entradas = [args.base_validacao_1, args.base_validacao_2]
    for entrada in entradas:
        if not entrada.is_file():
            raise FileNotFoundError(f"base de modelagem não encontrada: {entrada}")
    if args.base_validacao_1.resolve() == args.base_validacao_2.resolve():
        raise ValueError("as duas validações devem usar bases contextuais distintas")
    destinos = [args.output_db.resolve(), args.selection_json.resolve()]
    if len(set(destinos)) != len(destinos):
        raise ValueError("o SQLite e o JSON de seleção devem ser arquivos distintos")
    if any(entrada.resolve() in destinos for entrada in entradas):
        raise ValueError("as saídas devem ser diferentes das bases de modelagem")
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
            metodologia_sha256 TEXT NOT NULL,
            semente INTEGER NOT NULL,
            criterio_selecao TEXT NOT NULL,
            teste_final_consultado INTEGER NOT NULL CHECK (teste_final_consultado = 0),
            status TEXT NOT NULL
        );

        CREATE TABLE fontes_modelagem (
            contexto TEXT PRIMARY KEY
                CHECK (contexto IN ('validacao_1', 'validacao_2')),
            caminho TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            execucao_modelagem_id INTEGER NOT NULL,
            schema_version INTEGER NOT NULL
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

        CREATE TABLE cenarios_sensibilidade_cobertura (
            id INTEGER PRIMARY KEY,
            configuracao_id INTEGER NOT NULL,
            divisao TEXT NOT NULL,
            cobertura_minima_negativo INTEGER NOT NULL
                CHECK (cobertura_minima_negativo BETWEEN 1 AND 5),
            limiar REAL NOT NULL,
            linhas_avaliadas INTEGER NOT NULL,
            positivos INTEGER NOT NULL,
            negativos INTEGER NOT NULL,
            avaliavel INTEGER NOT NULL CHECK (avaliavel IN (0, 1)),
            motivo_nao_avaliavel TEXT,
            FOREIGN KEY (configuracao_id) REFERENCES configuracoes_modelos(id),
            UNIQUE (
                configuracao_id,
                divisao,
                cobertura_minima_negativo
            ),
            CHECK (linhas_avaliadas = positivos + negativos),
            CHECK (
                (avaliavel = 1 AND motivo_nao_avaliavel IS NULL)
                OR
                (avaliavel = 0 AND motivo_nao_avaliavel IS NOT NULL)
            )
        );

        CREATE TABLE metricas_sensibilidade_cobertura (
            id INTEGER PRIMARY KEY,
            cenario_id INTEGER NOT NULL,
            metrica TEXT NOT NULL,
            valor REAL NOT NULL,
            FOREIGN KEY (cenario_id)
                REFERENCES cenarios_sensibilidade_cobertura(id),
            UNIQUE (cenario_id, metrica)
        );

        CREATE TABLE previsoes_validacao (
            id INTEGER PRIMARY KEY,
            configuracao_id INTEGER NOT NULL,
            divisao TEXT NOT NULL,
            evento_id INTEGER NOT NULL,
            data_choque TEXT NOT NULL,
            grupo_choque TEXT NOT NULL,
            alvo INTEGER NOT NULL CHECK (alvo IN (0, 1)),
            dias_observados_destino INTEGER NOT NULL
                CHECK (dias_observados_destino BETWEEN 1 AND 5),
            probabilidade REAL NOT NULL CHECK (probabilidade BETWEEN 0 AND 1),
            previsao INTEGER NOT NULL CHECK (previsao IN (0, 1)),
            FOREIGN KEY (configuracao_id) REFERENCES configuracoes_modelos(id),
            UNIQUE (configuracao_id, divisao, evento_id)
        );

        CREATE INDEX idx_metricas_validacao_configuracao
            ON metricas_validacao(configuracao_id, divisao);
        CREATE INDEX idx_previsoes_validacao_evento
            ON previsoes_validacao(divisao, evento_id);
        CREATE INDEX idx_sensibilidade_cobertura_configuracao
            ON cenarios_sensibilidade_cobertura(
                configuracao_id, divisao, cobertura_minima_negativo
            );
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
                    COLUNA_COBERTURA: validacao[COLUNA_COBERTURA]
                    .astype(int)
                    .to_numpy(),
                    "probabilidade": probabilidades,
                }
            )

        limiar, _ = selecionar_limiar(previsoes_por_divisao)
        metricas_por_divisao: dict[str, dict[str, float]] = {}
        sensibilidade_por_divisao: dict[str, list[dict[str, Any]]] = {}
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
            sensibilidade_por_divisao[divisao] = (
                calcular_sensibilidade_cobertura(previsoes, limiar)
            )

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
                "sensibilidade_cobertura": sensibilidade_por_divisao,
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
    fontes: dict[str, dict[str, Any]],
    metodologia_sha256: str,
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
                    metodologia_sha256, semente, criterio_selecao,
                    teste_final_consultado, status
                ) VALUES (1, ?, ?, ?, ?, ?, ?, 0, 'concluida')
                """,
                (
                    SCHEMA_VERSION,
                    iniciada_em,
                    agora_iso(),
                    metodologia_sha256,
                    args.semente,
                    "maior F1 médio; desempate por PR-AUC, MCC e ordem fixa",
                ),
            )
            connection.executemany(
                """
                INSERT INTO fontes_modelagem (
                    contexto, caminho, sha256, execucao_modelagem_id,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        contexto,
                        str(fonte["caminho"]),
                        str(fonte["sha256"]),
                        int(fonte["execucao"]["id"]),
                        int(fonte["execucao"]["schema_version"]),
                    )
                    for contexto, fonte in sorted(fontes.items())
                ],
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
                for divisao, cenarios in resultado[
                    "sensibilidade_cobertura"
                ].items():
                    for cenario in cenarios:
                        cursor = connection.execute(
                            """
                            INSERT INTO cenarios_sensibilidade_cobertura (
                                configuracao_id, divisao,
                                cobertura_minima_negativo, limiar,
                                linhas_avaliadas, positivos, negativos,
                                avaliavel, motivo_nao_avaliavel
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                configuracao_id,
                                divisao,
                                cenario["cobertura_minima_negativo"],
                                resultado["limiar"],
                                cenario["linhas_avaliadas"],
                                cenario["positivos"],
                                cenario["negativos"],
                                int(cenario["avaliavel"]),
                                cenario["motivo_nao_avaliavel"],
                            ),
                        )
                        connection.executemany(
                            """
                            INSERT INTO metricas_sensibilidade_cobertura (
                                cenario_id, metrica, valor
                            ) VALUES (?, ?, ?)
                            """,
                            [
                                (int(cursor.lastrowid), nome, valor)
                                for nome, valor in cenario["metricas"].items()
                            ],
                        )
                for divisao, previsoes in resultado["previsoes"].items():
                    connection.executemany(
                        """
                        INSERT INTO previsoes_validacao (
                            configuracao_id, divisao, evento_id, data_choque,
                            grupo_choque, alvo, dias_observados_destino,
                            probabilidade, previsao
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                configuracao_id,
                                divisao,
                                int(linha.evento_id),
                                str(linha.data_choque),
                                str(linha.grupo_choque),
                                int(linha.efeito_domino),
                                int(linha.dias_observados_destino),
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
        cenarios_esperados = len(resultados) * 2 * len(
            COBERTURAS_MINIMAS_NEGATIVO
        )
        cenarios_gravados = connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT configuracao_id, divisao,
                    cobertura_minima_negativo
                FROM cenarios_sensibilidade_cobertura
            )
            """
        ).fetchone()[0]
        if int(cenarios_gravados) != cenarios_esperados:
            raise RuntimeError(
                "a saída não contém todos os cenários de sensibilidade da cobertura"
            )
    finally:
        connection.close()


def documento_selecao(
    args: argparse.Namespace,
    fontes: dict[str, dict[str, Any]],
    metodologia_sha256: str,
    atributos: Any,
    selecionado: dict[str, Any],
) -> dict[str, Any]:
    configuracao = selecionado["configuracao"]
    metricas_resumo = ("f1", "mcc", "precisao", "revocacao")
    resumo_sensibilidade = {
        divisao: [
            {
                "cobertura_minima_negativo": cenario[
                    "cobertura_minima_negativo"
                ],
                "linhas_avaliadas": cenario["linhas_avaliadas"],
                "positivos": cenario["positivos"],
                "negativos": cenario["negativos"],
                "avaliavel": cenario["avaliavel"],
                "motivo_nao_avaliavel": cenario[
                    "motivo_nao_avaliavel"
                ],
                **{
                    nome: cenario["metricas"].get(nome)
                    for nome in metricas_resumo
                },
            }
            for cenario in cenarios
        ]
        for divisao, cenarios in selecionado[
            "sensibilidade_cobertura"
        ].items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "gerado_em": agora_iso(),
        "bases_modelagem": {
            contexto: {
                "caminho": str(fonte["caminho"]),
                "sha256": str(fonte["sha256"]),
                "execucao_id": int(fonte["execucao"]["id"]),
                "schema_version": int(fonte["execucao"]["schema_version"]),
                "parametros_json": json.loads(
                    str(fonte["execucao"]["parametros_json"])
                ),
            }
            for contexto, fonte in sorted(fontes.items())
        },
        "metodologia_sha256": metodologia_sha256,
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
        "analise_sensibilidade_cobertura": {
            "regra_principal_cobertura_minima_negativo": 1,
            "coberturas_minimas_negativo": list(
                COBERTURAS_MINIMAS_NEGATIVO
            ),
            "positivos_confirmados_mantidos": True,
            "participa_selecao": False,
            "resumo_modelo_selecionado": resumo_sensibilidade,
        },
        "divisoes_consultadas": list(VISOES_VALIDACAO),
        "teste_final_consultado": False,
    }


def main() -> None:
    args = criar_parser().parse_args()
    validar_argumentos(args)
    iniciada_em = agora_iso()
    especificacoes = (
        (
            "validacao_1",
            args.base_validacao_1,
            ("treino_validacao_1", "validacao_1"),
        ),
        (
            "validacao_2",
            args.base_validacao_2,
            ("treino_validacao_2", "validacao_2"),
        ),
    )
    fontes: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    atributos = None
    metodologia_sha256 = None

    for contexto, path, visoes in especificacoes:
        hash_base = sha256_arquivo(path)
        base = abrir_base_somente_leitura(path)
        try:
            execucao = validar_base(base, visoes, contexto)
            atributos_contexto = carregar_atributos(base)
            if atributos is None:
                atributos = atributos_contexto
            elif atributos.como_dict() != atributos_contexto.como_dict():
                raise ValueError(
                    "os catálogos de atributos das duas validações divergem"
                )

            assinatura = str(execucao["metodologia_sha256"])
            if metodologia_sha256 is None:
                metodologia_sha256 = assinatura
            elif metodologia_sha256 != assinatura:
                raise ValueError(
                    "as duas validações foram produzidas por metodologias diferentes"
                )

            for nome in visoes:
                frames[nome] = carregar_visao(
                    base,
                    nome,
                    atributos_contexto,
                )
        finally:
            base.close()
        fontes[contexto] = {
            "caminho": path.resolve(),
            "sha256": hash_base,
            "execucao": execucao,
        }

    assert atributos is not None
    assert metodologia_sha256 is not None
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

    resultados = avaliar_configuracoes(frames, atributos, args.semente)
    selecionado = escolher_resultado(resultados)
    documento = documento_selecao(
        args,
        fontes,
        metodologia_sha256,
        atributos,
        selecionado,
    )

    temporario_db = caminho_temporario(args.output_db, ".sqlite")
    temporario_json = caminho_temporario(args.selection_json, ".json")
    try:
        escrever_resultados(
            temporario_db,
            args,
            fontes,
            metodologia_sha256,
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
