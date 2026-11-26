#!/usr/bin/env python3
"""Agrega janelas do experimento encerrado de estabilidade temporal."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
ChaveRelacao = tuple[str, str, str]


def abrir_somente_leitura(caminho: Path) -> sqlite3.Connection:
    if not caminho.is_file():
        raise FileNotFoundError(f"banco não encontrado: {caminho}")
    conexao = sqlite3.connect(
        f"{caminho.resolve().as_uri()}?mode=ro",
        uri=True,
    )
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA query_only = ON")
    return conexao


def resposta_dinamica(
    matrizes_lags: list[np.ndarray],
    indice_origem: int,
    indice_destino: int,
    horizonte: int,
) -> float:
    choque = np.zeros(matrizes_lags[0].shape[0])
    choque[indice_origem] = 1.0
    respostas = [choque]
    for passo in range(1, horizonte + 1):
        resposta = np.zeros_like(choque)
        for defasagem, matriz in enumerate(matrizes_lags, start=1):
            if defasagem <= passo:
                resposta += matriz @ respostas[passo - defasagem]
        respostas.append(resposta)
    return float(
        sum(resposta[indice_destino] for resposta in respostas[1:])
    )


def carregar_respostas_dinamicas(
    conexao: sqlite3.Connection,
    horizonte: int,
) -> dict[ChaveRelacao, float]:
    respostas: dict[ChaveRelacao, float] = {}
    redes = conexao.execute(
        """
        SELECT id, produto, mercados_json, lag_selecionado
        FROM redes_produtos
        WHERE status = 'testada'
        ORDER BY id
        """
    ).fetchall()
    for rede in redes:
        mercados = tuple(json.loads(rede["mercados_json"]))
        lag = int(rede["lag_selecionado"])
        matrizes = [
            np.zeros((len(mercados), len(mercados))) for _ in range(lag)
        ]
        for indice_destino, destino in enumerate(mercados):
            resultado = conexao.execute(
                """
                SELECT MIN(id) AS id
                FROM resultados_condicionais
                WHERE rede_id = ? AND destino = ?
                """,
                (rede["id"], destino),
            ).fetchone()
            if resultado is None or resultado["id"] is None:
                raise RuntimeError(
                    f"equação ausente para {rede['produto']}/{destino}"
                )
            coeficientes = conexao.execute(
                """
                SELECT mercado_preditor, defasagem, coeficiente
                FROM coeficientes_condicionais
                WHERE resultado_id = ? AND defasagem > 0
                """,
                (resultado["id"],),
            ).fetchall()
            for coeficiente in coeficientes:
                indice_preditor = mercados.index(
                    coeficiente["mercado_preditor"]
                )
                matrizes[int(coeficiente["defasagem"]) - 1][
                    indice_destino,
                    indice_preditor,
                ] = float(coeficiente["coeficiente"])

        for indice_origem, origem in enumerate(mercados):
            for indice_destino, destino in enumerate(mercados):
                if origem == destino:
                    continue
                respostas[(rede["produto"], origem, destino)] = (
                    resposta_dinamica(
                        matrizes,
                        indice_origem,
                        indice_destino,
                        horizonte,
                    )
                )
    return respostas


def carregar_janela(
    janela: dict[str, Any],
    horizonte: int,
    diretorio_janelas: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    caminho = diretorio_janelas / f"{janela['id']}.sqlite"
    with abrir_somente_leitura(caminho) as conexao:
        if conexao.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"falha de integridade em {caminho}")
        execucao = conexao.execute(
            """
            SELECT configuracao_redes_sha256, status
            FROM execucoes_condicionais
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
        if execucao is None or execucao["status"] != "concluida":
            raise RuntimeError(f"execução não concluída em {caminho}")
        respostas = carregar_respostas_dinamicas(conexao, horizonte)
        linhas: list[dict[str, Any]] = []
        for linha in conexao.execute(
            """
            SELECT resultado.produto, resultado.origem, resultado.destino,
                   rede.lag_selecionado, resultado.n_observacoes,
                   resultado.p_valor, resultado.q_valor_fdr,
                   resultado.significativo_fdr,
                   diagnostico.sem_autocorrelacao_irrestrito,
                   avaliacao.status AS status_avaliacao,
                   avaliacao.melhora_rmse, avaliacao.melhora_mae,
                   avaliacao.altas_avaliadas,
                   avaliacao.melhora_rmse_altas,
                   avaliacao.melhora_mae_altas
            FROM resultados_condicionais AS resultado
            JOIN redes_produtos AS rede ON rede.id = resultado.rede_id
            JOIN diagnosticos_residuos AS diagnostico
              ON diagnostico.resultado_id = resultado.id
            JOIN avaliacoes_preditivas AS avaliacao
              ON avaliacao.resultado_id = resultado.id
            ORDER BY resultado.produto, resultado.origem, resultado.destino
            """
        ):
            registro = dict(linha)
            chave = (
                registro["produto"],
                registro["origem"],
                registro["destino"],
            )
            registro["resposta_acumulada"] = respostas[chave]
            registro["resposta_positiva"] = int(respostas[chave] > 0)
            linhas.append(registro)
        resumo = {
            linha["chave"]: int(linha["valor"])
            for linha in conexao.execute(
                "SELECT chave, valor FROM resumo_condicional"
            )
        }
    metadados = {
        **janela,
        "configuracao_redes_sha256": execucao["configuracao_redes_sha256"],
        "direcoes_testadas": len(linhas),
        "produtos_testados": resumo["produtos_testados"],
    }
    return metadados, linhas


def carregar_resultados_finais(
    caminho: Path,
) -> dict[ChaveRelacao, dict[str, int]]:
    with abrir_somente_leitura(caminho) as conexao:
        return {
            (linha["produto"], linha["origem"], linha["destino"]): {
                "significativo_fdr": int(linha["significativo_fdr"]),
                "residuos_adequados": int(
                    linha["sem_autocorrelacao_irrestrito"]
                ),
                "melhora_rmse": int(linha["melhora_rmse"]),
                "melhora_mae": int(linha["melhora_mae"]),
                "melhora_rmse_altas": int(linha["melhora_rmse_altas"]),
                "melhora_mae_altas": int(linha["melhora_mae_altas"]),
            }
            for linha in conexao.execute(
                """
                SELECT resultado.produto, resultado.origem,
                       resultado.destino, resultado.significativo_fdr,
                       diagnostico.sem_autocorrelacao_irrestrito,
                       avaliacao.melhora_rmse, avaliacao.melhora_mae,
                       avaliacao.melhora_rmse_altas,
                       avaliacao.melhora_mae_altas
                FROM resultados_condicionais AS resultado
                JOIN diagnosticos_residuos AS diagnostico
                  ON diagnostico.resultado_id = resultado.id
                JOIN avaliacoes_preditivas AS avaliacao
                  ON avaliacao.resultado_id = resultado.id
                """
            )
        }


def proporcao(quantidade: int, total: int) -> float:
    return quantidade / total if total else 0.0


def agregar_relacoes(
    resultados: list[dict[str, Any]],
    limites: dict[str, Any],
    finais: dict[str, dict[ChaveRelacao, dict[str, int]]],
) -> list[dict[str, Any]]:
    grupos: dict[tuple[str, ChaveRelacao], list[dict[str, Any]]] = defaultdict(
        list
    )
    for resultado in resultados:
        chave = (
            resultado["produto"],
            resultado["origem"],
            resultado["destino"],
        )
        grupos[(resultado["contexto"], chave)].append(resultado)

    agregados: list[dict[str, Any]] = []
    for (contexto, chave), linhas in sorted(grupos.items()):
        total = len(linhas)
        significativas = sum(linha["significativo_fdr"] for linha in linhas)
        residuos = sum(
            linha["sem_autocorrelacao_irrestrito"] for linha in linhas
        )
        positivas = sum(linha["resposta_positiva"] for linha in linhas)
        avaliadas = [
            linha for linha in linhas if linha["status_avaliacao"] == "avaliado"
        ]
        altas_avaliadas = [linha for linha in linhas if linha["altas_avaliadas"]]
        lags = Counter(int(linha["lag_selecionado"]) for linha in linhas)
        lag_modal, ocorrencias_lag = min(
            lags.items(),
            key=lambda item: (-item[1], item[0]),
        )
        frequencia_significancia = proporcao(significativas, total)
        frequencia_residuos = proporcao(residuos, total)
        frequencia_positiva = proporcao(positivas, total)
        selecionada = (
            total >= int(limites["minimo_janelas_testadas"])
            and frequencia_significancia
            >= float(limites["frequencia_minima_significancia"])
            and frequencia_residuos
            >= float(limites["frequencia_minima_residuos"])
            and frequencia_positiva
            >= float(limites["frequencia_minima_resposta_positiva"])
        )
        final = finais[contexto].get(chave, {})
        agregados.append(
            {
                "contexto": contexto,
                "produto": chave[0],
                "origem": chave[1],
                "destino": chave[2],
                "janelas_testadas": total,
                "janelas_significativas": significativas,
                "frequencia_significancia": frequencia_significancia,
                "janelas_residuos_adequados": residuos,
                "frequencia_residuos_adequados": frequencia_residuos,
                "lag_modal": lag_modal,
                "frequencia_lag_modal": proporcao(ocorrencias_lag, total),
                "resposta_acumulada_media": float(
                    np.mean(
                        [linha["resposta_acumulada"] for linha in linhas]
                    )
                ),
                "janelas_resposta_positiva": positivas,
                "frequencia_resposta_positiva": frequencia_positiva,
                "previsoes_avaliadas": len(avaliadas),
                "frequencia_melhora_rmse": proporcao(
                    sum(linha["melhora_rmse"] for linha in avaliadas),
                    len(avaliadas),
                ),
                "frequencia_melhora_mae": proporcao(
                    sum(linha["melhora_mae"] for linha in avaliadas),
                    len(avaliadas),
                ),
                "altas_avaliadas": len(altas_avaliadas),
                "frequencia_melhora_rmse_altas": proporcao(
                    sum(
                        linha["melhora_rmse_altas"]
                        for linha in altas_avaliadas
                    ),
                    len(altas_avaliadas),
                ),
                "frequencia_melhora_mae_altas": proporcao(
                    sum(
                        linha["melhora_mae_altas"]
                        for linha in altas_avaliadas
                    ),
                    len(altas_avaliadas),
                ),
                "selecionada_estabilidade": int(selecionada),
                "significativa_final": final.get("significativo_fdr"),
                "residuos_adequados_final": final.get("residuos_adequados"),
                "melhora_rmse_final": final.get("melhora_rmse"),
                "melhora_mae_final": final.get("melhora_mae"),
                "melhora_rmse_altas_final": final.get(
                    "melhora_rmse_altas"
                ),
                "melhora_mae_altas_final": final.get("melhora_mae_altas"),
            }
        )
    return agregados


def serializar_relacao(registro: dict[str, Any]) -> dict[str, Any]:
    return {
        chave: registro[chave]
        for chave in (
            "produto",
            "origem",
            "destino",
            "janelas_testadas",
            "frequencia_significancia",
            "frequencia_residuos_adequados",
            "lag_modal",
            "frequencia_lag_modal",
            "resposta_acumulada_media",
            "frequencia_resposta_positiva",
            "frequencia_melhora_rmse",
            "frequencia_melhora_mae",
            "frequencia_melhora_rmse_altas",
            "frequencia_melhora_mae_altas",
            "significativa_final",
            "residuos_adequados_final",
            "melhora_rmse_final",
            "melhora_mae_final",
            "melhora_rmse_altas_final",
            "melhora_mae_altas_final",
        )
    }


def calcular_jaccard(
    primeiro: set[ChaveRelacao],
    segundo: set[ChaveRelacao],
) -> float:
    uniao = primeiro | segundo
    return len(primeiro & segundo) / len(uniao) if uniao else 1.0


def construir_relatorio(
    configuracao: dict[str, Any],
    janelas: list[dict[str, Any]],
    agregados: list[dict[str, Any]],
    finais: dict[str, dict[ChaveRelacao, dict[str, int]]],
) -> dict[str, Any]:
    selecionadas: dict[str, set[ChaveRelacao]] = {}
    significativas_finais: dict[str, set[ChaveRelacao]] = {}
    validacoes: dict[str, Any] = {}
    for contexto in ("validacao_1", "validacao_2"):
        registros = [
            registro
            for registro in agregados
            if registro["contexto"] == contexto
        ]
        escolhidas = [
            registro
            for registro in registros
            if registro["selecionada_estabilidade"]
        ]
        selecionadas[contexto] = {
            (registro["produto"], registro["origem"], registro["destino"])
            for registro in escolhidas
        }
        significativas_finais[contexto] = {
            chave
            for chave, resultado in finais[contexto].items()
            if resultado["significativo_fdr"]
        }
        validacoes[contexto] = {
            "janelas_executadas": sum(
                janela["contexto"] == contexto for janela in janelas
            ),
            "direcoes_eligiveis": len(registros),
            "relacoes_selecionadas": len(escolhidas),
            "selecionadas_significativas_no_modelo_final": sum(
                bool(registro["significativa_final"])
                for registro in escolhidas
            ),
            "selecionadas_com_residuos_adequados_no_modelo_final": sum(
                bool(registro["residuos_adequados_final"])
                for registro in escolhidas
            ),
            "selecionadas_com_ganho_geral_no_teste": sum(
                bool(registro["melhora_rmse_final"])
                and bool(registro["melhora_mae_final"])
                for registro in escolhidas
            ),
            "selecionadas_com_ganho_nas_altas_no_teste": sum(
                bool(registro["melhora_rmse_altas_final"])
                and bool(registro["melhora_mae_altas_final"])
                for registro in escolhidas
            ),
            "relacoes": [
                serializar_relacao(registro) for registro in escolhidas
            ],
        }

    comuns = selecionadas["validacao_1"] & selecionadas["validacao_2"]
    finais_comuns = (
        significativas_finais["validacao_1"]
        & significativas_finais["validacao_2"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "gerado_em": datetime.now(UTC).isoformat(),
        "metodo": configuracao["metodo"],
        "limites": configuracao["limites"],
        "horizonte_resposta_dias_uteis": configuracao[
            "horizonte_resposta_dias_uteis"
        ],
        "validacoes": validacoes,
        "comparacao": {
            "jaccard_condicional_final": calcular_jaccard(
                significativas_finais["validacao_1"],
                significativas_finais["validacao_2"],
            ),
            "jaccard_estabilidade_temporal": calcular_jaccard(
                selecionadas["validacao_1"],
                selecionadas["validacao_2"],
            ),
            "significativas_condicionais_nas_duas": len(finais_comuns),
            "selecionadas_nas_duas": len(comuns),
            "relacoes_selecionadas_nas_duas": [
                {"produto": produto, "origem": origem, "destino": destino}
                for produto, origem, destino in sorted(comuns)
            ],
        },
    }


def criar_esquema(conexao: sqlite3.Connection) -> None:
    conexao.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE execucoes_estabilidade (
            id INTEGER PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            iniciada_em TEXT NOT NULL,
            concluida_em TEXT,
            configuracao_json TEXT NOT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE janelas (
            id TEXT PRIMARY KEY,
            contexto TEXT NOT NULL,
            data_inicio_treino TEXT NOT NULL,
            data_corte_treino TEXT NOT NULL,
            data_fim_teste TEXT NOT NULL,
            produtos_testados INTEGER NOT NULL,
            direcoes_testadas INTEGER NOT NULL
        );

        CREATE TABLE resultados_janelas (
            janela_id TEXT NOT NULL,
            contexto TEXT NOT NULL,
            produto TEXT NOT NULL,
            origem TEXT NOT NULL,
            destino TEXT NOT NULL,
            lag_selecionado INTEGER NOT NULL,
            n_observacoes INTEGER NOT NULL,
            p_valor REAL NOT NULL,
            q_valor_fdr REAL NOT NULL,
            significativo_fdr INTEGER NOT NULL,
            residuos_adequados INTEGER NOT NULL,
            status_avaliacao TEXT NOT NULL,
            melhora_rmse INTEGER NOT NULL,
            melhora_mae INTEGER NOT NULL,
            altas_avaliadas INTEGER NOT NULL,
            melhora_rmse_altas INTEGER NOT NULL,
            melhora_mae_altas INTEGER NOT NULL,
            resposta_acumulada REAL NOT NULL,
            resposta_positiva INTEGER NOT NULL,
            PRIMARY KEY (janela_id, produto, origem, destino),
            FOREIGN KEY (janela_id) REFERENCES janelas(id)
        );

        CREATE TABLE relacoes_estabilidade (
            contexto TEXT NOT NULL,
            produto TEXT NOT NULL,
            origem TEXT NOT NULL,
            destino TEXT NOT NULL,
            janelas_testadas INTEGER NOT NULL,
            janelas_significativas INTEGER NOT NULL,
            frequencia_significancia REAL NOT NULL,
            janelas_residuos_adequados INTEGER NOT NULL,
            frequencia_residuos_adequados REAL NOT NULL,
            lag_modal INTEGER NOT NULL,
            frequencia_lag_modal REAL NOT NULL,
            resposta_acumulada_media REAL NOT NULL,
            janelas_resposta_positiva INTEGER NOT NULL,
            frequencia_resposta_positiva REAL NOT NULL,
            previsoes_avaliadas INTEGER NOT NULL,
            frequencia_melhora_rmse REAL NOT NULL,
            frequencia_melhora_mae REAL NOT NULL,
            altas_avaliadas INTEGER NOT NULL,
            frequencia_melhora_rmse_altas REAL NOT NULL,
            frequencia_melhora_mae_altas REAL NOT NULL,
            selecionada_estabilidade INTEGER NOT NULL,
            significativa_final INTEGER,
            residuos_adequados_final INTEGER,
            melhora_rmse_final INTEGER,
            melhora_mae_final INTEGER,
            melhora_rmse_altas_final INTEGER,
            melhora_mae_altas_final INTEGER,
            PRIMARY KEY (contexto, produto, origem, destino)
        );

        CREATE INDEX idx_estabilidade_selecionada
            ON relacoes_estabilidade(contexto, selecionada_estabilidade);
        """
    )


def gravar_banco(
    caminho: Path,
    configuracao: dict[str, Any],
    janelas: list[dict[str, Any]],
    resultados: list[dict[str, Any]],
    agregados: list[dict[str, Any]],
) -> None:
    if caminho.exists():
        raise FileExistsError(f"a saída já existe: {caminho}")
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{caminho.stem}-",
        suffix=".tmp.sqlite",
        dir=caminho.parent,
        delete=False,
    )
    temporario = Path(handle.name)
    handle.close()
    conexao: sqlite3.Connection | None = None
    try:
        conexao = sqlite3.connect(temporario)
        criar_esquema(conexao)
        iniciada_em = datetime.now(UTC).isoformat()
        conexao.execute(
            """
            INSERT INTO execucoes_estabilidade (
                id, schema_version, iniciada_em, configuracao_json, status
            ) VALUES (1, ?, ?, ?, 'executando')
            """,
            (
                SCHEMA_VERSION,
                iniciada_em,
                json.dumps(configuracao, ensure_ascii=False, sort_keys=True),
            ),
        )
        conexao.executemany(
            """
            INSERT INTO janelas VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    janela["id"],
                    janela["contexto"],
                    janela["data_inicio_treino"],
                    janela["data_corte_treino"],
                    janela["data_fim_teste"],
                    janela["produtos_testados"],
                    janela["direcoes_testadas"],
                )
                for janela in janelas
            ],
        )
        conexao.executemany(
            """
            INSERT INTO resultados_janelas VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    resultado["janela_id"],
                    resultado["contexto"],
                    resultado["produto"],
                    resultado["origem"],
                    resultado["destino"],
                    resultado["lag_selecionado"],
                    resultado["n_observacoes"],
                    resultado["p_valor"],
                    resultado["q_valor_fdr"],
                    resultado["significativo_fdr"],
                    resultado["sem_autocorrelacao_irrestrito"],
                    resultado["status_avaliacao"],
                    resultado["melhora_rmse"],
                    resultado["melhora_mae"],
                    resultado["altas_avaliadas"],
                    resultado["melhora_rmse_altas"],
                    resultado["melhora_mae_altas"],
                    resultado["resposta_acumulada"],
                    resultado["resposta_positiva"],
                )
                for resultado in resultados
            ],
        )
        colunas_agregadas = tuple(agregados[0]) if agregados else ()
        if colunas_agregadas:
            marcadores = ", ".join("?" for _ in colunas_agregadas)
            nomes_colunas = ", ".join(colunas_agregadas)
            conexao.executemany(
                f"INSERT INTO relacoes_estabilidade ({nomes_colunas}) "
                f"VALUES ({marcadores})",
                [
                    tuple(registro[coluna] for coluna in colunas_agregadas)
                    for registro in agregados
                ],
            )
        conexao.execute(
            """
            UPDATE execucoes_estabilidade
            SET concluida_em = ?, status = 'concluida'
            WHERE id = 1
            """,
            (datetime.now(UTC).isoformat(),),
        )
        conexao.commit()
        if conexao.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("falha de integridade no banco de estabilidade")
        conexao.close()
        conexao = None
        os.replace(temporario, caminho)
    finally:
        if conexao is not None:
            conexao.close()
        if temporario.exists():
            temporario.unlink()


def gravar_json(caminho: Path, valor: dict[str, Any]) -> None:
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(valor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def executar(args: argparse.Namespace) -> dict[str, Any]:
    configuracao = json.loads(args.configuracao.read_text(encoding="utf-8"))
    if int(configuracao.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("schema da configuração de estabilidade incompatível")
    horizonte = int(configuracao["horizonte_resposta_dias_uteis"])
    finais = {
        "validacao_1": carregar_resultados_finais(args.validacao_1),
        "validacao_2": carregar_resultados_finais(args.validacao_2),
    }
    metadados_janelas: list[dict[str, Any]] = []
    resultados: list[dict[str, Any]] = []
    hashes_configuracao: set[str] = set()
    for janela in configuracao["janelas"]:
        metadados, linhas = carregar_janela(
            janela,
            horizonte,
            args.diretorio_janelas,
        )
        hashes_configuracao.add(metadados["configuracao_redes_sha256"])
        metadados_janelas.append(metadados)
        for linha in linhas:
            resultados.append(
                {
                    "janela_id": janela["id"],
                    "contexto": janela["contexto"],
                    **linha,
                }
            )
    if hashes_configuracao != {configuracao["configuracao_redes_sha256"]}:
        raise ValueError("as janelas não usam a mesma configuração de redes")

    agregados = agregar_relacoes(
        resultados,
        configuracao["limites"],
        finais,
    )
    relatorio = construir_relatorio(
        configuracao,
        metadados_janelas,
        agregados,
        finais,
    )
    gravar_banco(
        args.output_db,
        configuracao,
        metadados_janelas,
        resultados,
        agregados,
    )
    gravar_json(args.output_json, relatorio)
    for contexto, resumo in relatorio["validacoes"].items():
        print(f"{contexto}:")
        print(f"  janelas: {resumo['janelas_executadas']}")
        print(f"  relações selecionadas: {resumo['relacoes_selecionadas']}")
        print(
            "  selecionadas com ganho geral/altas: "
            f"{resumo['selecionadas_com_ganho_geral_no_teste']}/"
            f"{resumo['selecionadas_com_ganho_nas_altas_no_teste']}"
        )
    print("Comparação:")
    for chave, valor in relatorio["comparacao"].items():
        if not isinstance(valor, list):
            print(f"  {chave}: {valor}")
    return relatorio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuracao", type=Path, required=True)
    parser.add_argument("--diretorio-janelas", type=Path, required=True)
    parser.add_argument("--validacao-1", type=Path, required=True)
    parser.add_argument("--validacao-2", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_db.parent.mkdir(parents=True, exist_ok=True)
    executar(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
