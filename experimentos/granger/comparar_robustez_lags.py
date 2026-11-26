#!/usr/bin/env python3
"""Compara o experimento encerrado de limites máximos de lags."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LAGS_ESPERADOS = (5, 10, 15, 20)
ChaveRelacao = tuple[str, str, str]


@dataclass(frozen=True)
class Resultado:
    status: str
    n_observacoes: int | None
    lag_selecionado: int | None
    bic: float | None
    p_valor: float | None
    q_valor_fdr: float | None
    significativo_fdr: bool


@dataclass(frozen=True)
class Execucao:
    caminho: Path
    max_lag: int
    max_lag_amostra: int
    banco_origem_sha256: str
    parametros: dict[str, Any]
    resultados: dict[ChaveRelacao, Resultado]
    bics: dict[ChaveRelacao, dict[int, float]]


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


def carregar_execucao(caminho: Path) -> Execucao:
    with abrir_somente_leitura(caminho) as conexao:
        integridade = conexao.execute("PRAGMA quick_check").fetchone()[0]
        if integridade != "ok":
            raise RuntimeError(f"{caminho}: falha de integridade: {integridade}")

        registro = conexao.execute(
            """
            SELECT banco_origem_sha256, parametros_json, status
            FROM execucoes_granger
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if registro is None or registro["status"] != "concluida":
            raise RuntimeError(f"{caminho}: execução de Granger não concluída")

        parametros = json.loads(registro["parametros_json"])
        max_lag = int(parametros["max_lag"])
        max_lag_amostra = int(parametros.get("max_lag_amostra", max_lag))

        resultados: dict[ChaveRelacao, Resultado] = {}
        for linha in conexao.execute(
            """
            SELECT produto, origem, destino, status, n_observacoes,
                   lag_selecionado, bic, p_valor, q_valor_fdr,
                   significativo_fdr
            FROM resultados_granger
            ORDER BY produto, origem, destino
            """
        ):
            chave = (linha["produto"], linha["origem"], linha["destino"])
            resultados[chave] = Resultado(
                status=linha["status"],
                n_observacoes=linha["n_observacoes"],
                lag_selecionado=linha["lag_selecionado"],
                bic=linha["bic"],
                p_valor=linha["p_valor"],
                q_valor_fdr=linha["q_valor_fdr"],
                significativo_fdr=bool(linha["significativo_fdr"]),
            )

        bics: dict[ChaveRelacao, dict[int, float]] = {}
        observacoes: dict[ChaveRelacao, set[int]] = {}
        for linha in conexao.execute(
            """
            SELECT resultado.produto, resultado.origem, resultado.destino,
                   lag.lag, lag.bic, lag.n_observacoes
            FROM resultados_lags AS lag
            JOIN resultados_granger AS resultado
              ON resultado.id = lag.resultado_id
            ORDER BY resultado.produto, resultado.origem,
                     resultado.destino, lag.lag
            """
        ):
            chave = (linha["produto"], linha["origem"], linha["destino"])
            bics.setdefault(chave, {})[int(linha["lag"])] = float(linha["bic"])
            observacoes.setdefault(chave, set()).add(int(linha["n_observacoes"]))

        amostras_inconsistentes = [
            chave for chave, quantidades in observacoes.items() if len(quantidades) != 1
        ]
        if amostras_inconsistentes:
            raise RuntimeError(
                f"{caminho}: uma direção usa amostras diferentes entre os lags"
            )

    return Execucao(
        caminho=caminho,
        max_lag=max_lag,
        max_lag_amostra=max_lag_amostra,
        banco_origem_sha256=registro["banco_origem_sha256"],
        parametros=parametros,
        resultados=resultados,
        bics=bics,
    )


def parametros_comparaveis(parametros: dict[str, Any]) -> dict[str, Any]:
    ignorados = {"max_lag", "max_lag_amostra", "selecao_lag"}
    return {
        chave: valor
        for chave, valor in parametros.items()
        if chave not in ignorados
    }


def validar_execucoes(execucoes: list[Execucao]) -> list[Execucao]:
    ordenadas = sorted(execucoes, key=lambda execucao: execucao.max_lag)
    limites = tuple(execucao.max_lag for execucao in ordenadas)
    if limites != LAGS_ESPERADOS:
        raise ValueError(
            f"os bancos devem representar max_lag={LAGS_ESPERADOS}; "
            f"encontrado: {limites}"
        )

    amostras = {execucao.max_lag_amostra for execucao in ordenadas}
    if amostras != {max(LAGS_ESPERADOS)}:
        raise ValueError(
            "todas as execuções devem usar max_lag_amostra=20; "
            f"encontrado: {sorted(amostras)}"
        )

    referencia = ordenadas[0]
    parametros_referencia = parametros_comparaveis(referencia.parametros)
    chaves_referencia = set(referencia.resultados)

    for execucao in ordenadas[1:]:
        if execucao.banco_origem_sha256 != referencia.banco_origem_sha256:
            raise ValueError("as execuções não usam o mesmo banco pré-processado")
        if parametros_comparaveis(execucao.parametros) != parametros_referencia:
            raise ValueError("as execuções possuem parâmetros não comparáveis")
        if set(execucao.resultados) != chaves_referencia:
            raise ValueError("as execuções não possuem as mesmas direções")

        for chave in chaves_referencia:
            atual = execucao.resultados[chave]
            base = referencia.resultados[chave]
            if atual.status != base.status:
                raise ValueError(f"status divergente para {formatar_chave(chave)}")
            if atual.n_observacoes != base.n_observacoes:
                raise ValueError(
                    f"amostra divergente para {formatar_chave(chave)}"
                )

    validar_bics_compartilhados(ordenadas)
    return ordenadas


def validar_bics_compartilhados(execucoes: list[Execucao]) -> None:
    referencia_por_lag: dict[tuple[ChaveRelacao, int], float] = {}
    for execucao in execucoes:
        for chave, bics in execucao.bics.items():
            for lag, bic in bics.items():
                identificador = (chave, lag)
                referencia = referencia_por_lag.setdefault(identificador, bic)
                if not math.isclose(bic, referencia, rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(
                        "BIC divergente para o mesmo modelo e amostra: "
                        f"{formatar_chave(chave)}, lag {lag}"
                    )


def calcular_delta_bic(bics: dict[int, float]) -> float | None:
    ordenados = sorted(bics.values())
    return ordenados[1] - ordenados[0] if len(ordenados) >= 2 else None


def resumir_execucao(execucao: Execucao) -> dict[str, int]:
    testados = [
        resultado
        for resultado in execucao.resultados.values()
        if resultado.status == "testado"
    ]
    deltas = [
        delta
        for bics in execucao.bics.values()
        if (delta := calcular_delta_bic(bics)) is not None
    ]
    return {
        "direcoes_testadas": len(testados),
        "significativas_fdr": sum(
            resultado.significativo_fdr for resultado in testados
        ),
        "lag_no_teto": sum(
            resultado.lag_selecionado == execucao.max_lag
            for resultado in testados
        ),
        "delta_bic_menor_2": sum(delta < 2 for delta in deltas),
        "delta_bic_menor_6": sum(delta < 6 for delta in deltas),
    }


def comparar_execucoes(execucoes: list[Execucao]) -> dict[str, Any]:
    ordenadas = validar_execucoes(execucoes)
    chaves = sorted(ordenadas[0].resultados)
    alteracoes: list[dict[str, Any]] = []
    lags_estaveis = 0
    significativas_todas = 0
    nao_significativas_todas = 0

    for chave in chaves:
        resultados = [execucao.resultados[chave] for execucao in ordenadas]
        flags = [resultado.significativo_fdr for resultado in resultados]
        lags = [resultado.lag_selecionado for resultado in resultados]

        if len(set(lags)) == 1:
            lags_estaveis += 1
        if all(flags):
            significativas_todas += 1
        elif not any(flags):
            nao_significativas_todas += 1

        if len(set(flags)) > 1 or len(set(lags)) > 1:
            alteracoes.append(
                {
                    "produto": chave[0],
                    "origem": chave[1],
                    "destino": chave[2],
                    "lags_selecionados": {
                        str(execucao.max_lag): resultado.lag_selecionado
                        for execucao, resultado in zip(
                            ordenadas,
                            resultados,
                            strict=True,
                        )
                    },
                    "significativo_fdr": {
                        str(execucao.max_lag): resultado.significativo_fdr
                        for execucao, resultado in zip(
                            ordenadas,
                            resultados,
                            strict=True,
                        )
                    },
                }
            )

    return {
        "schema_version": 1,
        "gerado_em": datetime.now(UTC).isoformat(),
        "banco_origem_sha256": ordenadas[0].banco_origem_sha256,
        "max_lag_amostra": ordenadas[0].max_lag_amostra,
        "configuracoes": {
            str(execucao.max_lag): resumir_execucao(execucao)
            for execucao in ordenadas
        },
        "comparacao": {
            "direcoes": len(chaves),
            "lag_estavel_em_todos_limites": lags_estaveis,
            "significativas_em_todos_limites": significativas_todas,
            "nao_significativas_em_todos_limites": nao_significativas_todas,
            "significancia_instavel": sum(
                1
                for alteracao in alteracoes
                if len(set(alteracao["significativo_fdr"].values())) > 1
            ),
            "lag_instavel": sum(
                1
                for alteracao in alteracoes
                if len(set(alteracao["lags_selecionados"].values())) > 1
            ),
        },
        "alteracoes": alteracoes,
    }


def formatar_chave(chave: ChaveRelacao) -> str:
    return f"{chave[0]}: {chave[1]} -> {chave[2]}"


def exibir_relatorio(relatorio: dict[str, Any]) -> None:
    print("=== ROBUSTEZ DOS LAGS ===")
    print(f"Hash da entrada: {relatorio['banco_origem_sha256']}")
    print(f"Amostra fixada em: {relatorio['max_lag_amostra']} lags")

    for limite, resumo in relatorio["configuracoes"].items():
        print(f"\nmax_lag={limite}")
        for chave, valor in resumo.items():
            print(f"  {chave}: {valor}")

    print("\n=== ESTABILIDADE ===")
    for chave, valor in relatorio["comparacao"].items():
        print(f"{chave}: {valor}")

    instaveis = [
        alteracao
        for alteracao in relatorio["alteracoes"]
        if len(set(alteracao["significativo_fdr"].values())) > 1
    ]
    print("\n=== RELAÇÕES COM SIGNIFICÂNCIA INSTÁVEL ===")
    if not instaveis:
        print("Nenhuma.")
        return
    for alteracao in instaveis:
        chave = (
            alteracao["produto"],
            alteracao["origem"],
            alteracao["destino"],
        )
        print(
            f"{formatar_chave(chave)} | "
            f"lags={alteracao['lags_selecionados']} | "
            f"FDR={alteracao['significativo_fdr']}"
        )


def gravar_relatorio(caminho: Path, relatorio: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bancos", nargs=4, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    execucoes = [carregar_execucao(caminho) for caminho in args.bancos]
    relatorio = comparar_execucoes(execucoes)
    exibir_relatorio(relatorio)
    if args.output_json is not None:
        gravar_relatorio(args.output_json, relatorio)
        print(f"\nRelatório gravado em: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
