#!/usr/bin/env python3
"""Compara a modelagem bivariada e assimétrica nas duas validações."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from treinamento.comum import calcular_metricas


METRICAS_PRIMARIAS = ("f1", "pr_auc", "mcc")
METRICAS_RELATORIO = (
    "f1",
    "pr_auc",
    "mcc",
    "precisao",
    "revocacao",
    "acuracia_balanceada",
    "brier",
)


@dataclass(frozen=True)
class ResultadoModelagem:
    tipo: str
    banco: Path
    selecao: Path
    semente: int
    catalogo_configuracoes_sha256: str
    atributos_sha256: str
    modelo: dict[str, Any]
    resumo: dict[str, float]
    metricas: dict[str, dict[str, float]]
    amostras: dict[str, dict[str, int]]


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


def assinatura_json(valor: object) -> str:
    conteudo = json.dumps(
        valor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(conteudo).hexdigest()


def carregar_resultado(
    tipo: str,
    banco: Path,
    selecao: Path,
    modelo_referencia: dict[str, Any] | None = None,
) -> ResultadoModelagem:
    if not selecao.is_file():
        raise FileNotFoundError(f"seleção não encontrada: {selecao}")
    documento = json.loads(selecao.read_text(encoding="utf-8"))
    if documento.get("teste_final_consultado") is not False:
        raise ValueError(f"{tipo}: a seleção não protege o teste final")
    if documento.get("divisoes_consultadas") != [
        "treino_validacao_1",
        "validacao_1",
        "treino_validacao_2",
        "validacao_2",
    ]:
        raise ValueError(f"{tipo}: divisões de desenvolvimento inesperadas")

    with abrir_somente_leitura(banco) as conexao:
        if conexao.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"{tipo}: falha de integridade no banco")
        execucao = conexao.execute(
            """
            SELECT semente, teste_final_consultado, status
            FROM execucoes_validacao
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
        if execucao is None or execucao["status"] != "concluida":
            raise ValueError(f"{tipo}: a validação não está concluída")
        if int(execucao["teste_final_consultado"]) != 0:
            raise ValueError(f"{tipo}: a validação consultou o teste final")

        configuracoes = [
            {
                "codigo": linha["codigo"],
                "familia": linha["familia"],
                "parametros": json.loads(linha["parametros_json"]),
            }
            for linha in conexao.execute(
                """
                SELECT codigo, familia, parametros_json
                FROM configuracoes_modelos
                ORDER BY id
                """
            )
        ]
        if modelo_referencia is None:
            linhas_modelo = conexao.execute(
                """
                SELECT c.id, c.codigo, c.familia, c.parametros_json, r.limiar
                FROM configuracoes_modelos c
                JOIN resumo_configuracoes r ON r.configuracao_id = c.id
                WHERE c.selecionada = 1
                """
            ).fetchall()
            if len(linhas_modelo) != 1:
                raise ValueError(
                    f"{tipo}: deve existir uma configuração selecionada"
                )
            linha_modelo = linhas_modelo[0]
            limiar = float(linha_modelo["limiar"])
        else:
            linha_modelo = conexao.execute(
                """
                SELECT id, codigo, familia, parametros_json
                FROM configuracoes_modelos
                WHERE codigo = ?
                """,
                (str(modelo_referencia["codigo"]),),
            ).fetchone()
            if linha_modelo is None:
                raise ValueError(
                    f"{tipo}: a configuração de referência não foi avaliada"
                )
            parametros = json.loads(str(linha_modelo["parametros_json"]))
            if (
                linha_modelo["familia"] != modelo_referencia["familia"]
                or parametros != modelo_referencia["parametros"]
            ):
                raise ValueError(
                    f"{tipo}: a configuração de referência possui outra definição"
                )
            limiar = float(modelo_referencia["limiar"])

        previsoes = pd.read_sql_query(
            """
            SELECT divisao, alvo, grupo_choque, probabilidade
            FROM previsoes_validacao
            WHERE configuracao_id = ?
            ORDER BY divisao, evento_id
            """,
            conexao,
            params=(int(linha_modelo["id"]),),
        )

    metricas: dict[str, dict[str, float]] = {}
    amostras: dict[str, dict[str, int]] = {}
    for divisao in ("validacao_1", "validacao_2"):
        frame = previsoes.loc[previsoes["divisao"] == divisao]
        if frame.empty:
            raise ValueError(f"{tipo}: previsões ausentes em {divisao}")
        metricas[divisao] = calcular_metricas(
            frame["alvo"],
            frame["probabilidade"].to_numpy(),
            limiar,
            frame["grupo_choque"],
        )
        amostras[divisao] = {
            "linhas": int(len(frame)),
            "positivos": int((frame["alvo"] == 1).sum()),
            "negativos": int((frame["alvo"] == 0).sum()),
        }

    modelo_documento = documento.get("modelo", {})
    if modelo_referencia is None:
        if modelo_documento.get("codigo") != linha_modelo["codigo"]:
            raise ValueError(
                f"{tipo}: o JSON e o SQLite selecionam modelos diferentes"
            )
        if float(modelo_documento.get("limiar")) != limiar:
            raise ValueError(
                f"{tipo}: o JSON e o SQLite registram limiares diferentes"
            )

    def media(metrica: str) -> float:
        return sum(
            metricas[divisao][metrica]
            for divisao in ("validacao_1", "validacao_2")
        ) / 2

    return ResultadoModelagem(
        tipo=tipo,
        banco=banco,
        selecao=selecao,
        semente=int(execucao["semente"]),
        catalogo_configuracoes_sha256=assinatura_json(configuracoes),
        atributos_sha256=str(documento["atributos_sha256"]),
        modelo={
            "codigo": str(linha_modelo["codigo"]),
            "familia": str(linha_modelo["familia"]),
            "parametros": json.loads(str(linha_modelo["parametros_json"])),
            "limiar": limiar,
        },
        resumo={
            "f1_medio": media("f1"),
            "pr_auc_media": media("pr_auc"),
            "mcc_medio": media("mcc"),
            "precisao_media": media("precisao"),
            "revocacao_media": media("revocacao"),
        },
        metricas=metricas,
        amostras=amostras,
    )


def comparar_resultados(
    bivariado: ResultadoModelagem,
    assimetrico: ResultadoModelagem,
    comparacao_domino: dict[str, Any],
) -> dict[str, Any]:
    if bivariado.semente != assimetrico.semente:
        raise ValueError("as alternativas usam sementes diferentes")
    if (
        bivariado.catalogo_configuracoes_sha256
        != assimetrico.catalogo_configuracoes_sha256
    ):
        raise ValueError("as alternativas avaliam configurações diferentes")
    if bivariado.atributos_sha256 != assimetrico.atributos_sha256:
        raise ValueError("as alternativas usam atributos diferentes")
    if bivariado.modelo != assimetrico.modelo:
        raise ValueError(
            "as alternativas não usam o mesmo modelo, parâmetros e limiar"
        )
    if comparacao_domino.get("inclui_teste_final") is not False:
        raise ValueError("a comparação do Efeito Dominó não protege o teste final")

    comparacoes: dict[str, Any] = {}
    melhora_por_divisao: dict[str, bool] = {}
    for divisao in ("validacao_1", "validacao_2"):
        metricas_divisao: dict[str, Any] = {}
        for metrica in METRICAS_RELATORIO:
            referencia = bivariado.metricas[divisao][metrica]
            alternativa = assimetrico.metricas[divisao][metrica]
            metricas_divisao[metrica] = {
                "bivariado": referencia,
                "assimetrico": alternativa,
                "delta": alternativa - referencia,
            }
        melhora_por_divisao[divisao] = all(
            metricas_divisao[metrica]["delta"] > 0
            for metrica in METRICAS_PRIMARIAS
        )
        comparacoes[divisao] = {
            "amostras": {
                "bivariado": bivariado.amostras[divisao],
                "assimetrico": assimetrico.amostras[divisao],
                "proporcao_linhas_preservada": (
                    assimetrico.amostras[divisao]["linhas"]
                    / bivariado.amostras[divisao]["linhas"]
                ),
            },
            "metricas": metricas_divisao,
            "melhora_f1_pr_auc_mcc": melhora_por_divisao[divisao],
            "eventos_efeito_domino": comparacao_domino["contextos"][divisao][
                "periodos"
            ]["teste"],
        }

    melhora_consistente = all(melhora_por_divisao.values())
    return {
        "gerado_em": datetime.now(UTC).isoformat(),
        "inclui_teste_final": False,
        "criterio_melhoria": (
            "F1, PR-AUC e MCC da alternativa devem superar a referência "
            "separadamente nas duas validações"
        ),
        "protocolo_comparacao": (
            "modelo, hiperparâmetros e limiar selecionados pela referência "
            "bivariada e congelados na avaliação assimétrica"
        ),
        "semente": bivariado.semente,
        "catalogo_configuracoes_sha256": (
            bivariado.catalogo_configuracoes_sha256
        ),
        "atributos_sha256": bivariado.atributos_sha256,
        "configuracao_congelada": bivariado.modelo,
        "selecao_assimetrica_participa_da_decisao": False,
        "resumos": {
            "bivariado": bivariado.resumo,
            "assimetrico": assimetrico.resumo,
        },
        "validacoes": comparacoes,
        "melhora_consistente_nas_duas_validacoes": melhora_consistente,
        "decisao": (
            "candidato_a_substituir_bivariado"
            if melhora_consistente
            else "manter_granger_bivariado"
        ),
    }


def exibir_relatorio(relatorio: dict[str, Any]) -> None:
    print("=== MODELAGEM: BIVARIADO X ASSIMÉTRICO ===")
    for divisao, comparacao in relatorio["validacoes"].items():
        print(f"\n{divisao}")
        amostras = comparacao["amostras"]
        print(
            "  amostras: "
            f"{amostras['bivariado']['linhas']} -> "
            f"{amostras['assimetrico']['linhas']}"
        )
        for metrica in METRICAS_PRIMARIAS:
            valores = comparacao["metricas"][metrica]
            print(
                f"  {metrica}: {valores['bivariado']:.4f} -> "
                f"{valores['assimetrico']:.4f} "
                f"(delta={valores['delta']:+.4f})"
            )
        print(
            "  melhora nas três métricas: "
            f"{comparacao['melhora_f1_pr_auc_mcc']}"
        )
    print(f"\nDecisão: {relatorio['decisao']}")


def gravar_relatorio(caminho: Path, relatorio: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporario.replace(caminho)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bivariado-db", type=Path, required=True)
    parser.add_argument("--bivariado-json", type=Path, required=True)
    parser.add_argument("--assimetrico-db", type=Path, required=True)
    parser.add_argument("--assimetrico-json", type=Path, required=True)
    parser.add_argument("--comparacao-domino", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    comparacao_domino = json.loads(
        args.comparacao_domino.read_text(encoding="utf-8")
    )
    bivariado = carregar_resultado(
        "bivariado",
        args.bivariado_db,
        args.bivariado_json,
    )
    assimetrico = carregar_resultado(
        "assimetrico",
        args.assimetrico_db,
        args.assimetrico_json,
        bivariado.modelo,
    )
    relatorio = comparar_resultados(
        bivariado,
        assimetrico,
        comparacao_domino,
    )
    exibir_relatorio(relatorio)
    gravar_relatorio(args.output_json, relatorio)
    print(f"\nRelatório gravado em: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
