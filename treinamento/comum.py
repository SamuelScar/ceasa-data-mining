"""Componentes compartilhados pelo treinamento e pela avaliação final."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SCHEMA_VERSION = 2
ALVO = "efeito_domino"
COLUNA_GRUPO = "grupo_choque"
COLUNA_COBERTURA = "dias_observados_destino"
COBERTURAS_MINIMAS_NEGATIVO = (1, 2, 3, 4, 5)
COLUNAS_CALENDARIO_CATEGORICAS = {"mes", "trimestre", "dia_semana"}
VISOES_VALIDACAO = (
    "treino_validacao_1",
    "validacao_1",
    "treino_validacao_2",
    "validacao_2",
)
VISOES_FINAIS = ("treino_final", "teste_final")
VISOES_POR_CONTEXTO = {
    "validacao_1": ("treino_validacao_1", "validacao_1"),
    "validacao_2": ("treino_validacao_2", "validacao_2"),
    "final": VISOES_FINAIS,
}


@dataclass(frozen=True)
class AtributosModelo:
    categoricos: tuple[str, ...]
    numericos: tuple[str, ...]

    @property
    def todos(self) -> tuple[str, ...]:
        return self.categoricos + self.numericos

    def como_dict(self) -> dict[str, list[str]]:
        return {
            "categoricos": list(self.categoricos),
            "numericos": list(self.numericos),
        }


@dataclass(frozen=True)
class ConfiguracaoModelo:
    codigo: str
    familia: str
    parametros: dict[str, Any]


class TaxaHistoricaPorRelacao(ClassifierMixin, BaseEstimator):
    """Baseline suavizada que aprende apenas a taxa histórica da relação."""

    def __init__(self, suavizacao: float = 10.0):
        self.suavizacao = suavizacao

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TaxaHistoricaPorRelacao":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("a baseline histórica exige um DataFrame")
        colunas = ["produto", "origem", "destino"]
        ausentes = set(colunas) - set(X.columns)
        if ausentes:
            raise ValueError(
                "atributos ausentes para a baseline histórica: "
                + ", ".join(sorted(ausentes))
            )
        y_array = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1], dtype=int)
        self.taxa_global_ = float(np.mean(y_array))
        frame = X[colunas].astype("string").fillna("<ausente>").copy()
        frame[ALVO] = y_array
        agregado = frame.groupby(colunas, dropna=False)[ALVO].agg(["sum", "count"])
        taxas = (
            agregado["sum"] + self.suavizacao * self.taxa_global_
        ) / (agregado["count"] + self.suavizacao)
        self.taxas_ = taxas.to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        colunas = ["produto", "origem", "destino"]
        frame = X[colunas].astype("string").fillna("<ausente>")
        probabilidades = np.array(
            [
                self.taxas_.get(tuple(linha), self.taxa_global_)
                for linha in frame.itertuples(index=False, name=None)
            ],
            dtype=float,
        )
        return np.column_stack([1.0 - probabilidades, probabilidades])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def sha256_arquivo(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def abrir_base_somente_leitura(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"base de modelagem não encontrada: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def validar_base(
    connection: sqlite3.Connection,
    visoes_obrigatorias: tuple[str, ...],
    contexto_esperado: str | None = None,
) -> dict[str, Any]:
    objetos = {
        (str(linha[0]), str(linha[1]))
        for linha in connection.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    tabelas = {nome for nome, tipo in objetos if tipo == "table"}
    visoes = {nome for nome, tipo in objetos if tipo == "view"}
    tabelas_necessarias = {
        "execucoes_modelagem",
        "catalogo_atributos",
        "auditoria_eventos",
    }
    ausentes = tabelas_necessarias - tabelas
    if ausentes:
        raise ValueError(
            "base de modelagem incompleta; tabelas ausentes: "
            + ", ".join(sorted(ausentes))
        )
    ausentes = set(visoes_obrigatorias) - visoes
    if ausentes:
        raise ValueError(
            "base de modelagem incompleta; visões ausentes: "
            + ", ".join(sorted(ausentes))
        )
    execucao = connection.execute(
        "SELECT * FROM execucoes_modelagem ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if execucao is None or execucao["status"] != "concluida":
        raise ValueError("a base não possui uma execução de modelagem concluída")
    resultado = dict(execucao)
    if contexto_esperado is not None:
        if int(resultado.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("versão incompatível da base de modelagem contextual")
        contexto = resultado.get("contexto")
        if contexto != contexto_esperado:
            raise ValueError(
                f"a base deveria representar {contexto_esperado}, "
                f"mas registra {contexto!r}"
            )
        assinatura = str(resultado.get("metodologia_sha256") or "")
        if len(assinatura) != 64:
            raise ValueError("a base não registra uma assinatura metodológica válida")
    return resultado


def carregar_atributos(connection: sqlite3.Connection) -> AtributosModelo:
    linhas = connection.execute(
        """
        SELECT nome, papel
        FROM catalogo_atributos
        WHERE usar_modelo = 1
        ORDER BY id
        """
    ).fetchall()
    if not linhas:
        raise ValueError("o catálogo não possui atributos habilitados para o modelo")

    categoricos: list[str] = []
    numericos: list[str] = []
    for linha in linhas:
        nome = str(linha["nome"])
        papel = str(linha["papel"])
        if papel == "preditora_categorica" or nome in COLUNAS_CALENDARIO_CATEGORICAS:
            categoricos.append(nome)
        elif papel == "preditora_numerica":
            numericos.append(nome)
        else:
            raise ValueError(f"papel de atributo não reconhecido: {nome}={papel}")
    return AtributosModelo(tuple(categoricos), tuple(numericos))


def assinatura_atributos(atributos: AtributosModelo) -> str:
    conteudo = json.dumps(
        atributos.como_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(conteudo).hexdigest()


def carregar_visao(
    connection: sqlite3.Connection,
    nome: str,
    atributos: AtributosModelo,
) -> pd.DataFrame:
    permitidas = set(VISOES_VALIDACAO) | set(VISOES_FINAIS)
    if nome not in permitidas:
        raise ValueError(f"visão de modelagem não permitida: {nome}")
    frame = pd.read_sql_query(
        f"""
        SELECT v.*, a.dias_observados_destino
        FROM "{nome}" v
        JOIN auditoria_eventos a ON a.id = v.id
        ORDER BY v.id
        """,
        connection,
    )
    if frame.empty:
        raise ValueError(f"a visão {nome} está vazia")
    obrigatorias = {
        "id",
        "data_choque",
        COLUNA_GRUPO,
        COLUNA_COBERTURA,
        ALVO,
        *atributos.todos,
    }
    ausentes = obrigatorias - set(frame.columns)
    if ausentes:
        raise ValueError(
            f"a visão {nome} não contém: " + ", ".join(sorted(ausentes))
        )
    if frame["id"].duplicated().any():
        raise ValueError(f"a visão {nome} possui eventos duplicados")
    alvos = set(frame[ALVO].dropna().astype(int).unique())
    if frame[ALVO].isna().any() or not alvos.issubset({0, 1}):
        raise ValueError(f"a visão {nome} contém rótulos inválidos")
    if len(alvos) != 2:
        raise ValueError(f"a visão {nome} precisa conter as duas classes")
    cobertura = pd.to_numeric(frame[COLUNA_COBERTURA], errors="coerce")
    cobertura_valida = (
        cobertura.notna()
        & (cobertura == np.floor(cobertura))
        & cobertura.between(
            min(COBERTURAS_MINIMAS_NEGATIVO),
            max(COBERTURAS_MINIMAS_NEGATIVO),
        )
    )
    if not cobertura_valida.all():
        raise ValueError(f"a visão {nome} contém cobertura futura inválida")
    frame[COLUNA_COBERTURA] = cobertura.astype(int)
    return frame


def validar_separacao_temporal(
    treino: pd.DataFrame,
    avaliacao: pd.DataFrame,
    nome_treino: str,
    nome_avaliacao: str,
) -> None:
    datas_treino = pd.to_datetime(treino["data_choque"], errors="raise")
    datas_avaliacao = pd.to_datetime(avaliacao["data_choque"], errors="raise")
    if datas_treino.max() >= datas_avaliacao.min():
        raise ValueError(
            f"ordem temporal inválida entre {nome_treino} e {nome_avaliacao}"
        )
    sobrepostos = set(treino["id"]) & set(avaliacao["id"])
    if sobrepostos:
        raise ValueError(
            f"há eventos compartilhados entre {nome_treino} e {nome_avaliacao}"
        )


def preparar_matriz(
    frame: pd.DataFrame,
    atributos: AtributosModelo,
) -> tuple[pd.DataFrame, pd.Series]:
    X = frame.loc[:, list(atributos.todos)].copy()
    for coluna in atributos.categoricos:
        valores = X[coluna]
        X[coluna] = valores.astype("object").where(valores.notna(), np.nan)
    for coluna in atributos.numericos:
        X[coluna] = pd.to_numeric(X[coluna], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    y = frame[ALVO].astype(int)
    return X, y


def configuracoes_candidatas() -> list[ConfiguracaoModelo]:
    configuracoes = [
        ConfiguracaoModelo("dummy_prior", "dummy", {"strategy": "prior"}),
        ConfiguracaoModelo(
            "historica_s05", "taxa_historica_relacao", {"suavizacao": 5.0}
        ),
        ConfiguracaoModelo(
            "historica_s20", "taxa_historica_relacao", {"suavizacao": 20.0}
        ),
    ]
    for c in (0.1, 1.0, 10.0):
        for peso in (None, "balanced"):
            sufixo = "normal" if peso is None else "balanceada"
            configuracoes.append(
                ConfiguracaoModelo(
                    f"logistica_c{c:g}_{sufixo}",
                    "regressao_logistica",
                    {"C": c, "class_weight": peso},
                )
            )
    for profundidade in (4, 8):
        for peso in (None, "balanced"):
            sufixo = "normal" if peso is None else "balanceada"
            configuracoes.append(
                ConfiguracaoModelo(
                    f"arvore_d{profundidade}_{sufixo}",
                    "arvore_decisao",
                    {
                        "max_depth": profundidade,
                        "min_samples_leaf": 10,
                        "class_weight": peso,
                    },
                )
            )
    for folha in (2, 5):
        for peso in (None, "balanced"):
            sufixo = "normal" if peso is None else "balanceada"
            configuracoes.append(
                ConfiguracaoModelo(
                    f"floresta_f{folha}_{sufixo}",
                    "random_forest",
                    {
                        "n_estimators": 300,
                        "min_samples_leaf": folha,
                        "max_features": "sqrt",
                        "class_weight": peso,
                    },
                )
            )
    for taxa in (0.05, 0.1):
        configuracoes.append(
            ConfiguracaoModelo(
                f"gradient_boosting_lr{taxa:g}",
                "gradient_boosting",
                {
                    "n_estimators": 150,
                    "learning_rate": taxa,
                    "max_depth": 2,
                },
            )
        )
    for c in (0.5, 1.0, 2.0):
        for peso in (None, "balanced"):
            sufixo = "normal" if peso is None else "balanceada"
            configuracoes.append(
                ConfiguracaoModelo(
                    f"svm_c{c:g}_{sufixo}",
                    "svm_rbf",
                    {"C": c, "class_weight": peso},
                )
            )
    return configuracoes


def construir_preprocessador(atributos: AtributosModelo) -> ColumnTransformer:
    numerico = Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="median", add_indicator=True)),
            ("escala", StandardScaler()),
        ]
    )
    categorico = Pipeline(
        steps=[
            (
                "imputacao",
                SimpleImputer(strategy="constant", fill_value="<ausente>"),
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numericos", numerico, list(atributos.numericos)),
            ("categoricos", categorico, list(atributos.categoricos)),
        ],
        remainder="drop",
    )


def construir_estimador(
    configuracao: ConfiguracaoModelo,
    atributos: AtributosModelo,
    semente: int,
) -> BaseEstimator:
    familia = configuracao.familia
    parametros = configuracao.parametros
    if familia == "taxa_historica_relacao":
        return TaxaHistoricaPorRelacao(**parametros)
    if familia == "dummy":
        modelo: BaseEstimator = DummyClassifier(**parametros)
    elif familia == "regressao_logistica":
        modelo = LogisticRegression(
            **parametros,
            max_iter=2_000,
            random_state=semente,
        )
    elif familia == "arvore_decisao":
        modelo = DecisionTreeClassifier(**parametros, random_state=semente)
    elif familia == "random_forest":
        modelo = RandomForestClassifier(
            **parametros,
            random_state=semente,
            n_jobs=-1,
        )
    elif familia == "gradient_boosting":
        modelo = GradientBoostingClassifier(**parametros, random_state=semente)
    elif familia == "svm_rbf":
        modelo = SVC(
            **parametros,
            kernel="rbf",
            probability=True,
            random_state=semente,
        )
    else:
        raise ValueError(f"família de modelo desconhecida: {familia}")
    return Pipeline(
        steps=[
            ("preprocessador", construir_preprocessador(atributos)),
            ("modelo", modelo),
        ]
    )


def probabilidades_positivas(modelo: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    probabilidades = np.asarray(modelo.predict_proba(X), dtype=float)
    classes = np.asarray(modelo.classes_)
    indices = np.flatnonzero(classes == 1)
    if probabilidades.ndim != 2 or len(indices) != 1:
        raise ValueError("o modelo não produziu probabilidade para a classe positiva")
    resultado = probabilidades[:, int(indices[0])]
    if not np.isfinite(resultado).all() or ((resultado < 0) | (resultado > 1)).any():
        raise ValueError("o modelo produziu probabilidades inválidas")
    return resultado


def pesos_por_grupo(grupos: pd.Series) -> np.ndarray:
    contagens = grupos.astype("string").value_counts(dropna=False)
    pesos = grupos.astype("string").map(lambda grupo: 1.0 / contagens[grupo])
    return pesos.to_numpy(dtype=float)


def calcular_metricas(
    y: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
    limiar: float,
    grupos: pd.Series | None = None,
) -> dict[str, float]:
    y_array = np.asarray(y, dtype=int)
    previsoes = (probabilidades >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, previsoes, labels=[0, 1]).ravel()
    metricas = {
        "f1": float(f1_score(y_array, previsoes, zero_division=0)),
        "precisao": float(precision_score(y_array, previsoes, zero_division=0)),
        "revocacao": float(recall_score(y_array, previsoes, zero_division=0)),
        "acuracia": float(accuracy_score(y_array, previsoes)),
        "acuracia_balanceada": float(balanced_accuracy_score(y_array, previsoes)),
        "mcc": float(matthews_corrcoef(y_array, previsoes)),
        "pr_auc": float(average_precision_score(y_array, probabilidades)),
        "roc_auc": float(roc_auc_score(y_array, probabilidades)),
        "brier": float(brier_score_loss(y_array, probabilidades)),
        "verdadeiros_negativos": float(tn),
        "falsos_positivos": float(fp),
        "falsos_negativos": float(fn),
        "verdadeiros_positivos": float(tp),
    }
    if grupos is not None:
        pesos = pesos_por_grupo(grupos)
        metricas.update(
            {
                "f1_ponderado_grupo": float(
                    f1_score(y_array, previsoes, sample_weight=pesos, zero_division=0)
                ),
                "precisao_ponderada_grupo": float(
                    precision_score(
                        y_array,
                        previsoes,
                        sample_weight=pesos,
                        zero_division=0,
                    )
                ),
                "revocacao_ponderada_grupo": float(
                    recall_score(
                        y_array,
                        previsoes,
                        sample_weight=pesos,
                        zero_division=0,
                    )
                ),
                "acuracia_ponderada_grupo": float(
                    accuracy_score(y_array, previsoes, sample_weight=pesos)
                ),
            }
        )
    return metricas


def calcular_sensibilidade_cobertura(
    previsoes: pd.DataFrame,
    limiar: float,
) -> list[dict[str, Any]]:
    obrigatorias = {
        ALVO,
        COLUNA_GRUPO,
        COLUNA_COBERTURA,
        "probabilidade",
    }
    ausentes = obrigatorias - set(previsoes.columns)
    if ausentes:
        raise ValueError(
            "previsões sem colunas para a sensibilidade de cobertura: "
            + ", ".join(sorted(ausentes))
        )

    resultados: list[dict[str, Any]] = []
    for cobertura_minima in COBERTURAS_MINIMAS_NEGATIVO:
        mascara = (previsoes[ALVO] == 1) | (
            (previsoes[ALVO] == 0)
            & (previsoes[COLUNA_COBERTURA] >= cobertura_minima)
        )
        cenario = previsoes.loc[mascara]
        contagens = cenario[ALVO].value_counts().to_dict()
        positivos = int(contagens.get(1, 0))
        negativos = int(contagens.get(0, 0))
        avaliavel = positivos > 0 and negativos > 0
        motivo_nao_avaliavel = None
        if not avaliavel:
            motivo_nao_avaliavel = (
                "cenário sem as duas classes após o filtro de cobertura"
            )
        resultados.append(
            {
                "cobertura_minima_negativo": cobertura_minima,
                "linhas_avaliadas": int(len(cenario)),
                "positivos": positivos,
                "negativos": negativos,
                "avaliavel": avaliavel,
                "motivo_nao_avaliavel": motivo_nao_avaliavel,
                "metricas": (
                    calcular_metricas(
                        cenario[ALVO],
                        cenario["probabilidade"].to_numpy(),
                        limiar,
                        cenario[COLUNA_GRUPO],
                    )
                    if avaliavel
                    else {}
                ),
            }
        )
    return resultados


def selecionar_limiar(
    previsoes_por_divisao: dict[str, pd.DataFrame],
) -> tuple[float, dict[str, float]]:
    melhor_limiar = 0.5
    melhor_resumo: dict[str, float] = {}
    melhor_chave: tuple[float, float, float, float] | None = None
    for limiar in np.round(np.arange(0.05, 0.951, 0.05), 2):
        por_divisao = [
            calcular_metricas(
                frame[ALVO],
                frame["probabilidade"].to_numpy(),
                float(limiar),
            )
            for frame in previsoes_por_divisao.values()
        ]
        resumo = {
            metrica: float(np.mean([valores[metrica] for valores in por_divisao]))
            for metrica in ("f1", "mcc", "precisao", "revocacao")
        }
        chave = (
            resumo["f1"],
            resumo["mcc"],
            resumo["precisao"],
            -abs(float(limiar) - 0.5),
        )
        if melhor_chave is None or chave > melhor_chave:
            melhor_chave = chave
            melhor_limiar = float(limiar)
            melhor_resumo = resumo
    return melhor_limiar, melhor_resumo


def parametros_json(parametros: dict[str, Any]) -> str:
    return json.dumps(parametros, ensure_ascii=False, sort_keys=True)
