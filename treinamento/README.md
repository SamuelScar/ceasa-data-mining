# Treinamento dos modelos

Este arquivo é um apoio operacional. A descrição metodológica principal, com
as regras e justificativas, está no `docs/relatorio/relatorio.tex`.

## Escopo

A implementação consome exclusivamente
`resultados/modelagem/base_modelagem.sqlite`. Não há integração com clima,
câmbio, inflação, transporte ou qualquer banco externo.

O fluxo possui duas etapas que podem ser executadas individualmente:

1. seleção do modelo, dos hiperparâmetros e do limiar nas validações;
2. ajuste em todo o desenvolvimento e avaliação no teste final isolado.

O teste final não integra o `pipeline-completo`. Existe um pipeline dedicado
somente para encadear essas duas etapas mediante comando explícito.

## Pré-requisito

Antes do treinamento, a base precisa ser materializada:

```bash
docker compose run --build --rm base-modelagem
```

Ela deve conter uma execução concluída, o catálogo de atributos e as seis views
temporais. O treinamento valida colunas, rótulos, presença das duas classes,
identificadores duplicados, ordem cronológica e separação dos eventos.

## Pipeline dedicado de treinamento e avaliação

Quando a metodologia e a seleção estiverem prontas para a avaliação oficial,
execute:

```bash
docker compose run --build --rm pipeline-treinamento-avaliacao
```

O comando exige `resultados/modelagem/base_modelagem.sqlite` e executa, nesta
ordem:

1. `validacao-modelos`, refazendo as saídas de desenvolvimento;
2. `avaliacao-final`, usando a configuração que acaba de ser selecionada.

A segunda etapa só começa se a primeira terminar com sucesso. Antes da
validação, o pipeline verifica se já existe
`resultados/treinamento/avaliacao_final.sqlite` ou
`resultados/treinamento/modelo_final.pkl`. Se qualquer um existir, toda a
execução é recusada. Essa verificação antecipada evita substituir os resultados
de validação e somente depois descobrir que os artefatos oficiais não poderiam
ser sobrescritos.

Esse comando consulta o teste final na segunda etapa. Portanto, ele não deve
ser usado durante comparações exploratórias; para isso, use apenas o comando
individual `validacao-modelos` descrito abaixo.

## Seleção nas validações temporais

Execute:

```bash
docker compose run --build --rm validacao-modelos
```

O comando consulta somente:

- `treino_validacao_1`: eventos de 2022 e 2023;
- `validacao_1`: eventos de 2024;
- `treino_validacao_2`: eventos de 2022 a 2024;
- `validacao_2`: eventos de janeiro a outubro de 2025.

As views `treino_final` e `teste_final` não são consultadas por esse comando.

### Preparação dos atributos

Cada configuração recebe um pipeline novo em cada divisão. Nenhuma
transformação é aprendida globalmente.

- Numéricos: conversão de valores inválidos ou infinitos em ausentes,
  imputação pela mediana do treino, indicador de ausência e padronização.
- Categóricos: categoria explícita para ausência e codificação one-hot.
- Categorias inéditas na avaliação são ignoradas sem reajustar o codificador.
- Mês, trimestre e dia da semana são tratados como categorias; o ano permanece
  numérico.

Isso evita usar estatísticas da validação e do teste para preparar o treino.

### Configurações comparadas

São avaliadas 25 configurações:

- prevalência do treino: 1;
- taxa histórica por produto, origem e destino: suavização 5 ou 20;
- Regressão Logística: `C` 0,1, 1 ou 10, com e sem pesos de classe;
- Árvore de Decisão: profundidade 4 ou 8, mínimo de 10 amostras por folha, com
  e sem pesos de classe;
- Random Forest: 300 árvores, mínimo de 2 ou 5 amostras por folha,
  `max_features="sqrt"`, com e sem pesos de classe;
- Gradient Boosting: 150 árvores, profundidade 2 e taxa 0,05 ou 0,10;
- SVM RBF: `C` 0,5, 1 ou 2, com e sem pesos de classe.

A semente padrão é 42 e pode ser alterada por `TREINAMENTO_SEMENTE`.

Não se aplica SMOTE. A proporção positiva não apresenta desbalanceamento
extremo, e amostras sintéticas poderiam misturar relações, grupos e dependências
temporais inexistentes. Os experimentos com pesos de classe e limiar cumprem
essa função sem criar novos eventos.

### Escolha do limiar e do modelo

Cada configuração é ajustada separadamente nos dois treinos. Para suas
probabilidades de validação, são avaliados limiares de 0,05 a 0,95 em passos de
0,05.

O limiar de cada configuração é escolhido por:

1. maior F1 médio nas duas validações;
2. maior MCC médio;
3. maior precisão média;
4. maior proximidade de 0,50.

Depois, a configuração final é escolhida por:

1. maior F1 médio;
2. maior PR-AUC média;
3. maior MCC médio;
4. ordem fixa das configurações, apenas para desempate determinístico.

Além dessas medidas, são gravadas revocação, acurácia, acurácia balanceada,
ROC-AUC, erro de Brier e matriz de confusão.

### Dependência entre eventos

Mais de um destino pode estar associado ao mesmo choque de origem. Por isso,
também são calculadas métricas de sensibilidade nas quais cada linha recebe
peso `1 / quantidade_de_linhas_do_grupo`.

Cada `grupo_choque` passa a ter peso total 1. Essas métricas não escolhem o
modelo; servem para verificar se grupos com muitos destinos dominam o resultado
por linha.

### Saídas da seleção

O comando gera:

- `resultados/treinamento/validacao.sqlite`: configurações, métricas e previsões
  das duas validações;
- `resultados/treinamento/configuracao_selecionada.json`: hash da base,
  atributos, semente, modelo, hiperparâmetros, limiar e resumo da seleção.

As saídas são construídas em arquivos temporários e substituídas somente após a
verificação de integridade do SQLite. O serviço de validação usa `--overwrite`
porque essa etapa pertence ao desenvolvimento e pode ser repetida.

## Avaliação final isolada

Execute somente quando a seleção estiver encerrada:

```bash
docker compose run --build --rm avaliacao-final
```

Antes do ajuste, o comando confere:

- formato do JSON e declaração de que o teste não foi consultado;
- lista exata das quatro views de desenvolvimento;
- modelo e hiperparâmetros pertencentes ao catálogo implementado;
- hash e identificador da execução da base;
- lista e assinatura dos atributos.

Depois dessas verificações, o modelo é ajustado em `treino_final`. A visão
`teste_final` só é lida após o término desse ajuste. O limiar selecionado não é
recalculado.

São gerados:

- `resultados/treinamento/avaliacao_final.sqlite`: métricas, previsões e
  importâncias disponíveis;
- `resultados/treinamento/modelo_final.pkl`: pipeline completo ajustado,
  acompanhado de metadados e hashes.

Por padrão, a avaliação final recusa sobrescrever esses artefatos. O serviço do
Docker Compose não passa `--overwrite`.

## Pendência dos rótulos negativos

A regra atual atribui rótulo 0 quando existe ao menos uma observação futura no
destino e nenhuma resposta atinge o limite. Ela não exige cinco observações
completas. Na execução atual, 908 dos 2.934 negativos possuem menos de cinco
observações futuras.

Esses casos são preservados pelo treinamento e não são corrigidos ou removidos
automaticamente. A exigência de cobertura completa ainda será decidida antes da
execução definitiva ou avaliada por uma análise de sensibilidade específica.

## Estado atual

Os programas estão implementados, mas não foram executados. A base
`resultados/modelagem/base_modelagem.sqlite` ainda precisa ser materializada;
portanto, ainda não existem resultados de treinamento ou avaliação final.
