# Treinamento dos modelos

Este arquivo é um apoio operacional. A descrição metodológica principal, com
as regras e justificativas, está no `docs/relatorio/relatorio.tex`.

## Escopo

A implementação consome exclusivamente as bases contextuais produzidas pelo
projeto. Não há integração com clima, câmbio, inflação, transporte ou qualquer
banco externo.

O fluxo possui duas etapas que podem ser executadas individualmente:

1. seleção do modelo, dos hiperparâmetros e do limiar nas validações;
2. ajuste em todo o desenvolvimento e avaliação no teste final isolado.

O `pipeline-completo` executa apenas a primeira etapa. A avaliação oficial tem
um pipeline próprio e exige uma seleção previamente congelada.

## Pré-requisito

Antes da seleção, duas bases precisam ser materializadas:

```bash
MODELAGEM_CONTEXTO=validacao_1 docker compose run --build --rm base-modelagem
MODELAGEM_CONTEXTO=validacao_2 docker compose run --build --rm base-modelagem
```

Cada base contém somente o treino e a avaliação de seu contexto. O treinamento
valida contexto, assinatura metodológica, catálogo de atributos, rótulos,
presença das duas classes, identificadores, ordem cronológica e separação dos
eventos.

## Execução completa

Para reconstruir as duas cadeias de desenvolvimento, gerar as bases e executar
a seleção, use:

```bash
docker compose run --build --rm pipeline-completo
```

Esse comando não constrói nem consulta o teste final. O manifesto registra
explicitamente `inclui_teste_final: false`.

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

Cada par de views vem de uma cadeia própria de pré-processamento, Granger e
Efeito Dominó. As views `treino_final` e `teste_final` não são construídas nem
consultadas por esse comando.

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

A escolha continua usando todas as linhas classificáveis. A análise de
sensibilidade da cobertura é calculada somente depois da definição do limiar e
não participa da seleção do modelo.

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
  das duas validações, incluindo a cobertura futura e os cinco cenários da
  análise de sensibilidade;
- `resultados/treinamento/configuracao_selecionada.json`: hashes das duas bases,
  assinatura metodológica, atributos, semente, modelo, hiperparâmetros, limiar
  e resumos da seleção e da sensibilidade de cobertura.

As saídas são construídas em arquivos temporários e substituídas somente após a
verificação de integridade do SQLite. O serviço de validação usa `--overwrite`
porque essa etapa pertence ao desenvolvimento e pode ser repetida.

## Avaliação final isolada

Execute somente quando a seleção estiver encerrada, informando o JSON
congelado:

```bash
PIPELINE_SELECAO_JSON=resultados/versoes/EXECUCAO/treinamento/configuracao_selecionada.json \
  docker compose run --build --rm pipeline-avaliacao-final
```

O pipeline reconstrói a cadeia com treino até `2025-10-31`, gera uma base
exclusiva do contexto final e, antes do ajuste, confere:

- formato do JSON e declaração de que o teste não foi consultado;
- lista exata das quatro views de desenvolvimento;
- modelo e hiperparâmetros pertencentes ao catálogo implementado;
- registro das duas bases usadas no desenvolvimento;
- igualdade da assinatura metodológica entre validações e contexto final;
- lista e assinatura dos atributos;
- política congelada da análise de sensibilidade da cobertura.

Depois dessas verificações, o modelo é ajustado em `treino_final`. A visão
`teste_final` só é lida após o término desse ajuste. O limiar selecionado não é
recalculado.

São preservados em `resultados/avaliacoes-finais/<nome>/`:

- `resultados/treinamento/avaliacao_final.sqlite`: métricas principais,
  sensibilidade da cobertura, previsões e importâncias disponíveis;
- `resultados/treinamento/modelo_final.pkl`: pipeline completo ajustado,
  acompanhado de metadados e hashes.

Cada avaliação usa um diretório novo e não sobrescreve artefatos oficiais.

## Sensibilidade dos rótulos negativos à cobertura

A regra principal atribui rótulo 0 quando existe ao menos uma observação futura
no destino e nenhuma resposta atinge o limite. Ela não exige cinco observações
completas, pois a frequência de publicação varia entre os mercados.

Esses casos são preservados pelo treinamento e não são corrigidos ou removidos
automaticamente. Para medir a dependência do resultado em relação a eles, cada
validação registra cinco cenários: cobertura mínima de 1, 2, 3, 4 ou 5
observações para os negativos. Todos os positivos confirmados são mantidos em
todos os cenários.

O cenário de uma observação reproduz a amostra principal. Os demais são
diagnósticos: não alteram o treinamento, o limiar nem a escolha do modelo. A
avaliação final repete a mesma análise com a política já congelada no JSON de
seleção. `dias_observados_destino` é armazenado para auditoria, mas não integra
os atributos preditores.

## Estado atual

Situação em 24 de novembro de 2026:

- os pipelines de desenvolvimento e avaliação final já possuem um painel fixo
  no terminal, com progresso geral, etapa atual, tempo decorrido e preservação
  dos logs acima do painel;
- o pré-processamento permanece sequencial e recebeu duas otimizações que
  preservam o resultado: uso explícito do índice de proveniência e cálculo
  conjunto das medianas das três colunas de preço;
- a execução sequencial otimizada terminou em 31min49,22s, com 2.387.646
  cotações selecionadas, 2.380.499 identidades consolidadas e 2.295.374
  cotações gravadas;
- o experimento com dois processos terminou em 34min30,05s, ficando 8,4% mais
  lento. Os bancos foram comparados integralmente e apresentaram o mesmo
  conteúdo lógico, por isso o paralelismo foi removido;
- o comparador de bancos e a instrumentação dos tempos de leitura,
  transformação e gravação foram mantidos para as próximas otimizações;
- `resultados/versoes/execucao-20261124-160203` contém somente um manifesto com
  status `em_andamento` e nenhuma etapa concluída. Essa execução foi
  interrompida e não representa uma versão válida do pipeline;
- o fluxo de desenvolvimento ainda precisa ser reexecutado por completo para
  gerar as tabelas de sensibilidade e um novo
  `configuracao_selecionada.json` compatível;
- a avaliação final permanece protegida e o teste oficial não deve ser
  consultado antes do congelamento da nova seleção.

### Próximo ponto de trabalho

O gargalo comprovado está na leitura e, principalmente, na gravação SQLite, não
na transformação em Pandas. O próximo experimento deve comparar, sobre uma
amostra de aproximadamente 50 mil linhas reais, a escrita atual do `to_sql` com
inserções de múltiplas linhas usando `method="multi"`. Os bancos gerados no
microbenchmark devem ser comparados integralmente antes de alterar o pipeline.

Se houver ganho sem qualquer diferença de conteúdo, a otimização pode ser
aplicada ao pré-processamento e validada em uma execução completa. Depois disso,
o pipeline de desenvolvimento deve ser reexecutado desde o início. A avaliação
final só poderá ocorrer após revisar e congelar a nova seleção.

O banco local
`resultados/preprocessamento/cotacoes_preprocessadas_paralelo.sqlite` foi
preservado apenas para auditoria e pode ser removido quando essa comparação não
for mais necessária.
