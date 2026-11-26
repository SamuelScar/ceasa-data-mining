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
- igualdade da assinatura metodológica entre validações e contexto final, com
  compatibilidade restrita à migração do Granger schema 2 para 3 quando ela
  acrescenta somente os diagnósticos de altas já definidos;
- lista e assinatura dos atributos;
- política congelada da análise de sensibilidade da cobertura.

Depois dessas verificações, o modelo é ajustado em `treino_final`. A visão
`teste_final` só é lida após o término desse ajuste. O limiar selecionado não é
recalculado.

Essa compatibilidade não aceita alterações em parâmetros, relações, atributos
ou critérios estatísticos. As assinaturas da seleção e da base final continuam
preservadas nos artefatos para auditoria.

São preservados em `resultados/avaliacoes-finais/<nome>/`:

- `resultados/treinamento/avaliacao_final.sqlite`: métricas principais,
  sensibilidade da cobertura, previsões e importâncias disponíveis;
- `resultados/treinamento/modelo_final.pkl`: pipeline completo ajustado,
  acompanhado de metadados e hashes.

Cada avaliação usa um diretório novo e não sobrescreve artefatos oficiais.

Se as quatro primeiras etapas tiverem sido concluídas e somente a avaliação do
modelo falhar, a execução pode ser retomada sem reconstruir os bancos:

```bash
PIPELINE_SELECAO_JSON=resultados/versoes/EXECUCAO/treinamento/configuracao_selecionada.json \
PIPELINE_AVALIACAO_FINAL_VERSAO=avaliacao-final-AAAAMMDD-HHMMSS \
PIPELINE_AVALIACAO_FINAL_RETOMAR=1 \
  docker compose run --build --rm pipeline-avaliacao-final
```

A retomada exige a mesma seleção, verifica os quatro artefatos existentes, não
sobrescreve métricas ou modelo e preserva no manifesto a falha anterior e os
hashes do código usado na recuperação.

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

## Resultado da investigação do treinamento

A execução de desenvolvimento `execucao-20261125-230051` foi usada para testar
alternativas sem consultar novamente o teste final. O baseline congelado obteve
F1 médio de `0,6115`, PR-AUC média de `0,5194` e MCC médio de `0,2359` nas duas
validações temporais.

Foram avaliados:

- ponderação por grupo e cobertura, além da interação produto--origem--destino;
- redução de atributos e regularizações L1, L2 e elastic net;
- exclusão ou ponderação de rótulos próximos ao limite e regressão da margem;
- filtros das relações por resíduos adequados e q-valor;
- horizontes de resposta de 1, 3 e 5 dias úteis com negativos condicionados à
  cobertura completa.

Nenhuma alternativa melhorou simultaneamente PR-AUC e MCC nas duas validações.
O horizonte de cinco dias com cobertura completa aumentou F1 e PR-AUC, mas
removeu 28 negativos da validação 1 e 76 da validação 2 sem alterar os rótulos
restantes. Com isso, o MCC caiu de `0,3092` para `0,1985` na validação 1 e de
`0,1627` para `0,1233` na validação 2. O ganho aparente veio da retirada de
negativos difíceis, não de melhor separação entre as classes.

As decisões mantidas são: passagem das relações significativas após FDR,
janela de resposta de cinco dias úteis, regra atual para negativos e
classificador congelado. Os executores e resultados descartáveis desses
experimentos não integram o pipeline oficial; as conclusões ficam registradas
neste documento.

## Estado atual

Situação em 26 de novembro de 2026:

- os pipelines de desenvolvimento e avaliação final possuem acompanhamento
  interativo e preservam os artefatos de cada execução versionada;
- o pré-processamento permanece sequencial, com otimizações que preservaram o
  conteúdo lógico do banco principal;
- o experimento de pré-processamento paralelo e a execução versionada
  interrompida foram descartados; seus artefatos locais também foram removidos;
- o fluxo contextual foi concluído na execução `execucao-20261125-230051` e
  produziu a configuração congelada usada pelos experimentos;
- a avaliação final `avaliacao-final-20261126-140147` foi concluída com F1 de
  `0,5912`, PR-AUC de `0,5808` e MCC de `0,1958`;
- o teste final permanece protegido e não deve ser consultado novamente durante
  a evolução metodológica.

## Evolução da causalidade de Granger

O foco atual é melhorar as relações e defasagens produzidas pelo Granger para
fornecer sinais mais úteis ao Efeito Dominó e, em seguida, ao treinamento. Uma
configuração não será considerada melhor apenas por encontrar mais relações
significativas.

Cada alteração no Granger deve ser avaliada em três níveis:

1. **Granger:** estabilidade das relações e direções, distribuição dos lags,
   significância após FDR, diagnósticos do modelo e ganho preditivo fora da
   amostra;
2. **Efeito Dominó:** cobertura dos eventos, quantidade de casos
   classificáveis e proporção de confirmações;
3. **Treinamento:** F1, PR-AUC e MCC nas duas validações temporais.

O método-base já implementado utiliza retornos logarítmicos, ADF e KPSS,
Granger linear bivariado, seleção do lag pelo menor BIC entre 1 e 10 e correção
global de Benjamini--Hochberg. As extensões estudadas a partir de Shojaie e Fox
ainda não foram incorporadas ao método principal.

### Ordem das melhorias

1. robustez da seleção de lags e diagnósticos do modelo;
2. sensibilidade ao preenchimento e à construção das séries;
3. Granger multivariado condicional por produto;
4. estabilidade das relações em janelas temporais;
5. regularização, assimetria e não linearidade somente se justificadas pelos
   resultados anteriores.

### Resultado do experimento de lags

Os limites máximos de 5, 10, 15 e 20 foram comparados sobre a mesma amostra,
fixada em 20 lags. O experimento confirmou o mesmo hash de entrada, a mesma
quantidade de observações e BICs idênticos para os modelos compartilhados.

Resultados principais:

- 257 das 268 direções mantiveram o mesmo lag em todos os limites;
- somente três direções mudaram de significância após FDR;
- o limite 5 produziu 53 relações significativas e 25 escolhas no teto;
- os limites 10, 15 e 20 produziram as mesmas 56 relações significativas;
- nenhum modelo disponível até 20 superou o lag 10.

O limite escolhido permanece `max_lag=10`. Nas execuções definitivas,
`max_lag_amostra` também permanece em 10 para não descartar observações sem
necessidade. O valor 20 pertence somente ao experimento de comparabilidade.

### Resultado dos resíduos e da previsão

O Granger passou a registrar, sem alterar a entrada do Efeito Dominó:

- Ljung--Box dos resíduos restritos e irrestritos no maior segmento contínuo
  de treino;
- RMSE e MAE fora da amostra para os dois modelos;
- ganhos percentuais do modelo irrestrito;
- classificação experimental de relação robusta.

Uma relação robusta precisa ser significativa após FDR, não apresentar
autocorrelação residual no modelo irrestrito e melhorar simultaneamente RMSE e
MAE fora da amostra. A avaliação exige pelo menos 30 observações posteriores e
usa previsão de um passo à frente com os coeficientes fixados no treino.

O pipeline de desenvolvimento `execucao-20261125-230051` avaliou o critério sem
consultar o teste final. Entre as relações significativas após FDR:

- na validação 1, 8 de 11 passaram pelos resíduos, 4 melhoraram RMSE, apenas 1
  melhorou MAE e 1 foi classificada como robusta;
- na validação 2, 13 de 30 passaram pelos resíduos, 20 melhoraram RMSE, 4
  melhoraram MAE e nenhuma foi classificada como robusta;
- a única relação robusta da validação 1, `batata: curitiba -> cascavel`, não
  manteve ganho preditivo na validação 2.

Portanto, nenhuma relação foi robusta nos dois períodos. O filtro experimental
não deve substituir a significância FDR no Efeito Dominó. O F1 médio de
`0,6115` obtido pela execução ainda corresponde ao fluxo com todas as relações
significativas, não ao filtro robusto.

### Resultado da sensibilidade ao preenchimento

O experimento compara `ffill_limite` 0, 1 e 3 nas duas
validações, mantendo os demais parâmetros fixos e reutilizando exatamente o
mesmo pré-processado entre os três limites de cada contexto.

Além da estabilidade das relações, a comparação resume Ljung--Box, RMSE, MAE e
o ganho nos dias de alta do destino. As altas são definidas pelo maior valor
entre zero e o percentil 90 dos retornos do destino no treino, sem consultar o
período avaliado. Essas métricas são diagnósticas e não alteram o filtro atual.

O experimento pode ser executado com:

```bash
docker compose run --build --rm --no-deps -T granger \
  python -m experimentos.granger.executar_sensibilidade_ffill
```

Os pré-processados temporários são descartados após as três execuções válidas
de cada contexto. Depois da comparação, permanecem somente os dois bancos com
`ffill_limite=3`, o manifesto e a comparação em JSON. Os bancos dos limites 0
e 1 têm tamanho e hash registrados no manifesto antes de serem descartados.

O experimento `execucao-20261126-005710` confirmou a manutenção de
`ffill_limite=3`. O limite 1 teve Jaccard ligeiramente maior, mas o limite 3
preservou mais relações significativas nas duas validações e mais relações com
resíduos adequados e ganho simultâneo de RMSE e MAE nas altas. Nenhuma
configuração produziu uma relação robusta geral nos dois períodos.

### Resultado do experimento condicional

O experimento comparou o Granger bivariado com redes condicionais por produto.
Como os sete mercados não possuem cobertura conjunta, a primeira
janela de treino seleciona a maior rede numericamente viável sem consultar a
significância. Essa configuração é congelada e reutilizada na validação 2.

Cada rede possui uma ordem comum escolhida pelo BIC. O teste de uma direção
remove somente os lags da origem e mantém o destino e os demais mercados como
controles. A comparação registra relações bivariadas retidas, removidas e novas,
além da estabilidade temporal e dos ganhos nas altas.

O resultado reteve 3 de 11 relações bivariadas comparáveis na validação 1 e 8
de 27 na validação 2. Quatro relações permaneceram significativas nas duas
janelas, com Jaccard de `0,4`, mas nenhuma foi robusta para altas nos dois
períodos. O condicionamento funcionou como filtro de relações indiretas, porém
ainda não produziu uma rede preditivamente estável para substituir o Granger
bivariado no Efeito Dominó.

### Resultado da estabilidade temporal

O experimento reutiliza as redes condicionais e cria janelas expansivas
com pelo menos 18 meses de treino, fechamentos trimestrais e seis meses de
avaliação. Uma relação será selecionada quando for testável em pelo menos três
janelas e atingir frequência mínima de 60% em significância FDR, resíduos
adequados e resposta dinâmica positiva durante cinco dias úteis.

A mesma regra é aplicada às duas validações. Também são registradas a
estabilidade do lag e as frequências de melhora de RMSE e MAE, sem usá-las para
recalibrar a validação 2. Os bancos de cada janela são temporários; apenas um
SQLite compacto, a configuração, o manifesto e a comparação em JSON são
preservados.

Os seis meses de avaliação permitem alcançar o mínimo de dez ocorrências acima
do percentil 90 exigido pelas métricas de altas. As avaliações podem se
sobrepor, mas nunca antecedem o respectivo corte de treino.

O executor usa os bancos existentes e não refaz o pré-processamento:

```bash
docker compose run --build --rm --no-deps -T granger \
  python -m experimentos.granger.executar_estabilidade_temporal \
  resultados/experimentos/sensibilidade-ffill/execucao-20261126-005710
```

As dez janelas foram executadas em `58,593` segundos. A validação 1 selecionou
duas relações e a validação 2 também selecionou duas, mas somente
`morango: maringa -> foz-do-iguacu` apareceu nas duas. O Jaccard caiu de `0,4`
no Granger condicional final para `0,3333` após o filtro temporal.

Embora a relação comum tenha mantido lag 1, significância frequente, resíduos
adequados e resposta positiva, ela não preservou ganho de RMSE, MAE ou das
altas no teste completo da validação 2. Portanto, o filtro temporal não melhora
o método principal e não deve alimentar o Efeito Dominó.

### Validação executada

- 24 testes unitários de robustez de lags, sensibilidade ao preenchimento,
  Granger condicional e estabilidade temporal passaram em `0,337` segundos
  depois da separação em `experimentos/granger/`;
- o experimento condicional concluiu em aproximadamente 23 segundos;
- o experimento de estabilidade temporal concluiu as 11 etapas em `58,593`
  segundos e preservou somente `368 KB` de artefatos;
- os quatro bancos dos limites 0 e 1 foram removidos após a comparação,
  liberando aproximadamente `80,7 MiB`;
- nenhum dos experimentos consultou o período de teste final.

O período final de novembro de 2025 a outubro de 2026 permanece protegido e
não pode orientar essa escolha.
