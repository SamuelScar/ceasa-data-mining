# Séries temporais e teste de Granger

Esta etapa lê o banco gerado pelo pré-processamento, constrói séries temporais
diárias e executa testes bivariados de causalidade de Granger. A execução é
feita por um script Python normal dentro do Docker, sem depender de notebooks.

O notebook existente permanece preservado como registro da exploração inicial,
mas não será ampliado nem será a fonte oficial dos resultados.

## Fluxo

```text
resultados/preprocessamento/cotacoes_preprocessadas.sqlite
    -> mediana diária em R$/kg
    -> séries por produto e entreposto
    -> qualificação dos pares
    -> estacionariedade
    -> teste de Granger e correção FDR
    -> resultados/granger/granger.sqlite
```

## Execução

O corte temporal do pré-processamento deve ser igual ao fim do treino. Ele
limita tanto as conversões inferidas quanto as referências dos outliers:

```bash
CEASA_DATA_CORTE_TREINO=2025-10-31 \
  docker compose run --build --rm preprocessamento
```

Depois que o pré-processamento terminar:

```bash
docker compose run --build --rm granger
```

O arquivo `resultados/granger/granger.sqlite` é construído em um arquivo
temporário e somente substitui uma saída anterior depois da conclusão e da
validação estrutural. O script recusa um banco pré-processado antigo, incompleto,
produzido com outro corte ou sem referências de outliers aprendidas somente no
período de treino.

## Recorte fixado

Entrepostos:

- `recife`;
- `ceasa-grande-vitoria`;
- `curitiba`;
- `maringa`;
- `cascavel`;
- `foz-do-iguacu`;
- `londrina`.

Produtos:

- `batata`;
- `tomate`;
- `cebola`;
- `cenoura`;
- `morango`;
- `beterraba`;
- `melao`;
- `pimentao verde`;
- `pimentao`;
- `batata doce`;
- `banana prata`;
- `quiabo`.

## Construção das séries

- Utiliza preços positivos em R$/kg e desconsidera todos os outliers
  sinalizados pelo pré-processamento.
- Aceita conversões `exata`, `calculada`, `padronizada` e
  `inferida_alta_confianca`.
- Calcula a mediana diária por produto e entreposto.
- Mantém somente dias úteis e registra quantas cotações de cada tipo de
  conversão participaram da mediana.
- Exige 400 dias úteis realmente observados em comum durante o treino para
  qualificar cada par.
- Somente depois da qualificação, preenche falhas de até três dias úteis com o
  último preço disponível.
- Quebra a série em segmentos quando a falha ultrapassa esse limite. Retornos e
  defasagens nunca atravessam segmentos diferentes.
- Calcula retornos logarítmicos dentro de cada segmento.

## Recorte temporal

- início das séries: `2022-01-04`;
- treino e Granger: até `2025-10-31`;
- teste reservado: `2025-11-01` a `2026-10-31`.

O período de teste não participa da seleção de pares, dos diagnósticos de
estacionariedade, da escolha de lags, do ajuste dos coeficientes nem dos testes
de Granger. Ele é usado apenas para medir o erro preditivo fora da amostra. Em
execuções de desenvolvimento, esse período deve pertencer a uma validação
temporal; o teste final protegido só pode ser avaliado depois do congelamento.

## Estacionariedade e Granger

Os testes ADF e KPSS são aplicados aos retornos do maior segmento contínuo de
treino de cada par. As duas séries precisam atender simultaneamente a
`ADF p < 0,05` e `KPSS p > 0,05`. Pares reprovados permanecem registrados, mas
não são testados por Granger.

Para cada direção dos pares aprovados:

- são avaliados lags de 1 a 10 dias úteis;
- todos os lags usam a mesma amostra, definida pelo lag máximo;
- o lag é selecionado pelo menor BIC;
- o p-valor usa a distribuição F exata do teste entre os modelos restrito e
  irrestrito;
- todos os resultados, inclusive não significativos, são preservados;
- os p-valores selecionados recebem correção global de Benjamini--Hochberg;
- a significância principal exige `q < 0,05` após a correção FDR.

O teste indica ganho preditivo temporal e não demonstra causalidade econômica
por si só. Os coeficientes e as defasagens são armazenados para interpretação
posterior.

## Decisão metodológica final

Depois das duas validações temporais, o pipeline oficial permanece com o
Granger bivariado clássico descrito acima. A configuração congelada é:

```text
min_dias_comuns_observados = 400
ffill_limite = 3
max_lag = 10
max_lag_amostra = 10
selecao_lag = BIC em amostra fixa
alpha = 0,05
correcao = Benjamini-Hochberg global
relacoes_consumidas = significativo_fdr
```

Os pipelines de desenvolvimento e avaliação final passam esses argumentos
explicitamente. As variáveis `GRANGER_*` continuam disponíveis somente para o
serviço individual e para experimentos reproduzíveis.

## Experimento de robustez dos lags

O argumento `--max-lag` define quantos modelos serão testados. O argumento
`--max-lag-amostra` define a maior defasagem reservada antes do início da
amostra comum. Quando o segundo argumento não é informado, ele assume o valor
de `--max-lag` e preserva o comportamento estável descrito acima.

Para comparar limites de 5, 10, 15 e 20 sem alterar a amostra entre as
configurações, execute:

```bash
for lag in 5 10 15 20; do
  docker compose run --rm --no-deps -T granger \
    python granger/executar_granger.py \
    resultados/preprocessamento/cotacoes_preprocessadas.sqlite \
    --output-db "resultados/granger/robustez-lag-${lag}.sqlite" \
    --max-lag "$lag" \
    --max-lag-amostra 20 \
    --overwrite
done
```

Os quatro bancos devem usar o mesmo banco pré-processado e os mesmos parâmetros
fora os limites avaliados. A comparação valida essas condições, confirma que os
BICs dos lags compartilhados são idênticos e registra as mudanças de lag e de
significância após FDR:

```bash
docker compose run --rm --no-deps -T granger \
  python -m experimentos.granger.comparar_robustez_lags \
  resultados/granger/robustez-lag-5.sqlite \
  resultados/granger/robustez-lag-10.sqlite \
  resultados/granger/robustez-lag-15.sqlite \
  resultados/granger/robustez-lag-20.sqlite \
  --output-json resultados/granger/robustez-lags.json
```

Esses arquivos são experimentais. O banco estável
`resultados/granger/granger.sqlite` não é substituído durante a comparação.

O experimento concluído com amostra comum de 20 lags mostrou que os limites 10,
15 e 20 selecionaram as mesmas relações e defasagens. O limite 5 perdeu três
relações após FDR e deixou 25 direções no teto. Por isso, o método permanece
com `max_lag=10`; o valor 20 foi usado somente para tornar essa comparação
justa.

## Diagnósticos e previsão fora da amostra

Depois da seleção do lag, o modelo registra duas verificações adicionais:

- o teste de Ljung--Box é aplicado aos resíduos do maior segmento contínuo de
  treino, com ajuste de graus de liberdade pela ordem autorregressiva;
- os modelos restrito e irrestrito fazem previsões de um passo à frente no
  período de teste, sem reajustar os coeficientes aprendidos no treino.

A avaliação preditiva exige pelo menos 30 observações de teste e registra RMSE,
MAE e o ganho percentual do modelo irrestrito. Uma relação é classificada como
robusta somente quando atende simultaneamente aos quatro critérios:

1. significância após a correção FDR;
2. ausência de autocorrelação residual no modelo irrestrito, com
   `Ljung-Box p >= alpha`;
3. RMSE irrestrito menor fora da amostra;
4. MAE irrestrito menor fora da amostra.

Essa classificação é experimental. O Efeito Dominó continua consumindo a view
`relacoes_granger_significativas` até que os resultados sejam avaliados nas
validações temporais e o novo critério seja aprovado.

Além das métricas gerais, a avaliação registra os erros somente nos dias em que
o retorno do mercado de destino supera o maior valor entre zero e o percentil
90 aprendido no treino. Esse recorte exige pelo menos dez ocorrências no teste.
As métricas de altas são diagnósticas e não alteram a significância FDR nem a
classificação de relação robusta.

## Sensibilidade ao preenchimento

O experimento de preenchimento compara `ffill_limite` 0, 1 e 3 separadamente
nas validações 1 e 2. Dentro de cada validação, as três execuções usam o mesmo
banco pré-processado e alteram somente o limite de preenchimento. O comparador
confere o hash da entrada, os parâmetros e as direções registradas antes de
calcular a estabilidade das relações.

O executor gera um pré-processado por validação, executa os três limites e
descarta esse banco temporário somente depois das três execuções concluírem. Se
alguma etapa falhar, os artefatos parciais são preservados para diagnóstico.
O período de teste final não é construído nem consultado.

```bash
docker compose run --build --rm --no-deps -T granger \
  python -m experimentos.granger.executar_sensibilidade_ffill
```

Depois da comparação, são preservados somente os dois bancos com
`ffill_limite=3`, o manifesto e `comparacao.json` em um diretório novo dentro
de `resultados/experimentos/sensibilidade-ffill/`. Os quatro bancos dos limites
0 e 1 têm tamanho e hash registrados no manifesto antes de serem descartados.

O experimento concluído em `execucao-20261126-005710` manteve
`ffill_limite=3`. Embora o limite 1 tenha apresentado o maior Jaccard entre as
validações, o limite 3 preservou mais relações estáveis com ganho específico
nas altas. Nenhum limite produziu uma relação robusta pelo critério geral nas
duas validações.

## Granger multivariado condicional

O experimento seguinte verifica se uma origem continua acrescentando
informação sobre o destino depois de controlar os demais mercados incluídos na
rede do produto. Ele não substitui o Granger bivariado durante a avaliação.

Os sete mercados não possuem cobertura conjunta suficiente. Por isso, as redes
são selecionadas somente pela cobertura de 2022–2023, sem consultar p-valores:

1. escolhe-se a maior rede com pelo menos três mercados;
2. exige-se um mínimo de seis observações por coeficiente da equação no lag
   máximo;
3. empates são resolvidos pela maior cobertura e pela ordem lexical;
4. a rede selecionada é congelada para a validação 2;
5. uma única ordem do VAR é escolhida pelo BIC para todas as direções da rede;
6. cada direção é testada removendo somente os lags da origem e mantendo os
   lags do destino e dos mercados de controle;
7. os p-valores recebem correção global de Benjamini--Hochberg.

Como a rede condicional possui menos direções que o universo bivariado, o
comparador também recalcula o FDR bivariado somente sobre a mesma família de
relações testáveis. Assim, relações retidas, removidas ou novas não são
atribuídas apenas à diferença no número de testes.

O executor usa os bancos `ffill-3.sqlite` já preservados e não refaz o
pré-processamento:

```bash
docker compose run --build --rm --no-deps -T granger \
  python -m experimentos.granger.executar_condicional \
  resultados/experimentos/sensibilidade-ffill/execucao-20261126-005710
```

Os bancos condicionais, a configuração congelada das redes, o manifesto e a
comparação em JSON são gravados em `granger-condicional/` dentro da execução de
sensibilidade. O teste final permanece protegido.

O experimento concluído reteve 3 de 11 relações bivariadas comparáveis na
validação 1 e 8 de 27 na validação 2. Quatro relações foram significativas nos
dois períodos, com Jaccard de `0,4`, mas nenhuma apresentou robustez nas altas
em ambas as validações. Portanto, o resultado condicional permanece
experimental e não substitui a entrada atual do Efeito Dominó.

## Estabilidade temporal do Granger condicional

O experimento repete o modelo em janelas expansivas com pelo menos 18
meses de treino. As janelas fecham a cada trimestre e usam os seis meses
seguintes somente para medir o ganho preditivo. Redes, FDR e critérios de
seleção permanecem iguais nas duas validações.

Esse horizonte fornece observações suficientes para tentar atingir o mínimo de
dez altas acima do percentil 90. As avaliações podem se sobrepor, mas sempre
ocorrem depois do corte de treino da respectiva janela.

Uma relação é selecionada pela estabilidade quando:

1. foi testada em pelo menos três janelas;
2. passou no FDR em pelo menos 60% delas;
3. apresentou resíduos adequados em pelo menos 60%;
4. apresentou resposta dinâmica positiva em pelo menos 60%.

A resposta dinâmica aplica um impulso unitário positivo na origem e acumula a
resposta prevista do destino nos cinco dias úteis seguintes. Ela identifica a
direção estimada do efeito no VAR, mas não transforma o teste de Granger em
causalidade econômica ou estrutural.

```bash
docker compose run --build --rm --no-deps -T granger \
  python -m experimentos.granger.executar_estabilidade_temporal \
  resultados/experimentos/sensibilidade-ffill/execucao-20261126-005710
```

Os bancos das janelas são descartados depois da agregação. A saída preservada
em `granger-condicional/estabilidade-temporal/` contém somente
`estabilidade.sqlite`, `comparacao.json`, `configuracao.json` e o manifesto.
O teste final permanece protegido.

A execução concluída processou três janelas na validação 1 e sete na validação
2 em `58,593` segundos. Cada validação selecionou duas relações, mas somente
`morango: maringa -> foz-do-iguacu` apareceu nas duas. O Jaccard das relações
selecionadas foi `0,3333`, abaixo dos `0,4` do Granger condicional sem o filtro.

A relação comum manteve lag 1, resíduos adequados e resposta positiva em todas
as janelas, mas não preservou ganho de RMSE, MAE ou das altas no teste completo
da validação 2. O filtro de estabilidade não substitui o método principal nem
deve ser consumido pelo Efeito Dominó.

## Registro das validações

- 24 testes unitários dos experimentos de lags, preenchimento, Granger
  condicional e estabilidade temporal foram aprovados em `0,337` segundos
  depois da separação em `experimentos/granger/`;
- o Granger condicional completo concluiu em aproximadamente 23 segundos;
- a estabilidade temporal concluiu as 11 etapas em `58,593` segundos;
- a execução ponta a ponta da estabilidade confirmou a criação do SQLite e do
  JSON compactos e o descarte dos bancos temporários;
- os quatro bancos dos limites 0 e 1 foram removidos da execução de
  sensibilidade, liberando aproximadamente `80,7 MiB`;
- os manifestos declaram `inclui_teste_final=false`.

Também foram concluídos os experimentos de assimetria, mudança de regime e
inferência robusta. A assimetria preservou duas relações robustas nas altas,
mas perdeu cobertura no Efeito Dominó e não melhorou simultaneamente F1,
PR-AUC e MCC nas duas validações. O diagnóstico de regimes não encontrou
evidência repetida que justificasse Threshold VAR.

Na inferência robusta, HAC reteve 10 relações na validação 1 e 22 na validação
2, contra 11 e 30 do método clássico. Nove relações HAC foram significativas
nos dois períodos, elevando o Jaccard de `0,2812` no método clássico para
`0,3913`. Houve uma relação robusta nas altas por HAC nos dois períodos,
`batata: curitiba -> londrina`, mas ela não foi confirmada por HC3 na primeira
validação (`q=0,05753`). Como o critério foi definido antes da execução, o
limite não foi alterado depois do resultado. A recomendação registrada foi
`manter_inferencia_classica`.

Os cinco testes da inferência robusta passaram em `0,017` segundo e os três
testes do diagnóstico de regimes em `0,022` segundo. Considerando as baterias
anteriores documentadas e os cinco testes de montagem das séries, foram
aprovados 47 testes relacionados à evolução do Granger e aos consumidores
experimentais.

As alternativas de montagem das séries também foram encerradas. A agregação
semanal elevou o Jaccard das relações significativas de `0,2812` para `0,4545`,
mas não preservou nenhuma relação robusta nas duas validações. O calendário
operacional derivado exclusivamente da base CEASA elevou o Jaccard para
`0,3171`, também sem relações robustas em comum. Nenhuma alternativa substitui
as séries diárias oficiais.

## Banco gerado

As principais estruturas de `resultados/granger/granger.sqlite` são:

- `execucoes_granger`: parâmetros, hash e origem da execução;
- `mercados_recorte` e `produtos_recorte`: universo fixado;
- `series_temporais`: preços observados e preenchidos, retornos e segmentos;
- `pares_candidatos`: cobertura e motivo de aceitação ou descarte;
- `series_pares`: séries alinhadas dos pares elegíveis;
- `diagnosticos_estacionariedade`: resultados ADF e KPSS;
- `resultados_granger`: uma linha por produto, par e direção;
- `resultados_lags`: estatísticas de todos os lags avaliados;
- `coeficientes_granger`: coeficientes do lag escolhido;
- `diagnosticos_residuos`: Ljung--Box dos modelos restrito e irrestrito;
- `avaliacoes_preditivas`: RMSE, MAE, métricas das altas, ganhos e classificação
  de robustez;
- `resumo_granger`: contagens finais da execução;
- `relacoes_granger_significativas`: view com relações aprovadas após FDR.
- `relacoes_granger_robustas`: view experimental com relações que também
  passam pelos resíduos e pela previsão fora da amostra.

Os parâmetros podem ser alterados por argumentos ou pelas variáveis descritas
em `.env.example`. Os valores efetivamente utilizados ficam registrados no
banco de saída.
