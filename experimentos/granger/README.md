# Experimentos de Granger

Esta pasta preserva estudos metodológicos já executados. Eles não integram o
`pipeline-completo`, não alteram automaticamente o Granger principal e não são
necessários para executar o Efeito Dominó.

Os códigos permanecem versionados somente para reproduzir e auditar os
resultados registrados no relatório:

- `comparar_robustez_lags.py`: comparação dos limites de 5, 10, 15 e 20 lags;
- `executar_sensibilidade_ffill.py` e `comparar_sensibilidade_ffill.py`:
  comparação dos limites de preenchimento 0, 1 e 3;
- `executar_modelo_condicional.py`, `executar_condicional.py` e
  `comparar_granger_condicional.py`: Granger multivariado por produto;
- `executar_estabilidade_temporal.py` e
  `analisar_estabilidade_temporal.py`: frequência das relações em janelas
  expansivas.

O experimento concluído para separar o efeito das altas e das quedas usa:

- `executar_modelo_assimetrico.py`: ajusta, para cada direção bivariada, um
  modelo restrito com o histórico do destino e as quedas da origem e testa se
  as altas da origem acrescentam informação;
- `executar_assimetria.py`: executa o mesmo método nas validações 1 e 2 sem
  consultar o teste final;
- `comparar_granger_assimetrico.py`: compara significância, sinal dinâmico,
  resíduos e ganho preditivo, inclusive nos períodos de alta do destino;
- `selecao_assimetrica_congelada.json`: preserva as duas relações aprovadas,
  seus critérios, métricas e hashes para auditoria da decisão.

O experimento seguinte verifica se essa melhora chega ao Efeito Dominó:

- `executar_domino_assimetrico.py`: combina as séries do banco bivariado com as
  relações assimétricas significativas e positivas do respectivo treino;
- `comparar_domino_assimetrico.py`: compara eventos, cobertura, classes e
  proporção de propagação com a referência bivariada;
- `executar_experimento_domino_assimetrico.py`: orquestra as quatro execuções e
  a comparação sem consultar o teste final.

Os três testes dessa etapa passaram em `0,015` segundo. A execução real foi
concluída em 30 segundos e preservou as duas classes nas duas validações, mas
não aumentou a proporção de efeitos confirmados. No teste da validação 2, os
eventos classificáveis caíram de 480 para 171.

A comparação seguinte chega até os classificadores:

- `gerar_base_modelagem_experimental.py`: gera as quatro bases supervisionadas
  usando as séries já auditadas no Granger e o manifesto dos pré-processamentos
  contextuais descartados;
- `comparar_modelagem_assimetrica.py`: compara F1, PR-AUC e MCC, além das
  amostras e métricas auxiliares, congelando o modelo, os hiperparâmetros e o
  limiar selecionados pela referência bivariada;
- `executar_experimento_modelagem_assimetrica.py`: constrói as bases, executa o
  mesmo catálogo de 25 configurações nas duas alternativas e grava a decisão;
- `tests/test_modelagem_assimetrica.py`: protege a normalização da proveniência,
  o critério de melhora e a composição das sete etapas.

As 25 configurações são executadas nas duas alternativas para preservar as
probabilidades auditáveis, mas a seleção assimétrica não participa da decisão:
a comparação usa exatamente a configuração e o limiar escolhidos pela
referência bivariada. Os três testes dessa etapa passaram em `0,001` segundo e
a execução real terminou em `172,597` segundos. A alternativa não melhorou F1,
PR-AUC e MCC simultaneamente nas duas validações e preservou somente 171 das 480
linhas na segunda. A decisão registrada foi `manter_granger_bivariado`.

O diagnóstico usado como barreira anterior ao Threshold VAR contém:

- `diagnosticar_regimes.py`: mede estabilidade do VAR, Jarque--Bera, ARCH LM,
  CUSUM e influência de observações; também compara, apenas como triagem,
  modelos de dois regimes definidos no treino;
- `tests/test_diagnostico_regimes.py`: protege a construção da triagem, a
  estabilidade da matriz companheira e a exigência de evidência nas duas
  validações.

O diagnóstico não produz novas relações causais e não integra o pipeline
principal. A execução foi concluída em aproximadamente 48 segundos. Todos os
VARs permaneceram estáveis, mas foram encontradas heterocedasticidade em 35 de
84 e 70 de 182 direções, e caudas pesadas em 83 e 182 direções. Nenhuma
configuração threshold apresentou evidência repetida nas duas validações com
`delta BIC >= 6` e ganho simultâneo de RMSE e MAE. Threshold VAR foi descartado.
Os três testes específicos desse diagnóstico passaram em `0,022` segundo.

```bash
docker compose run --build --rm --no-deps granger \
  python -m experimentos.granger.diagnosticar_regimes \
  resultados/experimentos/sensibilidade-ffill/execucao-20261126-005710
```

O experimento seguinte testa inferência robusta sem mudar coeficientes, lags,
amostras ou previsões:

- `executar_inferencia_robusta.py`: calcula testes Wald com HAC de Newey--West
  segmentada e HC3 para as direções bivariadas existentes;
- `comparar_inferencia_robusta.py`: compara FDR, estabilidade e robustez nas
  duas validações;
- `executar_experimento_inferencia_robusta.py`: orquestra as duas execuções e a
  comparação, sem consultar o teste final;
- `tests/test_inferencia_robusta.py`: protege as covariâncias, a detecção
  sintética, o critério de avanço e o isolamento do teste final.

HAC foi o método principal e HC3 a análise de sensibilidade. Cada método recebeu
sua própria correção Benjamini--Hochberg global. Os cinco testes passaram em
`0,017` segundo e a execução real terminou em 14 segundos, sem erros.

```bash
docker compose run --build --rm --no-deps granger \
  python -m experimentos.granger.executar_experimento_inferencia_robusta \
  resultados/experimentos/sensibilidade-ffill/execucao-20261126-005710
```

HAC reteve 10 e 22 relações nas validações, contra 11 e 30 do método clássico.
O Jaccard das relações significativas aumentou de `0,2812` para `0,3913`, mas a
única relação robusta nas altas por HAC nos dois períodos não foi confirmada
por HC3 na primeira validação (`q=0,05753`). Nenhuma relação cumpriu o critério
de avanço definido antes da execução. A recomendação final foi
`manter_inferencia_classica`.

Os experimentos anteriores foram encerrados porque nenhuma alternativa superou
o Granger bivariado nas duas validações temporais. O modelo assimétrico foi a
primeira alternativa a preservar relações robustas nas altas nos dois períodos:
`batata: curitiba -> cascavel` e `batata: londrina -> cascavel`, ambas com lag
de um dia útil. O arquivo congelado registra a evidência obtida, mas não deve
ser reutilizado como filtro para medir novamente as mesmas validações. Ele
ainda não substitui o Granger principal nem autoriza consultar o teste final.

O experimento de qualidade das séries temporais contém:

- `executar_experimento_series_temporais.py`: reutiliza os bancos diários
  `ffill-3.sqlite`, sem repetir o pré-processamento, e compara o método atual
  com um filtro diário de qualidade e com séries semanais;
- `tests/test_series_temporais_granger.py`: protege o corte das semanas nas
  fronteiras temporais, a mediana semanal, o filtro diário, o recálculo do FDR
  e a seleção feita somente na primeira validação.

O filtro diário preserva o mínimo original de 400 dias, exige cobertura de
75% e ao menos 15% de retornos não zero em cada mercado. O FDR é recalculado
somente entre as direções que passam por esses critérios. A alternativa semanal
usa apenas preços observados, mediana semanal, nenhum preenchimento, coberturas
de 75% e 85% e limites de dois e quatro lags. Semanas que cruzam uma fronteira
entre treino e teste são descartadas.

```bash
docker compose run --rm --no-deps -T granger \
  python -m experimentos.granger.executar_experimento_series_temporais \
  resultados/experimentos/sensibilidade-ffill/execucao-20261126-005710
```

Na primeira execução, os quatro testes disponíveis passaram em `0,130`
segundo. A configuração semanal escolhida somente na validação 1 foi cobertura
de 75% e dois lags. Ela
aumentou as relações significativas comuns de 9 para 25 e o Jaccard de `0,2812`
para `0,4545`. Também produziu 3 e 15 relações robustas separadamente nas duas
validações, mas nenhuma relação foi robusta em ambas. O filtro diário elevou o
Jaccard para `0,3913`, sem criar robustez comum. Por isso, as alternativas ficam
registradas como diagnóstico e não substituem o Granger diário oficial.

A última alternativa de montagem das séries usa o calendário operacional da
própria base CEASA:

- `executar_experimento_calendario_operacional.py`: considera um mercado ativo
  quando existe qualquer cotação bruta nele em um dia de segunda a sexta;
- cada par utiliza somente as sessões em que os dois mercados estavam ativos;
- o preenchimento de até três ausências é contado em sessões do próprio mercado,
  sem inserir feriados ou fechamentos na linha temporal;
- a seleção compara, somente na validação 1, o calendário isolado e sua combinação
  com cobertura mínima de 75% e taxa mínima de 15% de retornos não zero.

```bash
docker compose run --rm --no-deps -T granger \
  python -m experimentos.granger.executar_experimento_calendario_operacional \
  resultados/experimentos/sensibilidade-ffill/execucao-20261126-005710
```

Os cinco testes de séries temporais passaram em `0,164` segundo. A configuração
com filtro de qualidade foi escolhida na primeira validação e confirmada na
segunda. As relações significativas comuns aumentaram de 9 para 13 e o Jaccard
de `0,2812` para `0,3171`, mas nenhuma relação foi robusta em qualquer uma das
duas validações. Na primeira, as 17 relações significativas não produziram
nenhum ganho simultâneo de RMSE e MAE; na segunda, somente 2 das 37 produziram
esse ganho e falharam nos demais critérios. O calendário operacional também não
substitui o método diário oficial e encerra os experimentos de montagem das
séries temporais.

Os 24 testes unitários anteriores passaram em 0,337 segundos depois da mudança
dos arquivos. Os quatro testes específicos do modelo assimétrico passaram em
0,067 segundo, os três testes do Efeito Dominó assimétrico em `0,015` segundo e
os três testes da modelagem em `0,001` segundo. A execução assimétrica concluiu
sem erros de modelo em 26 segundos, com duas relações robustas nas altas em
comum e Jaccard de `0,5`, mas o ganho não chegou ao fluxo completo. Somados aos
três testes do diagnóstico de regimes, aos cinco da inferência robusta e aos
cinco da montagem das séries, foram aprovados 47 testes registrados nessa
evolução.
