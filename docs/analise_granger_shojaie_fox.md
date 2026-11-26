# Análise metodológica de Granger a partir de Shojaie e Fox

## Estado do documento

Este documento registra as possibilidades metodológicas avaliadas e os
experimentos concluídos. As implementações experimentais foram separadas em
`experimentos/granger/`: elas não integram o pipeline principal nem alteram o
fluxo atual. A execução estável permanece como referência de comparação.

Data da análise: 18 de novembro de 2026. Última atualização: 26 de novembro de
2026.

## Referência principal

Shojaie, A.; Fox, E. B. *Granger Causality: A Review and Recent Advances*.
Annual Review of Statistics and Its Application, v. 9, p. 289–319, 2022.
DOI: `10.1146/annurev-statistics-040120-010930`.

PDF consultado: `docs/artigos/annurev-statistics-040120-010930.pdf`.

O artigo funciona como um mapa metodológico. Ele não justifica diretamente
escolhas específicas deste projeto, como o mínimo de 400 dias comuns, o
preenchimento de três dias úteis ou o máximo de dez lags. Esses parâmetros
precisam de avaliação experimental própria.

## Conclusão principal

As três melhorias mais relevantes para o projeto são:

1. recalcular o Granger dentro de cada janela de validação temporal;
2. testar Granger multivariado em rede por produto, utilizando somente os
   mercados existentes no banco;
3. tornar a seleção de lags mais robusta, avaliando a sensibilidade ao limite
   máximo e, se necessário, métodos regularizados.

Não é recomendável começar por Granger neural. Primeiro é necessário garantir
uma validação temporal completa, examinar a estabilidade das relações e
comparar o modelo bivariado com uma rede condicional.

Depois dessas verificações, o Granger assimétrico foi a primeira alternativa a
apresentar relações robustas nas altas nas duas validações. A comparação de
ponta a ponta mostrou, porém, perda de cobertura e ausência de melhora
consistente na classificação. O método assimétrico não substitui o Granger
bivariado.

## Interpretação correta de causalidade de Granger

Uma série de origem causa uma série de destino no sentido de Granger quando o
passado da origem melhora a previsão do destino, mesmo depois de considerado o
passado do próprio destino e das demais variáveis relevantes incluídas no
modelo.

Isso representa precedência temporal e capacidade preditiva incremental. Não
demonstra sozinho um mecanismo econômico direto.

Uma formulação adequada para o relatório é:

> Os retornos do mercado de origem apresentaram capacidade preditiva
> incremental para os retornos do mercado de destino, no sentido de Granger,
> no recorte analisado.

Uma relação observada pode existir porque:

- o mercado de origem participa efetivamente da formação de preços do destino;
- ambos respondem a um terceiro mercado;
- ambos respondem a um choque comum de oferta, logística ou produção;
- a frequência de coleta faz um mercado parecer temporalmente anterior;
- a relação existiu somente em parte do período analisado.

## Hipóteses do Granger clássico e situação do projeto

| Hipótese | Situação atual | Consequência |
| --- | --- | --- |
| Séries contínuas | Atendida pelos preços e retornos | Modelos para contagens ou categorias não são prioritários |
| Relações lineares | Assumidas, mas não verificadas | Respostas assimétricas ou não lineares podem não ser detectadas |
| Tempo regular | Construído por dias úteis e preenchimento | A frequência real de coleta pode produzir relações artificiais |
| Lag conhecido | Selecionado pelo BIC entre 1 e 10 | O teto de 10 continua sendo uma decisão externa |
| Estacionariedade | Testada individualmente nos retornos | Não garante estabilidade das relações ao longo dos anos |
| Observação sem erro | Não plenamente garantida | Ausência, conversões e frequência de coleta introduzem incerteza |
| Sistema completo | Não atendida pelo modelo bivariado | Outros mercados ou choques comuns podem explicar relações aparentes |
| Ausência de efeitos instantâneos | Não verificável com segurança | Dados diários sem horário não identificam direção no mesmo dia |

## Funcionamento atual do projeto

O fluxo atual é:

```text
preços diários
→ retornos logarítmicos
→ ADF e KPSS
→ Granger bivariado
→ escolha do lag pelo BIC
→ correção de Benjamini–Hochberg
→ relações significativas
→ eventos de Efeito Dominó
→ base de modelagem
→ validação do classificador
```

A implementação atual possui decisões metodológicas importantes:

- calendário de segunda a sexta;
- preenchimento limitado a três dias úteis;
- separação em segmentos para não atravessar lacunas maiores;
- cálculo de retornos logarítmicos dentro dos segmentos;
- uso somente do período de treino para o Granger;
- testes ADF e KPSS;
- mesma amostra para comparar os lags de 1 a 10;
- escolha pelo menor BIC;
- preservação dos resultados significativos e não significativos;
- correção global de Benjamini–Hochberg;
- armazenamento de parâmetros, resultados e procedência dos bancos.

Na execução estável `execucao-20261118-201856`, foram registrados:

- 252 pares candidatos;
- 134 pares elegíveis e estacionários;
- 268 direções testadas;
- 80 relações nominalmente significativas;
- 62 relações significativas depois da correção FDR.

## Ponto crítico: validação temporal do fluxo completo

O Granger atual seleciona as relações utilizando todo o período de treino até
`2025-10-31`. A mesma lista é usada para construir eventos de 2022, 2023, 2024
e 2025. Os limites percentuais usados para localizar choques também são
calculados uma única vez com todo o período de treino.

As divisões de desenvolvimento são:

- treinamento da primeira divisão: 2022 e 2023;
- primeira validação: 2024;
- treinamento da segunda divisão: 2022 a 2024;
- segunda validação: janeiro a outubro de 2025.

Embora os transformadores e classificadores sejam reajustados dentro de cada
divisão, a seleção anterior das relações, dos lags e dos limites já consultou
os períodos usados como validação.

Isso não contamina o teste final de novembro de 2025 a outubro de 2026, pois o
Granger não utiliza esse período. Entretanto, as validações internas não são
inteiramente fora da amostra para o pipeline completo.

O fluxo metodologicamente desejado é:

```text
Divisão 1:
2022–2023 → selecionar relações, lags e limites
2024      → validar

Divisão 2:
2022–2024 → selecionar relações, lags e limites
2025      → validar

Avaliação final:
2022–outubro/2025 → selecionar relações, lags e limites
novembro/2025–outubro/2026 → avaliar uma única vez
```

Essa correção deve anteceder a comparação entre alternativas de Granger.

## Granger em rede sem fontes externas

Network Granger não significa rede neural e não exige uma base externa. A rede
pode ser construída somente com os mercados disponíveis para cada produto.

Em vez de testar dois mercados isoladamente, o modelo pergunta se a origem
ainda melhora a previsão do destino depois de considerar também o passado dos
demais mercados disponíveis para aquele produto.

Isso permite distinguir:

- relações diretas condicionais;
- relações aparentemente criadas por um terceiro mercado;
- relações indiretas ou mediadas;
- mudanças de direção depois do condicionamento.

Uma relação que desaparece no modelo condicional não representa piora. Ela é
evidência de que a associação bivariada pode ser explicada por outras séries
observadas.

### Viabilidade com a cobertura atual

Uma análise preliminar da execução estável indicou a seguinte disponibilidade,
respeitando segmentos contínuos e até dez lags:

| Produto | Mercados utilizáveis juntos | Linhas utilizáveis |
| --- | ---: | ---: |
| Batata | 6 | 794 |
| Batata-doce | 6 | 427 |
| Cebola | 6 | 425 |
| Cenoura | 6 | 690 |
| Melão | 6 | 414 |
| Morango | 6 | 863 |
| Quiabo | 6 | 889 |
| Tomate | 6 | 435 |
| Beterraba | 5 | 471 |
| Pimentão | 5 | 438 |
| Banana-prata | 4 | 496 |
| Pimentão-verde | 4 | 496 |

Uma rede por produto é viável, mas não se deve exigir os mesmos mercados em
todos os produtos. A seleção precisa seguir uma regra de cobertura definida
antes de observar a significância.

Uma rede com seis mercados e dez lags possui 61 parâmetros por equação,
incluindo o intercepto. Isso é estimável, mas pode ficar instável nos produtos
com aproximadamente 414 a 435 linhas, especialmente porque as observações
estão distribuídas em segmentos.

O primeiro modelo em rede deve selecionar uma ordem comum pelo BIC e não
assumir dez lags automaticamente.

### Cobertura nas validações temporais

A cobertura foi reavaliada nas duas validações usando `ffill_limite=3`. Exigir
os sete mercados simultaneamente é inviável: nenhum produto possui amostra
utilizável depois de reservar dez lags. Portanto, o primeiro experimento
condicional usa redes específicas por produto.

A seleção é feita somente no treino da validação 1, antes de observar qualquer
significância. A regra escolhe a maior rede com pelo menos seis observações por
coeficiente da equação no lag máximo. Em empate, prevalecem a maior cobertura e
a ordem lexical dos mercados. A rede escolhida em 2022–2023 é congelada para a
validação 2.

| Produto | Quantidade de mercados | Linhas utilizáveis em 2022–2023 |
| --- | ---: | ---: |
| Banana-prata | 3 | 305 |
| Batata | 6 | 369 |
| Batata-doce | 3 | 308 |
| Beterraba | 3 | 313 |
| Cebola | 4 | 311 |
| Cenoura | 5 | 331 |
| Morango | 6 | 446 |
| Pimentão-verde | 3 | 305 |
| Quiabo | 6 | 462 |
| Tomate | 3 | 308 |

Melão e pimentão não formaram uma rede mínima de três mercados com cobertura
suficiente e permanecem fora deste experimento. O limite de seis observações
por coeficiente é uma regra experimental de estabilidade numérica, não uma
recomendação universal do artigo.

### Resultado do condicionamento

O experimento condicional reteve 3 de 11 relações bivariadas comparáveis na
validação 1 e 8 de 27 na validação 2. Quatro relações permaneceram
significativas nos dois períodos, resultando em Jaccard de `0,4`. Nenhuma
relação foi simultaneamente robusta para altas nas duas validações.

O condicionamento reduziu relações potencialmente indiretas, mas não resolveu
a instabilidade temporal nem o ganho fora da amostra. Por isso, a etapa
seguinte mede a frequência de seleção em janelas expansivas antes de considerar
um estimador mais complexo.

### Limites da rede interna

A rede ajuda a controlar outros mercados observados, mas não controla:

- clima não observado;
- produção e safra;
- custos logísticos;
- fornecedores comuns;
- mudanças estruturais;
- erros ou irregularidades de coleta;
- efeitos instantâneos dentro do mesmo dia.

Bases externas continuam opcionais e não são necessárias para o primeiro
experimento multivariado.

## Seleção dos lags

### O que já é automático

O método atual ajusta, para cada direção, modelos de lag 1 até 10, calcula o BIC
na mesma amostra e seleciona o menor valor. A escolha dentro desse intervalo já
é automática.

O limite máximo de dez lags permanece fixado externamente. Uma relação com
efeito relevante no 11º ou 15º dia nunca poderá ser selecionada.

### Distribuição atual

| Lag | Direções selecionadas |
| ---: | ---: |
| 1 | 158 |
| 2 | 57 |
| 3 | 16 |
| 4 | 8 |
| 5 | 17 |
| 6 | 1 |
| 7 | 0 |
| 8 | 0 |
| 9 | 2 |
| 10 | 9 |

Nove direções atingiram exatamente o teto de dez lags. Todas possuem Maringá
como destino e seis são significativas após FDR.

Também aparecem melhorias de BIC concentradas em cinco e dez dias úteis em
algumas relações. Isso pode refletir transmissão de preços, periodicidade
semanal, frequência de atualização ou efeito do preenchimento. Essas
explicações precisam ser separadas experimentalmente.

Em 30 das 268 direções, a diferença entre o melhor e o segundo melhor BIC é
inferior a 2. Nesses casos, a escolha exata do lag é pouco definida.

Em 89 dos 134 pares, as direções opostas escolheram lags diferentes. Isso é
permitido no modelo bivariado, mas exige tratamento explícito em um sistema VAR
multivariado.

O lag é selecionado pelo BIC e o teste F é calculado nos mesmos dados. O p-valor
clássico não representa explicitamente a incerteza gerada pela busca entre
vários lags.

As alternativas metodológicas identificadas foram:

- seleção em uma janela e confirmação em outra;
- ordem comum do sistema definida antes dos testes condicionais.

Bootstrap não será usado nesta etapa por ampliar a complexidade e o custo do
experimento. A confirmação em períodos temporais distintos já foi incorporada
ao protocolo.

## Métodos automáticos apresentados pelo artigo

### Truncating Lasso

Elimina matrizes inteiras de lags. Quando um lag é eliminado, todos os lags
posteriores também são forçados a zero. É apropriado quando a influência deve
diminuir regularmente com o tempo. Essa hipótese não deve ser aplicada antes
de investigar os padrões em cinco e dez dias úteis.

### Adaptive Thresholded Lasso

Permite eliminar um lag e manter outro posterior. Pode ser útil quando existem
efeitos periódicos ou lacunas entre lags relevantes.

### Hierarchical Group Lasso

Agrupa coeficientes por série, direção e lag e permite selecionar relações e
profundidade temporal de maneira estruturada.

A ordem recomendada para avaliação é:

1. BIC comum em um VAR convencional;
2. sensibilidade ao teto máximo de lags;
3. Group Lasso para seleção de relações, se necessário;
4. HLag ou método adaptativo se houver evidência de lags heterogêneos.

## Estacionariedade e estabilidade temporal

ADF e KPSS verificam propriedades das séries individuais. Eles não demonstram
que os coeficientes ou as relações entre mercados permaneçam constantes.

Uma relação pode existir em 2022 e 2023, desaparecer em 2024 e mudar de direção
em 2025, mesmo quando os retornos passam nos testes de estacionariedade.

Na execução atual, todos os 268 diagnósticos passaram. Os p-valores KPSS iguais
a `0,1` representam o limite superior tabulado pelo procedimento, não
necessariamente o valor exato.

Antes de modelos de mudança de regime, devem ser avaliadas:

- janelas expansivas;
- janelas móveis de 18 ou 24 meses;
- frequência de reaparecimento de cada relação;
- estabilidade do sinal dos coeficientes;
- estabilidade do lag;
- sobreposição entre redes de períodos diferentes.

Modelos switching VAR ou VAR por partes só devem ser considerados se essa
análise indicar quebras estruturais relevantes.

## Frequência, preenchimento e calendário

O calendário de segunda a sexta e o preenchimento de até três dias regularizam
as séries, mas não garantem que todos os mercados sejam observados na mesma
frequência.

O preenchimento pode gerar retornos iguais a zero durante ausências e
concentrar a variação na próxima observação real.

Na base de treino, algumas proporções aproximadas de preços preenchidos são:

- banana-prata: 15,5%;
- pimentão-verde: 14,4%;
- tomate: 12,4%;
- batata-doce: 11,8%;
- beterraba: 11,3%.

O estudo de sensibilidade executado comparou:

- nenhum preenchimento;
- preenchimento de um dia;
- preenchimento de três dias, como no método atual;
- segmentos com observações efetivas;
- um calendário operacional inferido exclusivamente das cotações da CEASA;
- agregação semanal como comparação de robustez.

A agregação semanal e o calendário operacional aumentaram a estabilidade das
relações significativas, mas nenhuma alternativa preservou relações robustas
nas duas validações. Portanto, o calendário diário de segunda a sexta com
preenchimento limitado a três dias permanece no método oficial.

Métodos estruturais para frequências mistas exigem hipóteses fortes e não são a
primeira opção para este projeto.

## Não linearidade

O artigo apresenta Granger não linear com MLPs, LSTMs e penalidades de grupo.
Esses modelos representam relações complexas, mas necessitam de mais dados e
possuem interpretação mais difícil.

Os conjuntos atuais possuem aproximadamente 414 a 889 linhas utilizáveis por
produto antes das divisões temporais. Esse volume é limitado para uma rede
neural como método principal.

Antes de Granger neural, é mais apropriado avaliar:

- respostas diferentes para aumentos e quedas;
- Threshold VAR;
- termos de interação;
- regressões robustas a caudas pesadas;
- Granger linear regularizado.

Uma hipótese assimétrica é especialmente pertinente porque o Efeito Dominó
procura predominantemente altas de preços.

### Resultado do Granger assimétrico

O experimento separou o retorno da origem em parcelas positiva e negativa. O
modelo restrito contém o histórico do destino e as defasagens negativas da
origem; o irrestrito acrescenta as defasagens positivas. Assim, o teste F mede
especificamente se as altas da origem fornecem informação adicional depois de
controlar as quedas.

O mesmo protocolo foi executado nos bancos `ffill_limite=3` das duas
validações, sem construir ou consultar o teste final. Foram processadas 84
direções na validação 1 e 182 na validação 2, sem erros de modelo. O experimento
registrou:

| Métrica | Validação 1 | Validação 2 |
| --- | ---: | ---: |
| Significativas após FDR | 9 | 13 |
| Significativas com resposta positiva | 8 | 12 |
| Robustas nas métricas gerais | 0 | 1 |
| Robustas nas altas | 3 | 3 |

Cinco relações foram significativas nos dois períodos. Duas também mantiveram
resposta positiva, resíduos adequados e melhora de RMSE e MAE nas altas:

| Relação | Lag | Ganho RMSE nas altas V1/V2 | Ganho MAE nas altas V1/V2 |
| --- | ---: | ---: | ---: |
| `batata: curitiba -> cascavel` | 1 | 0,13% / 2,64% | 1,32% / 3,74% |
| `batata: londrina -> cascavel` | 1 | 1,38% / 5,23% | 5,26% / 7,39% |

O Jaccard das relações robustas nas altas foi `0,5`. No Granger condicional,
nenhuma relação havia atendido esse critério nas duas validações. Portanto,
houve melhora específica para antecipar altas, com evidência mais forte em
`batata: londrina -> cascavel`. Como a validação 1 não produziu relação robusta
nas métricas gerais, não se deve apresentar o resultado como melhora global de
previsão.

A decisão e sua procedência foram congeladas em
`experimentos/granger/selecao_assimetrica_congelada.json`. Essa lista registra
a evidência observada, mas não pode ser usada para medir novamente os mesmos
períodos como se fosse uma seleção independente.

## Diagnósticos adicionais

O modelo atual já registra Ljung--Box para autocorrelação residual e ganho
preditivo fora da amostra, além de OLS, soma de resíduos, estatística F,
p-valor e BIC. O pipeline principal ainda não registra diagnósticos explícitos
de:

- estabilidade do VAR;
- heterocedasticidade;
- caudas pesadas;
- influência de observações extremas;
- estabilidade dos coeficientes.

O módulo experimental de mudança de regime calculou esses indicadores nas duas
validações e sustentou a escolha da inferência robusta como etapa seguinte.
Quando aplicável, devem ser registrados:

```text
ordem selecionada
BIC, AIC e HQIC
diagnóstico de autocorrelação residual
estabilidade das raízes do VAR
distribuição e variância dos resíduos
erro preditivo fora da amostra
diferença entre modelos restrito e irrestrito
frequência da relação em janelas temporais
```

Como a definição de Granger é preditiva, o ganho fora da amostra deve ser
considerado junto com o teste estatístico.

## Plano experimental proposto

### Etapa 0 — Validação temporal completa

Para cada divisão temporal, recalcular:

1. séries disponíveis para o corte;
2. seleção dos lags;
3. testes de Granger;
4. correção FDR;
5. relações significativas;
6. limites de choque;
7. eventos de Efeito Dominó;
8. base de modelagem;
9. treinamento e avaliação no período posterior.

Essa etapa deve ser implementada antes das demais comparações.

### Etapa 1 — Robustez do Granger bivariado

Comparar máximos de 5, 10, 15 e 20 lags, mantendo a mesma amostra para que os
critérios de informação sejam comparáveis.

Registrar:

- relações que atingem o teto;
- estabilidade das relações;
- diferença entre os dois melhores BICs;
- mudanças de direção;
- ganho preditivo fora da amostra;
- sensibilidade ao preenchimento.

### Etapa 2 — Network Granger condicional

Para cada produto:

1. selecionar mercados por uma regra prévia de cobertura;
2. escolher uma ordem comum do VAR;
3. ajustar o sistema multivariado;
4. testar em bloco os coeficientes da origem na equação do destino;
5. aplicar a correção FDR com família de testes previamente definida;
6. registrar a rede resultante.

As relações devem ser classificadas como:

| Classificação | Interpretação |
| --- | --- |
| Mantida | Significativa no modelo bivariado e no condicional |
| Desapareceu | Significativa somente no bivariado |
| Emergiu | Significativa somente depois do condicionamento |
| Mudou de direção | A direção predominante mudou no modelo condicional |

### Etapa 3 — Estabilidade temporal

O executor repete os modelos em janelas expansivas com 18 meses
iniciais, fechamentos trimestrais e avaliação nos seis meses seguintes. Esse
período permite alcançar o mínimo de dez altas acima do percentil 90. Ele
registra:

- proporção de janelas em que cada relação aparece;
- estabilidade da direção;
- estabilidade do lag;
- sinal e amplitude dos coeficientes;
- sobreposição de Jaccard entre redes.

A direção do efeito é resumida por uma resposta dinâmica unitária acumulada em
cinco dias úteis. Uma relação precisa ser testável em pelo menos três janelas e
atingir frequência mínima de 60% para FDR, resíduos adequados e resposta
positiva. Os bancos intermediários são descartados depois da agregação.

A execução processou três janelas na validação 1 e sete na validação 2 em
`58,593` segundos. Cada contexto selecionou duas relações, com apenas uma em
comum: `morango: maringa -> foz-do-iguacu`. O Jaccard foi `0,3333`, inferior
aos `0,4` do Granger condicional final.

A relação comum apresentou lag 1, resíduos adequados e resposta positiva em
todas as janelas, mas não preservou ganho de RMSE, MAE nem das altas no teste
completo da validação 2. A estabilidade estrutural isolada não foi suficiente
para produzir uma entrada melhor para o Efeito Dominó.

### Etapa 4 — Regularização

Se o VAR comum apresentar instabilidade ou excesso de parâmetros e houver
amostra temporal suficiente para escolher a penalização, avaliar:

- Group Lasso por relação;
- HLag;
- seleção da penalização por validação temporal.

Em modelos penalizados, p-valores clássicos não devem ser a única evidência.
No recorte atual, a pequena amostra efetiva torna a validação temporal da
penalização frágil. Portanto, Group Lasso, HLag e bootstrap permanecem fora do
escopo atual.

### Etapa 5 — Não linearidade

O primeiro experimento desta etapa, o modelo assimétrico para altas e quedas,
foi concluído. Ele superou as alternativas anteriores no critério específico
de robustez nas altas, preservando duas relações nas duas validações. Produzir
mais relações significativas não foi usado como critério suficiente.

A melhora não se sustentou no fluxo completo. O Efeito Dominó perdeu cobertura
e a modelagem assimétrica não melhorou simultaneamente F1, PR-AUC e MCC nas duas
validações. O Granger bivariado permanece como método principal.

O diagnóstico anterior ao Threshold VAR foi concluído e não sustentou essa
extensão. Os VARs permaneceram estáveis, mas houve heterocedasticidade em 35 de
84 direções na validação 1 e em 70 de 182 na validação 2, além de caudas pesadas
em 83 e 182 direções, respectivamente. Houve quebra CUSUM em 9 direções apenas
na segunda validação. A triagem threshold encontrou 5 configurações viáveis,
também somente na segunda validação, e nenhuma combinou `delta BIC >= 6` com
ganho simultâneo de RMSE e MAE nos dois períodos. Além disso, 792 de 798
candidatos usaram limiar zero e, na prática, repetiriam a separação entre altas
e quedas já avaliada. Threshold VAR foi, portanto, descartado.

O experimento seguinte manteve os mesmos coeficientes OLS, lags, amostras e
previsões do Granger bivariado e alterou somente a inferência. O teste principal
usou a covariância HAC de Newey--West calculada separadamente em cada segmento;
HC3 funcionou como análise de sensibilidade. A correção Benjamini--Hochberg foi
refeita separadamente para os dois métodos. Essa alternativa respondeu
diretamente à heterocedasticidade observada sem incluir dados externos, gerar
dados artificiais ou ampliar o modelo.

## Registro de execução e testes

As verificações realizadas nesta evolução foram:

-  24 testes unitários de robustez de lags, sensibilidade ao preenchimento,
  Granger condicional e estabilidade temporal aprovados em `0,337` segundos
  depois da separação em `experimentos/granger/`;
- sensibilidade a `ffill_limite` 0, 1 e 3 concluída nas duas validações;
- Granger condicional concluído em aproximadamente 23 segundos;
- estabilidade temporal concluída em `58,593` segundos, com dez janelas e uma
  etapa de agregação;
- criação confirmada de `estabilidade.sqlite`, `comparacao.json`,
  `configuracao.json` e manifesto, totalizando `368 KB`;
- descarte confirmado dos bancos temporários das janelas;
- remoção dos quatro bancos `ffill_limite` 0 e 1 após a comparação, liberando
  aproximadamente `80,7 MiB` sem remover os bancos escolhidos com limite 3;
- proteção do teste final registrada nos manifestos;
- quatro testes unitários específicos do Granger assimétrico aprovados em
  `0,067` segundo;
- Granger assimétrico concluído em 26 segundos, sem erros de modelo, com duas
  relações robustas nas altas nas duas validações;
- seleção assimétrica congelada com os hashes do relatório e dos dois bancos de
  validação;
- três testes do Efeito Dominó assimétrico aprovados em `0,015` segundo e
  comparação concluída em 30 segundos;
- três testes da comparação de modelagem aprovados em `0,001` segundo;
- comparação completa da modelagem concluída em `172,597` segundos, sem
  consultar o teste final;
- diagnóstico de regimes concluído em aproximadamente 48 segundos, sem erros
  de modelo e sem evidência repetida que justificasse Threshold VAR.
- cinco testes da montagem das séries aprovados em `0,164` segundo;
- comparação diária e semanal concluída em aproximadamente 44 segundos;
- calendário operacional derivado da CEASA concluído em aproximadamente 54
  segundos, sem consultar o teste final.

A bateria registrada cobre os 24 testes anteriores, os quatro testes do modelo
assimétrico, os três testes do Efeito Dominó assimétrico e os três testes da
modelagem. Os três testes do diagnóstico de regimes passaram em `0,022` segundo
e os cinco testes da inferência robusta em `0,017` segundo. Com os cinco testes
de montagem das séries, todos os 47 testes registrados nessa evolução foram
aprovados.

## Critérios para considerar uma melhoria real

Uma versão não é melhor somente por produzir mais relações significativas.

| Dimensão | Métricas ou evidências |
| --- | --- |
| Estrutura | Relações mantidas, removidas, novas e invertidas |
| Estabilidade | Frequência por janela e sobreposição de Jaccard |
| Lags | Distribuição, escolhas no teto e diferença de BIC |
| Inferência | p-valores e q-valores clássicos, HC3 e HAC |
| Previsão Granger | Erro restrito e irrestrito fora da amostra |
| Efeito Dominó | Eventos, cobertura e proporção positiva |
| Classificação | F1, PR-AUC e MCC nas validações temporais |
| Complexidade | Parâmetros, tempo de execução e perda de observações |

A execução de desenvolvimento `execucao-20261125-230051` registrou:

- F1 médio: `0,6115339`;
- PR-AUC média: `0,5193795`;
- MCC médio: `0,2359359`.

Esses números são a referência histórica, mas deverão ser reinterpretados
depois que a seleção de Granger e os limites forem recalculados dentro de cada
divisão temporal.

O teste final não deve ser consultado repetidamente para escolher alterações.
O versionamento garante reprodutibilidade, mas não elimina o viés causado por
usar o resultado final como critério de desenvolvimento.

## Estado da implementação

```text
1. Concluído: recalcular Granger e limites nas divisões temporais
2. Concluído: diagnósticos e sensibilidade do máximo de lags
3. Concluído: sensibilidade ao preenchimento
4. Concluído: Network Granger condicional por produto
5. Concluído: frequência de seleção e direção em janelas expansivas
6. Concluído: Granger assimétrico para altas e quedas
7. Concluído: comparar o Efeito Dominó bivariado e assimétrico nas validações
8. Concluído: comparar os classificadores bivariados e assimétricos nas duas
   validações; decisão de manter o Granger bivariado
9. Concluído: diagnosticar mudança de regime; Threshold VAR descartado
10. Concluído: inferência robusta HC3/HAC nas duas validações; decisão de manter
    a inferência clássica
11. Encerrado sem avanço: nenhuma relação HAC cumpriu também a confirmação HC3
    nos dois períodos
12. Concluído: filtro diário de qualidade e agregação semanal; nenhuma relação
    robusta foi comum às duas validações
13. Concluído: calendário operacional derivado da CEASA; nenhuma relação foi
    robusta nas duas validações
14. Fora do escopo atual: dados externos, bootstrap, regularização e modelos
    neurais
```

Cada etapa deverá produzir uma nova execução versionada e ser comparada com a
execução estável anterior. As alterações devem ser feitas separadamente para
permitir atribuir as diferenças observadas à mudança metodológica correta.

## Resultado ponta a ponta da assimetria

A comparação verificou se a melhora específica do Granger chegava aos dois
consumidores seguintes sem reutilizar, como filtro, a lista de relações comuns
descoberta depois das validações. O protocolo adotado foi:

1. manter o Efeito Dominó atual como referência;
2. em cada divisão, usar as relações assimétricas significativas e com resposta
   positiva que foram aprendidas no treino daquela própria divisão;
3. calcular os limites de choque somente no respectivo treino;
4. manter janela, limiar, dados, atributos e classificador iguais entre as duas
   alternativas;
5. comparar quantidade e cobertura dos eventos, proporção de propagação, F1,
   PR-AUC e MCC nas validações 1 e 2;
6. aceitar a troca somente se o ganho aparecesse nos dois períodos sem redução
   incompatível de cobertura;
7. manter o teste final protegido.

Os indicadores de robustez calculados no período posterior continuam sendo
evidência de avaliação. Eles não devem participar da seleção de relações feita
antes desse período.

A comparação do Efeito Dominó foi concluída em 30 segundos e seus três testes
unitários passaram em `0,015` segundo. Na validação 1, as relações passaram de
11 para 8 e os eventos classificáveis do teste de 198 para 180; a proporção
confirmada caiu de `41,41%` para `40,56%`. Na validação 2, as relações passaram
de 30 para 12, os eventos classificáveis de 480 para 171 e a proporção
confirmada de `39,38%` para `39,18%`.

As duas classes foram preservadas nos treinos e nas avaliações, portanto a
comparação de classificadores continua tecnicamente possível. Isso não
representa melhora: a queda de cobertura, especialmente de `64,4%` nos eventos
classificáveis da validação 2, deve acompanhar a leitura das métricas.

A modelagem foi concluída com o mesmo catálogo de 25 configurações e semente
42. A referência selecionou regressão logística com `C=10`, pesos balanceados e
limiar `0,25`. Essa configuração foi congelada e reaplicada à base assimétrica,
sem permitir que a seleção independente da alternativa influenciasse a decisão.

| Contexto | F1 bivariado/assimétrico | PR-AUC bivariado/assimétrico | MCC bivariado/assimétrico |
| --- | ---: | ---: | ---: |
| Validação 1 | 0,6396 / 0,5890 | 0,5282 / 0,5548 | 0,3092 / 0,2534 |
| Validação 2 | 0,5835 / 0,5882 | 0,5106 / 0,5079 | 0,1627 / 0,2090 |

Na primeira validação, somente PR-AUC melhorou. Na segunda, F1 e MCC melhoraram,
mas PR-AUC piorou. Além disso, a segunda validação preservou apenas 171 das 480
linhas bivariadas. Como a alternativa não superou as três métricas nos dois
períodos, a decisão registrada foi `manter_granger_bivariado`.

## Resultado da inferência robusta HC3/HAC

O módulo `experimentos/granger/executar_inferencia_robusta.py` recalculou a
evidência estatística de cada direção já testada. Ele:

1. reutiliza o lag escolhido, a amostra comum e os coeficientes OLS do Granger
   bivariado;
2. calcula testes Wald conjuntos para os coeficientes defasados da origem;
3. usa HAC de Newey--West segmentada como método principal, sem criar
   autocovariâncias artificiais entre trechos descontínuos;
4. usa HC3 como análise de sensibilidade à heterocedasticidade;
5. refaz a correção FDR global separadamente para HC3 e HAC;
6. preserva os diagnósticos residuais e as comparações preditivas existentes;
7. executa somente as validações 1 e 2, mantendo o teste final protegido.

O experimento não muda os coeficientes, as previsões nem a qualidade dos dados;
ele verifica se as relações continuam significativas quando os erros-padrão
deixam de depender das hipóteses clássicas de variância constante e ausência de
autocorrelação. Uma relação só teria seguido para uma nova comparação com o
Efeito Dominó se fosse HAC robusta nas duas validações e também significativa
por HC3 nos dois períodos.

Os cinco testes passaram em `0,017` segundo e a execução real terminou em 14
segundos, sem erros de modelo. HAC reteve 10 relações na validação 1 e 22 na
validação 2; o método clássico havia retido 11 e 30. Nove relações HAC foram
significativas nos dois períodos e o Jaccard aumentou de `0,2812`, no clássico,
para `0,3913`.

`batata: curitiba -> londrina` foi a única relação robusta nas altas por HAC
nos dois períodos. Na validação 1, porém, seu `q` por HC3 foi `0,05753`; na
validação 2 foi `0,000206`. Como o limite de `0,05` e a confirmação pelos dois
métodos foram definidos antes da execução, o critério não foi relaxado depois
de observar o resultado. Nenhuma relação avançou para uma nova comparação com
o Efeito Dominó e a decisão final foi `manter_inferencia_classica`.

## Resultado das alternativas de montagem das séries

A comparação diária e semanal preservou as fronteiras temporais e selecionou
a configuração somente na validação 1. A série semanal com cobertura mínima de
75% e dois lags aumentou as relações significativas comuns de 9 para 25 e o
Jaccard de `0,2812` para `0,4545`. Apesar disso, as 3 relações robustas da
primeira validação e as 15 da segunda não tiveram interseção. O filtro diário
de qualidade elevou o Jaccard para `0,3913`, também sem robustez comum.

O último experimento construiu um calendário operacional usando somente a base
CEASA: um mercado foi considerado ativo quando possuía qualquer cotação bruta
em um dia útil, e cada par utilizou apenas sessões em que ambos estavam ativos.
A configuração com filtro de qualidade elevou as relações significativas
comuns de 9 para 13 e o Jaccard para `0,3171`, mas não produziu nenhuma relação
robusta em qualquer validação. Na primeira, nenhuma das 17 relações
significativas melhorou RMSE e MAE simultaneamente; na segunda, apenas 2 de 37
obtiveram esse ganho e falharam nos demais critérios.

Com esses resultados, a investigação atual sobre Granger e montagem das séries
foi encerrada. O pipeline oficial usa séries diárias, o método bivariado
clássico, `ffill_limite=3`, 10 lags em amostra fixa, seleção por BIC e FDR
global. HAC, agregação semanal, filtros de qualidade e calendário operacional
permanecem documentados como análises de sensibilidade.
