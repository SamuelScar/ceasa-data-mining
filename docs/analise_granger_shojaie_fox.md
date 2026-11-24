# Análise metodológica de Granger a partir de Shojaie e Fox

## Estado do documento

Este documento registra possibilidades metodológicas para o projeto. Ele não
representa uma implementação aprovada e não altera o fluxo atual. As propostas
devem ser avaliadas e implementadas gradualmente, preservando a execução
estável como referência de comparação.

Data da análise: 18 de novembro de 2026.

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

As alternativas a avaliar são:

- bootstrap que repita a seleção e o teste;
- seleção em uma janela e confirmação em outra;
- ordem comum do sistema definida antes dos testes condicionais.

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

Isso exige um estudo de sensibilidade com:

- nenhum preenchimento;
- preenchimento de um dia;
- preenchimento de três dias, como no método atual;
- segmentos com observações efetivas;
- calendários específicos, se disponíveis;
- agregação semanal apenas como comparação de robustez.

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

## Diagnósticos ainda ausentes

O modelo atual calcula OLS, soma de resíduos, estatística F, p-valor e BIC. Não
foram identificados diagnósticos explícitos de:

- autocorrelação residual;
- estabilidade do VAR;
- heterocedasticidade;
- caudas pesadas;
- influência de observações extremas;
- estabilidade dos coeficientes;
- ganho preditivo fora da amostra.

Uma futura implementação deve registrar, quando aplicável:

```text
ordem selecionada
BIC, AIC e HQIC
diagnóstico de autocorrelação residual
estabilidade das raízes do VAR
distribuição e variância dos resíduos
erro preditivo fora da amostra
diferença entre modelos restrito e irrestrito
frequência da relação em bootstrap e janelas temporais
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

Repetir os modelos por janelas e registrar:

- proporção de janelas em que cada relação aparece;
- estabilidade da direção;
- estabilidade do lag;
- sinal e amplitude dos coeficientes;
- sobreposição de Jaccard entre redes.

### Etapa 4 — Regularização

Se o VAR comum apresentar instabilidade ou excesso de parâmetros, avaliar:

- Group Lasso por relação;
- HLag;
- seleção da penalização por validação temporal;
- bootstrap ou frequência de seleção.

Em modelos penalizados, p-valores clássicos não devem ser a única evidência.

### Etapa 5 — Não linearidade

Somente depois das etapas anteriores:

- Threshold VAR;
- modelos assimétricos para altas e quedas;
- MLP Granger esparsa como comparação exploratória.

## Critérios para considerar uma melhoria real

Uma versão não é melhor somente por produzir mais relações significativas.

| Dimensão | Métricas ou evidências |
| --- | --- |
| Estrutura | Relações mantidas, removidas, novas e invertidas |
| Estabilidade | Frequência por janela e sobreposição de Jaccard |
| Lags | Distribuição, escolhas no teto e diferença de BIC |
| Inferência | p-valores, q-valores, bootstrap e estabilidade de seleção |
| Previsão Granger | Erro restrito e irrestrito fora da amostra |
| Efeito Dominó | Eventos, cobertura e proporção positiva |
| Classificação | F1, PR-AUC e MCC nas validações temporais |
| Complexidade | Parâmetros, tempo de execução e perda de observações |

A configuração estável atual registrou na validação:

- F1 médio: `0,5829075`;
- PR-AUC média: `0,5808559`;
- MCC médio: `0,2294662`.

Esses números são a referência histórica, mas deverão ser reinterpretados
depois que a seleção de Granger e os limites forem recalculados dentro de cada
divisão temporal.

O teste final não deve ser consultado repetidamente para escolher alterações.
O versionamento garante reprodutibilidade, mas não elimina o viés causado por
usar o resultado final como critério de desenvolvimento.

## Ordem recomendada para implementação futura

```text
1. Recalcular Granger e limites dentro das divisões temporais
2. Adicionar diagnósticos e sensibilidade do máximo de lags
3. Implementar Network Granger condicional por produto
4. Testar estabilidade temporal e sensibilidade ao preenchimento
5. Avaliar Group Lasso ou HLag se o VAR comum ficar instável
6. Avaliar não linearidade somente depois
```

Cada etapa deverá produzir uma nova execução versionada e ser comparada com a
execução estável anterior. As alterações devem ser feitas separadamente para
permitir atribuir as diferenças observadas à mudança metodológica correta.
