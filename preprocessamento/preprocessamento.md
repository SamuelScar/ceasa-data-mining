# Pré-processamento das cotações CEASA

O pré-processamento lê `data/cotacoes.sqlite` sem alterá-lo e gera
`resultados/preprocessamento/cotacoes_preprocessadas.sqlite`. A saída reúne as
cotações consolidadas, os preços tratados e os indicadores de qualidade
necessários para as etapas posteriores do trabalho.

Este arquivo é um apoio operacional. A descrição metodológica principal está
no relatório do projeto.

## Execução

Na raiz do repositório:

```bash
docker compose run --build --rm preprocessamento
```

O arquivo existente somente é substituído depois que o novo SQLite é concluído
e validado. Se a execução falhar, o banco anterior é preservado.

O `.env` pode definir `CEASA_DATA_LIMITE_INFERENCIA` no formato `AAAA-MM-DD`.
Quando a variável está vazia, o processamento continua normalmente e as regras
inferidas permanecem apenas como candidatas.

## Transformações realizadas

- Seleciona a coleta mais recente de cada cotação lógica.
- Consolida versões simultâneas pela mediana separada dos preços mínimo, comum
  e máximo.
- Normaliza diferenças textuais de produto, preservando a descrição original.
- Mapeia apresentações para massa, volume, contagem ou unidade nativa.
- Calcula preço por kg, unidade-base e item somente quando existe informação
  suficiente.
- Interpreta dúzia como 12 itens e cento como 100 itens, sem inventar peso.
- Preserva intervalos de massa e quantidade sem transformá-los em valor único.
- Reconhece intervalos com hífen antes da unidade, como `Cx.17a18-Kg`, sem
  reduzir o intervalo a um único peso.
- Preserva pesos totais explícitos mesmo quando o texto também contém um
  intervalo de itens, como em bandejas de 750 g com 3 a 4 unidades.
- Reconhece contagens explícitas de unidades e vasos mesmo quando a fonte não
  fornece uma unidade estruturada.
- Corrige apresentações com `kg` concatenado ao sufixo `PE`, como
  `Cx.20 KgPE`.
- Identifica a origem das cotações, incluindo o complemento PROHORT.
- Classifica possíveis outliers sem corrigir nem remover os valores originais.

## Decisões adotadas

- O banco bruto permanece como fonte de auditoria e nunca é sobrescrito.
- Cada identidade lógica gera somente uma cotação no banco derivado.
- Conversões podem ser `exata`, `calculada`, `padronizada`,
  `inferida_alta_confianca` ou `incerta`.
- Pesos não são estimados quando a apresentação não fornece evidência
  suficiente.
- Cotações classificadas como `sem_unidade`, `sem_quantidade_exata` ou
  `ambigua` não são gravadas na tabela final. As quantidades excluídas são
  registradas no resumo e as linhas permanecem auditáveis em
  `cotacoes_excluidas_mapeamento`.
- Unidades nativas e intervalos corretamente identificados são preservados.
- Produto inválido, preço comum não positivo e dimensões incompletas recebem
  marcadores e ficam fora da view `cotacoes_analiticas`.
- Quando `mínimo <= comum <= máximo` não é respeitado, o preço comum é mantido
  e os limites mínimo e máximo tratados ficam nulos.
- Regras inferidas exigem concordância do peso explícito, suporte mínimo de
  registros e presença em datas suficientes. O preço não é usado para escolher
  o peso.
- Outliers com preço comum igual a R$ 0,01, razão inferior a 0,01 ou superior
  a 20 vezes a mediana são classificados como `erro_provavel`. Grupos com pelo
  menos cinco ocorrências suspeitas recebem `inconsistencia_recorrente`; as
  demais recebem `extremo_isolado`. Os limites podem ser alterados pelos
  argumentos do script e ficam registrados nos parâmetros da execução.
- Todos os outliers suspeitos são preservados no banco para auditoria, mas são
  desconsiderados de `cotacoes_analiticas_principais`. Não existe etapa de
  revisão manual obrigatória e nenhum preço é corrigido ou apagado.

## Banco gerado

As principais estruturas são:

- `cotacoes_preprocessadas`: cotações consolidadas e tratadas;
- `cotacoes_analiticas`: registros com produto, preço comum e dimensões válidas;
- `cotacoes_analiticas_principais`: camada analítica sem erros prováveis e sem
  os demais outliers suspeitos;
- `cotacoes_excluidas_mapeamento`: registros que ficaram fora da tabela final,
  com valores originais e motivo da exclusão;
- `catalogo_apresentacoes`: regras e evidências usadas nas apresentações;
- `regras_conversao_inferida`: regras candidatas ou aplicadas;
- `outliers_suspeitos`: valores classificados e preservados para auditoria;
- `outliers_erros_provaveis`: casos desconsiderados da visão principal;
- `outliers_desconsiderados_analise`: todos os valores preservados no banco,
  mas fora da camada analítica principal;
- `resumo_preprocessamento`: contagens da entrada, saída e exclusões;
- `execucoes_preprocessamento`: parâmetros e estado da execução.

Ao final, o script verifica a integridade do SQLite, a unicidade das cotações,
as contagens de consolidação e exclusão e a consistência dos marcadores.

## Situação atual

O pré-processamento da base atual está concluído. Na execução validada mais
recente, 2.388.035 cotações da origem resultaram em 2.380.218 identidades
consolidadas. Foram excluídos da tabela final 85.124 registros sem mapeamento
suficiente: 85.107 sem unidade, 16 sem quantidade exata e 1 ambíguo. Todas
essas linhas permanecem disponíveis em `cotacoes_excluidas_mapeamento` para
auditoria.

O banco derivado contém 2.295.094 cotações, das quais 2.091.042 atendem aos
critérios da camada analítica. A visão `cotacoes_analiticas_principais` possui
2.090.331 registros após desconsiderar os 711 outliers sinalizados. As 316.087
conversões incertas foram preservadas com seus marcadores e somente devem ser
usadas em comparações compatíveis com as unidades disponíveis.

Foram registradas 24 regras de conversão inferida. A execução utilizou
31 de outubro de 2025 como data limite: 13 regras atenderam aos critérios e
foram aplicadas, enquanto 11 permaneceram como candidatas. Essas regras
aplicadas resultaram em 3.250 conversões de alta confiança no banco derivado.
