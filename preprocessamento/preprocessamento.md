# Pré-processamento das cotações CEASA

O pré-processamento lê `data/cotacoes.sqlite` sem alterá-lo e gera
`data/cotacoes_preprocessadas.sqlite`. A saída reúne as cotações consolidadas,
os preços tratados e os indicadores de qualidade necessários para as etapas
posteriores do trabalho.

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
- Corrige apresentações com `kg` concatenado ao sufixo `PE`, como
  `Cx.20 KgPE`.
- Identifica a origem das cotações, incluindo o complemento PROHORT.
- Marca possíveis outliers sem corrigi-los ou removê-los.

## Decisões adotadas

- O banco bruto permanece como fonte de auditoria e nunca é sobrescrito.
- Cada identidade lógica gera somente uma cotação no banco derivado.
- Conversões podem ser `exata`, `calculada`, `padronizada`,
  `inferida_alta_confianca` ou `incerta`.
- Pesos não são estimados quando a apresentação não fornece evidência
  suficiente.
- Cotações classificadas como `sem_unidade`, `sem_quantidade_exata` ou
  `ambigua` não são gravadas na tabela final. As quantidades excluídas são
  registradas no resumo do processamento.
- Unidades nativas e intervalos corretamente identificados são preservados.
- Produto inválido, preço comum não positivo e dimensões incompletas recebem
  marcadores e ficam fora da view `cotacoes_analiticas`.
- Quando `mínimo <= comum <= máximo` não é respeitado, o preço comum é mantido
  e os limites mínimo e máximo tratados ficam nulos.
- Regras inferidas exigem concordância do peso explícito, suporte mínimo de
  registros e presença em datas suficientes. O preço não é usado para escolher
  o peso.

## Banco gerado

As principais estruturas são:

- `cotacoes_preprocessadas`: cotações consolidadas e tratadas;
- `cotacoes_analiticas`: registros com produto, preço comum e dimensões válidas;
- `catalogo_apresentacoes`: regras e evidências usadas nas apresentações;
- `regras_conversao_inferida`: regras candidatas ou aplicadas;
- `outliers_suspeitos`: valores sinalizados para avaliação posterior;
- `resumo_preprocessamento`: contagens da entrada, saída e exclusões;
- `execucoes_preprocessamento`: parâmetros e estado da execução.

Ao final, o script verifica a integridade do SQLite, a unicidade das cotações,
as contagens de consolidação e exclusão e a consistência dos marcadores.

## Situação atual

O pré-processamento continua em desenvolvimento. Na execução mais recente,
2.380.608 cotações da origem resultaram em 2.372.920 identidades consolidadas.
Após a exclusão de 124.337 registros sem mapeamento suficiente, foram gravadas
2.248.583 cotações no banco derivado. Também foram registradas 69 regras de
conversão candidatas, sem aplicação automática.
