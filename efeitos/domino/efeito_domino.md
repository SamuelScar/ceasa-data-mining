# Caracterização do Efeito Dominó

Esta etapa lê `resultados/granger/granger.sqlite` e procura ocorrências de
propagação de uma alta de preço entre relações significativas no teste de
Granger. Ela não treina um modelo e não altera os bancos produzidos pelas
etapas anteriores.

## Regra inicial

Para cada produto e relação significativa `origem -> destino`:

1. calcula-se, somente no treino, o limite de alta de cada série;
2. procura-se uma alta observada acima desse limite no mercado de origem;
3. exige-se uma cotação observada no destino na mesma data;
4. movimentos fortes simultâneos são registrados, mas não classificados como
   propagação;
5. procura-se uma alta acumulada no destino nos cinco dias úteis seguintes;
6. o evento recebe `1` quando há resposta e `0` quando existe cobertura, mas a
   resposta não ocorre.

O limite de alta é o maior valor entre 5% e o percentil 90 dos retornos
observados daquela série no período de treino. Cotações preenchidas não iniciam
nem confirmam eventos.

## Execução

O pré-processamento e o teste de Granger precisam estar concluídos. Em seguida:

```bash
docker compose run --build --rm efeito-domino
```

A saída padrão é `resultados/efeitos/domino/efeito_domino.sqlite`. A gravação é
atômica: uma saída anterior somente é substituída depois da validação do novo
banco.

## Configuração

Os valores podem ser alterados no `.env`:

```dotenv
DOMINO_ALTA_MINIMA=0.05
DOMINO_PERCENTIL=0.90
DOMINO_JANELA_DIAS=5
```

## Estruturas geradas

- `execucoes_efeito_domino`: origem, parâmetros e estado da execução;
- `limites_altas`: limites calculados por produto e mercado;
- `relacoes_analisadas`: relações significativas herdadas de Granger;
- `eventos_efeito_domino`: choques, respostas e classificação;
- `resumo_efeito_domino`: contagens finais;
- `eventos_classificaveis`: eventos que receberam `0` ou `1`;
- `efeitos_domino_confirmados`: eventos em que a propagação foi observada.

Eventos sem uma cotação observada no destino, sem cobertura durante a janela ou
com movimentos simultâneos permanecem registrados, mas não recebem rótulo.

## Resultado da execução atual

Foram analisadas as 62 relações significativas de Granger. O processo encontrou
6.149 choques na origem, dos quais 4.815 puderam receber classificação. Entre os
classificáveis, 1.881 (39,1%) foram confirmados e 2.934 (60,9%) não apresentaram
resposta no destino dentro da janela definida.

A proporção de confirmações foi semelhante no treino e no teste: 39,0% no
treino (1.515 de 3.881) e 39,2% no teste (366 de 934). Todas as 62 relações
analisadas apresentaram ao menos um evento classificável e uma confirmação.

Dos 1.334 eventos não classificáveis, 841 foram movimentos simultâneos, 470 não
tinham uma base observada no destino e 23 não possuíam cobertura posterior. Das
confirmações, 610 ocorreram no primeiro dia útil e 449 no segundo.

Esses números mostram que a regra encontrou propagações em toda a rede
selecionada por Granger, mas ainda não demonstram que cada ocorrência seja
independente. Existem 433 grupos nos quais mais de um choque foi associado à
mesma combinação de produto, destino e data de resposta; isso representa 647
registros adicionais e precisa ser considerado antes do treinamento. Também é
necessário comparar essas taxas com relações não significativas ou datas
embaralhadas para medir quanto do resultado pode ocorrer por coincidência ou
por choques comuns de mercado.
