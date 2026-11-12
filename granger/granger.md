# Séries temporais e teste de Granger

Esta etapa lê o banco gerado pelo pré-processamento, constrói séries temporais
diárias e executa testes bivariados de causalidade de Granger. A execução é
feita por um script Python normal dentro do Docker, sem depender de notebooks.

O notebook existente permanece preservado como registro da exploração inicial,
mas não será ampliado nem será a fonte oficial dos resultados.

## Fluxo

```text
data/cotacoes_preprocessadas.sqlite
    -> mediana diária em R$/kg
    -> séries por produto e entreposto
    -> qualificação dos pares
    -> estacionariedade
    -> teste de Granger e correção FDR
    -> data/granger.sqlite
```

## Execução

O limite de inferência do pré-processamento deve ser igual ao fim do treino:

```bash
CEASA_DATA_LIMITE_INFERENCIA=2025-10-31 \
  docker compose run --build --rm preprocessamento
```

Depois que o pré-processamento terminar:

```bash
docker compose run --build --rm granger
```

O arquivo `data/granger.sqlite` é construído em um arquivo temporário e somente
substitui uma saída anterior depois da conclusão e da validação estrutural. O
script recusa um banco pré-processado antigo, incompleto ou produzido com um
limite de inferência diferente do corte de treino.

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

O período de teste é armazenado para a futura modelagem, mas não participa da
seleção de pares, dos diagnósticos de estacionariedade, da escolha de lags nem
dos testes de Granger.

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

## Banco gerado

As principais estruturas de `data/granger.sqlite` são:

- `execucoes_granger`: parâmetros, hash e origem da execução;
- `mercados_recorte` e `produtos_recorte`: universo fixado;
- `series_temporais`: preços observados e preenchidos, retornos e segmentos;
- `pares_candidatos`: cobertura e motivo de aceitação ou descarte;
- `series_pares`: séries alinhadas dos pares elegíveis;
- `diagnosticos_estacionariedade`: resultados ADF e KPSS;
- `resultados_granger`: uma linha por produto, par e direção;
- `resultados_lags`: estatísticas de todos os lags avaliados;
- `coeficientes_granger`: coeficientes do lag escolhido;
- `resumo_granger`: contagens finais da execução;
- `relacoes_granger_significativas`: view com relações aprovadas após FDR.

Os parâmetros podem ser alterados por argumentos ou pelas variáveis descritas
em `.env.example`. Os valores efetivamente utilizados ficam registrados no
banco de saída.
