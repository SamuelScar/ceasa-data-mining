# Pipeline completo

Este arquivo é um apoio operacional. A descrição metodológica principal está
no relatório do projeto.

O pipeline completo executa, no mesmo contêiner e em ordem:

1. pré-processamento das cotações;
2. construção das séries e teste de Granger;
3. caracterização do Efeito Dominó;
4. construção da base de modelagem.

O treinamento e a avaliação final ficam fora deste pipeline. Em especial, o
`pipeline-completo` não consulta o período reservado em `teste_final`. Quando a
seleção estiver encerrada, as duas etapas podem ser encadeadas explicitamente
com `docker compose run --build --rm pipeline-treinamento-avaliacao`. O fluxo
operacional está em [`treinamento/README.md`](../treinamento/README.md), e as
regras metodológicas completas permanecem no relatório principal.

Execute com:

```bash
docker compose run --build --rm pipeline-completo
```

O terminal mostra uma barra de progresso geral, a etapa atual, as etapas já
concluídas, quantas ainda faltam e o tempo decorrido. As mensagens internas de
cada processo continuam visíveis.

Se uma etapa falhar, a execução é interrompida e as seguintes não são
iniciadas. Como cada etapa substitui sua saída de forma atômica, uma falha não
deve sobrescrever o último banco válido daquela etapa.

As saídas são gravadas em:

- `resultados/preprocessamento/cotacoes_preprocessadas.sqlite`;
- `resultados/granger/granger.sqlite`;
- `resultados/efeitos/domino/efeito_domino.sqlite`;
- `resultados/modelagem/base_modelagem.sqlite`.
