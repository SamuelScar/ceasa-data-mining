# Pipeline completo

O pipeline completo executa, no mesmo contêiner e em ordem:

1. pré-processamento das cotações;
2. construção das séries e teste de Granger;
3. caracterização do Efeito Dominó.

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
- `resultados/efeitos/domino/efeito_domino.sqlite`.
