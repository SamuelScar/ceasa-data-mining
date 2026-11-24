# Pipelines temporais

Este arquivo é um apoio operacional. A descrição metodológica principal está
no relatório do projeto.

## Desenvolvimento

O `pipeline-completo` executa somente o ciclo de desenvolvimento:

1. pré-processamento, Granger, Efeito Dominó e base da validação 1;
2. pré-processamento, Granger, Efeito Dominó e base da validação 2;
3. comparação e seleção dos modelos nas duas validações;
4. análise de sensibilidade com cobertura mínima de uma a cinco observações
   futuras para os negativos, sem interferir na seleção.

Os cortes são fixos e passados diretamente aos scripts:

- validação 1: treino até `2023-12-31` e avaliação até `2024-12-31`;
- validação 2: treino até `2024-12-31` e avaliação até `2025-10-31`.

Execute:

```bash
docker compose run --build --rm pipeline-completo
```

Cada rodada recebe um nome `execucao-AAAAMMDD-HHMMSS` e fica em
`resultados/versoes/<nome>/`. O manifesto registra código, parâmetros, hashes,
duração, contextos e `inclui_teste_final: false`. Uma falha preserva os
artefatos parciais para diagnóstico e não inicia as etapas seguintes.

## Avaliação final protegida

O teste oficial não integra o pipeline de desenvolvimento. Depois de encerrar
as decisões metodológicas, informe explicitamente a configuração congelada:

```bash
PIPELINE_SELECAO_JSON=resultados/versoes/EXECUCAO/treinamento/configuracao_selecionada.json \
  docker compose run --build --rm pipeline-avaliacao-final
```

Esse pipeline reconstrói a cadeia final com treino até `2025-10-31`, gera uma
base exclusiva e somente então ajusta o modelo selecionado e consulta o período
de `2025-11-01` a `2026-10-31`.

A avaliação recusa uma base cuja assinatura metodológica seja diferente da
assinatura das duas validações ou uma seleção sem a política de sensibilidade
da cobertura. A mesma análise é repetida no teste apenas depois do ajuste do
modelo congelado. A seleção, os bancos, o modelo e o manifesto ficam em
`resultados/avaliacoes-finais/<nome>/`. Cada execução usa um diretório novo e
não sobrescreve resultados oficiais anteriores.
