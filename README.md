# Mineração de Dados Aplicada à Dinâmica de Preços nas CEASAs

Projeto de mineração de dados para analisar a dinâmica de preços agrícolas entre
diferentes CEASAs. O trabalho está organizado em três frentes: **Efeito
Dominó**, **Efeito Foguete e Pena** e **Volatilidade**. A implementação atual
abrange o Efeito Dominó.

## Documentos

A documentação do projeto está organizada em:

- [relatório atual](docs/relatorio/relatorio.pdf);
- [código-fonte do relatório](docs/relatorio/relatorio.tex);
- [bibliografia](docs/relatorio/refs.bib);
- [artigos utilizados](docs/artigos/);
- [experimentos encerrados de Granger](experimentos/granger/README.md);
- [apoio operacional do treinamento](treinamento/README.md).

O relatório é a documentação principal da metodologia, dos critérios de
tratamento e dos resultados do trabalho.

## Pré-processamento

O pré-processamento preserva o banco bruto e gera um novo SQLite com uma
cotação consolidada por identidade, preços normalizados e marcadores de
qualidade.

```bash
docker compose run --build --rm preprocessamento
```

## Séries temporais e teste de Granger

A etapa de Granger é executada por um script Python normal, sem depender de
notebooks. O serviço correspondente já possui uma entrada própria no Docker
Compose:

```bash
docker compose run --build --rm granger
```

O serviço utilizará
`resultados/preprocessamento/cotacoes_preprocessadas.sqlite` e gerará
`resultados/granger/granger.sqlite`, preservando as séries, diagnósticos e todos
os testes realizados.

### Experimentos encerrados

Os estudos de robustez dos lags, sensibilidade ao preenchimento, Granger
condicional e estabilidade temporal ficam isolados em `experimentos/granger/`.
Eles documentam decisões metodológicas e não são executados pelo
`pipeline-completo`.

## Caracterização do Efeito Dominó

Depois de gerar `resultados/granger/granger.sqlite`, execute a caracterização
dos eventos:

```bash
docker compose run --build --rm efeito-domino
```

O processo gera `resultados/efeitos/domino/efeito_domino.sqlite` sem alterar os
bancos anteriores.

## Base de modelagem

Cada cadeia temporal gera uma base própria. Para uma execução individual,
informe o contexto compatível com as datas registradas no banco de Granger:

```bash
MODELAGEM_CONTEXTO=validacao_1 \
  docker compose run --build --rm base-modelagem
```

Os contextos aceitos são `validacao_1`, `validacao_2` e `final`. O gerador
recusa datas incompatíveis e registra uma assinatura da metodologia e das
regras dos atributos.

## Treinamento e validação

Depois de gerar as bases independentes das duas validações, compare os modelos:

```bash
docker compose run --build --rm validacao-modelos
```

Essa etapa exige `resultados/modelagem/validacao_1.sqlite` e
`resultados/modelagem/validacao_2.sqlite`. Ela verifica que as assinaturas
metodológicas coincidem, não constrói nem consulta o teste final e grava as
métricas em `resultados/treinamento/validacao.sqlite` e a configuração em
`resultados/treinamento/configuracao_selecionada.json`. As saídas incluem a
sensibilidade das métricas ao exigir de uma a cinco observações futuras para os
negativos, sem usar essa análise para escolher o modelo.

A avaliação oficial possui um pipeline separado e exige explicitamente o JSON
congelado pela validação:

```bash
PIPELINE_SELECAO_JSON=resultados/versoes/EXECUCAO/treinamento/configuracao_selecionada.json \
  docker compose run --build --rm pipeline-avaliacao-final
```

Esse comando reconstrói apenas o contexto final, confere a assinatura
metodológica e preserva os resultados em `resultados/avaliacoes-finais/`.

## Pipeline completo

Para executar todo o fluxo de desenvolvimento, incluindo as duas cadeias
temporais e a seleção dos modelos:

```bash
docker compose run --build --rm pipeline-completo
```

O comando executa os contextos `validacao_1` e `validacao_2`, constrói suas
bases e seleciona o modelo. Cada rodada fica em `resultados/versoes/`, com
manifesto e hashes. O manifesto registra `inclui_teste_final: false`.

O teste oficial nunca faz parte desse comando. Ele só pode ser consultado pelo
pipeline separado `pipeline-avaliacao-final`.

## Ambiente Python com Docker

Construa a imagem e inicie o JupyterLab:

```bash
docker compose up --build
```

Depois, acesse o endereço exibido pelo JupyterLab no terminal. O serviço fica disponível somente em `127.0.0.1:8888`.

Para executar um script Python sem iniciar o JupyterLab:

```bash
docker compose run --rm workspace python caminho/do/script.py
```

O diretório do projeto é montado em `/workspace`. O banco bruto permanece em
`data/`, enquanto todos os bancos e arquivos derivados ficam em `resultados/`.
Essa pasta é criada automaticamente pelos comandos e permanece fora do Git.

As dependências Python são gerenciadas pelo `uv`, declaradas no `pyproject.toml` e fixadas no `uv.lock`.

Para adicionar uma dependência:

```bash
docker compose run --rm workspace uv add nome-do-pacote
docker compose build
```
