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
- [artigos utilizados](docs/artigos/).

## Pré-processamento

O pré-processamento preserva o banco bruto e gera um novo SQLite com uma
cotação consolidada por identidade, preços normalizados e marcadores de
qualidade.

```bash
docker compose run --build --rm preprocessamento
```

Consulte a [documentação do pré-processamento](preprocessamento/preprocessamento.md)
para conhecer as transformações e decisões adotadas.

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
os testes realizados. Consulte a
[documentação da etapa de Granger](granger/granger.md) para acompanhar as
decisões metodológicas e as estruturas geradas.

## Caracterização do Efeito Dominó

Depois de gerar `resultados/granger/granger.sqlite`, execute a caracterização
dos eventos:

```bash
docker compose run --build --rm efeito-domino
```

O processo gera `resultados/efeitos/domino/efeito_domino.sqlite` sem alterar os
bancos anteriores. Consulte a
[documentação do Efeito Dominó](efeitos/domino/efeito_domino.md) para conhecer
a regra de classificação e os resultados atuais.

## Pipeline completo

Para executar sequencialmente pré-processamento, Granger e Efeito Dominó:

```bash
docker compose run --build --rm pipeline-completo
```

O comando mostra a etapa atual, a porcentagem geral, as etapas concluídas e
restantes e o tempo decorrido. Consulte a
[documentação do pipeline](pipeline/pipeline.md) para conhecer o fluxo e o
comportamento em caso de falha.

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
