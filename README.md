# Mineração de Dados Aplicada à Dinâmica de Preços nas CEASAs

Projeto de mineração de dados para analisar a dinâmica de preços agrícolas entre diferentes CEASAs.

Inicialmente, o trabalho investiga o **Efeito Dominó**, buscando identificar se os preços de outras CEASAs ajudam a prever aumentos em um mercado específico.

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

A etapa de Granger será executada por um script Python normal, sem depender de
notebooks. O serviço correspondente já possui uma entrada própria no Docker
Compose:

```bash
docker compose run --build --rm granger
```

O serviço utilizará `data/cotacoes_preprocessadas.sqlite` e gerará
`data/granger.sqlite`, preservando as séries, diagnósticos e todos os testes
realizados. Consulte a
[documentação da etapa de Granger](granger/granger.md) para acompanhar as
decisões metodológicas e as estruturas geradas.

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

O diretório do projeto é montado em `/workspace`. O dataset permanece fora da imagem Docker e pode ser acessado pelo diretório `data/` após a extração.

As dependências Python são gerenciadas pelo `uv`, declaradas no `pyproject.toml` e fixadas no `uv.lock`.

Para adicionar uma dependência:

```bash
docker compose run --rm workspace uv add nome-do-pacote
docker compose build
```
