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

## Caracterização do Efeito Dominó

Depois de gerar `resultados/granger/granger.sqlite`, execute a caracterização
dos eventos:

```bash
docker compose run --build --rm efeito-domino
```

O processo gera `resultados/efeitos/domino/efeito_domino.sqlite` sem alterar os
bancos anteriores.

## Base de modelagem

Com os três bancos derivados disponíveis, gere a base supervisionada e suas
partições temporais:

```bash
docker compose run --build --rm base-modelagem
```

O processo gera `resultados/modelagem/base_modelagem.sqlite`, com os atributos
históricos, as informações de auditoria e as divisões de treino, validação e
teste final.

## Treinamento e validação

Depois de gerar a base de modelagem, compare os modelos nas duas validações
temporais de desenvolvimento:

```bash
docker compose run --build --rm validacao-modelos
```

Essa etapa usa somente o banco deste projeto. Ela não consulta a visão de teste
final e grava as métricas em `resultados/treinamento/validacao.sqlite` e a
configuração escolhida em
`resultados/treinamento/configuracao_selecionada.json`.

A avaliação oficial é um comando separado e deve ser executada apenas quando a
seleção estiver encerrada:

```bash
docker compose run --build --rm avaliacao-final
```

Por segurança, esse serviço não sobrescreve uma avaliação anterior. O teste
final também não faz parte do `pipeline-completo`.

Para executar as duas etapas em sequência, depois de gerar a base de modelagem:

```bash
docker compose run --build --rm pipeline-treinamento-avaliacao
```

Esse pipeline dedicado primeiro refaz a validação e a seleção e, se ela for
concluída, prossegue para a avaliação oficial. Como ele abre o teste final, deve
ser usado somente quando a metodologia e a seleção estiverem encerradas. Antes
de iniciar, o comando recusa a execução caso já exista o banco da avaliação ou
o modelo final, evitando alterar a validação antes de descobrir que o resultado
oficial está protegido.

## Pipeline completo

Para executar sequencialmente pré-processamento, Granger, Efeito Dominó e
construção da base de modelagem, sem iniciar treinamento nem abrir o teste
final:

```bash
docker compose run --build --rm pipeline-completo
```

O comando mostra a etapa atual, a porcentagem geral, as etapas concluídas e
restantes e o tempo decorrido.

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
