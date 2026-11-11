
# Diário de Bordo — Mineração de Dados CEASA (Efeito Dominó)

**Data de realização:** 11 de novembro de 2026

**Projeto:** Mineração de dados aplicada à dinâmica de preços nas CEASAs (UFSJ/DCOMP)

**Objetivo do dia:** Resolver o ambiente de execução local, parametrizar o pré-processamento, estruturar a janela temporal e rodar os testes empíricos de causalidade de Granger para a primeira pergunta de pesquisa.

## Update 1: Configuração do Ambiente e Execução do Pré-Processamento

* **Contexto inicial:** Configuração do ambiente local no Ubuntu (`ceasa-data-mining`), resolução de dependências do Docker (`docker-compose-v2`), e posicionamento da base bruta em `data/cotacoes.sqlite`.

* **Execução da pipeline:** Executado o container de pré-processamento via `docker compose run --rm preprocessamento`.

* **Métricas da execução inicial:**

  * **2.385.884** cotações brutas na origem.

  * **2.378.073** cotações lógicas consolidadas (pela mediana de preços).

  * **2.253.313** registros gravados no banco derivado `data/cotacoes_preprocessadas.sqlite`.

  * **1.828.678** cotações convertidas diretamente para R\$/kg.

  * **709** possíveis *outliers* apenas sinalizados na tabela `outliers_suspeitos` (sem exclusão automática).

  * **124.760** registros excluídos temporariamente por falta de unidade explícita ou quantidade exata.

## Update 2: Diagnóstico da Base e Identificação dos 5 Parâmetros

* **Abertura do JupyterLab:** Inicialização do container `workspace` e exploração analítica via SQL direto na view `cotacoes_convertidas_kg`.

* **Diagnósticos observados:**

  1. **Regras inferidas inativas:** Existência de 49 regras candidatas prontas no catálogo (ex.: pimentão caixa 10 kg, beterraba caixa 20 kg, maxixe caixa 18 kg) que permaneciam inativas apenas pela ausência da variável `CEASA_DATA_LIMITE_INFERENCIA`.

  2. **Concentração dos mercados:** Dos 22 entrepostos da base, apenas 7 entrepostos (distribuídos em PE, ES e PR) possuem histórico multi-anual consistente (+1.000 datas). Os demais 12 mercados (SP, MG, CE, DF, RJ) iniciaram coleta somente em junho/julho de 2026 (1 a 73 datas), devendo ser filtrados da análise temporal.

  3. **Janela de sobreposição diária:** A interseção contínua entre mercados de estados distintos inicia-se em `2022-01-04`, garantindo cerca de 1.100 dias úteis coincidentes.

* **Proposta dos 5 Parâmetros de Decisão:**

  1. *Mercados alvo:* Foco nos 7 entrepostos consolidados (`recife`, `ceasa-grande-vitoria`, `curitiba`, `maringa`, `cascavel`, `foz-do-iguacu`, `londrina`).

  2. *Data limite de treino:* Fixação de `CEASA_DATA_LIMITE_INFERENCIA=2025-10-31` no `.env`, garantindo quase 4 anos de treino (jan/2022 a out/2025) e 1 ano fechado de teste (nov/2025 a nov/2026) sem vazamento de dados (*data leakage*).

  3. *Escopo de pares:* Segmentação entre pares **Inter-estados** (longa distância: PE $$\leftrightarrow$$ PR $$\leftrightarrow$$ ES) e pares **Intra-estado** (curta distância: interior vs. capital no PR).

  4. *Alinhamento das séries:* Preço mediano diário por `(produto, entreposto, data)` com `ffill(limit=3)` em calendário de dias úteis e transformação para retornos logarítmicos ($$\Delta \ln P_t$$).

  5. *Lags do teste:* Busca em grade de 1 a 10 dias úteis, com seleção do lag ótimo orientada pelo critério de informação Bayesiano (BIC).

## Update 3: Aplicação da Data de Corte e Resgate de Dados Excluídos

* **Ajuste operacional:** Criação do arquivo `.env` com `CEASA_DATA_LIMITE_INFERENCIA=2025-10-31` e reexecução atômica do pré-processamento.

* **Ganhos quantitativos:**

  * As cotações gravadas subiram de **2.253.313** para **2.289.830** (resgate direto de **36.517** cotações antes descartadas como `sem_unidade`).

  * As exclusões auditáveis caíram de **124.760** para **88.243**.

  * A base analítica passou a contar com maior densidade amostral para produtos como pimentão, beterraba e morango.

## Update 4: Resultados Empíricos do Teste de Causalidade de Granger

Execução do teste de Granger bivariado nos retornos logarítmicos com seleção de lag por BIC ($$p < 0{,}05$$) no período de treino (`2022-01-04` a `2025-10-31`), resultando em evidências estatísticas claras da transmissão temporal de preços:

### 1. Dinâmica Inter-estados (PE $$\leftrightarrow$$ PR)

* **Cebola:** Transmissão expressiva de `Recife (PE) -> Curitiba (PR)` com $$F = 25{,}032$$ ($$p < 0{,}00001$$, lag 1) e presença de causalidade bilateral reversa de `Curitiba (PR) -> Recife (PE)` com $$F = 8{,}295$$ ($$p = 0{,}00409$$, lag 1).

* **Morango:** Transmissão unidirecional clara partindo dos polos produtores do Sul em direção ao Nordeste:

  * `Maringá (PR) -> Recife (PE)`: $$F = 8{,}631$$ ($$p = 0{,}00002$$, lag 3).

  * `Foz do Iguaçu (PR) -> Recife (PE)`: $$F = 5{,}306$$ ($$p = 0{,}02018$$, lag 1).

  * `Curitiba (PR) -> Recife (PE)`: $$F = 4{,}623$$ ($$p = 0{,}02979$$, lag 1).

* **Cenoura:** Causalidade significativa de `Recife (PE) -> Curitiba (PR)` ($$F = 8{,}469$$, lag 1) e `Curitiba (PR) -> Recife (PE)` ($$F = 7{,}099$$, lag 1).

* **Batata:** Relações estatisticamente relevantes entre `Maringá (PR) -> Recife (PE)` ($$F = 8{,}203$$, lag 1) e `Recife (PE) -> Foz do Iguaçu (PR)` ($$F = 6{,}348$$, lag 2).

* **Melão:** Transmissão oriunda do polo produtor nordestino para os entrepostos paranaenses:

  * `Recife (PE) -> Curitiba (PR)`: $$F = 5{,}806$$ ($$p = 0{,}01528$$, lag 1).

  * `Recife (PE) -> Foz do Iguaçu (PR)`: $$F = 5{,}224$$ ($$p = 0{,}02113$$, lag 1).

### 2. Dinâmica Intra-estado (Paraná — Rede Regional)

* A propagação intra-estadual apresentou magnitudes estatísticas ainda mais elevadas ($$F > 15$$), evidenciando o entreposto central da capital (`Curitiba`) como polo difusor para os polos de consumo do interior:

  * **Tomate:** `Curitiba -> Cascavel` ($$F = 29{,}204$$, lag 2) e `Curitiba -> Foz do Iguaçu` ($$F = 28{,}978$$, lag 4).

  * **Batata:** `Curitiba -> Londrina` ($$F = 22{,}363$$, lag 3) e `Londrina -> Cascavel` ($$F = 19{,}946$$, lag 1).

  * **Morango:** `Curitiba -> Londrina` ($$F = 19{,}402$$, lag 1) e `Curitiba -> Maringá` ($$F = 12{,}640$$, lag 2).

  * **Pimentão:** `Londrina -> Cascavel` ($$F = 18{,}359$$, lag 1).

  * **Batata doce:** `Curitiba -> Cascavel` ($$F = 13{,}515$$, lag 1).

## Próximos Passos Imediatos

2. Atualizar a Tabela 1 do `docs/relatorio/relatorio.tex` com os novos indicadores do pré-processamento.

3. Redigir a Seção 6 do relatório documentando a metodologia das séries, os critérios BIC de lag e a comprovação empírica do Efeito Dominó.

4. Estruturar a modelagem preditiva utilizando os pares e defasagens estatisticamente validados como variáveis preditoras (*features*).