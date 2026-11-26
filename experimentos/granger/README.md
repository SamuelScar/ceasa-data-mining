# Experimentos de Granger

Esta pasta preserva estudos metodológicos já executados. Eles não integram o
`pipeline-completo`, não alteram automaticamente o Granger principal e não são
necessários para executar o Efeito Dominó.

Os códigos permanecem versionados somente para reproduzir e auditar os
resultados registrados no relatório:

- `comparar_robustez_lags.py`: comparação dos limites de 5, 10, 15 e 20 lags;
- `executar_sensibilidade_ffill.py` e `comparar_sensibilidade_ffill.py`:
  comparação dos limites de preenchimento 0, 1 e 3;
- `executar_modelo_condicional.py`, `executar_condicional.py` e
  `comparar_granger_condicional.py`: Granger multivariado por produto;
- `executar_estabilidade_temporal.py` e
  `analisar_estabilidade_temporal.py`: frequência das relações em janelas
  expansivas.

Os experimentos foram encerrados porque nenhuma alternativa superou o Granger
bivariado nas duas validações temporais. Seus bancos temporários não são
versionados. Novas execuções só devem ocorrer quando for necessário reproduzir
os resultados ou comparar outra metodologia sob o mesmo protocolo.

Os 24 testes unitários dos experimentos passaram em 0,337 segundos depois da
mudança dos arquivos, validando os novos caminhos e importações.
