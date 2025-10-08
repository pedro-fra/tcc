# Estrutura Final do Projeto - TCC Previsao de Vendas

Projeto limpo e organizado para analise de resultados e documentacao da metodologia.

## Scripts Principais (Raiz)

### Pipeline de Execucao
1. **run_preprocessing.py** (8.1KB)
   - Preprocessamento completo dos dados brutos
   - Cria features para modelos de ML e series temporais
   - Gera dados em `data/processed_data/model_data/`

2. **run_arima_forecasting.py** (7.0KB)
   - Executa modelo ARIMA completo
   - Testes de estacionariedade (ADF, KPSS)
   - Gera predicoes e plots

3. **run_exponential_forecasting.py** (7.9KB)
   - Executa Exponential Smoothing
   - Analise de tendencia e sazonalidade
   - Gera predicoes e plots

4. **compare_models_simple.py** (9.2KB)
   - Comparacao final de todos os modelos
   - ARIMA, Theta, Exponential Smoothing, XGBoost
   - Gera MODEL_COMPARISON_RESULTS.md

## Codigo Fonte (src/)

### Modelos (`src/models/`)
- **arima_model.py** - Modelo ARIMA com AutoARIMA (Darts)
- **theta.py** - Theta Method com AutoTheta (Darts)
- **exponential_smoothing_model.py** - Exponential Smoothing (Darts)
- **xgboost_model.py** - XGBoost com 45 features engenheiradas

### Preprocessamento (`src/preprocessing/`)
- **data_preprocessor.py** - Pipeline completo de preprocessamento
- **utils.py** - Funcoes auxiliares (anonimizacao, validacao)
- **model_formatters.py** - Conversao de dados para diferentes formatos
- **stl_decomposition.py** - Decomposicao STL para analise

### Avaliacao (`src/evaluation/`)
- **model_evaluator.py** - Metricas de avaliacao (MAE, RMSE, MAPE)
- **time_series_cv.py** - Validacao cruzada temporal
- **residual_analysis.py** - Analise de residuos (Ljung-Box, normalidade)

### Visualizacao (`src/visualization/`)
- **arima_plots.py** - Plots para ARIMA
- **theta_plots.py** - Plots para Theta
- **exponential_plots.py** - Plots para Exponential Smoothing
- **xgboost_plots.py** - Plots para XGBoost
- **plot_config.py** - Configuracoes de visualizacao
- **plot_translations.py** - Traducoes para graficos

### Configuracao (`src/`)
- **config.py** - Configuracoes centralizadas do projeto

## Dados

### Entrada (`data/`)
- **fat_factory.csv** - Dataset original (37K+ transacoes, 2014-2025)

### Processados (`data/processed_data/`)
- `model_data/time_series/` - Series temporais para modelos Darts
- `model_data/xgboost/` - Features tabulares para XGBoost
- `arima/` - Resultados e modelo ARIMA
- `theta/` - Resultados e modelo Theta
- `exponential_smoothing/` - Resultados e modelo Exponential Smoothing
- `xgboost/` - Resultados e modelo XGBoost

### Visualizacoes (`data/plots/`)
- `arima/` - Graficos do modelo ARIMA
- `theta/` - Graficos do modelo Theta
- `exponential_smoothing/` - Graficos do Exponential Smoothing
- `xgboost/` - Graficos do XGBoost

## Documentacao

### Arquivos Principais
- **README.md** - Descricao geral do projeto
- **CLAUDE.md** - Instrucoes para Claude Code
- **MODEL_COMPARISON_RESULTS.md** - Resultados finais da comparacao

### Dados de Pesquisa (`data/`)
- **TCC1_PedroFra.md** - Documento completo do TCC com revisao de literatura
- **metodologia.md** - Metodologia detalhada de implementacao

## Pasta Optuna (Preservada)

### Otimizacao de Hiperparametros (`optuna_optimization/`)
- Contem implementacoes de otimizacao Bayesiana
- Arquivos markdown preservados conforme solicitado
- Scripts de otimizacao para todos os modelos

## Resumo de Arquivos Removidos

### Scripts Obsoletos Removidos:
- test_xgboost_model.py
- test_theta_model.py
- test_optimized_models.py
- test_improvements.py
- analyze_data_quality.py
- analyze_xgboost_features.py
- compare_all_models.py (duplicado/incompleto)
- run_model_evaluation.py (substituido por compare_models_simple.py)

### Arquivos Markdown Intermediarios Removidos:
- BASELINE_RESULTS.md
- IMPROVEMENTS_COMPARISON.md
- FEATURE_SELECTION_RESULTS.md
- PROJECT_STRUCTURE.md

### Diretorios Temporarios Removidos:
- src/visualization/Scripts/ (ambiente virtual acidental)
- Lib/ (pasta acidental)
- __pycache__/ (todos removidos)
- data/plots/feature_selection/
- data/plots/improvements/
- data/processed_data/arima_comparison/
- data/processed_data/exponential_comparison/

## Metricas de Limpeza

- **Scripts Python removidos:** 8 arquivos
- **Arquivos Markdown removidos:** 4 arquivos
- **Diretorios temporarios removidos:** 10+ pastas
- **Codigo verificado:** Todos os arquivos passaram no Ruff linting
- **Imports nao utilizados:** 0 detectados

## Proximos Passos

Com o projeto limpo, e possivel:
1. Documentar metodologia completa no TCC
2. Analisar resultados finais em MODEL_COMPARISON_RESULTS.md
3. Usar plots gerados para ilustracoes no documento
4. Revisar codigo fonte para documentacao tecnica

## Resultado Final

**Melhor Modelo:** XGBoost
- MAE: R$ 6,246,417.50
- RMSE: R$ 7,885,915.95
- MAPE: 16.32%
- Rank: #1 em todas as metricas
