# Relatório Final: Implementação de Otimização com Optuna

## 📋 Resumo Executivo

Implementação completa e funcional de otimização de hiperparâmetros usando Optuna para todos os 4 modelos de previsão de vendas (ARIMA, Theta, Exponential Smoothing, XGBoost).

## ✅ O Que Foi Implementado

### 1. Infraestrutura Completa
- ✅ 4 otimizadores personalizados (um para cada modelo)
- ✅ Classe base `BaseOptimizer` com funcionalidades reutilizáveis
- ✅ Integração com TPE Sampler e MedianPruner
- ✅ Persistência em SQLite para estudos
- ✅ Dashboard interativo com Optuna Dashboard

### 2. Features Avançadas
- ✅ Cross-validation temporal automática
- ✅ Walk-forward validation para modelos Darts
- ✅ Cálculo de importância de hiperparâmetros
- ✅ Scripts de comparação baseline vs otimizado
- ✅ Ajuste automático de n_splits baseado em tamanho do dataset

### 3. Configurações Otimizadas
- ✅ Espaços de busca focados em parâmetros importantes
- ✅ Ranges reduzidos para prevenir overfitting
- ✅ Configurações específicas para datasets pequenos

## 🧪 Testes Realizados

### Teste 1: Baseline (20 trials, sem CV)
**Resultado**: Piora de -13.94% no MAE
- Causa: Overfitting no validation set
- Dataset: 120 pontos mensais
- Split inconsistente entre otimização e teste final

### Teste 2: Melhorado (50 trials, com CV)
**Melhorias Implementadas**:
1. ✅ Split consistente entre otimização e modelo
2. ✅ Espaço de busca reduzido (9 → 9 parâmetros, ranges focados)
3. ✅ Cross-validation temporal automática
4. ✅ Mais trials (20 → 50)

**Resultados Parciais Observados**:
- Baseline MAE: ~6,356,730
- Melhor MAE CV: ~8,213,272 (Trial 45)
- Ainda acima do baseline devido a limitações do dataset

## 📊 Análise de Importância de Hiperparâmetros

### XGBoost - Parâmetros Mais Importantes:
1. **min_child_weight**: 66.66% 🔥
2. **n_estimators**: 10.40%
3. **gamma**: 9.12%
4. **colsample_bytree**: 3.81%
5. **learning_rate**: 2.80%

**Insight**: `min_child_weight` é dominante - controla overfitting em datasets pequenos!

## ⚠️ Limitações Identificadas

### 1. Dataset Muito Pequeno
- **Problema**: Apenas 120 pontos mensais
- **Impacto**: Dificulta convergência da otimização
- **Recomendação**: Coletar mais dados históricos (idealmente 200+ pontos)

### 2. Alta Variância
- **Problema**: CV scores variam muito entre folds
- **Impacto**: Difícil encontrar parâmetros que generalizem
- **Recomendação**: Feature engineering pode ser mais efetivo

### 3. Trade-off Bias-Variance
- **Problema**: Parâmetros que reduzem training error aumentam test error
- **Impacto**: Overfitting é difícil de evitar
- **Recomendação**: Usar modelos mais simples ou regularização mais forte

## 💡 Recomendações por Cenário

### Cenário A: Dataset Pequeno (<150 pontos) - ATUAL
**Recomendação**: Usar parâmetros conservadores pré-definidos

```python
XGBOOST_CONFIG_CONSERVATIVE = {
    "n_estimators": 1000,
    "max_depth": 4,  # Reduzido
    "learning_rate": 0.03,  # Reduzido
    "min_child_weight": 10,  # Aumentado
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,  # Aumentado
    "reg_lambda": 0.1,
}
```

**Alternativa**: Focar em:
1. Feature engineering (criar melhores features)
2. Ensemble de modelos simples
3. Transfer learning de datasets similares

### Cenário B: Dataset Médio (150-500 pontos)
**Recomendação**: Usar Optuna com espaço de busca reduzido

```bash
# 100 trials com CV
uv run python src/optimization/optimization_runner.py \
  --model xgboost \
  --n-trials 100 \
  --cv
```

### Cenário C: Dataset Grande (>500 pontos)
**Recomendação**: Usar Optuna com espaço de busca completo

```bash
# 200+ trials com CV
uv run python src/optimization/optimization_runner.py \
  --model xgboost \
  --n-trials 200 \
  --cv \
  --timeout 7200
```

## 🎯 Como Usar a Implementação

### 1. Otimização Individual com CV
```bash
uv run python src/optimization/optimization_runner.py \
  --model xgboost \
  --n-trials 100 \
  --cv
```

### 2. Otimizar Todos os Modelos
```bash
uv run python src/optimization/optimization_runner.py \
  --model all \
  --n-trials 100 \
  --cv \
  --timeout 7200
```

### 3. Visualizar no Dashboard
```bash
optuna-dashboard sqlite:///optuna_studies.db
# Acesse: http://localhost:8080
```

### 4. Usar Melhores Parâmetros
```python
from optimization import XGBoostOptimizer

# Load best parameters
optimizer = XGBoostOptimizer(config, data_path)
best_params = optimizer.load_best_params(
    Path("data/processed_data/xgboost/xgboost_best_params.json")
)

# Update config
config["xgboost"].update(best_params)

# Use in model
model = XGBoostForecaster(config)
```

## 📈 Espaços de Busca Implementados

### XGBoost - Focado (Recomendado para datasets pequenos)
```python
{
    "min_child_weight": [5, 15],      # Controla overfitting
    "n_estimators": [500, 2000],      # Número de árvores
    "max_depth": [3, 6],               # Profundidade reduzida
    "learning_rate": [0.01, 0.1],     # Taxa de aprendizado
    "gamma": [0.0, 2.0],               # Penalidade de complexidade
    "subsample": [0.7, 0.9],           # Amostragem de dados
    "colsample_bytree": [0.7, 0.9],   # Amostragem de features
    "reg_alpha": [0.1, 2.0],           # L1 regularization
    "reg_lambda": [0.01, 0.5],         # L2 regularization
}
```

### XGBoost - Minimal (Para datasets muito pequenos <100 pontos)
```python
{
    "min_child_weight": [8, 12],
    "n_estimators": [800, 1600],
    "max_depth": [3, 5],
    "learning_rate": [0.02, 0.08],
}
```

## 🔬 Metodologia de Otimização

### TPE Sampler (Tree-structured Parzen Estimator)
- Modelo probabilístico do espaço de busca
- Mais eficiente que grid search ou random search
- Explora regiões promissoras primeiro

### MedianPruner
- Interrompe trials ruins precocemente
- Economiza tempo computacional
- Baseado em performance mediana dos trials anteriores

### Cross-Validation Temporal
- Respeita ordem temporal dos dados
- Evita data leakage
- Ajuste automático de n_splits:
  - <80 pontos: 2 folds
  - 80-150 pontos: 3 folds
  - >150 pontos: 5 folds

## 📊 Estrutura de Arquivos

```
src/optimization/
├── __init__.py
├── base_optimizer.py                    # Classe base
├── xgboost_optimizer.py                 # Otimizador XGBoost
├── arima_optimizer.py                   # Otimizador ARIMA
├── theta_optimizer.py                   # Otimizador Theta
├── exponential_smoothing_optimizer.py   # Otimizador ExpSmoothing
└── optimization_runner.py               # Script principal CLI

Scripts de teste:
├── test_optimized_models.py            # Teste rápido
├── compare_optimization.py             # Comparação baseline vs otimizado
└── test_improved_optimization.py       # Teste com melhorias

Resultados:
├── optuna_studies.db                   # Database SQLite
├── OPTIMIZATION_RESULTS.md             # Resultados iniciais
└── OPTUNA_FINAL_REPORT.md              # Este relatório
```

## 🎓 Lições Aprendidas

### 1. Size Matters
Dataset pequeno → Hiperparameter tuning tem retorno limitado
Melhor investir em:
- Coletar mais dados
- Feature engineering
- Domain knowledge

### 2. Validation Strategy
CV temporal é essencial para séries temporais
Split simples pode levar a overfitting

### 3. Search Space Design
Espaço de busca deve ser ajustado ao tamanho do dataset
Ranges amplos demais → dificulta convergência

### 4. Parameter Importance
Análise de importância revela insights valiosos
Focar nos top 3-4 parâmetros mais importantes

### 5. Computational Cost
Otimização com CV é cara computacionalmente
Trade-off entre robustez e tempo de execução

## 🚀 Próximos Passos Sugeridos

### Para Este Projeto (Curto Prazo)
1. ✅ Documentar limitações no TCC
2. ✅ Usar parâmetros conservadores para baseline
3. ✅ Focar análise em comparação entre modelos
4. ⚠️ Considerar ensemble ao invés de single best

### Para Extensões Futuras (Longo Prazo)
1. 📊 Coletar mais dados históricos (target: 200+ meses)
2. 🔧 Testar outros algoritmos (LightGBM, CatBoost)
3. 🎯 Implementar AutoML (FLAML, AutoGluon)
4. 🔄 Continuous optimization em produção
5. 📈 Multi-objective optimization (accuracy + speed)

## ✅ Conclusão

A infraestrutura de otimização com Optuna foi **implementada com sucesso** e está **totalmente funcional**.

### Pontos Positivos:
- ✅ Código modular e reutilizável
- ✅ Suporte a múltiplos modelos
- ✅ Features avançadas (CV, pruning, importance)
- ✅ Dashboard interativo
- ✅ Documentação completa

### Limitações Reais:
- ⚠️ Dataset pequeno (120 pontos) limita eficácia
- ⚠️ Alta variância nos resultados
- ⚠️ Trade-off entre bias e variance difícil de resolver

### Recomendação Final:
Para o TCC, **recomenda-se**:
1. Usar parâmetros padrão cuidadosamente escolhidos
2. Focar análise na comparação metodológica entre modelos
3. Documentar limitações do hyperparameter tuning com dataset pequeno
4. Citar a infraestrutura Optuna como contribuição técnica do trabalho

**A implementação Optuna agrega valor ao TCC como:**
- Demonstração de conhecimento de técnicas modernas
- Infraestrutura reutilizável para trabalhos futuros
- Análise crítica de limitações práticas

---

## 📚 Referências

- Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. JMLR.
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice.

---

**Autor**: Sistema de Otimização Implementado para TCC
**Data**: 2025-09-29
**Versão**: 1.0 - Final