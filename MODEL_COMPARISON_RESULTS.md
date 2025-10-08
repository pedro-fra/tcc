# Comparacao de Modelos de Previsao de Vendas

## Resumo Executivo

**Melhor Modelo:** XGBoost

**MAE (Melhor):** R$ 6,246,417.50

## Tabela Comparativa

| Modelo | MAE | RMSE | MAPE (%) | Rank MAE | Rank RMSE | Rank MAPE | Rank Medio |
|--------|-----|------|----------|----------|-----------|-----------|------------|
| XGBoost | 6,246,417.50 | 7,885,915.95 | 16.32 | 1 | 1 | 1 | 1.00 |
| Theta | 17,327,600.78 | 21,287,394.49 | 39.71 | 2 | 2 | 2 | 2.00 |
| Exponential Smoothing | 21,846,386.39 | 25,914,518.67 | 63.00 | 3 | 3 | 3 | 3.00 |
| ARIMA | 28,710,800.45 | 32,311,238.76 | 78.73 | 4 | 4 | 4 | 4.00 |

## Interpretacao das Metricas

### MAE (Mean Absolute Error)
- Metrica principal para previsao de vendas
- Representa o erro medio em reais das previsoes
- Menor valor indica melhor desempenho

### RMSE (Root Mean Squared Error)
- Penaliza erros grandes mais severamente que MAE
- Util quando erros grandes sao criticos para o negocio
- Menor valor indica melhor desempenho

### MAPE (Mean Absolute Percentage Error)
- Erro percentual medio
- Facilita comparacao entre diferentes escalas
- Menor valor indica melhor desempenho

## Analise por Modelo

### 1. XGBoost

- **MAE:** R$ 6,246,417.50
- **RMSE:** R$ 7,885,915.95
- **MAPE:** 16.32%
- **Rank Medio:** 1.00

### 2. Theta

- **MAE:** R$ 17,327,600.78
- **RMSE:** R$ 21,287,394.49
- **MAPE:** 39.71%
- **Rank Medio:** 2.00

### 3. Exponential Smoothing

- **MAE:** R$ 21,846,386.39
- **RMSE:** R$ 25,914,518.67
- **MAPE:** 63.00%
- **Rank Medio:** 3.00

### 4. ARIMA

- **MAE:** R$ 28,710,800.45
- **RMSE:** R$ 32,311,238.76
- **MAPE:** 78.73%
- **Rank Medio:** 4.00

## Conclusao

O modelo **XGBoost** apresentou o melhor desempenho geral com MAE de R$ 6,246,417.50.

## Proximos Passos

1. Comparar resultados com baseline do Power BI
2. Avaliar viabilidade de ensemble dos melhores modelos
3. Validar resultados com stakeholders
