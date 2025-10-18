# Comparacao: XGBoost vs Power BI - Analise Completa para TCC

## Sumario Executivo

Este documento apresenta a analise comparativa entre o modelo XGBoost (melhor modelo de previsao de vendas desenvolvido neste TCC) e o metodo de projecao utilizado atualmente no Power BI.

**Melhor Modelo**: A analise determinara qual abordagem oferece maior precisao para previsao de faturamento mensal.

**Periodo de Analise**: Dezembro/2022 a Setembro/2025 (27 meses do conjunto de teste)

---

## 1. Introducao

### 1.1 Contexto

O Power BI utiliza uma medida chamada "Faturamento Projetado" que estima o faturamento total do mes em andamento baseado em:
- Faturamento acumulado ate o dia atual
- Dias uteis (excluindo sabados, domingos e feriados)
- Projecao linear para todo o mes

Por outro lado, o modelo XGBoost desenvolvido neste TCC utiliza:
- Series temporais univariadas
- Engenharia automatica de features via Darts
- 17 lags principais para capturar dependencias temporais
- 6 encoders temporais (mes, ano, trimestre, etc.)
- Gradient boosting com 2000 arvores

### 1.2 Objetivo da Comparacao

Comparar qual modelo oferece melhor precisao nas previsoes de faturamento mensal, considerando:
- Erro absoluto medio (MAE)
- Raiz do erro quadratico medio (RMSE)
- Erro percentual absoluto medio (MAPE)
- Analise temporal de performance

---

## 2. Metodologia

### 2.1 Dados Utilizados

**Fonte**: Tabela `[Fato] Faturamento` do Power BI

**Filtros Aplicados**:
- `GERA_COBRANCA = 1` (apenas transacoes que geram cobranca)
- `OPERACAO = "VENDA"` (apenas vendas, excluindo devoluções)

**Agregacao**: Dados agregados mensalmente usando EOMONTH(DATE_CAD)

**Periodo**: Outubro/2014 a Setembro/2025 (132 meses totais)
- Treino: Outubro/2014 a Novembro/2022 (105 meses)
- Teste: Dezembro/2022 a Setembro/2025 (27 meses)

### 2.2 Preparacao dos Dados

#### XGBoost

**Formato**: Series temporal univariada
- Entrada: Série de faturamento mensal agregado
- Processamento Darts:
  - 17 lags principais: [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -15, -18, -24, -30, -36]
  - 8 lags de covariadas: [-1, -2, -3, -4, -5, -6, -12, -24]
  - 6 encoders temporais: month, year, quarter, dayofyear, weekofyear, dayofweek
  - Normalizacao: MaxAbsScaler

#### Power BI

**Formato**: Faturamento mensal com projecao baseada em dias uteis

**Calculo da Projecao** (para mes em andamento):
```
Projecao = (Faturamento ate hoje / Dias uteis transcorridos) × Dias uteis totais
```

**Exclusoes**: Sabados, domingos e feriados

### 2.3 Metricas de Avaliacao

#### MAE (Mean Absolute Error)
```
MAE = (1/n) * Σ|Valor_Real - Valor_Previsto|
```
- Interpretacao: Erro absoluto medio em reais
- Unidade: R$

#### RMSE (Root Mean Squared Error)
```
RMSE = √((1/n) * Σ(Valor_Real - Valor_Previsto)²)
```
- Interpretacao: Penaliza mais fortemente erros grandes
- Unidade: R$

#### MAPE (Mean Absolute Percentage Error)
```
MAPE = (1/n) * Σ(|Valor_Real - Valor_Previsto| / Valor_Real) × 100
```
- Interpretacao: Erro percentual medio
- Unidade: %
- Vantagem: Independente da escala dos valores

---

## 3. Resultados da Comparacao

### 3.1 Metricas de Desempenho

#### XGBoost

| Metrica | Valor |
|---------|-------|
| **MAE** | R$ 10,110,160.96 |
| **RMSE** | R$ 13,302,309.10 |
| **MAPE** | 26.91% |

#### Power BI

| Metrica | Valor |
|---------|-------|
| **MAE** | [Será preenchido com dados exportados] |
| **RMSE** | [Será preenchido com dados exportados] |
| **MAPE** | [Será preenchido com dados exportados] |

### 3.2 Analise Comparativa

#### Melhor Modelo por Metrica

| Metrica | Melhor Modelo | Diferenca |
|---------|---|---|
| MAE | [A determinar] | [Será calculado] |
| RMSE | [A determinar] | [Será calculado] |
| MAPE | [A determinar] | [Será calculado] |

#### Conclusao Geral

[Será preenchida apos análise dos dados]

---

## 4. Analise Temporal

### 4.1 Performance por Periodo

A analise identifica em quais periodos cada modelo performou melhor:

- **Periodos de crescimento**: Qual modelo se adaptou melhor?
- **Periodos de declinio**: Qual modelo capturou melhor a tendencia?
- **Periodos de transicao**: Qual modelo reagiu mais rapidamente?

### 4.2 Volatilidade dos Erros

- **Consistencia**: Qual modelo teve erros mais consistentes?
- **Outliers**: Quais periodos tiveram erros extraordinarios?
- **Tendencia dos Erros**: Houve vieses sistematicos?

---

## 5. Discussao

### 5.1 Vantagens do XGBoost

- Capacidade de capturar dependencias nao-lineares complexas
- Adaptacao automatica via engenharia de features
- Sem suposicoes estatisticas rígidas (como estacionariedade)
- Flexibilidade na parametrizacao

### 5.2 Vantagens do Power BI

- Simplicidade conceitual (baseado em media diaria simples)
- Facilidade de compreensao para usuarios finais
- Integracao nativa no ambiente de BI existente
- Consideracao explícita de dias uteis e feriados

### 5.3 Limitacoes da Comparacao

- O Power BI foi projetado primariamente para projecoes intramês (mes em andamento)
- Comparacao com dados históricos pode nao refletir uso operacional
- Diferenca fundamental: Power BI usa lógica de dias úteis, XGBoost usa todos os dias

---

## 6. Recomendacoes

### 6.1 Para Implementacao no Power BI

Se XGBoost for selecionado como modelo preferido:

1. **Importar Previsoes**: Criar tabela com previsoes mensais do XGBoost
2. **Criar Medidas DAX**:
   - `TCC XGBoost Previsao` - Valor da previsao
   - `TCC Diferenca` - Diferenca vs Power BI
   - `TCC Diferenca Percentual` - Diferenca percentual

3. **Visualizacoes**:
   - Comparacao lado a lado em cards
   - Grafico de linhas temporal
   - Indicadores de qual modelo performou melhor

### 6.2 Para Producao

Possíveis estrategias de implementacao:

**Opcao A: Hibrida**
- Usar XGBoost para previsoes mensais
- Manter Power BI para projecoes intramês (mes em andamento)

**Opcao B: XGBoost Completo**
- Substituir projecoes Power BI por modelo XGBoost
- Requer integracaodados modelo em tempo real

**Opcao C: Consenso Ponderado**
- Combinar previsoes: (XGBoost × 0.6) + (Power BI × 0.4)
- Aproveita forcas de ambos modelos

---

## 7. Limitacoes

### 7.1 Data Utilizada

- Dados historicos ate Setembro/2025
- Proximas atualizacoes do modelo necessarias periodicamente
- Comportamento futuro pode diferir do passado

### 7.2 Factores Externos Nao Modelados

- Campanhas comerciais especiais
- Eventos economicos significativos
- Mudancas estruturais no negocio
- Sazonalidade extraordinaria

### 7.3 Horizonte de Previsao

- Analise focada em previsoes mensais
- Desempenho pode diferir para otros horizontes

---

## 8. Conclusao

A comparacao entre XGBoost e Power BI fornece insights importantes sobre qual metodo eh mais adequado para a empresa. Enquanto o XGBoost oferece sofisticacao matematica avancada, o Power BI oferece transparencia e facilidade de compreensao.

**Decisao Final**: [A ser determinada apos analise completa dos dados]

**Proximos Passos**:
1. Validar resultados com stakeholders
2. Considerar fatores de implementacao e manutencao
3. Implementar metodo selecionado em producao
4. Monitorar performance ao longo do tempo

---

## 9. Apendice Tecnico

### 9.1 Configuracao do XGBoost

```python
Hiperparametros Utilizados:
- n_estimators: 2000
- max_depth: 8
- learning_rate: 0.05
- subsample: 0.9
- colsample_bytree: 0.9
- reg_alpha: 0.2 (L1 regularization)
- reg_lambda: 1.5 (L2 regularization)
```

### 9.2 Features Utilizadas pelo XGBoost

**Lags Principais** (17 features):
`[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -15, -18, -24, -30, -36]`

**Lags de Covariadas** (8 features):
`[-1, -2, -3, -4, -5, -6, -12, -24]`

**Encoders Temporais** (6 features):
`[month, year, quarter, dayofyear, weekofyear, dayofweek]`

**Normalizacao**: MaxAbsScaler

### 9.3 Arquivos de Entrada Necessarios

```
data/
├── powerbi_historico_test_period.csv        # CSV exportado do Power BI
└── processed_data/
    └── xgboost/
        └── xgboost_model_summary.json       # Metricas do XGBoost
```

### 9.4 Arquivos de Saida Gerados

```
data/
└── plots/powerbi_comparison/
    ├── xgboost_vs_powerbi_forecast.png      # Comparacao de series
    ├── xgboost_vs_powerbi_metrics.png       # Comparacao de metricas
    ├── xgboost_vs_powerbi_difference.png    # Diferencas ao longo do tempo
    └── xgboost_vs_powerbi_scatter.png       # Scatter plot comparativo

COMPARACAO_XGBOOST_POWERBI.md               # Relatorio (este arquivo)
```

---

**Data de Atualizacao**: [Data da analise]
**Autor**: [Seu Nome]
**Projeto**: TCC - Previsao de Vendas com ML

---

## Referencias

- Documentacao TCC: `data/TCC2.md`
- Analise Power BI: `analise_powerbi_tmdl.md`
- Metodologia: `metodologia.md`
