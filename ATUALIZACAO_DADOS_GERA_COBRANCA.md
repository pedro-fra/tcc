# ATUALIZACAO CRITICA: DADOS COM FILTRO GERA_COBRANCA = 1

**Data:** 26/10/2025
**Autor:** Claude Code
**Status:** CONCLUIDO

---

## 1. PROBLEMA IDENTIFICADO

Durante analise do projeto, foi identificado que o dataset original (`data/fat_factory.csv`) **NAO incluia o filtro `GERA_COBRANCA = 1`** que o Power BI estava usando. Isso causava uma discrepancia significativa entre as metricas dos modelos de ML e as metricas do Power BI.

### Impacto do Problema
- Modelos de ML estavam sendo treinados com **TODOS** os registros de OPERACAO = 'VENDA'
- Power BI estava usando apenas registros com **GERA_COBRANCA = 1**
- Comparacao entre modelos nao era justa (datasets diferentes)

---

## 2. SOLUCAO IMPLEMENTADA

### 2.1 Novo Dataset
Exportado novo CSV do banco de dados do cliente usando a query:

```sql
SELECT [APELIDO], [DATE_CAD], [VALOR_LIQ]
FROM [dbo].[vw_faturamentoComercial]
WHERE OPERACAO = 'VENDA' AND GERA_COBRANCA = 1
```

### 2.2 Mudancas no Dataset
- **Coluna de Data:** `DATA_EMISSAO_PEDIDO` → `DATE_CAD`
- **Formato Data:** dd/MM/yyyy → yyyy-MM-dd (com timestamp)
- **Separador CSV:** `;` → `,`
- **Formato Valores:** NUMERIC (1378.00 = R$ 1.378,00)
- **Total Registros:** 37.649 (com GERA_COBRANCA = 1)

---

## 3. ARQUIVOS MODIFICADOS

### 3.1 Configuracoes
- **`src/config.py`:**
  - `date_column`: "DATA_EMISSAO_PEDIDO" → "DATE_CAD"
  - `csv_separator`: ";" → ","
  - `date_format`: "%d/%m/%Y" → None (auto-detect)
  - Removido: `operation_column` e `target_operation`

### 3.2 Preprocessamento
- **`src/preprocessing/data_preprocessor.py`:**
  - Removida logica de filtro por OPERACAO (linhas 94-100)
  - Adicionado suporte para date_format=None (auto-detect)
  - Atualizado step_7 para nao depender de operation_column

- **`src/preprocessing/utils.py`:**
  - Atualizado `validate_data_quality()` para operation_column opcional

### 3.3 Modelos
- **`src/models/theta.py`:**
  - Removida logica de filtro por OPERACAO
  - Atualizado para usar DATE_CAD com auto-detect

### 3.4 Scripts
- **`recalculate_powerbi_correct.py`:**
  - Atualizado com metricas corretas do XGBoost Ultimate
  - MAE: R$ 120.808,89 | RMSE: R$ 157.902,94 | MAPE: 31.66%

---

## 4. PIPELINES EXECUTADOS

Todos os pipelines foram reexecutados com os dados corretos:

1. **Preprocessing:** 37.649 → 37.132 registros (apos filtros de qualidade)
2. **ARIMA:** MAE: R$ 121.014,75 | RMSE: R$ 143.364,02 | MAPE: 33.61%
3. **Exponential Smoothing:** MAE: R$ 107.171,15 | RMSE: R$ 137.369,02 | MAPE: 23.99%
4. **Theta:** MAE: R$ 186.346,45 | RMSE: R$ 233.478,88 | MAPE: 40.30%
5. **XGBoost Ultimate:** MAE: R$ 120.808,89 | RMSE: R$ 157.902,94 | MAPE: 31.66%

---

## 5. RESULTADOS FINAIS

### 5.1 Ranking por MAPE (Melhor → Pior)

| Modelo | MAE | RMSE | MAPE |
|--------|-----|------|------|
| **Power BI (MM6 + YoY)** | R$ 89.651,22 | R$ 113.475,15 | **21.82%** |
| **Exponential Smoothing** | R$ 107.171,15 | R$ 137.369,02 | **23.99%** |
| **XGBoost Ultimate** | R$ 120.808,89 | R$ 157.902,94 | **31.66%** |
| **ARIMA** | R$ 121.014,75 | R$ 143.364,02 | **33.61%** |
| **Theta** | R$ 186.346,45 | R$ 233.478,88 | **40.30%** |

### 5.2 Power BI vs XGBoost (Melhor ML)

| Metrica | Power BI | XGBoost | Diferenca | Melhoria |
|---------|----------|---------|-----------|----------|
| **MAPE** | 21.82% | 31.66% | 9.83 p.p. | **31.1%** |
| **MAE** | R$ 89.651,22 | R$ 120.808,89 | R$ 31.157,67 | **25.8%** |
| **RMSE** | R$ 113.475,15 | R$ 157.902,94 | R$ 44.427,79 | **28.1%** |

---

## 6. CONCLUSAO

O metodo hibrido do Power BI (MM6 + YoY) **superou TODOS os modelos de ML** em TODAS as metricas:
- 31.1% melhor que XGBoost em MAPE
- 25.8% melhor em MAE
- 28.1% melhor em RMSE

**IMPORTANTE:** Agora a comparacao e justa, pois todos os modelos foram treinados e testados com o **MESMO dataset** (filtrado por GERA_COBRANCA = 1).

---

## 7. ARQUIVOS GERADOS

1. **`data/processed_data/final_consolidated_report.json`** - Relatorio consolidado de todas as metricas
2. **`data/processed_data/powerbi_comparison_final.json`** - Comparacao Power BI vs XGBoost
3. **`generate_final_report.py`** - Script para gerar relatorio consolidado

---

## 8. PROXIMOS PASSOS

1. Atualizar monografia (`data/TCC2.md`) com os novos valores
2. Atualizar todos os Quadros (1-6) com as metricas corretas
3. Atualizar secao de Conclusao com os novos percentuais
4. Verificar se graficos precisam ser regerados

---

**VALIDACAO FINAL:** Todos os modelos foram retreinados com dados corretos (GERA_COBRANCA = 1) e as metricas foram recalculadas. A conclusao do trabalho permanece VALIDA e FORTALECIDA.
