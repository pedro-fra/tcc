# Guia Simplificado: Exportar Dados do Power BI para Comparação

## Fluxo Resumido

```
Power BI (2 medidas)
    ↓
Tabela com 3 colunas
    ↓
Exportar CSV (27 meses)
    ↓
Python (compara com XGBoost)
    ↓
Relatório final
```

---

## Parte 1: Adicionar Medidas DAX

### Passo 1.1: Adicionar as 2 Medidas

1. Abra seu arquivo `.pbix` no Power BI Desktop
2. Vá para **"Modelagem"** → **"Nova Medida"**
3. Copie do arquivo `powerbi_medidas_tcc.txt` e adicione:

**Medida 1:**
```dax
TCC Faturamento Realizado =
    CALCULATE(
        SUM('[Fato] Faturamento'[VALOR_LIQ]),
        KEEPFILTERS('[Fato] Faturamento'[GERA_COBRANCA] = 1),
        KEEPFILTERS('[Fato] Faturamento'[OPERACAO] = "VENDA")
    )
```

**Medida 2:**
```dax
TCC Faturamento Projetado Power BI =
    VAR vDiasNoMes =
        CALCULATE(
            COUNTROWS('[Dim] Calendário'),
            KEEPFILTERS('[Dim] Calendário'[MesNum] = MONTH(TODAY())),
            KEEPFILTERS('[Dim] Calendário'[Ano] = YEAR(TODAY())),
            ALL(Aux_MesmoPeriodo)
        )
    VAR vDiasAteHoje =
        CALCULATE(
            DISTINCTCOUNT('[Dim] Calendário'[Data]),
            KEEPFILTERS('[Dim] Calendário'[Data] <= TODAY()),
            KEEPFILTERS('[Dim] Calendário'[MesNum] = MONTH(TODAY())),
            KEEPFILTERS('[Dim] Calendário'[Ano] = YEAR(TODAY())),
            ALL(Aux_MesmoPeriodo)
        )
    VAR vFaturamentoAteHoje = [TCC Faturamento Realizado]
    VAR vMediaDiaria = DIVIDE(vFaturamentoAteHoje, vDiasAteHoje)
    VAR vProjecao = vMediaDiaria * vDiasNoMes
    RETURN
    SWITCH(
        TRUE(),
        MONTH(TODAY()) = MONTH([Primeira Data do Filtro]) && vDiasAteHoje > 0,
        vProjecao,
        vFaturamentoAteHoje
    )
```

**Pronto!** Apenas 2 medidas.

---

## Parte 2: Criar Tabela para Exportação

### Passo 2.1: Criar Tabela com 3 Colunas

1. Clique em **"Tabela"** no painel de Visualizações
2. Arraste para o canvas:
   - **`[Dim] Calendário → Data`** (coluna de data)
   - **`TCC Faturamento Realizado`** (medida)
   - **`TCC Faturamento Projetado Power BI`** (medida)

**Resultado:**
```
| Data       | TCC Faturamento Realizado | TCC Faturamento Projetado Power BI |
|------------|---------------------------|----------------------------------|
| dez/2022   | 50.000.000                | 50.000.000                       |
| jan/2023   | 55.000.000                | 55.000.000                       |
| fev/2023   | ...                       | ...                              |
...
```

### Passo 2.2: Filtrar Período (27 meses)

1. Clique com botão direito na tabela → **"Editar interações"**
2. Adicione um filtro:
   - **`[Dim] Calendário[Data]`** → **Entre**
   - De: **01/12/2022**
   - Até: **30/09/2025**

---

## Parte 3: Exportar Dados para CSV

### Passo 3.1: Exportar Tabela

1. **Clique com botão direito** na tabela
2. Selecione **"Exportar dados"** → **"CSV"**
3. Salve como: **`powerbi_historico_test_period.csv`**
4. Local: **`c:\Users\PedroFrá\Documents\coding\tcc\data\`**

**Resultado do CSV:**
```
Data,TCC Faturamento Realizado,TCC Faturamento Projetado Power BI
2022-12-31,50000000,50000000
2023-01-31,55000000,55000000
2023-02-28,48000000,48000000
...
```

**Verificar**: Deve ter 27 linhas (27 meses de dez/2022 a set/2025)

---

## Parte 4: Executar Script Python

### Passo 4.1: Abrir Terminal e Executar

```bash
cd c:\Users\PedroFrá\Documents\coding\tcc
uv run python run_comparison_powerbi.py data/powerbi_historico_test_period.csv
```

### Passo 4.2: Verificar Resultado

O script gera:

1. **`COMPARACAO_XGBOOST_POWERBI.md`** - Relatório completo com:
   - Métricas XGBoost: MAE, RMSE, MAPE
   - Métricas Power BI: MAE, RMSE, MAPE
   - Qual modelo é melhor
   - Diferenças entre eles

2. **`data/plots/powerbi_comparison/`** - 4 Gráficos PNG:
   - `xgboost_vs_powerbi_forecast.png` - Linhas (Real vs XGBoost vs PBI)
   - `xgboost_vs_powerbi_metrics.png` - Barras de métricas
   - `xgboost_vs_powerbi_difference.png` - Diferenças em R$ e %
   - `xgboost_vs_powerbi_scatter.png` - Scatter plot

3. **`data/processed_data/comparison_results.json`** - Dados em JSON

---

## Checklist

- [ ] Medida 1: `TCC Faturamento Realizado` adicionada
- [ ] Medida 2: `TCC Faturamento Projetado Power BI` adicionada
- [ ] Tabela criada com 3 colunas
- [ ] Período filtrado: dez/2022 a set/2025 (27 meses)
- [ ] CSV exportado: `powerbi_historico_test_period.csv`
- [ ] CSV tem 27 linhas
- [ ] Script Python executado
- [ ] Relatório `COMPARACAO_XGBOOST_POWERBI.md` gerado

---

**Tempo**: ~30 minutos
