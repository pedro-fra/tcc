# Análise do Problema nos Dados do Power BI

## Problema Identificado

Os valores exportados do Power BI estão **DIÁRIOS** e não **MENSAIS**. Isso explica por que:
1. Cada linha tem valores diferentes (são transações de dias diferentes)
2. Todas as linhas mostram "Faturamento Realizado" = "Projeção Power BI" (ambas são iguais para dados históricos)
3. Na linha de TOTAL, os valores são diferentes (agregação mensal vs agregação diária)

## Dados Exportados

- **Total de linhas**: 1.781 (dados diários)
- **Período**: 03/01/2019 a 02/10/2025
- **Estrutura**:
  ```
  Data | TCC Faturamento Realizado | TCC Faturamento Projetado Power BI
  2019-01-03 | R$ 11.244 | R$ 11.244
  2019-01-04 | R$ 11.887 | R$ 11.887
  ...
  ```

## O Problema

A medida `TCC Faturamento Projetado Power BI` está retornando **dados DIÁRIOS** porque:

1. A tabela do Power BI está em **nível de dia** (não agregada)
2. O filtro `[Dim] Calendário[Data]` está pegando CADA DIA
3. A projeção é calculada para CADA DIA individual
4. Resultado: Para dados históricos, realizado = projeção

## O que Deveria Ser

Para uma comparação justa com XGBoost, os dados precisam ser:
- **Agregados mensalmente** (não diários)
- **27 meses** (dez/2022 a set/2025)
- **Formato**: Data (EOMONTH), Realizado (soma do mês), Projeção (projeção do mês)

**Exemplo correto:**
```
Data | TCC Faturamento Realizado | TCC Faturamento Projetado Power BI
2022-12-31 | 1.200.000.000 | 1.200.000.000
2023-01-31 | 1.350.000.000 | 1.350.000.000
2023-02-28 | 1.100.000.000 | 1.100.000.000
...
```

## Solução

### Opção 1: Agregar no Power BI (Recomendado)

1. Crie uma **nova tabela** agregando os dados por MÊS
2. Use esta estrutura:
   - Coluna: `[Dim] Calendário[MesAno]` ou `[Dim] Calendário[Data]` (EOMONTH)
   - Valores: Soma de `TCC Faturamento Realizado`
   - Valores: Soma de `TCC Faturamento Projetado Power BI`

3. **Passo a passo no Power BI**:
   - Selecione a coluna Data da tabela
   - **Clique direito** → **Agrupar por** → **Data** → **Mês**
   - Isso cria uma tabela agregada

4. Exporte esta tabela agregada como CSV

### Opção 2: Filtrar Período no Power BI

1. Antes de exportar, adicione um filtro:
   - **`[Dim] Calendário[Data]`** >= **01/12/2022**
   - **`[Dim] Calendário[Data]`** <= **30/09/2025**

2. Isso reduzirá para ~840 linhas (27 meses × ~30 dias)

3. Ainda será necessário agregar por mês

### Opção 3: Agregar em Python (Se necessário)

Se você já exportou os dados diários, podemos agregar em Python:

```python
import pandas as pd

# Carregar dados
df = pd.read_csv('data/PBI.csv')
df['Data'] = pd.to_datetime(df['Data'])

# Remover formatação R$ e converter para número
df['TCC Faturamento Realizado'] = df['TCC Faturamento Realizado'].str.replace('R$ ', '').str.replace('.', '').astype(float)
df['TCC Faturamento Projetado Power BI'] = df['TCC Faturamento Projetado Power BI'].str.replace('R$ ', '').str.replace('.', '').astype(float)

# Agregar por mês (EOMONTH)
df['Mes'] = df['Data'].dt.to_period('M')
df_mensual = df.groupby('Mes').agg({
    'TCC Faturamento Realizado': 'sum',
    'TCC Faturamento Projetado Power BI': 'sum'
}).reset_index()

# Filtrar período (dez/2022 a set/2025)
df_mensual = df_mensual[
    (df_mensual['Mes'] >= '2022-12') &
    (df_mensual['Mes'] <= '2025-09')
]

# Exportar
df_mensual.to_csv('data/powerbi_historico_test_period.csv', index=False)
```

## Recomendação

**Use a Opção 1** (agregar no Power BI):
1. Mais rápido
2. Já terá os dados corretos
3. Não precisa de script adicional

**Passos resumidos**:
1. No Power BI, crie visualização com:
   - Eixo: `MesAno` (ou Data agrupada por mês)
   - Valores: `TCC Faturamento Realizado` (sum)
   - Valores: `TCC Faturamento Projetado Power BI` (sum)
2. Exporte para CSV
3. Execute o script Python com este novo CSV

## Verificação

Após exportar corretamente, o CSV deve ter:
- **27 linhas** (sem contar header)
- **Período**: dez/2022 a set/2025
- **Valores em milhões** (da ordem de 50M a 150M por mês)
- **Realizado e Projeção similares** (para dados históricos)

---

**Próximo passo**: Agregar os dados mensalmente e re-exportar o CSV.
