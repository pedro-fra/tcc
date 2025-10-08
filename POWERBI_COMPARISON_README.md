# Comparacao XGBoost vs Power BI

Este sistema compara as previsoes de vendas do XGBoost (melhor modelo de Machine Learning) com as previsoes baseline do Power BI.

## Arquivos Criados

### Scripts Principais
- `compare_with_powerbi.py`: Script principal para executar a comparacao completa
- `create_powerbi_sample.py`: Script auxiliar para criar arquivo CSV de exemplo

### Modulos
- `src/evaluation/powerbi_comparison.py`: Classe de comparacao e calculo de metricas
- `src/visualization/powerbi_comparison_plots.py`: Geracao de graficos comparativos

### Documentacao
- `powerbi_export_template.txt`: Instrucoes para exportar dados do Power BI
- `POWERBI_COMPARISON_README.md`: Este arquivo

## Como Usar

### Passo 1: Exportar Dados do Power BI

Voce precisa exportar as previsoes do Power BI em um arquivo CSV. Siga as instrucoes em `powerbi_export_template.txt`.

O arquivo deve conter 3 colunas:
- `Data`: Data no formato YYYY-MM-DD (primeiro dia de cada mes)
- `Valor_Real`: Valores reais observados (para meses passados)
- `Valor_Projetado`: Valores projetados pelo Power BI (para todos os meses)

Salve o arquivo como: `data/powerbi_forecast.csv`

### Passo 2: (Opcional) Criar Arquivo de Exemplo

Se quiser testar o sistema primeiro com dados de exemplo:

```bash
uv run python create_powerbi_sample.py
```

Isso criara `data/powerbi_forecast_sample.csv` que voce pode usar para testar.

### Passo 3: Executar a Comparacao

Execute o script principal:

```bash
uv run python compare_with_powerbi.py
```

O script ira:
1. Carregar as previsoes do Power BI
2. Carregar as previsoes do XGBoost
3. Alinhar as datas entre os modelos
4. Calcular metricas de comparacao
5. Gerar visualizacoes
6. Salvar resultados em arquivos

## Resultados Gerados

Os resultados sao salvos em `data/plots/powerbi_comparison/`:

### Graficos
- `xgboost_vs_powerbi_forecast_comparison.png`: Comparacao de series temporais
- `xgboost_vs_powerbi_difference.png`: Analise de diferencas (4 subplots)

### Arquivos de Dados
- `xgboost_vs_powerbi_results.json`: Resultados completos em formato JSON

## Metricas Calculadas

### Metricas de Comparacao Direta (ML vs Power BI)
- Diferenca total (absoluta e percentual)
- Diferenca media absoluta
- Diferenca percentual media
- Correlacao entre previsoes
- Periodos com sobre/subestimacao

### Metricas de Acuracia (vs Valores Reais)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Bias (vies sistematico)
- Melhoria percentual do ML sobre Power BI

## Interpretacao dos Resultados

### Graficos de Comparacao de Previsoes
Mostra as series temporais lado a lado:
- Linha preta: Valores reais (quando disponiveis)
- Linha vermelha tracejada: Power BI
- Linha azul: Modelo ML

### Graficos de Analise de Diferencas
4 subplots mostrando:
1. **Diferenca Absoluta**: Valor ML - Valor Power BI (em R$)
   - Verde: ML previu maior que Power BI
   - Vermelho: ML previu menor que Power BI

2. **Diferenca Percentual**: (ML - Power BI) / Power BI * 100
   - Mostra magnitude relativa das diferencas

3. **Histograma de Diferencas**: Distribuicao das diferencas percentuais
   - Linha vertical vermelha: media
   - Permite identificar vies sistematico

4. **Scatter Plot de Correlacao**: Power BI vs ML
   - Linha vermelha: linha perfeita (y=x)
   - Pontos proximos da linha = alta concordancia

### Grafico de Metricas Consolidadas
Compara todos os modelos em 4 metricas:
1. **Diferenca Percentual Media**: Quanto cada modelo difere do Power BI
2. **Correlacao**: Quao similar sao as previsoes
3. **Comparacao de MAE**: Precisao de Power BI vs ML
4. **Melhoria em MAE**: Quanto cada modelo ML melhora sobre Power BI
   - Verde: ML e melhor
   - Vermelho: Power BI e melhor

## Exemplo de Uso para o TCC

### Analise Comparativa
```python
# Analisar resultados
import json

# Carregar resultados
with open('data/plots/powerbi_comparison/xgboost_vs_powerbi_results.json') as f:
    results = json.load(f)

# Ver metricas
xgb_results = results['results']['xgboost']
print("Comparacao XGBoost vs Power BI:")
print(xgb_results['forecast_comparison'])
print("\nAcuracia vs Valores Reais:")
print(xgb_results['accuracy_evaluation'])
```

### Discussao para o TCC
Use os resultados para:
1. Comparar abordagem tradicional (Power BI) vs XGBoost (melhor modelo ML)
2. Avaliar se XGBoost traz ganhos significativos sobre Power BI
3. Identificar em quais periodos XGBoost e superior
4. Discutir tradeoffs (complexidade vs acuracia)
5. Recomendar qual abordagem usar em producao

## Ajustes das Medidas DAX

Para que o Power BI preveja ate dezembro, use as medidas ajustadas fornecidas anteriormente que incluem:
- Projecao para mes atual baseada em dias uteis
- Projecao para meses futuros baseada em media historica
- Totalizador que soma todas as projecoes

## Solucao de Problemas

### Erro: "Power BI forecast file not found"
- Certifique-se de exportar o CSV do Power BI
- Coloque o arquivo em `data/powerbi_forecast.csv`
- Ou use `create_powerbi_sample.py` para criar um arquivo de teste

### Erro: "No overlapping dates"
- Verifique se as datas no CSV do Power BI estao no formato correto (YYYY-MM-DD)
- Certifique-se de que o periodo cobre os mesmos meses das previsoes ML
- As previsoes ML geralmente cobrem os ultimos meses do dataset

### Erro: "Forecast file not found"
- Execute o XGBoost primeiro (run_preprocessing.py, depois treinar o modelo)
- Certifique-se de que os arquivos de previsao existem em `data/processed_data/xgboost/`

## Proximos Passos

Apos executar a comparacao:
1. Analise os graficos gerados
2. Verifique se XGBoost trouxe melhoria significativa sobre Power BI
3. Use os resultados para a secao de resultados do TCC
4. Discuta as implicacoes praticas na conclusao
5. Recomende qual abordagem usar em producao

## Contato

Se precisar de ajustes ou tiver duvidas sobre a analise, consulte a metodologia completa em `data/metodologia.md`.
