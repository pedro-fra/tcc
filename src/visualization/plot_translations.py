"""
Portuguese translations for ARIMA plotting module.
Contains all text labels, titles, and descriptions in Portuguese (Brazil).
"""

# Main titles and labels
PLOT_TITLES = {
    "main_eda": "Analise Exploratoria - Serie Temporal de Vendas",
    "exponential_main_eda": "Analise Exploratoria - Suavizacao Exponencial",
    "decomposition": "Analise de Decomposicao da Serie Temporal",
    "seasonality": "Analise de Sazonalidade",
    "statistical": "Analise de Propriedades Estatisticas",
    "distribution": "Analise de Distribuicao",
    "prediction_comparison": "Modelo ARIMA: Previsoes vs Vendas Reais",
    "residual_analysis": "Analise de Residuos",
    "performance_metrics": "Visualizacao de Metricas de Performance",
    "stationarity_tests": "Resultados dos Testes de Estacionariedade",
    "model_summary": "Resumo do Modelo ARIMA",
    "acf_pacf_analysis": "Analise ACF/PACF para Selecao de Parametros ARIMA",
    "acf_plot": "Funcao de Autocorrelacao (ACF)",
    "pacf_plot": "Funcao de Autocorrelacao Parcial (PACF)",
    "parameter_suggestions": "Sugestoes de Parametros",
    "significance_analysis": "Analise de Significancia",
    "differencing_analysis": "Analise de Diferenciacao da Serie Temporal",
    "original_series": "Serie Temporal Original",
    "first_difference": "Primeira Diferenca",
    "stationarity_results": "Resultados dos Testes de Estacionariedade",
    "differencing_summary": "Resumo da Analise de Diferenciacao",
    # Exponential Smoothing specific titles
    "exponential_decomposition": "Decomposicao da Serie Temporal - Suavizacao Exponencial",
    "exponential_seasonality": "Analise de Sazonalidade - Suavizacao Exponencial",
    "exponential_statistical": "Propriedades Estatisticas - Suavizacao Exponencial",
    "exponential_distribution": "Analise de Distribuicao - Suavizacao Exponencial",
    "exponential_prediction_comparison": "Comparacao de Previsoes - Suavizacao Exponencial",
    "exponential_residual_analysis": "Analise de Residuos - Suavizacao Exponencial",
    "exponential_performance_metrics": "Metricas de Performance - Suavizacao Exponencial",
    "exponential_trend_seasonality": "Analise de Tendencia e Sazonalidade",
    "exponential_model_summary": "Resumo do Modelo - Suavizacao Exponencial",
    "rolling_stats": "Estatisticas Moveis",
    "monthly_averages": "Vendas Medias por Mes",
    "yearly_trends": "Totais de Vendas Anuais",
    "trend_component": "Componente de Tendencia",
    "seasonal_component": "Componente Sazonal",
    "residual_component": "Componente Residual",
    "monthly_distribution": "Distribuicao de Vendas Mensais",
    "quarterly_sales": "Vendas Medias por Trimestre",
    "yoy_comparison": "Comparacao Mensal Ano a Ano",
    "lag_plot": "Grafico de Defasagem",
    "rolling_mean_std": "Media e Desvio Padrao Moveis",
    "sales_distribution": "Distribuicao de Vendas",
    "qq_plot": "Grafico Q-Q",
    "box_plot": "Grafico de Caixa",
    "time_series_forecast": "Previsao da Serie Temporal",
    "detailed_test_period": "Periodo de Teste Detalhado",
    "residuals_over_time": "Residuos ao Longo do Tempo",
    "residuals_vs_fitted": "Residuos vs Valores Ajustados",
    "residual_distribution": "Distribuicao dos Residuos",
    "qq_residuals": "Grafico Q-Q dos Residuos",
    "actual_vs_predicted": "Real vs Previsto",
    "error_metrics": "Metricas de Erro",
    "percentage_error_time": "Erro Percentual ao Longo do Tempo",
    "cumulative_error": "Erro Cumulativo",
    "original_with_trend": "Serie Original com Tendencia",
    "trend_seasonality_analysis": "Analise de Tendencia e Sazonalidade",
    "seasonal_pattern": "Padrao Sazonal",
    "model_selection_guide": "Guia de Selecao do Modelo",
    "model_performance": "Performance do Modelo",
}

# Subplot titles
SUBPLOT_TITLES = {
    "sales_over_time": "Vendas ao Longo do Tempo",
    "rolling_stats": "Estatisticas Moveis",
    "monthly_averages": "Vendas Medias por Mes",
    "yearly_trends": "Totais de Vendas Anuais",
    "original_series": "Serie Temporal Original",
    "trend_component": "Componente de Tendencia",
    "seasonal_component": "Componente Sazonal",
    "residual_component": "Componente Residual",
    "sales_with_trend": "Vendas com Tendencia",
    "detrended_series": "Serie Sem Tendencia",
    "monthly_distribution": "Distribuicao de Vendas Mensais",
    "quarterly_sales": "Vendas Medias por Trimestre",
    "yoy_comparison": "Comparacao Mensal Ano a Ano",
    "lag_plot": "Grafico de Defasagem",
    "acf_plot": "Funcao de Autocorrelacao (ACF)",
    "pacf_plot": "Funcao de Autocorrelacao Parcial (PACF)",
    "rolling_mean_std": "Media e Desvio Padrao Moveis",
    "first_difference": "Primeira Diferenca",
    "sales_distribution": "Distribuicao de Vendas",
    "qq_plot": "Grafico Q-Q (Distribuicao Normal)",
    "box_plot": "Grafico de Caixa",
    "distribution_kde": "Distribuicao com KDE",
    "time_series_forecast": "Previsao da Serie Temporal",
    "detailed_test_period": "Visao Detalhada: Periodo de Teste",
    "residuals_over_time": "Residuos ao Longo do Tempo",
    "residuals_vs_fitted": "Residuos vs Valores Ajustados",
    "residual_distribution": "Distribuicao dos Residuos",
    "qq_residuals": "Grafico Q-Q dos Residuos",
    "actual_vs_predicted": "Real vs Previsto",
    "error_metrics": "Metricas de Erro",
    "percentage_error_time": "Erro Percentual ao Longo do Tempo",
    "cumulative_error": "Erro Absoluto Cumulativo",
    "adf_test": "Teste ADF",
    "kpss_test": "Teste KPSS",
    "model_performance": "Performance do Modelo",
    "model_config": "Configuracao do Modelo",
    "future_extensions": "Extensoes Futuras",
    "model_comparison": "Comparacao de Modelos",
}

# Axis labels
AXIS_LABELS = {
    "sales_value": "Valor de Vendas (R$)",
    "average_sales": "Vendas Medias (R$)",
    "total_sales": "Vendas Totais (R$)",
    "month": "Mes",
    "year": "Ano",
    "quarter": "Trimestre",
    "lag": "Defasagem",
    "acf": "ACF",
    "pacf": "PACF",
    "differenced_values": "Valores Diferenciados",
    "density": "Densidade",
    "residuals": "Residuos",
    "fitted_values": "Valores Ajustados",
    "actual_sales": "Vendas Reais",
    "predicted_sales": "Vendas Previstas",
    "error_value": "Valor do Erro",
    "percentage_error": "Erro Percentual Absoluto (%)",
    "cumulative_error": "Erro Cumulativo",
    "test_statistic": "Estatistica do Teste",
    "metric_value": "Valor da Metrica",
    "date": "Data",
    "sales": "Vendas (R$)",
    "difference": "Diferenca",
    "lags": "Defasagens",
    "autocorrelation": "Autocorrelacao",
    "partial_autocorrelation": "Autocorrelacao Parcial",
    "confidence_interval": "Intervalo de Confianca",
    "suggested_p": "Parametro p Sugerido",
    "suggested_d": "Parametro d Sugerido",
    "suggested_q": "Parametro q Sugerido",
    "pattern_interpretation": "Interpretacao do Padrao",
    "interpretation_guide": "Guia de Interpretacao",
}

# Legend labels
LEGEND_LABELS = {
    "original": "Original",
    "rolling_mean": "Media Movel",
    "rolling_std": "Desvio Padrao Movel",
    "trend": "Tendencia",
    "seasonal": "Sazonal",
    "residual": "Residual",
    "training_data": "Dados de Treino",
    "actual_test": "Real (Teste)",
    "predicted": "Previsto",
    "train_test_split": "Divisao Treino/Teste",
    "histogram": "Histograma",
    "kde": "KDE",
    "critical_values": "Valores Criticos",
    "adf_statistic": "Estatistica ADF",
    "kpss_statistic": "Estatistica KPSS",
    "mean": "Media",
    "original_series": "Serie Original",
    "first_difference": "Primeira Diferenca",
    "actual": "Real",
}

# Month names in Portuguese
MONTH_NAMES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

# Quarter names in Portuguese
QUARTER_NAMES = ["1o Tri", "2o Tri", "3o Tri", "4o Tri"]

# Model configuration text
MODEL_CONFIG_TEXT = """
Configuracao do Modelo ARIMA:
• Selecao automatica de parametros
• Sazonal: Habilitado (12 meses)
• Periodo de Teste: {test_samples} meses
• Periodo de Treino: {train_samples} meses

Resumo de Performance:
• Erro Absoluto Medio: {mae:.0f}
• Raiz do Erro Quadratico Medio: {rmse:.0f}
• Erro Percentual Absoluto Medio: {mape:.1f}%
• Precisao Direcional: {directional_accuracy:.1f}%
"""

# Additional text elements
ADDITIONAL_TEXT = {
    "future_analysis": "Analises adicionais\npodem ser incluidas aqui",
    "model_comparison_text": "Comparacao com\noutros metodos",
    "processing_eda": "Criando graficos de analise exploratoria",
    "processing_predictions": "Criando graficos de comparacao de previsoes",
    "processing_summary": "Criando grafico de resumo do modelo",
    "plots_saved": "Graficos EDA salvos em",
    "prediction_plots_saved": "Graficos de previsao salvos em",
    "lag_months": "meses",
    "samples": "amostras",
    "periods": "periodos",
    "months": "meses",
}
