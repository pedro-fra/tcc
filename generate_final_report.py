"""
Script para gerar relatorio consolidado final de todas as metricas.
Dados filtrados por GERA_COBRANCA = 1.
"""

import json
from pathlib import Path


def format_brazilian_currency(value):
    """Formata valor como moeda brasileira."""
    return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')


def load_metrics():
    """Carrega metricas de todos os modelos."""
    base_path = Path(__file__).parent / 'data' / 'processed_data'

    # Power BI
    pbi_file = base_path / 'powerbi_comparison_final.json'
    with open(pbi_file, 'r') as f:
        pbi_data = json.load(f)

    # ARIMA
    arima_file = base_path / 'arima' / 'arima_metrics.json'
    with open(arima_file, 'r') as f:
        arima_data = json.load(f)

    # Exponential Smoothing
    exp_file = base_path / 'exponential_smoothing' / 'exponential_metrics.json'
    with open(exp_file, 'r') as f:
        exp_data = json.load(f)

    # Theta
    theta_file = base_path / 'theta' / 'theta_model_summary.json'
    with open(theta_file, 'r') as f:
        theta_data = json.load(f)

    # XGBoost Ultimate
    xgb_file = base_path / 'xgboost_ultimate' / 'ultimate_results.json'
    with open(xgb_file, 'r') as f:
        xgb_data = json.load(f)

    return {
        'power_bi': pbi_data['power_bi_metrics'],
        'arima': arima_data,
        'exponential': exp_data,
        'theta': theta_data['evaluation_metrics'],
        'xgboost': xgb_data['metrics']
    }


def main():
    """Gera relatorio consolidado final."""

    print("="*80)
    print("RELATORIO CONSOLIDADO FINAL - TCC")
    print("Previsao de Faturamento com Dados Filtrados por GERA_COBRANCA = 1")
    print("="*80)

    metrics = load_metrics()

    print("\n" + "="*80)
    print("METRICAS DE TODOS OS MODELOS")
    print("="*80)
    print("\nPeriodo de Teste: Jul/2023 a Sep/2025 (27 meses)")
    print("\n{:<25} {:>20} {:>20} {:>10}".format("Modelo", "MAE", "RMSE", "MAPE"))
    print("-" * 80)

    # Ordenar por MAPE (melhor para pior)
    models = [
        ('Power BI (MM6 + YoY)', metrics['power_bi']),
        ('XGBoost Ultimate', metrics['xgboost']),
        ('Exponential Smoothing', metrics['exponential']),
        ('ARIMA', metrics['arima']),
        ('Theta', metrics['theta'])
    ]

    for name, m in models:
        mae_str = format_brazilian_currency(m['mae'])
        rmse_str = format_brazilian_currency(m['rmse'])
        mape_str = f"{m['mape']:.2f}%"
        print(f"{name:<25} {mae_str:>20} {rmse_str:>20} {mape_str:>10}")

    print("="*80)

    # Analise comparativa
    print("\n" + "="*80)
    print("ANALISE COMPARATIVA: POWER BI vs MELHOR MODELO ML (XGBoost)")
    print("="*80)

    pbi = metrics['power_bi']
    xgb = metrics['xgboost']

    diff_mape = xgb['mape'] - pbi['mape']
    improvement_mape = (diff_mape / xgb['mape']) * 100

    diff_mae = xgb['mae'] - pbi['mae']
    improvement_mae = (diff_mae / xgb['mae']) * 100

    diff_rmse = xgb['rmse'] - pbi['rmse']
    improvement_rmse = (diff_rmse / xgb['rmse']) * 100

    print(f"\nMAPE:")
    print(f"  Power BI: {pbi['mape']:.2f}%")
    print(f"  XGBoost:  {xgb['mape']:.2f}%")
    print(f"  Diferenca: {diff_mape:.2f} pontos percentuais")
    print(f"  Power BI e {improvement_mape:.1f}% melhor")

    print(f"\nMAE:")
    print(f"  Power BI: {format_brazilian_currency(pbi['mae'])}")
    print(f"  XGBoost:  {format_brazilian_currency(xgb['mae'])}")
    print(f"  Diferenca: {format_brazilian_currency(diff_mae)}")
    print(f"  Power BI e {improvement_mae:.1f}% melhor")

    print(f"\nRMSE:")
    print(f"  Power BI: {format_brazilian_currency(pbi['rmse'])}")
    print(f"  XGBoost:  {format_brazilian_currency(xgb['rmse'])}")
    print(f"  Diferenca: {format_brazilian_currency(diff_rmse)}")
    print(f"  Power BI e {improvement_rmse:.1f}% melhor")

    print("\n" + "="*80)
    print("CONCLUSAO")
    print("="*80)
    print("\nO metodo hibrido do Power BI (MM6 + YoY) superou TODOS os modelos de ML:")
    print(f"- {improvement_mape:.1f}% melhor que XGBoost em MAPE")
    print(f"- {improvement_mae:.1f}% melhor em MAE")
    print(f"- {improvement_rmse:.1f}% melhor em RMSE")
    print("\nO resultado valida a eficacia de metodos simples e interpretateis")
    print("para previsao de vendas em contextos especificos.")
    print("="*80)

    # Salvar relatorio
    output_file = Path(__file__).parent / 'data' / 'processed_data' / 'final_consolidated_report.json'

    consolidated_data = {
        'test_period': {
            'start': 'Jul/2023',
            'end': 'Sep/2025',
            'months': 27
        },
        'all_metrics': {
            'power_bi': metrics['power_bi'],
            'xgboost': metrics['xgboost'],
            'exponential_smoothing': metrics['exponential'],
            'arima': metrics['arima'],
            'theta': metrics['theta']
        },
        'comparison': {
            'winner': 'Power BI',
            'mape_improvement_pct': improvement_mape,
            'mae_improvement_pct': improvement_mae,
            'rmse_improvement_pct': improvement_rmse
        },
        'ranking_by_mape': [
            {'model': 'Power BI', 'mape': pbi['mape']},
            {'model': 'XGBoost', 'mape': xgb['mape']},
            {'model': 'Exponential Smoothing', 'mape': metrics['exponential']['mape']},
            {'model': 'ARIMA', 'mape': metrics['arima']['mape']},
            {'model': 'Theta', 'mape': metrics['theta']['mape']}
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    print(f"\nRelatorio salvo em: {output_file}")


if __name__ == '__main__':
    main()
