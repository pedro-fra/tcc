"""
Script para recalcular metricas corretas do Power BI a partir do CSV exportado.

Este script resolve o problema identificado no Quadro 6 da monografia onde
os valores de MAE e RMSE do Power BI estavam incorretos.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_brazilian_currency(value):
    """
    Converte valores em formato brasileiro para float.

    Args:
        value: String no formato 'R$ 123.456,78' ou float

    Returns:
        Valor float correspondente
    """
    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()
    value_str = value_str.replace('R$', '').replace(' ', '')
    value_str = value_str.replace('.', '').replace(',', '.')

    return float(value_str)


def calculate_metrics(actual, predicted):
    """
    Calcula MAE, RMSE e MAPE.

    Args:
        actual: Array com valores reais
        predicted: Array com valores previstos

    Returns:
        Dict com metricas MAE, RMSE, MAPE
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    errors = actual - predicted
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(abs_errors / actual) * 100

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape)
    }


def main():
    """Funcao principal para recalcular metricas do Power BI"""

    csv_path = Path(__file__).parent / 'data' / 'PBI_Previsoes.csv'

    print(f"Lendo dados de {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"\nTotal de registros: {len(df)}")
    print(f"Colunas: {df.columns.tolist()}")

    df['Real_Parsed'] = df['TCC Real Teste'].apply(parse_brazilian_currency)
    df['Erro_Absoluto_Parsed'] = df['TCC Erro Absoluto'].apply(parse_brazilian_currency)

    actual_values = df['Real_Parsed'].values
    predicted_values = df['TCC Previsao Hibrido'].values

    print(f"\nPrimeiros 5 valores reais: {actual_values[:5]}")
    print(f"Primeiros 5 valores previstos: {predicted_values[:5]}")

    metrics = calculate_metrics(actual_values, predicted_values)

    print("\n" + "="*60)
    print("METRICAS CORRETAS DO POWER BI (HIBRIDO MM6 + YOY)")
    print("="*60)
    print(f"MAE:  R$ {metrics['mae']:,.2f}")
    print(f"RMSE: R$ {metrics['rmse']:,.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print("="*60)

    xgboost_metrics = {
        'mae': 10110160.96,
        'rmse': 13302309.10,
        'mape': 26.91
    }

    print("\n" + "="*60)
    print("COMPARACAO: POWER BI vs XGBOOST")
    print("="*60)
    print(f"\nMetrica         Power BI         XGBoost          Melhor")
    print("-" * 60)
    print(f"MAE (R$)    {metrics['mae']:>15,.2f}  {xgboost_metrics['mae']:>15,.2f}  {'PBI' if metrics['mae'] < xgboost_metrics['mae'] else 'XGB'}")
    print(f"RMSE (R$)   {metrics['rmse']:>15,.2f}  {xgboost_metrics['rmse']:>15,.2f}  {'PBI' if metrics['rmse'] < xgboost_metrics['rmse'] else 'XGB'}")
    print(f"MAPE (%)    {metrics['mape']:>15.2f}  {xgboost_metrics['mape']:>15.2f}  {'PBI' if metrics['mape'] < xgboost_metrics['mape'] else 'XGB'}")
    print("="*60)

    mape_improvement = ((xgboost_metrics['mape'] - metrics['mape']) / xgboost_metrics['mape']) * 100
    print(f"\nMelhoria do Power BI em MAPE: {mape_improvement:.1f}%")

    output_data = {
        'power_bi_metrics': metrics,
        'xgboost_metrics': xgboost_metrics,
        'comparison': {
            'mape_improvement_pct': mape_improvement,
            'winner': 'Power BI' if metrics['mape'] < xgboost_metrics['mape'] else 'XGBoost'
        },
        'test_period': {
            'start': 'Jul/23',
            'end': 'Sep/25',
            'months': len(df)
        }
    }

    output_file = Path(__file__).parent / 'data' / 'processed_data' / 'powerbi_metrics_corrected.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResultados salvos em: {output_file}")

    print("\n" + "="*60)
    print("VALORES PARA ATUALIZAR NO QUADRO 6 DO TCC2.md:")
    print("="*60)
    print(f"Power BI MAE:  R$ {metrics['mae']:,.2f}")
    print(f"Power BI RMSE: R$ {metrics['rmse']:,.2f}")
    print(f"Power BI MAPE: {metrics['mape']:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
