"""
Script para calcular metricas corretas do Power BI com formatacao brasileira.
Valores no formato NUMERIC (1378.00 = R$ 1.378,00).
Dados filtrados por GERA_COBRANCA = 1.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_brazilian_currency(value):
    """
    Converte valores em formato brasileiro para float.
    R$ 369.258,95 -> 369258.95
    """
    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip()
    value_str = value_str.replace("R$", "").replace(" ", "")
    value_str = value_str.replace(".", "").replace(",", ".")

    return float(value_str)


def format_brazilian_currency(value):
    """
    Formata valor como moeda brasileira.
    369258.95 -> R$ 369.258,95
    """
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def calculate_metrics(actual, predicted):
    """Calcula MAE, RMSE e MAPE"""
    actual = np.array(actual)
    predicted = np.array(predicted)

    errors = actual - predicted
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(abs_errors / actual) * 100

    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def main():
    csv_path = Path(__file__).parent / "data" / "PBI_Previsoes.csv"

    print(f"Lendo dados de {csv_path}\n")
    df = pd.read_csv(csv_path)

    print(f"Total de registros: {len(df)}")
    print(f"Colunas: {df.columns.tolist()}\n")

    df["Real_Parsed"] = df["TCC Real Teste"].apply(parse_brazilian_currency)

    actual_values = df["Real_Parsed"].values
    predicted_values = df["TCC Previsao Hibrido"].values

    print(f"Primeiros 5 valores reais: {actual_values[:5]}")
    print(f"Primeiros 5 valores previstos: {predicted_values[:5]}\n")

    metrics = calculate_metrics(actual_values, predicted_values)

    print("=" * 70)
    print("METRICAS DO POWER BI (HIBRIDO MM6 + YOY)")
    print("=" * 70)
    print(f"MAE:  {format_brazilian_currency(metrics['mae'])}")
    print(f"RMSE: {format_brazilian_currency(metrics['rmse'])}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print("=" * 70)

    xgboost_metrics = {"mae": 120808.89, "rmse": 157902.94, "mape": 31.66}

    print("\n" + "=" * 70)
    print("COMPARACAO: POWER BI vs XGBOOST")
    print("=" * 70)
    print("\nMetrica         Power BI                XGBoost                 Melhor")
    print("-" * 70)
    print(
        f"MAE         {format_brazilian_currency(metrics['mae']):>20}    {format_brazilian_currency(xgboost_metrics['mae']):>20}    {'PBI' if metrics['mae'] < xgboost_metrics['mae'] else 'XGB'}"
    )
    print(
        f"RMSE        {format_brazilian_currency(metrics['rmse']):>20}    {format_brazilian_currency(xgboost_metrics['rmse']):>20}    {'PBI' if metrics['rmse'] < xgboost_metrics['rmse'] else 'XGB'}"
    )
    print(
        f"MAPE        {metrics['mape']:>18.2f}%    {xgboost_metrics['mape']:>18.2f}%    {'PBI' if metrics['mape'] < xgboost_metrics['mape'] else 'XGB'}"
    )
    print("=" * 70)

    diff_mape_pp = xgboost_metrics["mape"] - metrics["mape"]
    melhoria_mape = ((xgboost_metrics["mape"] - metrics["mape"]) / xgboost_metrics["mape"]) * 100

    diff_mae = xgboost_metrics["mae"] - metrics["mae"]
    melhoria_mae = ((xgboost_metrics["mae"] - metrics["mae"]) / xgboost_metrics["mae"]) * 100

    print(f"\nDiferenca MAPE: {diff_mape_pp:.2f} pontos percentuais")
    print(f"Melhoria relativa do Power BI em MAPE: {melhoria_mape:.1f}%")
    print(f"\nDiferenca MAE: {format_brazilian_currency(diff_mae)}")
    print(f"Melhoria relativa do Power BI em MAE: {melhoria_mae:.1f}%")

    output_data = {
        "power_bi_metrics": metrics,
        "xgboost_metrics": xgboost_metrics,
        "comparison": {
            "mape_diff_pp": diff_mape_pp,
            "mape_improvement_pct": melhoria_mape,
            "mae_diff": diff_mae,
            "mae_improvement_pct": melhoria_mae,
            "winner": "Power BI" if metrics["mape"] < xgboost_metrics["mape"] else "XGBoost",
        },
        "test_period": {"start": "Jul/23", "end": "Sep/25", "months": len(df)},
    }

    output_file = (
        Path(__file__).parent / "data" / "processed_data" / "powerbi_comparison_final.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResultados salvos em: {output_file}")

    print("\n" + "=" * 70)
    print("VALORES PARA O QUADRO 6 (com formatacao brasileira):")
    print("=" * 70)
    print(f"Power BI MAE:  {format_brazilian_currency(metrics['mae'])}")
    print(f"Power BI RMSE: {format_brazilian_currency(metrics['rmse'])}")
    print(f"Power BI MAPE: {metrics['mape']:.2f}%")
    print()
    print(f"XGBoost MAE:  {format_brazilian_currency(xgboost_metrics['mae'])}")
    print(f"XGBoost RMSE: {format_brazilian_currency(xgboost_metrics['rmse'])}")
    print(f"XGBoost MAPE: {xgboost_metrics['mape']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
