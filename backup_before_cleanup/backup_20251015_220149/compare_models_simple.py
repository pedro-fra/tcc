"""
Script simplificado para comparar todos os modelos de previsao.
Usa a API run_complete_pipeline() de cada modelo.
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import PROCESSED_DATA_DIR, load_config
from models.arima_model import ArimaForecaster
from models.exponential_smoothing_model import ExponentialSmoothingForecaster
from models.theta import ThetaForecaster
from models.xgboost_model import XGBoostForecaster


def main():
    print("=" * 80)
    print("COMPARACAO COMPLETA DE MODELOS DE PREVISAO DE VENDAS")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Prepare data paths
    time_series_data_dir = PROCESSED_DATA_DIR / "model_data" / "time_series"
    raw_data_file = "data/fat_factory.csv"

    all_results = {}

    # Test ARIMA
    print("\n" + "=" * 80)
    print("1. AVALIANDO ARIMA")
    print("=" * 80)
    try:
        arima_forecaster = ArimaForecaster(config)
        arima_output_dir = PROCESSED_DATA_DIR / "arima_comparison"
        arima_results = arima_forecaster.run_complete_pipeline(
            data_path=time_series_data_dir, output_dir=arima_output_dir
        )
        if arima_results["success"]:
            all_results["ARIMA"] = {
                "MAE": arima_results["metrics"]["mae"],
                "RMSE": arima_results["metrics"]["rmse"],
                "MAPE": arima_results["metrics"]["mape"],
            }
            print(f"MAE: {arima_results['metrics']['mae']:,.2f}")
            print(f"RMSE: {arima_results['metrics']['rmse']:,.2f}")
            print(f"MAPE: {arima_results['metrics']['mape']:.2f}%")
        else:
            print(f"Erro: {arima_results.get('error', 'Desconhecido')}")
    except Exception as e:
        print(f"Erro ao executar ARIMA: {e}")

    # Test Theta
    print("\n" + "=" * 80)
    print("2. AVALIANDO THETA METHOD")
    print("=" * 80)
    try:
        theta_forecaster = ThetaForecaster(config)
        theta_results = theta_forecaster.run_complete_pipeline(raw_data_file)
        if "metrics" in theta_results:
            all_results["Theta"] = {
                "MAE": theta_results["metrics"]["mae"],
                "RMSE": theta_results["metrics"]["rmse"],
                "MAPE": theta_results["metrics"]["mape"],
            }
            print(f"MAE: {theta_results['metrics']['mae']:,.2f}")
            print(f"RMSE: {theta_results['metrics']['rmse']:,.2f}")
            print(f"MAPE: {theta_results['metrics']['mape']:.2f}%")
        else:
            print("Erro: Estrutura de retorno inesperada")
    except Exception as e:
        print(f"Erro ao executar Theta: {e}")

    # Test Exponential Smoothing
    print("\n" + "=" * 80)
    print("3. AVALIANDO EXPONENTIAL SMOOTHING")
    print("=" * 80)
    try:
        exp_forecaster = ExponentialSmoothingForecaster(config)
        exp_output_dir = PROCESSED_DATA_DIR / "exponential_comparison"
        exp_results = exp_forecaster.run_complete_pipeline(
            data_path=time_series_data_dir, output_dir=exp_output_dir
        )
        if exp_results["success"]:
            all_results["Exponential Smoothing"] = {
                "MAE": exp_results["metrics"]["mae"],
                "RMSE": exp_results["metrics"]["rmse"],
                "MAPE": exp_results["metrics"]["mape"],
            }
            print(f"MAE: {exp_results['metrics']['mae']:,.2f}")
            print(f"RMSE: {exp_results['metrics']['rmse']:,.2f}")
            print(f"MAPE: {exp_results['metrics']['mape']:.2f}%")
        else:
            print(f"Erro: {exp_results.get('error', 'Desconhecido')}")
    except Exception as e:
        print(f"Erro ao executar Exponential Smoothing: {e}")

    # Test XGBoost
    print("\n" + "=" * 80)
    print("4. AVALIANDO XGBOOST (45 features aprimoradas)")
    print("=" * 80)
    try:
        xgb_forecaster = XGBoostForecaster(config)
        xgb_results = xgb_forecaster.run_complete_pipeline(raw_data_file)
        if "metrics" in xgb_results:
            all_results["XGBoost"] = {
                "MAE": xgb_results["metrics"]["mae"],
                "RMSE": xgb_results["metrics"]["rmse"],
                "MAPE": xgb_results["metrics"]["mape"],
            }
            print(f"MAE: {xgb_results['metrics']['mae']:,.2f}")
            print(f"RMSE: {xgb_results['metrics']['rmse']:,.2f}")
            print(f"MAPE: {xgb_results['metrics']['mape']:.2f}%")
        else:
            print("Erro: Estrutura de retorno inesperada")
    except Exception as e:
        print(f"Erro ao executar XGBoost: {e}")

    # Create comparison table
    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA DE RESULTADOS")
    print("=" * 80)

    if all_results:
        comparison_df = pd.DataFrame(all_results).T
        comparison_df = comparison_df.round(2)

        # Add rankings
        comparison_df["Rank_MAE"] = comparison_df["MAE"].rank()
        comparison_df["Rank_RMSE"] = comparison_df["RMSE"].rank()
        comparison_df["Rank_MAPE"] = comparison_df["MAPE"].rank()
        comparison_df["Rank_Medio"] = (
            comparison_df["Rank_MAE"] + comparison_df["Rank_RMSE"] + comparison_df["Rank_MAPE"]
        ) / 3

        comparison_df = comparison_df.sort_values("Rank_Medio")

        print("\n" + comparison_df.to_string())

        # Save results
        save_comparison_results(comparison_df, all_results)

        # Print best model
        print("\n" + "=" * 80)
        print("MELHOR MODELO")
        print("=" * 80)
        best_model = comparison_df.index[0]
        best_metrics = all_results[best_model]
        print(f"\nModelo: {best_model}")
        print(f"MAE: R$ {best_metrics['MAE']:,.2f}")
        print(f"RMSE: R$ {best_metrics['RMSE']:,.2f}")
        print(f"MAPE: {best_metrics['MAPE']:.2f}%")
        print(f"Rank Medio: {comparison_df.loc[best_model, 'Rank_Medio']:.2f}")
    else:
        print("\nNenhum modelo foi executado com sucesso.")


def save_comparison_results(comparison_df: pd.DataFrame, all_results: dict):
    """Salva resultados da comparacao em arquivo markdown."""
    output_file = Path("MODEL_COMPARISON_RESULTS.md")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Comparacao de Modelos de Previsao de Vendas\n\n")
        f.write("## Resumo Executivo\n\n")

        best_model = comparison_df.index[0]
        best_mae = comparison_df.loc[best_model, "MAE"]
        f.write(f"**Melhor Modelo:** {best_model}\n\n")
        f.write(f"**MAE (Melhor):** R$ {best_mae:,.2f}\n\n")

        f.write("## Tabela Comparativa\n\n")
        f.write(
            "| Modelo | MAE | RMSE | MAPE (%) | Rank MAE | Rank RMSE | Rank MAPE | Rank Medio |\n"
        )
        f.write(
            "|--------|-----|------|----------|----------|-----------|-----------|------------|\n"
        )

        for model_name in comparison_df.index:
            row = comparison_df.loc[model_name]
            f.write(
                f"| {model_name} | "
                f"{row['MAE']:,.2f} | "
                f"{row['RMSE']:,.2f} | "
                f"{row['MAPE']:.2f} | "
                f"{row['Rank_MAE']:.0f} | "
                f"{row['Rank_RMSE']:.0f} | "
                f"{row['Rank_MAPE']:.0f} | "
                f"{row['Rank_Medio']:.2f} |\n"
            )

        f.write("\n## Interpretacao das Metricas\n\n")
        f.write("### MAE (Mean Absolute Error)\n")
        f.write("- Metrica principal para previsao de vendas\n")
        f.write("- Representa o erro medio em reais das previsoes\n")
        f.write("- Menor valor indica melhor desempenho\n\n")

        f.write("### RMSE (Root Mean Squared Error)\n")
        f.write("- Penaliza erros grandes mais severamente que MAE\n")
        f.write("- Util quando erros grandes sao criticos para o negocio\n")
        f.write("- Menor valor indica melhor desempenho\n\n")

        f.write("### MAPE (Mean Absolute Percentage Error)\n")
        f.write("- Erro percentual medio\n")
        f.write("- Facilita comparacao entre diferentes escalas\n")
        f.write("- Menor valor indica melhor desempenho\n\n")

        f.write("## Analise por Modelo\n\n")

        for i, model_name in enumerate(comparison_df.index, 1):
            row = comparison_df.loc[model_name]
            f.write(f"### {i}. {model_name}\n\n")
            f.write(f"- **MAE:** R$ {row['MAE']:,.2f}\n")
            f.write(f"- **RMSE:** R$ {row['RMSE']:,.2f}\n")
            f.write(f"- **MAPE:** {row['MAPE']:.2f}%\n")
            f.write(f"- **Rank Medio:** {row['Rank_Medio']:.2f}\n\n")

        f.write("## Conclusao\n\n")
        f.write(f"O modelo **{best_model}** apresentou o melhor desempenho geral ")
        f.write(f"com MAE de R$ {best_mae:,.2f}.\n\n")

        f.write("## Proximos Passos\n\n")
        f.write("1. Comparar resultados com baseline do Power BI\n")
        f.write("2. Avaliar viabilidade de ensemble dos melhores modelos\n")
        f.write("3. Validar resultados com stakeholders\n")

    print(f"\nResultados salvos em: {output_file}")


if __name__ == "__main__":
    main()
