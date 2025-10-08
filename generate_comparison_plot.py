"""
Script para gerar grafico comparativo de previsoes de todos os modelos.
Visualiza previsoes de ARIMA, Theta, Exponential Smoothing e XGBoost em um unico grafico.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import DATA_DIR, PROCESSED_DATA_DIR


def load_model_predictions():
    """Carrega predicoes de todos os modelos executando pipeline completo."""
    from config import load_config
    from models.arima_model import ArimaForecaster
    from models.exponential_smoothing_model import ExponentialSmoothingForecaster
    from models.theta import ThetaForecaster
    from models.xgboost_model import XGBoostForecaster

    config = load_config()
    predictions = {}

    raw_data_file = "data/fat_factory.csv"
    time_series_data_dir = PROCESSED_DATA_DIR / "model_data" / "time_series"

    print("Executando modelos para obter predicoes...")

    # ARIMA
    print("  - ARIMA...")
    try:
        arima_forecaster = ArimaForecaster(config)
        arima_output_dir = PROCESSED_DATA_DIR / "arima"
        arima_results = arima_forecaster.run_complete_pipeline(
            data_path=time_series_data_dir, output_dir=arima_output_dir
        )
        if arima_results["success"]:
            predictions["ARIMA"] = pd.DataFrame(
                {
                    "actual": arima_forecaster.test_series.values().flatten(),
                    "predicted": arima_forecaster.predictions.values().flatten(),
                },
                index=arima_forecaster.test_series.time_index,
            )
    except Exception as e:
        print(f"    Erro: {e}")

    # Theta
    print("  - Theta...")
    try:
        theta_forecaster = ThetaForecaster(config)
        theta_results = theta_forecaster.run_complete_pipeline(raw_data_file)
        if "metrics" in theta_results:
            predictions["Theta"] = pd.DataFrame(
                {
                    "actual": theta_forecaster.test_series.values().flatten(),
                    "predicted": theta_forecaster.predictions.values().flatten(),
                },
                index=theta_forecaster.test_series.time_index,
            )
    except Exception as e:
        print(f"    Erro: {e}")

    # Exponential Smoothing
    print("  - Exponential Smoothing...")
    try:
        exp_forecaster = ExponentialSmoothingForecaster(config)
        exp_output_dir = PROCESSED_DATA_DIR / "exponential_smoothing"
        exp_results = exp_forecaster.run_complete_pipeline(
            data_path=time_series_data_dir, output_dir=exp_output_dir
        )
        if exp_results["success"]:
            predictions["Exponential Smoothing"] = pd.DataFrame(
                {
                    "actual": exp_forecaster.test_series.values().flatten(),
                    "predicted": exp_forecaster.predictions.values().flatten(),
                },
                index=exp_forecaster.test_series.time_index,
            )
    except Exception as e:
        print(f"    Erro: {e}")

    # XGBoost
    print("  - XGBoost...")
    try:
        xgb_forecaster = XGBoostForecaster(config)
        xgb_results = xgb_forecaster.run_complete_pipeline(raw_data_file)
        if "metrics" in xgb_results:
            predictions["XGBoost"] = pd.DataFrame(
                {
                    "actual": xgb_forecaster.y_test.values,
                    "predicted": xgb_forecaster.predictions,
                },
                index=xgb_forecaster.X_test.index,
            )
    except Exception as e:
        print(f"    Erro: {e}")

    return predictions


def create_comparison_plot(predictions):
    """Cria grafico comparativo de todos os modelos."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Cores distintas para cada modelo
    colors = {
        "ARIMA": "#E74C3C",  # Vermelho
        "Theta": "#3498DB",  # Azul
        "Exponential Smoothing": "#2ECC71",  # Verde
        "XGBoost": "#9B59B6",  # Roxo
    }

    # Estilos de linha
    linestyles = {
        "ARIMA": "--",
        "Theta": "-.",
        "Exponential Smoothing": ":",
        "XGBoost": "-",
    }

    # Plotar valores reais (pegar do primeiro modelo disponivel)
    first_model = list(predictions.values())[0]
    if "actual" in first_model.columns:
        ax.plot(
            first_model.index,
            first_model["actual"],
            label="Valores Reais",
            color="black",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.8,
        )

    # Plotar previsoes de cada modelo
    for model_name, df in predictions.items():
        if "predicted" in df.columns:
            ax.plot(
                df.index,
                df["predicted"],
                label=f"{model_name}",
                color=colors[model_name],
                linewidth=2,
                linestyle=linestyles[model_name],
                marker="s",
                markersize=5,
                alpha=0.7,
            )

    # Configuracoes do grafico
    ax.set_xlabel("Periodo", fontsize=14, fontweight="bold")
    ax.set_ylabel("Vendas (R$)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Comparacao de Previsoes - Todos os Modelos\nDataset: Vendas Mensais (2023-2025)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legenda
    ax.legend(
        loc="upper left",
        fontsize=12,
        frameon=True,
        shadow=True,
        fancybox=True,
        ncol=1,
    )

    # Formatar eixo Y com separador de milhares
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Rotacionar labels do eixo X
    plt.xticks(rotation=45, ha="right")

    # Ajustar layout
    plt.tight_layout()

    return fig


def create_detailed_comparison_plot(predictions):
    """Cria grafico com subplot de erros tambem."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

    colors = {
        "ARIMA": "#E74C3C",
        "Theta": "#3498DB",
        "Exponential Smoothing": "#2ECC71",
        "XGBoost": "#9B59B6",
    }

    linestyles = {"ARIMA": "--", "Theta": "-.", "Exponential Smoothing": ":", "XGBoost": "-"}

    # Subplot 1: Previsoes
    first_model = list(predictions.values())[0]
    if "actual" in first_model.columns:
        ax1.plot(
            first_model.index,
            first_model["actual"],
            label="Valores Reais",
            color="black",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.8,
        )

    for model_name, df in predictions.items():
        if "predicted" in df.columns:
            ax1.plot(
                df.index,
                df["predicted"],
                label=f"{model_name}",
                color=colors[model_name],
                linewidth=2,
                linestyle=linestyles[model_name],
                marker="s",
                markersize=5,
                alpha=0.7,
            )

    ax1.set_ylabel("Vendas (R$)", fontsize=14, fontweight="bold")
    ax1.set_title(
        "Comparacao de Previsoes - Todos os Modelos",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.legend(loc="upper left", fontsize=11, frameon=True, shadow=True, ncol=2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # Subplot 2: Erros percentuais
    for model_name, df in predictions.items():
        if "actual" in df.columns and "predicted" in df.columns:
            errors = ((df["predicted"] - df["actual"]) / df["actual"]) * 100
            ax2.plot(
                df.index,
                errors,
                label=f"{model_name}",
                color=colors[model_name],
                linewidth=2,
                linestyle=linestyles[model_name],
                marker="s",
                markersize=4,
                alpha=0.7,
            )

    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Periodo", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Erro Percentual (%)", fontsize=14, fontweight="bold")
    ax2.set_title("Erros de Previsao por Modelo", fontsize=14, fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.legend(loc="best", fontsize=10, frameon=True, shadow=True, ncol=2)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def main():
    """Executa geracao do grafico comparativo."""
    print("=" * 80)
    print("GERANDO GRAFICO COMPARATIVO DE PREVISOES")
    print("=" * 80)

    # Carregar predicoes
    print("\nCarregando predicoes dos modelos...")
    predictions = load_model_predictions()

    if not predictions:
        print("ERRO: Nenhuma predicao encontrada!")
        print(
            "Execute os modelos primeiro (run_arima_forecasting.py, etc.) ou compare_models_simple.py"
        )
        return

    print(f"Modelos encontrados: {', '.join(predictions.keys())}")

    # Criar diretorio de plots se nao existir
    plots_dir = DATA_DIR / "plots" / "comparison"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Gerar grafico simples
    print("\nGerando grafico comparativo simples...")
    fig1 = create_comparison_plot(predictions)
    output_file1 = plots_dir / "all_models_comparison.png"
    fig1.savefig(output_file1, dpi=300, bbox_inches="tight")
    print(f"Salvo: {output_file1}")

    # Gerar grafico detalhado com erros
    print("\nGerando grafico comparativo detalhado com erros...")
    fig2 = create_detailed_comparison_plot(predictions)
    output_file2 = plots_dir / "all_models_comparison_detailed.png"
    fig2.savefig(output_file2, dpi=300, bbox_inches="tight")
    print(f"Salvo: {output_file2}")

    print("\n" + "=" * 80)
    print("GRAFICOS GERADOS COM SUCESSO!")
    print("=" * 80)
    print(f"\nDiretorio: {plots_dir}")
    print("\nArquivos gerados:")
    print("  1. all_models_comparison.png - Grafico simples de previsoes")
    print("  2. all_models_comparison_detailed.png - Grafico com subplot de erros")

    # Calcular e mostrar estatisticas
    print("\n" + "=" * 80)
    print("ESTATISTICAS DE PREVISAO")
    print("=" * 80)

    for model_name, df in predictions.items():
        if "actual" in df.columns and "predicted" in df.columns:
            mae = (df["predicted"] - df["actual"]).abs().mean()
            mape = ((df["predicted"] - df["actual"]).abs() / df["actual"]).mean() * 100
            print(f"\n{model_name}:")
            print(f"  MAE: R$ {mae:,.2f}")
            print(f"  MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()
