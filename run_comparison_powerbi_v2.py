"""
Script de execucao da comparacao entre XGBoost e Power BI (Versao 2)

Este script eh o ponto de entrada para realizar a analise comparativa
entre as previsoes do modelo XGBoost e as projecoes do Power BI.

Uso:
    uv run python run_comparison_powerbi_v2.py <caminho_csv_powerbi>

Exemplo:
    uv run python run_comparison_powerbi_v2.py data/powerbi_historico_test_period.csv
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))

from config import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_xgboost_summary() -> dict:
    """Carrega o resumo do modelo XGBoost"""
    summary_file = PROCESSED_DATA_DIR / "xgboost" / "xgboost_model_summary.json"

    if not summary_file.exists():
        raise FileNotFoundError(f"Arquivo XGBoost nao encontrado: {summary_file}")

    with open(summary_file, "r") as f:
        return json.load(f)


def load_powerbi_data(csv_path: str) -> pd.DataFrame:
    """Carrega dados do Power BI"""
    df = pd.read_csv(csv_path, parse_dates=["Data"])
    logger.info(f"Dados Power BI carregados: {len(df)} registros")
    return df


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calcula metricas de erro"""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape}


def generate_forecast_comparison_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Gera grafico de comparacao de previsoes"""
    logger.info("Gerando grafico de comparacao de previsoes")

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        df["Data"],
        df["Faturamento_Real"],
        marker="o",
        label="Faturamento Real",
        linewidth=2,
        color="black",
    )
    ax.plot(
        df["Data"],
        df["XGBoost_Previsao"],
        marker="s",
        label="XGBoost",
        linewidth=2,
        color="blue",
        linestyle="--",
    )
    ax.plot(
        df["Data"],
        df["PowerBI_Projecao"],
        marker="^",
        label="Power BI",
        linewidth=2,
        color="green",
        linestyle="--",
    )

    ax.set_xlabel("Periodo")
    ax.set_ylabel("Faturamento (R$)")
    ax.set_title("Comparacao: XGBoost vs Power BI vs Faturamento Real")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_file = output_dir / "xgboost_vs_powerbi_forecast.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Grafico salvo em {output_file}")
    plt.close()


def generate_metrics_comparison_plot(
    xgboost_metrics: dict, powerbi_metrics: dict, output_dir: Path
) -> None:
    """Gera grafico comparativo de metricas"""
    logger.info("Gerando grafico de comparacao de metricas")

    metrics_names = ["MAE", "RMSE", "MAPE"]
    xgboost_values = [
        xgboost_metrics["mae"],
        xgboost_metrics["rmse"],
        xgboost_metrics["mape"],
    ]
    powerbi_values = [
        powerbi_metrics["mae"],
        powerbi_metrics["rmse"],
        powerbi_metrics["mape"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, xgboost_values, width, label="XGBoost", color="blue")
    ax.bar(x + width / 2, powerbi_values, width, label="Power BI", color="green")

    ax.set_xlabel("Metricas")
    ax.set_ylabel("Valor")
    ax.set_title("Comparacao de Metricas: XGBoost vs Power BI")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    output_file = output_dir / "xgboost_vs_powerbi_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Grafico salvo em {output_file}")
    plt.close()


def generate_markdown_report(
    powerbi_metrics: dict, xgboost_metrics: dict, summary_data: dict, output_file: str
) -> None:
    """Gera relatorio em markdown"""
    logger.info(f"Gerando relatorio em {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Comparacao: XGBoost vs Power BI\n\n")

        f.write("## Periodo de Analise\n\n")
        f.write(f"- **Inicio**: {summary_data['inicio']}\n")
        f.write(f"- **Fim**: {summary_data['fim']}\n")
        f.write(f"- **Total de meses**: {summary_data['total_meses']}\n\n")

        f.write("## Metricas de Desempenho\n\n")

        f.write("### XGBoost\n\n")
        f.write("| Metrica | Valor |\n")
        f.write("|---------|-------|\n")
        f.write(f"| MAE | R$ {xgboost_metrics['mae']:,.2f} |\n")
        f.write(f"| RMSE | R$ {xgboost_metrics['rmse']:,.2f} |\n")
        f.write(f"| MAPE | {xgboost_metrics['mape']:.2f}% |\n\n")

        f.write("### Power BI\n\n")
        f.write("| Metrica | Valor |\n")
        f.write("|---------|-------|\n")
        f.write(f"| MAE | R$ {powerbi_metrics['mae']:,.2f} |\n")
        f.write(f"| RMSE | R$ {powerbi_metrics['rmse']:,.2f} |\n")
        f.write(f"| MAPE | {powerbi_metrics['mape']:.2f}% |\n\n")

        f.write("## Conclusao\n\n")

        melhor_modelo = (
            "XGBoost" if xgboost_metrics["mape"] < powerbi_metrics["mape"] else "Power BI"
        )
        diferenca = abs(xgboost_metrics["mape"] - powerbi_metrics["mape"])

        f.write(f"**Melhor Modelo**: {melhor_modelo}\n\n")
        f.write(f"**Diferenca MAPE**: {diferenca:.2f} pontos percentuais\n\n")

        f.write("## Visualizacoes Geradas\n\n")
        f.write("As seguintes visualizacoes foram geradas em `data/plots/powerbi_comparison/`:\n\n")
        f.write("1. `xgboost_vs_powerbi_forecast.png` - Comparacao temporal\n")
        f.write("2. `xgboost_vs_powerbi_metrics.png` - Comparacao de metricas\n")


def main():
    """Funcao principal"""
    logger.info("Iniciando script de comparacao XGBoost vs Power BI")

    if len(sys.argv) < 2:
        logger.error("Uso: uv run python run_comparison_powerbi_v2.py <caminho_csv_powerbi>")
        sys.exit(1)

    powerbi_csv = sys.argv[1]

    try:
        xgboost_summary = load_xgboost_summary()
        powerbi_df = load_powerbi_data(powerbi_csv)

        xgboost_metrics = xgboost_summary["metrics"]
        actual = powerbi_df["TCC Faturamento Realizado"].values
        powerbi_forecast = powerbi_df["TCC Faturamento Projetado Power BI"].values

        powerbi_metrics = calculate_metrics(actual, powerbi_forecast)

        plots_dir = Path(__file__).parent / "data" / "plots" / "powerbi_comparison"
        plots_dir.mkdir(parents=True, exist_ok=True)

        summary_data = {
            "inicio": powerbi_df["Data"].min().strftime("%Y-%m-%d"),
            "fim": powerbi_df["Data"].max().strftime("%Y-%m-%d"),
            "total_meses": len(powerbi_df),
        }

        report_file = Path(__file__).parent / "COMPARACAO_XGBOOST_POWERBI.md"
        generate_markdown_report(powerbi_metrics, xgboost_metrics, summary_data, str(report_file))

        generate_metrics_comparison_plot(xgboost_metrics, powerbi_metrics, plots_dir)

        print("\n" + "=" * 70)
        print("COMPARACAO CONCLUIDA COM SUCESSO!")
        print("=" * 70)

        print(f"\nPeriodo de Analise: {summary_data['inicio']} a {summary_data['fim']}")
        print(f"Total de Meses: {summary_data['total_meses']}")

        print("\n>>> XGBoost Metricas:")
        print(f"    MAE:  R$ {xgboost_metrics['mae']:,.2f}")
        print(f"    RMSE: R$ {xgboost_metrics['rmse']:,.2f}")
        print(f"    MAPE: {xgboost_metrics['mape']:.2f}%")

        print("\n>>> Power BI Metricas:")
        print(f"    MAE:  R$ {powerbi_metrics['mae']:,.2f}")
        print(f"    RMSE: R$ {powerbi_metrics['rmse']:,.2f}")
        print(f"    MAPE: {powerbi_metrics['mape']:.2f}%")

        melhor = "XGBoost" if xgboost_metrics["mape"] < powerbi_metrics["mape"] else "Power BI"
        diferenca = abs(xgboost_metrics["mape"] - powerbi_metrics["mape"])

        print("\n>>> Conclusao:")
        print(f"    Melhor Modelo: {melhor}")
        print(f"    Diferenca MAPE: {diferenca:.2f} pp")

        print("\n>>> Arquivos Gerados:")
        print(f"    - Relatorio: {report_file}")
        print(f"    - Graficos: {plots_dir}")

        print("\n" + "=" * 70)

    except Exception as e:
        logger.error(f"Erro durante comparacao: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
