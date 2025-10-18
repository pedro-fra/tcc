"""
Modulo de comparacao entre previsoes do XGBoost e Power BI

Este modulo realiza a analise comparativa entre as previsoes do modelo
XGBoost e as projecoes calculadas pelo Power BI para o periodo de teste.

Funcoes principais:
- compare_models: Funcao principal de comparacao
- load_xgboost_predictions: Carrega previsoes do XGBoost
- load_powerbi_data: Carrega dados exportados do Power BI
- calculate_metrics: Calcula metricas de comparacao
- generate_plots: Gera visualizacoes comparativas
"""

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class XGBoostPowerBIComparator:
    """Classe para comparacao entre XGBoost e Power BI"""

    def __init__(self, processed_data_dir: Path):
        """
        Inicializa o comparador

        Args:
            processed_data_dir: Diretorio com dados processados
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.xgboost_results = None
        self.powerbi_data = None
        self.comparison_results = {}
        self.plots_dir = (
            Path(__file__).parent.parent.parent / "data" / "plots" / "powerbi_comparison"
        )
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def load_xgboost_predictions(self) -> pd.DataFrame:
        """
        Carrega previsoes do XGBoost do periodo de teste

        Returns:
            DataFrame com previsoes do XGBoost
        """
        logger.info("Carregando previsoes do XGBoost")

        xgboost_dir = self.processed_data_dir / "xgboost"
        if not xgboost_dir.exists():
            raise FileNotFoundError(f"Diretorio XGBoost nao encontrado: {xgboost_dir}")

        summary_file = xgboost_dir / "xgboost_model_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError(f"Arquivo de resumo XGBoost nao encontrado: {summary_file}")

        with open(summary_file, "r") as f:
            summary = json.load(f)

        logger.info(f"Metricas XGBoost carregadas: MAE={summary['metrics']['mae']:.2f}")

        self.xgboost_results = summary
        return summary

    def load_powerbi_data(self, csv_path: str) -> pd.DataFrame:
        """
        Carrega dados historicos exportados do Power BI

        Args:
            csv_path: Caminho para o CSV exportado do Power BI

        Returns:
            DataFrame com dados do Power BI
        """
        logger.info(f"Carregando dados do Power BI de {csv_path}")

        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Arquivo Power BI nao encontrado: {csv_file}")

        self.powerbi_data = pd.read_csv(csv_file, parse_dates=["Data"])
        logger.info(f"Dados Power BI carregados: {len(self.powerbi_data)} registros")

        return self.powerbi_data

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calcula metricas de erro para as previsoes

        Args:
            actual: Valores reais
            predicted: Valores previstos

        Returns:
            Dicionario com metricas MAE, RMSE, MAPE
        """
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        return {"mae": mae, "rmse": rmse, "mape": mape}

    def generate_forecast_comparison_plot(self, df: pd.DataFrame) -> None:
        """
        Gera grafico de comparacao de previsoes ao longo do tempo

        Args:
            df: DataFrame com dados comparativos
        """
        logger.info("Gerando grafico de comparacao de previsoes")

        fig, ax = plt.subplots(figsize=(14, 7))

        if "Data" in df.columns:
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
        else:
            ax.plot(df.index, df["Faturamento_Real"], marker="o", label="Faturamento Real")
            ax.plot(df.index, df["XGBoost_Previsao"], marker="s", label="XGBoost")
            ax.plot(df.index, df["PowerBI_Projecao"], marker="^", label="Power BI")

        ax.set_xlabel("Periodo")
        ax.set_ylabel("Faturamento (R$)")
        ax.set_title("Comparacao: XGBoost vs Power BI vs Faturamento Real")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_file = self.plots_dir / "xgboost_vs_powerbi_forecast.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Grafico de previsoes salvo em {output_file}")
        plt.close()

    def generate_metrics_comparison_plot(
        self, xgboost_metrics: Dict, powerbi_metrics: Dict
    ) -> None:
        """
        Gera grafico comparativo de metricas

        Args:
            xgboost_metrics: Metricas do XGBoost
            powerbi_metrics: Metricas do Power BI
        """
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

        output_file = self.plots_dir / "xgboost_vs_powerbi_metrics.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Grafico de metricas salvo em {output_file}")
        plt.close()

    def generate_difference_plot(self, df: pd.DataFrame) -> None:
        """
        Gera grafico de diferenca entre XGBoost e Power BI

        Args:
            df: DataFrame com dados comparativos
        """
        logger.info("Gerando grafico de diferenca")

        df["Diferenca"] = df["XGBoost_Previsao"] - df["PowerBI_Projecao"]
        df["Diferenca_Pct"] = (df["Diferenca"] / df["PowerBI_Projecao"]) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        if "Data" in df.columns:
            ax1.bar(
                df["Data"],
                df["Diferenca"],
                color=["blue" if x > 0 else "red" for x in df["Diferenca"]],
            )
            ax1.set_ylabel("Diferenca (R$)")
        else:
            ax1.bar(
                df.index,
                df["Diferenca"],
                color=["blue" if x > 0 else "red" for x in df["Diferenca"]],
            )

        ax1.set_title("Diferenca Absoluta: XGBoost - Power BI")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis="y")

        if "Data" in df.columns:
            ax2.bar(
                df["Data"],
                df["Diferenca_Pct"],
                color=["blue" if x > 0 else "red" for x in df["Diferenca_Pct"]],
            )
            ax2.set_xlabel("Periodo")
        else:
            ax2.bar(
                df.index,
                df["Diferenca_Pct"],
                color=["blue" if x > 0 else "red" for x in df["Diferenca_Pct"]],
            )

        ax2.set_ylabel("Diferenca (%)")
        ax2.set_title("Diferenca Percentual: (XGBoost - Power BI) / Power BI")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45)
        plt.tight_layout()

        output_file = self.plots_dir / "xgboost_vs_powerbi_difference.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Grafico de diferenca salvo em {output_file}")
        plt.close()

    def generate_scatter_plot(self, df: pd.DataFrame) -> None:
        """
        Gera scatter plot comparando XGBoost vs Power BI

        Args:
            df: DataFrame com dados comparativos
        """
        logger.info("Gerando scatter plot de comparacao")

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(
            df["PowerBI_Projecao"],
            df["XGBoost_Previsao"],
            alpha=0.6,
            s=100,
            color="blue",
        )

        min_val = min(df["PowerBI_Projecao"].min(), df["XGBoost_Previsao"].min())
        max_val = max(df["PowerBI_Projecao"].max(), df["XGBoost_Previsao"].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Linha de Igualdade")

        correlation = df["PowerBI_Projecao"].corr(df["XGBoost_Previsao"])
        ax.text(
            0.05,
            0.95,
            f"Correlacao: {correlation:.4f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_xlabel("Power BI (R$)")
        ax.set_ylabel("XGBoost (R$)")
        ax.set_title("Scatter Plot: Power BI vs XGBoost")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_file = self.plots_dir / "xgboost_vs_powerbi_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Scatter plot salvo em {output_file}")
        plt.close()

    def generate_summary_report(self) -> Dict:
        """
        Gera relatorio resumido da comparacao

        Returns:
            Dicionario com resumo da analise
        """
        logger.info("Gerando relatorio resumido")

        summary = {
            "periodo_analise": {
                "inicio": self.powerbi_data["Data"].min().strftime("%Y-%m-%d")
                if "Data" in self.powerbi_data.columns
                else "N/A",
                "fim": self.powerbi_data["Data"].max().strftime("%Y-%m-%d")
                if "Data" in self.powerbi_data.columns
                else "N/A",
                "total_meses": len(self.powerbi_data),
            },
            "xgboost_metricas": self.xgboost_results["metrics"],
            "powerbi_metricas": self.comparison_results.get("powerbi_metrics", {}),
            "comparacao": {
                "melhor_modelo": "XGBoost"
                if self.xgboost_results["metrics"]["mape"]
                < self.comparison_results.get("powerbi_metrics", {}).get("mape", float("inf"))
                else "Power BI",
                "diferenca_mape_pct": abs(
                    self.xgboost_results["metrics"]["mape"]
                    - self.comparison_results.get("powerbi_metrics", {}).get("mape", 0)
                ),
            },
        }

        return summary


def compare_models(powerbi_csv_path: str, processed_data_dir: Path = None) -> Dict:
    """
    Funcao principal para comparacao entre XGBoost e Power BI

    Args:
        powerbi_csv_path: Caminho para CSV exportado do Power BI
        processed_data_dir: Diretorio com dados processados

    Returns:
        Dicionario com resultados da comparacao
    """
    if processed_data_dir is None:
        processed_data_dir = Path(__file__).parent.parent.parent / "data" / "processed_data"

    logger.info("Iniciando comparacao XGBoost vs Power BI")

    comparator = XGBoostPowerBIComparator(processed_data_dir)

    xgboost_results = comparator.load_xgboost_predictions()
    powerbi_data = comparator.load_powerbi_data(powerbi_csv_path)

    if (
        "TCC Faturamento Realizado" in powerbi_data.columns
        and "TCC Faturamento Projetado Power BI" in powerbi_data.columns
    ):
        actual = powerbi_data["TCC Faturamento Realizado"].values
        powerbi_forecast = powerbi_data["TCC Faturamento Projetado Power BI"].values

        powerbi_metrics = comparator.calculate_metrics(actual, powerbi_forecast)
        comparator.comparison_results["powerbi_metrics"] = powerbi_metrics

        logger.info(f"Metricas Power BI calculadas: MAPE={powerbi_metrics['mape']:.2f}%")

        powerbi_data_renamed = powerbi_data.copy()
        powerbi_data_renamed.columns = ["Data", "Faturamento_Real", "PowerBI_Projecao"]
        comparator.generate_forecast_comparison_plot(powerbi_data_renamed)
        comparator.generate_difference_plot(powerbi_data_renamed)

        comparator.generate_metrics_comparison_plot(xgboost_results["metrics"], powerbi_metrics)

        summary = comparator.generate_summary_report()

        comparator.comparison_results["summary"] = summary

        logger.info("Comparacao concluida com sucesso")

        return comparator.comparison_results

    else:
        logger.warning("Colunas esperadas nao encontradas no CSV do Power BI")
        return {}
