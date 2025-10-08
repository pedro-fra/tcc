"""
Visualization module for comparing ML forecasts with Power BI baseline.
Generates publication-quality plots for TCC analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from src.visualization.plot_config import get_plot_style

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PowerBIComparisonPlots:
    """
    Generate comparison visualizations between ML and Power BI forecasts.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize plot generator.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(get_plot_style())

    def plot_forecast_comparison(
        self,
        dates: pd.DatetimeIndex,
        actual: Optional[pd.Series],
        powerbi: pd.Series,
        ml_forecast: pd.Series,
        model_name: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot side-by-side comparison of Power BI and ML forecasts.

        Args:
            dates: Date index for x-axis
            actual: Actual values (optional)
            powerbi: Power BI forecast values
            ml_forecast: ML model forecast values
            model_name: Name of ML model
            save_path: Path to save plot
        """
        logger.info(f"Creating forecast comparison plot for {model_name}")

        fig, ax = plt.subplots(figsize=(14, 7))

        if actual is not None:
            ax.plot(
                dates[: len(actual)],
                actual,
                label="Valor Real",
                color="#2C3E50",
                linewidth=2.5,
                marker="o",
                markersize=6,
            )

        ax.plot(
            dates,
            powerbi,
            label="Power BI",
            color="#E74C3C",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=5,
        )

        ax.plot(
            dates,
            ml_forecast,
            label=f"{model_name}",
            color="#3498DB",
            linewidth=2,
            linestyle="-",
            marker="^",
            markersize=5,
        )

        ax.set_xlabel("Data", fontsize=12, fontweight="bold")
        ax.set_ylabel("Valor (R$)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Comparacao de Previsoes: {model_name} vs Power BI",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        ax.legend(fontsize=11, loc="best", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R$ {x / 1e6:.1f}M"))

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"{model_name.lower()}_vs_powerbi_forecast_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Forecast comparison plot saved to {save_path}")
        plt.close()

    def plot_difference_analysis(
        self,
        dates: pd.DatetimeIndex,
        powerbi: pd.Series,
        ml_forecast: pd.Series,
        model_name: str,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Plot difference analysis between forecasts.

        Args:
            dates: Date index
            powerbi: Power BI forecast values
            ml_forecast: ML forecast values
            model_name: Name of ML model
            save_path: Path to save plot
        """
        logger.info(f"Creating difference analysis plot for {model_name}")

        diff = ml_forecast.values - powerbi.values
        pct_diff = (diff / powerbi.values) * 100

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        colors = ["#27AE60" if d > 0 else "#E74C3C" for d in diff]
        ax1.bar(dates, diff, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax1.set_ylabel("Diferenca Absoluta (R$)", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"Diferenca: {model_name} - Power BI",
            fontsize=12,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"R$ {x / 1e6:.1f}M"))

        ax2 = fig.add_subplot(gs[1, :])
        colors = ["#27AE60" if d > 0 else "#E74C3C" for d in pct_diff]
        ax2.bar(dates, pct_diff, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.set_ylabel("Diferenca Percentual (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Diferenca Percentual", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        ax3 = fig.add_subplot(gs[2, 0])
        ax3.hist(pct_diff, bins=20, color="#3498DB", alpha=0.7, edgecolor="black")
        ax3.axvline(
            x=pct_diff.mean(),
            color="#E74C3C",
            linestyle="--",
            linewidth=2,
            label=f"Media: {pct_diff.mean():.1f}%",
        )
        ax3.set_xlabel("Diferenca Percentual (%)", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Frequencia", fontsize=10, fontweight="bold")
        ax3.set_title("Distribuicao das Diferencas", fontsize=11, fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        ax4 = fig.add_subplot(gs[2, 1])
        ax4.scatter(powerbi, ml_forecast, alpha=0.6, s=80, color="#3498DB")
        min_val = min(powerbi.min(), ml_forecast.min())
        max_val = max(powerbi.max(), ml_forecast.max())
        ax4.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.7)
        ax4.set_xlabel("Power BI (R$)", fontsize=10, fontweight="bold")
        ax4.set_ylabel(f"{model_name} (R$)", fontsize=10, fontweight="bold")
        ax4.set_title("Correlacao de Previsoes", fontsize=11, fontweight="bold")
        ax4.grid(True, alpha=0.3)

        correlation = np.corrcoef(powerbi.values, ml_forecast.values)[0, 1]
        ax4.text(
            0.05,
            0.95,
            f"Correlacao: {correlation:.3f}",
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        for ax in [ax1, ax2]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        if save_path is None:
            save_path = self.output_dir / f"{model_name.lower()}_vs_powerbi_difference.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Difference analysis plot saved to {save_path}")
        plt.close()

    def plot_metrics_comparison(
        self, comparison_df: pd.DataFrame, save_path: Optional[Path] = None
    ) -> None:
        """
        Plot metrics comparison across all models.

        Args:
            comparison_df: DataFrame with comparison metrics
            save_path: Path to save plot
        """
        logger.info("Creating metrics comparison plot")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Comparacao de Metricas: Modelos ML vs Power BI",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )

        if "mean_pct_diff" in comparison_df.columns:
            ax = axes[0, 0]
            comparison_df.plot(
                x="model",
                y="mean_pct_diff",
                kind="bar",
                ax=ax,
                color="#3498DB",
                legend=False,
            )
            ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
            ax.set_title("Diferenca Percentual Media", fontweight="bold")
            ax.set_xlabel("Modelo", fontweight="bold")
            ax.set_ylabel("Diferenca (%)", fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        if "correlation" in comparison_df.columns:
            ax = axes[0, 1]
            comparison_df.plot(
                x="model",
                y="correlation",
                kind="bar",
                ax=ax,
                color="#27AE60",
                legend=False,
            )
            ax.set_title("Correlacao com Power BI", fontweight="bold")
            ax.set_xlabel("Modelo", fontweight="bold")
            ax.set_ylabel("Correlacao", fontweight="bold")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        if all(col in comparison_df.columns for col in ["powerbi_mae", "ml_mae"]):
            ax = axes[1, 0]
            x = np.arange(len(comparison_df))
            width = 0.35
            ax.bar(
                x - width / 2,
                comparison_df["powerbi_mae"],
                width,
                label="Power BI",
                color="#E74C3C",
                alpha=0.8,
            )
            ax.bar(
                x + width / 2,
                comparison_df["ml_mae"],
                width,
                label="ML",
                color="#3498DB",
                alpha=0.8,
            )
            ax.set_title("Comparacao de MAE", fontweight="bold")
            ax.set_xlabel("Modelo", fontweight="bold")
            ax.set_ylabel("MAE", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df["model"], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        if "mae_improvement_pct" in comparison_df.columns:
            ax = axes[1, 1]
            colors = [
                "#27AE60" if x > 0 else "#E74C3C" for x in comparison_df["mae_improvement_pct"]
            ]
            comparison_df.plot(
                x="model",
                y="mae_improvement_pct",
                kind="bar",
                ax=ax,
                color=colors,
                legend=False,
            )
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
            ax.set_title("Melhoria em MAE vs Power BI", fontweight="bold")
            ax.set_xlabel("Modelo", fontweight="bold")
            ax.set_ylabel("Melhoria (%)", fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "all_models_metrics_comparison.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Metrics comparison plot saved to {save_path}")
        plt.close()

    def plot_forecast_table(
        self,
        dates: pd.DatetimeIndex,
        actual: Optional[pd.Series],
        powerbi: pd.Series,
        ml_forecasts: Dict[str, pd.Series],
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Create a detailed comparison table plot.

        Args:
            dates: Date index
            actual: Actual values
            powerbi: Power BI forecasts
            ml_forecasts: Dictionary of ML model forecasts
            save_path: Path to save plot
        """
        logger.info("Creating forecast comparison table")

        data = {"Data": dates, "Power BI": powerbi}

        if actual is not None:
            data["Valor Real"] = actual

        for model_name, forecast in ml_forecasts.items():
            data[model_name] = forecast

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis("tight")
        ax.axis("off")

        table_data = []
        for _, row in df.iterrows():
            formatted_row = [row["Data"].strftime("%Y-%m")]
            for col in df.columns[1:]:
                val = row[col]
                if pd.notna(val):
                    formatted_row.append(f"R$ {val / 1e6:.2f}M")
                else:
                    formatted_row.append("-")
            table_data.append(formatted_row)

        headers = ["Data"] + list(df.columns[1:])

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#3498DB")
            table[(0, i)].set_text_props(weight="bold", color="white")

        for i in range(1, len(table_data) + 1):
            if i % 2 == 0:
                for j in range(len(headers)):
                    table[(i, j)].set_facecolor("#ECF0F1")

        plt.title(
            "Tabela Comparativa de Previsoes",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        if save_path is None:
            save_path = self.output_dir / "forecast_comparison_table.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Forecast table saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    print("Power BI comparison plots module loaded successfully!")
