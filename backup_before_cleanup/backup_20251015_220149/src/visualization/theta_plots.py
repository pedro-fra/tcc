"""
Comprehensive plotting module for Theta model analysis and results.
Generates exploratory data analysis and prediction comparison visualizations.
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from darts import TimeSeries

# Import centralized plot configuration
from .plot_config import (
    BAR_CONFIG,
    FIGURE_CONFIG,
    HISTOGRAM_CONFIG,
    LINE_CONFIG,
    LINE_STYLES,
    VISUAL_COLORS,
    apply_plot_style,
    configure_axes,
    get_line_style,
    save_figure,
)
from .plot_translations import AXIS_LABELS, LEGEND_LABELS, PLOT_TITLES, SUBPLOT_TITLES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply centralized plot style
apply_plot_style()


class ThetaVisualizer:
    """
    Comprehensive visualization class for Theta method forecasting analysis.
    Generates professional plots with consistent styling for TCC documentation.
    """

    def __init__(self):
        """Initialize Theta visualizer with standardized configuration."""
        self.style = "seaborn-v0_8"
        self.figsize = FIGURE_CONFIG["figsize"]
        self.dpi = FIGURE_CONFIG["dpi"]

        logger.info("ThetaVisualizer initialized with standardized configuration")

    def create_exploratory_data_analysis(
        self, series: TimeSeries, output_dir: Path
    ) -> Dict[str, str]:
        """
        Create comprehensive exploratory data analysis plots for Theta method.

        Args:
            series: TimeSeries object to analyze
            output_dir: Directory to save plots

        Returns:
            Dictionary with paths to saved plot files
        """
        logger.info("Creating exploratory data analysis plots for Theta method")

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_files = {}

        # 1. Time Series Overview
        plot_files["overview"] = self._create_time_series_overview(series, output_dir)

        # 2. Statistical Analysis
        plot_files["statistics"] = self._create_statistical_analysis(series, output_dir)

        # 3. Distribution Analysis
        plot_files["distribution"] = self._create_distribution_analysis(series, output_dir)

        # 4. Trend and Seasonality Analysis
        plot_files["trend_seasonality"] = self._create_trend_seasonality_analysis(
            series, output_dir
        )

        logger.info("Exploratory data analysis plots completed")
        return plot_files

    def _create_time_series_overview(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Create time series overview plot with moving statistics.

        Args:
            series: TimeSeries to plot
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating time series overview plot")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise Exploratoria - Metodo Theta", fontsize=16, fontweight="bold")

        # Convert to DataFrame for plotting
        df = series.to_dataframe()
        dates = df.index
        values = df.iloc[:, 0].values

        # 1. Time series with moving average (top left)
        window = min(12, len(series) // 4)
        rolling_mean = df.rolling(window=window, center=True).mean()
        rolling_std = df.rolling(window=window).std()

        historical_style = LINE_STYLES["historical_data"]
        trend_style = LINE_STYLES["trend"]

        axes[0, 0].plot(dates, values, **historical_style, label=LEGEND_LABELS["original"])
        axes[0, 0].plot(
            dates,
            rolling_mean.iloc[:, 0],
            color=VISUAL_COLORS["predictions"],
            linewidth=LINE_STYLES["predictions"]["linewidth"],
            alpha=LINE_STYLES["predictions"]["alpha"],
            label=f"{LEGEND_LABELS['rolling_mean']} ({window}M)",
        )
        configure_axes(
            axes[0, 0],
            title=SUBPLOT_TITLES["sales_with_trend"],
            xlabel=AXIS_LABELS["date"],
            ylabel=AXIS_LABELS["sales_value"],
        )

        # 2. Rolling statistics (top right)
        axes[0, 1].plot(dates, values, **historical_style, label=LEGEND_LABELS["original"])
        axes[0, 1].plot(
            dates,
            rolling_mean.iloc[:, 0],
            color=VISUAL_COLORS["predictions"],
            linewidth=LINE_STYLES["predictions"]["linewidth"],
            alpha=LINE_STYLES["predictions"]["alpha"],
            label=f"{LEGEND_LABELS['rolling_mean']} ({window}M)",
        )
        axes[0, 1].plot(
            dates,
            rolling_std.iloc[:, 0],
            **trend_style,
            label=f"{LEGEND_LABELS['rolling_std']} ({window}M)",
        )
        configure_axes(
            axes[0, 1],
            title=SUBPLOT_TITLES["rolling_stats"],
            xlabel=AXIS_LABELS["date"],
            ylabel=AXIS_LABELS["sales_value"],
        )

        # 3. Monthly averages (bottom left)
        monthly_avg = df.groupby(df.index.month).mean()
        axes[1, 0].bar(range(1, 13), monthly_avg.iloc[:, 0], **BAR_CONFIG)
        axes[1, 0].set_title(SUBPLOT_TITLES["monthly_averages"])
        axes[1, 0].set_xlabel(AXIS_LABELS["month"])
        axes[1, 0].set_ylabel(AXIS_LABELS["average_sales"])
        axes[1, 0].set_xticks(range(1, 13))
        month_names = [
            "Jan",
            "Fev",
            "Mar",
            "Abr",
            "Mai",
            "Jun",
            "Jul",
            "Ago",
            "Set",
            "Out",
            "Nov",
            "Dez",
        ]
        axes[1, 0].set_xticklabels(month_names)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Annual totals (bottom right)
        annual_totals = df.groupby(df.index.year).sum()
        primary_style = LINE_STYLES["primary_data"]
        axes[1, 1].plot(
            annual_totals.index, annual_totals.iloc[:, 0], **primary_style, marker="o", markersize=6
        )
        axes[1, 1].set_title(SUBPLOT_TITLES["yearly_trends"])
        axes[1, 1].set_xlabel(AXIS_LABELS["year"])
        axes[1, 1].set_ylabel(AXIS_LABELS["total_sales"])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        filename = output_dir / "01_theta_time_series_overview.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_statistical_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Create statistical analysis plots.

        Args:
            series: TimeSeries to analyze
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating statistical analysis plots")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise Estatistica - Metodo Theta", fontsize=16, fontweight="bold")

        df = series.to_dataframe()
        values = df.iloc[:, 0].values

        # 1. Rolling mean and std (top left)
        window = min(12, len(series) // 4)
        rolling_mean = df.rolling(window=window).mean()
        rolling_std = df.rolling(window=window).std()

        historical_style_stats = LINE_STYLES["historical_data"]
        prediction_style_stats = LINE_STYLES["predictions"]

        axes[0, 0].plot(df.index, values, **historical_style_stats, label=LEGEND_LABELS["original"])
        axes[0, 0].plot(
            df.index,
            rolling_mean.iloc[:, 0],
            **prediction_style_stats,
            label=f"{LEGEND_LABELS['rolling_mean']} ({window}M)",
        )
        axes[0, 0].fill_between(
            df.index,
            (rolling_mean.iloc[:, 0] - rolling_std.iloc[:, 0]),
            (rolling_mean.iloc[:, 0] + rolling_std.iloc[:, 0]),
            alpha=0.3,
            color=VISUAL_COLORS["predictions"],
        )
        axes[0, 0].set_title(PLOT_TITLES["rolling_mean_std"])
        axes[0, 0].set_xlabel(AXIS_LABELS["date"])
        axes[0, 0].set_ylabel(AXIS_LABELS["sales_value"])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. First difference (top right)
        diff_values = np.diff(values)
        trend_style_diff = LINE_STYLES["trend"]
        axes[0, 1].plot(
            df.index[1:],
            diff_values,
            color=trend_style_diff["color"],
            linewidth=trend_style_diff["linewidth"],
            alpha=trend_style_diff["alpha"],
        )
        axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[0, 1].set_title(PLOT_TITLES["first_difference"])
        axes[0, 1].set_xlabel(AXIS_LABELS["date"])
        axes[0, 1].set_ylabel(AXIS_LABELS["differenced_values"])
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Quarterly analysis (bottom left)
        quarterly_avg = df.groupby(df.index.quarter).mean()
        quarter_names = ["1o Tri", "2o Tri", "3o Tri", "4o Tri"]

        axes[1, 0].bar(range(1, 5), quarterly_avg.iloc[:, 0], **BAR_CONFIG)
        axes[1, 0].set_title(PLOT_TITLES["quarterly_sales"])
        axes[1, 0].set_xlabel(AXIS_LABELS["quarter"])
        axes[1, 0].set_ylabel(AXIS_LABELS["average_sales"])
        axes[1, 0].set_xticks(range(1, 5))
        axes[1, 0].set_xticklabels(quarter_names)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Year-over-year growth (bottom right)
        annual_totals = df.groupby(df.index.year).sum()
        if len(annual_totals) > 1:
            growth_rates = annual_totals.pct_change().dropna() * 100
            axes[1, 1].bar(
                growth_rates.index.astype(str),
                growth_rates.iloc[:, 0],
                color=VISUAL_COLORS["bars"],
                edgecolor=VISUAL_COLORS["bar_edge"],
                alpha=BAR_CONFIG["alpha"],
            )
            axes[1, 1].axhline(y=0, color="black", linestyle="-", alpha=0.5)
            axes[1, 1].set_title(PLOT_TITLES["yoy_comparison"])
            axes[1, 1].set_xlabel(AXIS_LABELS["year"])
            axes[1, 1].set_ylabel("Taxa de Crescimento (%)")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "02_theta_statistical_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_distribution_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Create distribution analysis plots.

        Args:
            series: TimeSeries to analyze
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating distribution analysis plots")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise de Distribuicao - Metodo Theta", fontsize=16, fontweight="bold")

        df = series.to_dataframe()
        values = df.iloc[:, 0].values

        # 1. Histogram with KDE (top left)
        axes[0, 0].hist(values, **HISTOGRAM_CONFIG)
        try:
            from scipy import stats

            kde = stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 100)
            pred_style_kde = LINE_STYLES["predictions"]
            axes[0, 0].plot(
                x_range,
                kde(x_range),
                color=pred_style_kde["color"],
                linewidth=pred_style_kde["linewidth"],
                label="KDE",
            )
            axes[0, 0].legend()
        except ImportError:
            pass

        axes[0, 0].set_title(PLOT_TITLES["distribution"])
        axes[0, 0].set_xlabel(AXIS_LABELS["sales_value"])
        axes[0, 0].set_ylabel(AXIS_LABELS["density"])
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plot by month (top right)
        monthly_data = []
        month_labels = []
        for month in range(1, 13):
            month_values = df[df.index.month == month].iloc[:, 0].values
            if len(month_values) > 0:
                monthly_data.append(month_values)
                month_labels.append(month)

        if monthly_data:
            bp = axes[0, 1].boxplot(monthly_data, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(VISUAL_COLORS["bars"])
                patch.set_alpha(BAR_CONFIG["alpha"])
            axes[0, 1].set_title(PLOT_TITLES["monthly_distribution"])
            axes[0, 1].set_xlabel(AXIS_LABELS["month"])
            axes[0, 1].set_ylabel(AXIS_LABELS["sales_value"])
            axes[0, 1].set_xticklabels(month_labels)
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q plot (bottom left)
        try:
            from scipy import stats

            (osm, osr), (slope, intercept, r) = stats.probplot(values, dist="norm", plot=None)

            axes[1, 0].scatter(
                osm, osr, color=VISUAL_COLORS["primary_data"], alpha=LINE_CONFIG["secondary_alpha"]
            )
            axes[1, 0].plot(
                osm,
                slope * osm + intercept,
                color=VISUAL_COLORS["predictions"],
                linewidth=LINE_CONFIG["primary_linewidth"],
            )
            axes[1, 0].set_title(PLOT_TITLES["qq_plot"])
            axes[1, 0].set_xlabel("Quantis Teoricos")
            axes[1, 0].set_ylabel("Quantis da Amostra")
            axes[1, 0].grid(True, alpha=0.3)
        except ImportError:
            axes[1, 0].text(
                0.5,
                0.5,
                "Scipy not available",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )

        # 4. Summary statistics (bottom right)
        stats_data = {
            "Media": values.mean(),
            "Mediana": np.median(values),
            "Desvio": values.std(),
            "CV (%)": (values.std() / values.mean()) * 100,
        }

        axes[1, 1].axis("off")
        stats_text = "\n".join([f"{key}: {value:.2f}" for key, value in stats_data.items()])
        axes[1, 1].text(
            0.1,
            0.9,
            "Estatisticas Descritivas:",
            fontsize=14,
            fontweight="bold",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(0.1, 0.7, stats_text, fontsize=12, transform=axes[1, 1].transAxes)

        plt.tight_layout()

        filename = output_dir / "03_theta_distribution_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_trend_seasonality_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Create trend and seasonality analysis plots for Theta method.

        Args:
            series: Original TimeSeries data
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating trend and seasonality analysis plots")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(
            "Analise de Tendencia e Sazonalidade - Metodo Theta", fontsize=16, fontweight="bold"
        )

        # Convert original series to dataframe for plotting
        original_df = series.to_dataframe()
        dates = original_df.index
        values = original_df.iloc[:, 0].values

        # Original series with trend line (top left)
        primary_style_orig = LINE_STYLES["primary_data"]
        axes[0, 0].plot(dates, values, **primary_style_orig, label=LEGEND_LABELS["original_series"])

        # Add trend line
        z = np.polyfit(range(len(values)), values, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(
            dates,
            p(range(len(values))),
            color=VISUAL_COLORS["predictions"],
            linestyle="--",
            linewidth=LINE_CONFIG["primary_linewidth"],
            label=LEGEND_LABELS["trend"],
        )

        axes[0, 0].set_title(PLOT_TITLES["original_with_trend"])
        axes[0, 0].set_xlabel(AXIS_LABELS["date"])
        axes[0, 0].set_ylabel(AXIS_LABELS["sales_value"])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Detrended series (top right)
        detrended = values - p(range(len(values)))
        trend_style = LINE_STYLES["trend"]
        axes[0, 1].plot(dates, detrended, **trend_style)
        axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[0, 1].set_title(SUBPLOT_TITLES["detrended_series"])
        axes[0, 1].set_xlabel(AXIS_LABELS["date"])
        axes[0, 1].set_ylabel("Valores Sem Tendencia")
        axes[0, 1].grid(True, alpha=0.3)

        # Seasonal patterns (bottom left)
        monthly_avg = original_df.groupby(original_df.index.month).mean()
        month_names = [
            "Jan",
            "Fev",
            "Mar",
            "Abr",
            "Mai",
            "Jun",
            "Jul",
            "Ago",
            "Set",
            "Out",
            "Nov",
            "Dez",
        ]

        trend_style_seasonal = LINE_STYLES["trend"]
        axes[1, 0].plot(
            range(1, 13),
            monthly_avg.iloc[:, 0],
            color=trend_style_seasonal["color"],
            linestyle="-",
            linewidth=trend_style_seasonal["linewidth"],
            marker="o",
            markersize=8,
            alpha=trend_style_seasonal["alpha"],
        )
        axes[1, 0].set_title(PLOT_TITLES["seasonal_pattern"])
        axes[1, 0].set_xlabel(AXIS_LABELS["month"])
        axes[1, 0].set_ylabel(AXIS_LABELS["average_sales"])
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(month_names)
        axes[1, 0].grid(True, alpha=0.3)

        # Theta method information (bottom right)
        axes[1, 1].axis("off")

        theta_info = """
Metodo Theta:

• Metodo de decomposicao para previsao
• Combina tendencia linear com sazonalidade
• Automaticamente seleciona parametros
• Eficaz para series com padroes sazonais
• Boa performance em dados de vendas

Aplicacao:
• Identificacao automatica de sazonalidade
• Previsao robusta com tendencias
• Intervalos de confianca probabilisticos
        """

        axes[1, 1].text(
            0.05,
            0.95,
            theta_info.strip(),
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        filename = output_dir / "04_theta_trend_seasonality.png"
        save_figure(fig, filename)

        return str(filename)

    def create_prediction_comparison_plots(
        self,
        actual: TimeSeries,
        predicted: TimeSeries,
        train_series: TimeSeries,
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        Create prediction comparison plots for Theta method.

        Args:
            actual: Actual test values
            predicted: Predicted values
            train_series: Training data for context
            output_dir: Output directory

        Returns:
            Dictionary with paths to saved plot files
        """
        logger.info("Creating prediction comparison plots")

        plot_files = {}

        # 1. Main forecast plot
        plot_files["forecast"] = self._create_forecast_plot(
            actual, predicted, train_series, output_dir
        )

        # 2. Residual analysis
        plot_files["residuals"] = self._create_residual_analysis(actual, predicted, output_dir)

        return plot_files

    def _create_forecast_plot(
        self,
        actual: TimeSeries,
        predicted: TimeSeries,
        train_series: TimeSeries,
        output_dir: Path,
    ) -> str:
        """
        Create main forecast comparison plot.

        Args:
            actual: Actual test values
            predicted: Predicted values
            train_series: Training data
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating forecast comparison plot")

        fig, ax = plt.subplots(1, 1, figsize=FIGURE_CONFIG["figsize"])

        # Get line styles from centralized config
        actual_config = get_line_style("primary_data")
        pred_config = get_line_style("predictions")
        train_config = get_line_style("historical_data")

        # Plot training data
        train_series.plot(
            ax=ax,
            color=train_config["color"],
            linewidth=train_config["linewidth"],
            alpha=train_config["alpha"],
            label=LEGEND_LABELS["training_data"],
        )

        # Plot actual test data
        actual.plot(
            ax=ax,
            color=actual_config["color"],
            linewidth=actual_config["linewidth"],
            alpha=actual_config["alpha"],
            label=LEGEND_LABELS["actual"],
        )

        # Plot predictions
        predicted.plot(
            ax=ax,
            color=pred_config["color"],
            linewidth=pred_config["linewidth"],
            alpha=pred_config["alpha"],
            label=LEGEND_LABELS["predicted"],
        )

        # Configure plot
        configure_axes(
            ax,
            title="Comparacao de Previsoes - Metodo Theta",
            xlabel=AXIS_LABELS["date"],
            ylabel=AXIS_LABELS["sales_value"],
        )

        filename = output_dir / "05_theta_forecast_comparison.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_residual_analysis(
        self, actual: TimeSeries, predicted: TimeSeries, output_dir: Path
    ) -> str:
        """
        Create residual analysis plots.

        Args:
            actual: Actual values
            predicted: Predicted values
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating residual analysis plots")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise de Residuos - Metodo Theta", fontsize=16, fontweight="bold")

        # Calculate residuals
        actual_values = actual.values().flatten()
        pred_values = predicted.values().flatten()
        residuals = actual_values - pred_values

        actual_df = actual.to_dataframe()

        # 1. Residuals over time (top left)
        trend_style_resid = LINE_STYLES["trend"]
        axes[0, 0].plot(
            actual_df.index,
            residuals,
            color=trend_style_resid["color"],
            linewidth=trend_style_resid["linewidth"],
            alpha=trend_style_resid["alpha"],
        )
        axes[0, 0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[0, 0].set_title(PLOT_TITLES["residuals_over_time"])
        axes[0, 0].set_xlabel(AXIS_LABELS["date"])
        axes[0, 0].set_ylabel(AXIS_LABELS["residuals"])
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuals vs fitted values (top right)
        axes[0, 1].scatter(pred_values, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[0, 1].set_title(PLOT_TITLES["residuals_vs_fitted"])
        axes[0, 1].set_xlabel(AXIS_LABELS["fitted_values"])
        axes[0, 1].set_ylabel(AXIS_LABELS["residuals"])
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuals distribution (bottom left)
        axes[1, 0].hist(residuals, **HISTOGRAM_CONFIG)
        axes[1, 0].set_title(PLOT_TITLES["residual_distribution"])
        axes[1, 0].set_xlabel(AXIS_LABELS["residuals"])
        axes[1, 0].set_ylabel(AXIS_LABELS["density"])
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Q-Q plot of residuals (bottom right)
        try:
            from scipy import stats

            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)

            axes[1, 1].scatter(osm, osr, alpha=0.6)
            axes[1, 1].plot(osm, slope * osm + intercept, "r-", linewidth=2)
            axes[1, 1].set_title(PLOT_TITLES["qq_residuals"])
            axes[1, 1].set_xlabel("Quantis Teoricos")
            axes[1, 1].set_ylabel("Quantis dos Residuos")
            axes[1, 1].grid(True, alpha=0.3)
        except ImportError:
            axes[1, 1].text(
                0.5,
                0.5,
                "Scipy not available",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        plt.tight_layout()

        filename = output_dir / "06_theta_residual_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def create_performance_plots(
        self,
        actual: TimeSeries,
        predicted: TimeSeries,
        metrics: Dict[str, float],
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        Create model performance visualization plots.

        Args:
            actual: Actual values
            predicted: Predicted values
            metrics: Dictionary containing evaluation metrics
            output_dir: Output directory

        Returns:
            Dictionary with paths to saved plot files
        """
        logger.info("Creating performance analysis plots")

        plot_files = {}

        # 1. Error analysis
        plot_files["errors"] = self._create_error_analysis(actual, predicted, metrics, output_dir)

        # 2. Model summary
        plot_files["summary"] = self._create_model_summary(actual, predicted, metrics, output_dir)

        return plot_files

    def _create_error_analysis(
        self,
        actual: TimeSeries,
        predicted: TimeSeries,
        metrics: Dict[str, float],
        output_dir: Path,
    ) -> str:
        """
        Create error analysis plots.

        Args:
            actual: Actual values
            predicted: Predicted values
            metrics: Evaluation metrics
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating error analysis plots")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise de Erros - Metodo Theta", fontsize=16, fontweight="bold")

        # Get metric values
        mae_score = metrics.get("mae", 0)
        rmse_score = metrics.get("rmse", 0)
        mape_score = metrics.get("mape", 0)

        actual_values = actual.values().flatten()
        pred_values = predicted.values().flatten()
        actual_df = actual.to_dataframe()

        # 1. Actual vs Predicted scatter (top left)
        axes[0, 0].scatter(actual_values, pred_values, alpha=0.7)

        # Perfect prediction line
        min_val = min(min(actual_values), min(pred_values))
        max_val = max(max(actual_values), max(pred_values))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7)

        axes[0, 0].set_title(PLOT_TITLES["actual_vs_predicted"])
        axes[0, 0].set_xlabel(AXIS_LABELS["actual_sales"])
        axes[0, 0].set_ylabel(AXIS_LABELS["predicted_sales"])
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Error metrics bar chart (top right)
        metrics_names = ["MAE", "RMSE", "MAPE (%)"]
        metric_values = [mae_score, rmse_score, mape_score]

        bars = axes[0, 1].bar(
            metrics_names,
            metric_values,
            color=VISUAL_COLORS["bars"],
            edgecolor=VISUAL_COLORS["bar_edge"],
            alpha=BAR_CONFIG["alpha"],
        )
        axes[0, 1].set_title(PLOT_TITLES["error_metrics"])
        axes[0, 1].set_ylabel(AXIS_LABELS["metric_value"])
        axes[0, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # 3. Percentage error over time (bottom left)
        percentage_errors = np.abs((actual_values - pred_values) / actual_values) * 100
        axes[1, 0].plot(
            actual_df.index,
            percentage_errors,
            color=VISUAL_COLORS["secondary"],
            linewidth=LINE_CONFIG["secondary_linewidth"],
            alpha=LINE_CONFIG["secondary_alpha"],
        )
        axes[1, 0].set_title(PLOT_TITLES["percentage_error_time"])
        axes[1, 0].set_xlabel(AXIS_LABELS["date"])
        axes[1, 0].set_ylabel(AXIS_LABELS["percentage_error"])
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Cumulative absolute error (bottom right)
        cumulative_error = np.cumsum(np.abs(actual_values - pred_values))

        primary_style_cum = LINE_STYLES["primary_data"]
        axes[1, 1].plot(
            actual_df.index,
            cumulative_error,
            color=primary_style_cum["color"],
            linewidth=primary_style_cum["linewidth"],
            alpha=primary_style_cum["alpha"],
        )
        axes[1, 1].set_title(PLOT_TITLES["cumulative_error"])
        axes[1, 1].set_xlabel(AXIS_LABELS["date"])
        axes[1, 1].set_ylabel(AXIS_LABELS["cumulative_error"])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "07_theta_error_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_model_summary(
        self,
        actual: TimeSeries,
        predicted: TimeSeries,
        metrics: Dict[str, float],
        output_dir: Path,
    ) -> str:
        """
        Create model summary plot.

        Args:
            actual: Actual values
            predicted: Predicted values
            metrics: Evaluation metrics
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating model summary plot")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Resumo do Modelo - Metodo Theta", fontsize=16, fontweight="bold")

        # 1. Performance metrics bar chart (top left)
        metric_names = ["MAE", "RMSE", "MAPE (%)"]
        metric_values = [metrics.get("mae", 0), metrics.get("rmse", 0), metrics.get("mape", 0)]

        bars = axes[0, 0].bar(
            metric_names,
            metric_values,
            color=VISUAL_COLORS["bars"],
            edgecolor=VISUAL_COLORS["bar_edge"],
            alpha=BAR_CONFIG["alpha"],
        )
        axes[0, 0].set_title(PLOT_TITLES["model_performance"])
        axes[0, 0].set_ylabel(AXIS_LABELS["metric_value"])
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # 2. Model information (top right)
        axes[0, 1].axis("off")
        model_info = f"""
Metodo Theta - Resumo:

Tipo: Decomposicao de Series Temporais
Parametros: Automaticos
Sazonalidade: 12 meses

Metricas de Performance:
• MAE: {metrics.get("mae", 0):.2f}
• RMSE: {metrics.get("rmse", 0):.2f}
• MAPE: {metrics.get("mape", 0):.2f}%

Periodo de Teste: {len(actual)} observacoes
        """

        axes[0, 1].text(
            0.05,
            0.95,
            model_info.strip(),
            transform=axes[0, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
        )

        # 3. Forecast quality assessment (bottom left)
        axes[1, 0].axis("off")

        # Calculate quality metrics
        actual_values = actual.values().flatten()
        pred_values = predicted.values().flatten()

        # Calculate correlation
        correlation = np.corrcoef(actual_values, pred_values)[0, 1]

        # Calculate directional accuracy
        actual_changes = np.diff(actual_values)
        pred_changes = np.diff(pred_values)
        directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(pred_changes)) * 100

        quality_info = f"""
Avaliacao de Qualidade:

Correlacao: {correlation:.3f}
Acuracia Direcional: {directional_accuracy:.1f}%

Interpretacao:
• Correlacao > 0.8: Excelente
• Correlacao > 0.6: Boa
• Correlacao > 0.4: Regular
• Correlacao < 0.4: Fraca
        """

        axes[1, 0].text(
            0.05,
            0.95,
            quality_info.strip(),
            transform=axes[1, 0].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        # 4. Conclusions and recommendations (bottom right)
        axes[1, 1].axis("off")

        # Determine model quality
        mape = metrics.get("mape", 100)
        if mape < 10:
            quality = "Excelente"
        elif mape < 20:
            quality = "Boa"
        elif mape < 30:
            quality = "Regular"
        else:
            quality = "Fraca"

        conclusions = f"""
Conclusoes:

Qualidade da Previsao: {quality}

Recomendacoes:
• MAPE < 10%: Modelo pronto para uso
• MAPE 10-20%: Bom para planejamento
• MAPE 20-30%: Use com cautela
• MAPE > 30%: Revisar modelo

Metodo Theta e eficaz para:
• Series com sazonalidade clara
• Dados com tendencia estavel
• Previsoes de medio prazo
        """

        axes[1, 1].text(
            0.05,
            0.95,
            conclusions.strip(),
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        filename = output_dir / "08_theta_model_summary.png"
        save_figure(fig, filename)

        return str(filename)
