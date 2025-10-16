"""
Exponential Smoothing visualization module.
Creates comprehensive plots and charts for Exponential Smoothing model analysis and results.
All text in Portuguese (Brazil) without accents.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries

# Import centralized plot configuration
from .plot_config import (
    BAR_CONFIG,
    FIGURE_CONFIG,
    HISTOGRAM_CONFIG,
    LINE_CONFIG,
    LINE_STYLES,
    PREDICTION_COMPARISON_CONFIG,
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

# Set matplotlib to use non-interactive backend
plt.switch_backend("Agg")

# Apply centralized plot style
apply_plot_style()


class ExponentialVisualizer:
    """
    Comprehensive visualization class for Exponential Smoothing model analysis.
    Generates exploratory data analysis, prediction comparison, and performance plots.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Exponential Smoothing visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.style = "seaborn-v0_8"
        self.figsize = (12, 8)
        self.dpi = 300

        # Apply centralized plot style - DO NOT override this
        # Style is already applied via apply_plot_style() at module level

    def create_exploratory_data_analysis(
        self, series: TimeSeries, output_dir: Path
    ) -> Dict[str, str]:
        """
        Create comprehensive exploratory data analysis plots.

        Args:
            series: TimeSeries data for analysis
            output_dir: Directory to save plots

        Returns:
            Dictionary with paths to generated plot files
        """
        logger.info("Criando graficos de analise exploratoria para Suavizacao Exponencial")

        plot_files = {}

        # 1. Time series overview
        plot_files["overview"] = self._create_time_series_overview(series, output_dir)

        # 2. Decomposition analysis
        plot_files["decomposition"] = self._create_decomposition_analysis(series, output_dir)

        # 3. Seasonality analysis
        plot_files["seasonality"] = self._create_seasonality_analysis(series, output_dir)

        # 4. Statistical properties
        plot_files["statistics"] = self._create_statistical_properties(series, output_dir)

        # 5. Distribution analysis
        plot_files["distribution"] = self._create_distribution_analysis(series, output_dir)

        logger.info("Graficos EDA de Suavizacao Exponencial salvos em " + str(output_dir))

        return plot_files

    def _create_time_series_overview(self, series: TimeSeries, output_dir: Path) -> str:
        """Create time series overview plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(PLOT_TITLES["exponential_main_eda"], fontsize=16, fontweight="bold")

        # Convert to dataframe for plotting
        df = series.to_dataframe()
        dates = df.index
        values = df.iloc[:, 0].values

        # 1. Sales over time (top left)
        primary_style = LINE_STYLES["primary_data"]
        axes[0, 0].plot(dates, values, **primary_style)
        configure_axes(
            axes[0, 0],
            title=SUBPLOT_TITLES["sales_over_time"],
            xlabel=AXIS_LABELS["date"],
            ylabel=AXIS_LABELS["sales_value"],
        )

        # 2. Rolling statistics (top right)
        window = min(12, len(series) // 4)  # 12 months or 1/4 of data, same as ARIMA
        rolling_mean = df.rolling(window=window).mean()
        rolling_std = df.rolling(window=window).std()

        historical_style = LINE_STYLES["historical_data"]
        trend_style = LINE_STYLES["trend"]

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
        axes[0, 1].legend()

        # 3. Monthly averages (bottom left)
        monthly_avg = df.groupby(df.index.month).mean()
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

        axes[1, 0].bar(range(1, 13), monthly_avg.iloc[:, 0], **BAR_CONFIG)
        configure_axes(
            axes[1, 0],
            title=SUBPLOT_TITLES["monthly_averages"],
            xlabel=AXIS_LABELS["month"],
            ylabel=AXIS_LABELS["average_sales"],
        )
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(month_names)

        # 4. Yearly trends (bottom right)
        yearly_totals = df.groupby(df.index.year).sum()
        primary_style = LINE_STYLES["primary_data"]
        axes[1, 1].plot(
            yearly_totals.index,
            yearly_totals.iloc[:, 0],
            color=primary_style["color"],
            linewidth=primary_style["linewidth"],
            alpha=primary_style["alpha"],
            marker="o",
            markersize=6,
        )
        configure_axes(
            axes[1, 1],
            title=SUBPLOT_TITLES["yearly_trends"],
            xlabel=AXIS_LABELS["year"],
            ylabel=AXIS_LABELS["total_sales"],
        )

        plt.tight_layout()

        filename = output_dir / "01_exponential_time_series_overview.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(filename)

    def _create_decomposition_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """Create decomposition analysis plot."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            fig.suptitle(PLOT_TITLES["exponential_decomposition"], fontsize=16, fontweight="bold")

            # Convert to pandas series
            df = series.to_dataframe()
            ts_data = df.iloc[:, 0]

            # Perform decomposition
            decomposition = seasonal_decompose(ts_data, model="additive", period=12)

            # Plot each component
            decomposition.observed.plot(ax=axes[0], title=PLOT_TITLES["original_series"])
            axes[0].set_ylabel(AXIS_LABELS["sales_value"])

            decomposition.trend.plot(ax=axes[1], title=PLOT_TITLES["trend_component"])
            axes[1].set_ylabel(AXIS_LABELS["trend"])

            decomposition.seasonal.plot(ax=axes[2], title=PLOT_TITLES["seasonal_component"])
            axes[2].set_ylabel(AXIS_LABELS["seasonal"])

            decomposition.resid.plot(ax=axes[3], title=PLOT_TITLES["residual_component"])
            axes[3].set_ylabel(AXIS_LABELS["residuals"])
            axes[3].set_xlabel(AXIS_LABELS["date"])

            for ax in axes:
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            filename = output_dir / "02_exponential_decomposition_analysis.png"
            plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
            plt.close()

            return str(filename)

        except Exception as e:
            logger.warning(f"Could not create decomposition plot: {e}")
            return ""

    def _create_seasonality_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """Create seasonality analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(PLOT_TITLES["exponential_seasonality"], fontsize=16, fontweight="bold")

        df = series.to_dataframe()

        # 1. Monthly distribution (top left)
        monthly_data = df.groupby(df.index.month).apply(lambda x: x.iloc[:, 0].tolist())
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

        box_data = [monthly_data.get(i, []) for i in range(1, 13)]
        axes[0, 0].boxplot(box_data, labels=month_names)
        axes[0, 0].set_title(PLOT_TITLES["monthly_distribution"])
        axes[0, 0].set_xlabel(AXIS_LABELS["month"])
        axes[0, 0].set_ylabel(AXIS_LABELS["sales_value"])
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Quarterly analysis (top right)
        quarterly_avg = df.groupby(df.index.quarter).mean()
        quarter_names = ["1o Tri", "2o Tri", "3o Tri", "4o Tri"]

        axes[0, 1].bar(range(1, 5), quarterly_avg.iloc[:, 0], **BAR_CONFIG)
        axes[0, 1].set_title(PLOT_TITLES["quarterly_sales"])
        axes[0, 1].set_xlabel(AXIS_LABELS["quarter"])
        axes[0, 1].set_ylabel(AXIS_LABELS["average_sales"])
        axes[0, 1].set_xticks(range(1, 5))
        axes[0, 1].set_xticklabels(quarter_names)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Year-over-year comparison (bottom left)
        df_pivot = df.copy()
        df_pivot["Year"] = df_pivot.index.year
        df_pivot["Month"] = df_pivot.index.month

        years = sorted(df_pivot["Year"].unique())
        for year in years[-3:]:  # Last 3 years
            year_data = df_pivot[df_pivot["Year"] == year]
            monthly_avg = year_data.groupby("Month").mean()
            axes[1, 0].plot(
                monthly_avg.index, monthly_avg.iloc[:, 0], "o-", label=f"{year}", linewidth=2
            )

        axes[1, 0].set_title(PLOT_TITLES["yoy_comparison"])
        axes[1, 0].set_xlabel(AXIS_LABELS["month"])
        axes[1, 0].set_ylabel(AXIS_LABELS["average_sales"])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Lag plot (bottom right)
        axes[1, 1].scatter(df.iloc[:-1, 0], df.iloc[1:, 0], alpha=0.6)
        axes[1, 1].set_title(PLOT_TITLES["lag_plot"])
        axes[1, 1].set_xlabel(f"{AXIS_LABELS['sales_value']} (t)")
        axes[1, 1].set_ylabel(f"{AXIS_LABELS['sales_value']} (t+1)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "03_exponential_seasonality_analysis.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(filename)

    def _create_statistical_properties(self, series: TimeSeries, output_dir: Path) -> str:
        """Create statistical properties plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(PLOT_TITLES["exponential_statistical"], fontsize=16, fontweight="bold")

        df = series.to_dataframe()
        values = df.iloc[:, 0].values

        # 1. Rolling mean and std (top left)
        window = min(12, len(series) // 4)  # Consistent with other plots
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

        # 3. Autocorrelation (bottom left)
        try:
            from statsmodels.tsa.stattools import acf

            lags = min(40, len(values) // 4)
            acf_values = acf(values, nlags=lags)
            axes[1, 0].stem(range(len(acf_values)), acf_values)
            axes[1, 0].axhline(y=0, color="black", linestyle="-", alpha=0.3)
            axes[1, 0].set_title(PLOT_TITLES["acf_plot"])
            axes[1, 0].set_xlabel(AXIS_LABELS["lag"])
            axes[1, 0].set_ylabel(AXIS_LABELS["acf"])
            axes[1, 0].grid(True, alpha=0.3)
        except ImportError:
            axes[1, 0].text(
                0.5,
                0.5,
                "ACF nao disponivel",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )

        # 4. Statistics summary (bottom right)
        axes[1, 1].axis("off")
        stats_text = f"""
Estatisticas Descritivas:

Media: {np.mean(values):,.0f}
Mediana: {np.median(values):,.0f}
Desvio Padrao: {np.std(values):,.0f}
Minimo: {np.min(values):,.0f}
Maximo: {np.max(values):,.0f}

Coeficiente de Variacao: {(np.std(values) / np.mean(values)) * 100:.2f}%
Assimetria: {pd.Series(values).skew():.3f}
Curtose: {pd.Series(values).kurtosis():.3f}
        """

        axes[1, 1].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        filename = output_dir / "04_exponential_statistical_properties.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(filename)

    def _create_distribution_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """Create distribution analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(PLOT_TITLES["exponential_distribution"], fontsize=16, fontweight="bold")

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

        axes[0, 0].set_title(PLOT_TITLES["sales_distribution"])
        axes[0, 0].set_xlabel(AXIS_LABELS["sales_value"])
        axes[0, 0].set_ylabel(AXIS_LABELS["density"])
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Q-Q plot (top right)
        try:
            from scipy import stats

            stats.probplot(values, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title(PLOT_TITLES["qq_plot"])
            axes[0, 1].grid(True, alpha=0.3)
        except ImportError:
            axes[0, 1].text(
                0.5,
                0.5,
                "Q-Q plot nao disponivel",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
            )

        # 3. Box plot (bottom left)
        axes[1, 0].boxplot(values, vert=True)
        axes[1, 0].set_title(PLOT_TITLES["box_plot"])
        axes[1, 0].set_ylabel(AXIS_LABELS["sales_value"])
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Distribution summary (bottom right)
        axes[1, 1].axis("off")

        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        perc_values = np.percentile(values, percentiles)

        dist_text = f"""
Analise de Distribuicao:

Percentis:
P5: {perc_values[0]:,.0f}
P25: {perc_values[1]:,.0f}
P50: {perc_values[2]:,.0f}
P75: {perc_values[3]:,.0f}
P95: {perc_values[4]:,.0f}

IQR: {perc_values[3] - perc_values[1]:,.0f}
Amplitude: {values.max() - values.min():,.0f}
        """

        axes[1, 1].text(
            0.1,
            0.9,
            dist_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()

        filename = output_dir / "05_exponential_distribution_analysis.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(filename)

    def create_prediction_comparison_plots(
        self,
        actual: TimeSeries,
        predicted: TimeSeries,
        train_series: TimeSeries,
        output_dir: Path,
        trend_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Create prediction comparison plots.

        Args:
            actual: Actual test values
            predicted: Predicted values
            train_series: Training series
            output_dir: Directory to save plots
            trend_analysis: Trend/seasonality analysis results

        Returns:
            Dictionary with paths to generated plot files
        """
        logger.info("Criando graficos de comparacao de previsoes para Suavizacao Exponencial")

        plot_files = {}

        # 1. Prediction comparison
        plot_files["comparison"] = self._create_prediction_comparison(
            actual, predicted, train_series, output_dir
        )

        # 2. Residual analysis
        plot_files["residuals"] = self._create_residual_analysis(actual, predicted, output_dir)

        # 3. Performance metrics
        plot_files["metrics"] = self._create_performance_metrics(actual, predicted, output_dir)

        logger.info("Prediction plots saved to " + str(output_dir))

        return plot_files

    def _create_prediction_comparison(
        self, actual: TimeSeries, predicted: TimeSeries, train_series: TimeSeries, output_dir: Path
    ) -> str:
        """Create prediction vs actual comparison plot using standardized config."""
        fig, axes = plt.subplots(2, 1, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(
            "Modelo Suavizacao Exponencial: Previsoes vs Vendas Reais",
            fontsize=16,
            fontweight="bold",
        )

        # Main comparison plot - usando configuracoes padronizadas
        train_config = PREDICTION_COMPARISON_CONFIG["training_data"]
        actual_config = PREDICTION_COMPARISON_CONFIG["actual_test"]
        pred_config = PREDICTION_COMPARISON_CONFIG["predicted"]

        train_series.plot(ax=axes[0], label=LEGEND_LABELS["training_data"], **train_config)
        actual.plot(ax=axes[0], label=LEGEND_LABELS["actual_test"], **actual_config)
        predicted.plot(ax=axes[0], label=LEGEND_LABELS["predicted"], **pred_config)

        # Linha de divisao treino/teste
        split_config = get_line_style("split")
        axes[0].axvline(
            x=train_series.end_time(), label=LEGEND_LABELS["train_test_split"], **split_config
        )

        # Configurar eixos
        configure_axes(
            axes[0], title=SUBPLOT_TITLES["time_series_forecast"], ylabel=AXIS_LABELS["sales_value"]
        )
        axes[0].legend()

        # Visao detalhada do periodo de teste
        actual.plot(
            ax=axes[1],
            label=LEGEND_LABELS["actual"],
            color=actual_config["color"],
            linewidth=actual_config["linewidth"],
            marker="o",
            markersize=4,
            alpha=actual_config["alpha"],
        )
        predicted.plot(
            ax=axes[1],
            label=LEGEND_LABELS["predicted"],
            color=pred_config["color"],
            linewidth=pred_config["linewidth"],
            marker="o",
            markersize=4,
            alpha=pred_config["alpha"],
        )

        # Configurar eixos
        configure_axes(
            axes[1], title=SUBPLOT_TITLES["detailed_test_period"], ylabel=AXIS_LABELS["sales_value"]
        )
        axes[1].legend()

        plt.tight_layout()

        filename = output_dir / "06_exponential_prediction_comparison.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_residual_analysis(
        self, actual: TimeSeries, predicted: TimeSeries, output_dir: Path
    ) -> str:
        """Create residual analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(PLOT_TITLES["exponential_residual_analysis"], fontsize=16, fontweight="bold")

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

            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title(PLOT_TITLES["qq_residuals"])
            axes[1, 1].grid(True, alpha=0.3)
        except ImportError:
            axes[1, 1].text(
                0.5,
                0.5,
                "Q-Q plot nao disponivel",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

        plt.tight_layout()

        filename = output_dir / "07_exponential_residual_analysis.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(filename)

    def _create_performance_metrics(
        self, actual: TimeSeries, predicted: TimeSeries, output_dir: Path
    ) -> str:
        """Create performance metrics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(PLOT_TITLES["exponential_performance_metrics"], fontsize=16, fontweight="bold")

        # Calculate metrics
        from darts.metrics import mae, mape, rmse

        mae_score = mae(actual, predicted)
        rmse_score = rmse(actual, predicted)
        mape_score = mape(actual, predicted)

        actual_values = actual.values().flatten()
        pred_values = predicted.values().flatten()

        # 1. Actual vs Predicted scatter (top left)
        axes[0, 0].scatter(actual_values, pred_values, alpha=0.6)

        # Add perfect prediction line
        min_val = min(actual_values.min(), pred_values.min())
        max_val = max(actual_values.max(), pred_values.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.8)

        axes[0, 0].set_title(PLOT_TITLES["actual_vs_predicted"])
        axes[0, 0].set_xlabel(AXIS_LABELS["actual_sales"])
        axes[0, 0].set_ylabel(AXIS_LABELS["predicted_sales"])
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Error metrics bar chart (top right)
        metrics = ["MAE", "RMSE", "MAPE (%)"]
        metric_values = [mae_score, rmse_score, mape_score]

        bars = axes[0, 1].bar(
            metrics,
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
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # 3. Percentage error over time (bottom left)
        percentage_errors = np.abs((actual_values - pred_values) / actual_values) * 100
        actual_df = actual.to_dataframe()

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

        filename = output_dir / "08_exponential_performance_metrics.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(filename)

    def create_trend_seasonality_plots(
        self,
        original_series: TimeSeries,
        trend_analysis: Dict[str, Any],
        output_dir: Path,
    ) -> str:
        """
        Create trend and seasonality analysis visualization.

        Args:
            original_series: Original time series
            trend_analysis: Results from trend/seasonality analysis
            output_dir: Directory to save plots

        Returns:
            String path to saved plot file
        """
        logger.info("Criando graficos de analise de tendencia e sazonalidade")

        # Set up the plot
        # Using centralized plot style - DO NOT override
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(PLOT_TITLES["exponential_trend_seasonality"], fontsize=16, fontweight="bold")

        # Convert original series to dataframe for plotting
        original_df = original_series.to_dataframe()
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

        # Trend/seasonality strength (top right)
        axes[0, 1].axis("off")

        trend_strength = trend_analysis.get("trend_strength", 0)
        seasonal_strength = trend_analysis.get("seasonal_strength", 0)
        trend_type = trend_analysis.get("trend_type", "unknown")
        has_seasonality = trend_analysis.get("has_seasonality", False)
        recommended_model = trend_analysis.get("recommended_model", "N/A")

        analysis_text = f"""
{PLOT_TITLES["trend_seasonality_analysis"]}

Forca da Tendencia: {trend_strength:.4f}
Forca da Sazonalidade: {seasonal_strength:.4f}

Tipo de Tendencia: {trend_type}
Tem Sazonalidade: {"Sim" if has_seasonality else "Nao"}

Modelo Recomendado:
{recommended_model}
        """

        axes[0, 1].text(
            0.1,
            0.9,
            analysis_text,
            transform=axes[0, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # Seasonal pattern (bottom left)
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

        # Model selection guide (bottom right)
        axes[1, 1].axis("off")

        guide_text = f"""
{PLOT_TITLES["model_selection_guide"]}

Criterios de Selecao:

• Sem tendencia, sem sazonalidade:
  → Suavizacao Exponencial Simples

• Com tendencia, sem sazonalidade:
  → Metodo de Holt (Linear)

• Sem tendencia, com sazonalidade:
  → Suavizacao Exponencial Sazonal

• Com tendencia e sazonalidade:
  → Metodo de Holt-Winters

Escolha Automatica: Habilitada
        """

        axes[1, 1].text(
            0.1,
            0.9,
            guide_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()

        filename = output_dir / "09_exponential_trend_seasonality.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(filename)

    def create_model_summary_plot(self, metrics: Dict[str, float], output_dir: Path) -> str:
        """
        Create model summary visualization.

        Args:
            metrics: Model performance metrics
            output_dir: Directory to save plots

        Returns:
            String path to saved plot file
        """
        logger.info("Criando grafico de resumo do modelo de Suavizacao Exponencial")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(PLOT_TITLES["exponential_model_summary"], fontsize=16, fontweight="bold")

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
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Model configuration (top right)
        config_text = f"""
Configuracao do Modelo:
• Modelo: Suavizacao Exponencial
• Selecao Automatica: Habilitada
• Periodo de Teste: {metrics.get("test_samples", 0)} meses
• Horizonte de Previsao: {metrics.get("forecast_horizon", 0)} meses

Resumo de Performance:
• Erro Absoluto Medio: {metrics.get("mae", 0):.0f}
• Raiz do Erro Quadratico Medio: {metrics.get("rmse", 0):.0f}
• Erro Percentual Absoluto Medio: {metrics.get("mape", 0):.1f}%
        """

        axes[0, 1].text(
            0.1,
            0.5,
            config_text,
            transform=axes[0, 1].transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
        )
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axis("off")
        axes[0, 1].set_title("Configuracao do Modelo")

        # Placeholder for additional charts
        axes[1, 0].text(
            0.5,
            0.5,
            "Analises adicionais\npodem ser incluidas aqui",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
        )
        axes[1, 0].set_title("Extensoes Futuras")
        axes[1, 0].axis("off")

        axes[1, 1].text(
            0.5,
            0.5,
            "Comparacao com\noutros metodos",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
        )
        axes[1, 1].set_title("Comparacao de Modelos")
        axes[1, 1].axis("off")

        plt.tight_layout()

        filename = output_dir / "10_exponential_model_summary.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return str(filename)


if __name__ == "__main__":
    print("Exponential Smoothing visualization module loaded successfully!")
