"""
Comprehensive plotting module for ARIMA model analysis and results.
Generates exploratory data analysis and prediction comparison visualizations.
"""

import locale
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from darts import TimeSeries
from darts.utils.statistics import extract_trend_and_seasonality

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

# Import Portuguese translations
from .plot_translations import (
    ADDITIONAL_TEXT,
    AXIS_LABELS,
    LEGEND_LABELS,
    MONTH_NAMES,
    PLOT_TITLES,
    SUBPLOT_TITLES,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply centralized plot style
apply_plot_style()

# Configure matplotlib for Portuguese
try:
    locale.setlocale(locale.LC_TIME, "pt_BR.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, "Portuguese_Brazil.1252")
    except locale.Error:
        pass  # Keep default if Portuguese locale not available


class ArimaVisualizer:
    """
    Comprehensive visualization class for ARIMA model analysis.
    Generates EDA, stationarity analysis, and prediction comparison plots.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ARIMA visualizer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.figsize = (12, 8)
        self.dpi = 300

    def create_exploratory_data_analysis(
        self, series: TimeSeries, output_dir: Path
    ) -> Dict[str, str]:
        """
        Create comprehensive exploratory data analysis plots.

        Args:
            series: Time series data
            output_dir: Directory to save plots

        Returns:
            Dictionary with plot file paths
        """
        logger.info(ADDITIONAL_TEXT["processing_eda"])

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_files = {}

        # 1. Time series overview
        plot_files["overview"] = self._plot_time_series_overview(series, output_dir)

        # 2. Decomposition analysis
        plot_files["decomposition"] = self._plot_decomposition_analysis(series, output_dir)

        # 3. Seasonality analysis
        plot_files["seasonality"] = self._plot_seasonality_analysis(series, output_dir)

        # 4. Statistical properties
        plot_files["statistics"] = self._plot_statistical_properties(series, output_dir)

        # 5. Distribution analysis
        plot_files["distribution"] = self._plot_distribution_analysis(series, output_dir)

        logger.info(f"{ADDITIONAL_TEXT['plots_saved']} {output_dir}")
        return plot_files

    def _plot_time_series_overview(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Plot time series overview with trend and key statistics.
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(PLOT_TITLES["main_eda"], fontsize=16, fontweight="bold")

        # Convert to pandas for easier plotting
        df = series.to_dataframe()
        dates = df.index
        values = df.iloc[:, 0].values

        # Main time series plot
        primary_style = LINE_STYLES["primary_data"]
        axes[0, 0].plot(dates, values, **primary_style)
        configure_axes(
            axes[0, 0], title=SUBPLOT_TITLES["sales_over_time"], ylabel=AXIS_LABELS["sales_value"]
        )
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Rolling statistics
        window = min(12, len(series) // 4)  # 12 months or 1/4 of data
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
        configure_axes(axes[0, 1], title=SUBPLOT_TITLES["rolling_stats"])
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Monthly averages
        monthly_avg = df.groupby(df.index.month).mean()
        axes[1, 0].bar(range(1, 13), monthly_avg.iloc[:, 0], **BAR_CONFIG)
        configure_axes(
            axes[1, 0],
            title=SUBPLOT_TITLES["monthly_averages"],
            xlabel=AXIS_LABELS["month"],
            ylabel=AXIS_LABELS["average_sales"],
        )
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(MONTH_NAMES, rotation=45)

        # Yearly trends
        yearly_sum = df.groupby(df.index.year).sum()
        primary_style = LINE_STYLES["primary_data"]
        axes[1, 1].plot(
            yearly_sum.index,
            yearly_sum.iloc[:, 0],
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
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        filename = output_dir / "01_time_series_overview.png"
        save_figure(fig, filename)

        return str(filename)

    def _plot_decomposition_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Plot time series decomposition (trend, seasonal, residual).
        """
        try:
            # Extract trend and seasonality using Darts
            trend, seasonal = extract_trend_and_seasonality(series, freq=12)

            fig, axes = plt.subplots(4, 1, figsize=(FIGURE_CONFIG["figsize"][0], 12))
            fig.suptitle("Time Series Decomposition Analysis", fontsize=16, fontweight="bold")

            # Original series
            series.plot(ax=axes[0], label="Original")
            axes[0].set_title("Original Time Series")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Trend component
            if trend is not None:
                trend.plot(ax=axes[1], label="Trend", color=VISUAL_COLORS["predictions"])
                axes[1].set_title("Trend Component")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            # Seasonal component
            if seasonal is not None:
                seasonal.plot(ax=axes[2], label="Seasonal", color=VISUAL_COLORS["seasonal"])
                axes[2].set_title("Seasonal Component")
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)

            # Residual component
            if trend is not None and seasonal is not None:
                residual = series - trend - seasonal
                residual.plot(ax=axes[3], label="Residual", color=VISUAL_COLORS["trend"])
                axes[3].set_title("Residual Component")
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)

            plt.tight_layout()

        except Exception as e:
            logger.warning(f"Decomposition failed: {e}, creating alternative plot")

            # Fallback: manual decomposition
            fig, axes = plt.subplots(2, 1, figsize=(FIGURE_CONFIG["figsize"][0], 8))
            fig.suptitle("Time Series Analysis", fontsize=16, fontweight="bold")

            # Original with moving average
            df = series.to_dataframe()
            window = 12
            moving_avg = df.rolling(window=window, center=True).mean()

            historical_style = LINE_STYLES["historical_data"]
            axes[0].plot(df.index, df.iloc[:, 0], **historical_style, label="Original")
            axes[0].plot(
                df.index,
                moving_avg.iloc[:, 0],
                color=VISUAL_COLORS["predictions"],
                linewidth=LINE_CONFIG["primary_linewidth"],
                label=f"Trend ({window}M MA)",
            )
            axes[0].set_title("Sales with Trend")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Detrended series
            detrended = df.iloc[:, 0] - moving_avg.iloc[:, 0]
            trend_style = LINE_STYLES["trend"]
            axes[1].plot(df.index, detrended, **trend_style)
            axes[1].set_title("Detrended Series")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

        filename = output_dir / "02_decomposition_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def _plot_seasonality_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Plot seasonality analysis and patterns.
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(PLOT_TITLES["seasonality"], fontsize=16, fontweight="bold")

        df = series.to_dataframe()
        values = df.iloc[:, 0]

        # Monthly boxplot
        monthly_data = [values[df.index.month == month].values for month in range(1, 13)]
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        axes[0, 0].boxplot(monthly_data, labels=months)
        axes[0, 0].set_title(SUBPLOT_TITLES["monthly_distribution"])
        axes[0, 0].set_ylabel("Sales (R$)")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Quarterly analysis
        quarterly_avg = df.groupby(df.index.quarter).mean()
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        axes[0, 1].bar(quarters, quarterly_avg.iloc[:, 0], **BAR_CONFIG)
        axes[0, 1].set_title(SUBPLOT_TITLES["quarterly_sales"])
        axes[0, 1].set_ylabel("Average Sales (R$)")
        axes[0, 1].grid(True, alpha=0.3)

        # Year-over-year comparison
        years = sorted(df.index.year.unique())
        for year in years[-5:]:  # Last 5 years
            year_data = df[df.index.year == year]
            if len(year_data) >= 6:  # At least 6 months of data
                axes[1, 0].plot(
                    year_data.index.month,
                    year_data.iloc[:, 0],
                    color=VISUAL_COLORS["primary_data"],
                    linewidth=LINE_CONFIG["secondary_linewidth"],
                    alpha=LINE_CONFIG["secondary_alpha"],
                    marker="o",
                    label=str(year),
                )

        axes[1, 0].set_title(SUBPLOT_TITLES["yoy_comparison"])
        axes[1, 0].set_xlabel("Month")
        axes[1, 0].set_ylabel("Sales (R$)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Lag plot (check for autocorrelation patterns)
        lag = 12  # 12-month lag
        if len(values) > lag:
            axes[1, 1].scatter(
                values[:-lag],
                values[lag:],
                color=VISUAL_COLORS["primary_data"],
                alpha=LINE_CONFIG["secondary_alpha"],
            )
            axes[1, 1].set_title(f"Lag Plot (Lag={lag} months)")
            axes[1, 1].set_xlabel("Sales(t)")
            axes[1, 1].set_ylabel(f"Sales(t+{lag})")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "03_seasonality_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def _plot_statistical_properties(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Plot statistical properties and stationarity tests.
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Statistical Properties Analysis", fontsize=16, fontweight="bold")

        df = series.to_dataframe()
        values = df.iloc[:, 0].values

        # ACF plot (manual implementation)
        from statsmodels.tsa.stattools import acf

        lags = min(40, len(values) // 4)
        acf_values = acf(values, nlags=lags, alpha=0.05)

        axes[0, 0].plot(range(lags + 1), acf_values[0], "b-")
        axes[0, 0].fill_between(
            range(lags + 1), acf_values[1][:, 0], acf_values[1][:, 1], alpha=0.2, color="blue"
        )
        axes[0, 0].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[0, 0].set_title("Autocorrelation Function (ACF)")
        axes[0, 0].set_xlabel("Lag")
        axes[0, 0].set_ylabel("ACF")
        axes[0, 0].grid(True, alpha=0.3)

        # PACF plot
        from statsmodels.tsa.stattools import pacf

        pacf_values = pacf(values, nlags=lags, alpha=0.05)

        axes[0, 1].plot(range(lags + 1), pacf_values[0], "r-")
        axes[0, 1].fill_between(
            range(lags + 1), pacf_values[1][:, 0], pacf_values[1][:, 1], alpha=0.2, color="red"
        )
        axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[0, 1].set_title("Partial Autocorrelation Function (PACF)")
        axes[0, 1].set_xlabel("Lag")
        axes[0, 1].set_ylabel("PACF")
        axes[0, 1].grid(True, alpha=0.3)

        # Rolling statistics
        window = 12
        rolling_mean = df.rolling(window=window).mean()
        rolling_std = df.rolling(window=window).std()

        axes[1, 0].plot(df.index, df.iloc[:, 0], alpha=0.3, label="Original")
        axes[1, 0].plot(df.index, rolling_mean.iloc[:, 0], "red", label="Rolling Mean")
        axes[1, 0].plot(df.index, rolling_std.iloc[:, 0], "green", label="Rolling Std")
        axes[1, 0].set_title("Rolling Mean and Standard Deviation")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Differenced series
        diff_values = np.diff(values)
        axes[1, 1].plot(df.index[1:], diff_values, alpha=0.8)
        axes[1, 1].set_title("First Difference")
        axes[1, 1].set_ylabel("Differenced Values")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "04_statistical_properties.png"
        save_figure(fig, filename)

        return str(filename)

    def _plot_distribution_analysis(self, series: TimeSeries, output_dir: Path) -> str:
        """
        Plot distribution analysis of the time series.
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Distribution Analysis", fontsize=16, fontweight="bold")

        df = series.to_dataframe()
        values = df.iloc[:, 0].values

        # Histogram
        axes[0, 0].hist(values, **HISTOGRAM_CONFIG)
        axes[0, 0].set_title("Sales Distribution")
        axes[0, 0].set_xlabel("Sales Value (R$)")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats

        stats.probplot(values, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot (Normal Distribution)")
        axes[0, 1].grid(True, alpha=0.3)

        # Box plot
        axes[1, 0].boxplot(values)
        axes[1, 0].set_title("Box Plot")
        axes[1, 0].set_ylabel("Sales Value (R$)")
        axes[1, 0].grid(True, alpha=0.3)

        # Kernel density estimation
        axes[1, 1].hist(values, **HISTOGRAM_CONFIG, label="Histogram")

        # Add KDE
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        prediction_style = LINE_STYLES["predictions"]
        axes[1, 1].plot(
            x_range,
            kde(x_range),
            color=prediction_style["color"],
            linewidth=prediction_style["linewidth"],
            label="KDE",
        )
        axes[1, 1].set_title("Distribution with KDE")
        axes[1, 1].set_xlabel("Sales Value (R$)")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "05_distribution_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def create_prediction_comparison_plots(
        self,
        actual: TimeSeries,
        predicted: TimeSeries,
        train_series: TimeSeries,
        output_dir: Path,
        stationarity_results: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Create comprehensive prediction vs actual comparison plots.

        Args:
            actual: Actual test values
            predicted: Predicted values
            train_series: Training data for context
            output_dir: Directory to save plots
            stationarity_results: Stationarity test results

        Returns:
            Dictionary with plot file paths
        """
        logger.info(ADDITIONAL_TEXT["processing_predictions"])

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_files = {}

        # 1. Main prediction comparison
        plot_files["comparison"] = self._plot_prediction_comparison(
            actual, predicted, train_series, output_dir
        )

        # 2. Residual analysis
        plot_files["residuals"] = self._plot_residual_analysis(actual, predicted, output_dir)

        # 3. Performance metrics visualization
        plot_files["metrics"] = self._plot_performance_metrics(actual, predicted, output_dir)

        # 4. Stationarity test results
        if stationarity_results:
            plot_files["stationarity"] = self._plot_stationarity_results(
                stationarity_results, output_dir
            )

        logger.info(f"Prediction plots saved to {output_dir}")
        return plot_files

    def _plot_prediction_comparison(
        self, actual: TimeSeries, predicted: TimeSeries, train_series: TimeSeries, output_dir: Path
    ) -> str:
        """
        Plot main prediction vs actual comparison.
        """
        fig, axes = plt.subplots(2, 1, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(PLOT_TITLES["prediction_comparison"], fontsize=16, fontweight="bold")

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

        filename = output_dir / "06_prediction_comparison.png"
        save_figure(fig, filename)

        return str(filename)

    def _plot_residual_analysis(
        self, actual: TimeSeries, predicted: TimeSeries, output_dir: Path
    ) -> str:
        """
        Plot comprehensive residual analysis.
        """
        # Calculate residuals
        actual_values = actual.values().flatten()
        predicted_values = predicted.values().flatten()
        residuals = actual_values - predicted_values

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Residual Analysis", fontsize=16, fontweight="bold")

        # Residuals over time
        dates = actual.to_dataframe().index
        trend_style = LINE_STYLES["trend"]
        axes[0, 0].plot(
            dates,
            residuals,
            color=trend_style["color"],
            linewidth=trend_style["linewidth"],
            alpha=trend_style["alpha"],
            marker="o",
            markersize=4,
        )
        axes[0, 0].axhline(y=0, color=VISUAL_COLORS["predictions"], linestyle="--", alpha=0.7)
        axes[0, 0].set_title("Residuals Over Time")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals vs fitted
        axes[0, 1].scatter(
            predicted_values,
            residuals,
            color=VISUAL_COLORS["primary_data"],
            alpha=LINE_CONFIG["secondary_alpha"],
        )
        axes[0, 1].axhline(y=0, color=VISUAL_COLORS["predictions"], linestyle="--", alpha=0.7)
        axes[0, 1].set_title("Residuals vs Fitted Values")
        axes[0, 1].set_xlabel("Fitted Values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].grid(True, alpha=0.3)

        # Residual distribution
        axes[1, 0].hist(residuals, **HISTOGRAM_CONFIG)
        axes[1, 0].set_title("Residual Distribution")
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].grid(True, alpha=0.3)

        # Q-Q plot of residuals
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot of Residuals")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "07_residual_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def _plot_performance_metrics(
        self, actual: TimeSeries, predicted: TimeSeries, output_dir: Path
    ) -> str:
        """
        Plot performance metrics visualization.
        """
        from darts.metrics import mae, mape, rmse

        # Calculate metrics
        mae_score = mae(actual, predicted)
        rmse_score = rmse(actual, predicted)
        mape_score = mape(actual, predicted)

        actual_values = actual.values().flatten()
        predicted_values = predicted.values().flatten()

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Performance Metrics Visualization", fontsize=16, fontweight="bold")

        # Actual vs Predicted scatter
        axes[0, 0].scatter(
            actual_values,
            predicted_values,
            color=VISUAL_COLORS["primary_data"],
            alpha=LINE_CONFIG["secondary_alpha"],
        )

        # Perfect prediction line
        min_val = min(min(actual_values), min(predicted_values))
        max_val = max(max(actual_values), max(predicted_values))
        axes[0, 0].plot(
            [min_val, max_val],
            [min_val, max_val],
            color=VISUAL_COLORS["predictions"],
            linestyle="--",
            alpha=LINE_CONFIG["primary_alpha"],
        )

        axes[0, 0].set_title("Actual vs Predicted")
        axes[0, 0].set_xlabel("Actual Sales")
        axes[0, 0].set_ylabel("Predicted Sales")
        axes[0, 0].grid(True, alpha=0.3)

        # Error metrics bar chart
        metrics = ["MAE", "RMSE", "MAPE (%)"]
        values = [mae_score, rmse_score, mape_score]

        bars = axes[0, 1].bar(
            metrics,
            values,
            color=VISUAL_COLORS["bars"],
            edgecolor=VISUAL_COLORS["bar_edge"],
            alpha=BAR_CONFIG["alpha"],
        )
        axes[0, 1].set_title("Error Metrics")
        axes[0, 1].set_ylabel("Error Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Percentage error over time
        percentage_errors = abs((actual_values - predicted_values) / actual_values) * 100
        dates = actual.to_dataframe().index

        axes[1, 0].plot(
            dates,
            percentage_errors,
            color=VISUAL_COLORS["secondary"],
            linewidth=LINE_CONFIG["secondary_linewidth"],
            alpha=LINE_CONFIG["secondary_alpha"],
            marker="o",
            markersize=4,
        )
        axes[1, 0].axhline(
            y=percentage_errors.mean(),
            color=VISUAL_COLORS["predictions"],
            linestyle="--",
            label=f"Mean: {percentage_errors.mean():.1f}%",
        )
        axes[1, 0].set_title("Percentage Error Over Time")
        axes[1, 0].set_ylabel("Absolute Percentage Error (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative error
        cumulative_error = np.cumsum(abs(actual_values - predicted_values))
        primary_style = LINE_STYLES["primary_data"]
        axes[1, 1].plot(
            dates,
            cumulative_error,
            color=primary_style["color"],
            linewidth=primary_style["linewidth"],
            alpha=primary_style["alpha"],
        )
        axes[1, 1].set_title("Cumulative Absolute Error")
        axes[1, 1].set_ylabel("Cumulative Error")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "08_performance_metrics.png"
        save_figure(fig, filename)

        return str(filename)

    def _plot_stationarity_results(self, stationarity_results: Dict, output_dir: Path) -> str:
        """
        Plot stationarity test results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(FIGURE_CONFIG["figsize"][0], 6))
        fig.suptitle("Stationarity Test Results", fontsize=16, fontweight="bold")

        # ADF test results
        adf_data = stationarity_results["adf_test"]
        adf_statistic = adf_data["statistic"]
        adf_pvalue = adf_data["pvalue"]
        adf_critical = adf_data["critical_values"]

        # Plot ADF test
        critical_levels = list(adf_critical.keys())
        critical_values = list(adf_critical.values())

        axes[0].barh(
            critical_levels,
            critical_values,
            color=VISUAL_COLORS["bars"],
            edgecolor=VISUAL_COLORS["bar_edge"],
            alpha=BAR_CONFIG["alpha"],
            label="Critical Values",
        )
        axes[0].axvline(
            x=adf_statistic,
            color=VISUAL_COLORS["predictions"],
            linestyle="--",
            linewidth=2,
            label=f"ADF Statistic: {adf_statistic:.3f}",
        )
        axes[0].set_title(f"ADF Test (p-value: {adf_pvalue:.4f})")
        axes[0].set_xlabel("Test Statistic")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # KPSS test results
        kpss_data = stationarity_results["kpss_test"]
        kpss_statistic = kpss_data["statistic"]
        kpss_pvalue = kpss_data["pvalue"]
        kpss_critical = kpss_data["critical_values"]

        critical_levels = list(kpss_critical.keys())
        critical_values = list(kpss_critical.values())

        axes[1].barh(
            critical_levels,
            critical_values,
            color=VISUAL_COLORS["bars"],
            edgecolor=VISUAL_COLORS["bar_edge"],
            alpha=BAR_CONFIG["alpha"],
            label="Critical Values",
        )
        axes[1].axvline(
            x=kpss_statistic,
            color=VISUAL_COLORS["predictions"],
            linestyle="--",
            linewidth=2,
            label=f"KPSS Statistic: {kpss_statistic:.3f}",
        )
        axes[1].set_title(f"KPSS Test (p-value: {kpss_pvalue:.4f})")
        axes[1].set_xlabel("Test Statistic")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "09_stationarity_tests.png"
        save_figure(fig, filename)

        return str(filename)

    def create_model_summary_plot(self, metrics: Dict[str, float], output_dir: Path) -> str:
        """
        Create a comprehensive model summary visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("ARIMA Model Summary", fontsize=16, fontweight="bold")

        # Metrics comparison with benchmarks
        metric_names = ["MAE", "RMSE", "MAPE (%)", "R²"]
        metric_values = [
            metrics.get("mae", 0),
            metrics.get("rmse", 0),
            metrics.get("mape", 0),
            metrics.get("r_squared", 0),
        ]

        # Color code based on performance (green=good, yellow=ok, red=poor)
        colors = []
        for i, (name, value) in enumerate(zip(metric_names, metric_values)):
            if name == "MAPE (%)":
                if value <= 10:
                    colors.append("green")
                elif value <= 25:
                    colors.append("orange")
                else:
                    colors.append("red")
            elif name == "R²":
                if value >= 0.8:
                    colors.append("green")
                elif value >= 0.5:
                    colors.append("orange")
                else:
                    colors.append("red")
            else:
                colors.append("blue")

        bars = axes[0, 0].bar(
            metric_names,
            metric_values,
            color=VISUAL_COLORS["bars"],
            edgecolor=VISUAL_COLORS["bar_edge"],
            alpha=BAR_CONFIG["alpha"],
        )
        axes[0, 0].set_title("Model Performance Metrics")
        axes[0, 0].set_ylabel("Metric Value")
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

        # Model configuration (placeholder)
        config_text = f"""
ARIMA Model Configuration:
• Automatic parameter selection
• Seasonal: Enabled (12 months)
• Test Period: {metrics.get("test_samples", 0)} months
• Training Period: {metrics.get("forecast_horizon", 0)} months

Performance Summary:
• Mean Absolute Error: {metrics.get("mae", 0):.0f}
• Root Mean Square Error: {metrics.get("rmse", 0):.0f}
• Mean Absolute Percentage Error: {metrics.get("mape", 0):.1f}%
• Directional Accuracy: {metrics.get("directional_accuracy", 0) * 100:.1f}%
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
        axes[0, 1].set_title("Model Configuration")

        # Placeholder for additional charts
        axes[1, 0].text(
            0.5,
            0.5,
            "Additional analysis\ncan be added here",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
        )
        axes[1, 0].set_title("Future Extensions")
        axes[1, 0].axis("off")

        axes[1, 1].text(
            0.5,
            0.5,
            "Model comparison\nwith other methods",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
        )
        axes[1, 1].set_title("Model Comparison")
        axes[1, 1].axis("off")

        plt.tight_layout()

        filename = output_dir / "10_model_summary.png"
        save_figure(fig, filename)

        return str(filename)

    def create_acf_pacf_plots(self, acf_pacf_results: Dict[str, Any], output_dir: Path) -> str:
        """
        Create ACF and PACF analysis plots.

        Args:
            acf_pacf_results: ACF/PACF analysis results from ARIMA model
            output_dir: Directory to save plots

        Returns:
            String path to saved plot file
        """
        logger.info("Criando graficos de analise ACF/PACF")

        # Set up the plot with Portuguese style
        # Using centralized plot style - DO NOT override
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(PLOT_TITLES["acf_pacf_analysis"], fontsize=16, fontweight="bold")

        # Extract data
        acf_values = acf_pacf_results.get("acf_values", [])
        pacf_values = acf_pacf_results.get("pacf_values", [])
        acf_confint = acf_pacf_results.get("acf_confint", [])
        pacf_confint = acf_pacf_results.get("pacf_confint", [])
        lags = acf_pacf_results.get("lags", [])
        suggested_params = acf_pacf_results.get("suggested_params", {})
        pattern_interpretation = acf_pacf_results.get("analysis_summary", "")

        # Limit to first 20 lags for clarity
        max_lags = min(20, len(lags))
        lags_subset = lags[:max_lags]
        acf_subset = acf_values[:max_lags]
        pacf_subset = pacf_values[:max_lags]

        # ACF Plot (top left)
        axes[0, 0].stem(lags_subset, acf_subset, basefmt="b-")
        axes[0, 0].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add confidence intervals if available
        if acf_confint and len(acf_confint) >= max_lags:
            acf_conf_subset = acf_confint[:max_lags]
            conf_upper = [conf[1] for conf in acf_conf_subset]
            conf_lower = [conf[0] for conf in acf_conf_subset]
            axes[0, 0].fill_between(
                lags_subset,
                conf_lower,
                conf_upper,
                alpha=0.2,
                color="gray",
                label=AXIS_LABELS["confidence_interval"],
            )

        axes[0, 0].set_title(PLOT_TITLES["acf_plot"])
        axes[0, 0].set_xlabel(AXIS_LABELS["lags"])
        axes[0, 0].set_ylabel(AXIS_LABELS["autocorrelation"])
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # PACF Plot (top right)
        axes[0, 1].stem(lags_subset, pacf_subset, basefmt="r-")
        axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Add confidence intervals if available
        if pacf_confint and len(pacf_confint) >= max_lags:
            pacf_conf_subset = pacf_confint[:max_lags]
            conf_upper = [conf[1] for conf in pacf_conf_subset]
            conf_lower = [conf[0] for conf in pacf_conf_subset]
            axes[0, 1].fill_between(
                lags_subset,
                conf_lower,
                conf_upper,
                alpha=0.2,
                color="gray",
                label=AXIS_LABELS["confidence_interval"],
            )

        axes[0, 1].set_title(PLOT_TITLES["pacf_plot"])
        axes[0, 1].set_xlabel(AXIS_LABELS["lags"])
        axes[0, 1].set_ylabel(AXIS_LABELS["partial_autocorrelation"])
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Parameter suggestions (bottom left)
        axes[1, 0].axis("off")
        suggestions_text = f"""
{PLOT_TITLES["parameter_suggestions"]}

{AXIS_LABELS["suggested_p"]}: {suggested_params.get("suggested_p", "N/A")}
{AXIS_LABELS["suggested_d"]}: {suggested_params.get("suggested_d", "N/A")}
{AXIS_LABELS["suggested_q"]}: {suggested_params.get("suggested_q", "N/A")}

{AXIS_LABELS["pattern_interpretation"]}:
{pattern_interpretation}
        """

        axes[1, 0].text(
            0.1,
            0.9,
            suggestions_text,
            transform=axes[1, 0].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # Significance analysis (bottom right)
        axes[1, 1].axis("off")

        # Count significant lags
        significant_acf = sum(1 for val in acf_subset[1:] if abs(val) > 0.2)
        significant_pacf = sum(1 for val in pacf_subset[1:] if abs(val) > 0.2)

        significance_text = f"""
{PLOT_TITLES["significance_analysis"]}

ACF - Lags Significativos: {significant_acf}
PACF - Lags Significativos: {significant_pacf}

{AXIS_LABELS["interpretation_guide"]}:

• ACF corta rapidamente → Modelo MA
• PACF corta rapidamente → Modelo AR
• Ambos decaem gradualmente → Modelo ARMA
        """

        axes[1, 1].text(
            0.1,
            0.9,
            significance_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()

        filename = output_dir / "11_acf_pacf_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def create_differencing_analysis_plots(
        self, original_series: TimeSeries, differencing_results: Dict[str, Any], output_dir: Path
    ) -> str:
        """
        Create differencing analysis visualization.

        Args:
            original_series: Original time series
            differencing_results: Results from differencing analysis
            output_dir: Directory to save plots

        Returns:
            String path to saved plot file
        """
        logger.info("Criando graficos de analise de diferenciacao")

        # Set up the plot
        # Using centralized plot style - DO NOT override
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(PLOT_TITLES["differencing_analysis"], fontsize=16, fontweight="bold")

        # Convert original series to dataframe for plotting
        original_df = original_series.to_dataframe()
        dates = original_df.index
        values = original_df.iloc[:, 0].values

        # Original series (top left)
        primary_style = LINE_STYLES["primary_data"]
        axes[0, 0].plot(dates, values, **primary_style, label=LEGEND_LABELS["original_series"])
        axes[0, 0].set_title(PLOT_TITLES["original_series"])
        axes[0, 0].set_xlabel(AXIS_LABELS["date"])
        axes[0, 0].set_ylabel(AXIS_LABELS["sales"])
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # First difference (top right) if differencing was applied
        differences_applied = differencing_results.get("differences_applied", 0)
        if differences_applied > 0:
            diff_series = original_series.diff()
            diff_df = diff_series.to_dataframe().dropna()  # Remove NaN values from differencing
            diff_dates = diff_df.index
            diff_values = diff_df.iloc[:, 0].values

            axes[0, 1].plot(
                diff_dates,
                diff_values,
                color=VISUAL_COLORS["predictions"],
                linewidth=LINE_CONFIG["secondary_linewidth"],
                label=LEGEND_LABELS["first_difference"],
            )
            axes[0, 1].set_title(PLOT_TITLES["first_difference"])
            axes[0, 1].set_xlabel(AXIS_LABELS["date"])
            axes[0, 1].set_ylabel(AXIS_LABELS["difference"])
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Diferenciacao nao aplicada\n(Serie ja estacionaria)",
                transform=axes[0, 1].transAxes,
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"),
            )
            axes[0, 1].set_title(PLOT_TITLES["first_difference"])

        # Stationarity results (bottom left)
        axes[1, 0].axis("off")

        final_stationary = differencing_results.get("final_series_stationary", False)
        final_tests = differencing_results.get("final_stationarity_tests", {})

        stationarity_text = f"""
{PLOT_TITLES["stationarity_results"]}

Diferencas Aplicadas: {differences_applied}
Serie Final Estacionaria: {"Sim" if final_stationary else "Nao"}

Testes de Estacionariedade Final:
"""

        if final_tests:
            adf_result = final_tests.get("adf_test", {})
            kpss_result = final_tests.get("kpss_test", {})

            stationarity_text += f"""
ADF Test:
• Estacionaria: {"Sim" if adf_result.get("is_stationary") else "Nao"}
• p-value: {adf_result.get("pvalue", "N/A"):.4f}

KPSS Test:
• Estacionaria: {"Sim" if kpss_result.get("is_stationary") else "Nao"}
• p-value: {kpss_result.get("pvalue", "N/A"):.4f}
"""

        axes[1, 0].text(
            0.1,
            0.9,
            stationarity_text,
            transform=axes[1, 0].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # Summary and recommendations (bottom right)
        axes[1, 1].axis("off")

        recommendation = (
            final_tests.get("recommendation", "Analise nao disponivel")
            if final_tests
            else "Analise nao disponivel"
        )

        summary_text = f"""
{PLOT_TITLES["differencing_summary"]}

Recomendacao:
{recommendation}

Proximos Passos:
• Aplicar modelo ARIMA com d={differences_applied}
• Verificar residuos do modelo
• Validar previsoes
        """

        axes[1, 1].text(
            0.1,
            0.9,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()

        filename = output_dir / "12_differencing_analysis.png"
        save_figure(fig, filename)

        return str(filename)


if __name__ == "__main__":
    print("ARIMA visualization module loaded successfully!")
