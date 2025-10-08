"""
XGBoost visualization module for sales forecasting analysis.
Creates comprehensive plots for XGBoost model performance, feature importance, and predictions.
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import plot configuration
from .plot_config import (
    BAR_CONFIG,
    FIGURE_CONFIG,
    HISTOGRAM_CONFIG,
    LINE_CONFIG,
    LINE_STYLES,
    VISUAL_COLORS,
    configure_axes,
    save_figure,
)
from .plot_translations import AXIS_LABELS, LEGEND_LABELS, PLOT_TITLES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostVisualizer:
    """
    Visualization class for XGBoost model analysis and results.
    Creates standardized plots following the project's visual guidelines.
    """

    def __init__(self):
        """Initialize XGBoost visualizer with standardized configuration."""
        logger.info("XGBoostVisualizer initialized with standardized configuration")

    def create_feature_analysis(
        self,
        features: pd.DataFrame,
        feature_importance: np.ndarray,
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        Create comprehensive feature analysis plots.

        Args:
            features: Features DataFrame
            feature_importance: Feature importance array
            output_dir: Output directory

        Returns:
            Dictionary with paths to saved plot files
        """
        logger.info("Creating feature analysis plots for XGBoost")

        plot_files = {}

        # 1. Feature overview
        plot_files["overview"] = self._create_feature_overview(features, output_dir)

        # 2. Feature importance
        plot_files["importance"] = self._create_feature_importance_plot(
            features, feature_importance, output_dir
        )

        # 3. Feature correlations
        plot_files["correlations"] = self._create_correlation_analysis(features, output_dir)

        # 4. Distribution analysis
        plot_files["distributions"] = self._create_feature_distribution_analysis(
            features, output_dir
        )

        return plot_files

    def _create_feature_overview(self, features: pd.DataFrame, output_dir: Path) -> str:
        """
        Create feature overview plot showing basic statistics and information.

        Args:
            features: Features DataFrame
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating feature overview plot")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise de Features - XGBoost", fontsize=16, fontweight="bold")

        # 1. Feature count by type (top left)
        feature_types = {
            "Temporais": sum(
                1
                for col in features.columns
                if any(x in col for x in ["year", "month", "day", "quarter"])
            ),
            "Lag": sum(1 for col in features.columns if "lag" in col),
            "Media Movel": sum(1 for col in features.columns if "ma" in col),
            "Cyclicas": sum(1 for col in features.columns if any(x in col for x in ["sin", "cos"])),
            "Outras": len(features.columns)
            - sum(
                [
                    sum(
                        1
                        for col in features.columns
                        if any(x in col for x in ["year", "month", "day", "quarter"])
                    ),
                    sum(1 for col in features.columns if "lag" in col),
                    sum(1 for col in features.columns if "ma" in col),
                    sum(1 for col in features.columns if any(x in col for x in ["sin", "cos"])),
                ]
            ),
        }

        axes[0, 0].bar(feature_types.keys(), feature_types.values(), **BAR_CONFIG)
        axes[0, 0].set_title("Tipos de Features")
        axes[0, 0].set_ylabel("Quantidade")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Data range (top right)
        start_idx = features.index.min()
        end_idx = features.index.max()
        total_samples = len(features)

        info_text = f"""
        Informacoes Gerais:

        • Total de Features: {len(features.columns)}
        • Total de Amostras: {total_samples}
        • Intervalo de Indices: {start_idx} a {end_idx}
        • Features Numericas: {len(features.select_dtypes(include=[np.number]).columns)}
        • Memoria (MB): {features.memory_usage(deep=True).sum() / 1024**2:.2f}
        """

        axes[0, 1].axis("off")
        axes[0, 1].text(
            0.05,
            0.95,
            info_text.strip(),
            transform=axes[0, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # 3. Missing values heatmap (bottom left)
        missing_data = features.isnull().sum()
        if missing_data.sum() > 0:
            top_missing = missing_data[missing_data > 0].head(10)
            axes[1, 0].barh(range(len(top_missing)), top_missing.values, **BAR_CONFIG)
            axes[1, 0].set_yticks(range(len(top_missing)))
            axes[1, 0].set_yticklabels(top_missing.index)
            axes[1, 0].set_title("Valores Ausentes por Feature")
            axes[1, 0].set_xlabel("Quantidade")
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "Nenhum valor ausente encontrado",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
            )
            axes[1, 0].set_title("Valores Ausentes")

        axes[1, 0].grid(True, alpha=0.3)

        # 4. Feature statistics (bottom right)
        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            stats_summary = {
                "Media": numeric_features.mean().mean(),
                "Mediana": numeric_features.median().median(),
                "Desvio Padrao": numeric_features.std().mean(),
                "CV Medio (%)": (numeric_features.std() / numeric_features.mean()).mean() * 100,
            }

            axes[1, 1].axis("off")
            stats_text = "\n".join([f"{key}: {value:.2f}" for key, value in stats_summary.items()])
            axes[1, 1].text(
                0.1,
                0.9,
                "Estatisticas das Features:",
                fontsize=14,
                fontweight="bold",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].text(0.1, 0.7, stats_text, fontsize=12, transform=axes[1, 1].transAxes)

        plt.tight_layout()

        filename = output_dir / "01_xgboost_feature_overview.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_feature_importance_plot(
        self, features: pd.DataFrame, feature_importance: np.ndarray, output_dir: Path
    ) -> str:
        """
        Create feature importance plots.

        Args:
            features: Features DataFrame
            feature_importance: Feature importance values
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating feature importance plot")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(
            "Analise de Importancia das Features - XGBoost", fontsize=16, fontweight="bold"
        )

        # Create importance DataFrame
        feature_names = features.columns
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        ).sort_values("importance", ascending=False)

        # 1. Top 15 feature importance (top left)
        top_features = importance_df.head(15)
        y_pos = np.arange(len(top_features))

        axes[0, 0].barh(
            y_pos,
            top_features["importance"],
            color=BAR_CONFIG["color"],
            edgecolor=BAR_CONFIG["edgecolor"],
            alpha=BAR_CONFIG["alpha"],
        )
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(top_features["feature"])
        axes[0, 0].set_title("Top 15 Features Mais Importantes")
        axes[0, 0].set_xlabel("Importancia")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cumulative importance (top right)
        cumulative_importance = np.cumsum(importance_df["importance"].values)
        cumulative_percentage = cumulative_importance / cumulative_importance[-1] * 100

        axes[0, 1].plot(
            range(len(cumulative_percentage)),
            cumulative_percentage,
            **LINE_STYLES["primary_data"],
            marker="o",
            markersize=4,
        )
        axes[0, 1].axhline(y=80, color="red", linestyle="--", alpha=0.7, label="80%")
        axes[0, 1].axhline(y=95, color="orange", linestyle="--", alpha=0.7, label="95%")
        axes[0, 1].set_title("Importancia Cumulativa")
        axes[0, 1].set_xlabel("Numero de Features")
        axes[0, 1].set_ylabel("Importancia Cumulativa (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature importance by category (bottom left)
        feature_categories = {
            "Temporais": [
                f for f in feature_names if any(x in f for x in ["year", "month", "day", "quarter"])
            ],
            "Lag": [f for f in feature_names if "lag" in f],
            "Media Movel": [f for f in feature_names if "ma" in f],
            "Cyclicas": [f for f in feature_names if any(x in f for x in ["sin", "cos"])],
            "Outras": [
                f
                for f in feature_names
                if not any(
                    pattern in f
                    for pattern in ["year", "month", "day", "quarter", "lag", "ma", "sin", "cos"]
                )
            ],
        }

        category_importance = {}
        for category, category_features in feature_categories.items():
            if category_features:
                category_indices = [
                    i for i, f in enumerate(feature_names) if f in category_features
                ]
                category_importance[category] = feature_importance[category_indices].mean()
            else:
                category_importance[category] = 0

        axes[1, 0].bar(category_importance.keys(), category_importance.values(), **BAR_CONFIG)
        axes[1, 0].set_title("Importancia Media por Categoria")
        axes[1, 0].set_ylabel("Importancia Media")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Feature importance distribution (bottom right)
        axes[1, 1].hist(feature_importance, **HISTOGRAM_CONFIG)
        axes[1, 1].axvline(
            feature_importance.mean(),
            color="red",
            linestyle="--",
            label=f"Media: {feature_importance.mean():.4f}",
        )
        axes[1, 1].set_title("Distribuicao da Importancia")
        axes[1, 1].set_xlabel("Importancia")
        axes[1, 1].set_ylabel("Frequencia")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "02_xgboost_feature_importance.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_correlation_analysis(self, features: pd.DataFrame, output_dir: Path) -> str:
        """
        Create feature correlation analysis.

        Args:
            features: Features DataFrame
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating correlation analysis plot")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise de Correlacao das Features - XGBoost", fontsize=16, fontweight="bold")

        # Select only numeric features for correlation
        numeric_features = features.select_dtypes(include=[np.number])

        # 1. Correlation heatmap (top left) - top correlated features
        correlation_matrix = numeric_features.corr()

        # Get top correlated features (excluding self-correlation)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        correlation_matrix_masked = correlation_matrix.mask(mask)

        # Find top 20 correlations
        correlation_pairs = correlation_matrix_masked.unstack().dropna()
        top_correlations = correlation_pairs.abs().nlargest(20)

        if len(top_correlations) > 0:
            top_corr_features = list(
                set(
                    [pair[0] for pair in top_correlations.index]
                    + [pair[1] for pair in top_correlations.index]
                )
            )[:15]

            correlation_subset = correlation_matrix.loc[top_corr_features, top_corr_features]

            sns.heatmap(
                correlation_subset,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                ax=axes[0, 0],
                cbar_kws={"shrink": 0.8},
            )
            axes[0, 0].set_title("Correlacoes Principais")
        else:
            axes[0, 0].text(
                0.5,
                0.5,
                "Dados insuficientes para correlacao",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )

        # 2. Correlation distribution (top right)
        all_correlations = correlation_matrix_masked.unstack().dropna()
        axes[0, 1].hist(all_correlations, **HISTOGRAM_CONFIG)
        axes[0, 1].axvline(
            all_correlations.mean(),
            color="red",
            linestyle="--",
            label=f"Media: {all_correlations.mean():.3f}",
        )
        axes[0, 1].set_title("Distribuicao das Correlacoes")
        axes[0, 1].set_xlabel("Correlacao")
        axes[0, 1].set_ylabel("Frequencia")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. High correlation pairs (bottom left)
        high_correlations = (
            correlation_pairs[correlation_pairs.abs() > 0.7]
            .abs()
            .sort_values(ascending=False)
            .head(10)
        )

        if len(high_correlations) > 0:
            y_pos = np.arange(len(high_correlations))
            pair_labels = [f"{pair[0]} - {pair[1]}" for pair in high_correlations.index]

            axes[1, 0].barh(
                y_pos,
                high_correlations.values,
                color=BAR_CONFIG["color"],
                edgecolor=BAR_CONFIG["edgecolor"],
                alpha=BAR_CONFIG["alpha"],
            )
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(
                [label[:30] + "..." if len(label) > 30 else label for label in pair_labels]
            )
            axes[1, 0].set_title("Pares com Alta Correlacao (>0.7)")
            axes[1, 0].set_xlabel("Correlacao Absoluta")
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "Nenhuma correlacao alta encontrada",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set_title("Pares com Alta Correlacao")

        axes[1, 0].grid(True, alpha=0.3)

        # 4. Correlation summary (bottom right)
        axes[1, 1].axis("off")

        correlation_stats = {
            "Correlacoes Totais": len(all_correlations),
            "Correlacao Media": all_correlations.mean(),
            "Correlacao Maxima": all_correlations.abs().max(),
            "Correlacoes Altas (>0.7)": sum(all_correlations.abs() > 0.7),
            "Correlacoes Moderadas (0.3-0.7)": sum(
                (all_correlations.abs() > 0.3) & (all_correlations.abs() <= 0.7)
            ),
        }

        stats_text = "\n".join(
            [
                f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}"
                for key, value in correlation_stats.items()
            ]
        )

        axes[1, 1].text(
            0.1,
            0.9,
            "Resumo de Correlacoes:",
            fontsize=14,
            fontweight="bold",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].text(0.1, 0.7, stats_text, fontsize=12, transform=axes[1, 1].transAxes)

        plt.tight_layout()

        filename = output_dir / "03_xgboost_correlation_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_feature_distribution_analysis(
        self, features: pd.DataFrame, output_dir: Path
    ) -> str:
        """
        Create feature distribution analysis.

        Args:
            features: Features DataFrame
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating feature distribution analysis")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle(
            "Analise de Distribuicao das Features - XGBoost", fontsize=16, fontweight="bold"
        )

        # Select numeric features
        numeric_features = features.select_dtypes(include=[np.number])

        # 1. Feature variance analysis (top left)
        feature_variances = numeric_features.var().sort_values(ascending=False).head(15)

        y_pos = np.arange(len(feature_variances))
        axes[0, 0].barh(
            y_pos,
            feature_variances.values,
            color=BAR_CONFIG["color"],
            edgecolor=BAR_CONFIG["edgecolor"],
            alpha=BAR_CONFIG["alpha"],
        )
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(
            [name[:20] + "..." if len(name) > 20 else name for name in feature_variances.index]
        )
        axes[0, 0].set_title("Top 15 Features por Variancia")
        axes[0, 0].set_xlabel("Variancia")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Feature skewness (top right)
        feature_skewness = numeric_features.skew().abs().sort_values(ascending=False).head(15)

        y_pos = np.arange(len(feature_skewness))
        axes[0, 1].barh(
            y_pos,
            feature_skewness.values,
            color=BAR_CONFIG["color"],
            edgecolor=BAR_CONFIG["edgecolor"],
            alpha=BAR_CONFIG["alpha"],
        )
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(
            [name[:20] + "..." if len(name) > 20 else name for name in feature_skewness.index]
        )
        axes[0, 1].set_title("Top 15 Features por Assimetria")
        axes[0, 1].set_xlabel("Assimetria Absoluta")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution of feature means (bottom left)
        feature_means = numeric_features.mean()
        axes[1, 0].hist(feature_means, **HISTOGRAM_CONFIG)
        axes[1, 0].axvline(
            feature_means.mean(),
            color="red",
            linestyle="--",
            label=f"Media: {feature_means.mean():.2e}",
        )
        axes[1, 0].set_title("Distribuicao das Medias")
        axes[1, 0].set_xlabel("Media da Feature")
        axes[1, 0].set_ylabel("Frequencia")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Zero values analysis (bottom right)
        zero_counts = (numeric_features == 0).sum().sort_values(ascending=False).head(10)

        if zero_counts.sum() > 0:
            y_pos = np.arange(len(zero_counts))
            axes[1, 1].barh(
                y_pos,
                zero_counts.values,
                color=BAR_CONFIG["color"],
                edgecolor=BAR_CONFIG["edgecolor"],
                alpha=BAR_CONFIG["alpha"],
            )
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(
                [name[:20] + "..." if len(name) > 20 else name for name in zero_counts.index]
            )
            axes[1, 1].set_title("Features com Mais Valores Zero")
            axes[1, 1].set_xlabel("Quantidade de Zeros")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Nenhum valor zero encontrado",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Valores Zero")

        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = output_dir / "04_xgboost_distribution_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def create_prediction_comparison_plots(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: np.ndarray,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        Create prediction comparison plots.

        Args:
            X_test: Test features
            y_test: Test target values
            predictions: Model predictions
            X_train: Training features
            y_train: Training target values
            output_dir: Output directory

        Returns:
            Dictionary with paths to saved plot files
        """
        logger.info("Creating prediction comparison plots")

        plot_files = {}

        # 1. Main forecast plot
        plot_files["forecast"] = self._create_forecast_plot(
            X_test, y_test, predictions, X_train, y_train, output_dir
        )

        # 2. Residual analysis
        plot_files["residuals"] = self._create_residual_analysis(y_test, predictions, output_dir)

        return plot_files

    def _create_forecast_plot(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: np.ndarray,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        output_dir: Path,
    ) -> str:
        """
        Create main forecast comparison plot.

        Args:
            X_test: Test features
            y_test: Test target values
            predictions: Model predictions
            X_train: Training features
            y_train: Training target values
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating forecast comparison plot")

        fig, ax = plt.subplots(1, 1, figsize=FIGURE_CONFIG["figsize"])

        # Plot training data
        train_config = LINE_STYLES["historical_data"]
        ax.plot(
            X_train.index,
            y_train.values,
            color=train_config["color"],
            linewidth=train_config["linewidth"],
            alpha=train_config["alpha"],
            label=LEGEND_LABELS["training_data"],
        )

        # Plot actual test data
        actual_config = LINE_STYLES["primary_data"]
        ax.plot(
            X_test.index,
            y_test.values,
            color=actual_config["color"],
            linewidth=actual_config["linewidth"],
            alpha=actual_config["alpha"],
            label=LEGEND_LABELS["actual_test"],
        )

        # Plot predictions
        pred_config = LINE_STYLES["predictions"]
        ax.plot(
            X_test.index,
            predictions,
            color=pred_config["color"],
            linewidth=pred_config["linewidth"],
            alpha=pred_config["alpha"],
            label=LEGEND_LABELS["predicted"],
        )

        # Add vertical line to separate train/test
        if len(X_train) > 0 and len(X_test) > 0:
            split_date = X_test.index[0]
            ax.axvline(
                x=split_date,
                color="black",
                linestyle="--",
                alpha=0.7,
                label=LEGEND_LABELS["train_test_split"],
            )

        # Configure plot
        configure_axes(
            ax,
            title="Comparacao de Previsoes - XGBoost",
            xlabel=AXIS_LABELS["date"],
            ylabel=AXIS_LABELS["sales_value"],
        )

        # Format date axis for better readability
        import matplotlib.dates as mdates

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Every 6 months
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        filename = output_dir / "05_xgboost_forecast_comparison.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_residual_analysis(
        self, y_test: pd.Series, predictions: np.ndarray, output_dir: Path
    ) -> str:
        """
        Create residual analysis plots.

        Args:
            y_test: Test target values
            predictions: Model predictions
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating residual analysis plots")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise de Residuos - XGBoost", fontsize=16, fontweight="bold")

        # Calculate residuals
        residuals = y_test.values - predictions

        # 1. Residuals over time (top left)
        trend_style_resid = LINE_STYLES["trend"]
        axes[0, 0].plot(
            y_test.index,
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

        # Format date axis
        import matplotlib.dates as mdates

        axes[0, 0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 2. Residuals vs fitted values (top right)
        axes[0, 1].scatter(predictions, residuals, alpha=0.6)
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

        filename = output_dir / "06_xgboost_residual_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def create_performance_plots(
        self,
        y_test: pd.Series,
        predictions: np.ndarray,
        metrics: Dict[str, float],
        output_dir: Path,
    ) -> Dict[str, str]:
        """
        Create performance analysis plots.

        Args:
            y_test: Test target values
            predictions: Model predictions
            metrics: Performance metrics
            output_dir: Output directory

        Returns:
            Dictionary with paths to saved plot files
        """
        logger.info("Creating performance analysis plots")

        plot_files = {}

        # 1. Error analysis
        plot_files["errors"] = self._create_error_analysis(y_test, predictions, metrics, output_dir)

        # 2. Model summary
        plot_files["summary"] = self._create_model_summary(y_test, predictions, metrics, output_dir)

        return plot_files

    def _create_error_analysis(
        self,
        y_test: pd.Series,
        predictions: np.ndarray,
        metrics: Dict[str, float],
        output_dir: Path,
    ) -> str:
        """
        Create error analysis plots.

        Args:
            y_test: Test target values
            predictions: Model predictions
            metrics: Performance metrics
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating error analysis plots")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Analise de Erros - XGBoost", fontsize=16, fontweight="bold")

        # Extract metric values
        mae_score = metrics.get("mae", 0)
        rmse_score = metrics.get("rmse", 0)
        mape_score = metrics.get("mape", 0)

        # 1. Actual vs Predicted scatter (top left)
        axes[0, 0].scatter(y_test.values, predictions, alpha=0.7)

        # Perfect prediction line
        min_val = min(min(y_test.values), min(predictions))
        max_val = max(max(y_test.values), max(predictions))
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
        percentage_errors = np.abs((y_test.values - predictions) / y_test.values) * 100
        axes[1, 0].plot(
            y_test.index,
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
        cumulative_error = np.cumsum(np.abs(y_test.values - predictions))

        primary_style_cum = LINE_STYLES["primary_data"]
        axes[1, 1].plot(
            y_test.index,
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

        filename = output_dir / "07_xgboost_error_analysis.png"
        save_figure(fig, filename)

        return str(filename)

    def _create_model_summary(
        self,
        y_test: pd.Series,
        predictions: np.ndarray,
        metrics: Dict[str, float],
        output_dir: Path,
    ) -> str:
        """
        Create model summary plots.

        Args:
            y_test: Test target values
            predictions: Model predictions
            metrics: Performance metrics
            output_dir: Output directory

        Returns:
            Path to saved plot file
        """
        logger.info("Creating model summary plot")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_CONFIG["figsize"])
        fig.suptitle("Resumo do Modelo - XGBoost", fontsize=16, fontweight="bold")

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
XGBoost - Resumo:

Tipo: Gradient Boosting
Algoritmo: XGBoost Regressor
Features: Tabular com lag e cyclicas

Metricas de Performance:
• MAE: {metrics.get("mae", 0):.2f}
• RMSE: {metrics.get("rmse", 0):.2f}
• MAPE: {metrics.get("mape", 0):.2f}%

Periodo de Teste: {len(y_test)} observacoes
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
        correlation = np.corrcoef(y_test.values, predictions)[0, 1]
        directional_accuracy = (
            np.mean(np.sign(np.diff(y_test.values)) == np.sign(np.diff(predictions))) * 100
        )

        quality_info = f"""
Qualidade da Previsao:

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
Conclusoes - XGBoost:

Qualidade: {quality}
• MAPE < 10%: Excelente
• MAPE < 20%: Boa
• MAPE < 30%: Regular
• MAPE > 30%: Revisar modelo

XGBoost e eficaz para:
• Features tabulares complexas
• Relacoes nao-lineares
• Feature engineering avancado
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

        filename = output_dir / "08_xgboost_model_summary.png"
        save_figure(fig, filename)

        return str(filename)
