"""
STL Decomposition Module

Implements Seasonal and Trend decomposition using LOESS (STL) for time series analysis.
Provides functions to decompose time series into trend, seasonal, and residual components.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


class STLDecomposer:
    """
    Performs STL (Seasonal and Trend decomposition using LOESS) on time series data.

    STL is a robust method for decomposing a time series into three components:
    - Trend: Long-term progression of the series
    - Seasonal: Periodic fluctuations
    - Residual: Remainder after removing trend and seasonal components
    """

    def __init__(
        self,
        seasonal: int = 13,
        trend: Optional[int] = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        robust: bool = False,
    ):
        """
        Initialize STL decomposer with configuration parameters.

        Parameters
        ----------
        seasonal : int, default=13
            Length of the seasonal smoother. Must be odd and >= 7.
            Recommended: seasonal period + (even number)
            For monthly data with yearly seasonality: 13 (12 + 1)
        trend : int, optional
            Length of the trend smoother. Must be odd.
            If None, defaults to the smallest odd integer >= 1.5 * seasonal / (1 - 1.5 / seasonal_deg)
        seasonal_deg : int, default=1
            Degree of locally weighted regression for seasonal component (0 or 1)
        trend_deg : int, default=1
            Degree of locally weighted regression for trend component (0 or 1)
        robust : bool, default=False
            If True, uses robust weights to reduce influence of outliers
        """
        self.seasonal = seasonal
        self.trend = trend
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.robust = robust
        self.result = None

        logger.info(
            f"STLDecomposer initialized: seasonal={seasonal}, trend={trend}, robust={robust}"
        )

    def decompose(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual components.

        Parameters
        ----------
        series : pd.Series
            Time series with datetime index

        Returns
        -------
        trend : pd.Series
            Trend component
        seasonal : pd.Series
            Seasonal component
        residual : pd.Series
            Residual component
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")

        if series.isna().any():
            logger.warning("Series contains missing values, will be forward-filled")
            series = series.fillna(method="ffill")

        logger.info(f"Performing STL decomposition on series with {len(series)} points")

        try:
            stl = STL(
                series,
                seasonal=self.seasonal,
                trend=self.trend,
                seasonal_deg=self.seasonal_deg,
                trend_deg=self.trend_deg,
                robust=self.robust,
            )
            self.result = stl.fit()

            logger.info("STL decomposition completed successfully")

            return self.result.trend, self.result.seasonal, self.result.resid

        except Exception as e:
            logger.error(f"STL decomposition failed: {e}")
            raise

    def get_strength_of_trend(self) -> float:
        """
        Calculate strength of trend component.

        Returns
        -------
        float
            Strength of trend (0 to 1, higher = stronger trend)
        """
        if self.result is None:
            raise ValueError("Must call decompose() first")

        detrended = self.result.observed - self.result.trend
        strength = max(0, 1 - (np.var(self.result.resid) / np.var(detrended)))

        logger.info(f"Trend strength: {strength:.4f}")
        return strength

    def get_strength_of_seasonality(self) -> float:
        """
        Calculate strength of seasonal component.

        Returns
        -------
        float
            Strength of seasonality (0 to 1, higher = stronger seasonality)
        """
        if self.result is None:
            raise ValueError("Must call decompose() first")

        deseasonalized = self.result.observed - self.result.seasonal
        strength = max(0, 1 - (np.var(self.result.resid) / np.var(deseasonalized)))

        logger.info(f"Seasonal strength: {strength:.4f}")
        return strength

    def plot_decomposition(
        self,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 10),
    ):
        """
        Create visualization of STL decomposition components.

        Parameters
        ----------
        output_path : Path, optional
            Path to save the plot. If None, plot is displayed
        figsize : tuple, default=(12, 10)
            Figure size (width, height)
        """
        if self.result is None:
            raise ValueError("Must call decompose() first")

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # Original series
        axes[0].plot(self.result.observed, label="Original", color="#2E86AB")
        axes[0].set_ylabel("Original")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(self.result.trend, label="Trend", color="#A23B72")
        axes[1].set_ylabel("Trend")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(self.result.seasonal, label="Seasonal", color="#F18F01")
        axes[2].set_ylabel("Seasonal")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(self.result.resid, label="Residual", color="#C73E1D")
        axes[3].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        axes[3].set_ylabel("Residual")
        axes[3].set_xlabel("Date")
        axes[3].legend(loc="upper left")
        axes[3].grid(True, alpha=0.3)

        plt.suptitle(
            "STL Decomposition: Trend, Seasonal, and Residual Components",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"STL decomposition plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def get_summary(self) -> dict:
        """
        Get summary statistics of decomposition.

        Returns
        -------
        dict
            Summary including component statistics and strength metrics
        """
        if self.result is None:
            raise ValueError("Must call decompose() first")

        summary = {
            "trend_strength": self.get_strength_of_trend(),
            "seasonal_strength": self.get_strength_of_seasonality(),
            "trend_stats": {
                "mean": float(self.result.trend.mean()),
                "std": float(self.result.trend.std()),
                "min": float(self.result.trend.min()),
                "max": float(self.result.trend.max()),
            },
            "seasonal_stats": {
                "mean": float(self.result.seasonal.mean()),
                "std": float(self.result.seasonal.std()),
                "min": float(self.result.seasonal.min()),
                "max": float(self.result.seasonal.max()),
            },
            "residual_stats": {
                "mean": float(self.result.resid.mean()),
                "std": float(self.result.resid.std()),
                "min": float(self.result.resid.min()),
                "max": float(self.result.resid.max()),
            },
        }

        return summary


def apply_stl_decomposition(
    series: pd.Series,
    seasonal_period: int = 12,
    robust: bool = True,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Convenience function to apply STL decomposition with recommended settings.

    Parameters
    ----------
    series : pd.Series
        Time series to decompose
    seasonal_period : int, default=12
        Period of seasonality (12 for monthly data with yearly seasonality)
    robust : bool, default=True
        Use robust fitting to reduce influence of outliers
    output_dir : Path, optional
        Directory to save decomposition plot

    Returns
    -------
    dict
        Dictionary containing trend, seasonal, residual components and summary
    """
    seasonal = seasonal_period + 1 if seasonal_period % 2 == 0 else seasonal_period

    decomposer = STLDecomposer(seasonal=seasonal, robust=robust)
    trend, seasonal_comp, residual = decomposer.decompose(series)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "stl_decomposition.png"
        decomposer.plot_decomposition(output_path=plot_path)

    summary = decomposer.get_summary()

    return {
        "trend": trend,
        "seasonal": seasonal_comp,
        "residual": residual,
        "summary": summary,
    }
