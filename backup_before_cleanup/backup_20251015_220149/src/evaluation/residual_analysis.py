"""
Residual Analysis Module

Provides comprehensive residual diagnostics for time series models including:
- Statistical tests (Ljung-Box, Jarque-Bera, normality)
- Visual diagnostics (Q-Q plots, ACF/PACF, histograms)
- Outlier detection and analysis
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

logger = logging.getLogger(__name__)


class ResidualAnalyzer:
    """
    Performs comprehensive residual analysis for time series models.

    Provides statistical tests and visualizations to assess:
    - Residual randomness (white noise)
    - Normality of residuals
    - Autocorrelation structure
    - Outliers and influential points
    """

    def __init__(self, residuals: np.ndarray, model_name: str = "Model"):
        """
        Initialize residual analyzer.

        Parameters
        ----------
        residuals : array-like
            Model residuals (actual - predicted)
        model_name : str, default='Model'
            Name of the model for labeling
        """
        self.residuals = np.array(residuals).flatten()
        self.model_name = model_name
        self.n = len(self.residuals)

        self.residuals = self.residuals[~np.isnan(self.residuals)]

        logger.info(
            f"ResidualAnalyzer initialized for {model_name} with {len(self.residuals)} residuals"
        )

    def ljung_box_test(self, lags: Optional[int] = None, return_full: bool = False) -> Dict:
        """
        Perform Ljung-Box test for autocorrelation in residuals.

        The Ljung-Box test checks whether residuals exhibit significant autocorrelation,
        which would indicate the model has not captured all temporal patterns.

        Null hypothesis: Residuals are independently distributed (no autocorrelation)
        Alternative: Residuals exhibit autocorrelation

        Parameters
        ----------
        lags : int, optional
            Number of lags to test. If None, uses min(10, n/5)
        return_full : bool, default=False
            If True, return full test results for all lags

        Returns
        -------
        dict
            Test statistics and p-values
        """
        if lags is None:
            lags = min(10, max(1, int(self.n / 5)))

        try:
            lb_result = acorr_ljungbox(self.residuals, lags=lags, return_df=True)

            is_white_noise = (lb_result["lb_pvalue"] > 0.05).all()

            logger.info(
                f"Ljung-Box test: {'PASS' if is_white_noise else 'FAIL'} "
                f"(p-value at lag {lags}: {lb_result['lb_pvalue'].iloc[-1]:.4f})"
            )

            result = {
                "test_name": "Ljung-Box",
                "statistic": float(lb_result["lb_stat"].iloc[-1]),
                "p_value": float(lb_result["lb_pvalue"].iloc[-1]),
                "lags_tested": lags,
                "is_white_noise": bool(is_white_noise),
                "interpretation": (
                    "Residuals appear to be white noise (no autocorrelation)"
                    if is_white_noise
                    else "Residuals show significant autocorrelation"
                ),
            }

            if return_full:
                result["full_results"] = lb_result

            return result

        except Exception as e:
            logger.error(f"Ljung-Box test failed: {e}")
            return {"test_name": "Ljung-Box", "error": str(e)}

    def jarque_bera_test(self) -> Dict:
        """
        Perform Jarque-Bera test for normality of residuals.

        Tests whether residuals follow a normal distribution based on skewness
        and kurtosis.

        Null hypothesis: Residuals are normally distributed
        Alternative: Residuals are not normally distributed

        Returns
        -------
        dict
            Test statistics and interpretation
        """
        try:
            jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(self.residuals)

            is_normal = jb_pvalue > 0.05

            logger.info(
                f"Jarque-Bera test: {'PASS' if is_normal else 'FAIL'} (p-value: {jb_pvalue:.4f})"
            )

            return {
                "test_name": "Jarque-Bera",
                "statistic": float(jb_stat),
                "p_value": float(jb_pvalue),
                "skewness": float(skew),
                "kurtosis": float(kurtosis),
                "is_normal": bool(is_normal),
                "interpretation": (
                    "Residuals appear to be normally distributed"
                    if is_normal
                    else "Residuals deviate significantly from normal distribution"
                ),
            }

        except Exception as e:
            logger.error(f"Jarque-Bera test failed: {e}")
            return {"test_name": "Jarque-Bera", "error": str(e)}

    def shapiro_wilk_test(self) -> Dict:
        """
        Perform Shapiro-Wilk test for normality (more powerful for small samples).

        Null hypothesis: Residuals are normally distributed
        Alternative: Residuals are not normally distributed

        Returns
        -------
        dict
            Test statistics and interpretation
        """
        try:
            if self.n > 5000:
                logger.warning("Sample size > 5000, Shapiro-Wilk may be too sensitive")

            stat, p_value = stats.shapiro(self.residuals)

            is_normal = p_value > 0.05

            logger.info(
                f"Shapiro-Wilk test: {'PASS' if is_normal else 'FAIL'} (p-value: {p_value:.4f})"
            )

            return {
                "test_name": "Shapiro-Wilk",
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": bool(is_normal),
                "interpretation": (
                    "Residuals appear to be normally distributed"
                    if is_normal
                    else "Residuals deviate from normal distribution"
                ),
            }

        except Exception as e:
            logger.error(f"Shapiro-Wilk test failed: {e}")
            return {"test_name": "Shapiro-Wilk", "error": str(e)}

    def detect_outliers(
        self, threshold: float = 3.0, method: str = "iqr"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Detect outliers in residuals.

        Parameters
        ----------
        threshold : float, default=3.0
            For 'zscore': number of standard deviations
            For 'iqr': multiplier for IQR
        method : str, default='iqr'
            Method to use: 'zscore' or 'iqr'

        Returns
        -------
        outlier_indices : ndarray
            Indices of detected outliers
        outlier_info : dict
            Information about detected outliers
        """
        if method == "zscore":
            z_scores = np.abs(stats.zscore(self.residuals))
            outlier_mask = z_scores > threshold
        elif method == "iqr":
            q1 = np.percentile(self.residuals, 25)
            q3 = np.percentile(self.residuals, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (self.residuals < lower_bound) | (self.residuals > upper_bound)
        else:
            raise ValueError(f"Unknown method: {method}")

        outlier_indices = np.where(outlier_mask)[0]
        n_outliers = len(outlier_indices)
        outlier_pct = 100 * n_outliers / self.n

        logger.info(f"Detected {n_outliers} outliers ({outlier_pct:.2f}%) using {method} method")

        outlier_info = {
            "method": method,
            "threshold": threshold,
            "n_outliers": n_outliers,
            "outlier_percentage": float(outlier_pct),
            "outlier_values": self.residuals[outlier_indices].tolist(),
        }

        return outlier_indices, outlier_info

    def get_summary_statistics(self) -> Dict:
        """
        Compute summary statistics for residuals.

        Returns
        -------
        dict
            Comprehensive summary statistics
        """
        return {
            "n_observations": int(self.n),
            "mean": float(np.mean(self.residuals)),
            "median": float(np.median(self.residuals)),
            "std": float(np.std(self.residuals)),
            "variance": float(np.var(self.residuals)),
            "min": float(np.min(self.residuals)),
            "max": float(np.max(self.residuals)),
            "q1": float(np.percentile(self.residuals, 25)),
            "q3": float(np.percentile(self.residuals, 75)),
            "skewness": float(stats.skew(self.residuals)),
            "kurtosis": float(stats.kurtosis(self.residuals)),
            "range": float(np.ptp(self.residuals)),
        }

    def run_all_tests(self) -> Dict:
        """
        Run all statistical tests and return comprehensive report.

        Returns
        -------
        dict
            Complete diagnostic report
        """
        logger.info(f"Running comprehensive residual analysis for {self.model_name}")

        report = {
            "model_name": self.model_name,
            "summary_statistics": self.get_summary_statistics(),
            "ljung_box_test": self.ljung_box_test(),
            "jarque_bera_test": self.jarque_bera_test(),
            "shapiro_wilk_test": self.shapiro_wilk_test(),
        }

        outlier_indices, outlier_info = self.detect_outliers()
        report["outlier_analysis"] = outlier_info

        passes_all = (
            report["ljung_box_test"].get("is_white_noise", False)
            and report["jarque_bera_test"].get("is_normal", False)
            and outlier_info["outlier_percentage"] < 5.0
        )

        report["overall_assessment"] = {
            "passes_all_tests": passes_all,
            "issues_detected": [],
        }

        if not report["ljung_box_test"].get("is_white_noise", False):
            report["overall_assessment"]["issues_detected"].append("Autocorrelation in residuals")

        if not report["jarque_bera_test"].get("is_normal", False):
            report["overall_assessment"]["issues_detected"].append("Non-normal residuals")

        if outlier_info["outlier_percentage"] > 5.0:
            report["overall_assessment"]["issues_detected"].append(
                f"High outlier rate ({outlier_info['outlier_percentage']:.1f}%)"
            )

        logger.info(f"Analysis complete: {'PASS' if passes_all else 'ISSUES DETECTED'}")

        return report

    def plot_diagnostics(
        self, output_path: Optional[Path] = None, figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Create comprehensive diagnostic plots.

        Parameters
        ----------
        output_path : Path, optional
            Path to save the plot
        figsize : tuple, default=(15, 10)
            Figure size
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.residuals, linewidth=1)
        ax1.axhline(y=0, color="r", linestyle="--", linewidth=1)
        ax1.set_title(f"Residual Plot - {self.model_name}", fontweight="bold")
        ax1.set_xlabel("Observation")
        ax1.set_ylabel("Residual")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.residuals, bins=30, edgecolor="black", alpha=0.7)
        ax2.set_title("Histogram of Residuals", fontweight="bold")
        ax2.set_xlabel("Residual Value")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        qqplot(self.residuals, line="s", ax=ax3)
        ax3.set_title("Q-Q Plot", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[2, 0])
        try:
            plot_acf(self.residuals, lags=min(40, self.n // 2), ax=ax4)
            ax4.set_title("Autocorrelation Function (ACF)", fontweight="bold")
        except Exception as e:
            logger.warning(f"Could not plot ACF: {e}")
            ax4.text(0.5, 0.5, "ACF plot unavailable", ha="center", va="center")

        ax5 = fig.add_subplot(gs[2, 1])
        try:
            plot_pacf(self.residuals, lags=min(40, self.n // 2), ax=ax5)
            ax5.set_title("Partial Autocorrelation Function (PACF)", fontweight="bold")
        except Exception as e:
            logger.warning(f"Could not plot PACF: {e}")
            ax5.text(0.5, 0.5, "PACF plot unavailable", ha="center", va="center")

        plt.suptitle(
            f"Residual Diagnostics - {self.model_name}",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Diagnostic plots saved to {output_path}")
        else:
            plt.show()

        plt.close()


def analyze_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model",
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Convenience function to perform complete residual analysis.

    Parameters
    ----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Predicted values from model
    model_name : str, default='Model'
        Name of the model
    output_dir : Path, optional
        Directory to save diagnostic plots

    Returns
    -------
    dict
        Complete residual analysis report
    """
    residuals = np.array(actual) - np.array(predicted)

    analyzer = ResidualAnalyzer(residuals, model_name)
    report = analyzer.run_all_tests()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{model_name.lower().replace(' ', '_')}_residuals.png"
        analyzer.plot_diagnostics(output_path=plot_path)

    return report
