"""
Modulo para criacao de EDA geral dos dados apos pre-processamento.
Analise exploratoria abrangente da serie temporal de vendas.
"""

import locale
import logging
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from darts import TimeSeries
from scipy import stats
from scipy.stats import jarque_bera, normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from .plot_config import (
    apply_plot_style,
    configure_axes,
    save_figure,
)

# Configurar locale para formato brasileiro
try:
    locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, "Portuguese_Brazil.1252")
    except locale.Error:
        pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings desnecessarios
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class GeneralEDAPlotter:
    """
    Classe para criacao de plots de EDA geral dos dados preprocessados.
    """

    def __init__(self, output_dir: Path):
        """
        Inicializa o plotter com diretorio de saida.

        Args:
            output_dir: Diretorio onde salvar os plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Aplicar estilo base
        apply_plot_style()

        # Configurar nomes dos meses em portugues
        self.meses_pt = [
            "Janeiro",
            "Fevereiro",
            "Marco",
            "Abril",
            "Maio",
            "Junho",
            "Julho",
            "Agosto",
            "Setembro",
            "Outubro",
            "Novembro",
            "Dezembro",
        ]

    def format_currency(self, value: float) -> str:
        """
        Formata valor como moeda brasileira.

        Args:
            value: Valor numerico

        Returns:
            String formatada como moeda
        """
        try:
            return f"R$ {value:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except (ValueError, TypeError):
            return f"R$ {value}"

    def format_percentage(self, value: float) -> str:
        """
        Formata valor como percentual brasileiro.

        Args:
            value: Valor numerico

        Returns:
            String formatada como percentual
        """
        try:
            return f"{value:.1f}%".replace(".", ",")
        except (ValueError, TypeError):
            return f"{value}%"

    def load_time_series_data(self, data_path: Path) -> Tuple[pd.DataFrame, TimeSeries]:
        """
        Carrega dados da serie temporal.

        Args:
            data_path: Caminho para arquivo parquet

        Returns:
            Tuple com DataFrame pandas e TimeSeries darts
        """
        logger.info(f"Carregando dados de {data_path}")

        # Carregar dados
        df = pd.read_parquet(data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Criar TimeSeries darts
        ts = TimeSeries.from_dataframe(df, time_col=None, value_cols=["target"])

        logger.info(
            f"Dados carregados: {len(df)} pontos, periodo {df.index.min()} a {df.index.max()}"
        )

        return df, ts

    def plot_01_overview_serie_temporal(self, df: pd.DataFrame, ts: TimeSeries) -> None:
        """
        Cria overview geral da serie temporal.

        Args:
            df: DataFrame com dados
            ts: TimeSeries darts
        """
        logger.info("Criando plot 1: Visao Geral da Serie Temporal")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Visao Geral da Serie Temporal de Vendas", fontsize=18, fontweight="bold", y=0.98
        )

        # Plot 1: Serie temporal completa
        ax1 = axes[0, 0]
        ax1.plot(df.index, df["target"], linewidth=2, color="#1f77b4", alpha=0.8)
        configure_axes(
            ax1, title="Serie Temporal Completa", xlabel="Data", ylabel="Valor das Vendas (R$)"
        )

        # Formato dos valores no eixo Y
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 2: Tendencia media movel
        ax2 = axes[0, 1]
        ma_3 = df["target"].rolling(window=3).mean()
        ma_12 = df["target"].rolling(window=12).mean()

        ax2.plot(
            df.index, df["target"], linewidth=1, alpha=0.5, color="#1f77b4", label="Serie Original"
        )
        ax2.plot(df.index, ma_3, linewidth=2, color="#ff7f0e", label="Media Movel 3 Meses")
        ax2.plot(df.index, ma_12, linewidth=2, color="#2ca02c", label="Media Movel 12 Meses")

        configure_axes(
            ax2, title="Tendencias - Medias Moveis", xlabel="Data", ylabel="Valor das Vendas (R$)"
        )
        ax2.legend()
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 3: Distribuicao mensal
        ax3 = axes[1, 0]
        df_monthly = df.copy()
        df_monthly["mes"] = df_monthly.index.month
        monthly_data = [df_monthly[df_monthly["mes"] == i]["target"].values for i in range(1, 13)]

        box_plot = ax3.boxplot(monthly_data, labels=self.meses_pt, patch_artist=True)
        for patch in box_plot["boxes"]:
            patch.set_facecolor("#5490d3")
            patch.set_alpha(0.8)

        configure_axes(
            ax3,
            title="Distribuicao Mensal das Vendas",
            xlabel="Mes",
            ylabel="Valor das Vendas (R$)",
        )
        ax3.tick_params(axis="x", rotation=45)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 4: Estatisticas principais
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Calcular estatisticas
        stats_data = {
            "Total de Observacoes": len(df),
            "Periodo": f"{df.index.min().strftime('%m/%Y')} a {df.index.max().strftime('%m/%Y')}",
            "Valor Total": self.format_currency(df["target"].sum()),
            "Valor Medio Mensal": self.format_currency(df["target"].mean()),
            "Valor Mediano": self.format_currency(df["target"].median()),
            "Desvio Padrao": self.format_currency(df["target"].std()),
            "Coeficiente de Variacao": self.format_percentage(
                (df["target"].std() / df["target"].mean()) * 100
            ),
            "Valor Minimo": self.format_currency(df["target"].min()),
            "Valor Maximo": self.format_currency(df["target"].max()),
        }

        # Exibir estatisticas
        y_pos = 0.9
        for key, value in stats_data.items():
            ax4.text(0.1, y_pos, f"{key}:", fontsize=11, fontweight="bold")
            ax4.text(0.6, y_pos, str(value), fontsize=11)
            y_pos -= 0.09

        ax4.set_title("Estatisticas Descritivas", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()
        save_figure(fig, self.output_dir / "01_visao_geral_serie_temporal.png")
        logger.info("Plot 1 salvo com sucesso")

    def plot_02_decomposicao_stl(self, ts: TimeSeries) -> None:
        """
        Cria analise de decomposicao STL.

        Args:
            ts: TimeSeries darts
        """
        logger.info("Criando plot 2: Analise de Decomposicao STL")

        try:
            # Realizar decomposicao STL usando statsmodels
            from statsmodels.tsa.seasonal import STL

            # Converter para pandas para decomposicao
            df = ts.to_dataframe()

            # Aplicar decomposicao STL
            stl = STL(df.iloc[:, 0], seasonal=13, robust=True)
            decomposition = stl.fit()

            # Extrair componentes como arrays numpy
            trend_values = decomposition.trend.values
            seasonal_values = decomposition.seasonal.values
            residual_values = decomposition.resid.values
            dates = df.index

            fig, axes = plt.subplots(4, 1, figsize=(16, 14))
            fig.suptitle(
                "Decomposicao STL da Serie Temporal", fontsize=18, fontweight="bold", y=0.98
            )

            # Serie original
            ax1 = axes[0]
            ax1.plot(dates, df.iloc[:, 0].values, linewidth=2, color="#1f77b4")
            configure_axes(ax1, title="Serie Original", xlabel="", ylabel="Vendas (R$)")
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

            # Tendencia
            ax2 = axes[1]
            ax2.plot(dates, trend_values, linewidth=2, color="#ff7f0e")
            configure_axes(ax2, title="Componente de Tendencia", xlabel="", ylabel="Tendencia (R$)")
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

            # Sazonalidade
            ax3 = axes[2]
            ax3.plot(dates, seasonal_values, linewidth=2, color="#2ca02c")
            configure_axes(ax3, title="Componente Sazonal", xlabel="", ylabel="Sazonalidade (R$)")
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

            # Residuos
            ax4 = axes[3]
            ax4.plot(dates, residual_values, linewidth=1, color="#d62728", alpha=0.7)
            configure_axes(ax4, title="Residuos", xlabel="Data", ylabel="Residuos (R$)")
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

            plt.tight_layout()
            save_figure(fig, self.output_dir / "02_analise_decomposicao.png")
            logger.info("Plot 2 salvo com sucesso")

        except Exception as e:
            logger.error(f"Erro na decomposicao STL: {e}")
            # Criar plot alternativo com decomposicao simples
            self._plot_decomposicao_alternativa(ts)

    def _plot_decomposicao_alternativa(self, ts: TimeSeries) -> None:
        """
        Cria decomposicao alternativa se STL falhar.

        Args:
            ts: TimeSeries darts
        """
        logger.info("Criando decomposicao alternativa")

        # Converter para pandas para decomposicao statsmodels
        df = ts.to_dataframe()

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            decomp = seasonal_decompose(df.iloc[:, 0], model="additive", period=12)

            fig, axes = plt.subplots(4, 1, figsize=(16, 14))
            fig.suptitle("Decomposicao da Serie Temporal", fontsize=18, fontweight="bold", y=0.98)

            # Serie original
            axes[0].plot(df.index, df.iloc[:, 0], linewidth=2, color="#1f77b4")
            configure_axes(axes[0], title="Serie Original", ylabel="Vendas (R$)")
            axes[0].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: self.format_currency(x))
            )

            # Tendencia
            axes[1].plot(df.index, decomp.trend, linewidth=2, color="#ff7f0e")
            configure_axes(axes[1], title="Componente de Tendencia", ylabel="Tendencia (R$)")
            axes[1].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: self.format_currency(x))
            )

            # Sazonalidade
            axes[2].plot(df.index, decomp.seasonal, linewidth=2, color="#2ca02c")
            configure_axes(axes[2], title="Componente Sazonal", ylabel="Sazonalidade (R$)")
            axes[2].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: self.format_currency(x))
            )

            # Residuos
            axes[3].plot(df.index, decomp.resid, linewidth=1, color="#d62728", alpha=0.7)
            configure_axes(axes[3], title="Residuos", xlabel="Data", ylabel="Residuos (R$)")
            axes[3].yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: self.format_currency(x))
            )

            plt.tight_layout()
            save_figure(fig, self.output_dir / "02_analise_decomposicao.png")
            logger.info("Plot 2 alternativo salvo com sucesso")

        except Exception as e:
            logger.error(f"Erro na decomposicao alternativa: {e}")

    def plot_03_analise_sazonalidade(self, df: pd.DataFrame) -> None:
        """
        Cria analise detalhada de sazonalidade.

        Args:
            df: DataFrame com dados
        """
        logger.info("Criando plot 3: Analise de Sazonalidade")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Padroes Sazonais das Vendas", fontsize=18, fontweight="bold", y=0.98)

        # Preparar dados
        df_sazonal = df.copy()
        df_sazonal["mes"] = df_sazonal.index.month
        df_sazonal["ano"] = df_sazonal.index.year
        df_sazonal["trimestre"] = df_sazonal.index.quarter

        # Plot 1: Boxplot mensal
        ax1 = axes[0, 0]
        monthly_data = [df_sazonal[df_sazonal["mes"] == i]["target"].values for i in range(1, 13)]
        box_plot = ax1.boxplot(monthly_data, labels=self.meses_pt, patch_artist=True)

        for patch in box_plot["boxes"]:
            patch.set_facecolor("#5490d3")
            patch.set_alpha(0.8)

        configure_axes(
            ax1,
            title="Distribuicao Mensal das Vendas",
            xlabel="Mes",
            ylabel="Valor das Vendas (R$)",
        )
        ax1.tick_params(axis="x", rotation=45)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 2: Heatmap anual
        ax2 = axes[0, 1]
        pivot_data = df_sazonal.pivot_table(
            values="target", index="ano", columns="mes", aggfunc="sum"
        )

        sns.heatmap(
            pivot_data,
            ax=ax2,
            cmap="YlOrRd",
            fmt=".0f",
            xticklabels=self.meses_pt,
            annot=False,
            cbar_kws={"label": "Vendas (R$)"},
        )
        configure_axes(ax2, title="Heatmap de Vendas por Mes e Ano", xlabel="Mes", ylabel="Ano")

        # Plot 3: Perfil sazonal medio
        ax3 = axes[1, 0]
        perfil_mensal = df_sazonal.groupby("mes")["target"].mean()
        ax3.bar(range(1, 13), perfil_mensal.values, color="#5490d3", alpha=0.8, edgecolor="#1f77b4")
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(self.meses_pt, rotation=45)

        configure_axes(ax3, title="Perfil Sazonal Medio", xlabel="Mes", ylabel="Vendas Medias (R$)")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 4: Analise trimestral
        ax4 = axes[1, 1]
        trimestral = df_sazonal.groupby("trimestre")["target"].agg(["mean", "std"])

        x_pos = range(1, 5)
        ax4.bar(
            x_pos,
            trimestral["mean"].values,
            yerr=trimestral["std"].values,
            color="#5490d3",
            alpha=0.8,
            edgecolor="#1f77b4",
            capsize=5,
        )
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f"Q{i}" for i in range(1, 5)])

        configure_axes(
            ax4,
            title="Vendas por Trimestre (Media ± Desvio)",
            xlabel="Trimestre",
            ylabel="Vendas Medias (R$)",
        )
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        plt.tight_layout()
        save_figure(fig, self.output_dir / "03_analise_sazonalidade.png")
        logger.info("Plot 3 salvo com sucesso")

    def plot_04_propriedades_estatisticas(self, df: pd.DataFrame, ts: TimeSeries) -> None:
        """
        Cria analise de propriedades estatisticas.

        Args:
            df: DataFrame com dados
            ts: TimeSeries darts
        """
        logger.info("Criando plot 4: Propriedades Estatisticas")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Propriedades Estatisticas da Serie Temporal", fontsize=18, fontweight="bold", y=0.98
        )

        # Plot 1: Autocorrelacao (ACF)
        ax1 = axes[0, 0]
        try:
            from statsmodels.tsa.stattools import acf

            autocorr = acf(df["target"].dropna(), nlags=24, alpha=0.05)

            lags = range(len(autocorr[0]))
            ax1.plot(lags, autocorr[0], "b-", linewidth=2, label="ACF")
            ax1.fill_between(
                lags,
                autocorr[1][:, 0] - autocorr[0],
                autocorr[1][:, 1] - autocorr[0],
                alpha=0.2,
                color="blue",
            )
            ax1.axhline(0, color="black", linewidth=0.5)

            configure_axes(
                ax1,
                title="Funcao de Autocorrelacao (ACF)",
                xlabel="Lag (Meses)",
                ylabel="Autocorrelacao",
            )

        except Exception as e:
            logger.warning(f"Erro no calculo ACF: {e}")
            ax1.text(
                0.5, 0.5, "Erro no calculo ACF", ha="center", va="center", transform=ax1.transAxes
            )

        # Plot 2: Autocorrelacao Parcial (PACF)
        ax2 = axes[0, 1]
        try:
            from statsmodels.tsa.stattools import pacf

            partial_autocorr = pacf(df["target"].dropna(), nlags=24, alpha=0.05)

            lags = range(len(partial_autocorr[0]))
            ax2.plot(lags, partial_autocorr[0], "r-", linewidth=2, label="PACF")
            ax2.fill_between(
                lags,
                partial_autocorr[1][:, 0] - partial_autocorr[0],
                partial_autocorr[1][:, 1] - partial_autocorr[0],
                alpha=0.2,
                color="red",
            )
            ax2.axhline(0, color="black", linewidth=0.5)

            configure_axes(
                ax2,
                title="Funcao de Autocorrelacao Parcial (PACF)",
                xlabel="Lag (Meses)",
                ylabel="Autocorrelacao Parcial",
            )

        except Exception as e:
            logger.warning(f"Erro no calculo PACF: {e}")
            ax2.text(
                0.5, 0.5, "Erro no calculo PACF", ha="center", va="center", transform=ax2.transAxes
            )

        # Plot 3: Testes de estacionariedade
        ax3 = axes[1, 0]
        ax3.axis("off")

        # Teste ADF
        try:
            adf_result = adfuller(df["target"].dropna())
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_critical = adf_result[4]

            # Teste KPSS
            kpss_result = kpss(df["target"].dropna(), regression="ct")
            kpss_statistic = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_critical = kpss_result[3]

            # Exibir resultados
            results_text = [
                "TESTES DE ESTACIONARIEDADE",
                "",
                "Teste ADF (Augmented Dickey-Fuller):",
                f"  Estatistica: {adf_statistic:.4f}",
                f"  p-valor: {adf_pvalue:.4f}",
                "  Valores criticos:",
                f"    1%: {adf_critical['1%']:.4f}",
                f"    5%: {adf_critical['5%']:.4f}",
                f"    10%: {adf_critical['10%']:.4f}",
                f"  Resultado: {'Estacionaria' if adf_pvalue < 0.05 else 'Nao Estacionaria'}",
                "",
                "Teste KPSS (Kwiatkowski-Phillips-Schmidt-Shin):",
                f"  Estatistica: {kpss_statistic:.4f}",
                f"  p-valor: {kpss_pvalue:.4f}",
                "  Valores criticos:",
                f"    1%: {kpss_critical['1%']:.4f}",
                f"    5%: {kpss_critical['5%']:.4f}",
                f"    10%: {kpss_critical['10%']:.4f}",
                f"  Resultado: {'Estacionaria' if kpss_pvalue > 0.05 else 'Nao Estacionaria'}",
            ]

            y_pos = 0.95
            for line in results_text:
                if line.startswith("TESTES"):
                    ax3.text(
                        0.02, y_pos, line, fontsize=12, fontweight="bold", transform=ax3.transAxes
                    )
                elif line.startswith("Teste"):
                    ax3.text(
                        0.02, y_pos, line, fontsize=11, fontweight="bold", transform=ax3.transAxes
                    )
                elif line.startswith("  Resultado:"):
                    color = "green" if "Estacionaria" in line and "Nao" not in line else "red"
                    ax3.text(
                        0.02,
                        y_pos,
                        line,
                        fontsize=10,
                        color=color,
                        fontweight="bold",
                        transform=ax3.transAxes,
                    )
                else:
                    ax3.text(0.02, y_pos, line, fontsize=10, transform=ax3.transAxes)
                y_pos -= 0.045

        except Exception as e:
            logger.warning(f"Erro nos testes de estacionariedade: {e}")
            ax3.text(
                0.5,
                0.5,
                "Erro nos testes de estacionariedade",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )

        # Plot 4: Estatisticas resumo
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Calcular estatisticas adicionais
        data_values = df["target"].dropna()

        # Teste de normalidade
        jb_stat, jb_pvalue = jarque_bera(data_values)
        sw_stat, sw_pvalue = normaltest(data_values)

        # Teste de independencia (Ljung-Box)
        try:
            lb_result = acorr_ljungbox(data_values, lags=10, return_df=True)
            lb_pvalue = lb_result["lb_pvalue"].iloc[-1]
        except Exception:
            lb_pvalue = None

        stats_text = [
            "TESTES ESTATISTICOS",
            "",
            "Normalidade:",
            f"  Jarque-Bera: {jb_stat:.4f} (p={jb_pvalue:.4f})",
            f"  D'Agostino-Pearson: {sw_stat:.4f} (p={sw_pvalue:.4f})",
            f"  Normal: {'Sim' if jb_pvalue > 0.05 else 'Nao'}",
            "",
            "Autocorrelacao:",
            f"  Ljung-Box (lag 10): {lb_pvalue:.4f}"
            if lb_pvalue
            else "  Ljung-Box: Nao disponivel",
            f"  Independente: {'Sim' if lb_pvalue and lb_pvalue > 0.05 else 'Nao'}"
            if lb_pvalue
            else "",
            "",
            "Estatisticas Descritivas:",
            f"  Assimetria: {stats.skew(data_values):.4f}",
            f"  Curtose: {stats.kurtosis(data_values):.4f}",
            f"  CV: {(data_values.std() / data_values.mean()):.4f}",
        ]

        y_pos = 0.95
        for line in stats_text:
            if line.startswith("TESTES"):
                ax4.text(0.02, y_pos, line, fontsize=12, fontweight="bold", transform=ax4.transAxes)
            elif line.endswith(":") and not line.startswith("  "):
                ax4.text(0.02, y_pos, line, fontsize=11, fontweight="bold", transform=ax4.transAxes)
            else:
                ax4.text(0.02, y_pos, line, fontsize=10, transform=ax4.transAxes)
            y_pos -= 0.06

        plt.tight_layout()
        save_figure(fig, self.output_dir / "04_propriedades_estatisticas.png")
        logger.info("Plot 4 salvo com sucesso")

    def plot_05_analise_distribuicao(self, df: pd.DataFrame) -> None:
        """
        Cria analise de distribuicao dos valores.

        Args:
            df: DataFrame com dados
        """
        logger.info("Criando plot 5: Analise de Distribuicao")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Distribuicao dos Valores de Vendas", fontsize=18, fontweight="bold", y=0.98)

        data_values = df["target"].dropna()

        # Plot 1: Histograma com densidade
        ax1 = axes[0, 0]
        ax1.hist(
            data_values, bins=30, density=True, alpha=0.7, color="#5490d3", edgecolor="#1f77b4"
        )

        # Adicionar curva de densidade normal para comparacao
        mu, sigma = data_values.mean(), data_values.std()
        x = np.linspace(data_values.min(), data_values.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, normal_curve, "r-", linewidth=2, label="Distribuicao Normal")

        configure_axes(
            ax1,
            title="Histograma e Densidade",
            xlabel="Valor das Vendas (R$)",
            ylabel="Densidade de Probabilidade",
        )
        ax1.legend()
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 2: Q-Q Plot
        ax2 = axes[0, 1]
        stats.probplot(data_values, dist="norm", plot=ax2)
        ax2.get_lines()[0].set_markerfacecolor("#5490d3")
        ax2.get_lines()[0].set_markeredgecolor("#1f77b4")
        ax2.get_lines()[1].set_color("#d62728")

        configure_axes(
            ax2,
            title="Q-Q Plot (Normalidade)",
            xlabel="Quantis Teoricos",
            ylabel="Quantis Amostrais",
        )

        # Plot 3: Box Plot com outliers
        ax3 = axes[1, 0]
        box_plot = ax3.boxplot(data_values, patch_artist=True, vert=True)
        box_plot["boxes"][0].set_facecolor("#5490d3")
        box_plot["boxes"][0].set_alpha(0.8)

        # Identificar outliers
        Q1 = data_values.quantile(0.25)
        Q3 = data_values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data_values[(data_values < lower_bound) | (data_values > upper_bound)]

        configure_axes(
            ax3,
            title=f"Box Plot (Outliers: {len(outliers)})",
            xlabel="",
            ylabel="Valor das Vendas (R$)",
        )
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 4: Estatisticas de distribuicao
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Calcular percentis
        percentis = [5, 10, 25, 50, 75, 90, 95]
        percentil_values = [data_values.quantile(p / 100) for p in percentis]

        dist_stats = [
            "ESTATISTICAS DE DISTRIBUICAO",
            "",
            "Medidas de Tendencia Central:",
            f"  Media: {self.format_currency(data_values.mean())}",
            f"  Mediana: {self.format_currency(data_values.median())}",
            f"  Moda: {self.format_currency(data_values.mode().iloc[0] if not data_values.mode().empty else 0)}",
            "",
            "Medidas de Dispersao:",
            f"  Desvio Padrao: {self.format_currency(data_values.std())}",
            f"  Variancia: {self.format_currency(data_values.var())}",
            f"  Coef. Variacao: {self.format_percentage((data_values.std() / data_values.mean()) * 100)}",
            "",
            "Medidas de Forma:",
            f"  Assimetria: {stats.skew(data_values):.4f}",
            f"  Curtose: {stats.kurtosis(data_values):.4f}",
            "",
            "Percentis Principais:",
            f"  P5: {self.format_currency(percentil_values[0])}",
            f"  P25: {self.format_currency(percentil_values[2])}",
            f"  P50: {self.format_currency(percentil_values[3])}",
            f"  P75: {self.format_currency(percentil_values[4])}",
            f"  P95: {self.format_currency(percentil_values[6])}",
        ]

        y_pos = 0.95
        for line in dist_stats:
            if line.startswith("ESTATISTICAS"):
                ax4.text(0.02, y_pos, line, fontsize=12, fontweight="bold", transform=ax4.transAxes)
            elif line.endswith(":") and not line.startswith("  "):
                ax4.text(0.02, y_pos, line, fontsize=11, fontweight="bold", transform=ax4.transAxes)
            else:
                ax4.text(0.02, y_pos, line, fontsize=10, transform=ax4.transAxes)
            y_pos -= 0.042

        plt.tight_layout()
        save_figure(fig, self.output_dir / "05_analise_distribuicao.png")
        logger.info("Plot 5 salvo com sucesso")

    def plot_06_evolucao_temporal(self, df: pd.DataFrame) -> None:
        """
        Cria analise de evolucao temporal detalhada.

        Args:
            df: DataFrame com dados
        """
        logger.info("Criando plot 6: Evolucao Temporal")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Evolucao Temporal das Vendas", fontsize=18, fontweight="bold", y=0.98)

        # Preparar dados anuais
        df_anual = df.copy()
        df_anual["ano"] = df_anual.index.year
        vendas_anuais = df_anual.groupby("ano")["target"].sum()

        # Plot 1: Evolucao anual
        ax1 = axes[0, 0]
        ax1.plot(
            vendas_anuais.index,
            vendas_anuais.values,
            marker="o",
            linewidth=3,
            markersize=8,
            color="#1f77b4",
            markerfacecolor="#ff7f0e",
        )

        configure_axes(
            ax1, title="Evolucao das Vendas Anuais", xlabel="Ano", ylabel="Vendas Anuais (R$)"
        )
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 2: Taxa de crescimento anual
        ax2 = axes[0, 1]
        crescimento = vendas_anuais.pct_change() * 100
        colors = ["green" if x > 0 else "red" for x in crescimento.dropna()]

        ax2.bar(crescimento.index[1:], crescimento.dropna().values, color=colors, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.5)

        configure_axes(
            ax2, title="Taxa de Crescimento Anual", xlabel="Ano", ylabel="Crescimento (%)"
        )
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_percentage(x)))

        # Plot 3: Vendas mensais com destaque sazonal
        ax3 = axes[1, 0]
        df_mensal = df.copy()
        df_mensal["mes"] = df_mensal.index.month
        df_mensal["ano"] = df_mensal.index.year

        # Plotar alguns anos para mostrar sazonalidade
        anos_recentes = sorted(df_mensal["ano"].unique())[-5:]  # Últimos 5 anos
        for ano in anos_recentes:
            dados_ano = df_mensal[df_mensal["ano"] == ano]
            vendas_mensais = dados_ano.groupby("mes")["target"].sum()
            ax3.plot(
                vendas_mensais.index,
                vendas_mensais.values,
                marker="o",
                linewidth=2,
                label=str(ano),
                alpha=0.8,
            )

        configure_axes(
            ax3, title="Padroes Sazonais por Ano", xlabel="Mes", ylabel="Vendas Mensais (R$)"
        )
        ax3.legend()
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels([m[:3] for m in self.meses_pt])
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Plot 4: Analise de tendencia
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Calcular estatisticas de tendencia
        from scipy import stats as scipy_stats

        # Tendencia linear nos dados anuais
        anos = vendas_anuais.index.values
        valores = vendas_anuais.values
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(anos, valores)

        # Crescimento medio
        crescimento_medio = crescimento.mean()
        crescimento_mediano = crescimento.median()

        # Volatilidade
        cv_anual = (vendas_anuais.std() / vendas_anuais.mean()) * 100
        cv_mensal = (df["target"].std() / df["target"].mean()) * 100

        tendencia_stats = [
            "ANALISE DE TENDENCIA",
            "",
            "Tendencia Linear:",
            f"  Inclinacao: {self.format_currency(slope)}/ano",
            f"  R²: {r_value**2:.4f}",
            f"  p-valor: {p_value:.4f}",
            f"  Significativa: {'Sim' if p_value < 0.05 else 'Nao'}",
            "",
            "Crescimento:",
            f"  Medio Anual: {self.format_percentage(crescimento_medio)}",
            f"  Mediano Anual: {self.format_percentage(crescimento_mediano)}",
            f"  Desvio do Crescimento: {self.format_percentage(crescimento.std())}",
            "",
            "Volatilidade:",
            f"  CV Anual: {self.format_percentage(cv_anual)}",
            f"  CV Mensal: {self.format_percentage(cv_mensal)}",
            "",
            "Periodo de Analise:",
            f"  Inicio: {df.index.min().strftime('%m/%Y')}",
            f"  Fim: {df.index.max().strftime('%m/%Y')}",
            f"  Duracao: {len(df)} meses",
        ]

        y_pos = 0.95
        for line in tendencia_stats:
            if line.startswith("ANALISE"):
                ax4.text(0.02, y_pos, line, fontsize=12, fontweight="bold", transform=ax4.transAxes)
            elif line.endswith(":") and not line.startswith("  "):
                ax4.text(0.02, y_pos, line, fontsize=11, fontweight="bold", transform=ax4.transAxes)
            else:
                ax4.text(0.02, y_pos, line, fontsize=10, transform=ax4.transAxes)
            y_pos -= 0.045

        plt.tight_layout()
        save_figure(fig, self.output_dir / "06_evolucao_temporal.png")
        logger.info("Plot 6 salvo com sucesso")

    def plot_07_analise_correlacao(self, df: pd.DataFrame) -> None:
        """
        Cria analise de correlacao e dependencias.

        Args:
            df: DataFrame com dados
        """
        logger.info("Criando plot 7: Analise de Correlacao")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Analise de Correlacao Temporal", fontsize=18, fontweight="bold", y=0.98)

        # Criar lags para analise de correlacao
        df_lags = df.copy()
        max_lags = 12

        for lag in range(1, max_lags + 1):
            df_lags[f"lag_{lag}"] = df_lags["target"].shift(lag)

        # Plot 1: Correlacao com lags
        ax1 = axes[0, 0]
        lag_columns = [f"lag_{i}" for i in range(1, max_lags + 1)]
        correlations = [df_lags["target"].corr(df_lags[col]) for col in lag_columns]

        ax1.bar(
            range(1, max_lags + 1), correlations, color="#5490d3", alpha=0.8, edgecolor="#1f77b4"
        )
        ax1.axhline(0, color="black", linewidth=0.5)

        configure_axes(ax1, title="Correlacao com Lags", xlabel="Lag (Meses)", ylabel="Correlacao")

        # Plot 2: Matriz de correlacao (lags selecionados)
        ax2 = axes[0, 1]
        selected_lags = ["target"] + [f"lag_{i}" for i in [1, 3, 6, 12]]
        corr_matrix = df_lags[selected_lags].corr()

        im = ax2.imshow(corr_matrix.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(selected_lags)))
        ax2.set_yticks(range(len(selected_lags)))
        ax2.set_xticklabels(["Atual", "Lag 1", "Lag 3", "Lag 6", "Lag 12"])
        ax2.set_yticklabels(["Atual", "Lag 1", "Lag 3", "Lag 6", "Lag 12"])

        # Adicionar valores na matriz
        for i in range(len(selected_lags)):
            for j in range(len(selected_lags)):
                ax2.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                )

        configure_axes(ax2, title="Matriz de Correlacao (Lags Selecionados)", xlabel="", ylabel="")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label("Correlacao")

        # Plot 3: Autocorrelacao parcial detalhada
        ax3 = axes[1, 0]
        try:
            from statsmodels.tsa.stattools import pacf

            pacf_values = pacf(df["target"].dropna(), nlags=max_lags)

            ax3.bar(
                range(max_lags + 1), pacf_values, color="#ff7f0e", alpha=0.8, edgecolor="#d62728"
            )
            ax3.axhline(0, color="black", linewidth=0.5)

            # Adicionar linhas de confianca
            confidence_interval = 1.96 / np.sqrt(len(df))
            ax3.axhline(confidence_interval, color="red", linestyle="--", alpha=0.7)
            ax3.axhline(-confidence_interval, color="red", linestyle="--", alpha=0.7)

            configure_axes(
                ax3, title="Autocorrelacao Parcial Detalhada", xlabel="Lag (Meses)", ylabel="PACF"
            )

        except Exception as e:
            logger.warning(f"Erro no PACF detalhado: {e}")
            ax3.text(
                0.5, 0.5, "Erro no calculo PACF", ha="center", va="center", transform=ax3.transAxes
            )

        # Plot 4: Estatisticas de dependencia
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Calcular estatisticas de dependencia
        autocorr_1 = df["target"].corr(df["target"].shift(1))
        autocorr_12 = df["target"].corr(df["target"].shift(12))

        # Durbin-Watson
        try:
            from statsmodels.stats.stattools import durbin_watson

            dw_stat = durbin_watson(df["target"].dropna().values)
        except Exception:
            dw_stat = None

        # Correlacao com componentes temporais
        df_temp = df.copy()
        df_temp["mes"] = df_temp.index.month
        df_temp["ano"] = df_temp.index.year - df_temp.index.year.min()

        corr_mes = df_temp["target"].corr(df_temp["mes"])
        corr_ano = df_temp["target"].corr(df_temp["ano"])

        dependencia_stats = [
            "ESTATISTICAS DE DEPENDENCIA",
            "",
            "Autocorrelacao:",
            f"  Lag 1: {autocorr_1:.4f}",
            f"  Lag 12: {autocorr_12:.4f}",
            f"  Media (lags 1-12): {np.mean(correlations):.4f}",
            "",
            "Testes de Independencia:",
            f"  Durbin-Watson: {dw_stat:.4f}" if dw_stat else "  Durbin-Watson: Nao disponivel",
            f"  Interpretacao: {'Positiva' if dw_stat and dw_stat < 1.5 else 'Negativa' if dw_stat and dw_stat > 2.5 else 'Nenhuma'}"
            if dw_stat
            else "",
            "",
            "Correlacao Temporal:",
            f"  Com Mes: {corr_mes:.4f}",
            f"  Com Ano: {corr_ano:.4f}",
            "",
            "Lags Significativos:",
            f"  Lag 1: {'Sim' if abs(autocorr_1) > 0.2 else 'Nao'}",
            f"  Lag 12: {'Sim' if abs(autocorr_12) > 0.2 else 'Nao'}",
            f"  Total > 0.2: {sum(1 for c in correlations if abs(c) > 0.2)}",
        ]

        y_pos = 0.95
        for line in dependencia_stats:
            if line.startswith("ESTATISTICAS"):
                ax4.text(0.02, y_pos, line, fontsize=12, fontweight="bold", transform=ax4.transAxes)
            elif line.endswith(":") and not line.startswith("  "):
                ax4.text(0.02, y_pos, line, fontsize=11, fontweight="bold", transform=ax4.transAxes)
            else:
                ax4.text(0.02, y_pos, line, fontsize=10, transform=ax4.transAxes)
            y_pos -= 0.045

        plt.tight_layout()
        save_figure(fig, self.output_dir / "07_analise_correlacao.png")
        logger.info("Plot 7 salvo com sucesso")

    def plot_08_resumo_executivo(self, df: pd.DataFrame, ts: TimeSeries) -> None:
        """
        Cria resumo executivo com dashboard de metricas.

        Args:
            df: DataFrame com dados
            ts: TimeSeries darts
        """
        logger.info("Criando plot 8: Resumo Executivo")

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            "Resumo Executivo - Caracteristicas dos Dados", fontsize=20, fontweight="bold", y=0.98
        )

        # Layout do dashboard
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # Calcular metricas principais
        total_vendas = df["target"].sum()
        media_mensal = df["target"].mean()
        mediana_mensal = df["target"].median()
        desvio_padrao = df["target"].std()
        cv = (desvio_padrao / media_mensal) * 100

        # Vendas anuais
        df_anual = df.copy()
        df_anual["ano"] = df_anual.index.year
        vendas_anuais = df_anual.groupby("ano")["target"].sum()
        crescimento_medio = vendas_anuais.pct_change().mean() * 100

        # Sazonalidade
        df_mensal = df.copy()
        df_mensal["mes"] = df_mensal.index.month
        sazonalidade = df_mensal.groupby("mes")["target"].mean()
        mes_maior_venda = sazonalidade.idxmax()
        mes_menor_venda = sazonalidade.idxmin()

        # Panel 1: Metricas Principais
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis("off")

        principais_metrics = [
            "METRICAS PRINCIPAIS",
            "",
            f"Total de Vendas: {self.format_currency(total_vendas)}",
            f"Media Mensal: {self.format_currency(media_mensal)}",
            f"Mediana Mensal: {self.format_currency(mediana_mensal)}",
            f"Desvio Padrao: {self.format_currency(desvio_padrao)}",
            f"Coef. Variacao: {self.format_percentage(cv)}",
            "",
            f"Crescimento Anual Medio: {self.format_percentage(crescimento_medio)}",
            f"Numero de Observacoes: {len(df)} meses",
            f"Periodo: {df.index.min().strftime('%m/%Y')} a {df.index.max().strftime('%m/%Y')}",
        ]

        y_pos = 0.95
        for line in principais_metrics:
            if line.startswith("METRICAS"):
                ax1.text(0.05, y_pos, line, fontsize=14, fontweight="bold", transform=ax1.transAxes)
            elif line == "":
                pass
            else:
                ax1.text(0.05, y_pos, line, fontsize=12, transform=ax1.transAxes)
            y_pos -= 0.08

        # Panel 2: Caracteristicas Sazonais
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.axis("off")

        sazonais_metrics = [
            "CARACTERISTICAS SAZONAIS",
            "",
            f"Mes de Maior Venda: {self.meses_pt[mes_maior_venda - 1]}",
            f"Valor: {self.format_currency(sazonalidade.max())}",
            "",
            f"Mes de Menor Venda: {self.meses_pt[mes_menor_venda - 1]}",
            f"Valor: {self.format_currency(sazonalidade.min())}",
            "",
            f"Amplitude Sazonal: {self.format_currency(sazonalidade.max() - sazonalidade.min())}",
            f"Ratio Max/Min: {sazonalidade.max() / sazonalidade.min():.2f}x",
        ]

        y_pos = 0.95
        for line in sazonais_metrics:
            if line.startswith("CARACTERISTICAS"):
                ax2.text(0.05, y_pos, line, fontsize=14, fontweight="bold", transform=ax2.transAxes)
            elif line == "":
                pass
            else:
                ax2.text(0.05, y_pos, line, fontsize=12, transform=ax2.transAxes)
            y_pos -= 0.08

        # Panel 3: Serie temporal resumida
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(df.index, df["target"], linewidth=2, color="#1f77b4", alpha=0.8)

        # Adicionar media movel de 12 meses
        ma_12 = df["target"].rolling(window=12).mean()
        ax3.plot(df.index, ma_12, linewidth=3, color="#ff7f0e", label="Media Movel 12 Meses")

        configure_axes(
            ax3, title="Serie Temporal com Tendencia", xlabel="Data", ylabel="Vendas (R$)"
        )
        ax3.legend()
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Panel 4: Distribuicao mensal
        ax4 = fig.add_subplot(gs[2, :2])
        monthly_data = [df_mensal[df_mensal["mes"] == i]["target"].values for i in range(1, 13)]
        box_plot = ax4.boxplot(
            monthly_data, labels=[m[:3] for m in self.meses_pt], patch_artist=True
        )

        for patch in box_plot["boxes"]:
            patch.set_facecolor("#5490d3")
            patch.set_alpha(0.8)

        configure_axes(ax4, title="Distribuicao Mensal", xlabel="Mes", ylabel="Vendas (R$)")
        ax4.tick_params(axis="x", rotation=45)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Panel 5: Tendencia anual
        ax5 = fig.add_subplot(gs[2, 2:])
        ax5.plot(
            vendas_anuais.index,
            vendas_anuais.values,
            marker="o",
            linewidth=3,
            markersize=8,
            color="#1f77b4",
            markerfacecolor="#ff7f0e",
        )

        configure_axes(ax5, title="Evolucao Anual", xlabel="Ano", ylabel="Vendas Anuais (R$)")
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: self.format_currency(x)))

        # Panel 6: Insights e Recomendacoes
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis("off")

        # Gerar insights automaticos
        insights = self._gerar_insights(df, vendas_anuais, sazonalidade, crescimento_medio, cv)

        # Dividir insights em duas colunas
        mid_point = len(insights) // 2

        # Coluna esquerda
        y_pos = 0.95
        for line in insights[:mid_point]:
            if line.startswith("INSIGHTS"):
                ax6.text(0.02, y_pos, line, fontsize=14, fontweight="bold", transform=ax6.transAxes)
            elif line.startswith("•"):
                ax6.text(0.02, y_pos, line, fontsize=11, transform=ax6.transAxes)
            else:
                ax6.text(0.02, y_pos, line, fontsize=12, fontweight="bold", transform=ax6.transAxes)
            y_pos -= 0.12

        # Coluna direita
        y_pos = 0.95
        for line in insights[mid_point:]:
            if line.startswith("INSIGHTS"):
                ax6.text(0.52, y_pos, line, fontsize=14, fontweight="bold", transform=ax6.transAxes)
            elif line.startswith("•"):
                ax6.text(0.52, y_pos, line, fontsize=11, transform=ax6.transAxes)
            else:
                ax6.text(0.52, y_pos, line, fontsize=12, fontweight="bold", transform=ax6.transAxes)
            y_pos -= 0.12

        plt.tight_layout()
        save_figure(fig, self.output_dir / "08_resumo_executivo.png")
        logger.info("Plot 8 salvo com sucesso")

    def _gerar_insights(
        self,
        df: pd.DataFrame,
        vendas_anuais: pd.Series,
        sazonalidade: pd.Series,
        crescimento_medio: float,
        cv: float,
    ) -> List[str]:
        """
        Gera insights automaticos sobre os dados.

        Args:
            df: DataFrame com dados
            vendas_anuais: Serie com vendas anuais
            sazonalidade: Serie com sazonalidade mensal
            crescimento_medio: Taxa de crescimento medio
            cv: Coeficiente de variacao

        Returns:
            Lista de insights
        """
        insights = ["INSIGHTS E RECOMENDACOES"]

        # Insight sobre tendencia
        if crescimento_medio > 5:
            insights.extend(
                [
                    "Tendencia:",
                    f"• Crescimento consistente de {self.format_percentage(crescimento_medio)} ao ano",
                    "• Tendencia positiva sustentavel",
                ]
            )
        elif crescimento_medio < -5:
            insights.extend(
                [
                    "Tendencia:",
                    f"• Declinio de {self.format_percentage(abs(crescimento_medio))} ao ano",
                    "• Requer atencao estrategica",
                ]
            )
        else:
            insights.extend(
                [
                    "Tendencia:",
                    "• Crescimento estavel com baixa volatilidade",
                    "• Padrao previsivel adequado para forecasting",
                ]
            )

        # Insight sobre sazonalidade
        amplitude_sazonal = sazonalidade.max() / sazonalidade.min()
        if amplitude_sazonal > 2:
            insights.extend(
                [
                    "Sazonalidade:",
                    f"• Alta sazonalidade (ratio {amplitude_sazonal:.1f}x)",
                    "• Importante para planejamento de estoque",
                ]
            )
        else:
            insights.extend(
                ["Sazonalidade:", "• Sazonalidade moderada", "• Padroes mensais bem definidos"]
            )

        # Insight sobre volatilidade
        if cv > 30:
            insights.extend(
                [
                    "Volatilidade:",
                    f"• Alta variabilidade (CV = {self.format_percentage(cv)})",
                    "• Previsoes com maior incerteza",
                ]
            )
        else:
            insights.extend(
                [
                    "Volatilidade:",
                    f"• Volatilidade controlada (CV = {self.format_percentage(cv)})",
                    "• Dados adequados para modelagem",
                ]
            )

        return insights

    def create_all_plots(self, data_path: Path) -> None:
        """
        Cria todos os plots de EDA.

        Args:
            data_path: Caminho para arquivo de dados
        """
        logger.info("Iniciando criacao de todos os plots de EDA geral")

        try:
            # Carregar dados
            df, ts = self.load_time_series_data(data_path)

            # Criar todos os plots
            self.plot_01_overview_serie_temporal(df, ts)
            self.plot_02_decomposicao_stl(ts)
            self.plot_03_analise_sazonalidade(df)
            self.plot_04_propriedades_estatisticas(df, ts)
            self.plot_05_analise_distribuicao(df)
            self.plot_06_evolucao_temporal(df)
            self.plot_07_analise_correlacao(df)
            self.plot_08_resumo_executivo(df, ts)

            logger.info(f"Todos os plots de EDA salvos em {self.output_dir}")

        except Exception as e:
            logger.error(f"Erro na criacao dos plots: {e}")
            raise


if __name__ == "__main__":
    # Teste da classe
    from pathlib import Path

    output_dir = Path("data/plots/eda")
    data_path = Path("data/processed_data/model_data/time_series/time_series_aggregated.parquet")

    if data_path.exists():
        plotter = GeneralEDAPlotter(output_dir)
        plotter.create_all_plots(data_path)
        print("EDA geral criada com sucesso!")
    else:
        print(f"Arquivo de dados nao encontrado: {data_path}")
