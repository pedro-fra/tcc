"""
Arquivo de configuracao centralizada para formatacao de graficos.
Define todas as cores, estilos e configuracoes visuais para garantir consistencia.
"""

import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURACOES GERAIS DE ESTILO
# ============================================================================

# Estilo base do matplotlib
PLOT_STYLE = "seaborn-v0_8"

# Configuracoes de figura
FIGURE_CONFIG = {"figsize": (15, 10), "dpi": 300, "facecolor": "white", "edgecolor": "none"}

# ============================================================================
# CONFIGURACOES DE CORES
# ============================================================================

# Esquema de cores otimizado para qualidade visual
VISUAL_COLORS = {
    "primary_data": "#1f77b4",  # Azul profissional para dados principais
    "historical_data": "#1f77b4",  # Mesmo azul para dados históricos (com alpha reduzido)
    "predictions": "#d62728",  # Vermelho para previsões
    "trend": "#ff7f0e",  # Laranja para tendências
    "seasonal": "#2ca02c",  # Verde para componentes sazonais
    "bars": "#5490d3",  # Azul mais claro para barras
    "secondary": "#9467bd",  # Roxo para elementos auxiliares
    # Elementos de apoio
    "grid": "gray",  # Grade em cinza
    "split_line": "gray",  # Linha de divisão treino/teste
    "background": None,  # Background padrão do seaborn
    "bar_edge": "#1f77b4",  # Bordas sutis para barras
}

# ============================================================================
# CONFIGURACOES DE LINHA
# ============================================================================

LINE_CONFIG = {
    "primary_linewidth": 2,  # Largura para dados principais
    "secondary_linewidth": 1.5,  # Largura para dados históricos
    "linestyle": "-",  # Linhas sólidas
    "primary_alpha": 1.0,  # Alpha para dados importantes
    "secondary_alpha": 0.7,  # Alpha para dados de contexto
    "marker": "o",  # Marcador padrão
    "markersize": 4,  # Tamanho dos marcadores
}

# Configurações específicas por tipo de linha com hierarquia visual
LINE_STYLES = {
    "primary_data": {
        "color": VISUAL_COLORS["primary_data"],
        "linewidth": LINE_CONFIG["primary_linewidth"],
        "linestyle": LINE_CONFIG["linestyle"],
        "alpha": LINE_CONFIG["primary_alpha"],
    },
    "historical_data": {
        "color": VISUAL_COLORS["historical_data"],
        "linewidth": LINE_CONFIG["secondary_linewidth"],
        "linestyle": LINE_CONFIG["linestyle"],
        "alpha": LINE_CONFIG["secondary_alpha"],
    },
    "predictions": {
        "color": VISUAL_COLORS["predictions"],
        "linewidth": LINE_CONFIG["primary_linewidth"],
        "linestyle": LINE_CONFIG["linestyle"],
        "alpha": LINE_CONFIG["primary_alpha"],
    },
    "trend": {
        "color": VISUAL_COLORS["trend"],
        "linewidth": LINE_CONFIG["secondary_linewidth"],
        "linestyle": LINE_CONFIG["linestyle"],
        "alpha": LINE_CONFIG["secondary_alpha"],
    },
    "seasonal": {
        "color": VISUAL_COLORS["seasonal"],
        "linewidth": LINE_CONFIG["secondary_linewidth"],
        "linestyle": LINE_CONFIG["linestyle"],
        "alpha": LINE_CONFIG["secondary_alpha"],
    },
    "split": {
        "color": VISUAL_COLORS["split_line"],
        "linewidth": 1,
        "linestyle": ":",
        "alpha": 0.7,
    },
}

# ============================================================================
# CONFIGURACOES DE BARRAS E COLUNAS
# ============================================================================

BAR_CONFIG = {
    "color": VISUAL_COLORS["bars"],
    "edgecolor": VISUAL_COLORS["bar_edge"],
    "linewidth": 0.8,
    "alpha": 0.8,  # Leve transparência para barras
    "width": 0.8,
}

# ============================================================================
# CONFIGURACOES DE TEXTO E LABELS
# ============================================================================

TEXT_CONFIG = {
    "title_fontsize": 16,
    "title_fontweight": "bold",
    "label_fontsize": 12,
    "label_fontweight": "bold",  # Todas as labels em negrito
    "legend_fontsize": 10,
    "tick_labelsize": 10,
}

# ============================================================================
# CONFIGURACOES DE GRID E LAYOUT
# ============================================================================

GRID_CONFIG = {
    "visible": True,
    "alpha": 0.3,
    "color": VISUAL_COLORS["grid"],
    "linestyle": "-",
    "linewidth": 0.5,
}

LAYOUT_CONFIG = {
    "bbox_inches": "tight",
    "pad_inches": 0.1,
}

# ============================================================================
# FUNCOES UTILITARIAS
# ============================================================================


def apply_plot_style():
    """Aplica o estilo base para todos os graficos."""
    plt.style.use(PLOT_STYLE)


def configure_axes(ax, title=None, xlabel=None, ylabel=None):
    """
    Aplica configuracoes padrao aos eixos.

    Args:
        ax: Objeto axes do matplotlib
        title: Titulo do grafico (opcional)
        xlabel: Label do eixo X (opcional)
        ylabel: Label do eixo Y (opcional)
    """
    # Configurar grid
    ax.grid(**GRID_CONFIG)

    # Configurar labels
    if title:
        ax.set_title(
            title,
            fontsize=TEXT_CONFIG["title_fontsize"],
            fontweight=TEXT_CONFIG["title_fontweight"],
        )

    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=TEXT_CONFIG["label_fontsize"],
            fontweight=TEXT_CONFIG["label_fontweight"],
        )

    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=TEXT_CONFIG["label_fontsize"],
            fontweight=TEXT_CONFIG["label_fontweight"],
        )

    # Configurar ticks
    ax.tick_params(labelsize=TEXT_CONFIG["tick_labelsize"])

    # A legenda sera configurada automaticamente pelo matplotlib


def get_line_style(line_type):
    """
    Retorna configuracoes de linha para um tipo especifico.

    Args:
        line_type: Tipo da linha ('primary_data', 'historical_data', 'predictions', 'trend', 'seasonal', 'split')

    Returns:
        dict: Configuracoes de estilo da linha
    """
    return LINE_STYLES.get(line_type, LINE_STYLES["primary_data"])


def save_figure(fig, filename, **kwargs):
    """
    Salva figura com configuracoes padrao.

    Args:
        fig: Figura do matplotlib
        filename: Nome do arquivo para salvar
        **kwargs: Argumentos adicionais para savefig
    """
    save_config = LAYOUT_CONFIG.copy()
    save_config.update(kwargs)

    fig.savefig(filename, dpi=FIGURE_CONFIG["dpi"], **save_config)
    plt.close(fig)


# ============================================================================
# CONFIGURACOES ESPECIFICAS POR TIPO DE GRAFICO
# ============================================================================

# Configurações para gráficos de comparação de previsão
PREDICTION_COMPARISON_CONFIG = {
    "training_data": {
        "color": VISUAL_COLORS["historical_data"],
        "alpha": LINE_CONFIG["secondary_alpha"],  # Dados históricos com menor destaque
        "linewidth": LINE_CONFIG["secondary_linewidth"],
        "linestyle": LINE_CONFIG["linestyle"],
    },
    "actual_test": {
        "color": VISUAL_COLORS["primary_data"],
        "alpha": LINE_CONFIG["primary_alpha"],
        "linewidth": LINE_CONFIG["primary_linewidth"],
        "linestyle": LINE_CONFIG["linestyle"],
    },
    "predicted": {
        "color": VISUAL_COLORS["predictions"],
        "alpha": LINE_CONFIG["primary_alpha"],
        "linewidth": LINE_CONFIG["primary_linewidth"],
        "linestyle": LINE_CONFIG["linestyle"],
    },
    "detailed_markers": True,
    "marker_size": 4,
}

# Configurações para histogramas e distribuições
HISTOGRAM_CONFIG = {
    "bins": 30,
    "color": VISUAL_COLORS["bars"],
    "edgecolor": VISUAL_COLORS["bar_edge"],
    "alpha": 0.8,
    "density": True,
}

# Configurações para box plots
BOXPLOT_CONFIG = {
    "boxprops": {
        "facecolor": VISUAL_COLORS["bars"],
        "edgecolor": VISUAL_COLORS["bar_edge"],
        "alpha": 0.8,
    },
    "medianprops": {"color": VISUAL_COLORS["predictions"], "linewidth": 2},
    "whiskerprops": {"color": VISUAL_COLORS["bar_edge"]},
    "capprops": {"color": VISUAL_COLORS["bar_edge"]},
    "flierprops": {
        "marker": "o",
        "markerfacecolor": VISUAL_COLORS["predictions"],
        "markeredgecolor": VISUAL_COLORS["bar_edge"],
        "markersize": 4,
    },
}
