#!/usr/bin/env python3
"""
Gera graficos dos dados do Power BI seguindo o padrao visual do projeto.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/PBI_Previsoes.csv")


def limpar_valor(val):
    if pd.isna(val) or val == "":
        return np.nan
    if isinstance(val, str):
        return float(val.replace("R$", "").replace(" ", "").replace(",", ".").strip())
    return float(val)


df["Real"] = df["TCC Real Teste"].apply(limpar_valor)
df["Hibrido"] = df["TCC Previsao Hibrido"].apply(limpar_valor)
df["Erro_Abs"] = df["TCC Erro Absoluto"].apply(limpar_valor)
df["Erro_Pct_CSV"] = (
    df["TCC Erro Percentual"]
    .str.strip('"%')
    .str.replace(",", ".")
    .apply(lambda x: float(x) if x else np.nan)
)

meses_map = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

df["mes_num"] = df["MonthName"].str.split("/").str[0].map(meses_map)

# Cores do projeto
VISUAL_COLORS = {
    "primary_data": "#1f77b4",
    "predictions": "#d62728",
    "trend": "#ff7f0e",
    "seasonal": "#2ca02c",
    "bars": "#5490d3",
}

plt.style.use("seaborn-v0_8-darkgrid")

# ============================================================================
# GRAFICO 1: Previsoes vs Real (similar ao pattern do projeto)
# ============================================================================

fig, ax = plt.subplots(figsize=(15, 8), dpi=300, facecolor="white")

x_pos = np.arange(len(df))

ax.plot(
    x_pos,
    df["Real"] * 100000,
    marker="o",
    linewidth=2.5,
    markersize=5,
    label="Faturamento Real",
    color=VISUAL_COLORS["primary_data"],
    alpha=1.0,
)
ax.plot(
    x_pos,
    df["Hibrido"] * 100000,
    marker="s",
    linewidth=2.5,
    markersize=5,
    label="Previsao Hibrido (50% MM6 + 50% YoY)",
    linestyle="--",
    color=VISUAL_COLORS["predictions"],
    alpha=1.0,
)

ax.fill_between(
    x_pos, df["Real"] * 100000, df["Hibrido"] * 100000, alpha=0.15, color="gray", label="Diferenca"
)

ax.set_xlabel("Periodo (Jul/23 - Set/25)", fontsize=12, fontweight="bold")
ax.set_ylabel("Faturamento (R$ x 100.000)", fontsize=12, fontweight="bold")
ax.set_title(
    "Analise Power BI: Previsoes Hibrido vs Faturamento Real", fontsize=14, fontweight="bold"
)
ax.set_xticks(x_pos[::3])
ax.set_xticklabels(df["MonthName"].iloc[::3], rotation=45, ha="right")
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/plots/powerbi/01_previsoes_vs_real.png", dpi=300, bbox_inches="tight")
print("Salvo: data/plots/powerbi/01_previsoes_vs_real.png")
plt.close()

# ============================================================================
# GRAFICO 2: Erros Percentuais Absolutos
# ============================================================================

fig, ax = plt.subplots(figsize=(15, 8), dpi=300, facecolor="white")

# Usar valor absoluto dos erros percentuais
df["Erro_Pct_Abs"] = df["Erro_Pct_CSV"].abs()

cores_erros = ["green" if x < 15 else "orange" if x < 30 else "red" for x in df["Erro_Pct_Abs"]]

bars = ax.bar(
    x_pos, df["Erro_Pct_Abs"], color=cores_erros, alpha=0.7, edgecolor="black", linewidth=0.8
)

# MAPE do Power BI (23.45%)
mape_pbi = 23.45
ax.axhline(
    y=mape_pbi,
    color="blue",
    linestyle="--",
    linewidth=2.5,
    label=f"MAPE ({mape_pbi:.2f}%)",
)

ax.set_xlabel("Periodo", fontsize=12, fontweight="bold")
ax.set_ylabel("Erro Percentual Absoluto (%)", fontsize=12, fontweight="bold")
ax.set_title("Analise Power BI: Erros Percentuais Mensais", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos[::3])
ax.set_xticklabels(df["MonthName"].iloc[::3], rotation=45, ha="right")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("data/plots/powerbi/02_erros_percentuais.png", dpi=300, bbox_inches="tight")
print("Salvo: data/plots/powerbi/02_erros_percentuais.png")
plt.close()

# ============================================================================
# GRAFICO 3: Erros Absolutos em Reais
# ============================================================================

fig, ax = plt.subplots(figsize=(15, 8), dpi=300, facecolor="white")

cores_erros = [
    "green" if x < 5000000 else "orange" if x < 10000000 else "red" for x in df["Erro_Abs"] * 100000
]

ax.bar(
    x_pos, df["Erro_Abs"] * 100000, color=cores_erros, alpha=0.7, edgecolor="black", linewidth=0.8
)

mae_media = (df["Erro_Abs"] * 100000).mean()
mae_original = df["Erro_Abs"].mean()
ax.axhline(
    y=mae_media, color="blue", linestyle="--", linewidth=2.5, label=f"MAE (R$ {mae_original:,.2f})"
)

ax.set_xlabel("Periodo", fontsize=12, fontweight="bold")
ax.set_ylabel("Erro Absoluto (R$ x 100.000)", fontsize=12, fontweight="bold")
ax.set_title("Analise Power BI: Erros Absolutos Mensais", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos[::3])
ax.set_xticklabels(df["MonthName"].iloc[::3], rotation=45, ha="right")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("data/plots/powerbi/03_erros_absolutos.png", dpi=300, bbox_inches="tight")
print("Salvo: data/plots/powerbi/03_erros_absolutos.png")
plt.close()

# ============================================================================
# GRAFICO 4: Distribuicao de Erros
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, facecolor="white")

# Histograma de erros percentuais (usando valor absoluto)
axes[0].hist(
    df["Erro_Pct_Abs"],
    bins=10,
    color=VISUAL_COLORS["bars"],
    alpha=0.7,
    edgecolor="black",
    linewidth=1,
)
axes[0].axvline(
    x=df["Erro_Pct_Abs"].mean(), color="red", linestyle="--", linewidth=2, label="Media"
)
axes[0].axvline(
    x=df["Erro_Pct_Abs"].median(), color="green", linestyle="--", linewidth=2, label="Mediana"
)
axes[0].set_xlabel("Erro Percentual Absoluto (%)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Frequencia", fontsize=11, fontweight="bold")
axes[0].set_title("Distribuicao de Erros Percentuais", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis="y")

# Box plot (usando valor absoluto)
data_box = [df["Erro_Pct_Abs"]]
bp = axes[1].boxplot(data_box, tick_labels=["Erro %"], patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor(VISUAL_COLORS["bars"])
    patch.set_alpha(0.7)
axes[1].set_ylabel("Erro Percentual (%)", fontsize=11, fontweight="bold")
axes[1].set_title("Box Plot - Distribuicao de Erros", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3, axis="y")

plt.suptitle("Analise Power BI: Distribuicao de Erros", fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig("data/plots/powerbi/04_distribuicao_erros.png", dpi=300, bbox_inches="tight")
print("Salvo: data/plots/powerbi/04_distribuicao_erros.png")
plt.close()

# ============================================================================
# GRAFICO 5: Metricas Consolidadas
# ============================================================================

fig = plt.figure(figsize=(15, 10), dpi=300, facecolor="white")

# Leitura das metricas
metricas_df = pd.read_csv("data/PBI_Previsoes_Metricas.csv")
mape_val = float(metricas_df["TCC MAPE"].values[0])
mae_val = float(
    metricas_df["TCC MAE"].values[0].replace("R$", "").replace(" ", "").replace(",", ".")
)
rmse_val = float(metricas_df["TCC RMSE"].values[0])
vies_val = float(metricas_df["TCC Vies"].values[0].strip("%").replace(",", ".")) / 100
acuracia_val = float(metricas_df["TCC Acuracia"].values[0].strip("%").replace(",", ".")) / 100

gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# MAPE
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(
    ["MAPE"],
    [mape_val],
    color=VISUAL_COLORS["predictions"],
    alpha=0.7,
    edgecolor="black",
    linewidth=2,
)
ax1.set_ylabel("Percentual (%)", fontsize=11, fontweight="bold")
ax1.set_title("MAPE", fontsize=12, fontweight="bold")
ax1.set_ylim([0, 30])
ax1.text(0, mape_val + 1, f"{mape_val:.2f}%", ha="center", fontweight="bold", fontsize=11)
ax1.grid(True, alpha=0.3, axis="y")

# MAE
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(
    ["MAE"], [mae_val], color=VISUAL_COLORS["seasonal"], alpha=0.7, edgecolor="black", linewidth=2
)
ax2.set_ylabel("Valor (R$)", fontsize=11, fontweight="bold")
ax2.set_title("MAE", fontsize=12, fontweight="bold")
ax2.text(0, mae_val + 3000, f"R$ {mae_val:,.0f}", ha="center", fontweight="bold", fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")

# RMSE
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(
    ["RMSE"], [rmse_val], color=VISUAL_COLORS["trend"], alpha=0.7, edgecolor="black", linewidth=2
)
ax3.set_ylabel("Valor (R$)", fontsize=11, fontweight="bold")
ax3.set_title("RMSE", fontsize=12, fontweight="bold")
ax3.text(0, rmse_val + 5000, f"R$ {rmse_val:,.0f}", ha="center", fontweight="bold", fontsize=10)
ax3.grid(True, alpha=0.3, axis="y")

# VIES
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(["Vies"], [vies_val * 100], color="purple", alpha=0.7, edgecolor="black", linewidth=2)
ax4.set_ylabel("Percentual (%)", fontsize=11, fontweight="bold")
ax4.set_title("Vies (Bias)", fontsize=12, fontweight="bold")
ax4.text(
    0, vies_val * 100 + 0.1, f"{vies_val * 100:.2f}%", ha="center", fontweight="bold", fontsize=11
)
ax4.grid(True, alpha=0.3, axis="y")

# ACURACIA
ax5 = fig.add_subplot(gs[2, 0])
ax5.bar(
    ["Acuracia"],
    [acuracia_val * 100],
    color=VISUAL_COLORS["primary_data"],
    alpha=0.7,
    edgecolor="black",
    linewidth=2,
)
ax5.set_ylabel("Percentual (%)", fontsize=11, fontweight="bold")
ax5.set_title("Acuracia", fontsize=12, fontweight="bold")
ax5.set_ylim([0, 100])
ax5.text(
    0,
    acuracia_val * 100 + 2,
    f"{acuracia_val * 100:.2f}%",
    ha="center",
    fontweight="bold",
    fontsize=11,
)
ax5.grid(True, alpha=0.3, axis="y")

# Resumo de metricas (tabela)
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis("off")
metricas_texto = f"""
METRICAS CONSOLIDADAS - POWER BI (METODO HIBRIDO)

MAPE (Mean Absolute Percentage Error)
  {mape_val:.2f}%

MAE (Mean Absolute Error)
  R$ {mae_val:,.2f}

RMSE (Root Mean Squared Error)
  R$ {rmse_val:,.2f}

Vies (Bias)
  {vies_val * 100:.2f}%

Acuracia
  {acuracia_val * 100:.2f}%

Periodo de Teste: 27 meses (Jul/23 - Set/25)
"""
ax6.text(
    0.1,
    0.9,
    metricas_texto,
    transform=ax6.transAxes,
    fontsize=10,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

plt.suptitle("Analise Power BI: Metricas de Desempenho", fontsize=14, fontweight="bold")
plt.savefig("data/plots/powerbi/05_metricas_consolidadas.png", dpi=300, bbox_inches="tight")
print("Salvo: data/plots/powerbi/05_metricas_consolidadas.png")
plt.close()

# ============================================================================
# GRAFICO 6: Comparacao Temporal (Real vs Previsto)
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=300, facecolor="white")

# Grafico superior: Linhas
ax1.plot(
    x_pos,
    df["Real"] * 100000,
    marker="o",
    linewidth=2.5,
    markersize=5,
    label="Real",
    color=VISUAL_COLORS["primary_data"],
    alpha=1.0,
)
ax1.plot(
    x_pos,
    df["Hibrido"] * 100000,
    marker="s",
    linewidth=2.5,
    markersize=5,
    label="Previsao",
    linestyle="--",
    color=VISUAL_COLORS["predictions"],
    alpha=1.0,
)
ax1.fill_between(x_pos, df["Real"] * 100000, df["Hibrido"] * 100000, alpha=0.15, color="gray")
ax1.set_ylabel("Faturamento (R$ x 100.000)", fontsize=11, fontweight="bold")
ax1.set_title("Faturamento Real vs Previsao Hibrido", fontsize=12, fontweight="bold")
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x_pos[::3])
ax1.set_xticklabels(df["MonthName"].iloc[::3], rotation=45, ha="right")

# Grafico inferior: Erros (usar valor absoluto)
cores_erros_temporal = [
    "green" if x < 15 else "orange" if x < 30 else "red" for x in df["Erro_Pct_Abs"]
]
ax2.bar(
    x_pos,
    df["Erro_Pct_Abs"],
    color=cores_erros_temporal,
    alpha=0.7,
    edgecolor="black",
    linewidth=0.8,
)
ax2.axhline(y=mape_pbi, color="blue", linestyle="--", linewidth=2, label=f"MAPE ({mape_pbi:.2f}%)")
ax2.set_xlabel("Periodo", fontsize=11, fontweight="bold")
ax2.set_ylabel("Erro Percentual (%)", fontsize=11, fontweight="bold")
ax2.set_title("Erros Percentuais Mensais", fontsize=12, fontweight="bold")
ax2.set_xticks(x_pos[::3])
ax2.set_xticklabels(df["MonthName"].iloc[::3], rotation=45, ha="right")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

plt.suptitle("Analise Temporal: Power BI Metodo Hibrido", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/plots/powerbi/06_analise_temporal.png", dpi=300, bbox_inches="tight")
print("Salvo: data/plots/powerbi/06_analise_temporal.png")
plt.close()

print("\nTodos os graficos foram gerados com sucesso em data/plots/powerbi/")
