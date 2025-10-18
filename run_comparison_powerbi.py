"""
Script de execucao da comparacao entre XGBoost e Power BI

Este script eh o ponto de entrada para realizar a analise comparativa
entre as previsoes do modelo XGBoost e as projecoes do Power BI.

Uso:
    uv run python run_comparison_powerbi.py <caminho_csv_powerbi>

Exemplo:
    uv run python run_comparison_powerbi.py data/powerbi_historico_test_period.csv
"""

import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from config import PROCESSED_DATA_DIR
from evaluation.compare_xgboost_powerbi import compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_markdown_report(results: dict, output_file: str) -> None:
    """
    Gera relatorio em markdown com resultados da comparacao

    Args:
        results: Dicionario com resultados da comparacao
        output_file: Caminho para arquivo de saida
    """
    logger.info(f"Gerando relatorio markdown em {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Comparacao: XGBoost vs Power BI\n\n")

        if "summary" in results:
            summary = results["summary"]

            f.write("## Periodo de Analise\n\n")
            f.write(f"- **Inicio**: {summary['periodo_analise']['inicio']}\n")
            f.write(f"- **Fim**: {summary['periodo_analise']['fim']}\n")
            f.write(f"- **Total de meses**: {summary['periodo_analise']['total_meses']}\n\n")

            f.write("## Metricas de Desempenho\n\n")
            f.write("### XGBoost\n\n")
            xgb_metrics = summary["xgboost_metricas"]
            f.write("| Metrica | Valor |\n")
            f.write("|---------|-------|\n")
            f.write(f"| MAE | R$ {xgb_metrics['mae']:,.2f} |\n")
            f.write(f"| RMSE | R$ {xgb_metrics['rmse']:,.2f} |\n")
            f.write(f"| MAPE | {xgb_metrics['mape']:.2f}% |\n\n")

            f.write("### Power BI\n\n")
            if "powerbi_metricas" in summary:
                pbi_metrics = summary["powerbi_metricas"]
                f.write("| Metrica | Valor |\n")
                f.write("|---------|-------|\n")
                f.write(f"| MAE | R$ {pbi_metrics['mae']:,.2f} |\n")
                f.write(f"| RMSE | R$ {pbi_metrics['rmse']:,.2f} |\n")
                f.write(f"| MAPE | {pbi_metrics['mape']:.2f}% |\n\n")

            f.write("## Conclusao\n\n")
            f.write(f"**Melhor Modelo**: {summary['comparacao']['melhor_modelo']}\n\n")
            f.write(
                f"**Diferenca MAPE**: {summary['comparacao']['diferenca_mape_pct']:.2f} pontos percentuais\n\n"
            )

        f.write("## Visualizacoes Geradas\n\n")
        f.write("As seguintes visualizacoes foram geradas em `data/plots/powerbi_comparison/`:\n\n")
        f.write("1. `xgboost_vs_powerbi_forecast.png` - Comparacao temporal das previsoes\n")
        f.write("2. `xgboost_vs_powerbi_metrics.png` - Comparacao de metricas\n")
        f.write("3. `xgboost_vs_powerbi_difference.png` - Diferencas entre modelos\n")
        f.write("4. `xgboost_vs_powerbi_scatter.png` - Scatter plot comparativo\n")

    logger.info("Relatorio markdown gerado com sucesso")


def main():
    """Funcao principal"""
    logger.info("Iniciando script de comparacao XGBoost vs Power BI")

    if len(sys.argv) < 2:
        logger.error("Uso: uv run python run_comparison_powerbi.py <caminho_csv_powerbi>")
        logger.error(
            "Exemplo: uv run python run_comparison_powerbi.py data/powerbi_historico_test_period.csv"
        )
        sys.exit(1)

    powerbi_csv = sys.argv[1]
    powerbi_path = Path(powerbi_csv)

    if not powerbi_path.exists():
        logger.error(f"Arquivo nao encontrado: {powerbi_csv}")
        logger.info("Certifique-se de que o arquivo foi exportado do Power BI")
        logger.info("Caminhos esperados:")
        logger.info("  - data/powerbi_historico_test_period.csv")
        sys.exit(1)

    try:
        logger.info(f"Carregando dados do Power BI de {powerbi_csv}")

        results = compare_models(powerbi_csv, PROCESSED_DATA_DIR)

        logger.info("Comparacao concluida com sucesso")

        report_file = Path(__file__).parent / "COMPARACAO_XGBOOST_POWERBI.md"
        generate_markdown_report(results, str(report_file))

        results_json = Path(__file__).parent / "data" / "processed_data" / "comparison_results.json"
        results_json.parent.mkdir(parents=True, exist_ok=True)

        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Resultados salvos em {results_json}")

        print("\n" + "=" * 70)
        print("COMPARACAO CONCLUIDA COM SUCESSO!")
        print("=" * 70)

        if "summary" in results:
            summary = results["summary"]
            xgb = summary["xgboost_metricas"]
            pbi = summary.get("powerbi_metricas", {})

            print(
                f"\nPeriodo de Analise: {summary['periodo_analise']['inicio']} a {summary['periodo_analise']['fim']}"
            )
            print(f"Total de Meses: {summary['periodo_analise']['total_meses']}")

            print("\n>>> XGBoost Metricas:")
            print(f"    MAE:  R$ {xgb['mae']:,.2f}")
            print(f"    RMSE: R$ {xgb['rmse']:,.2f}")
            print(f"    MAPE: {xgb['mape']:.2f}%")

            if pbi:
                print("\n>>> Power BI Metricas:")
                print(f"    MAE:  R$ {pbi['mae']:,.2f}")
                print(f"    RMSE: R$ {pbi['rmse']:,.2f}")
                print(f"    MAPE: {pbi['mape']:.2f}%")

                print("\n>>> Conclusao:")
                print(f"    Melhor Modelo: {summary['comparacao']['melhor_modelo']}")
                print(f"    Diferenca MAPE: {summary['comparacao']['diferenca_mape_pct']:.2f} pp")

        print("\n>>> Arquivos Gerados:")
        print(f"    - Relatorio: {report_file}")
        print(f"    - Resultados JSON: {results_json}")
        print("    - Graficos: data/plots/powerbi_comparison/")

        print("\n" + "=" * 70)

    except Exception as e:
        logger.error(f"Erro durante comparacao: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
