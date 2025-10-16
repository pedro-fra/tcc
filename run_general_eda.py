"""
Script para executar a EDA geral dos dados apos pre-processamento.
Gera analise exploratoria abrangente da serie temporal de vendas.
"""

import logging
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import PROCESSED_DATA_DIR, create_directories
from src.visualization.general_eda_plots import GeneralEDAPlotter

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Executa a EDA geral completa.
    """
    logger.info("Iniciando EDA geral dos dados preprocessados")

    # Criar diretorios necessarios
    create_directories()

    # Definir caminhos
    data_path = PROCESSED_DATA_DIR / "model_data" / "time_series" / "time_series_aggregated.parquet"
    output_dir = Path("data") / "plots" / "eda"

    # Verificar se dados existem
    if not data_path.exists():
        logger.error(f"Arquivo de dados nao encontrado: {data_path}")
        logger.info(
            "Execute o script run_preprocessing.py primeiro para gerar os dados preprocessados"
        )
        return False

    try:
        # Inicializar plotter
        logger.info(f"Criando plots em: {output_dir}")
        plotter = GeneralEDAPlotter(output_dir)

        # Criar todos os plots
        plotter.create_all_plots(data_path)

        # Listar plots criados
        logger.info("=== EDA GERAL COMPLETADA ===")
        logger.info("Plots criados:")

        plot_files = [
            "01_visao_geral_serie_temporal.png",
            "02_analise_decomposicao.png",
            "03_analise_sazonalidade.png",
            "04_propriedades_estatisticas.png",
            "05_analise_distribuicao.png",
            "06_evolucao_temporal.png",
            "07_analise_correlacao.png",
            "08_resumo_executivo.png",
        ]

        for plot_file in plot_files:
            plot_path = output_dir / plot_file
            if plot_path.exists():
                logger.info(f"  ✓ {plot_file}")
            else:
                logger.warning(f"  ✗ {plot_file} - NAO CRIADO")

        logger.info(f"Todos os plots salvos em: {output_dir.absolute()}")
        logger.info("EDA geral concluida com sucesso!")

        return True

    except Exception as e:
        logger.error(f"Erro na execucao da EDA geral: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 60)
        print("EDA GERAL COMPLETADA COM SUCESSO!")
        print("=" * 60)
        print("Os seguintes plots foram criados:")
        print("• 01_visao_geral_serie_temporal.png - Overview completo da serie")
        print("• 02_analise_decomposicao.png - Decomposicao STL")
        print("• 03_analise_sazonalidade.png - Padroes sazonais detalhados")
        print("• 04_propriedades_estatisticas.png - ACF, PACF e testes")
        print("• 05_analise_distribuicao.png - Distribuicao e normalidade")
        print("• 06_evolucao_temporal.png - Tendencias e crescimento")
        print("• 07_analise_correlacao.png - Correlacoes temporais")
        print("• 08_resumo_executivo.png - Dashboard executivo")
        print("\nLocalização: data/plots/eda/")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("ERRO NA EXECUCAO DA EDA GERAL")
        print("=" * 60)
        print("Verifique os logs acima para detalhes do erro.")
        print("Certifique-se de que:")
        print("• Os dados foram preprocessados (execute run_preprocessing.py)")
        print("• Todas as dependencias estao instaladas")
        print("• Nao ha problemas de permissao nos diretorios")
        sys.exit(1)
