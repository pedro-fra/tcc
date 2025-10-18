"""
Script para processar dados diarios do Power BI e agregar para mensal

Este script:
1. Carrega o CSV diario do Power BI
2. Pega o ultimo dia de cada mes (projecao completa)
3. Salva como CSV mensal para comparacao com XGBoost
"""

import pandas as pd
from pathlib import Path

def process_powerbi_daily_to_monthly(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Processa dados diarios do Power BI e agrega para mensal

    Args:
        input_csv: Caminho do arquivo diario do Power BI
        output_csv: Caminho do arquivo mensal de saida

    Returns:
        DataFrame agregado por mes
    """
    print(f"Carregando dados diarios de {input_csv}...")

    df = pd.read_csv(input_csv)
    df['Data'] = pd.to_datetime(df['Data'])

    print(f"Dados carregados: {len(df)} linhas")
    print(f"Periodo: {df['Data'].min().date()} a {df['Data'].max().date()}")

    df['TCC Faturamento Realizado'] = df['TCC Faturamento Realizado'].str.replace('R$ ', '').str.replace('.', '').astype(float)
    df['TCC Faturamento Projetado Power BI'] = df['TCC Faturamento Projetado Power BI'].str.replace('R$ ', '').str.replace('.', '').astype(float)

    df['YearMonth'] = df['Data'].dt.to_period('M')

    agregado = df.groupby('YearMonth').apply(
        lambda x: pd.Series({
            'Data': x['Data'].max(),
            'TCC Faturamento Realizado': x['TCC Faturamento Realizado'].sum(),
            'TCC Faturamento Projetado Power BI': x['TCC Faturamento Projetado Power BI'].max()
        }),
        include_groups=False
    ).reset_index(drop=True)

    agregado = agregado[(agregado['Data'] >= '2022-12-01') & (agregado['Data'] <= '2025-09-30')]

    agregado['Data'] = agregado['Data'].dt.strftime('%Y-%m-%d')

    print(f"\nDados agregados: {len(agregado)} meses")
    print(f"Periodo: {agregado['Data'].min()} a {agregado['Data'].max()}")

    print(f"\nSalvando em {output_csv}...")
    agregado.to_csv(output_csv, index=False)

    print("Concluido!")
    print("\nPrimeiras linhas:")
    print(agregado.head())
    print("\nUltimas linhas:")
    print(agregado.tail())

    return agregado


if __name__ == "__main__":
    input_file = Path(__file__).parent / "data" / "PBI.csv"
    output_file = Path(__file__).parent / "data" / "powerbi_historico_test_period.csv"

    process_powerbi_daily_to_monthly(str(input_file), str(output_file))
