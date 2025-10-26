"""
Script de investigacao da escala dos dados - identifica inconsistencia entre MAE e MAPE.

Objetivo: Entender por que XGBoost tem MAE de R$10 milhoes mas MAPE de apenas 26.91%.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_consistency():
    """Analisa consistencia entre MAE e MAPE"""

    print("="*80)
    print("ANALISE DE CONSISTENCIA: MAE vs MAPE")
    print("="*80)

    pbi_csv = Path(__file__).parent / 'data' / 'PBI_Previsoes.csv'
    xgb_json = Path(__file__).parent / 'data' / 'processed_data' / 'xgboost' / 'xgboost_model_summary.json'

    with open(xgb_json, 'r') as f:
        xgb_data = json.load(f)

    print("\n1. DADOS DO XGBOOST (do JSON):")
    print(f"   MAE:  {xgb_data['metrics']['mae']:,.2f}")
    print(f"   RMSE: {xgb_data['metrics']['rmse']:,.2f}")
    print(f"   MAPE: {xgb_data['metrics']['mape']:.2f}%")

    df_pbi = pd.read_csv(pbi_csv)

    def parse_currency(val):
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).replace('R$', '').replace(' ', '').replace('.', '').replace(',', '.')
        return float(val_str)

    df_pbi['Real_Parsed'] = df_pbi['TCC Real Teste'].apply(parse_currency)

    mean_real = df_pbi['Real_Parsed'].mean()
    median_real = df_pbi['Real_Parsed'].median()

    print("\n2. VALORES REAIS (do CSV Power BI - 27 meses):")
    print(f"   Media:    R$ {mean_real:,.2f}")
    print(f"   Mediana:  R$ {median_real:,.2f}")
    print(f"   Min:      R$ {df_pbi['Real_Parsed'].min():,.2f}")
    print(f"   Max:      R$ {df_pbi['Real_Parsed'].max():,.2f}")

    print("\n3. VERIFICACAO DE CONSISTENCIA ENTRE MAE E MAPE:")
    mae_xgb = xgb_data['metrics']['mae']
    mape_xgb = xgb_data['metrics']['mape']

    print(f"\n   Se MAE = R$ {mae_xgb:,.2f}")
    print(f"   E valores reais tem media = R$ {mean_real:,.2f}")
    print(f"   Entao MAPE esperado seria:")
    mape_esperado_media = (mae_xgb / mean_real) * 100
    mape_esperado_mediana = (mae_xgb / median_real) * 100

    print(f"      - Usando media:   {mape_esperado_media:.2f}%")
    print(f"      - Usando mediana: {mape_esperado_mediana:.2f}%")
    print(f"\n   Mas MAPE reportado = {mape_xgb:.2f}%")

    print("\n4. CONCLUSAO:")
    if mape_esperado_media > 100:
        print("   >> INCONSISTENCIA DETECTADA! <<")
        print(f"   MAE de R${mae_xgb:,.2f} sobre valores de ~R${mean_real:,.2f}")
        print(f"   deveria gerar MAPE de {mape_esperado_media:.0f}%, nao {mape_xgb:.2f}%!")
        print("\n   HIPOTESES:")
        print("   A) MAE esta em escala errada (multiplicado por 1000?)")
        print("   B) MAPE esta correto mas MAE foi salvo errado no JSON")
        print("   C) Dados de teste do XGBoost sao diferentes do CSV Power BI")

        mae_correto_opcao1 = (mape_xgb / 100) * mean_real
        print(f"\n   Se MAPE 26.91% esta correto, MAE deveria ser: R$ {mae_correto_opcao1:,.2f}")

        escala = mae_xgb / mae_correto_opcao1
        print(f"   Fator de escala detectado: {escala:.2f}x")

    else:
        print("   Valores parecem consistentes!")

    print("\n5. COMPARACAO COM POWER BI:")
    powerbi_corrected = Path(__file__).parent / 'data' / 'processed_data' / 'powerbi_metrics_corrected.json'
    if powerbi_corrected.exists():
        with open(powerbi_corrected, 'r') as f:
            pbi_metrics = json.load(f)['power_bi_metrics']

        print(f"   Power BI MAE:  R$ {pbi_metrics['mae']:,.2f}")
        print(f"   Power BI MAPE: {pbi_metrics['mape']:.2f}%")

        check_pbi = (pbi_metrics['mae'] / mean_real) * 100
        print(f"\n   Verificacao Power BI: {pbi_metrics['mae']:,.2f} / {mean_real:,.2f} * 100 = {check_pbi:.2f}%")
        print(f"   MAPE reportado: {pbi_metrics['mape']:.2f}%")

        if abs(check_pbi - pbi_metrics['mape']) < 1:
            print("   >> Power BI: CONSISTENTE! <<")
        else:
            print("   >> Power BI: INCONSISTENTE! <<")

    print("="*80)


if __name__ == '__main__':
    analyze_consistency()
