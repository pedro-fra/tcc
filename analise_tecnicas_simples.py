import pandas as pd
import numpy as np

# Simular dados do seu faturamento
df = pd.read_csv('data/PBI_Previsoes.csv')

def clean_value(val):
    if isinstance(val, str):
        return float(val.replace('R$ ', '').replace(',', '.'))
    return val

df['Real'] = df['TCC Real Teste'].apply(clean_value)
df['Data'] = pd.to_datetime(df['MonthName'], format='%b/%y')

print("="*90)
print("TECNICAS SIMPLES DE PREVISAO PARA POWER BI")
print("="*90)

# 1. MÉDIA MÓVEL SIMPLES (SMA)
print("\n1. MEDIA MOVEL SIMPLES (SMA) - Most Common")
print("-" * 90)

sma_3 = df['Real'].rolling(window=3, min_periods=1).mean()
sma_6 = df['Real'].rolling(window=6, min_periods=1).mean()
sma_12 = df['Real'].rolling(window=12, min_periods=1).mean()

# Comparar com previsão SMA-3
mae_sma3 = np.abs(sma_3.shift(1) - df['Real']).mean()
mape_sma3 = (np.abs(sma_3.shift(1) - df['Real']) / df['Real']).mean() * 100

print(f"SMA-3 (ultimos 3 meses):")
print(f"  MAE:  R$ {mae_sma3:,.2f}")
print(f"  MAPE: {mape_sma3:.2f}%")
print(f"  Uso: Muito responsivo, bom para dados volateis")
print(f"  Codigo DAX: AVERAGE(LASTPERIODS(3, [Faturamento]))")

mae_sma6 = np.abs(sma_6.shift(1) - df['Real']).mean()
mape_sma6 = (np.abs(sma_6.shift(1) - df['Real']) / df['Real']).mean() * 100

print(f"\nSMA-6 (ultimos 6 meses):")
print(f"  MAE:  R$ {mae_sma6:,.2f}")
print(f"  MAPE: {mape_sma6:.2f}%")
print(f"  Uso: Bom equilibrio entre reatividade e suavizacao")
print(f"  Codigo DAX: AVERAGE(LASTPERIODS(6, [Faturamento]))")

mae_sma12 = np.abs(sma_12.shift(1) - df['Real']).mean()
mape_sma12 = (np.abs(sma_12.shift(1) - df['Real']) / df['Real']).mean() * 100

print(f"\nSMA-12 (ultimos 12 meses):")
print(f"  MAE:  R$ {mae_sma12:,.2f}")
print(f"  MAPE: {mape_sma12:.2f}%")
print(f"  Uso: Suavizado, ignora variacoes curtas, capta sazonalidade anual")

# 2. CRESCIMENTO LINEAR (Trend)
print("\n\n2. CRESCIMENTO LINEAR (Linear Trend)")
print("-" * 90)

x = np.arange(len(df))
y = df['Real'].values
coeffs = np.polyfit(x, y, 1)
linear_trend = np.polyval(coeffs, x)
linear_prev = np.polyval(coeffs, x - 1)

mae_trend = np.abs(linear_prev - y).mean()
mape_trend = (np.abs(linear_prev - y) / y).mean() * 100

print(f"Regressao Linear Simples:")
print(f"  Slope (inclinacao): R$ {coeffs[0]:,.2f} por mes")
print(f"  MAE:  R$ {mae_trend:,.2f}")
print(f"  MAPE: {mape_trend:.2f}%")
print(f"  Uso: Bom para dados com tendencia clara")
print(f"  Problema: Seus dados NAO tem tendencia linear!")

# 3. CRESCIMENTO PERCENTUAL FIXO (% crescimento)
print("\n\n3. CRESCIMENTO PERCENTUAL FIXO (%)")
print("-" * 90)

pct_change = df['Real'].pct_change()
avg_pct_change = pct_change.mean()
prev_pct = df['Real'].iloc[:-1].values * (1 + avg_pct_change)
mae_pct = np.abs(prev_pct - df['Real'].iloc[1:].values).mean()
mape_pct = (np.abs(prev_pct - df['Real'].iloc[1:].values) / df['Real'].iloc[1:].values).mean() * 100

print(f"Crescimento Medio: {avg_pct_change*100:.2f}% ao mes")
print(f"MAE:  R$ {mae_pct:,.2f}")
print(f"MAPE: {mape_pct:.2f}%")
print(f"Uso: Rapido e simples, assume taxa constante")
print(f"Problema: Seus dados flutuam muito!")

# 4. YEAR-OVER-YEAR (Comparação com mesmo mês ano anterior)
print("\n\n4. YEAR-OVER-YEAR (YoY) - Already Tested")
print("-" * 90)

yoy = df['Real'].shift(12)
mae_yoy = np.abs(yoy - df['Real']).dropna().mean()
mape_yoy = (np.abs(yoy - df['Real']) / df['Real']).dropna().mean() * 100

print(f"Comparacao com mes 12 meses atras:")
print(f"  MAE:  R$ {mae_yoy:,.2f}")
print(f"  MAPE: {mape_yoy:.2f}%")
print(f"  Uso: Excelente para dados com sazonalidade anual")
print(f"  Seu resultado: Muito bom! (20,19%)")

# 5. NAIVE SAZONAL
print("\n\n5. NAIVE SAZONAL (Seasonal Naive)")
print("-" * 90)

seasonal_naive = df['Real'].shift(1)
mae_sn = np.abs(seasonal_naive - df['Real']).dropna().mean()
mape_sn = (np.abs(seasonal_naive - df['Real']) / df['Real']).dropna().mean() * 100

print(f"Previsao = valor do mes anterior")
print(f"  MAE:  R$ {mae_sn:,.2f}")
print(f"  MAPE: {mape_sn:.2f}%")
print(f"  Uso: Baseline simples, mas muito reativo")

# 6. MÉDIA DE TODOS OS DADOS
print("\n\n6. MEDIA HISTORICA (Historical Average)")
print("-" * 90)

media_historica = df['Real'].mean()
mae_media = np.abs(media_historica - df['Real']).mean()
mape_media = (np.abs(media_historica - df['Real']) / df['Real']).mean() * 100

print(f"Previsao constante = Media de todos os dados")
print(f"  Valor constante: R$ {media_historica:,.2f}")
print(f"  MAE:  R$ {mae_media:,.2f}")
print(f"  MAPE: {mape_media:.2f}%")
print(f"  Uso: Baseline muito basico, para comparacao apenas")

print("\n\n" + "="*90)
print("RESUMO COMPARATIVO - QUAL USAR?")
print("="*90)

comparacao = pd.DataFrame({
    'Tecnica': ['SMA-3', 'SMA-6', 'SMA-12', 'YoY', 'Naive', 'Cresc %', 'Linear', 'Media', 'XGBoost'],
    'MAE': [f'{mae_sma3:,.0f}', f'{mae_sma6:,.0f}', f'{mae_sma12:,.0f}',
            f'{mae_yoy:,.0f}', f'{mae_sn:,.0f}', f'{mae_pct:,.0f}',
            f'{mae_trend:,.0f}', f'{mae_media:,.0f}', '10.1M'],
    'MAPE': [f'{mape_sma3:.1f}%', f'{mape_sma6:.1f}%', f'{mape_sma12:.1f}%',
             f'{mape_yoy:.1f}%', f'{mape_sn:.1f}%', f'{mape_pct:.1f}%',
             f'{mape_trend:.1f}%', f'{mape_media:.1f}%', '26.9%'],
    'Simplicidade': ['Muito', 'Muito', 'Simples', 'Simples',
                     'Muito', 'Simples', 'Media', 'Muito', 'Complexo']
})

print(comparacao.to_string(index=False))

print("\n\n" + "="*90)
print("RECOMENDACAO PARA SEU TCC")
print("="*90)

print("""
TOP 3 OPCOES:

1. MELHOR PERFORORMANCE: YoY (Year-over-Year) *** RECOMENDADO ***
   MAE: R$ 163k | MAPE: 20.1%
   Razao:
     - Ja testado e funcionando bem
     - Captura padrao sazonal anual (seus dados tem isso)
     - Muito simples no DAX
     - Codigo: CALCULATE([Faturamento], DATEADD(Data, -12, MONTH))

2. BASELINE SIMPLES: SMA-12 (Media Movel 12 meses)
   MAE: R$ 195k | MAPE: 30.7%
   Razao:
     - Facil de implementar
     - Bom equilibrio
     - Menos dependente de ciclo exato
     - Mostra limite do metodo simples

3. COMPARACAO BASELINE: SMA-6 (Media Movel 6 meses)
   MAE: R$ 184k | MAPE: 35.3%
   Razao:
     - Mais reativo a mudancas
     - Nao assume ciclo anual
     - Mostra problema de ser muito sensivel

OPCAO NUKE SIMPLES: Naive (mes anterior)
   MAE: R$ 210k | MAPE: 30.4%
   - Super trivial, mas nao muito bom

CONCLUSAO DO TCC:
  Usar 2 metodos simples:

  1. Baseline: SMA-6 (MAPE 35,3%)
  2. Otimizado: YoY (MAPE 20,1%)
  3. ML Avancado: XGBoost (MAPE 26,9%)

  Resultado: YoY eh competitivo com XGBoost!
  Mostra que uma tecnica bem pensada pode ser tao boa quanto ML complexo.
""")

print("\n" + "="*90)
