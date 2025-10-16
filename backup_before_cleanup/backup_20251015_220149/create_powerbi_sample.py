"""
Helper script to create a sample Power BI forecast CSV file.
This demonstrates the expected format for the comparison.
"""

from pathlib import Path

import pandas as pd


def create_sample_powerbi_csv():
    """
    Create a sample Power BI forecast CSV with the expected format.
    """

    dates = pd.date_range(start="2024-01-01", end="2025-12-01", freq="MS")

    sample_data = {
        "Data": dates,
        "Valor_Real": [None] * len(dates),
        "Valor_Projetado": [None] * len(dates),
    }

    for i in range(len(dates)):
        base_value = 5000000 + (i * 100000)
        seasonal = 500000 * (1 + 0.3 * ((i % 12) / 12))
        noise = 200000 * (0.5 - (i % 3) / 6)

        projected = base_value + seasonal + noise
        sample_data["Valor_Projetado"][i] = projected

        if dates[i] < pd.Timestamp.now():
            actual = projected * (0.95 + 0.1 * ((i % 5) / 5))
            sample_data["Valor_Real"][i] = actual

    df = pd.DataFrame(sample_data)

    output_file = Path("data/powerbi_forecast_sample.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False, float_format="%.2f")

    print(f"Sample Power BI forecast created: {output_file}")
    print("\nFormat:")
    print(df.head(10).to_string())
    print("\n...")
    print(df.tail(5).to_string())
    print(f"\nTotal periods: {len(df)}")
    print(f"Periods with actual values: {df['Valor_Real'].notna().sum()}")
    print(f"Periods with only projections: {df['Valor_Real'].isna().sum()}")


if __name__ == "__main__":
    create_sample_powerbi_csv()
