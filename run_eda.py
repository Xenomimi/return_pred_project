from pathlib import Path

from src.data_loader import load_data
from src.preprocessing import preprocessing_pipeline
from src.feature_engineering import build_features_transaction_level
from src.config import DATA_PATH

from src.eda import (
    basic_info, descriptive_stats, target_distribution,
    histograms, boxplots_by_target, scatter_by_target,
    correlation_with_target, correlation_heatmap
)

# Katalog na wyniki EDA
Path("outputs/eda").mkdir(parents=True, exist_ok=True)

# Wczytanie + preprocessing
df = load_data(DATA_PATH)
df, report = preprocessing_pipeline(df)
print("AUDYT:", report)

# Feature engineering
tx = build_features_transaction_level(df)

# Podstawowe info
basic_info(tx)

# Statystyki opisowe
stats = descriptive_stats(tx)
stats.to_csv("outputs/eda/descriptive_stats.csv")
print("\nStatystyki opisowe (pierwsze 15 wierszy):")
print(stats.head(15))
print("\nZapisano: outputs/eda/descriptive_stats.csv")

# Rozkład targetu
target_distribution(tx, target_col="Returned", save_path="outputs/eda/target_distribution.png")

# Histogramy
histograms(
    tx,
    cols=["TotalRevenue_sum", "DiscountRatio", "UnitPrice", "UniqueItems_n", "ItemsPurchased_sum"],
    save_dir="outputs/eda"
)

# Boxploty vs target
boxplots_by_target(
    tx,
    cols=["TotalRevenue_sum", "DiscountRatio", "UnitPrice", "UniqueItems_n", "ItemsPurchased_sum"],
    target_col="Returned",
    save_dir="outputs/eda"
)

# Scatter: rabat vs wartość koszyka
scatter_by_target(
    tx,
    x="TotalRevenue_sum",
    y="DiscountRatio",
    target_col="Returned",
    save_path="outputs/eda/scatter_revenue_discount.png"
)

# Korelacje (bez Transaction ID)
tx_corr = tx.drop(columns=["Transaction ID"], errors="ignore")

corr_to_target = correlation_with_target(tx_corr, target_col="Returned", top_n=15)
corr_to_target.to_csv("outputs/eda/corr_to_target.csv")
print("\nZapisano: outputs/eda/corr_to_target.csv")

correlation_heatmap(tx_corr, save_path="outputs/eda/corr_heatmap.png")

print("\nEDA zakończone. Wykresy i CSV są w outputs/eda/")
