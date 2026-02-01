import pandas as pd
import numpy as np

# Kolumny, które zdradzają zwrot / powstają po zwrocie.
# Używamy ich do stworzenia targetu, ale nie wchodzic do X.
LEAKAGE_COLS = [
    "Refunds",
    "Refunded Item Count",
    "Final Revenue",
    "Overall Revenue",
    "Final Quantity",
    "Sales Tax",
]


def _frequency_encoding_map(series: pd.Series) -> dict:
    """Zwraca mapę: wartość -> częstość (0..1)."""
    freq = series.value_counts(normalize=True)
    return freq.to_dict()


def build_features_transaction_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Buduje dane na poziomie transakcji (Transaction ID).

    Logika:
    - Target Returned (0/1):
        Returned = 1, jeśli w transakcji wystąpił jakikolwiek zwrot
        (Refunded Item Count < 0 lub Refunds < 0)
    - Cechy liczymy TYLKO z wierszy zakupowych (Purchased Item Count > 0),
      żeby model nie widział sygnałów zwrotu w feature'ach (data leakage).
    - Kategoryczne kodujemy prostym Frequency Encoding (Category, Version),
      bo Version może być czasem pojedynczą liczbą/tekstem.

    Zwraca:
    - DataFrame: 1 wiersz = 1 transakcja, z kolumną targetu "Returned".
    """

    df = df.copy()

    # Usuwamy duplikaty całych wierszy
    df = df.drop_duplicates()

    # Target Returned na poziomie transakcji
    # Wiersz zwrotu rozpoznajemy po ujemnych wartościach
    is_refund_row = (df["Refunded Item Count"] < 0) | (df["Refunds"] < 0)
    returned_by_tx = (
        df.assign(_refund=is_refund_row)
        .groupby("Transaction ID")["_refund"]
        .any()
        .astype(int)
    )

    # Bierzemy tylko wiersze zakupowe do liczenia cech
    purchases = df[df["Purchased Item Count"] > 0].copy()

    # Parsowanie daty
    dt = pd.to_datetime(purchases["Date"], errors="coerce", dayfirst=True)
    purchases["_date"] = dt

    # Frequency encoding dla Category i Version
    cat_freq_map = _frequency_encoding_map(purchases["Category"])
    ver_freq_map = _frequency_encoding_map(purchases["Version"])

    purchases["Category_freq"] = purchases["Category"].map(cat_freq_map).fillna(0.0)
    purchases["Version_freq"] = purchases["Version"].map(ver_freq_map).fillna(0.0)

    # prefix z Item Code i jego freq encoding
    if "Item Code" in purchases.columns:
        purchases["ItemCodePrefix"] = purchases["Item Code"].astype(str).str.split("-").str[0]
        prefix_map = _frequency_encoding_map(purchases["ItemCodePrefix"])
        purchases["ItemCodePrefix_freq"] = purchases["ItemCodePrefix"].map(prefix_map).fillna(0.0)

    # Agregacje po Transaction ID
    g = purchases.groupby("Transaction ID")

    tx = pd.DataFrame({
        "Transaction ID": g.size().index,

        # daty (bierzemy najwcześniejszą datę zakupu w transakcji)
        "PurchaseDate": g["_date"].min(),

        # wolumen
        "ItemsPurchased_sum": g["Purchased Item Count"].sum(),
        "FinalQuantity_sum": g["Final Quantity"].sum(),
        "UniqueItems_n": g["Item ID"].nunique(),
        "UniqueCategories_n": g["Category"].nunique(),

        # kasa
        "TotalRevenue_sum": g["Total Revenue"].sum(),
        "PriceReductions_sum": g["Price Reductions"].sum(),
        "SalesTax_sum": g["Sales Tax"].sum(),

        # agregacje z FE
        "Category_freq_mean": g["Category_freq"].mean(),
        "Version_freq_mean": g["Version_freq"].mean(),
    })

    if "ItemCodePrefix_freq" in purchases.columns:
        tx["ItemCodePrefix_freq_mean"] = g["ItemCodePrefix_freq"].mean()

    # Cechy pochodne
    # Uwaga: dzielenie przez 0 zabezpieczamy
    tx["DiscountRatio"] = tx["PriceReductions_sum"] / tx["TotalRevenue_sum"].replace(0, np.nan)
    tx["DiscountRatio"] = tx["DiscountRatio"].fillna(0.0)

    tx["TaxRatio"] = tx["SalesTax_sum"] / tx["TotalRevenue_sum"].replace(0, np.nan)
    tx["TaxRatio"] = tx["TaxRatio"].fillna(0.0)

    tx["UnitPrice"] = tx["TotalRevenue_sum"] / tx["FinalQuantity_sum"].replace(0, np.nan)
    tx["UnitPrice"] = tx["UnitPrice"].fillna(0.0)

    # Rozbijamy datę na proste cechy
    tx["Year"] = tx["PurchaseDate"].dt.year.fillna(0).astype(int)
    tx["Month"] = tx["PurchaseDate"].dt.month.fillna(0).astype(int)
    tx["DayOfWeek"] = tx["PurchaseDate"].dt.weekday.fillna(0).astype(int)
    tx["IsWeekend"] = tx["DayOfWeek"].isin([5, 6]).astype(int)
    tx["Quarter"] = tx["PurchaseDate"].dt.quarter.fillna(0).astype(int)

    # surową datę można wywalić
    tx = tx.drop(columns=["PurchaseDate"])

    # Dołączamy target
    tx["Returned"] = tx["Transaction ID"].map(returned_by_tx).fillna(0).astype(int)

    tx = tx.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return tx
