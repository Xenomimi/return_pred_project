import numpy as np
import pandas as pd

from src.feature_engineering import _frequency_encoding_map, build_features_transaction_level


def test_frequency_encoding_map_sums_to_1():
    s = pd.Series(["A", "A", "B", "C"])
    m = _frequency_encoding_map(s)

    assert set(m.keys()) == {"A", "B", "C"}
    assert abs(sum(m.values()) - 1.0) < 1e-9
    assert m["A"] > m["B"]


def _toy_df():
    # 2 transakcje, pierwsza ma dodatkowy wiersz zwrotu
    return pd.DataFrame([
        # zakupy tx=1
        {"Transaction ID": 1, "Purchased Item Count": 1, "Final Quantity": 1,
         "Total Revenue": 100.0, "Price Reductions": -10.0, "Sales Tax": 20.0,
         "Refunded Item Count": 0.0, "Refunds": 0.0,
         "Date": "01/01/2019", "Category": "A", "Version": "1", "Item Code": "ABC-001", "Item ID": 111},

        # zwrot tx=1 (nie powinien wchodzic do cech, tylko do targetu)
        {"Transaction ID": 1, "Purchased Item Count": 0, "Final Quantity": -1,
         "Total Revenue": 0.0, "Price Reductions": 0.0, "Sales Tax": -20.0,
         "Refunded Item Count": -1.0, "Refunds": -100.0,
         "Date": "05/01/2019", "Category": "A", "Version": "1", "Item Code": "ABC-001", "Item ID": 111},

        # zakupy tx=2 (bez zwrotu)
        {"Transaction ID": 2, "Purchased Item Count": 2, "Final Quantity": 2,
         "Total Revenue": 200.0, "Price Reductions": 0.0, "Sales Tax": 40.0,
         "Refunded Item Count": 0.0, "Refunds": 0.0,
         "Date": "02/01/2019", "Category": "B", "Version": "v2", "Item Code": "XYZ-999", "Item ID": 222},
    ])


def test_build_features_one_row_per_transaction():
    tx = build_features_transaction_level(_toy_df())
    assert tx.shape[0] == 2
    assert set(tx["Transaction ID"].astype(int).tolist()) == {1, 2}


def test_returned_target_is_correct():
    tx = build_features_transaction_level(_toy_df())
    returned = dict(zip(tx["Transaction ID"].astype(int), tx["Returned"].astype(int)))

    assert returned[1] == 1
    assert returned[2] == 0


def test_leakage_protection_features_from_purchase_rows_only():
    tx = build_features_transaction_level(_toy_df())
    row1 = tx.loc[tx["Transaction ID"] == 1].iloc[0]

    # wiersz zwrotu ma Sales Tax = -20, ale cecha powinna byc liczona tylko z zakupow => 20
    assert float(row1["SalesTax_sum"]) == 20.0


def test_safe_divisions_no_nan_or_inf():
    df = _toy_df()

    # transakcja z revenue=0 i quantity=0 w zakupach, zeby sprawdzic zabezpieczenia
    df = pd.concat([df, pd.DataFrame([{
        "Transaction ID": 3, "Purchased Item Count": 1, "Final Quantity": 0,
        "Total Revenue": 0.0, "Price Reductions": 0.0, "Sales Tax": 0.0,
        "Refunded Item Count": 0.0, "Refunds": 0.0,
        "Date": "03/01/2019", "Category": "C", "Version": "v3", "Item Code": "C-1", "Item ID": 333
    }])], ignore_index=True)

    tx = build_features_transaction_level(df)
    numeric = tx.select_dtypes(include="number")

    assert not np.isinf(numeric.to_numpy()).any()
    assert not numeric.isna().any().any()

    row3 = tx.loc[tx["Transaction ID"] == 3].iloc[0]
    assert float(row3["DiscountRatio"]) == 0.0
    assert float(row3["TaxRatio"]) == 0.0
    assert float(row3["UnitPrice"]) == 0.0


def test_time_features_created():
    tx = build_features_transaction_level(_toy_df())
    for col in ["Year", "Month", "DayOfWeek", "IsWeekend", "Quarter"]:
        assert col in tx.columns
