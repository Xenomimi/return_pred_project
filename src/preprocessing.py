import pandas as pd


def audit_data_quality(df: pd.DataFrame) -> dict:
    """
    Prosty audyt jakości danych.
    Zwraca słownik z informacjami o duplikatach i brakach (NaN).
    """
    report = {}

    report["n_rows"] = int(len(df))
    report["n_cols"] = int(df.shape[1])

    # Duplikaty całych wierszy
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Braki danych
    na_count = df.isna().sum()
    report["any_nan"] = bool(na_count.any())
    report["nan_total"] = int(na_count.sum())

    # Top 10 kolumn z największą liczbą NaN
    report["nan_by_col_top10"] = (
        na_count[na_count > 0]
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    return report


def remove_full_row_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usuwa duplikaty całych wierszy.
    """
    df = df.copy()
    df = df.drop_duplicates()
    return df


def preprocessing_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline do wstępnego przetwarzania:
    - audyt jakości
    - usunięcie duplikatów całych wierszy
    - ponowny audyt po czyszczeniu
    """
    before = audit_data_quality(df)

    df_clean = remove_full_row_duplicates(df)

    after = audit_data_quality(df_clean)

    return df_clean, {"before": before, "after": after}
