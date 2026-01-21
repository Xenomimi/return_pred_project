import pandas as pd
from pathlib import Path


def load_data(path: Path) -> pd.DataFrame:
    """
    Ładuje dane z pliku do DataFrame na podstawie przekazanej ścieżki.
    W przypadku nieprawidłowej ścieżku rzuca wyjątkiem
    :param path: Ścieżka do danych
    :return: Dane w postaci DataFrame
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df
