from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import load_data


def test_load_data_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nope.csv")


def test_load_data_reads_csv(tmp_path: Path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(p, index=False)

    df = load_data(p)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]
