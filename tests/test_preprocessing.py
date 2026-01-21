import pandas as pd

from src.preprocessing import audit_data_quality, preprocessing_pipeline, remove_full_row_duplicates


def test_audit_data_quality_counts_duplicates_and_nan():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", None]})
    rep = audit_data_quality(df)

    assert rep["n_rows"] == 3
    assert rep["duplicate_rows"] == 1
    assert rep["any_nan"] is True
    assert rep["nan_total"] == 1


def test_remove_full_row_duplicates_removes_only_full_rows():
    df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "y"]})
    out = remove_full_row_duplicates(df)

    # tylko jeden wiersz (1,"x") jest duplikatem calosciowym
    assert len(out) == 2


def test_preprocessing_pipeline_report_structure_and_effect():
    df = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})
    out, report = preprocessing_pipeline(df)

    assert "before" in report and "after" in report
    assert report["before"]["duplicate_rows"] == 1
    assert report["after"]["duplicate_rows"] == 0
    assert len(out) == 1