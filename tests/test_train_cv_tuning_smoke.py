import pandas as pd

from src.cv import run_cv
from src.models import logreg_model
from src.train import train_and_evaluate
from src.tuning import tune_xgb_optuna


def test_train_and_evaluate_runs():
    X = pd.DataFrame({"x1": [0, 1, 0, 1], "x2": [1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1])

    res = train_and_evaluate(logreg_model(), X, X, y, y)
    assert "roc_auc" in res
    assert "cm" in res
    assert isinstance(res["cm"], list)


def test_run_cv_returns_expected_keys():
    X = pd.DataFrame({"x1": [0, 1, 0, 1, 0, 1], "x2": [1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])

    res = run_cv(logreg_model(), X, y, n_splits=3)
    for k in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        assert k in res
        assert "mean" in res[k] and "std" in res[k]


def test_optuna_tuning_returns_required_params():
    X = pd.DataFrame({"x1": [0, 1, 0, 1, 0, 1, 0, 1], "x2": [1, 0, 1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    best = tune_xgb_optuna(X, y, n_trials=2, n_splits=2, random_state=42)
    # minimalna walidacja zwrotu
    assert "n_estimators" in best
    assert best["objective"] == "binary:logistic"
    assert best["eval_metric"] == "auc"
