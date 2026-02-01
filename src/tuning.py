import optuna
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


def tune_xgb_optuna(
    X,
    y,
    n_trials: int = 10,
    random_state: int = 42,
    n_splits: int = 10,
    n_jobs: int = -1,
) -> dict:
    """
    Strojenie hiperparametrów XGBoost za pomocą Optuny.
    Optymalizujemy ROC-AUC w 10-krotnej walidacji krzyżowej

    returns: best_params (dict): najlepsze parametry do XGBClassifier
    """

    # Stratyfikowana walidacja krzyżowa
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Dociążenie klasy pozytywnej (ważne przy niezbalansowanych danych)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    def objective(trial: optuna.Trial) -> float:
        # Parametry do strojenia (sensowny, mały zakres)
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=n_jobs,
            # ważne: dociążenie klasy + szybsze uczenie
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    # dopinamy parametry stałe
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": n_jobs,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
    })

    return best_params
