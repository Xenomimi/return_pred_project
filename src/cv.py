import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate


def run_cv(model, X, y, random_state: int = 42, n_splits: int = 10) -> dict:
    """
    10-krotna walidacja krzyżowa (stratyfikowana).
    Zwraca średnie i odchylenia dla kilku metryk.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    for k, v in scores.items():
        if k.startswith("test_"):
            name = k.replace("test_", "")
            summary[name] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
            }

    return summary
