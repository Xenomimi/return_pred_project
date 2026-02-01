from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_roc_curves(models: dict, X_test, y_test, save_path: str) -> None:
    """Rysuje krzywe ROC dla wielu JUŻ wytrenowanych modeli."""
    save_path = str(save_path)
    _ensure_dir(Path(save_path).parent)

    plt.figure()
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, proba, name=name)

    plt.title("ROC curve (hold-out)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curves(models: dict, X_test, y_test, save_path: str) -> None:
    """Rysuje Precision–Recall curve dla wielu JUŻ wytrenowanych modeli."""
    save_path = str(save_path)
    _ensure_dir(Path(save_path).parent)

    plt.figure()
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=name)

    plt.title("Precision–Recall curve (hold-out)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrices(models: dict, X_test, y_test, save_dir: str) -> None:
    """Zapisuje CM dla wielu JUŻ wytrenowanych modeli."""
    save_dir = _ensure_dir(save_dir)

    for name, model in models.items():
        preds = model.predict(X_test)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
        ax.set_title(f"Confusion matrix (hold-out) - {name}")
        fig.tight_layout()
        fig.savefig(save_dir / f"cm_{name}.png", dpi=200)
        plt.close(fig)


def plot_feature_importance(model, feature_names, save_path: str, top_n: int = 20, title: str | None = None) -> None:
    """
    Zapisuje wykres ważności cech
    Działa dla RandomForest / XGBoost.
    """
    if not hasattr(model, "feature_importances_"):
        return

    importances = np.asarray(model.feature_importances_, dtype=float)
    idx = np.argsort(importances)[::-1][:top_n]

    _ensure_dir(Path(save_path).parent)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx][::-1], importances[idx][::-1])
    ax.set_xlabel("feature_importances_")
    ax.set_title(title or "Feature importance (top)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
