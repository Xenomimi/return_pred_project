from sklearn.model_selection import train_test_split
from pathlib import Path
import json

from src.config import DATA_PATH
from src.data_loader import load_data
from src.preprocessing import preprocessing_pipeline
from src.feature_engineering import build_features_transaction_level

from src.models import baseline_model, xgb_model, logreg_model
from src.train import train_and_evaluate, undersample_train
from src.cv import run_cv
from src.tuning import tune_xgb_optuna

from src.model_viz import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_feature_importance,
)


def print_comparison_table(title: str, before: dict, after: dict) -> None:
    """Prosta tabelka porównawcza metryk przed/po."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    is_cv = isinstance(before.get("roc_auc"), dict)

    if is_cv:
        rows = ["roc_auc", "f1", "precision", "recall", "accuracy"]
        print(f"{'Metric':<12} | {'Before (mean±std)':<22} | {'After (mean±std)':<22}")
        print("-" * 70)
        for m in rows:
            if m not in before or m not in after:
                continue
            b = before[m]
            a = after[m]
            print(f"{m:<12} | {b['mean']:.4f} ± {b['std']:.4f}        | {a['mean']:.4f} ± {a['std']:.4f}")
    else:
        rows = ["roc_auc", "acc", "f1", "precision", "recall", "cm"]
        print(f"{'Metric':<10} | {'Before':<18} | {'After':<18}")
        print("-" * 55)
        for m in rows:
            if m in before and m in after:
                b = before[m]
                a = after[m]
                if isinstance(b, (float, int)) and isinstance(a, (float, int)):
                    print(f"{m:<10} | {b:<18.4f} | {a:<18.4f}")
                else:
                    print(f"{m:<10} | {str(b):<18} | {str(a):<18}")


def _extract_model(res: dict, fallback_model):
    """
    Jeżeli train_and_evaluate zwraca 'model' -> użyj.
    Jeśli nie -> fallback_model (już dopasowany gdzie indziej) albo dopasuj ręcznie.
    """
    if isinstance(res, dict) and "model" in res and res["model"] is not None:
        return res["model"]
    return fallback_model


def main():
    # 1) Dane
    df = load_data(DATA_PATH)
    df, report = preprocessing_pipeline(df)
    print("AUDYT:", report)

    tx = build_features_transaction_level(df)
    X = tx.drop(columns=["Returned", "Transaction ID"])
    y = tx["Returned"].astype(int)

    # 2) Hold-out split (test zawsze w naturalnym rozkładzie)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train mean (pos rate):", float(y_train.mean()))
    print("y_test mean (pos rate):", float(y_test.mean()))

    # 3) Zbalansowany train (undersampling 1:1) – tylko do uczenia
    X_train_bal, y_train_bal = undersample_train(X_train, y_train, random_state=42)
    print("X_train_bal shape:", X_train_bal.shape)
    print("y_train_bal mean (pos rate):", float(y_train_bal.mean()))

    # 4) 10-fold CV (na pełnych danych – realistycznie i stabilnie)
    print("\n=== 10-fold CV (na pełnych danych, bez strojenia) ===")
    lr_cv = run_cv(logreg_model(), X, y)
    rf_cv = run_cv(baseline_model(), X, y)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw = neg / max(pos, 1)
    xgb_cv_before = run_cv(xgb_model(scale_pos_weight=spw), X, y)

    print("CV LogisticRegression:", lr_cv)
    print("CV RandomForest:", rf_cv)
    print("scale_pos_weight:", spw)
    print("CV XGBoost:", xgb_cv_before)

    # 5) Hold-out (SCENARIUSZ A: trening na pełnym train)
    print("\n=== Hold-out (train pełny) ===")
    lr_hold_full = train_and_evaluate(logreg_model(), X_train, X_test, y_train, y_test)
    rf_hold_full = train_and_evaluate(baseline_model(), X_train, X_test, y_train, y_test)
    xgb_hold_full = train_and_evaluate(xgb_model(scale_pos_weight=spw), X_train, X_test, y_train, y_test)

    print("LogisticRegression:", lr_hold_full)
    print("RandomForest:", rf_hold_full)
    print("XGBoost:", xgb_hold_full)

    # 6) Hold-out (SCENARIUSZ B: trening na zbalansowanym train)
    print("\n=== Hold-out (train zbalansowany 1:1) ===")
    lr_hold_bal = train_and_evaluate(logreg_model(), X_train_bal, X_test, y_train_bal, y_test)
    rf_hold_bal = train_and_evaluate(baseline_model(), X_train_bal, X_test, y_train_bal, y_test)

    # przy undersamplingu zwykle scale_pos_weight = 1.0
    xgb_hold_bal = train_and_evaluate(xgb_model(scale_pos_weight=1.0), X_train_bal, X_test, y_train_bal, y_test)

    print("LogisticRegression (balanced):", lr_hold_bal)
    print("RandomForest (balanced):", rf_hold_bal)
    print("XGBoost (balanced):", xgb_hold_bal)

    # 7) Strojenie Optuna (trzymamy na pełnych danych; cel: ROC-AUC w CV)
    print("\n=== Optuna tuning (XGBoost, na pełnych danych) ===")
    best_params = tune_xgb_optuna(X, y, n_trials=10, random_state=42, n_splits=10)
    print("Najlepsze parametry z Optuny:")
    print(best_params)

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/best_xgb_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print("Zapisano outputs/best_xgb_params.json")

    # 8) Po tuningu: CV + hold-out (pełny train)
    print("\n=== 10-fold CV (PO strojeniu) ===")
    xgb_cv_after = run_cv(xgb_model(override_params=best_params), X, y)
    print("CV XGBoost tuned:", xgb_cv_after)

    print("\n=== Hold-out (PO strojeniu, train pełny) ===")
    xgb_hold_tuned_full = train_and_evaluate(
        xgb_model(override_params=best_params), X_train, X_test, y_train, y_test
    )
    print("XGBoost tuned:", xgb_hold_tuned_full)

    print_comparison_table(
        title="Porównanie 10-fold CV: XGBoost przed vs po Optuna",
        before=xgb_cv_before,
        after=xgb_cv_after,
    )
    print_comparison_table(
        title="Porównanie hold-out (train pełny): XGBoost przed vs po Optuna",
        before=xgb_hold_full,
        after=xgb_hold_tuned_full,
    )

    # 9) Wykresy – wybieramy jeden scenariusz do wizualizacji (polecam: train pełny + tuned)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    # Uwaga: to wymaga, żeby train_and_evaluate zwracał też "model"
    # Jeśli nie zwraca, to w train.py dopisz: "model": model
    models_for_plots = {
        "LogReg_full": _extract_model(lr_hold_full, None),
        "RF_full": _extract_model(rf_hold_full, None),
        "XGB_full": _extract_model(xgb_hold_full, None),
        "XGB_tuned_full": _extract_model(xgb_hold_tuned_full, None),
    }

    # Jeżeli nie masz "model" w wynikach, zrobimy szybki fallback: uczymy raz do wykresów
    if any(v is None for v in models_for_plots.values()):
        print("[INFO] Brak 'model' w train_and_evaluate -> fituję modele raz do wykresów.")
        models_for_plots = {
            "LogReg_full": logreg_model().fit(X_train, y_train),
            "RF_full": baseline_model().fit(X_train, y_train),
            "XGB_full": xgb_model(scale_pos_weight=spw).fit(X_train, y_train),
            "XGB_tuned_full": xgb_model(override_params=best_params).fit(X_train, y_train),
        }

    plot_roc_curves(models_for_plots, X_test, y_test, "outputs/models/roc_holdout.png")
    plot_pr_curves(models_for_plots, X_test, y_test, "outputs/models/pr_holdout.png")
    plot_confusion_matrices(models_for_plots, X_test, y_test, "outputs/models")

    plot_feature_importance(
        models_for_plots["RF_full"], list(X.columns),
        "outputs/models/fi_rf.png",
        title="RandomForest feature importance (hold-out, train pełny)",
    )
    plot_feature_importance(
        models_for_plots["XGB_tuned_full"], list(X.columns),
        "outputs/models/fi_xgb_tuned.png",
        title="XGBoost tuned feature importance (hold-out, train pełny)",
    )

    print("Zapisano wykresy modeli do outputs/models/")


if __name__ == "__main__":
    main()
