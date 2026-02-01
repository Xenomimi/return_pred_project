from __future__ import annotations
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import pandas as pd


def undersample_train(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Undersampling klasy większościowej (0) do liczebności klasy 1.
    Balansujemy TYLKO zbiór treningowy.
    """
    df = X.copy()
    df["__y__"] = y.values

    df_pos = df[df["__y__"] == 1]
    df_neg = df[df["__y__"] == 0]

    n = len(df_pos)
    df_neg_down = df_neg.sample(n=n, random_state=random_state)

    df_bal = pd.concat([df_pos, df_neg_down], axis=0).sample(frac=1.0, random_state=random_state)

    y_bal = df_bal["__y__"].astype(int)
    X_bal = df_bal.drop(columns=["__y__"])
    return X_bal, y_bal


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    # uczymy raz
    model.fit(X_train, y_train)

    # predykcje do metryk i wykresów
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "y_pred": preds,
        "y_proba": proba,
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, preds),
        "acc": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "cm": confusion_matrix(y_test, preds).tolist()
    }
