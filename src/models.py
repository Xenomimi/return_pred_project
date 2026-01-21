from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.config import XGB_PARAMS


def logreg_model():
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ])


def baseline_model():
    return RandomForestClassifier()


def xgb_model(scale_pos_weight: float | None = None, override_params: dict | None = None):
    params = dict(XGB_PARAMS)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    if override_params is not None:
        params.update(override_params)
    return XGBClassifier(**params)
