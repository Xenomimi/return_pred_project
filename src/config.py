from pathlib import Path

DATA_PATH = Path("data/order_dataset.csv")
TARGET_COL = "returned"
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": RANDOM_STATE
}
