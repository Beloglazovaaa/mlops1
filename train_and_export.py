import argparse
import json
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42
TIME_COL = "transaction_time"
DROP_COLS = ["name_1", "name_2", "street", "jobs", "merch", "us_state", "post_code"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - train_and_export - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_and_export")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TIME_COL not in df.columns:
        logger.warning("Column '%s' not found. Skipping time features.", TIME_COL)
        return df
    dt = pd.to_datetime(df[TIME_COL], errors="coerce")
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df.drop(columns=[TIME_COL], inplace=True, errors="ignore")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
    return df


def fit_label_encoders(df: pd.DataFrame) -> dict:
    encoders = {}
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        cats = pd.Series(df[col].astype("string").unique())
        mapping = {v: int(i) for i, v in enumerate(cats.fillna("__nan__"))}
        encoders[col] = mapping
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        mapping = encoders.get(col)
        if mapping is None:
            # fallback: кодируем по категориям, если почему-то энкодера нет
            df[col] = pd.Categorical(df[col]).codes
        else:
            df[col] = (
                df[col].astype("string")
                .fillna("__nan__")
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )
    return df


def train_and_export(train_path: str, test_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) читаем данные
    logger.info("Loading train: %s", train_path)
    train = pd.read_csv(train_path)
    logger.info("Loading test:  %s", test_path)
    test = pd.read_csv(test_path)

    assert "target" in train.columns, "train.csv must contain 'target' column"
    y = train["target"].astype(int)
    X = train.drop(columns=["target"])

    # 2) общий препроцесс (как в проде): склейка для устойчивых категорий
    df_all = pd.concat([X, test], keys=["train", "test"])
    df_all = add_time_features(df_all)
    df_all = drop_columns(df_all)

    encoders = fit_label_encoders(df_all)
    df_all_enc = apply_encoders(df_all, encoders)

    X_train = df_all_enc.loc["train"]
    X_test = df_all_enc.loc["test"]  # для фиксации порядка фич
    X_train.index = X.index
    X_test.index = test.index
    feature_names = list(X_train.columns)

    pos_rate = float(y.mean())
    logger.info("Train shape: %s, Pos rate: %.4f", X_train.shape, pos_rate)

    # 3) валидационное разбиение
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4) модель с балансировкой классов (часто критично для fraud)
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_tr, y_tr)

    # 5) подбор порога по F1
    if hasattr(model, "predict_proba"):
        val_proba = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.0, 1.0, 101)
        f1_by_t = [f1_score(y_val, (val_proba >= t).astype(int)) for t in thresholds]
        best_idx = int(np.argmax(f1_by_t))
        best_thr = float(thresholds[best_idx])
        best_f1 = float(f1_by_t[best_idx])
    else:
        # fallback: если модели нет predict_proba (для RF он есть)
        val_pred = model.predict(X_val)
        best_thr = 0.5
        best_f1 = float(f1_score(y_val, val_pred))

    logger.info("Best F1 on val: %.4f at threshold=%.3f", best_f1, best_thr)

    # 6) сохраняем артефакты
    joblib.dump(model, out / "model.pkl")
    joblib.dump(encoders, out / "encoders.pkl")
    with open(out / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    with open(out / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thr}, f)

    logger.info("Saved artifacts to %s: model.pkl, encoders.pkl, feature_names.json, threshold.json", out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train RF and export inference artifacts")
    ap.add_argument("--train", required=True, help="Path to train.csv (with 'target')")
    ap.add_argument("--test", required=True, help="Path to test.csv (same schema)")
    ap.add_argument("--out", default="models", help="Output directory for artifacts")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_export(args.train, args.test, args.out)

