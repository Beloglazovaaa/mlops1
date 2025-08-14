import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/app/models")


def _load_model() -> Tuple[object, list]:
    import joblib
    model = joblib.load(MODELS_DIR / "model.pkl")  # RandomForestClassifier
    with open(MODELS_DIR / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return model, feature_names


def _load_threshold() -> float:
    t_path = MODELS_DIR / "threshold.json"
    if t_path.exists():
        try:
            with open(t_path, "r", encoding="utf-8") as f:
                return float(json.load(f)["threshold"])
        except Exception as e:
            logger.warning("Failed to read threshold.json, fallback to 0.5: %s", e)
    return 0.5


def get_top_features(n: int = 5) -> dict:
    model, feature_names = _load_model()
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {}
    idx = np.argsort(importances)[::-1][:n]
    return {feature_names[i]: float(importances[i]) for i in idx}


def plot_density(pred_proba: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(9, 6))
    plt.hist(pred_proba, bins=50, density=True, alpha=0.6)
    plt.title("Density of predicted probabilities")
    plt.xlabel("Probability of class 1")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def make_pred(X: pd.DataFrame, original_index):
    model, _ = _load_model()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        thr = _load_threshold()
        preds = (proba >= thr).astype(int)
        logger.info("Prediction complete: %d rows, pos_rate=%.4f, thr=%.3f",
                    len(X), preds.mean(), thr)
    else:
        preds = model.predict(X)
        proba = preds.astype(float)
        logger.info("Prediction complete (no proba): %d rows, pos_rate=%.4f",
                    len(X), preds.mean())

    submission = pd.DataFrame({
        "index": original_index,
        "prediction": preds
    })
    return submission, proba
