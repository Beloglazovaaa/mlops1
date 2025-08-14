import json
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path('/app/models')
TIME_COL = 'transaction_time'
DROP_COLS = ['name_1', 'name_2', 'street', 'jobs', 'merch', 'us_state', 'post_code']


def _load_artifacts():
    import joblib
    encoders = joblib.load(MODELS_DIR / 'encoders.pkl')  # dict[col] -> {category:int}
    with open(MODELS_DIR / 'feature_names.json', 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    return encoders, feature_names


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df['hour'] = df[TIME_COL].dt.hour
    df['day_of_week'] = df[TIME_COL].dt.dayofweek
    df['month'] = df[TIME_COL].dt.month
    df['year'] = df[TIME_COL].dt.year
    df.drop(columns=[TIME_COL], inplace=True)
    return df


def _drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols, errors='ignore')


def _apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        mapping = encoders.get(col)
        if mapping is None:
            # на всякий: устойчивое кодирование без артефактов
            df[col] = pd.Categorical(df[col]).codes
        else:
            df[col] = (
                df[col].astype('string').fillna('__nan__').map(mapping).fillna(-1).astype(int)
            )
    return df


def load_and_preprocess(path_to_file: str) -> Tuple[pd.DataFrame, pd.Index]:
    logger.info("Loading data from %s", path_to_file)
    raw = pd.read_csv(path_to_file)
    original_index = raw.index

    df = _add_time_features(raw)
    df = _drop_columns(df)

    encoders, feature_names = _load_artifacts()
    df = _apply_encoders(df, encoders)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    logger.info("Preprocessed shape: %s", df.shape)
    return df, original_index
