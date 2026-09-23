# predict.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "churn_model_v1.joblib"
OHE_PATH = BASE_DIR / "models" / "ohe.joblib"
FEATURE_INFO_PATH = BASE_DIR / "models" / "feature_info.joblib"

_model = None
_ohe = None
_numeric_cols = None
_categorical_cols = None
_feature_names = None
_threshold = 0.5

def _load_all():
    global _model, _ohe, _numeric_cols, _categorical_cols, _feature_names, _threshold
    if _model is None:
        # Модель
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict):
            _model = data["model"]
            _threshold = data.get("threshold", 0.5)
        else:
            _model = data
        
        # Энкодер
        _ohe = joblib.load(OHE_PATH)
        
        # Инфо о фичах
        info = joblib.load(FEATURE_INFO_PATH)
        _numeric_cols = info["numeric_cols"]
        _categorical_cols = info["categorical_cols"]
        _feature_names = info["feature_names"]

def predict_churn(features_dict: dict) -> dict:
    _load_all()
    
    X_new = pd.DataFrame([features_dict])
    
    # Числовые
    X_num = X_new[_numeric_cols].to_numpy()
    
    # Категориальные → OneHot
    X_cat = _ohe.transform(X_new[_categorical_cols])
    
    # Конкатенация
    X_encoded = np.hstack([X_num, X_cat])
    
    # Приводим к DataFrame с теми же именами колонок, на которых модель обучалась
    X_df = pd.DataFrame(X_encoded, columns=_feature_names)
    
    proba = _model.predict_proba(X_df)[0, 1]
    label = int(proba > _threshold)
    return {
        "churn_probability": float(proba),
        "churn_label": label,
    }
    
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает предсказания для всего DataFrame.
    Возвращает копию df с двумя новыми колонками: churn_probability, churn_label
    """
    df = df.copy()
    probs = []
    labels = []

    for _, row in df.iterrows():
        payload = row.to_dict()
        pred = predict_churn(payload)
        probs.append(pred["churn_probability"])
        labels.append(pred["churn_label"])

    df["churn_probability"] = probs
    df["churn_label"] = labels
    return df