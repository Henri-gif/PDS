from __future__ import annotations
import joblib, numpy as np
from typing import Dict
from sklearn.ensemble import IsolationForest

def build_model(cfg: Dict) -> IsolationForest:
    mcfg = cfg.get('model_iforest', {})
    return IsolationForest(
        n_estimators=mcfg.get('n_estimators', 200),
        max_samples=mcfg.get('max_samples', 'auto'),
        contamination=mcfg.get('contamination', 0.12),
        random_state=mcfg.get('random_state', 42),
        n_jobs=-1
    )

def fit_model(model, X):
    model.fit(X)
    return model

def score_samples(model, X) -> np.ndarray:
    raw = -model.decision_function(X).ravel()
    rmin, rmax = raw.min(), raw.max()
    if not np.isfinite(rmin) or not np.isfinite(rmax) or (rmax - rmin) <= 1e-12:
        return np.zeros_like(raw)
    return (raw - rmin) / (rmax - rmin)

def save_model(model, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
