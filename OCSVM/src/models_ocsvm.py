from __future__ import annotations
import joblib
import numpy as np
from typing import Dict
from sklearn.svm import OneClassSVM

def build_model(cfg: Dict) -> OneClassSVM:
    mcfg = cfg.get('model', {})
    return OneClassSVM(
        kernel=mcfg.get('kernel', 'rbf'),
        gamma=mcfg.get('gamma', 'scale'),
        nu=mcfg.get('nu', 0.12),
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
