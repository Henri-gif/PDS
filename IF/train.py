from __future__ import annotations
import argparse, yaml
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.data import load_csv, preprocess_tracks
from src.features import compute_features, make_feature_matrix
from src import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    df = load_csv(cfg['data']['csv_path'], cfg)
    df = preprocess_tracks(df, cfg)
    df = compute_features(df, cfg)
    X, cols = make_feature_matrix(df, cfg)

    X = np.asarray(X, dtype=float); X[~np.isfinite(X)] = 0.0
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model = models.build_model(cfg)
    model = models.fit_model(model, Xs)

    save_dir = Path(cfg['train']['save_dir']); save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / cfg['train']['save_name']
    models.save_model(model, str(out_path))
    joblib.dump(scaler, save_dir / "scaler.pkl")

    print(f"[train] rows={len(df):,} features={len(cols)} saved={out_path}")

if __name__ == '__main__':
    main()
