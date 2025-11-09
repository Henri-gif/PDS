from __future__ import annotations
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd
import joblib
from src.data import load_csv, preprocess_tracks
from src.features import compute_features, make_feature_matrix
from src import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--out_csv', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    cfg = dict(cfg); cfg['data'] = dict(cfg['data'])
    cfg['data']['csv_path'] = args.input_csv

    df = load_csv(cfg['data']['csv_path'], cfg)
    df = preprocess_tracks(df, cfg)
    df = compute_features(df, cfg)
    X, cols = make_feature_matrix(df, cfg)

    X = np.asarray(X, dtype=float); X[~np.isfinite(X)] = 0.0
    scaler_path = Path(cfg['train']['save_dir']) / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    else:
        print(f"[warn] No scaler at {scaler_path}; using raw features.")

    model = models.load_model(str(Path(cfg['train']['save_dir'])/cfg['train']['save_name']))
    scores = models.score_samples(model, X)
    df['anomaly_score'] = scores

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ['mmsi','ts','lat','lon','sog','cog','anomaly_score']
    if 'label' in df.columns: keep_cols.append('label')
    df[keep_cols].to_csv(out, index=False)
    print(f"[predict] wrote {len(df):,} rows to {out}")

if __name__ == '__main__':
    main()
