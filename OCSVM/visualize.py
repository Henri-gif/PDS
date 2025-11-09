from __future__ import annotations
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import joblib
from src.data import load_csv, preprocess_tracks
from src.features import compute_features, make_feature_matrix
from src import models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--sample', type=int, default=50000)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    df = load_csv(cfg['data']['csv_path'], cfg)
    df = preprocess_tracks(df, cfg)
    df = compute_features(df, cfg)
    X, cols = make_feature_matrix(df, cfg)

    X = np.asarray(X, dtype=float); X[~np.isfinite(X)] = 0.0
    scaler_path = Path(cfg['train']['save_dir']) / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path); Xs = scaler.transform(X)
    else:
        Xs = X

    model = models.load_model(str(Path(cfg['train']['save_dir'])/cfg['train']['save_name']))
    scores = models.score_samples(model, Xs)
    df['anomaly_score'] = scores

    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)

    s = df['anomaly_score'].values
    vmin, vmax = np.percentile(s, 5), np.percentile(s, 95)
    if vmin == vmax:
        vmin, vmax = s.min(), s.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    out_dir = Path(cfg['eval']['plots_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    sc = plt.scatter(df['lon'], df['lat'], c=df['anomaly_score'], s=2, alpha=0.9, norm=norm, cmap='viridis')
    plt.colorbar(sc, label='Anomaly score')
    plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.title('Geographic anomaly intensity')
    plt.savefig(out_dir/'geo_scatter.png', dpi=200); plt.close()

    keep = ['mmsi','ts','lat','lon','sog','cog','anomaly_score']
    if 'label' in df.columns: keep.append('label')
    df[keep].to_csv(out_dir/'scored.csv', index=False)

    print(f'[visualize] saved plots and scored.csv to {out_dir}')

if __name__ == '__main__':
    main()
