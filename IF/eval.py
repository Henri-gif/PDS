from __future__ import annotations
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
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
    scaler_path = Path(cfg['train']['save_dir']) / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    else:
        print(f"[warn] No scaler at {scaler_path}; using raw features.")

    model = models.load_model(str(Path(cfg['train']['save_dir'])/cfg['train']['save_name']))
    scores = models.score_samples(model, X)
    df['anomaly_score'] = scores

    plots_dir = Path(cfg['eval']['plots_dir']); plots_dir.mkdir(parents=True, exist_ok=True)

    if 'label' in df.columns and df['label'].notna().any():
        y = df['label'].fillna(0).astype(int).values
        try:
            roc = roc_auc_score(y, scores)
        except Exception:
            roc = float('nan')
        ap = average_precision_score(y, scores)
        print(f"[eval] ROC-AUC={roc:.4f} PR-AUC={ap:.4f}")

        pr_p, pr_r, _ = precision_recall_curve(y, scores)
        plt.figure(); plt.plot(pr_r, pr_p); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall')
        plt.savefig(plots_dir/'pr_curve.png', dpi=150); plt.close()

        fpr, tpr, _ = roc_curve(y, scores)
        plt.figure(); plt.plot(fpr, tpr); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC')
        plt.savefig(plots_dir/'roc_curve.png', dpi=150); plt.close()

        q = cfg['eval'].get('score_quantile_threshold', 0.95)
        thr = float(np.quantile(scores, q))
        y_pred = (scores >= thr).astype(int)
        cm = pd.crosstab(pd.Series(y, name='true'), pd.Series(y_pred, name='pred'))
        cm.to_csv(plots_dir/'confusion_matrix.csv', index=True)
        print(f"[eval] Saved curves and confusion matrix to {plots_dir}")
    else:
        print('[eval] No labels found; exporting scores only.')

    out_scored = plots_dir/'scored.csv'
    keep_cols = ['mmsi','ts','lat','lon','sog','cog','anomaly_score']
    if 'label' in df.columns: keep_cols.append('label')
    df[keep_cols].to_csv(out_scored, index=False)
    print(f"[eval] Saved scored CSV to {out_scored}")

if __name__ == '__main__':
    main()
