# visualize_binary.py — binary geo-scatter for IF or OCSVM
from __future__ import annotations
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import joblib

# ---- Repo-agnostic imports: your repos expose a "models" module under src/ ----
from src.data import load_csv, preprocess_tracks
from src.features import compute_features, make_feature_matrix
from src import models  # <-- IF: src/__init__.py maps to models_iforest; OCSVM: to models_ocsvm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out", type=str, default=None, help="PNG path (default: outputs/geo_scatter_binary.png)")
    ap.add_argument("--score_quantile", type=float, default=0.95,
                    help="Quantile threshold used if no label col is given")
    ap.add_argument("--sample", type=int, default=80000, help="Max points to plot for speed")
    args = ap.parse_args()

    # ---- Load config / data / features ----
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_csv(cfg["data"]["csv_path"], cfg)
    df = preprocess_tracks(df, cfg)
    df = compute_features(df, cfg)
    X, cols = make_feature_matrix(df, cfg)
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = 0.0

    # ---- Scale + model ----
    save_dir = Path(cfg["train"]["save_dir"])
    scaler_path = save_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        Xs = scaler.transform(X)
    else:
        Xs = X

    model_path = save_dir / cfg["train"]["save_name"]
    model = models.load_model(str(model_path))

    # ---- Scores -> binary anomalies ----
    # IF & OCSVM both expose models.score_samples(X) returning higher = more anomalous
    scores = models.score_samples(model, Xs)

    # If we already have labels, prefer them for the binary plot; otherwise threshold scores
    if "label" in df.columns and df["label"].notna().any():
        y_bin = df["label"].fillna(0).astype(int).values
        used_labels = True
    else:
        thr = float(np.quantile(scores, args.score_quantile))
        y_bin = (scores >= thr).astype(int)
        used_labels = False

    df_plot = df.copy()
    df_plot["anomaly"] = y_bin

    # Optional: downsample for speed/clarity
    if len(df_plot) > args.sample:
        # Keep all anomalies + a sample of normals
        anomalies = df_plot[df_plot["anomaly"] == 1]
        normals = df_plot[df_plot["anomaly"] == 0].sample(
            max(args.sample - len(anomalies), 0), random_state=42
        )
        df_plot = pd.concat([anomalies, normals], ignore_index=True)

    # Ensure lon/lat columns exist (your CSV uses x=lon, y=lat and we remap in load_csv)
    for c in ("lon", "lat"):
        if c not in df_plot.columns:
            df_plot[c] = np.nan

    # ---- Binary geo-scatter ----
    plt.figure(figsize=(9, 8))
    normal = df_plot[df_plot["anomaly"] == 0]
    anomal = df_plot[df_plot["anomaly"] == 1]

    # Blue for normal, red for anomaly
    plt.scatter(normal["lon"], normal["lat"], s=3, c="#3366cc", alpha=0.6, label="Normal")
    plt.scatter(anomal["lon"], anomal["lat"], s=6, c="#cc3333", alpha=0.9, label="Anomaly")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    title_src = "labels" if used_labels else f"threshold@q={args.score_quantile:.2f}"
    plt.title(f"Binary Geo-Scatter ({title_src})")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.35)

    out_dir = Path(cfg["eval"]["plots_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = Path(args.out) if args.out else (out_dir / "geo_scatter_binary.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    print(f"[binary-plot] saved: {out_png}")

if __name__ == "__main__":
    main()
