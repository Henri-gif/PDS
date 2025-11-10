# Maritime Anomaly Detection — Isolation Forest (AIS)

Unsupervised anomaly detection for AIS vessel movement using **Isolation Forest**. Loads a CSV, engineers basic movement features, trains the model, scores anomalies, and saves plots/outputs.

---

## Requirements

* Python 3.10+
* `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pyyaml`, `joblib`, `tqdm`

**Setup**

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
pip install -U scikit-learn pandas numpy matplotlib pyyaml joblib tqdm
```

---

## Configure

Edit `configs/default.yaml` to point to your CSV and columns:

```yaml
data:
  csv_path: "path/to/synthetic_vessel_tracks_with_anomalies_20251007.csv"
  time_col: "t"          # timestamp
  mmsi_col: "AgentID"    # vessel id
  lat_col: "y"
  lon_col: "x"
  sog_col: "speed"
  label_col: "is_anomaly"  # optional (0/1). Needed for PR/ROC/CM.
train:
  save_dir: "outputs"
  save_name: "iforest_model.pkl"
eval:
  plots_dir: "outputs"
  score_quantile_threshold: 0.95   # used for binary map if no labels
```

---

## Run

**Train**

```bash
python train.py --config configs/default.yaml
```

**Evaluate** (creates scores, PR/ROC if labels exist, confusion matrix)

```bash
python eval.py --config configs/default.yaml
```

**Visualize**

```bash
# Continuous anomaly intensity map
python visualize.py --config configs/default.yaml

# Binary map (red=anomaly, blue=normal). Uses labels if present, else score quantile.
python visualize_binary.py --config configs/default.yaml --score_quantile 0.95
```

**Score a new CSV**

```bash
python predict.py --config configs/default.yaml --input_csv path/to/new.csv --out_csv outputs/if_scores_new.csv
```

---

## What you get (in `outputs/`)

* `iforest_model.pkl`, `scaler.pkl`
* `scored.csv` — `mmsi, ts, lat, lon, sog, cog, anomaly_score[, label]`
* `pr_curve.png`, `roc_curve.png` (if labels)
* `confusion_matrix.csv` (at the chosen threshold)
* `geo_scatter.png` (intensity), `geo_scatter_binary.png` (red/blue)

---

## Notes

* **Anomaly score:** we convert `decision_function` so **higher = more anomalous**.
* **Thresholds:** if no labels, binary map uses **score quantile** (default **0.95**). With labels, pick threshold from PR/ROC or an operational FPR.
* **Feature scaling:** features are standardized; NaN/Inf are handled before training.
* **Why counts can differ:** PR/ROC/CM use only rows with **valid labels & finite features**.

---

## Minimal repo layout

```
configs/default.yaml
src/{data.py,features.py,models_iforest.py,utils.py,__init__.py}
train.py  eval.py  predict.py
visualize.py  visualize_binary.py
outputs/  (created at runtime)
```

That’s it—point the config to your CSV, run `train.py`, then `eval.py` and `visualize_*` to inspect results.

