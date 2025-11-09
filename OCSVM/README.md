# Maritime Anomaly Detection — One-Class SVM

This repository implements unsupervised maritime movement anomaly detection on AIS-like CSV data using **One-Class SVM (sklearn)**.
It is designed for synthetic datasets generated from agent-based simulators (ports, coasts, protected areas, etc.).

## Features
- CSV ingestion with flexible column mapping (MMSI, timestamp, lat, lon, SOG, COG, heading, optional label).
- Track-wise preprocessing (sorting, dedup, sanity checks, gap filtering, resampling optional).
- Rich kinematic features per message: speed, acceleration, turn rate, course change, along-track distance, straight-line deviation, stop/go flags.
- Model training with reproducible config (YAML).
- Evaluation with ROC/PR curves if a `label` column is present (0: normal, 1: anomaly).
- Per-point anomaly scoring + CSV export.
- Basic explainability via permutation importance on anomaly scores to highlight influential features.
- Minimal external deps; pure-Python haversine (no geospatial stack needed).

## Quickstart
1) Create and activate a virtual environment (recommended).
2) Install requirements.
3) Edit `configs/default.yaml` (or pass CLI args) to point to your CSV.
4) Run the pipeline.

```bash
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
# or: py -m venv .venv && .venv\Scripts\activate  # Windows

pip install -r requirements.txt
python train.py --config configs/default.yaml
python eval.py  --config configs/default.yaml
python predict.py --config configs/default.yaml --input_csv path/to/new_data.csv --out_csv outputs/predictions.csv
python visualize.py --config configs/default.yaml --sample 50000
```

### Expected CSV schema
Your CSV **must** have at least these columns (case-insensitive names are accepted via config mapping):
- `mmsi` (int/str)
- `timestamp` (ISO8601 or epoch seconds)
- `lat` (degrees)
- `lon` (degrees)
- `sog` (speed over ground, kn)
- `cog` (course over ground, deg, 0–360)

Optional but helpful:
- `heading` (deg), `nav_status` (str/int), `label` (0 normal, 1 anomaly)

### Notes
- Unsupervised models assume most data is normal. Ensure your training set reflects that.
- If your simulator injects anomalies, use `label` only for **evaluation**, not training.
- Sampling/windowing: this code scores **per AIS message**. If you need window-level anomalies, aggregate scores by MMSI/time window after prediction.
