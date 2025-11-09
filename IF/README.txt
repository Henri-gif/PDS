Isolation Forest (AIS anomalies) — PACK ADAPTÉ CSV
======================================================
Schéma CSV (via config): t, AgentID, x, y, speed, is_anomaly, anomaly_type

1) Copiez le contenu de cette archive dans votre dépôt IF (à la racine).
2) Vérifiez/éditez configs/default.yaml (chemin vers le CSV).
3) Commandes (PowerShell) :
   python .\train.py --config .\configs\default.yaml
   python .\eval.py  --config .\configs\default.yaml
   python .\visualize.py --config .\configs\default.yaml
   python .\predict.py --config .\configs\default.yaml --input_csv "C:\chemin\autre.csv" --out_csv ".\outputs\if_scores.csv"
