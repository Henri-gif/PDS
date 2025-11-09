# ML-Powered Anomaly Detection Dashboard for AIS Data

## Overview

This dashboard is built using Panel and provides an interactive interface for loading AIS (Automatic Identification System) data from CSV files, mapping columns, training machine learning models for anomaly detection, visualizing results on maps and tables, and providing feedback on detected anomalies.

The dashboard supports Isolation Forest and One-Class SVM models for detecting anomalies in vessel track data. It includes geographic visualization (with GeoViews or Folium), data filtering, detailed anomaly inspection, and export of confirmed anomalies.

## Features

- Upload and load CSV files with AIS data (supports auto-mapping for columns like 'x', 'y', 'is_anomaly').
- Interactive column mapping for latitude, longitude, and ML features.
- Train ML models: Isolation Forest or One-Class SVM with configurable parameters.
- Detect anomalies with adjustable threshold.
- Interactive filters for location, anomaly scores, and features.
- Visualization:
  - Interactive map (GeoViews for geographic projection, Folium fallback, or Cartesian plot).
  - Data table with pagination.
  - Detailed view of individual anomalies with explanations.
- Provide feedback (confirm/reject) on anomalies and download confirmed ones as CSV.
- Stateful interface with real-time updates.

## Requirements

- Python 3.8+
- Core libraries:
  - panel
  - holoviews
  - hvplot
  - pandas
  - numpy
  - scikit-learn
- Optional for mapping:
  - geoviews (for geographic maps)
  - folium (fallback mapping)
- For full functionality, install with:
```
pip install panel holoviews hvplot pandas numpy scikit-learn geoviews folium
```
Note: GeoViews requires additional dependencies like cartopy and geopandas for proper geographic rendering.

## Usage

1. **Prepare Data**:
   - Ensure your CSV has columns for latitude, longitude, and features (e.g., 'speed').
   - Default CSV path: `data/synthetic_vessel_tracks_with_anomalies_20251007.csv` (update in code if needed).

2. **Run the Dashboard**:
   - Save the code as `anomaly_dashboard.py`.
   - Run with Panel:
     ```
     panel serve anomaly_dashboard.py --show
     ```
   - Access at http://localhost:5006/anomaly_dashboard.

3. **Interface Steps**:
   - **Upload/Load CSV**: Use file input or load default.
   - **Map Columns**: Select lat/lon and ML features, apply mapping.
   - **Configure ML**: Choose model, set parameters, train.
   - **Detect Anomalies**: Run detection, adjust threshold.
   - **Filter Data**: Use sliders for location/features/scores.
   - **View Results**: Map shows normal (blue) vs anomalies (red); table lists data.
   - **Inspect Anomalies**: Navigate with slider, view details/explanations, confirm/reject.
   - **Download**: Export confirmed anomalies as CSV.

## How It Works

1. **Data Loading**: Reads CSV, auto-detects schema if possible, cleans data.
2. **Column Mapping**: User selects lat/lon/features; data is filtered for valid coordinates.
3. **Model Training**: Scales features, trains selected model on all data (unsupervised).
4. **Anomaly Detection**: Computes scores, applies threshold; generates explanations using DBSCAN on normal data.
5. **Filtering**: Applies range filters to location/features/scores; updates views.
6. **Visualization**:
   - Map: Plots points with colors; prefers GeoViews, falls back to Folium or Cartesian.
   - Table: Paginated Tabulator widget.
   - Details: Shows selected anomaly info, explanation, feedback controls.
7. **Feedback**: Stores confirm/reject verdicts; exports confirmed rows.

## Customization

- **Default CSV**: Update `DEFAULT_CSV` path.
- **Model Parameters**: Adjust sliders for contamination/nu.
- **Features**: Select multiple numeric columns for ML.
- **Threshold**: Slider for anomaly cutoff.
- **Explanations**: Uses DBSCAN clustering on normal points for deviation analysis.
- Add more models by extending `train_model` and `detect_anomalies`.
