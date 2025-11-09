import io, os
from datetime import datetime
import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
import hvplot.pandas  # registers .hvplot
# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
# Try geoviews first; else Folium fallback
try:
    import geoviews as gv  # noqa: F401
    HAS_GV = True
except Exception:
    HAS_GV = False
try:
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False
pn.extension('tabulator', template='material')
# ---- CONFIG: default CSV to autoload (set your path here) ----
DEFAULT_CSV = "synthetic_vessel_tracks_with_anomalies_20251007.csv"  # Updated to match provided file
# ---------------- Widgets ----------------
file_input = pn.widgets.FileInput(name="Upload CSV", accept=".csv")
load_btn = pn.widgets.Button(name="Load file", button_type="primary")
status_msg = pn.pane.Markdown("📄 Please upload a CSV or click **Load file** to open the default.",
                              styles={'white-space': 'pre-wrap'})
# Column mapping widgets
lat_sel = pn.widgets.Select(name="Latitude column", options=[], value=None)
lon_sel = pn.widgets.Select(name="Longitude column", options=[], value=None)
feature_sels = pn.widgets.MultiSelect(name="Feature columns for ML", options=[], size=6)
apply_mapping_btn = pn.widgets.Button(name="Apply column mapping", button_type="primary", disabled=True)
# ML Model widgets
model_sel = pn.widgets.Select(
    name="Anomaly Detection Model",
    options=['Isolation Forest', 'One-Class SVM'],
    value='Isolation Forest'
)
if_contamination = pn.widgets.FloatSlider(name="IF Contamination", start=0.01, end=0.5, value=0.1, step=0.01)
ocsvm_nu = pn.widgets.FloatSlider(name="OCSVM nu", start=0.01, end=0.5, value=0.1, step=0.01)
threshold_slider = pn.widgets.FloatSlider(name="Anomaly Score Threshold", start=0.0, end=1.0, value=0.5, step=0.01)
train_btn = pn.widgets.Button(name="Train Model", button_type="primary", disabled=True)
detect_btn = pn.widgets.Button(name="Detect Anomalies", button_type="success", disabled=True)
# Location filter widgets
filters_box = pn.Column(name="Filters")
lat_range = pn.widgets.RangeSlider(name="Latitude range", start=-90, end=90, value=(-90, 90), visible=False)
lon_range = pn.widgets.RangeSlider(name="Longitude range", start=-180, end=180, value=(-180, 180), visible=False)
score_slider = pn.widgets.RangeSlider(name="Anomaly score range", start=0.0, end=1.0, value=(0.0, 1.0), visible=False)
apply_filters_btn = pn.widgets.Button(name="Apply filters", button_type="primary", disabled=True)
reset_filters_btn = pn.widgets.Button(name="Reset filters", button_type="default", disabled=True)
# Anomaly navigation widgets
idx_slider = pn.widgets.IntSlider(name="Anomaly index", start=0, end=0, value=0)
prev_btn = pn.widgets.Button(name="◀ Prev")
next_btn = pn.widgets.Button(name="Next ▶")
explanation_pane = pn.pane.Markdown("### Anomaly Explanation\n\nSelect an anomaly to see why it was detected.")
confirm_btn = pn.widgets.Button(name="Confirm anomaly", button_type="success", disabled=True)
reject_btn = pn.widgets.Button(name="Reject anomaly", button_type="warning", disabled=True)
feedback_note = pn.pane.Markdown("")
download_btn = pn.widgets.FileDownload(
    label="Download confirmed anomalies", filename="confirmed_anomalies.csv",
    button_type="primary", disabled=True, callback=None
)
# Panes
map_pane = pn.pane.HoloViews(object=hv.Curve([]), sizing_mode="stretch_both")
map_html = pn.pane.HTML(sizing_mode="stretch_both", visible=False)  # Folium fallback
table_pane = pn.widgets.Tabulator(pd.DataFrame(), sizing_mode="stretch_both", pagination='remote', page_size=25)
detail_pane = pn.Card(title="Anomaly details", collapsed=False)
# ---------------- State ----------------
class State:
    def __init__(self):
        self.df_raw = pd.DataFrame()
        self.df = pd.DataFrame()
        self.lat = None
        self.lon = None
        self.features = []
        self.feedback = {}  # original row index -> verdict

        # ML model attributes
        self.model = None
        self.scaler = StandardScaler()
        self.anomaly_scores = None
        self.anomaly_explanations = {}
        self.is_trained = False
        self.if_model = None
        self.ocsvm_model = None

    def log(self, txt):
        status_msg.object = f"{txt}"

    def load_csv(self, b: bytes) -> bool:
        self.log("📥 Reading CSV …")
        try:
            # Try different encodings and separators
            try:
                df = pd.read_csv(io.BytesIO(b))
            except:
                try:
                    df = pd.read_csv(io.BytesIO(b), sep=';')
                except:
                    df = pd.read_csv(io.BytesIO(b), encoding='latin-1')
        except Exception as e:
            status_msg.object = f"❌ Failed to read CSV: {e}"
            return False

        if df.empty:
            status_msg.object = "❌ CSV is empty."
            return False
        cols = list(df.columns)
        self.log(f"✅ Read {len(df)} rows, {len(cols)} columns")
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        cols = list(df.columns)
        # Auto-map your schema: x (lon), y (lat), is_anomaly (bool)
        if all(c in cols for c in ['x', 'y', 'is_anomaly']):
            self.log("🔎 Auto-mapping columns (x,y,is_anomaly).")
            df['x'] = pd.to_numeric(df['x'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            lab = df['is_anomaly']
            if lab.dtype == bool:
                df['is_anomaly'] = lab.astype(int)
            else:
                df['is_anomaly'] = (lab.astype(str).str.strip().str.lower().isin(['1', 'true', 'yes'])).astype(int)
            df = df.dropna(subset=['x', 'y']).reset_index(drop=True)
            self.df_raw = df
            self.df = self.df_raw.copy()
            self.lat, self.lon = 'y', 'x'
            exclude = {'x', 'y', 'is_anomaly'}
            self.features = [c for c in self.df_raw.columns
                             if c not in exclude and pd.api.types.is_numeric_dtype(self.df_raw[c])]
            build_filters()
            # reflect mapping in selectors (user can change)
            lat_sel.options = cols
            lon_sel.options = cols
            feature_sels.options = [c for c in cols if c not in exclude and pd.api.types.is_numeric_dtype(self.df_raw[c])]
            lat_sel.value = 'y'
            lon_sel.value = 'x'
            if 'speed' in self.features:
                feature_sels.value = ['speed']
            else:
                feature_sels.value = self.features[:3] if len(self.features) > 0 else []

            apply_mapping_btn.disabled = False
            apply_filters_btn.disabled = False
            reset_filters_btn.disabled = False
            self.log(f"🧭 Using lat=y, lon=x. Features: {self.features or 'none'}")
            refresh_views()
            return True
        # Manual mapping path
        self.log("ℹ️ Auto-map not possible — please map columns manually.")
        self.df_raw = df.reset_index(drop=True)
        self.df = self.df_raw.copy()
        lat_sel.options = cols
        lon_sel.options = cols

        # Improved auto-detection
        lat_sel.value = next((c for c in cols if c.lower() in ["lat", "latitude", "y"]), None)
        lon_sel.value = next((c for c in cols if c.lower() in ["lon", "longitude", "x"]), None)

        # Auto-detect feature columns
        exclude_cols = ['t', 'AgentID', 'is_anomaly', 'anomaly_type']
        feature_candidates = [c for c in cols if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        feature_sels.options = feature_candidates

        if 'speed' in feature_candidates:
            feature_sels.value = ['speed']
        else:
            feature_sels.value = feature_candidates[:3] if len(feature_candidates) > 0 else []
        apply_mapping_btn.disabled = False
        apply_filters_btn.disabled = True  # Wait until mapping is applied
        reset_filters_btn.disabled = True
        return True

    def apply_mapping(self) -> bool:
        if not all([lat_sel.value, lon_sel.value]):
            status_msg.object = "❌ Please select Latitude and Longitude columns."
            return False
        self.lat = lat_sel.value
        self.lon = lon_sel.value
        self.features = feature_sels.value
        # Convert to numeric and drop invalid coordinates
        self.df_raw[self.lat] = pd.to_numeric(self.df_raw[self.lat], errors="coerce")
        self.df_raw[self.lon] = pd.to_numeric(self.df_raw[self.lon], errors="coerce")

        initial_count = len(self.df_raw)
        self.df_raw = self.df_raw.dropna(subset=[self.lat, self.lon]).reset_index(drop=True)
        final_count = len(self.df_raw)

        if final_count < initial_count:
            self.log(f"⚠️ Dropped {initial_count - final_count} rows with invalid coordinates")
        self.df = self.df_raw.copy()
        build_filters()
        status_msg.object = (
            f"✅ Mapping applied: lat={self.lat}, lon={self.lon}"
            f"\n📊 Features for ML: {self.features or 'none'}"
        )
        apply_filters_btn.disabled = False
        reset_filters_btn.disabled = False
        train_btn.disabled = False  # Enable training after mapping

        refresh_views()
        return True

    def train_model(self):
        """Train the selected ML model"""
        if not self.features:
            self.log("❌ No features selected. Please apply column mapping first.")
            return False

        self.log("🏋️ Training model...")

        # Prepare features
        X = self.df[self.features].values

        if len(X) == 0:
            self.log("❌ No data available for training.")
            return False

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        try:
            if model_sel.value == 'Isolation Forest':
                self.if_model = IsolationForest(
                    contamination=if_contamination.value,
                    random_state=42,
                    n_estimators=100
                )
                self.if_model.fit(X_scaled)
                self.model = self.if_model
                self.log("✅ Isolation Forest trained successfully!")

            elif model_sel.value == 'One-Class SVM':
                self.ocsvm_model = OneClassSVM(
                    nu=ocsvm_nu.value,
                    kernel='rbf',
                    gamma='scale'
                )
                self.ocsvm_model.fit(X_scaled)
                self.model = self.ocsvm_model
                self.log("✅ One-Class SVM trained successfully!")

            self.is_trained = True
            detect_btn.disabled = False
            return True

        except Exception as e:
            self.log(f"❌ Model training failed: {e}")
            return False

    def detect_anomalies(self):
        """Detect anomalies using the trained model"""
        if not self.is_trained or self.model is None:
            self.log("❌ Please train a model first.")
            return False

        self.log("🔍 Detecting anomalies...")

        X = self.df[self.features].values
        X_scaled = self.scaler.transform(X)

        try:
            if isinstance(self.model, IsolationForest):
                scores = -self.model.decision_function(X_scaled)  # Higher = more anomalous
                predictions = self.model.predict(X_scaled)
                anomalies = predictions == -1

            elif isinstance(self.model, OneClassSVM):
                scores = -self.model.decision_function(X_scaled)  # Higher = more anomalous
                predictions = self.model.predict(X_scaled)
                anomalies = predictions == -1

            # Normalize scores to 0-1 range
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min())

            # Apply threshold
            threshold = threshold_slider.value
            thresholded_anomalies = scores_normalized > threshold

            # Store results
            self.df['anomaly_score'] = scores_normalized
            self.df['is_anomaly_pred'] = thresholded_anomalies.astype(int)
            self.anomaly_scores = scores_normalized

            # Generate explanations for anomalies
            self._generate_explanations(X_scaled, scores_normalized)

            # Update filter ranges
            score_slider.start = float(scores_normalized.min())
            score_slider.end = float(scores_normalized.max())
            score_slider.value = (float(scores_normalized.min()), float(scores_normalized.max()))
            score_slider.visible = True

            anomaly_count = np.sum(thresholded_anomalies)
            self.log(f"✅ Detected {anomaly_count} anomalies (threshold: {threshold:.3f})")

            confirm_btn.disabled = False
            reject_btn.disabled = False

            refresh_views()
            return True

        except Exception as e:
            self.log(f"❌ Anomaly detection failed: {e}")
            return False

    def _generate_explanations(self, X_scaled, scores):
        """Generate explanations for why points are anomalous using DBSCAN clustering"""
        self.anomaly_explanations = {}

        if len(scores) == 0:
            return

        # Use DBSCAN to find clusters of normal behavior
        normal_mask = scores < np.percentile(scores, 50)  # Use bottom 50% as "normal"
        if np.sum(normal_mask) > 10:  # Need enough normal points
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                normal_clusters = dbscan.fit_predict(X_scaled[normal_mask])

                # Calculate cluster centroids and stds for normal behavior
                cluster_stats = {}
                for cluster_id in np.unique(normal_clusters):
                    if cluster_id != -1:  # Ignore noise
                        cluster_points = X_scaled[normal_mask][normal_clusters == cluster_id]
                        cluster_stats[cluster_id] = {
                            'centroid': np.mean(cluster_points, axis=0),
                            'std': np.std(cluster_points, axis=0)
                        }

                # Generate explanations for top anomalies
                high_anomaly_mask = scores > np.percentile(scores, 90)
                high_anomaly_indices = np.where(high_anomaly_mask)[0]

                for idx in high_anomaly_indices:
                    point = X_scaled[idx]
                    explanation = self._explain_point(point, cluster_stats, self.features)
                    original_idx = self.df.index[idx]
                    self.anomaly_explanations[original_idx] = explanation

            except Exception as e:
                self.log(f"⚠️ Could not generate detailed explanations: {e}")

    def _explain_point(self, point, cluster_stats, feature_names):
        """Explain why a specific point is anomalous by comparing to normal clusters"""
        if not cluster_stats:
            return "Anomaly detected based on overall distribution."

        # Find closest normal cluster
        min_distance = float('inf')
        closest_cluster = None

        for cluster_id, stats in cluster_stats.items():
            distance = np.linalg.norm(point - stats['centroid'])
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id

        if closest_cluster is None:
            return "No clear normal pattern found for comparison."

        # Compare point to closest normal cluster
        centroid = cluster_stats[closest_cluster]['centroid']
        std = cluster_stats[closest_cluster]['std']

        deviations = np.abs(point - centroid) / (std + 1e-8)  # Add small value to avoid division by zero

        # Find most deviant features
        top_deviations = np.argsort(deviations)[-3:][::-1]  # Top 3 most deviant
        top_features = []
        top_dev_values = []
        for i in top_deviations:
            if i < len(feature_names):
                top_features.append(feature_names[i])
                top_dev_values.append(deviations[i])

        explanation = f"This point is anomalous because it deviates from normal patterns in:\n"
        for feat, dev in zip(top_features, top_dev_values):
            explanation += f"• **{feat}**: {dev:.1f} standard deviations from normal\n"

        explanation += f"\nThe point is significantly different from the closest normal cluster of data points."

        return explanation


STATE = State()
# ---------------- Filters ----------------
def build_filters():
    """Build all filter controls including location and feature filters"""
    controls = []

    # Location filters (always available when we have coordinates)
    if STATE.lat and STATE.lat in STATE.df_raw.columns:
        lat_min = float(STATE.df_raw[STATE.lat].min())
        lat_max = float(STATE.df_raw[STATE.lat].max())
        lat_range.start = lat_min
        lat_range.end = lat_max
        lat_range.value = (lat_min, lat_max)
        lat_range.visible = True

    if STATE.lon and STATE.lon in STATE.df_raw.columns:
        lon_min = float(STATE.df_raw[STATE.lon].min())
        lon_max = float(STATE.df_raw[STATE.lon].max())
        lon_range.start = lon_min
        lon_range.end = lon_max
        lon_range.value = (lon_min, lon_max)
        lon_range.visible = True

    # Score filter (for ML anomaly scores)
    score_slider.visible = False  # Will be enabled after anomaly detection
    # Feature filters
    for c in STATE.features:
        if c in STATE.df_raw.columns:
            s = STATE.df_raw[c]
            lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
            step = (hi - lo) / 200 if hi > lo else 0.001
            controls.append(pn.widgets.RangeSlider(name=c, start=lo, end=hi, value=(lo, hi), step=step))
    filters_box.objects = [
        pn.pane.Markdown("**Location Filters**"),
        lat_range,
        lon_range,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("**Anomaly Score Filter**"),
        score_slider,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("**Feature Filters**")
    ] + controls + [apply_filters_btn, reset_filters_btn]


def apply_filters(_=None):
    """Apply all active filters to the data"""
    df = STATE.df_raw.copy()

    # Apply location filters
    if lat_range.visible and STATE.lat in df.columns:
        lat_min, lat_max = lat_range.value
        df = df[(df[STATE.lat] >= lat_min) & (df[STATE.lat] <= lat_max)]

    if lon_range.visible and STATE.lon in df.columns:
        lon_min, lon_max = lon_range.value
        df = df[(df[STATE.lon] >= lon_min) & (df[STATE.lon] <= lon_max)]

    # Apply anomaly score filter (from ML detection)
    if 'anomaly_score' in df.columns:
        score_min, score_max = score_slider.value
        df = df[(df['anomaly_score'] >= score_min) & (df['anomaly_score'] <= score_max)]

    # Apply feature filters
    for w in filters_box.objects:
        if isinstance(w, pn.widgets.RangeSlider) and w.name in df.columns and w.name not in [STATE.lat, STATE.lon]:
            lo, hi = w.value
            df = df[(df[w.name] >= lo) & (df[w.name] <= hi)]

    STATE.df = df.reset_index(drop=True)

    # Update anomaly counts
    total_anomalies = len(anomalies_df())
    status_msg.object = f"🔍 Filtered to {len(STATE.df)} rows ({total_anomalies} anomalies)"
    refresh_views()


def reset_filters(_=None):
    """Reset all filters to their original ranges"""
    if STATE.lat and STATE.lat in STATE.df_raw.columns:
        lat_min = float(STATE.df_raw[STATE.lat].min())
        lat_max = float(STATE.df_raw[STATE.lat].max())
        lat_range.value = (lat_min, lat_max)

    if STATE.lon and STATE.lon in STATE.df_raw.columns:
        lon_min = float(STATE.df_raw[STATE.lon].min())
        lon_max = float(STATE.df_raw[STATE.lon].max())
        lon_range.value = (lon_min, lon_max)

    # Reset anomaly score filter
    if 'anomaly_score' in STATE.df_raw.columns:
        score_min = float(STATE.df_raw['anomaly_score'].min())
        score_max = float(STATE.df_raw['anomaly_score'].max())
        score_slider.value = (score_min, score_max)

    # Reset feature filters
    for w in filters_box.objects:
        if isinstance(w, pn.widgets.RangeSlider) and w.name in STATE.df_raw.columns and w.name not in [STATE.lat, STATE.lon]:
            lo = float(STATE.df_raw[w.name].min())
            hi = float(STATE.df_raw[w.name].max())
            w.value = (lo, hi)

    # Reapply with reset values
    apply_filters()


# ---------------- Views ----------------
def anomalies_df():
    """Get dataframe of only anomalies from current filtered data"""
    if STATE.df.empty or 'is_anomaly_pred' not in STATE.df.columns:
        return pd.DataFrame()
    return STATE.df[STATE.df['is_anomaly_pred'] == 1].reset_index(drop=False)


def update_map():
    """Update the map view with current filtered data"""
    if STATE.df.empty or not all([STATE.lat, STATE.lon]):
        map_pane.object = hv.Curve([])
        map_html.visible = False
        return

    df = STATE.df
    lat, lon = STATE.lat, STATE.lon

    # Check if required columns exist
    if lat not in df.columns or lon not in df.columns:
        map_pane.object = hv.Curve([])
        map_html.visible = False
        return

    # Use ML anomaly predictions if available, otherwise use original labels
    if 'is_anomaly_pred' in df.columns:
        normal = df[df['is_anomaly_pred'] == 0]
        anom = df[df['is_anomaly_pred'] == 1]
        title_suffix = " (ML Detection)"
    elif 'is_anomaly' in df.columns:
        normal = df[df['is_anomaly'] == 0]
        anom = df[df['is_anomaly'] == 1]
        title_suffix = " (Original Labels)"
    else:
        normal = df
        anom = pd.DataFrame()
        title_suffix = ""

    gv_success = False
    if HAS_GV and len(df) > 0:
        try:
            tiles = gv.tile_sources.OSM
            pts_n = normal.hvplot.points(x=lon, y=lat, geo=True, tiles=tiles, size=5, alpha=0.35,
                                         color="#1f77b4", xlabel='Longitude', ylabel='Latitude',
                                         hover_cols=[c for c in df.columns if c not in [lat, lon]],
                                         title=f"Anomaly Detection{title_suffix}", tools=['hover'])
            if len(anom) > 0:
                pts_a = anom.hvplot.points(x=lon, y=lat, geo=True, tiles=None, size=6, alpha=0.85,
                                           color="#d62728", xlabel='Longitude', ylabel='Latitude',
                                           hover_cols=[c for c in df.columns if c not in [lat, lon]], tools=['hover'])
                map_pane.object = (pts_n * pts_a).opts(active_tools=['wheel_zoom'])
            else:
                map_pane.object = pts_n.opts(active_tools=['wheel_zoom'])
            map_pane.visible = True
            map_html.visible = False
            gv_success = True
        except Exception as e:
            status_msg.object += f"\n⚠️ GeoViews map error: {e}"

    folium_success = False
    if not gv_success and HAS_FOLIUM and len(df) > 0:
        try:
            # Use mean of coordinates for center, fallback to default
            try:
                center_lat = df[lat].mean()
                center_lon = df[lon].mean()
            except:
                center_lat, center_lon = 48.39, -4.486

            m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap",
                           width='100%', height='100%')
            for _, r in normal.iterrows():
                folium.CircleMarker([r[lat], r[lon]], radius=2, color="#1f77b4", fill=True, fill_opacity=0.6).add_to(m)
            if len(anom) > 0:
                for _, r in anom.iterrows():
                    folium.CircleMarker([r[lat], r[lon]], radius=3, color="#d62728", fill=True, fill_opacity=0.9).add_to(m)
            map_html.object = m._repr_html_()
            map_html.visible = True
            map_pane.visible = False
            folium_success = True
        except Exception as e:
            status_msg.object += f"\n⚠️ Folium map error: {e}"

    if not gv_success and not folium_success and len(df) > 0:
        try:
            pts_n = normal.hvplot.points(x=lon, y=lat, size=5, alpha=0.35,
                                         color="#1f77b4", xlabel='Longitude', ylabel='Latitude',
                                         hover_cols=[c for c in df.columns if c not in [lat, lon]],
                                         title=f"Anomaly Detection{title_suffix} (Cartesian fallback)", tools=['hover'])
            if len(anom) > 0:
                pts_a = anom.hvplot.points(x=lon, y=lat, size=6, alpha=0.85,
                                           color="#d62728", xlabel='Longitude', ylabel='Latitude',
                                           hover_cols=[c for c in df.columns if c not in [lat, lon]], tools=['hover'])
                map_pane.object = pts_n * pts_a
            else:
                map_pane.object = pts_n
            map_pane.visible = True
            map_html.visible = False
            status_msg.object += "\n📊 Using Cartesian plot fallback. For geographic map, install 'geoviews' or 'folium'."
        except Exception as e:
            status_msg.object += f"\n⚠️ Cartesian plot error: {e}"
            map_pane.visible = False
            map_html.visible = False
    elif not gv_success and not folium_success:
        status_msg.object += "\n⚠️ Neither geoviews nor folium is available; map disabled."
        map_pane.visible = False
        map_html.visible = False


def update_table():
    """Update the table view with current filtered data"""
    table_pane.value = STATE.df


def update_stepper():
    """Update the anomaly navigation slider"""
    adf = anomalies_df()
    if adf.empty:
        idx_slider.start = idx_slider.end = 0
        idx_slider.value = 0
        confirm_btn.disabled = True
        reject_btn.disabled = True
    else:
        idx_slider.start = 0
        idx_slider.end = len(adf) - 1
        idx_slider.value = min(idx_slider.value, idx_slider.end)
        confirm_btn.disabled = False
        reject_btn.disabled = False


def update_details(_=None):
    """Update the anomaly details view"""
    adf = anomalies_df()
    if adf.empty:
        detail_pane.objects = [pn.pane.Markdown("_No anomalies in current view._")]
        explanation_pane.object = "### Anomaly Explanation\n\nNo anomalies to explain."
        return

    row = adf.iloc[idx_slider.value]
    orig_idx = int(row['index'])
    info = {
        "Row index (original)": orig_idx,
        "Latitude": f"{row[STATE.lat]:.6f}",
        "Longitude": f"{row[STATE.lon]:.6f}",
    }

    # Add anomaly score if available
    if 'anomaly_score' in row:
        info["Anomaly Score"] = f"{row['anomaly_score']:.6f}"
        info["Threshold"] = f"{threshold_slider.value:.3f}"

    # Add feature values
    for c in STATE.features:
        if c in adf.columns and pd.api.types.is_numeric_dtype(adf[c]):
            info[c] = f"{row[c]:.6f}"

    verdict = STATE.feedback.get(orig_idx, "—")
    feedback_note.object = f"**Feedback for this point:** {verdict}"

    rows = ''.join(f"<tr><td><b>{k}</b></td><td style='padding-left:10px'>{v}</td></tr>" for k, v in info.items())
    table_html = pn.pane.HTML(f"<table>{rows}</table>", styles={'font-size': '12.5px'})

    # Show current anomaly position in slider
    slider_info = pn.pane.Markdown(f"**Anomaly {idx_slider.value + 1} of {len(adf)}**")

    # Get explanation for this anomaly
    explanation = STATE.anomaly_explanations.get(orig_idx,
                                                 "Explanation not available. This point was detected as anomalous based on its overall deviation from normal patterns.")
    explanation_pane.object = f"### Anomaly Explanation\n\n{explanation}"

    detail_pane.objects = [
        table_html,
        slider_info,
        pn.Row(prev_btn, idx_slider, next_btn),
        pn.Row(confirm_btn, reject_btn),
        feedback_note
    ]


def refresh_views():
    """Refresh all views simultaneously"""
    update_map()
    update_table()
    update_stepper()
    update_details()


# ---------------- Feedback & download ----------------
def confirm(_=None):
    adf = anomalies_df()
    if adf.empty:
        return
    orig_idx = int(adf.iloc[idx_slider.value]['index'])
    STATE.feedback[orig_idx] = "confirmed"
    download_btn.disabled = False
    update_details()
    status_msg.object = f"✅ Anomaly {orig_idx} confirmed"


def reject(_=None):
    adf = anomalies_df()
    if adf.empty:
        return
    orig_idx = int(adf.iloc[idx_slider.value]['index'])
    STATE.feedback[orig_idx] = "rejected"
    download_btn.disabled = False
    update_details()
    status_msg.object = f"❌ Anomaly {orig_idx} rejected"


def build_confirmed():
    if not STATE.feedback:
        return io.BytesIO(b"")
    df = STATE.df_raw.reset_index(drop=False).set_index('index')
    df['feedback'] = ""
    for i, v in STATE.feedback.items():
        if i in df.index:
            df.loc[i, 'feedback'] = v
    out = df[df['feedback'] == 'confirmed'].reset_index()
    name = f"confirmed_anomalies_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"
    download_btn.filename = name
    return io.BytesIO(out.to_csv(index=False).encode('utf-8'))


def on_threshold_change(event):
    """Re-detect anomalies when threshold changes"""
    if STATE.is_trained:
        status_msg.object = f"🔄 Threshold changed to {event.new}, re-detecting anomalies..."
        STATE.detect_anomalies()


download_btn.callback = build_confirmed
# ---------------- Callbacks ----------------
def on_file_change(event):
    if event.new:
        status_msg.object = "⏳ File received. Parsing…"
        ok = STATE.load_csv(event.new)
        if ok and not all([STATE.lat, STATE.lon]):
            status_msg.object += "\n➡️ Map your columns and click **Apply column mapping**."


def manual_load(_):
    # Load from the file chooser if present; else from DEFAULT_CSV
    if file_input.value:
        status_msg.object = "⏳ Loading file from chooser…"
        STATE.load_csv(file_input.value)
    else:
        if os.path.exists(DEFAULT_CSV):
            status_msg.object = f"⏳ Loading default CSV: {DEFAULT_CSV}"
            with open(DEFAULT_CSV, "rb") as f:
                STATE.load_csv(f.read())
        else:
            status_msg.object = f"❌ Default CSV not found: {DEFAULT_CSV}"


def on_apply_mapping(_=None):
    STATE.apply_mapping()


def on_train_model(_):
    STATE.train_model()


def on_detect_anomalies(_):
    STATE.detect_anomalies()


# Connect all callbacks
file_input.param.watch(on_file_change, 'value')
load_btn.on_click(manual_load)
apply_mapping_btn.on_click(on_apply_mapping)
train_btn.on_click(on_train_model)
detect_btn.on_click(on_detect_anomalies)
apply_filters_btn.on_click(apply_filters)
reset_filters_btn.on_click(reset_filters)
threshold_slider.param.watch(on_threshold_change, 'value')
prev_btn.on_click(lambda e: setattr(idx_slider, 'value', max(idx_slider.start, idx_slider.value - 1)))
next_btn.on_click(lambda e: setattr(idx_slider, 'value', min(idx_slider.end, idx_slider.value + 1)))
confirm_btn.on_click(confirm)
reject_btn.on_click(reject)
idx_slider.param.watch(update_details, 'value')
# ---------------- Layout ----------------
mapper = pn.Card(
    pn.Column(
        pn.pane.Markdown("### Column Mapping"),
        lat_sel, lon_sel,
        pn.pane.Markdown("**Select features for ML anomaly detection:**"),
        feature_sels,
        apply_mapping_btn,
    ),
    title="Step 1 — Map columns", collapsed=False
)
model_card = pn.Card(
    pn.Column(
        pn.pane.Markdown("### ML Model Configuration"),
        model_sel,
        pn.Row(if_contamination, pn.pane.Markdown("(Isolation Forest contamination)")),
        pn.Row(ocsvm_nu, pn.pane.Markdown("(OCSVM nu parameter)")),
        threshold_slider,
        pn.Row(train_btn, detect_btn),
    ),
    title="Step 2 — Configure & Train ML Model", collapsed=False
)
filters_card = pn.Card(filters_box, title="Step 3 — Filter data", collapsed=False)
sidebar = pn.Column(
    "### Data",
    file_input,
    load_btn,
    status_msg,
    mapper,
    model_card,
    filters_card,
    sizing_mode="stretch_width",
    width=400,
)
tabs = pn.Tabs(
    ("Map", pn.Column(map_pane, map_html)),
    ("Table", table_pane),
    ("Details", pn.Column(detail_pane, explanation_pane)),
    dynamic=True,
)
main = pn.Column(
    "## ML-Powered Anomaly Detection Dashboard for Vessel Tracks",
    tabs,
    pn.Row(download_btn),
    sizing_mode="stretch_both"
)
# Set template parameters
pn.state.template.param.update(
    title="ML-Powered Anomaly Detection Dashboard for Vessel Tracks",
    header_background="#1b3a57"
)
# ---- Optional: autoload default CSV on startup ----
if os.path.exists(DEFAULT_CSV):
    status_msg.object = "🔄 Auto-loading default CSV..."
    with open(DEFAULT_CSV, "rb") as f:
        STATE.load_csv(f.read())
else:
    status_msg.object = f"📁 Please upload a CSV file or ensure default exists at: {DEFAULT_CSV}"
# Mark as servable
sidebar.servable(target='sidebar')
main.servable(target='main')