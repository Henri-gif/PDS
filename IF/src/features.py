from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict
from .utils import haversine_m, cartesian_dist, geographic_bearing_deg, wrap_angle_diff_deg

KNOTS_PER_MPS = 1.943844

def _infer_coord_system(lat, lon):
    if np.nanmin(lat) >= -90 and np.nanmax(lat) <= 90 and np.nanmin(lon) >= -180 and np.nanmax(lon) <= 180:
        return 'geographic'
    return 'cartesian'

def _compute_step_metrics(group: pd.DataFrame, coord_system: str) -> pd.DataFrame:
    g = group.copy()
    g['lat_prev'] = g['lat'].shift(1)
    g['lon_prev'] = g['lon'].shift(1)
    g['dt_s'] = g['dt_s'].fillna(0)

    if coord_system == 'geographic':
        g['dist_m'] = haversine_m(g['lat_prev'], g['lon_prev'], g['lat'], g['lon'])
        g['cog_obs'] = geographic_bearing_deg(g['lat_prev'], g['lon_prev'], g['lat'], g['lon'])
    else:
        dx = g['lon'] - g['lon_prev']
        dy = g['lat'] - g['lat_prev']
        g['dist_m'] = cartesian_dist(dx, dy)
        g['cog_obs'] = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0

    g.loc[g['dt_s'] <= 0, 'dist_m'] = 0.0
    g['speed_mps_obs'] = np.where(g['dt_s'] > 0, g['dist_m'] / g['dt_s'], 0.0)

    if 'sog' not in g.columns or g['sog'].isna().all():
        g['sog'] = g['speed_mps_obs'] * KNOTS_PER_MPS

    if 'cog' not in g.columns or g['cog'].isna().all():
        g['cog'] = g['cog_obs']

    g['dcog_abs'] = wrap_angle_diff_deg(g['cog'], g['cog'].shift(1))
    g['turn_rate'] = np.where(g['dt_s'] > 0, g['dcog_abs'] / g['dt_s'], 0.0)
    g['accel'] = np.where(g['dt_s'] > 0, (g['sog'] - g['sog'].shift(1)) / g['dt_s'], 0.0)
    g['speed_std_60s'] = g['sog'].rolling(window=5, min_periods=2).std().fillna(0.0)
    g['turn_rate_std_60s'] = g['turn_rate'].rolling(window=5, min_periods=2).std().fillna(0.0)

    # Deviation from straight line (local meters if geographic)
    lat_m = g['lat'].copy()
    lon_m = g['lon'].copy()
    if coord_system == 'geographic':
        lat0 = np.nanmedian(g['lat'])
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat0))
        lat_m = (g['lat'] - lat0) * m_per_deg_lat
        lon_m = (g['lon'] - np.nanmedian(g['lon'])) * m_per_deg_lon

    x = lon_m.values
    y = lat_m.values
    x_prev = pd.Series(x).shift(1).values
    y_prev = pd.Series(y).shift(1).values
    x_next = pd.Series(x).shift(-1).values
    y_next = pd.Series(y).shift(-1).values

    num = np.abs((y_next - y_prev) * x - (x_next - x_prev) * y + x_next*y_prev - y_next*x_prev)
    den = np.hypot(y_next - y_prev, x_next - x_prev)
    dev = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den>0)
    g['dev_from_line_m'] = pd.Series(dev, index=g.index).fillna(0.0)

    g['stop_flag'] = (g['sog'] < 0.5).astype(int)

    num_cols = g.select_dtypes(include=[np.number]).columns
    g[num_cols] = g[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return g

def compute_features(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.copy()
    coord_system = _infer_coord_system(df['lat'].values, df['lon'].values)
    feats = []
    for _, grp in df.groupby('mmsi', sort=False):
        feats.append(_compute_step_metrics(grp, coord_system))
    out = pd.concat(feats, axis=0).sort_index()
    return out

def make_feature_matrix(df: pd.DataFrame, cfg: Dict):
    cols = cfg.get('features', {}).get('use_columns', [])
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[cols].astype(float).values
    return X, cols
