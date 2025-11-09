from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def _to_datetime_utc(series: pd.Series) -> pd.Series:
    s = series
    if np.issubdtype(s.dtype, np.number):
        v = s.astype('float64')
        unit = 's'
        vmax = float(np.nanmax(v))
        if vmax > 1e12 or vmax > 1e10:
            unit = 'ms'
        dt = pd.to_datetime(v, unit=unit, utc=True, errors='coerce')
    else:
        dt = pd.to_datetime(s, utc=True, errors='coerce')
    return dt

def load_csv(path: str, cfg: Dict) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine='python')
    cols = (cfg or {}).get('data', {})

    mapping = {
        cols.get('time_col'): 'ts',
        cols.get('mmsi_col'): 'mmsi',
        cols.get('lat_col'): 'lat',
        cols.get('lon_col'): 'lon',
        cols.get('sog_col'): 'sog',
        cols.get('cog_col'): 'cog',
        cols.get('heading_col'): 'heading',
        cols.get('label_col'): 'label',
    }
    rename = {k: v for k, v in mapping.items() if k and k in df.columns}
    df = df.rename(columns=rename)

    for name in ['cog','heading','label']:
        if name not in df.columns:
            df[name] = np.nan

    if 'ts' in df.columns:
        try:
            df['ts'] = _to_datetime_utc(df['ts'])
        except Exception:
            pass

    for need in ['mmsi','lat','lon']:
        if need not in df.columns:
            raise ValueError(f'Missing required column after renaming: {need}')

    return df

def preprocess_tracks(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.dropna(subset=['mmsi','lat','lon']).copy()
    sort_cols = ['mmsi']
    if 'ts' in df.columns:
        sort_cols.append('ts')
    elif 't' in df.columns:
        sort_cols.append('t')
    df = df.sort_values(sort_cols).reset_index(drop=True)

    if 'ts' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['dt_s'] = df.groupby('mmsi')['ts'].diff().dt.total_seconds().fillna(0).clip(lower=0)
    elif 't' in df.columns:
        df['dt_s'] = df.groupby('mmsi')['t'].diff().astype('float64').fillna(0).clip(lower=0)
    else:
        df['dt_s'] = 0.0

    dcfg = (cfg or {}).get('data', {})
    max_gap_s = dcfg.get('max_gap_s', 3600)
    df['gap_flag'] = (df['dt_s'] > max_gap_s).astype(int)

    if 'sog' in df.columns:
        max_speed_kn = dcfg.get('max_speed_kn', 200)
        df.loc[df['sog'] > max_speed_kn, 'sog'] = max_speed_kn

    return df
