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

    if 'ts' not in df.columns:
        raise ValueError('Timestamp column not found after renaming. Check data.time_col in config.')
    if 'mmsi' not in df.columns:
        raise ValueError('MMSI/AgentID column not found after renaming. Check data.mmsi_col in config.')
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError('Latitude/Longitude columns not found after renaming. Check data.lat_col/lon_col in config.')

    df['ts'] = _to_datetime_utc(df['ts'])
    return df

def preprocess_tracks(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    df = df.dropna(subset=['mmsi','ts','lat','lon']).copy()
    df = df.sort_values(['mmsi','ts']).reset_index(drop=True)
    df['dt_s'] = df.groupby('mmsi')['ts'].diff().dt.total_seconds().fillna(0).clip(lower=0)
    dcfg = (cfg or {}).get('data', {})
    max_gap_s = dcfg.get('max_gap_s', 1800)
    df['gap_flag'] = (df['dt_s'] > max_gap_s).astype(int)
    if 'sog' in df.columns:
        max_speed_kn = dcfg.get('max_speed_kn', 200)
        df.loc[df['sog'] > max_speed_kn, 'sog'] = max_speed_kn
    return df
