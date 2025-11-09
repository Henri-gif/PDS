from __future__ import annotations
import numpy as np

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
EARTH_M = 6371008.8

def haversine_m(lat1, lon1, lat2, lon2):
    lat1r = lat1 * DEG2RAD
    lon1r = lon1 * DEG2RAD
    lat2r = lat2 * DEG2RAD
    lon2r = lon2 * DEG2RAD
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return EARTH_M * c

def cartesian_dist(dx, dy):
    return np.hypot(dx, dy)

def geographic_bearing_deg(lat1, lon1, lat2, lon2):
    lat1r = lat1 * DEG2RAD
    lat2r = lat2 * DEG2RAD
    dlon = (lon2 - lon1) * DEG2RAD
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r)*np.sin(lat2r) - np.sin(lat1r)*np.cos(lat2r)*np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    brng = (brng + 360.0) % 360.0
    return brng

def wrap_angle_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)
