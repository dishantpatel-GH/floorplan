"""Measured LiDAR noise model (D-011).

sigma_lidar(r) returns the 1-sigma noise (metres) of ONE high-confidence depth sample at range r, interpolated from the
table measured by scripts/noise_model.py (multi-view re-projection on the sample captures). It is used to:
  * weight points in plane and line fits (weight = 1 / sigma^2): near points count more than far ones;
  * set the sensor term of every interval.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_TABLE = Path(__file__).with_name("lidar_noise.json")


@lru_cache(maxsize=1)
def _table():
    t = json.loads(_TABLE.read_text())["table"]
    r = np.array([row["r_mid"] for row in t])
    s = np.array([row["sigma_mm"] for row in t]) / 1000.0
    s = np.maximum.accumulate(s)        # noise never decreases with range (monotone envelope)
    return r, s


def sigma_lidar(r) -> np.ndarray:
    r_tab, s_tab = _table()
    return np.interp(np.asarray(r, float), r_tab, s_tab, left=s_tab[0], right=s_tab[-1])


def weights(r) -> np.ndarray:
    return 1.0 / sigma_lidar(r) ** 2
