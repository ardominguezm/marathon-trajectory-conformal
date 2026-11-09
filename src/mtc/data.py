"""Data loading and feature construction.

Expected files (not tracked by git, see data/README.md):
  * segment_paces_data.csv         Boston, Chicago, New York 2021-2024 (N = 396,813)
  * berlin_marathon_2021_2024.csv  raw Berlin splits in cumulative seconds (N_raw = 137,847)
  * wall_labels_data.csv           optional, persistent-deterioration labels for the 3 development events

Cleaning follows the rules of Dominguez-Monterroza (2026), Physical Review E, doi:10.1103/1wch-rv2h:
all segment paces in (0, 20) min/km and finish time in (120, 480) min.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINTS = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
SEG_NAMES = ["0_5k", "5_10k", "10_15k", "15_20k", "20k_half",
             "half_25k", "25_30k", "30_35k", "35_40k", "40k_finish"]
SEG_COLS = [f"seg_time_{s}" for s in SEG_NAMES]
SEG_KM = np.array([5, 5, 5, 5, 1.0975, 3.9025, 5, 5, 5, 2.195])
CKPT_KM = np.cumsum(SEG_KM)[:-1]  # distance of each of the 9 checkpoints
MARATHON_KM = 42.195
DEV_EVENTS = ["Boston", "Chicago", "NewYork"]
ALL_EVENTS = DEV_EVENTS + ["Berlin"]
GROUP_EDGES = (214.2, 249.72, 297.31)  # finish-time cut points of the stored performance quartiles
GROUP_LABELS = ["Q1_fastest", "Q2", "Q3", "Q4_slowest"]
META_COLS = ["runner_id", "Year", "City", "male", "Age", "finish_time_min", "performance_group"]


def _group_from_finish(finish_min: np.ndarray) -> np.ndarray:
    return np.array(GROUP_LABELS)[np.digitize(finish_min, GROUP_EDGES)]


def load_development(path: str | Path) -> pd.DataFrame:
    cols = ["runner_id", "Year", "City", "Sex", "Age", "finish_time_min", "performance_group"] + SEG_COLS
    d = pd.read_csv(path, usecols=cols)
    d["Year"] = d["Year"].astype(int)
    d["male"] = (d["Sex"] == "M").astype(np.int8)
    d["Age"] = d["Age"].fillna(d["Age"].median())
    return d[META_COLS + SEG_COLS].reset_index(drop=True)


def load_berlin(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cum_cols = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K", "Finish Net"]
    cum = raw[cum_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    seg_sec = np.diff(np.c_[np.zeros(len(raw)), cum], axis=1)
    seg_min = seg_sec / 60.0
    pace = seg_min / SEG_KM
    finish = cum[:, -1] / 60.0
    keep = (np.isfinite(pace).all(1) & (pace > 0).all(1) & (pace < 20).all(1)
            & np.isfinite(finish) & (finish > 120) & (finish < 480))
    out = pd.DataFrame(seg_min[keep], columns=SEG_COLS)
    out.insert(0, "runner_id", ["BER_%d" % i for i in np.flatnonzero(keep)])
    out["Year"] = pd.to_numeric(raw["Year"], errors="coerce").to_numpy()[keep].astype(int)
    out["City"] = "Berlin"
    out["male"] = (raw["M/F"].astype(str).str.upper().to_numpy()[keep] == "M").astype(np.int8)
    age = pd.to_numeric(raw["Age"], errors="coerce")
    out["Age"] = age.fillna(age.median()).to_numpy()[keep]
    out["finish_time_min"] = finish[keep]
    out["performance_group"] = _group_from_finish(out["finish_time_min"].to_numpy())
    return out[META_COLS + SEG_COLS].reset_index(drop=True)


def load_all(dev_path, berlin_path=None) -> pd.DataFrame:
    parts = [load_development(dev_path)]
    if berlin_path is not None:
        parts.append(load_berlin(berlin_path))
    return pd.concat(parts, ignore_index=True)


def load_wall_labels(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["runner_id", "wall_persistent", "wall_km_persistent"])


def checkpoint_features(df: pd.DataFrame, n_seg: int) -> pd.DataFrame:
    """Interpretable pacing features using only the first n_seg segments (n_seg = 1..9)."""
    seg = df[SEG_COLS[:n_seg]].to_numpy(float)
    pace = seg / SEG_KM[:n_seg]
    X = pd.DataFrame(pace, columns=[f"p{i}" for i in range(n_seg)], index=df.index)
    X["cum_pace"] = seg.sum(1) / SEG_KM[:n_seg].sum()
    X["last_pace"] = pace[:, -1]
    X["accel"] = pace[:, -1] - (pace[:, -2] if n_seg > 1 else pace[:, -1])
    X["pace_var"] = pace.var(1)
    X["age"] = df["Age"].to_numpy()
    X["male"] = df["male"].to_numpy()
    return X.astype(np.float32)


def naive_extrapolation(df: pd.DataFrame, n_seg: int) -> np.ndarray:
    """Traditional projection: current cumulative pace held constant to the finish (minutes)."""
    return df[SEG_COLS[:n_seg]].to_numpy(float).sum(1) / SEG_KM[:n_seg].sum() * MARATHON_KM


def make_splits(df: pd.DataFrame, protocol: str = "temporal", seed: int = 0,
                events=DEV_EVENTS) -> dict[str, pd.DataFrame]:
    """Train 2021-22. 'temporal': calibrate 2023, test 2024. 'exchangeable': pool 2023-24, split 50/50."""
    d = df[df.City.isin(events)]
    train = d[d.Year.isin([2021, 2022])]
    if protocol == "temporal":
        cal, test = d[d.Year == 2023], d[d.Year == 2024]
    elif protocol == "exchangeable":
        pool = d[d.Year.isin([2023, 2024])]
        idx = np.random.RandomState(seed).permutation(len(pool))
        half = len(pool) // 2
        cal, test = pool.iloc[idx[:half]], pool.iloc[idx[half:]]
    else:
        raise ValueError(protocol)
    return {"train": train.reset_index(drop=True), "cal": cal.reset_index(drop=True),
            "test": test.reset_index(drop=True)}
