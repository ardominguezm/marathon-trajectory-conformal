from __future__ import annotations

import numpy as np
import pandas as pd


def interval_score(lo, hi, y, alpha: float):
    """Winkler/interval score (lower is better): width plus (2/alpha) times the distance of misses."""
    lo, hi, y = np.asarray(lo, float), np.asarray(hi, float), np.asarray(y, float)
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def summarize_intervals(lo, hi, y, alpha: float) -> dict:
    """Per-checkpoint and joint summaries for (n, K) interval arrays against a common target y (n,)."""
    y = np.asarray(y, float)[:, None]
    cov = (y >= lo) & (y <= hi)
    w = hi - lo
    return {
        "coverage_mean": float(cov.mean()),
        "coverage_min_ckpt": float(cov.mean(0).min()),
        "coverage_max_ckpt": float(cov.mean(0).max()),
        "joint_coverage": float(cov.all(1).mean()),
        "width_mean": float(w.mean()),
        "interval_score_mean": float(interval_score(lo, hi, np.broadcast_to(y, lo.shape), alpha).mean()),
    }


def slice_coverage(lo, hi, y, labels) -> pd.DataFrame:
    """Coverage of 1-D intervals within slices (labels array of same length)."""
    y, lab = np.asarray(y, float), np.asarray(labels)
    c = (y >= lo) & (y <= hi)
    return (pd.DataFrame({"slice": lab, "covered": c, "width": hi - lo})
            .groupby("slice").agg(n=("covered", "size"), coverage=("covered", "mean"), width=("width", "mean"))
            .reset_index())
