"""Simulation study: repeated interval forecasts of a single outcome from growing partial trajectories.

Synthetic runners cover 42.195 km in M equal segments. The pace of runner i on segment j is
    p_ij = mu_i * exp(drift_j + e_ij) * (1 + delta_i * 1{collapse_i, km_j >= onset_i}),
where mu_i is a log-normal baseline pace, drift_j a common positive drift, e_ij a stationary AR(1)
process with correlation rho, and the collapse ("wall") happens with a probability that increases with
mu_i. The outcome is the finish time y_i. Forecasts are issued at K equally spaced checkpoints.

Scenarios
  S1  exchangeable data: per-checkpoint vs Bonferroni vs joint bands, for K in {3, 5, 9, 17} and rho in {0, 0.5, 0.9}
  S2  goal-time alerts under S1 with K = 9, rho = 0.5
  S3  shift between calibration and test populations: covariate shift (faster field, same conditional law)
      and conditional shift (higher collapse probability and larger slowdown, as on a hot day)

Usage: python experiments/run_simulation.py [--reps 10] [--out results/simulation]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtc.conformal import RankMaxConformal, TrajectoryConformal  # noqa: E402

KM = 42.195
M = 17  # number of segments; checkpoints are subsets of the segment ends
ALPHA = 0.1
LGB = dict(n_estimators=150, learning_rate=0.08, num_leaves=31, min_child_samples=50,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=2)


def simulate(n, rng, rho=0.5, sigma=0.04, mu_shift=0.0, collapse_scale=1.0, delta_scale=1.0):
    """Return segment times (n, M) in minutes and finish times (n,)."""
    seg_km = np.full(M, KM / M)
    ends = np.cumsum(seg_km)
    mu = np.exp(rng.normal(np.log(5.6) + mu_shift, 0.17, n))          # min/km, median ~ 3 h 56
    drift = np.linspace(0.0, 0.05, M)                                   # gradual slowing
    e = np.empty((n, M))
    e[:, 0] = rng.normal(0, sigma, n)
    for j in range(1, M):
        e[:, j] = rho * e[:, j - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma, n)
    z = (np.log(mu) - np.log(5.6)) / 0.17
    p_col = np.clip(collapse_scale / (1 + np.exp(-(z * 0.8 - 1.2))), 0, 0.95)   # ~0.25 overall, higher if slow
    col = rng.random(n) < p_col
    onset = rng.uniform(25, 38, n)
    delta = rng.uniform(0.08, 0.35, n) * delta_scale
    mult = 1 + delta[:, None] * (col[:, None] & (ends[None, :] > onset[:, None]))
    pace = mu[:, None] * np.exp(drift[None, :] + e) * mult
    seg = pace * seg_km[None, :]
    return seg, seg.sum(1), col


def checkpoints(K):
    """Segment counts at which forecasts are issued: K roughly equally spaced among the M-1 interior ends."""
    return np.unique(np.round(np.linspace(1, M - 1, K)).astype(int))


def features(seg, n):
    km = KM / M
    pace = seg[:, :n] / km
    cum = seg[:, :n].sum(1) / (n * km)
    last = pace[:, -1]
    acc = pace[:, -1] - (pace[:, -2] if n > 1 else pace[:, -1])
    X = np.column_stack([pace, cum, last, acc, pace.var(1)])
    proj = cum * KM
    return X.astype(np.float32), proj


def fit_predict(seg_tr, y_tr, others, cps):
    import lightgbm as lgb
    out = {k: (np.zeros((len(v), len(cps))), np.zeros((len(v), len(cps)))) for k, v in others.items()}
    width = np.zeros(len(cps))
    for j, n in enumerate(cps):
        X, proj = features(seg_tr, n)
        r = y_tr - proj
        mL = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA / 2, **LGB).fit(X, r)
        mH = lgb.LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2, **LGB).fit(X, r)
        width[j] = np.median(mH.predict(X[:5000]) - mL.predict(X[:5000]))
        for name, s in others.items():
            Xo, po = features(s, n)
            out[name][0][:, j] = mL.predict(Xo) + po
            out[name][1][:, j] = mH.predict(Xo) + po
    return out, width


def evaluate(lo, hi, y):
    inside = (lo <= y[:, None]) & (y[:, None] <= hi)
    return dict(marginal=inside.mean(), worst=inside.mean(0).min(), joint=inside.all(1).mean(),
                width=(hi - lo).mean())


def calibrate_all(P, width, y_cal, rng):
    """Return {method: (lo, hi)} for the test set."""
    Lc, Hc = P["cal"]
    Lt, Ht = P["test"]
    res = {}
    for name, mode in [("Per-checkpoint", "per_checkpoint"), ("Bonferroni", "bonferroni")]:
        res[name] = TrajectoryConformal(ALPHA, mode).fit(Lc, Hc, y_cal).predict(Lt, Ht)
    sc = np.broadcast_to(width, Lc.shape)
    tc = TrajectoryConformal(ALPHA, "simultaneous").fit(Lc, Hc, y_cal, scale=sc)
    res["Constant-scale band"] = tc.predict(Lt, Ht, scale=np.broadcast_to(width, Lt.shape))
    idx = rng.permutation(len(y_cal))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    rm = RankMaxConformal(ALPHA, nested=True).fit(Lc[a], Hc[a], y_cal[a], Lc[b], Hc[b], y_cal[b])
    res["Rank-normalized band, nested"] = rm.predict(Lt, Ht)
    return res


def alerts(lo, y, proj5, early, goals=(180, 210, 240, 270, 300)):
    """False-alarm rate among goal reachers and detection among misses (by 30 km and overall), per runner-goal pair."""
    fa, det, det30, n_reach, n_miss = 0, 0, 0, 0, 0
    for g in goals:
        m = np.abs(proj5 - g) <= 15
        if not m.any():
            continue
        alert = (lo[m] > g).any(1)
        reach = y[m] <= g
        fa += (alert & reach).sum(); n_reach += reach.sum()
        det += (alert & ~reach).sum(); n_miss += (~reach).sum()
        det30 += ((lo[m][:, early] > g).any(1) & ~reach).sum()
    return dict(pairs=int(m_tot(goals, proj5)), reached=int(n_reach), false_alarm=fa / max(n_reach, 1),
                detected_30km=det30 / max(n_miss, 1), detected=det / max(n_miss, 1))


def m_tot(goals, proj5):
    return sum((np.abs(proj5 - g) <= 15).sum() for g in goals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--ntrain", type=int, default=15000)
    ap.add_argument("--ncal", type=int, default=10000)
    ap.add_argument("--ntest", type=int, default=20000)
    ap.add_argument("--out", default="results/simulation")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, alert_rows, shift_rows = [], [], []
    for rep in range(a.reps):
        rng = np.random.default_rng(1000 + rep)
        # S1 and S2
        for rho in (0.0, 0.5, 0.9):
            seg_tr, y_tr, _ = simulate(a.ntrain, rng, rho=rho)
            seg_ca, y_ca, _ = simulate(a.ncal, rng, rho=rho)
            seg_te, y_te, _ = simulate(a.ntest, rng, rho=rho)
            for K in (3, 5, 9, 17):
                cps = checkpoints(K) if K < 17 else np.arange(1, M)
                P, w = fit_predict(seg_tr, y_tr, {"cal": seg_ca, "test": seg_te}, cps)
                for meth, (lo, hi) in calibrate_all(P, w, y_ca, rng).items():
                    rows.append(dict(rep=rep, rho=rho, K=len(cps), method=meth, **evaluate(lo, hi, y_te)))
                    if K == 9 and rho == 0.5:
                        _, p5 = features(seg_te, 2)  # projection at 4.96 km
                        early = np.asarray(cps) * KM / M <= 30.0
                        alert_rows.append(dict(rep=rep, method=meth, **alerts(lo, y_te, p5, early)))
        # S3: shift, K = 9, rho = 0.5; train and calibrate on the base population
        seg_tr, y_tr, _ = simulate(a.ntrain, rng)
        seg_ca, y_ca, _ = simulate(a.ncal, rng)
        tests = {"No shift": simulate(a.ntest, rng),
                 "Covariate (faster field)": simulate(a.ntest, rng, mu_shift=-0.12),
                 "Conditional (hot day)": simulate(a.ntest, rng, collapse_scale=1.8, delta_scale=1.5)}
        cps = checkpoints(9)
        P, w = fit_predict(seg_tr, y_tr, {"cal": seg_ca, **{k: v[0] for k, v in tests.items()}}, cps)
        for shift, (seg_te, y_te, _) in tests.items():
            Q = {"cal": P["cal"], "test": P[shift]}
            for meth, (lo, hi) in calibrate_all(Q, w, y_ca, rng).items():
                shift_rows.append(dict(rep=rep, shift=shift, method=meth, **evaluate(lo, hi, y_te)))
        print(f"rep {rep} done", flush=True)
    pd.DataFrame(rows).to_csv(out / "sim_joint.csv", index=False)
    pd.DataFrame(alert_rows).to_csv(out / "sim_alerts.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(out / "sim_shift.csv", index=False)


if __name__ == "__main__":
    main()
