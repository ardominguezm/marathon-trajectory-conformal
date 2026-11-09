"""Decision-level evaluation (goal-time alerts), efficiency of simultaneous bands, and example trajectories.

Temporal protocol on Boston, Chicago, New York: train 2021-22, calibrate 2023, test 2024.
Bands: per-checkpoint CQR, Bonferroni, constant-scale simultaneous band, rank-max simultaneous band (2023 split in a
tuning half and a calibration half), and nested versions.

Goal-time alert. For a goal g, an alert "goal no longer reachable" fires at checkpoint k when the lower limit exceeds g.
Pairs (runner, g) are kept when the runner's 5K constant-pace projection is within 15 min of g, for g in
{180, 210, 240, 270, 300} min. False alarm: an alert fires although the runner finished within g. For runners who missed
g we report the share alerted by 40K and the median distance of the first alert. A practitioner heuristic that requires
two consecutive alerts on per-checkpoint intervals is included.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from common import base_parser, ROOT  # noqa
from mtc.conformal import RankMaxConformal, TrajectoryConformal
from mtc.data import CHECKPOINTS, CKPT_KM, load_development, make_splits, naive_extrapolation
from mtc.metrics import summarize_intervals
from mtc.models import quantile_trajectories

p = base_parser(__doc__)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
seeds = 1 if a.quick else a.seeds
dev = load_development(a.dev)
if a.quick:
    dev = dev.sample(frac=0.15, random_state=0)
GOALS = np.array([180, 210, 240, 270, 300.0])
summ, dec, ex = [], [], []


def first_alert(lo, g):
    """Index of first checkpoint with lo > g, or -1."""
    hit = lo > g[:, None]
    idx = np.where(hit.any(1), hit.argmax(1), -1)
    return idx


def first_alert_consecutive(lo, g):
    hit = lo > g[:, None]
    two = hit[:, 1:] & hit[:, :-1]
    return np.where(two.any(1), two.argmax(1) + 1, -1)


for seed in range(seeds):
    sp = make_splits(dev, "temporal", seed)
    cal, test = sp["cal"], sp["test"]
    res, width = quantile_trajectories(sp["train"], {"cal": cal, "test": test}, list(range(1, 10)), a.alpha, seed,
                                       20000 if a.quick else a.max_train,
                                       {"n_estimators": 60 if a.quick else a.n_estimators}, target=a.target)
    (Lc, Hc), (Lt, Ht) = res["cal"], res["test"]
    yc, yt = cal.finish_time_min.to_numpy(), test.finish_time_min.to_numpy()
    half = np.random.RandomState(seed).permutation(len(yc)); A, B = half[: len(yc) // 2], half[len(yc) // 2:]
    sc = lambda n: np.tile(width, (n, 1))
    bands = {}
    bands["per_checkpoint"] = TrajectoryConformal(a.alpha, "per_checkpoint").fit(Lc, Hc, yc).predict(Lt, Ht)
    bands["bonferroni"] = TrajectoryConformal(a.alpha, "bonferroni").fit(Lc, Hc, yc).predict(Lt, Ht)
    sim = TrajectoryConformal(a.alpha, "simultaneous").fit(Lc, Hc, yc, sc(len(yc)))
    bands["simultaneous_constant"] = sim.predict(Lt, Ht, sc(len(yt)))
    simB = TrajectoryConformal(a.alpha, "simultaneous").fit(Lc[B], Hc[B], yc[B], sc(len(B)))
    bands["simultaneous_constant_halfcal"] = simB.predict(Lt, Ht, sc(len(yt)))
    rm = RankMaxConformal(a.alpha).fit(Lc[A], Hc[A], yc[A], Lc[B], Hc[B], yc[B])
    bands["simultaneous_rankmax"] = rm.predict(Lt, Ht)
    for nm in ["simultaneous_constant", "simultaneous_rankmax"]:
        lo, hi = bands[nm]
        bands[nm + "_nested"] = (np.maximum.accumulate(lo, 1), np.minimum.accumulate(hi, 1))
    for nm, (lo, hi) in bands.items():
        r = summarize_intervals(lo, hi, yt, a.alpha)
        w = (hi - lo).mean(0)
        summ.append({"seed": seed, "method": nm, **r, **{f"width_{c}": w[k] for k, c in enumerate(CHECKPOINTS)}})
    # ---- goal-time alerts
    proj5 = naive_extrapolation(test, 1)
    rows_r, rows_g = [], []
    for g in GOALS:
        m = np.abs(proj5 - g) <= 15
        rows_r.append(np.flatnonzero(m)); rows_g.append(np.full(m.sum(), g))
    ri, gg = np.concatenate(rows_r), np.concatenate(rows_g)
    made = yt[ri] <= gg
    for nm, (lo, hi) in bands.items():
        for rule, fn in [("any", first_alert), ("two_consecutive", first_alert_consecutive)]:
            if rule == "two_consecutive" and nm not in ("per_checkpoint",):
                continue
            fa = fn(lo[ri], gg)
            alerted = fa >= 0
            km = np.where(alerted, CKPT_KM[np.maximum(fa, 0)], np.nan)
            dec.append({"seed": seed, "method": nm, "rule": rule, "n_pairs": len(ri), "n_made": int(made.sum()),
                        "false_alarm_rate": float(alerted[made].mean()),
                        "detection_rate": float(alerted[~made].mean()),
                        "median_alert_km_missed": float(np.nanmedian(km[~made])),
                        "detection_rate_by_30K": float((alerted & (fa <= 6))[~made].mean()),
                        "minutes_missed_median": float(np.median(yt[ri][~made] - gg[~made]))})
    # ---- example trajectories (seed 0): runners near finish-time quantiles where per-checkpoint misses somewhere
    if seed == 0:
        lo_p, hi_p = bands["per_checkpoint"]; lo_s, hi_s = bands["simultaneous_rankmax_nested"]
        miss_p = ~((yt[:, None] >= lo_p) & (yt[:, None] <= hi_p)).all(1)
        cov_s = ((yt[:, None] >= lo_s) & (yt[:, None] <= hi_s)).all(1)
        cand = np.flatnonzero(miss_p & cov_s)
        for qq in [0.25, 0.5, 0.75]:
            target = np.quantile(yt, qq)
            i = cand[np.argmin(np.abs(yt[cand] - target))]
            for k, c in enumerate(CHECKPOINTS):
                ex.append({"runner": f"q{int(qq*100)}", "checkpoint": c, "km": CKPT_KM[k], "finish": yt[i],
                           "pc_lo": lo_p[i, k], "pc_hi": hi_p[i, k], "band_lo": lo_s[i, k], "band_hi": hi_s[i, k],
                           "proj_cp": naive_extrapolation(test.iloc[[i]], k + 1)[0]})
    print(f"seed {seed} done", flush=True)

tag = "quick" if a.quick else "full"
S, D = pd.DataFrame(summ), pd.DataFrame(dec)
S.to_csv(out / f"decision_bands_{tag}.csv", index=False); D.to_csv(out / f"decision_alerts_{tag}.csv", index=False)
pd.DataFrame(ex).to_csv(out / f"example_trajectories_{tag}.csv", index=False)
pd.set_option("display.width", 220)
print(S.groupby("method")[["coverage_mean", "joint_coverage", "width_mean", "interval_score_mean", "width_5K", "width_40K"]].mean().round(3))
print(D.groupby(["method", "rule"])[["false_alarm_rate", "detection_rate", "detection_rate_by_30K", "median_alert_km_missed"]].mean().round(3))
