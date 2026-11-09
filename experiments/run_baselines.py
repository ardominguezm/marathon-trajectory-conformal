"""Baseline comparison under the temporal protocol: classical, ensemble and neural predictors, all conformalized."""
import numpy as np
import pandas as pd
from pathlib import Path
from common import base_parser, ROOT  # noqa
from mtc.conformal import TrajectoryConformal, conformal_quantile
from mtc.data import CHECKPOINTS, checkpoint_features, load_development, make_splits, naive_extrapolation
from mtc.models import LGBMHetero, MLPHetero, RFHetero, fit_hetero_split_cp, quantile_trajectories, _subsample

p = base_parser(__doc__)
p.add_argument("--max-train-dl", type=int, default=60000)
p.add_argument("--ckpts", nargs="*", type=int, default=[1, 2, 4, 7, 9], help="number of segments (5K=1, 10K=2, 20K=4, 30K=7, 40K=9)")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
seeds = 1 if a.quick else a.seeds
dev = load_development(a.dev)
if a.quick:
    dev = dev.sample(frac=0.15, random_state=0)
sp = make_splits(dev, "temporal")
cal, test = sp["cal"], sp["test"]
yc, yt = cal.finish_time_min.to_numpy(), test.finish_time_min.to_numpy()
rows = []


def log(method, seed, n, lo, hi, mu=None):
    w = hi - lo
    rows.append({"method": method, "seed": seed, "checkpoint": CHECKPOINTS[n - 1],
                 "coverage": float(((yt >= lo) & (yt <= hi)).mean()), "width_mean": float(w.mean()),
                 "interval_score": float(np.mean(w + (2 / a.alpha) * (np.maximum(lo - yt, 0) + np.maximum(yt - hi, 0)))),
                 "rmse_point": float(np.sqrt(np.mean((yt - mu) ** 2))) if mu is not None else np.nan})


MAXDL = 8000 if a.quick else a.max_train_dl
for seed in range(seeds):
    tr = sp["train"]
    for n in a.ckpts:
        # classical: constant-pace extrapolation + absolute-residual split CP
        mu_c, mu_t = naive_extrapolation(cal, n), naive_extrapolation(test, n)
        q = conformal_quantile(np.abs(yc - mu_c), a.alpha)
        log("naive_extrapolation_splitCP", seed, n, mu_t - q, mu_t + q, mu_t)
        # LightGBM quantile + CQR (the submitted base learner)
        for tgt_name, label in [("residual", "lgbm_quantile_CQR"), ("absolute", "lgbm_quantile_CQR_absolute_target")]:
            res, _ = quantile_trajectories(tr, {"cal": cal, "test": test}, [n], a.alpha, seed, a.max_train,
                                           {"n_estimators": 60 if a.quick else a.n_estimators}, target=tgt_name)
            tc = TrajectoryConformal(a.alpha, "per_checkpoint").fit(*res["cal"], yc)
            lo, hi = tc.predict(*res["test"])
            log(label, seed, n, lo[:, 0], hi[:, 0])
        # learners with locally weighted split CP
        trs = _subsample(tr, MAXDL, seed)
        Xtr, ytr = checkpoint_features(trs, n), trs.finish_time_min.to_numpy()
        Xc, Xt = checkpoint_features(cal, n), checkpoint_features(test, n)
        for name, model in [("lgbm_hetero_localCP", LGBMHetero(seed, n_estimators=60 if a.quick else 250)),
                            ("random_forest_localCP", RFHetero(seed, 20 if a.quick else 80)),
                            ("mlp_hetero_localCP", MLPHetero(seed, max_iter=15 if a.quick else 60))]:
            lo, hi, mu = fit_hetero_split_cp(model, Xtr, ytr, Xc, yc, Xt, a.alpha, naive_extrapolation(trs, n), mu_c, mu_t)
            log(name, seed, n, lo, hi, mu)
        print(f"seed {seed} checkpoint {CHECKPOINTS[n-1]} done", flush=True)
d = pd.DataFrame(rows)
d.to_csv(out / f"baselines_{'quick' if a.quick else 'full'}.csv", index=False)
print(d.groupby(["checkpoint", "method"])[["coverage", "width_mean", "interval_score", "rmse_point"]].mean().round(3).to_string())
