"""Same-day causal recalibration of a source-calibrated forecaster (labels arrive as runners finish)."""
import numpy as np
import pandas as pd
from pathlib import Path
from common import base_parser, maybe_path, ROOT  # noqa
from mtc.conformal import TrajectoryConformal, conformal_quantile
from mtc.data import ALL_EVENTS, CHECKPOINTS, SEG_COLS, load_all
from mtc.models import quantile_trajectories
from mtc.online import aci_delayed, sameday_quantiles

p = base_parser(__doc__)
p.add_argument("--targets", nargs="*", default=None)
p.add_argument("--n-fewshot", type=int, default=5000)
p.add_argument("--nmin", type=int, default=100)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
seeds = 1 if a.quick else a.seeds
N_SEGS = [2, 4, 6, 7, 8, 9]  # 10K, 20K, 25K, 30K, 35K, 40K
df = load_all(a.dev, maybe_path(a.berlin))
if a.quick:
    df = df.sample(frac=0.15, random_state=0)
events = a.targets or [e for e in ALL_EVENTS if (df.City == e).any()]
MAX_TRAIN = 20000 if a.quick else a.max_train
rows = []


def log(meta, method, q, S, w_base, grp, nlab):
    cov = S <= q
    width = w_base + 2 * q
    has = nlab >= a.nmin
    for sub, mask in [("all", np.ones(len(S), bool)), ("with_labels", has), ("without_labels", ~has)] + \
            [("group:" + g, grp == g) for g in np.unique(grp)]:
        if mask.sum() == 0:
            continue
        rows.append({**meta, "method": method, "subset": sub, "n": int(mask.sum()), "coverage": float(cov[mask].mean()),
                     "width_mean": float(width[mask].mean()), "frac_labeled": float(has.mean())})


for seed in range(seeds):
    rng = np.random.RandomState(seed)
    for tgt in events:
        src = df[(df.City != tgt) & (df.Year <= 2023)]
        idx = rng.permutation(len(src)); ntr = int(0.7 * len(src))
        tr, cal_src = src.iloc[idx[:ntr]], src.iloc[idx[ntr:]].iloc[:60000]
        t24 = df[(df.City == tgt) & (df.Year == 2024)]
        ti = rng.permutation(len(t24)); nfs = min(a.n_fewshot, len(t24) // 3)
        fs, test = t24.iloc[ti[:nfs]], t24.iloc[ti[nfs:]]
        res, _ = quantile_trajectories(tr, {"src": cal_src, "fs": fs, "test": test}, N_SEGS, a.alpha, seed, MAX_TRAIN,
                                       {"n_estimators": 60 if a.quick else a.n_estimators}, target=a.target)
        y_src, y_fs, yt = (d.finish_time_min.to_numpy() for d in (cal_src, fs, test))
        grp = test.performance_group.to_numpy()
        for j, n in enumerate(N_SEGS):
            ck = CHECKPOINTS[n - 1]
            meta = {"seed": seed, "target": tgt, "checkpoint": ck}
            S_src = TrajectoryConformal.scores(res["src"][0][:, [j]], res["src"][1][:, [j]], y_src)[:, 0]
            S_fs = TrajectoryConformal.scores(res["fs"][0][:, [j]], res["fs"][1][:, [j]], y_fs)[:, 0]
            Lt, Ht = res["test"][0][:, j], res["test"][1][:, j]
            S = np.maximum(Lt - yt, yt - Ht)
            t_pass = test[SEG_COLS[:n]].to_numpy(float).sum(1)  # net time at the checkpoint, proxy for clock time
            keep = t_pass < yt - 1e-6
            wb = Ht - Lt
            nl0 = np.zeros(len(yt), int)
            q0 = conformal_quantile(S_src, a.alpha)
            log(meta, "static_source", np.full(len(yt), q0), S, wb, grp, nl0)
            log(meta, "oracle_fewshot_static", np.full(len(yt), conformal_quantile(S_fs, a.alpha)), S, wb, grp, nl0)
            for m in ["naive", "shift_median", "shift_q"]:
                q, nl = sameday_quantiles(S_src, y_src, S, yt, t_pass, a.alpha, m, a.nmin)
                log(meta, m, q, S, wb, grp, nl)
            _, nl = sameday_quantiles(S_src, y_src, S, yt, t_pass, a.alpha, "naive", a.nmin)
            for gm in (0.005, 0.02):
                q = aci_delayed(S_src, S, yt, t_pass, a.alpha, gm)
                log(meta, f"aci_g{gm}", q, S, wb, grp, nl)
        print(f"seed {seed} target {tgt} done", flush=True)

d = pd.DataFrame(rows)
tag = "quick" if a.quick else "full"
d.to_csv(out / f"sameday_{tag}.csv", index=False)
x = d[d.subset == "all"].groupby(["target", "checkpoint", "method"]).coverage.mean().unstack("method").round(3)
print(x.to_string())
