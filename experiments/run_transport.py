"""Transport across events: leave-one-event-out training, four calibration schemes, target = 2024 edition."""
import numpy as np
import pandas as pd
from pathlib import Path
from common import base_parser, maybe_path, ROOT  # noqa
from mtc.conformal import TrajectoryConformal, conformal_quantile, density_ratio_weights
from mtc.data import ALL_EVENTS, CHECKPOINTS, checkpoint_features, load_all
from mtc.models import quantile_trajectories

p = base_parser(__doc__)
p.add_argument("--targets", nargs="*", default=None)
p.add_argument("--n-fewshot", type=int, default=5000)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
seeds = 1 if a.quick else a.seeds
N_SEGS = list(range(1, 10))
df = load_all(a.dev, maybe_path(a.berlin))
if a.quick:
    df = df.sample(frac=0.15, random_state=0)
events = a.targets or [e for e in ALL_EVENTS if (df.City == e).any()]
MAX_TRAIN = 20000 if a.quick else a.max_train
rows = []


def summarize(tag, lo, hi, yt, **meta):
    cov = (yt[:, None] >= lo) & (yt[:, None] <= hi)
    for k, ck in enumerate(CHECKPOINTS):
        w = (hi - lo)[:, k]
        rows.append({**meta, **tag, "checkpoint": ck, "coverage": float(cov[:, k].mean()),
                     "width_mean": float(w.mean()),
                     "interval_score": float(np.mean(w + (2 / a.alpha) * (np.maximum(lo[:, k] - yt, 0) + np.maximum(yt - hi[:, k], 0)))),
                     "joint_coverage": float(cov.all(1).mean()), "n_test": len(yt)})


for seed in range(seeds):
    rng = np.random.RandomState(seed)
    for tgt in events:
        src = df[(df.City != tgt) & (df.Year <= 2023)]
        idx = rng.permutation(len(src)); ntr = int(0.7 * len(src))
        tr, cal_src = src.iloc[idx[:ntr]], src.iloc[idx[ntr:]].iloc[:60000]
        prev = df[(df.City == tgt) & (df.Year == 2023)]
        t24 = df[(df.City == tgt) & (df.Year == 2024)]
        ti = rng.permutation(len(t24)); nfs = min(a.n_fewshot, len(t24) // 3)
        fs, test = t24.iloc[ti[:nfs]], t24.iloc[ti[nfs:]]
        if len(prev) < 500 or len(test) < 500:
            print("skip", tgt); continue
        res, width = quantile_trajectories(tr, {"src": cal_src, "prev": prev, "fs": fs, "test": test}, N_SEGS, a.alpha,
                                           seed, MAX_TRAIN, {"n_estimators": 60 if a.quick else a.n_estimators}, target=a.target)
        Lt, Ht = res["test"]; yt = test.finish_time_min.to_numpy()
        meta = {"seed": seed, "target": tgt}
        for scheme, key, frame in [("source_split", "src", cal_src), ("prev_edition", "prev", prev), ("fewshot_oracle", "fs", fs)]:
            Lc, Hc = res[key]; yc = frame.finish_time_min.to_numpy()
            for mode, nested in [("per_checkpoint", False), ("simultaneous", False), ("simultaneous", True)]:
                sc_c = np.tile(width, (len(yc), 1)) if mode == "simultaneous" else None
                sc_t = np.tile(width, (len(yt), 1)) if mode == "simultaneous" else None
                tc = TrajectoryConformal(a.alpha, mode, nested).fit(Lc, Hc, yc, sc_c)
                lo, hi = tc.predict(Lt, Ht, sc_t)
                summarize({"scheme": scheme, "mode": mode + ("_nested" if nested else "")}, lo, hi, yt, **meta)
        # covariate-shift weighted source calibration (per checkpoint, marginal; unlabeled target X only)
        Lc, Hc = res["src"]; yc = cal_src.finish_time_min.to_numpy()
        S = TrajectoryConformal.scores(Lc, Hc, yc)
        q = np.zeros(len(CHECKPOINTS))
        for j, n in enumerate(N_SEGS):
            ws, wt = density_ratio_weights(checkpoint_features(cal_src, n).iloc[:30000],
                                           checkpoint_features(test, n).iloc[:30000], seed=seed)
            q[j] = conformal_quantile(S[:30000, j], a.alpha, ws, wt)
        summarize({"scheme": "weighted_source", "mode": "per_checkpoint"}, Lt - q, Ht + q, yt, **meta)
        print(f"seed {seed} target {tgt} done", flush=True)

tag = ("quick" if a.quick else "full") + ("_absolute" if a.target == "absolute" else "")
pd.DataFrame(rows).to_csv(out / f"transport_{tag}.csv", index=False)
d = pd.DataFrame(rows)
print(d[d.checkpoint.isin(["20K", "30K"])].groupby(["target", "checkpoint", "scheme", "mode"]).coverage.mean().unstack("scheme").round(3).to_string())
