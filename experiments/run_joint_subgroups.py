"""Joint coverage over the nine checkpoints within subgroups (temporal protocol)."""
import numpy as np
import pandas as pd
from pathlib import Path
from common import base_parser, maybe_path, ROOT  # noqa
from mtc.conformal import RankMaxConformal, TrajectoryConformal
from mtc.data import load_development, load_wall_labels, make_splits
from mtc.models import quantile_trajectories

a = base_parser(__doc__).parse_args()
dev = load_development(a.dev)
if maybe_path(a.wall):
    dev = dev.merge(load_wall_labels(a.wall), on="runner_id", how="left")
rows = []
for seed in range(a.seeds):
    sp = make_splits(dev, "temporal", seed)
    cal, test = sp["cal"], sp["test"]
    res, width = quantile_trajectories(sp["train"], {"cal": cal, "test": test}, list(range(1, 10)), a.alpha, seed,
                                       a.max_train, {"n_estimators": a.n_estimators}, target=a.target)
    (Lc, Hc), (Lt, Ht) = res["cal"], res["test"]
    yc, yt = cal.finish_time_min.to_numpy(), test.finish_time_min.to_numpy()
    sc = lambda n: np.tile(width, (n, 1))
    half = np.random.RandomState(seed).permutation(len(yc)); A, B = half[: len(yc) // 2], half[len(yc) // 2:]
    bands = {
        "per_checkpoint": TrajectoryConformal(a.alpha, "per_checkpoint").fit(Lc, Hc, yc).predict(Lt, Ht),
        "constant_band": TrajectoryConformal(a.alpha, "simultaneous").fit(Lc, Hc, yc, sc(len(yc))).predict(Lt, Ht, sc(len(yt))),
        "rank_band": RankMaxConformal(a.alpha).fit(Lc[A], Hc[A], yc[A], Lc[B], Hc[B], yc[B]).predict(Lt, Ht),
    }
    dims = {"all": np.array(["all"] * len(test)), "performance_group": test.performance_group.to_numpy(),
            "sex": np.where(test.male.to_numpy() == 1, "male", "female"), "event": test.City.to_numpy()}
    if "wall_persistent" in test:
        dims["wall_persistent"] = test.wall_persistent.fillna(-1).astype(int).astype(str).to_numpy()
    for m, (lo, hi) in bands.items():
        joint = ((yt[:, None] >= lo) & (yt[:, None] <= hi)).all(1)
        for dim, lab in dims.items():
            for g in np.unique(lab):
                k = lab == g
                rows.append({"seed": seed, "method": m, "dimension": dim, "group": g, "n": int(k.sum()),
                             "joint_coverage": float(joint[k].mean())})
    print(f"seed {seed} done", flush=True)
d = pd.DataFrame(rows)
d.to_csv(ROOT / "results" / "joint_by_subgroup.csv", index=False)
print(d.groupby(["dimension", "group", "method"]).joint_coverage.mean().unstack("method").round(3).to_string())
