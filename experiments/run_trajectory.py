"""Trajectory validity: per-checkpoint vs Bonferroni vs simultaneous bands (plus Mondrian ablation)."""
import numpy as np
import pandas as pd
from common import base_parser, maybe_path, ROOT  # noqa
from mtc.conformal import TrajectoryConformal, hierarchical_mondrian_quantile
from mtc.data import CHECKPOINTS, load_development, load_wall_labels, make_splits, DEV_EVENTS
from mtc.metrics import slice_coverage, summarize_intervals
from mtc.models import quantile_trajectories

p = base_parser(__doc__)
p.add_argument("--protocol", choices=["temporal", "exchangeable"], default="temporal")
a = p.parse_args()
out = ROOT / "results" if a.out is None else __import__("pathlib").Path(a.out)
out.mkdir(parents=True, exist_ok=True)
seeds = 1 if a.quick else a.seeds
N_SEGS = list(range(1, 10))
dev = load_development(a.dev)
wall = maybe_path(a.wall)
if wall:
    dev = dev.merge(load_wall_labels(wall), on="runner_id", how="left")
if a.quick:
    dev = dev.sample(frac=0.15, random_state=0)
MAX_TRAIN = 20000 if a.quick else a.max_train
SLICE_CKPTS = ["20K", "30K"]


def mondrian_per_checkpoint(Sc, cal, test, Xc_cum, Xt_cum, alpha, nmin=150):
    """Original CP-PRO partition (pace quantile x sex x age) with backoff, applied independently per checkpoint."""
    K = Sc.shape[1]
    q = np.zeros((len(test), K))
    for k in range(K):
        nb = 4 if k <= 3 else 5 if k <= 5 else 6
        edges = np.quantile(Xc_cum[:, k], np.linspace(0, 1, nb + 1))
        pb_c = np.clip(np.digitize(Xc_cum[:, k], edges[1:-1]), 0, nb - 1)
        pb_t = np.clip(np.digitize(Xt_cum[:, k], edges[1:-1]), 0, nb - 1)
        ab_c, ab_t = np.digitize(cal.Age, [30, 40, 50, 60]), np.digitize(test.Age, [30, 40, 50, 60])
        kc = [pb_c * 100 + cal.male.to_numpy() * 10 + ab_c, pb_c * 10 + cal.male.to_numpy(), pb_c]
        kt = [pb_t * 100 + test.male.to_numpy() * 10 + ab_t, pb_t * 10 + test.male.to_numpy(), pb_t]
        q[:, k], _ = hierarchical_mondrian_quantile(Sc[:, k], kc, kt, alpha, nmin)
    return q


from mtc.data import SEG_COLS, SEG_KM  # noqa: E402


def cum_pace(df):
    seg = df[SEG_COLS].to_numpy(float)
    return np.cumsum(seg[:, :9], axis=1) / np.cumsum(SEG_KM[:9])


summ, byck, slices = [], [], []
for seed in range(seeds):
    sp = make_splits(dev, a.protocol, seed)
    cal, test = sp["cal"], sp["test"]
    tr, width = quantile_trajectories(sp["train"], {"cal": cal, "test": test}, N_SEGS, a.alpha, seed,
                                      MAX_TRAIN, {"n_estimators": 60 if a.quick else a.n_estimators}, target=a.target)
    (Lc, Hc), (Lt, Ht) = tr["cal"], tr["test"]
    yc, yt = cal.finish_time_min.to_numpy(), test.finish_time_min.to_numpy()
    sc_const_c, sc_const_t = np.tile(width, (len(cal), 1)), np.tile(width, (len(test), 1))
    sc_loc_c, sc_loc_t = np.maximum(Hc - Lc, 1.0), np.maximum(Ht - Lt, 1.0)
    methods = {}
    for name, mode, nested, sc in [
        ("per_checkpoint", "per_checkpoint", False, None),
        ("per_checkpoint_nested", "per_checkpoint", True, None),
        ("bonferroni", "bonferroni", False, None),
        ("simultaneous", "simultaneous", False, "const"),
        ("simultaneous_local", "simultaneous", False, "local"),
        ("simultaneous_nested", "simultaneous", True, "const"),
    ]:
        scc, sct = {None: (None, None), "const": (sc_const_c, sc_const_t), "local": (sc_loc_c, sc_loc_t)}[sc]
        tc = TrajectoryConformal(a.alpha, mode, nested).fit(Lc, Hc, yc, scc)
        methods[name] = tc.predict(Lt, Ht, sct)
    Sc = TrajectoryConformal.scores(Lc, Hc, yc)
    qm = mondrian_per_checkpoint(Sc, cal, test, cum_pace(cal), cum_pace(test), a.alpha)
    methods["mondrian_pace_sex_age"] = (Lt - qm, Ht + qm)
    for name, (lo, hi) in methods.items():
        r = summarize_intervals(lo, hi, yt, a.alpha)
        summ.append({"protocol": a.protocol, "seed": seed, "method": name, **r})
        cov = ((yt[:, None] >= lo) & (yt[:, None] <= hi)).mean(0)
        for k, ck in enumerate(CHECKPOINTS):
            byck.append({"protocol": a.protocol, "seed": seed, "method": name, "checkpoint": ck,
                         "coverage": cov[k], "width_mean": float((hi - lo)[:, k].mean()),
                         "interval_score": float(np.mean((hi - lo)[:, k] + (2 / a.alpha) * (np.maximum(lo[:, k] - yt, 0)
                                                 + np.maximum(yt - hi[:, k], 0))))})
        if name in {"per_checkpoint", "simultaneous", "mondrian_pace_sex_age"}:
            for ck in SLICE_CKPTS:
                k = CHECKPOINTS.index(ck)
                dims = {"performance_group": test.performance_group, "event": test.City}
                if "wall_persistent" in test:
                    dims["wall_persistent"] = test.wall_persistent.fillna(-1).astype(int)
                for dim, lab in dims.items():
                    s = slice_coverage(lo[:, k], hi[:, k], yt, lab.to_numpy())
                    s.insert(0, "dimension", dim)
                    s.insert(0, "checkpoint", ck); s.insert(0, "method", name); s.insert(0, "seed", seed)
                    s.insert(0, "protocol", a.protocol)
                    slices.append(s)
    print(f"seed {seed} done", flush=True)

tag = a.protocol + ("_absolute" if a.target == "absolute" else "") + ("_quick" if a.quick else "")
pd.DataFrame(summ).to_csv(out / f"trajectory_summary_{tag}.csv", index=False)
pd.DataFrame(byck).to_csv(out / f"trajectory_by_checkpoint_{tag}.csv", index=False)
pd.concat(slices).to_csv(out / f"trajectory_slices_{tag}.csv", index=False)
print(pd.DataFrame(summ).groupby("method").mean(numeric_only=True).round(3).to_string())
