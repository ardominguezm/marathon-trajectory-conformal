"""Wall-clock cost of fitting and querying the per-checkpoint LightGBM quantile pair and of calibration."""
import time, numpy as np, pandas as pd
from common import base_parser, ROOT  # noqa
from mtc.data import load_development, make_splits, checkpoint_features, naive_extrapolation
from mtc.models import quantile_trajectories
from mtc.conformal import TrajectoryConformal
a = base_parser(__doc__).parse_args()
sp = make_splits(load_development(a.dev), "temporal")
rows = []
for n in [1, 4, 9]:
    t0 = time.perf_counter()
    res, w = quantile_trajectories(sp["train"], {"cal": sp["cal"]}, [n], a.alpha, 0, a.max_train)
    fit_cal = time.perf_counter() - t0
    import lightgbm as lgb
    tr = sp["train"].iloc[:a.max_train]; X = checkpoint_features(tr, n); y = tr.finish_time_min.values - naive_extrapolation(tr, n)
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.05, n_estimators=250, learning_rate=0.06, num_leaves=63, min_child_samples=100, verbose=-1, n_jobs=2).fit(X, y)
    Xq = checkpoint_features(sp["test"].iloc[:10000], n)
    t1 = time.perf_counter(); m.predict(Xq); m.predict(Xq); pred = time.perf_counter() - t1
    rows.append({"segments": n, "fit_two_models_and_predict_cal_s": fit_cal, "predict_10k_runners_two_models_ms": 1000 * pred})
L, H = np.random.rand(123710, 9), np.random.rand(123710, 9) + 1; y = np.random.rand(123710)
t2 = time.perf_counter(); TrajectoryConformal(0.1, "simultaneous").fit(L, H, y); cal = time.perf_counter() - t2
d = pd.DataFrame(rows); d["calibration_simultaneous_s"] = cal
d.to_csv(ROOT / "results" / "timing.csv", index=False); print(d.round(3))
