import numpy as np
import pandas as pd
from mtc.conformal import TrajectoryConformal, conformal_quantile, hierarchical_mondrian_quantile
from mtc.metrics import summarize_intervals


def synth(n, K=9, rho=0.9, seed=0):
    """Same target y; checkpoint-specific predictions with strongly dependent errors."""
    r = np.random.RandomState(seed)
    y = r.normal(240, 40, n)
    common = r.normal(0, 1, n)
    L, H = np.zeros((n, K)), np.zeros((n, K))
    sd = np.linspace(25, 3, K)
    for k in range(K):
        e = rho * common + np.sqrt(1 - rho**2) * r.normal(0, 1, n)
        mu = y + sd[k] * e  # prediction error at checkpoint k
        L[:, k], H[:, k] = mu - 1.645 * sd[k], mu + 1.645 * sd[k]
    return L, H, y


def test_quantile_finite_sample_rank():
    s = np.arange(1, 100)  # n = 99
    assert conformal_quantile(s, 0.1) == 90  # ceil(0.9 * 100)
    assert np.isinf(conformal_quantile(s[:5], 0.1))


def test_weighted_equals_unweighted_with_unit_weights():
    s = np.random.RandomState(1).normal(size=999)
    assert abs(conformal_quantile(s, 0.1, weights=np.ones(999), w_test=1.0) - conformal_quantile(s, 0.1)) < 1e-12


def test_modes_coverage_ordering():
    alpha = 0.1
    Lc, Hc, yc = synth(20000, seed=1)
    Lt, Ht, yt = synth(20000, seed=2)
    res = {}
    for m in ["per_checkpoint", "bonferroni", "simultaneous"]:
        tc = TrajectoryConformal(alpha, m).fit(Lc, Hc, yc)
        lo, hi = tc.predict(Lt, Ht)
        res[m] = summarize_intervals(lo, hi, yt, alpha)
    # each checkpoint is calibrated marginally...
    assert abs(res["per_checkpoint"]["coverage_mean"] - 0.9) < 0.01
    # ...but joint coverage is not
    assert res["per_checkpoint"]["joint_coverage"] < 0.85
    # simultaneous band guarantees joint coverage
    assert res["simultaneous"]["joint_coverage"] >= 0.9 - 0.01
    assert res["bonferroni"]["joint_coverage"] >= 0.9 - 0.01
    # and is no wider than Bonferroni on average here
    assert res["simultaneous"]["width_mean"] <= res["bonferroni"]["width_mean"] * 1.02


def test_simultaneous_joint_coverage_over_replications():
    alpha, hits = 0.1, []
    for s in range(60):
        Lc, Hc, yc = synth(400, seed=100 + s)
        Lt, Ht, yt = synth(400, seed=500 + s)
        tc = TrajectoryConformal(alpha, "simultaneous").fit(Lc, Hc, yc)
        lo, hi = tc.predict(Lt, Ht)
        hits.append(summarize_intervals(lo, hi, yt, alpha)["joint_coverage"])
    assert np.mean(hits) >= 0.9 - 0.01  # finite-sample guarantee holds on average


def test_nested_intersection_preserves_joint_coverage():
    alpha = 0.1
    Lc, Hc, yc = synth(20000, seed=3)
    Lt, Ht, yt = synth(20000, seed=4)
    tc = TrajectoryConformal(alpha, "simultaneous", nested=True).fit(Lc, Hc, yc)
    lo, hi = tc.predict(Lt, Ht)
    assert (np.diff(hi, axis=1) <= 1e-9).all() and (np.diff(lo, axis=1) >= -1e-9).all()
    assert summarize_intervals(lo, hi, yt, alpha)["joint_coverage"] >= 0.89


def test_mondrian_backoff():
    r = np.random.RandomState(0)
    s = r.normal(size=1000)
    fine = np.r_[np.zeros(900, int), np.arange(1, 101)]  # one big cell + 100 singletons
    coarse = np.zeros(1000, int)
    q, lv = hierarchical_mondrian_quantile(s, [fine, coarse], [np.array([0, 5]), np.array([0, 0])], 0.1, nmin=50)
    assert lv[0] == 0 and lv[1] == 1  # cell 5 has 1 point: backs off


def test_rankmax_joint_validity_and_efficiency():
    from mtc.conformal import RankMaxConformal
    alpha = 0.1
    hits, w_rm, w_sim = [], [], []
    for s in range(30):
        Lt_, Ht_, yt_ = synth(3000, seed=900 + s)
        Lc, Hc, yc = synth(3000, seed=1900 + s)
        Lx, Hx, yx = synth(3000, seed=2900 + s)
        rm = RankMaxConformal(alpha).fit(Lt_, Ht_, yt_, Lc, Hc, yc)
        lo, hi = rm.predict(Lx, Hx)
        hits.append(summarize_intervals(lo, hi, yx, alpha)["joint_coverage"])
        w_rm.append((hi - lo).mean())
        tc = TrajectoryConformal(alpha, "simultaneous").fit(Lc, Hc, yc)
        lo2, hi2 = tc.predict(Lx, Hx)
        w_sim.append((hi2 - lo2).mean())
    assert np.mean(hits) >= 0.9 - 0.01
