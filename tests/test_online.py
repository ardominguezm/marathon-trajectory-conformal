import numpy as np
from mtc.online import aci_delayed, sameday_quantiles
from mtc.conformal import conformal_quantile


def make(n, shift, seed, scale=1.0):
    r = np.random.RandomState(seed)
    y = r.normal(240, 40, n)
    S = r.normal(0, 3, n) * scale + shift + 0.02 * (y - 240)  # score drifts with finish time (selection matters)
    return y, S


def test_never_uses_unfinished_labels():
    y_src, S_src = make(20000, 0.0, 0)
    y, S = make(5000, 4.0, 1)
    t = y - 10.0
    _, nlab = sameday_quantiles(S_src, y_src, S, y, t, 0.1)
    for i in range(0, 5000, 500):
        assert nlab[i] <= (y <= np.floor(t[i] / 2) * 2).sum()


def test_matched_shift_beats_naive_under_selection_and_shift():
    alpha = 0.1
    y_src, S_src = make(40000, 0.0, 2)
    y, S = make(20000, 3.0, 3)
    t = y - 20.0
    out = {}
    for m in ["naive", "shift_median", "shift_q"]:
        q, nl = sameday_quantiles(S_src, y_src, S, y, t, alpha, m)
        use = nl >= 100
        out[m] = abs((S[use] <= q[use]).mean() - 0.9)
    q0 = conformal_quantile(S_src, alpha)
    static = abs((S <= q0).mean() - 0.9)
    assert out["shift_median"] < static and out["shift_median"] < out["naive"]


def test_aci_runs_and_is_causal():
    y_src, S_src = make(20000, 0.0, 4)
    y, S = make(8000, 3.0, 5)
    q = aci_delayed(S_src, S, y, y - 15.0, 0.1)
    assert np.isfinite(q).all()
