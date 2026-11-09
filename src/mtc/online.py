"""Same-day (causal) recalibration when labels arrive only as runners finish.

Setting. A runner who passes checkpoint k at clock time t can only be calibrated with same-day runners who have already
FINISHED, i.e. whose finish time is <= t. Those finishers are the fastest runners, so their scores are a sample selected on the
outcome. All functions below use a runner's own label never, and other labels only if finish time <= the issue time.

Clock time is approximated by net time (all runners treated as starting together); wave starts are ignored.

Methods
  naive         : conformal quantile of the finishers' scores (selection-biased).
  shift_median  : q_source + shrunk difference of medians between finishers and source runners selected the same way
                  (source calibration rows with finish time <= t). Assumes the day effect is common across finish-time strata.
  shift_q       : same with the (1 - alpha) quantile instead of the median.
  aci           : adaptive conformal inference (Gibbs and Candes, 2021) driven by the finishers' miss indicators, with the
                  feedback delay implied by the race.
"""
from __future__ import annotations

import numpy as np

from .conformal import conformal_quantile


def sameday_quantiles(S_src, y_src, S_day, y_day, t_query, alpha, method="shift_median",
                      nmin: int = 100, n0: float = 300.0, grid: float = 2.0):
    """Return (q per query, number of usable same-day labels per query). Pools are taken at the start of the grid bin."""
    S_src, y_src, S_day, y_day, t_query = (np.asarray(a, float) for a in (S_src, y_src, S_day, y_day, t_query))
    q0 = conformal_quantile(S_src, alpha)
    o = np.argsort(y_day, kind="stable"); yd, Sd = y_day[o], S_day[o]
    o2 = np.argsort(y_src, kind="stable"); ys, Ss = y_src[o2], S_src[o2]
    bins = np.floor(t_query / grid) * grid
    q = np.full(len(t_query), q0)
    nlab = np.zeros(len(t_query), int)
    for b in np.unique(bins):
        m = int(np.searchsorted(yd, b, side="right"))
        sel = bins == b
        nlab[sel] = m
        ms = int(np.searchsorted(ys, b, side="right"))
        if m < nmin or ms < nmin:
            continue
        pool, matched = Sd[:m], Ss[:ms]
        w = m / (m + n0)
        if method == "naive":
            q[sel] = conformal_quantile(pool, alpha)
        elif method == "shift_median":
            q[sel] = q0 + w * (np.median(pool) - np.median(matched))
        elif method == "shift_q":
            q[sel] = q0 + w * (conformal_quantile(pool, alpha) - conformal_quantile(matched, alpha))
        else:
            raise ValueError(method)
    return q, nlab


def aci_delayed(S_src, S_day, y_day, t_query, alpha, gamma: float = 0.02, grid: float = 2.0):
    """ACI with race-implied delayed feedback. Queries are the day runners themselves (t_query[i] < y_day[i])."""
    S_src, S_day, y_day, t_query = (np.asarray(a, float) for a in (S_src, S_day, y_day, t_query))
    srt = np.sort(S_src); n = len(srt)

    def qa(a):
        a = float(np.clip(a, 1e-3, 0.5))
        k = int(np.ceil((1 - a) * (n + 1)))
        return np.inf if k > n else srt[k - 1]

    tb = np.floor(t_query / grid).astype(int); fb = np.floor(y_day / grid).astype(int)
    assert (tb <= fb).all(), "a label would be used before the forecast was issued"
    ord_q, ord_f = np.argsort(tb, kind="stable"), np.argsort(fb, kind="stable")
    q_iss = np.zeros(len(S_day)); a = alpha; pq = pf = 0
    for g in range(int(max(tb.max(), fb.max())) + 1):
        qg = qa(a)
        while pq < len(ord_q) and tb[ord_q[pq]] == g:
            q_iss[ord_q[pq]] = qg; pq += 1
        idx = []
        while pf < len(ord_f) and fb[ord_f[pf]] == g:
            idx.append(ord_f[pf]); pf += 1
        if idx:
            err = np.mean(S_day[idx] > q_iss[idx])
            a = a + gamma * (alpha - err)
    return q_iss
