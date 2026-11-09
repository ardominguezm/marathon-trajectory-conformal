"""Conformal calibration utilities.

TrajectoryConformal
    Wraps lower/upper quantile predictions made at K checkpoints for the SAME target (the finish time)
    and calibrates them in one of three ways.

    per_checkpoint : one CQR correction per checkpoint. Valid marginally at each checkpoint, but the
                     probability that the finish time lies inside the interval at *all* checkpoints is
                     in general far below 1 - alpha.
    bonferroni     : per-checkpoint corrections at level alpha / K (simultaneous but conservative).
    simultaneous   : one common multiplier q for the maximum scaled score over checkpoints,
                         S_i = max_k  s_ik / sigma_ik ,   interval_k = [L_k - q sigma_k, H_k + q sigma_k].
                     Because {y in interval_k for all k} = {S <= q}, split-conformal finite-sample theory
                     gives P(y in all K intervals) >= 1 - alpha under exchangeability of (x_i, y_i), for any
                     fixed positive scales sigma (a function of x and of data independent of the calibration
                     and test labels). Running intersections of valid simultaneous intervals remain valid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_quantile(scores, alpha: float, weights=None, w_test: float | None = None) -> float:
    """Finite-sample conformal quantile.

    Unweighted: the ceil((1 - alpha)(n + 1))-th smallest score (inf if that exceeds n).
    Weighted (Tibshirani et al., 2019): weighted quantile with mass w_test placed at +inf. Here a single
    w_test (default: mean calibration weight) is used for all test points, which is an approximation to
    the exact test-point-specific quantile and should be reported as such.
    """
    s = np.asarray(scores, float)
    n = len(s)
    if n == 0:
        return np.inf
    if weights is None:
        k = int(np.ceil((1 - alpha) * (n + 1)))
        return np.inf if k > n else float(np.partition(s, k - 1)[k - 1])
    w = np.asarray(weights, float)
    wt = float(w.mean()) if w_test is None else float(w_test)
    order = np.argsort(s)
    s, w = s[order], w[order]
    cdf = np.cumsum(w) / (w.sum() + wt)
    idx = int(np.searchsorted(cdf, 1 - alpha, side="left"))
    return np.inf if idx >= n else float(s[idx])


class TrajectoryConformal:
    def __init__(self, alpha: float = 0.1, mode: str = "per_checkpoint", nested: bool = False):
        if mode not in {"per_checkpoint", "bonferroni", "simultaneous"}:
            raise ValueError(mode)
        self.alpha, self.mode, self.nested = alpha, mode, nested

    @staticmethod
    def scores(L, H, y):
        """Signed CQR scores (n, K)."""
        y = np.asarray(y, float)[:, None]
        return np.maximum(L - y, y - H)

    def fit(self, L, H, y, scale=None, weights=None):
        L, H = np.asarray(L, float), np.asarray(H, float)
        n, K = L.shape
        scale = np.ones_like(L) if scale is None else np.asarray(scale, float)
        Z = self.scores(L, H, y) / scale
        if self.mode == "per_checkpoint":
            self.q_ = np.array([conformal_quantile(Z[:, k], self.alpha, weights) for k in range(K)])
        elif self.mode == "bonferroni":
            self.q_ = np.array([conformal_quantile(Z[:, k], self.alpha / K, weights) for k in range(K)])
        else:
            self.q_ = np.full(K, conformal_quantile(Z.max(1), self.alpha, weights))
        return self

    def predict(self, L, H, scale=None):
        L, H = np.asarray(L, float), np.asarray(H, float)
        scale = np.ones_like(L) if scale is None else np.asarray(scale, float)
        lo, hi = L - self.q_ * scale, H + self.q_ * scale
        if self.nested:
            lo, hi = np.maximum.accumulate(lo, axis=1), np.minimum.accumulate(hi, axis=1)
        return lo, hi


def hierarchical_mondrian_quantile(scores, cal_keys, test_keys, alpha: float, nmin: int = 150):
    """Mondrian CQR correction with backoff.

    cal_keys / test_keys: lists of arrays ordered from the finest to the coarsest partition. For every test
    point the finest cell with at least nmin calibration points is used; otherwise the global quantile.
    Returns (q per test point, index of the level used; -1 means global).
    """
    s = np.asarray(scores, float)
    q_glob = conformal_quantile(s, alpha)
    n_test = len(test_keys[0])
    q = np.full(n_test, np.nan)
    level = np.full(n_test, -1)
    for lv, (kc, kt) in enumerate(zip(cal_keys, test_keys)):
        g = pd.Series(s).groupby(np.asarray(kc))
        cell_q = {k: conformal_quantile(v.to_numpy(), alpha) for k, v in g if len(v) >= nmin}
        mapped = pd.Series(np.asarray(kt)).map(cell_q).to_numpy(float)
        fill = np.isnan(q) & ~np.isnan(mapped)
        q[fill], level[fill] = mapped[fill], lv
    q[np.isnan(q)] = q_glob
    return q, level


def density_ratio_weights(X_source, X_target, clip_quantile: float = 0.99, seed: int = 0):
    """Covariate-shift weights w(x) = p_target(x) / p_source(x) for source rows, from a probabilistic classifier.

    Returns (weights for X_source, mean weight of target rows under the same model).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xs, Xt = np.asarray(X_source, float), np.asarray(X_target, float)
    X = np.vstack([Xs, Xt])
    z = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed).fit(X, z)
    ratio = len(Xs) / len(Xt)

    def w(A):
        p = np.clip(clf.predict_proba(A)[:, 1], 1e-4, 1 - 1e-4)
        return ratio * p / (1 - p)

    ws, wt = w(Xs), w(Xt)
    cap = np.quantile(ws, clip_quantile)
    return np.minimum(ws, cap), float(np.minimum(wt, cap).mean())


class RankMaxConformal:
    """Simultaneous band with per-checkpoint rank normalisation (an empirical-copula style construction).

    A tuning split fixes increasing maps g_k(s) = empirical CDF of the checkpoint-k scores on that split. The joint score
    S = max_k g_k(s_k) is calibrated on a second, disjoint split. Interval_k = {y : g_k(s_k(y)) <= q}, i.e.
    [L_k - c_k, H_k + c_k] with c_k = g_k^{-1}(q). Because the g_k are fixed before calibration and increasing, the
    argument of Proposition 1 applies unchanged: P(y in all K intervals) >= 1 - alpha under exchangeability.
    Every checkpoint receives the same marginal level, which avoids spending the joint budget where scores are heavy-tailed.
    """

    def __init__(self, alpha: float = 0.1, nested: bool = False):
        self.alpha, self.nested = alpha, nested

    def fit(self, L_tune, H_tune, y_tune, L_cal, H_cal, y_cal):
        St = TrajectoryConformal.scores(L_tune, H_tune, y_tune)
        self.sorted_ = [np.sort(St[:, k]) for k in range(St.shape[1])]
        Sc = TrajectoryConformal.scores(L_cal, H_cal, y_cal)
        G = np.column_stack([np.searchsorted(self.sorted_[k], Sc[:, k], side="right") / len(self.sorted_[k])
                             for k in range(Sc.shape[1])])
        self.q_ = conformal_quantile(G.max(1), self.alpha)
        # c_k = sup{s : G_k(s) <= q}: the smallest tuning score whose ECDF value exceeds q. Closed intervals
        # [L - c_k, H + c_k] then contain the set {G_k(s_k) <= q}, so Proposition 1 applies.
        def _cut(srt):
            if not np.isfinite(self.q_):
                return np.inf
            j = int(round(self.q_ * len(srt)))
            return srt[j] if j < len(srt) else np.inf
        self.c_ = np.array([_cut(s) for s in self.sorted_])
        return self

    def predict(self, L, H):
        lo, hi = np.asarray(L, float) - self.c_, np.asarray(H, float) + self.c_
        if self.nested:
            lo, hi = np.maximum.accumulate(lo, axis=1), np.minimum.accumulate(hi, axis=1)
        return lo, hi
