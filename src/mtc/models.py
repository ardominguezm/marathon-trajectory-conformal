"""Base predictors. All are wrapped by split-conformal calibration downstream."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import checkpoint_features, naive_extrapolation

LGB_DEFAULTS = dict(n_estimators=250, learning_rate=0.06, num_leaves=63, min_child_samples=100,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=2)


def _subsample(df: pd.DataFrame, max_train: int | None, seed: int) -> pd.DataFrame:
    if max_train is None or len(df) <= max_train:
        return df
    return df.iloc[np.random.RandomState(seed).permutation(len(df))[:max_train]]


def quantile_trajectories(train, targets: dict, n_segs, alpha: float, seed: int = 0,
                          max_train: int | None = None, lgb_kw: dict | None = None, target: str = "residual"):
    """Fit LightGBM lower/upper quantile models per checkpoint on `train`; predict on every frame in `targets`.

    Returns ({name: (L, H)} with arrays of shape (n_name, len(n_segs)), train_width (len(n_segs),)).
    train_width is the median in-sample interval width, a fixed scale that does not use calibration data.

    target="residual" models y minus the constant-pace projection of the checkpoint (the interval is shifted back
    afterwards); target="absolute" models the finish time directly, as in the submitted CP-PRO. Both are valid after
    conformal calibration; the residual parametrisation is much sharper at late checkpoints because trees cannot
    extrapolate the near-deterministic relation between the current split and the finish time.
    """
    if target not in {"residual", "absolute"}:
        raise ValueError(target)
    import lightgbm as lgb

    kw = {**LGB_DEFAULTS, **(lgb_kw or {}), "random_state": seed}
    tr = _subsample(train, max_train, seed)
    y = tr["finish_time_min"].to_numpy()
    out = {k: (np.zeros((len(v), len(n_segs))), np.zeros((len(v), len(n_segs)))) for k, v in targets.items()}
    width = np.zeros(len(n_segs))
    ref = tr.iloc[:min(20000, len(tr))]
    for j, n in enumerate(n_segs):
        off = (lambda d: naive_extrapolation(d, n)) if target == "residual" else (lambda d: 0.0)
        X = checkpoint_features(tr, n)
        yt_ = y - off(tr)
        mL = lgb.LGBMRegressor(objective="quantile", alpha=alpha / 2, **kw).fit(X, yt_)
        mH = lgb.LGBMRegressor(objective="quantile", alpha=1 - alpha / 2, **kw).fit(X, yt_)
        Xr = checkpoint_features(ref, n)
        width[j] = np.median(mH.predict(Xr) - mL.predict(Xr))
        for name, df in targets.items():
            Xt = checkpoint_features(df, n)
            o = off(df)
            out[name][0][:, j], out[name][1][:, j] = mL.predict(Xt) + o, mH.predict(Xt) + o
    return out, width


class _Hetero:
    """Mean model plus a scale model fitted to absolute residuals; scores are |y - mu| / sigma."""

    def predict(self, X):
        mu = self.mean_predict(X)
        return mu, np.maximum(self.scale_predict(X), 0.25)


class LGBMHetero(_Hetero):
    def __init__(self, seed=0, **kw):
        self.kw = {**LGB_DEFAULTS, **kw, "random_state": seed}

    def fit(self, X, y):
        import lightgbm as lgb
        self.m = lgb.LGBMRegressor(**self.kw).fit(X, y)
        self.s = lgb.LGBMRegressor(**self.kw).fit(X, np.abs(y - self.m.predict(X)))
        return self

    def mean_predict(self, X): return self.m.predict(X)
    def scale_predict(self, X): return self.s.predict(X)


class RFHetero(_Hetero):
    """Random-forest ensemble; the scale model uses out-of-bag residuals (honest, unlike in-sample)."""

    def __init__(self, seed=0, n_estimators=80):
        self.seed, self.n = seed, n_estimators

    def fit(self, X, y):
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestRegressor
        self.m = RandomForestRegressor(n_estimators=self.n, min_samples_leaf=10, max_features=0.5, oob_score=True,
                                       n_jobs=2, random_state=self.seed).fit(X, y)
        self.s = lgb.LGBMRegressor(**{**LGB_DEFAULTS, "n_estimators": 120, "random_state": self.seed}).fit(
            X, np.abs(y - self.m.oob_prediction_))
        return self

    def mean_predict(self, X): return self.m.predict(X)
    def scale_predict(self, X): return self.s.predict(X)


class MLPHetero(_Hetero):
    """Feed-forward network (scikit-learn) for the mean and a second one for the scale (heteroscedastic)."""

    def __init__(self, seed=0, hidden=(64, 64), max_iter=60):
        self.seed, self.hidden, self.max_iter = seed, hidden, max_iter

    def _net(self):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=self.hidden, early_stopping=True, max_iter=self.max_iter,
                            n_iter_no_change=6, random_state=self.seed)

    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        self.sx = StandardScaler().fit(X)
        Z = self.sx.transform(X)
        self.mu_, self.sd_ = float(y.mean()), float(y.std())
        self.m = self._net().fit(Z, (y - self.mu_) / self.sd_)
        r = np.abs(y - (self.m.predict(Z) * self.sd_ + self.mu_))
        self.rm_ = float(r.mean())
        self.s = self._net().fit(Z, r / self.rm_)
        return self

    def mean_predict(self, X): return self.m.predict(self.sx.transform(X)) * self.sd_ + self.mu_
    def scale_predict(self, X): return np.maximum(self.s.predict(self.sx.transform(X)), 0.02) * self.rm_


def fit_hetero_split_cp(model, Xtr, ytr, Xcal, ycal, Xte, alpha, off_tr=0.0, off_cal=0.0, off_te=0.0):
    """Locally weighted split CP: interval mu +/- q sigma with q the conformal quantile of |y - mu| / sigma.

    off_* are optional per-row offsets (e.g. the constant-pace projection): the model is fitted to y - off and the
    offset is added back to the predicted mean.
    """
    from .conformal import conformal_quantile
    model.fit(Xtr, ytr - off_tr)
    mu_c, sg_c = model.predict(Xcal)
    mu_c = mu_c + off_cal
    q = conformal_quantile(np.abs(ycal - mu_c) / sg_c, alpha)
    mu_t, sg_t = model.predict(Xte)
    mu_t = mu_t + off_te
    return mu_t - q * sg_t, mu_t + q * sg_t, mu_t
