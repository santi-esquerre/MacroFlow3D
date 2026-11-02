"""Secondary (descriptive) analyses of SF-43 (specification item 1 Secondary, item 9 and decision G9b); none enters the primary family.

1. ``time_window``  SF-39-style time-window estimators (``family=time`` rows of ``sf42red``: window ``[0.75, 1] t_end,r`` ending at each
   realization's first exit; fixed-axis ``alpha_LAM``, ``alpha_Tfixed``, aligned ``alpha_TAM``): per arm mean, sd and 90 % Student interval
   over the realizations.  B&dD-style window: with the column ``t_end_adv`` (``t_end,r`` in advection times ``lambda / U``) the measured
   ``alpha_LAM`` is compared with B&dD's eq. 6 averaged over the same window,
   ``(1 / (0.25 t_end)) integral_{0.75 t_end}^{t_end} f_L(t) dt`` with the Table 1 parameters of the amplitude (ratio of means measured /
   predicted, with the Student interval of the per-realization ratio).
2. ``expI``         weight sensitivity (``exp(-I)`` weights on the same paths; not a comparator): for each base column (``alpha_TP``,
   ``alpha_LP``) the paired contrast of ``<col>:expI`` against ``<col>`` of one arm (log-ratio of means, delta-method paired Student
   interval of ``mcstat.gate5.gate5_paired``, and the mean paired difference).
3. ``par2_transverse``  Par2's transverse dispersivity in the matched configuration (window injection): ``alpha_TP`` (position basis) and
   ``alpha_TP:disp`` (displacement basis) of the Par2 arm, mean and 90 % Student interval, and the paired log-ratio against the consumer
   twin on the same seeds, for both bases.
4. ``bdd_fit``      fit of B&dD eq. 6, ``f_L(t) = a_LAF (1 - exp(-t / t_LF))^a_LF``, by nonlinear least squares to the pooled points
   ``(t_ij, alpha_L,ij)`` of the realizations of an arm (``curve_t``, ``curve_alpha``: arrays (n, K), NaN allowed; ``t`` in advection
   times times ``t_scale``), several starts (data-driven and the Table 1 values of the amplitude), box bounds
   ``a_LAF in [1e-6, 1e4]``, ``t_LF in [1e-3, 1e4]``, ``a_LF in [0.05, 3]``; bootstrap over seeds (rows resampled with replacement,
   refit from the full fit; ``B`` resamples, percentile 90 % intervals, failed refits counted).  The gap to Table 1 (0.24 / 1.27 / 4.35 /
   12.7, 1.35 / 2.21 / 4.83 / 20.5, 1 / 0.80 / 0.63 / 0.49 at ``sigma^2`` 0.25 / 1 / 2.25 / 4) is recorded as difference and ratio.
   With few distinct times per arm the three parameters are weakly identified: the bootstrap intervals say so.

``run(prereg, data)`` -- ``prereg["secondaries"]`` (every block optional)::

    level (0.90), seed (20261103), B (1000), t_scale (1.0),
    time_window: {arm: {"amplitude": amp}},  expI: {name: {"arm": arm}},
    par2_transverse: {amp: {"par2": arm, "consumer": arm}},
    bdd_fit: [{"name": str, "arm": arm, "amplitude": amp, "t_key": "curve_t", "alpha_key": "curve_alpha"}]

``data``: ``{"arms": {arm: {"seed": int array, "cols": {name: float array (n,) or (n, K)}}}}`` as ``sf43ana.gate5``.
"""
from __future__ import annotations

import zlib

from sf43cov import mcpath  # noqa: F401

import numpy as np
from scipy import optimize as _opt
from scipy import stats as _st

from mcstat import gate5 as _mg5

from .gate5 import arm_series, jsonable, pair_by_seed

# B&dD 2013 Table 1, longitudinal fit: amplitude -> (alpha_LAF, t_LF, a_LF)  (docs/theory/beaudoin-de-dreuzy-2013-key-claims.md)
BDD_TABLE = {"s025": (0.24, 1.35, 1.0), "s1": (1.27, 2.21, 0.80), "s225": (4.35, 4.83, 0.63), "s4": (12.7, 20.5, 0.49)}
WINDOW = (0.75, 1.0)
TIME_KEYS = ("alpha_LAM:time", "alpha_Tfixed:time", "alpha_TAM:time")
LOWER = np.array([1e-6, 1e-3, 0.05])
UPPER = np.array([1e4, 1e4, 3.0])


def bdd_f(t, a, tl, p):
    """B&dD eq. 6: ``a (1 - exp(-t / tl))^p``."""
    return a * (-np.expm1(-np.asarray(t, dtype=float) / tl)) ** p


def bdd_window_average(amp, t_end, window=WINDOW, npts=400):
    """Mean of eq. 6 (Table 1 parameters of ``amp``) over ``t in [window[0], window[1]] * t_end`` (midpoint rule); arrays broadcast."""
    a, tl, p = BDD_TABLE[amp]
    t_end = np.asarray(t_end, dtype=float)
    u = window[0] + (np.arange(npts) + 0.5) * (window[1] - window[0]) / npts
    return bdd_f(t_end[..., None] * u, a, tl, p).mean(axis=-1)


def _student(x, level):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return {"n": int(n), "mean": float(x.mean()) if n else float("nan"), "sd": float("nan"), "lo": float("nan"), "hi": float("nan")}
    sd = float(x.std(ddof=1))
    h = float(_st.t.isf(0.5 * (1.0 - level), n - 1)) * sd / np.sqrt(n)
    return {"n": int(n), "mean": float(x.mean()), "sd": sd, "lo": float(x.mean() - h), "hi": float(x.mean() + h)}


def _paired_log_ratio(a, b, level):
    """Delta-method paired log-ratio of means ``a / b`` (arrays aligned) or a note."""
    if a.size < 3 or not (a.mean() > 0 and b.mean() > 0):
        return {"note": "fewer than 3 pairs or a non-positive mean", "n": int(a.size)}
    r = _mg5.gate5_paired(a, b, level=level)
    return {"ratio": float(r.ratio), "lo": float(r.ratio_lo), "hi": float(r.ratio_hi), "log_ratio": float(r.theta), "se_log": float(r.se),
            "n": int(a.size), "mean_diff": float((a - b).mean())}


# ---------------------------------------------------------------------------------------------------------------- 1. time window
def time_window(cfg, spec, data):
    out = {}
    for arm, s in spec.items():
        res = {}
        for key in TIME_KEYS:
            try:
                _, v = arm_series(data, arm, key)
            except KeyError:
                continue
            res[key] = _student(v, cfg["level"])
        amp = s.get("amplitude")
        try:
            _, te = arm_series(data, arm, "t_end_adv")
            res["t_end_adv"] = _student(te, cfg["level"])
            if amp in BDD_TABLE and "alpha_LAM:time" in res:
                _, al = arm_series(data, arm, "alpha_LAM:time")
                ok = np.isfinite(al) & np.isfinite(te) & (te > 0)
                pred = bdd_window_average(amp, te[ok])
                res["bdd_window"] = {"predicted": _student(pred, cfg["level"]), "ratio_of_means": float(al[ok].mean() / pred.mean()),
                                     "ratio_per_realization": _student(al[ok] / pred, cfg["level"]), "table": dict(zip(("a_LAF", "t_LF", "a_LF"), BDD_TABLE[amp]))}
        except KeyError:
            res["bdd_window"] = {"note": "column t_end_adv absent: B&dD window comparison not computed"}
        out[arm] = res
    return out


# ---------------------------------------------------------------------------------------------------------------- 2. expI
def exp_i(cfg, spec, data):
    out = {}
    for name, s in spec.items():
        res = {}
        for base in ("alpha_TP", "alpha_LP"):
            try:
                sa, va = arm_series(data, s["arm"], base + ":expI")
                sb, vb = arm_series(data, s["arm"], base)
            except KeyError:
                continue
            _, a, b, info = pair_by_seed(sa, va, sb, vb)
            res[base] = dict(_paired_log_ratio(a, b, cfg["level"]), pairing=info, contrast="%s:expI / %s" % (base, base))
        out[name] = res
    return out


# ---------------------------------------------------------------------------------------------------------------- 3. Par2 transverse
def par2_transverse(cfg, spec, data):
    out = {}
    for amp, s in spec.items():
        res = {}
        for basis, key in (("position", "alpha_TP"), ("displacement", "alpha_TP:disp")):
            try:
                sp, vp = arm_series(data, s["par2"], key)
            except KeyError:
                res[basis] = {"note": "column %s absent in the Par2 arm" % key}
                continue
            blk = {"par2": _student(vp, cfg["level"])}
            if s.get("consumer"):
                try:
                    sc, vc = arm_series(data, s["consumer"], key)
                    _, a, b, info = pair_by_seed(sp, vp, sc, vc)
                    blk["par2_over_consumer"] = dict(_paired_log_ratio(a, b, cfg["level"]), pairing=info)
                except KeyError as e:
                    blk["par2_over_consumer"] = {"note": e.args[0]}
            res[basis] = blk
        out[amp] = res
    return out


# ---------------------------------------------------------------------------------------------------------------- 4. eq. 6 fit
def _starts(t, y, amp):
    ymax, tm = max(float(np.nanmax(y)), 1e-3), max(float(np.nanmedian(t)), 1e-2)
    st = [(1.3 * ymax, tm, 0.8), (2.0 * ymax, 3.0 * tm, 0.6), (ymax, 0.5 * tm, 1.0)]
    if amp in BDD_TABLE:
        st.append(BDD_TABLE[amp])
    return [np.clip(np.array(s, dtype=float), LOWER * 1.0001, UPPER * 0.9999) for s in st]


def fit_bdd(t, y, amp=None, x0=None):
    """Nonlinear least squares of eq. 6 to the points ``(t, y)`` (1-D, finite).  Returns ``(params, rmse)`` or ``(None, nan)``."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if t.size < 3:
        return None, float("nan")
    best, cost = None, np.inf
    starts = [np.clip(np.asarray(x0, dtype=float), LOWER * 1.0001, UPPER * 0.9999)] if x0 is not None else []
    for x in starts + _starts(t, y, amp):
        try:
            r = _opt.least_squares(lambda q: bdd_f(t, *q) - y, x, bounds=(LOWER, UPPER), x_scale=np.maximum(np.abs(x), 1e-3), method="trf")
        except (ValueError, FloatingPointError):
            continue
        if r.cost < cost:
            best, cost = r.x, r.cost
    if best is None:
        return None, float("nan")
    return best, float(np.sqrt(2.0 * cost / t.size))


def bdd_fit(cfg, item, data):
    amp = item.get("amplitude")
    _, T = arm_series(data, item["arm"], item.get("t_key", "curve_t"))
    _, Y = arm_series(data, item["arm"], item.get("alpha_key", "curve_alpha"))
    T, Y = np.atleast_2d(T.T).T * cfg["t_scale"], np.atleast_2d(Y.T).T
    ok = np.isfinite(T) & np.isfinite(Y) & (T > 0)
    n = T.shape[0]
    p0, rmse = fit_bdd(T[ok], Y[ok], amp)
    if p0 is None:
        return {"note": "the fit failed or fewer than 3 points", "n_points": int(ok.sum())}
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], zlib.crc32(item.get("name", item["arm"]).encode())]))
    boots, failed = [], 0
    for _ in range(cfg["B"]):
        idx = rng.integers(0, n, size=n)
        m = ok[idx]
        q, _r = fit_bdd(T[idx][m], Y[idx][m], amp, x0=p0)
        if q is None:
            failed += 1
        else:
            boots.append(q)
    names = ("a_LAF", "t_LF", "a_LF")
    out = {"n_seeds": int(n), "n_points": int(ok.sum()), "distinct_times": int(np.unique(np.round(T[ok], 9)).size), "rmse": rmse,
           "params": dict(zip(names, map(float, p0))), "bootstrap": {"B": cfg["B"], "failed": failed}}
    if boots:
        bq = np.array(boots)
        lo, hi = np.quantile(bq, [0.5 * (1.0 - cfg["level"]), 0.5 * (1.0 + cfg["level"])], axis=0)
        out["bootstrap"].update({nm: {"lo": float(lo[i]), "hi": float(hi[i]), "at_bound": float(np.mean((bq[:, i] <= LOWER[i] * 1.01) | (bq[:, i] >= UPPER[i] * 0.99)))}
                                 for i, nm in enumerate(names)})
    if amp in BDD_TABLE:
        tab = dict(zip(names, BDD_TABLE[amp]))
        out["table1"] = tab
        out["gap"] = {nm: {"difference": float(p0[i] - tab[nm]), "ratio": float(p0[i] / tab[nm])} for i, nm in enumerate(names)}
    return out


def run(prereg, data):
    p = prereg.get("secondaries", {})
    cfg = {"level": float(p.get("level", 0.90)), "seed": int(p.get("seed", 20261103)), "B": int(p.get("B", 1000)),
           "t_scale": float(p.get("t_scale", 1.0))}
    out = {"config": cfg}
    if "time_window" in p:
        out["time_window"] = time_window(cfg, p["time_window"], data)
    if "expI" in p:
        out["expI"] = exp_i(cfg, p["expI"], data)
    if "par2_transverse" in p:
        out["par2_transverse"] = par2_transverse(cfg, p["par2_transverse"], data)
    if "bdd_fit" in p:
        out["bdd_fit"] = {it.get("name", it["arm"]): bdd_fit(cfg, it, data) for it in p["bdd_fit"]}
    return jsonable(out)
