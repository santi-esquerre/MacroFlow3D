"""Reported estimands per amplitude (G4): alpha_TP, alpha_TAP (displacement and position bases), alpha_LP (with U_r), the halves of
the window as intervals (no claim), the far window and its ratio on distance arms, first-order references from the
pre-registration.  Sampling intervals: two-sided bootstrap-t at ``levels.interval`` (the procedure SF-42 qualified); Student and BCa
are reported next to it (BCa never a verdict).
"""
import numpy as np

from mcstat.intervals import bca_interval, bootstrap_t_interval, student_interval

from .gate4 import seq_rng

TRANSVERSE = ("alpha_TP", "alpha_TAP")


def mean_summary(x, cfg, label):
    """Mean, sd, n and the three 90 % (``levels.interval``) intervals of a 1-D sample."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    lv, bs = cfg["levels"]["interval"], cfg["bootstrap"]
    out = {"n": int(x.size), "mean": float(x.mean()) if x.size else float("nan")}
    if x.size < 3:
        return dict(out, sd=float("nan"), student=None, bootstrap_t=None, bca=None, halfwidth_bootstrap_t=float("nan"))
    st = student_interval(x, lv)
    bt = bootstrap_t_interval(x, lv, "two", bs["B_sampling"], rng=seq_rng(bs["seed"], "est", label))
    bc = bca_interval(x, lv, "two", bs["B_bca"], seed=int(bs["seed"]) % (2 ** 31))
    return dict(out, sd=float(x.std(ddof=1)), student=[float(st.lo), float(st.hi)], bootstrap_t=[float(bt.lo), float(bt.hi)],
                bca=None if bc.degenerate else [bc.lo, bc.hi], halfwidth_bootstrap_t=0.5 * float(bt.hi - bt.lo))


def _ratio_interval(num, den, cfg, label):
    """Ratio of means with a paired seed-bootstrap percentile interval (pairs with a non-finite member dropped)."""
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    if num.size < 3 or den.mean() == 0.0:
        return {"n": int(num.size), "ratio": float("nan"), "percentile": None}
    bs = cfg["bootstrap"]
    rng = seq_rng(bs["seed"], "ratio", label)
    idx = rng.integers(0, num.size, size=(bs["B_sampling"], num.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    lo, hi = np.nanpercentile(r, [50 * (1 - cfg["levels"]["interval"]), 50 * (1 + cfg["levels"]["interval"])])
    return {"n": int(num.size), "ratio": float(num.mean() / den.mean()), "percentile": [float(lo), float(hi)]}


def amplitude_estimands(prereg, amp, arms):
    """Dictionary of the reported estimands of one amplitude (production arm; distance arm when configured)."""
    c, cfg = prereg.amp(amp), prereg.raw
    prim, bases = cfg["primary_window"], cfg["bases"]
    P = arms[c["production"]]
    ref = cfg.get("first_order", {}).get(amp, {})
    out = {"production_arm": P.id, "N": P.N, "status": P.info["status"]}
    if P.N == 0:
        return out
    est = {}
    for e in TRANSVERSE:
        est[e] = {}
        for role, b in bases.items():
            s = mean_summary(P.get(prim, b, e), cfg, "%s/%s/%s/%s" % (amp, e, b, prim))
            s["role"] = role
            if e in ref:
                s["first_order"], s["ratio_to_first_order"] = ref[e], s["mean"] / ref[e] if ref[e] else float("nan")
            est[e][b] = s
    # alpha_LP: the estimand is the mean of per-realization values; the (Ubar^2/2) x mean-slope form is reported next to it
    b = bases["primary"]
    aL, U = P.get(prim, b, "alpha_LP"), P.get(prim, b, "U_r")
    lp = mean_summary(aL, cfg, "%s/alpha_LP/%s" % (amp, prim))
    Ubar = float(U.mean())
    lp["U_r"] = {"mean": Ubar, "sd": float(U.std(ddof=1)) if U.size > 1 else float("nan"),
                 "cv": float(U.std(ddof=1) / Ubar) if U.size > 1 and Ubar else float("nan")}
    lp["Ubar2_over2_x_mean_slope"] = float(Ubar ** 2 * np.mean(aL / U ** 2))  # alpha_LP = (U^2/2) slope/ell  =>  slope/ell = 2 alpha/U^2
    if "alpha_LP" in ref:
        lp["first_order"], lp["ratio_to_first_order"] = ref["alpha_LP"], lp["mean"] / ref["alpha_LP"] if ref["alpha_LP"] else float("nan")
    est["alpha_LP"] = lp
    out["estimators"] = est
    halves = {}
    for w in ("h1", "h2"):
        if w in cfg["windows"]:
            halves[w] = {"window": cfg["windows"][w]}
            for e in TRANSVERSE + ("alpha_LP",):
                for b in ([bases["primary"], bases["secondary"]] if e in TRANSVERSE else [bases["primary"]]):
                    x = P.get(w, b, e)
                    xf = x[np.isfinite(x)]
                    iv = student_interval(xf, cfg["levels"]["interval"]) if xf.size > 2 else None
                    halves[w]["%s/%s" % (e, b)] = {"n": int(xf.size), "mean": float(xf.mean()) if xf.size else float("nan"),
                                                   "student": None if iv is None else [float(iv.lo), float(iv.hi)]}
    out["halves_no_claim"] = halves
    if c.get("distance_arm") and "far" in cfg["windows"]:
        D = arms[c["distance_arm"]]
        far = {"arm": D.id, "N": D.N, "window": cfg["windows"]["far"], "estimators": {}, "ratio_far_to_main": {}}
        if D.N >= 3:
            for e in TRANSVERSE + ("alpha_LP",):
                for b in ([bases["primary"], bases["secondary"]] if e in TRANSVERSE else [bases["primary"]]):
                    k = "%s/%s" % (e, b)
                    far["estimators"][k] = mean_summary(D.get("far", b, e), cfg, "%s/dist/%s/far" % (amp, k))
                    far["ratio_far_to_main"][k] = _ratio_interval(D.get("far", b, e), D.get(prim, b, e), cfg, "%s/dist/%s" % (amp, k))
        out["distance"] = far
    return out
