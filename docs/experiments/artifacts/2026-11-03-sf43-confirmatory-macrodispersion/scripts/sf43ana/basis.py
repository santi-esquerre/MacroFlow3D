"""Basis contrasts (decision H2): position basis against displacement basis on the production analysis set.  Descriptive; never feeds a decision.

The primary estimators are taken in the DISPLACEMENT basis (``bases.primary``); the position basis (``bases.secondary``) is secondary.  Per amplitude:

1. ``flux`` -- the production arm (flux weights), every transverse estimator (``alpha_TP``, ``alpha_TAP``): the PAIRED difference
   ``position - displacement`` (mean, 90 % Student and bootstrap-t intervals over the seeds) and the ratio of means ``position / displacement``
   (paired seed-bootstrap percentile interval and the paired delta-method Student interval of ``mcstat.gate5``): ``pollock.paired_contrast``.
2. ``control_uniform`` -- the same contrasts on the UNIFORM-weight copy of the production set (the arm ``unif`` of the pre-registration:
   ``source`` = the production arm, ``selection.weights = unif``, the same ``hmax_div`` label and the same re-estimated size).  With
   uniform weights over the injection window and a transversely stationary medium, the displacement of a particle has the same law
   whatever its position, so the two bases estimate the same transverse growth and the expectation is a DIFFERENCE OF 0: ``interval_excludes_zero``
   (Student, bootstrap-t) and ``flag`` (either excludes 0) say whether the data are at odds with it.  A flag indicts the estimand pipeline or the
   transverse stationarity, not the physics; the control is descriptive and gates nothing.

``run(prereg, data)`` -- ``prereg["basis"]``::

    level (0.90), B (2000), seed (20261103), amplitudes: {amp: {"flux": <production arm>, "unif": <uniform-weight arm>}}
"""
from __future__ import annotations

from sf43cov import mcpath  # noqa: F401

from mcstat import gate5 as _mg5

from .gate4 import seq_rng
from .gate5 import arm_series, jsonable, pair_by_seed
from .pollock import column, paired_contrast

KEYS = {"level", "B", "seed", "amplitudes"}
AMP_KEYS = {"flux", "unif"}
ESTIMATORS = ("alpha_TP", "alpha_TAP")
EXPECTATION = "control_uniform: difference position - displacement = 0 (transverse stationarity, uniform weights); flag if the interval excludes 0"


def _contrast(cfg, arm, est, bases, tag):
    sec, prim = bases["secondary"], bases["primary"]
    try:
        s1, v1 = arm_series(cfg["data"], arm, column(est, sec))
        s2, v2 = arm_series(cfg["data"], arm, column(est, prim))
    except KeyError as e:
        return {"note": "unavailable: %s" % (e.args[0],)}
    _, x, y, info = pair_by_seed(s1, v1, s2, v2)
    if x.size == 0:
        return {"note": "no finite paired seeds", "pairing": info}
    out = paired_contrast(x, y, cfg["level"], cfg["B"], seq_rng(cfg["seed"], "basis", arm, est, tag))
    if x.size >= 3 and x.mean() > 0 and y.mean() > 0:  # Student interval of the ratio of means: the paired delta method of mcstat.gate5
        r = _mg5.gate5_paired(x, y, level=cfg["level"])
        out["ratio_student"] = [float(r.ratio_lo), float(r.ratio_hi)]
    out["pairing"] = info
    return out


def _excludes_zero(iv):
    return bool(iv is not None and (iv[0] > 0.0 or iv[1] < 0.0))


def run(prereg, data):
    p = prereg["basis"]
    cfg = {"level": float(p.get("level", 0.90)), "B": int(p.get("B", 2000)), "seed": int(p.get("seed", 20261103)), "data": data}
    bases = prereg["bases"]
    out = {"config": {"level": cfg["level"], "B": cfg["B"], "seed": cfg["seed"], "primary_basis": bases["primary"], "secondary_basis": bases["secondary"],
                      "difference": "%s - %s" % (bases["secondary"], bases["primary"]), "window": prereg["primary_window"]},
           "label": "reported; descriptive", "expectation": EXPECTATION, "amplitudes": {}}
    for amp, spec in p.get("amplitudes", {}).items():
        res = {"arms": dict(spec), "flux": {}, "control_uniform": {}}
        for est in ESTIMATORS:
            res["flux"][est] = _contrast(cfg, spec["flux"], est, bases, "flux")
            c = _contrast(cfg, spec["unif"], est, bases, "unif")
            if "difference" in c and c.get("n", 0) >= 3:
                st, bt = c.get("student"), c.get("bootstrap_t")
                c["interval_excludes_zero"] = {"student": _excludes_zero(st), "bootstrap_t": _excludes_zero(bt)}
                c["flag"] = bool(c["interval_excludes_zero"]["student"] or c["interval_excludes_zero"]["bootstrap_t"])
            res["control_uniform"][est] = c
        out["amplitudes"][amp] = res
    return jsonable(out)


def analyse(ctx):
    if not ctx.prereg.raw.get("basis"):
        return None
    return run(ctx.prereg.raw, ctx.data)
