"""Pollock comparator of SF-43 (specification acceptance B, item 8, decision G5.5 / G7).  REPORTED; outside the error budget and the Holm family.

Every consumer realization also runs the Pollock bridge (``pollock_bridge: true``): the same particles are traced on the face-centred
linear interpolation of the same flux field.  ``sf42red`` writes those plane moments as rows ``code = pollock`` with the columns, bases,
weights and sets of the consumer plane rows and the same seed and ``hmax_div`` label; the pre-registration lists them as arm entries with
``code: pollock`` and ``source`` = the engine arm that produced them.

1. ``contrast`` (acceptance B), per amplitude: for the production analysis set (flux weights) and every plane estimand (``alpha_TP``,
   ``alpha_TAP``, ``alpha_LP``, ``U_r``) in BOTH bases (displacement primary, position secondary): the PAIRED difference ``consumer -
   Pollock`` (mean, 90 % Student interval, 90 % bootstrap-t interval over the seeds) and the ratio of means ``consumer / Pollock`` (paired
   seed-bootstrap percentile interval).  What the contrast measures is the velocity representation (the consumer's spline
   field against the Pollock face-flux interpolation); it decides nothing.
2. ``decomposition`` (G5.5 / item 8), per amplitude on the Gate 5 pairs (consumer twin and Pollock twin with uniform weights, Par2): the
   pairwise log-ratio contrasts consumer/Pollock (velocity representation), Pollock/Par2 (Euler stepping plus random against lattice
   injection) and consumer/Par2 (total), each response paired by seed, with the delta-method Student interval of ``mcstat.gate5``
   (``g7.twins``).

``run(prereg, data)`` -- ``prereg["pollock"]``::

    level (0.90), B (2000), seed (20261103), responses (["alpha_LP", "alpha_TP:disp", "alpha_TP", "alpha_TAP:disp", "alpha_TAP"]),
    estimands (["alpha_TP", "alpha_TAP", "alpha_LP", "U_r"]),
    contrast:      {amp: {"consumer": arm, "pollock": arm}}
    decomposition: {amp: {"consumer": arm, "pollock": arm, "par2": arm}}

``data``: ``sf43ana.loader.build_data``.
"""
from __future__ import annotations

import numpy as np

from mcstat.intervals import bootstrap_t_interval, student_interval

from .g7 import twins
from .gate4 import seq_rng
from .gate5 import arm_series, jsonable, pair_by_seed

KEYS = {"level", "B", "seed", "responses", "estimands", "contrast", "decomposition"}
LABEL = "reported; outside the error budget and the Holm family"


def column(est, basis):
    """Column name of the loader's data contract: the position basis has the bare name."""
    return est if basis == "pos" else "%s:%s" % (est, basis)


def paired_contrast(c, p, level, B, rng):
    """Paired difference ``c - p`` and ratio of means of two aligned finite samples."""
    n = int(c.size)
    d = c - p
    out = {"n": n, "mean_consumer": float(c.mean()), "mean_pollock": float(p.mean()), "difference": float(d.mean())}
    if n < 3:
        return dict(out, note="fewer than 3 pairs")
    sd = float(d.std(ddof=1))
    out["sd_difference"] = sd
    if sd == 0.0:
        out.update(student=[float(d.mean())] * 2, bootstrap_t=[float(d.mean())] * 2, note="identical paired values: degenerate intervals")
    else:
        st = student_interval(d, level)
        bt = bootstrap_t_interval(d, level, "two", B, rng=rng)
        out.update(student=[float(st.lo), float(st.hi)], bootstrap_t=[float(bt.lo), float(bt.hi)])
    if p.mean() != 0.0:
        idx = rng.integers(0, n, size=(B, n))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = c[idx].mean(axis=1) / p[idx].mean(axis=1)
        a = 0.5 * (1.0 - level)
        lo, hi = np.nanquantile(r, [a, 1.0 - a])
        out.update(ratio=float(c.mean() / p.mean()), ratio_percentile=[float(lo), float(hi)])
    return out


def contrast(cfg, spec, data, bases):
    res = {}
    for est in cfg["estimands"]:
        for b in bases:
            col = column(est, b)
            key = "%s/%s" % (est, b)
            try:
                sc, vc = arm_series(data, spec["consumer"], col)
                sp, vp = arm_series(data, spec["pollock"], col)
            except KeyError as e:
                res[key] = {"note": "unavailable: %s" % (e.args[0],)}
                continue
            _, x, y, info = pair_by_seed(sc, vc, sp, vp)
            if x.size == 0:
                res[key] = {"note": "no paired seeds (the Pollock rows are absent or the arms share none)", "pairing": info}
                continue
            res[key] = dict(paired_contrast(x, y, cfg["level"], cfg["B"], seq_rng(cfg["seed"], "pollock", spec["consumer"], key)), pairing=info)
    return res


def run(prereg, data):
    p = prereg["pollock"]
    cfg = {"level": float(p.get("level", 0.90)), "B": int(p.get("B", 2000)), "seed": int(p.get("seed", 20261103)),
           "estimands": list(p.get("estimands", ["alpha_TP", "alpha_TAP", "alpha_LP", "U_r"]))}
    responses = list(p.get("responses", ["alpha_LP", "alpha_TP:disp", "alpha_TP", "alpha_TAP:disp", "alpha_TAP"]))
    bases = [prereg["bases"]["primary"], prereg["bases"]["secondary"]]
    out = {"label": LABEL, "config": dict(cfg, responses=responses, bases=bases, window=prereg["primary_window"]),
           "contrast": {}, "decomposition": {}}
    for amp, spec in p.get("contrast", {}).items():
        out["contrast"][amp] = contrast(cfg, spec, data, bases)
    for amp, spec in p.get("decomposition", {}).items():
        out["decomposition"][amp] = twins({"level": cfg["level"]}, spec, data, responses)
    return jsonable(out)


def analyse(ctx):
    if not ctx.prereg.raw.get("pollock"):
        return None
    return run(ctx.prereg.raw, ctx.data)
