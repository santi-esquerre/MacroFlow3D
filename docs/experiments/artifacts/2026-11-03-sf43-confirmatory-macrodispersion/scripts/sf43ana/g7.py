"""G7 of SF-43 (decision G7): attribution of a Gate 5 offset.  Descriptive, outside the primary family.

1. FREE DECOMPOSITION, per amplitude, paired by seed: consumer twin (uniform weights), Pollock twin (the consumer's Pollock-bridge plane
   outputs; Par2's face-centred linear interpolation is the bridge's velocity) and Par2.  Consumer-vs-Pollock isolates the velocity
   representation; Pollock-vs-Par2 isolates Euler stepping plus random-vs-lattice injection.  The three pairwise log-ratio contrasts of
   the means (``log(mean A / mean B)``, ``A/B`` in consumer/pollock, pollock/par2, consumer/par2) with the paired delta-method Student
   interval of ``mcstat.gate5.gate5_paired`` (``level``), for each response (``alpha_LP`` and ``alpha_TP`` by default).
2. FACTORIAL at ``sigma^2`` 0.25 and 1: the Par2 cells of the ``2^(4-1)`` resolution-IV fraction ``I = GIDR`` (G generator: SF-18
   against Par2 random-spectral, whole plot, unpaired; I injection: full plane against window; D time step ``dt`` against ``dt / 2``;
   R ``cg_rtol`` 1e-6 against 1e-10; the within-seed factors paired by seed), analysed with ``mcstat.splitplot.analyze``: effects on the
   log of cell means (bootstrap over seeds), within-seed contrasts by one-sample ``t``, G and its interactions by Welch with a two-stratum
   variance; aliases are reported in pairs (``GI = DR``, ``GD = IR``, ``GR = ID``, main effects with three-factor interactions).

``run(prereg, data)`` -- ``prereg["g7"]``::

    level (0.90), log_bootstrap (2000), seed (20261103), responses (["alpha_LP", "alpha_TP"]),
    twins:     {amp: {"consumer": arm, "pollock": arm, "par2": arm}}
    factorial: {amp: [{"g": +-1, "i": +-1, "d": +-1, "r": +-1, "arm": arm}, ...]}   (the 8 cells with g i d r = +1; g = +1 is SF-18)

``data``: ``{"arms": {arm: {"seed": int array, "cols": {name: float array (n,)}}}}`` (as ``sf43ana.gate5``).  Seeds with a NaN in any
cell of a generator are dropped from that generator (counted).  Returns ``{"twins": {amp: {response: {...}}}, "factorial": {amp:
{response: {...}}}}``.
"""
from __future__ import annotations

from sf43cov import mcpath  # noqa: F401

import numpy as np

from mcstat import gate5 as _mg5
from mcstat import splitplot as _sp

from .gate5 import arm_series, jsonable, pair_by_seed

PAIRS = (("consumer", "pollock"), ("pollock", "par2"), ("consumer", "par2"))
MEANING = {("consumer", "pollock"): "velocity representation", ("pollock", "par2"): "Euler stepping + random-vs-lattice injection",
           ("consumer", "par2"): "total"}


def _contrast(a, b, level):
    r = _mg5.gate5_paired(a, b, level=level)
    return {"log_ratio": float(r.theta), "ratio": float(r.ratio), "ratio_lo": float(r.ratio_lo), "ratio_hi": float(r.ratio_hi),
            "se_log": float(r.se), "dof": int(r.dof), "n": int(a.size), "mean_A": float(a.mean()), "mean_B": float(b.mean())}


def twins(cfg, spec, data, responses):
    out = {}
    for resp in responses:
        res = {}
        for (a, b) in PAIRS:
            name = "%s/%s" % (a, b)
            try:
                sa, va = arm_series(data, spec[a], resp)
                sb, vb = arm_series(data, spec[b], resp)
            except KeyError as e:
                res[name] = {"note": "unavailable: %s" % (e.args[0],)}
                continue
            _, x, y, info = pair_by_seed(sa, va, sb, vb)
            if x.size < 3 or not (x.mean() > 0 and y.mean() > 0):
                res[name] = {"note": "fewer than 3 pairs or a non-positive mean", "pairing": info}
                continue
            res[name] = dict(_contrast(x, y, cfg["level"]), meaning=MEANING[(a, b)], pairing=info)
        out[resp] = res
    return out


def _generator_data(cells, data, resp):
    """-> ({g: {(i, d, r): y (n_g,)}}, info) with the cells of one generator aligned on their common finite seeds."""
    by_g = {1: {}, -1: {}}
    for c in cells:
        g = int(c["g"])
        key = (int(c["i"]), int(c["d"]), int(c["r"]))
        if int(c["g"]) * key[0] * key[1] * key[2] != 1:
            raise ValueError("cell %r is not in the fraction I = GIDR (g i d r must be +1)" % (c,))
        s, v = arm_series(data, c["arm"], resp)
        by_g[g][key] = (s, v)
    out, info = {}, {}
    for g, cd in by_g.items():
        want = _sp.cells_of(g)
        if sorted(cd) != sorted(want):
            raise ValueError("generator %+d needs the cells %s, got %s" % (g, want, sorted(cd)))
        common = None
        for s, v in cd.values():
            ok = set(int(x) for x, y in zip(s, v) if np.isfinite(y))
            common = ok if common is None else common & ok
        seeds = sorted(common)
        out[g] = {}
        for k, (s, v) in cd.items():
            idx = {int(x): i for i, x in enumerate(s)}
            out[g][k] = np.array([v[idx[x]] for x in seeds], dtype=float)
        info[g] = {"n_seeds": len(seeds), "n_arm_seeds": {str(k): int(len(s)) for k, (s, _) in cd.items()}}
    return out, info


def factorial(cfg, cells, data, responses):
    out = {}
    for resp in responses:
        try:
            gd, info = _generator_data(cells, data, resp)
            if min(i["n_seeds"] for i in info.values()) < 2:
                out[resp] = {"note": "fewer than 2 complete seeds in a generator", "seeds": info}
                continue
            res = _sp.analyze(gd, level=cfg["level"], log_bootstrap=cfg["log_bootstrap"], seed=cfg["seed"])
        except (KeyError, ValueError) as e:
            out[resp] = {"note": "not analysed: %s" % (e.args[0] if e.args else e,)}
            continue
        res = dict(res)
        res["simple"] = {"g=%+d,%s" % k: v._asdict() for k, v in res["simple"].items()}
        res["cells"] = {"g=%+d,i=%+d,d=%+d,r=%+d" % k: v for k, v in res["cells"].items()}
        res["design"] = ["g=%+d,i=%+d,d=%+d,r=%+d" % c for c in res["design"]]
        res["seeds"] = info
        out[resp] = res
    return out


KEYS = {"level", "log_bootstrap", "seed", "responses", "twins", "factorial"}


def analyse(ctx):
    if not ctx.prereg.raw.get("g7"):
        return None
    return run(ctx.prereg.raw, ctx.data)


def run(prereg, data):
    p = prereg["g7"]
    cfg = {"level": float(p.get("level", 0.90)), "log_bootstrap": int(p.get("log_bootstrap", 2000)), "seed": int(p.get("seed", 20261103))}
    responses = list(p.get("responses", ["alpha_LP", "alpha_TP"]))
    out = {"config": dict(cfg, responses=responses), "twins": {}, "factorial": {}}
    for amp, spec in p.get("twins", {}).items():
        out["twins"][amp] = twins(cfg, spec, data, responses)
    for amp, cells in p.get("factorial", {}).items():
        out["factorial"][amp] = factorial(cfg, cells, data, responses)
    return jsonable(out)
