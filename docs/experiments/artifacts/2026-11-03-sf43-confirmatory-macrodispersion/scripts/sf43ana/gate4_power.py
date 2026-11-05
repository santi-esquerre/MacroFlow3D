"""Achieved Gate 4 power at the REALIZED sizes (specification acceptance C).  Descriptive: it never feeds a decision.

Per amplitude: the probability that the robust sign rejects -- BOTH transverse estimators, intersection-union -- at the smallest Holm level
``fwer / m`` (the first step of the step-down) and at ``fwer`` (the last), when the true transverse growths are the pre-registered PLANNING
ALTERNATIVES ``alternatives[amp] = {alpha_TP, alpha_TAP}``.  The alternatives are the displacement-basis means of the SF-42 R3 noise pilot
(``docs/experiments/2026-10-30-sf42-confirmatory-tooling.md``, section "Estimator noise by basis (R3, no new runs)", table rows 1025-1032, "mean
disp"); they are DESIGN INPUTS, not claims about ``alpha_T``.  The actual Holm power lies between the two levels (it is the level of the step at
which the hypothesis is reached).

METHOD -- a port of ``gate4_power_sim.py`` of the pre-activation review (``docs/experiments/artifacts/2026-11-02-sf43-pre-activation-review``): normal
draws, the pre-registered pipeline classes (``gate4.RobustTest`` with ``eh.EhBound`` and the mcstat contrast bounds), R replicates:

* sizes: the REALIZED sizes of the design arrays of the analysis (production N, the paired ladder, every width cell, the h_max pairs);
* production arm ``x ~ N(alt, SD_prod^2)`` with the OBSERVED per-seed SD of the production arm of the estimator;
* ladder: ``L_j = x_ip[:n] + D_j`` with ``D = (L_j - L_ip)_j`` (and the gate arm minus ``L_ip``) drawn jointly normal with the OBSERVED mean
  (scenario ``observed``) or zero mean (``no_bias``) and the OBSERVED covariance of those per-seed level differences (their SD and correlations);
* width: ``prod_fine ~ N(alt, observed SD^2)``, ``wide_fine ~ N(alt + dW, observed SD^2)`` (``dW`` = observed mean difference, or 0), the coarse
  arms as the fine arm minus a normal resolution step with the observed mean / SD of the paired difference; h_max: the paired difference
  ``h16 - h8`` is ``N(observed mean or 0, observed SD^2)``;
* scenario ``no_bias``: every budget contrast has true mean 0 (the budget only pays for its sampling noise); ``observed``: the contrast means
  are the observed ones (a world in which the discretization, width and h_max errors are as measured);
* the same standard-normal draws for the two estimators (the two transverse estimators agree to 1e-3 per realization in SF-42 R3, so their noise is
  taken comonotone: ``both`` is then the rejection of both on the same replicate; the Frechet upper bound ``min`` of the two margins is reported).

SIMPLIFICATIONS (documented, deliberate): (i) the sampling limit of the production mean is the STUDENT one, not the bootstrap-t of the analysis (a
bootstrap of 33 599 resamples per replicate is out of reach; under normal draws the two agree to second order and the bootstrap-t is qualified for
the analysis by ``sf43cov``); (ii) ``B_budget`` resamples for ``E_h`` (pre-registered ``bootstrap.B_budget`` by default); (iii) the planning
alternatives are the pilot means (no shrinkage); (iv) normal draws only (the analysis uses the observed data, which need not be normal).  The
review's scenario S2 (non-monotone level errors) and the production-width deficits are not simulated: ``observed`` covers the measured ones.

``run(prereg, arms)`` -- ``prereg["gate4_power"]``::

    R (400), B_budget (pre-registered bootstrap.B_budget), seed (20261103), scenarios (["no_bias", "observed"]),
    alternatives: {amp: {"alpha_TP": value, "alpha_TAP": value}}, source (free text citing the table)

An amplitude whose design is untestable by the floors is reported ``not computed`` (with the reasons).
"""
from __future__ import annotations

import zlib

import numpy as np

from .estimands import TRANSVERSE
from .gate4 import Design, RobustTest, design_from_arms, exceeds
from .gate5 import jsonable
from .qualification import realized_sizes

KEYS = {"R", "B_budget", "seed", "scenarios", "alternatives", "source", "chunk"}
SCENARIOS = ("no_bias", "observed")


def _sd(x):
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def observed_inputs(d):
    """Observed means / SDs / covariances of the design arrays of one estimator (the simulation's inputs)."""
    o = {"N": int(d.prod.size), "sd_prod": _sd(d.prod), "mean_prod": float(d.prod.mean())}
    if d.levels is not None:
        L, ip = d.levels, d.ip
        others = [j for j in range(L.shape[1]) if j != ip]
        D = L[:, others] - L[:, [ip]]
        if d.gate_extra is not None:
            D = np.column_stack([D, d.gate_extra - L[:, ip]])
        cov = np.atleast_2d(np.cov(D, rowvar=False)) if L.shape[0] > 1 else np.zeros((D.shape[1], D.shape[1]))
        o["ladder"] = {"n": int(L.shape[0]), "others": others, "mean": D.mean(axis=0), "cov": cov, "sd": np.sqrt(np.diag(cov)), "extra": d.gate_extra is not None}
    if d.wide_fine is not None:
        w = {"n_wide": int(d.wide_fine.size), "n_prod": int(d.prod_fine.size), "sd_wide": _sd(d.wide_fine), "sd_prod": _sd(d.prod_fine),
             "dW": float(d.wide_fine.mean() - d.prod_fine.mean())}
        if d.wide_coarse is not None:
            dw, dp = d.wide_fine - d.wide_coarse, d.prod_fine - d.prod_coarse
            w.update(step_wide_mean=float(dw.mean()), step_wide_sd=_sd(dw), step_prod_mean=float(dp.mean()), step_prod_sd=_sd(dp))
        o["width"] = w
    if d.h16 is not None:
        h = d.h16 - d.h8
        o["hmax"] = {"n": int(h.size), "mean": float(h.mean()), "sd": _sd(h)}
    return o


def _correlated(rng, mean, cov, shape):
    """``mean + A z`` with ``A A^T = cov`` (eigen-decomposition, negative round-off eigenvalues clipped), z standard normal of the given leading shape."""
    w, V = np.linalg.eigh(cov)
    A = V * np.sqrt(np.clip(w, 0.0, None))
    return mean + rng.standard_normal(shape + (len(mean),)) @ A.T


def simulate_design(d, o, alt, bias, rng, r):
    """A ``Design`` with a leading replicate axis of ``r`` simulated replicates (see the module docstring); ``bias``: 1.0 (``observed``) or 0.0."""
    prod = alt + o["sd_prod"] * rng.standard_normal((r, o["N"]))
    kw = {}
    if "ladder" in o:
        lo, n = o["ladder"], o["ladder"]["n"]
        xl = prod[:, :n] if n <= o["N"] else alt + o["sd_prod"] * rng.standard_normal((r, n))
        Dm = _correlated(rng, bias * lo["mean"], lo["cov"], (r, n))
        K = d.levels.shape[1]
        lev = np.empty((r, n, K))
        lev[..., d.ip] = xl
        for c, j in enumerate(lo["others"]):
            lev[..., j] = xl + Dm[..., c]
        kw.update(levels=lev, res=d.res, ip=d.ip, triples=d.triples, gates=d.gates, fallback_levels=d.fallback_levels)
        if lo["extra"]:
            kw["gate_extra"] = xl + Dm[..., -1]
    if "width" in o:
        w = o["width"]
        pf = alt + w["sd_prod"] * rng.standard_normal((r, w["n_prod"]))
        wf = alt + bias * w["dW"] + w["sd_wide"] * rng.standard_normal((r, w["n_wide"]))
        kw.update(wide_fine=wf, prod_fine=pf, prod_fine_paired=d.prod_fine_paired)
        if "step_wide_mean" in w:
            kw.update(wide_coarse=wf - (bias * w["step_wide_mean"] + w["step_wide_sd"] * rng.standard_normal(wf.shape)),
                      prod_coarse=pf - (bias * w["step_prod_mean"] + w["step_prod_sd"] * rng.standard_normal(pf.shape)))
    if "hmax" in o:
        h = o["hmax"]
        kw.update(h16=bias * h["mean"] + h["sd"] * rng.standard_normal((r, h["n"])), h8=np.zeros((r, h["n"])))
    return Design(prod, d.components, **kw)


def power_cell(prereg, amp, designs, alt, scenario, cfg, levels):
    """Rejection indicators (R, 2) per estimator at ``levels`` for one scenario; the estimators share the standard-normal draws."""
    bias = 1.0 if scenario == "observed" else 0.0
    null = prereg.null_value(amp)
    out = {}
    for e in TRANSVERSE:
        d = designs[e]
        o = observed_inputs(d)
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], zlib.crc32(amp.encode()), zlib.crc32(scenario.encode())]))
        rej = []
        for r in _chunks(cfg["R"], cfg["chunk"]):
            sim = simulate_design(d, o, alt[e], bias, rng, r)
            raw = dict(prereg.raw, bootstrap=dict(prereg.raw["bootstrap"], B_budget=cfg["B_budget"]))
            t = RobustTest(sim, null, raw, "power/%s/%s/%s" % (amp, e, scenario), sampling="student", B_budget=cfg["B_budget"], rng=rng)
            rej.append(exceeds(t.rl_at(np.asarray(levels)), null, t.rtol))
        out[e] = np.concatenate(rej, axis=0)
    return out


def _chunks(R, chunk):
    left = R
    while left > 0:
        yield min(chunk, left)
        left -= chunk


def _p(x):
    p = float(np.mean(x))
    return {"p": p, "mc_se": float(np.sqrt(p * (1.0 - p) / max(1, len(x))))}


def run(prereg, arms):
    g = prereg["gate4_power"]
    m = len(prereg.family_ids())
    fwer = prereg["levels"]["fwer"]
    cfg = {"R": int(g.get("R", 400)), "B_budget": int(g.get("B_budget", prereg["bootstrap"]["B_budget"])), "seed": int(g.get("seed", 20261103)),
           "chunk": int(g.get("chunk", 100)), "scenarios": list(g.get("scenarios", SCENARIOS))}
    levels = [fwer / m, fwer]
    out = {"label": "descriptive; never feeds a decision", "config": dict(cfg, levels={"fwer_over_m": levels[0], "fwer": levels[1]}, m=m,
                                                                           source=g.get("source")),
           "method": "see sf43ana/gate4_power.py (normal draws, observed SDs, Student sampling limit, comonotone estimators)", "amplitudes": {}}
    pb = prereg["bases"]["primary"]
    for amp in prereg.amps:
        alt = g.get("alternatives", {}).get(amp)
        if alt is None:
            continue
        designs = {e: design_from_arms(prereg, amp, arms, e, pb) for e in TRANSVERSE}
        bad = sorted({r for d in designs.values() for r in d.reasons})
        if bad:
            out["amplitudes"][amp] = {"status": "not computed: the design is untestable at the realized sizes", "reasons": bad}
            continue
        res = {"status": "computed", "alternatives": dict(alt), "null_value": prereg.null_value(amp), "basis": pb,
               "realized_sizes": realized_sizes(designs["alpha_TP"]),
               "observed": {e: {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "cov"}) for k, v in observed_inputs(designs[e]).items()}
                            for e in TRANSVERSE},
               "scenarios": {}}
        for sc in cfg["scenarios"]:
            rej = power_cell(prereg, amp, designs, alt, sc, cfg, levels)
            both = rej["alpha_TP"] & rej["alpha_TAP"]
            cell = {"estimators": {e: {"fwer_over_m": _p(rej[e][:, 0]), "fwer": _p(rej[e][:, 1])} for e in TRANSVERSE},
                    "both": {"fwer_over_m": _p(both[:, 0]), "fwer": _p(both[:, 1])},
                    "both_frechet_upper": {"fwer_over_m": min(float(rej[e][:, 0].mean()) for e in TRANSVERSE),
                                           "fwer": min(float(rej[e][:, 1].mean()) for e in TRANSVERSE)}}
            res["scenarios"][sc] = cell
        out["amplitudes"][amp] = res
    return jsonable(out)


def analyse(ctx):
    if not ctx.prereg.raw.get("gate4_power"):
        return None
    return run(ctx.prereg, ctx.arms)
