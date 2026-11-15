"""Gate 4 core: robust sign with error budget, intersection-union, shifted null, classes (decisions G5.4, G5.7 with the
pre-activation refinements H2-H6, H16).

Per amplitude, transverse estimator ``e`` in {alpha_TP, alpha_TAP} and basis (displacement primary, position secondary):

    RL_e(a) = L_e(a/2) - sum_{k=1..K} B_k(a / (2K))          (``mcstat.budget.RobustLimit``)
    reject the null  <=>  RL_e(a) > c                          (c = null_value = null_per_sigma2 * sigma^2 by default; 0 allowed)
    p_e = inf{a : RL_e(a) > c} on the mcstat log lattice (1.0 if never), amplitude p = max(p_TP, p_TAP).

The threshold ``c`` is explicit: the data are never shifted (H3; statistics review S12), so every budget component is computed from the
unshifted arms.  A decision is ``exceeds(RL, c)``: ``RL - c > tie_rtol * max(|RL|, |c|)`` (relative tolerance 1e-9: a tie is not a
rejection).  ``L_e`` is the BOOTSTRAP-t lower limit of the production mean (``sampling_method``, default; Student only on request),
the components are Student-studentized.

``eh.EhBound`` is ``E_h``; ``mcstat.budget.ContrastBound`` gives ``E_W``, the width x lambda/h interaction ``E_WxH`` and ``E_int``.
Enforced (monotone) bounds are those of ``RobustLimit``.

Floors (H13).  A hypothesis is untestable (``p = 1``, estimates descriptive) unless every arm it needs reaches its floor: the production
arm ``floors.production`` (100), the ladder ``floors.ladder`` (30 complete PAIRED seeds: the common prefix of all ladder arms, of the
gate arm and of the production-width arm of the same group), every width cell ``floors.width`` (30), the ``h_max`` passes
``floors.hmax`` (20 paired seeds); an amplitude may override them (``floors`` key); each arm's own ``floor`` applies as well.

Classes (G5.7 with levels; ``classify_estimator`` / ``classify_amplitude``), in order of precedence:
  physical (restricted)  Holm rejects the amplitude's robust-sign hypothesis (both estimators, intersection-union);
  numerical              the production two-sided interval (bootstrap-t, level ``levels.interval``) lies above the null value AND the
                         one-sided ``levels.ucl`` upper limit of the bias-corrected estimate is below ``numerical_fraction`` x the
                         production mean; bias-corrected = production mean + signed Richardson correction + signed width correction
                         (a(W*) - a(W_prod)); its limit is the ``1 - ucl`` percentile of a seed bootstrap that resamples jointly the
                         arms sharing seeds (the ladder and a width-production arm of the same group) and independently the others;
  unresolved (bounded)   otherwise, if ``U(ucl) + sum_k B_k(ucl / K)`` is finite (the reported bound);
  unresolved             anything else.  The two estimators must agree; discordance gives ``unresolved``.
"""
import sys
import traceback
import zlib

import numpy as np

from mcstat._util import boot_counts, ucb_sorted
from mcstat.budget import BootstrapTMean, RobustLimit, StudentMean, e_int, e_interaction, e_w
from mcstat.intervals import bootstrap_t_interval

from . import eh as eh_mod
from .loader import common_prefix

CLASS_PHYS = "physical (restricted)"
CLASS_NUM = "numerical"
CLASS_BOUNDED = "unresolved (bounded)"
CLASS_UNRES = "unresolved"
TRANSVERSE = ("alpha_TP", "alpha_TAP")


def exceeds(rl, c, rtol=1e-9):
    """``RL > c`` with a relative tolerance: ``RL - c > rtol * max(|RL|, |c|)`` (NaN -> False).  A tie is not a rejection."""
    rl = np.asarray(rl, dtype=float)
    with np.errstate(invalid="ignore"):
        return (rl - c) > rtol * np.maximum(np.abs(rl), abs(c))


def seq_rng(base_seed, *labels):
    """Deterministic, label-dependent stream (crc32 of the label; never Python's salted hash)."""
    key = [int(base_seed)] + [zlib.crc32(str(l).encode()) for l in labels]
    return np.random.default_rng(np.random.SeedSequence(key))


class Design:
    """Per-seed arrays of one (amplitude, estimator, basis); everything a robust test needs.

    ``prod`` (N,) production arm; ``levels`` (n, K) ladder on common seeds with ``res``/``ip``/``triples``; ``wide_fine``,
    ``prod_fine`` (unpaired), optional ``wide_coarse``, ``prod_coarse`` (paired within the width); ``prod_fine_paired``: the
    ``prod_fine`` array is aligned seed-for-seed with ``levels``; ``h16``, ``h8`` (paired passes).  ``components``: the configured
    list; ``reasons``: why the design is untestable (empty if testable).
    """

    def __init__(self, prod, components, levels=None, res=None, ip=None, triples=None, wide_fine=None, prod_fine=None,
                 wide_coarse=None, prod_coarse=None, prod_fine_paired=False, h16=None, h8=None, reasons=(), gates=(), gate_extra=None,
                 fallback_levels=None):
        self.prod = None if prod is None else np.asarray(prod, dtype=float)
        self.components = list(components)
        self.levels, self.res, self.ip, self.triples = levels, res, ip, triples
        self.gates, self.gate_extra, self.fallback_levels = list(gates), gate_extra, fallback_levels
        self.wide_fine, self.prod_fine, self.wide_coarse, self.prod_coarse = wide_fine, prod_fine, wide_coarse, prod_coarse
        self.prod_fine_paired = prod_fine_paired
        self.h16, self.h8 = h16, h8
        self.reasons = list(reasons)

    @property
    def testable(self):
        return not self.reasons


def design_from_arms(prereg, amp, arms, est, basis):
    """Design for estimator ``est`` and ``basis`` from the loaded ``ArmData``; untestable (with reasons) if an arm used is below its floor.

    Alignment: arms that share seeds are cut to their common prefix (the ladder together with a width-production arm of its
    group; the two resolutions of a width; the two ``h_max`` passes); the production arm keeps its own full prefix for the sampling
    limit.
    """
    c, prim = prereg.amp(amp), prereg["primary_window"]
    comps = [k for k in ("E_h", "E_W", "E_WxH", "E_int") if k in c.get("components", [])]
    reasons = []
    floors = dict(prereg["floors"], **c.get("floors", {}))

    def floor_check(what, n, role):
        if n < floors[role]:
            reasons.append("%s: %d complete seeds < floor %d (H13)" % (what, n, floors[role]))

    def use(aid):
        a = arms[aid]
        if not a.usable:
            reasons.append("arm %s below its floor (N=%d < %d)" % (aid, a.N, a.floor))
        return a

    def vals(a, n=None):
        return a.get(prim, basis, est, n)

    P = use(c["production"])
    floor_check("production arm %s" % P.id, P.N, "production")
    LA = [use(i) for i in c["ladder"]["arms"]] if "E_h" in comps else []
    gate_arms = {g["arm"]: use(g["arm"]) for g in c["ladder"].get("gates", []) if g["type"] == "extra_level"} if "E_h" in comps else {}
    wf = wc = pf = pc = h16 = h8 = None
    if "E_W" in comps or "E_WxH" in comps:
        W = c["width"]
        wf, pf = use(W["wide_fine"]), use(W["prod_fine"])
        if "E_WxH" in comps:
            wc, pc = use(W["wide_coarse"]), use(W["prod_coarse"])
        for cell in (wf, pf, wc, pc):
            if cell is not None:
                floor_check("width cell %s" % cell.id, cell.N, "width")
    if "E_int" in comps:
        h16, h8 = use(c["hmax"]["h16"]), use(c["hmax"]["h8"])
    d = Design(vals(P) if P.usable else None, comps)
    if reasons:
        d.reasons = sorted(set(reasons))
        return d
    paired = bool(LA and pf is not None and pf.group == LA[0].group)
    pg = [a for a in (pf, pc) if a is not None]
    if LA:
        joint = list(LA) + list(gate_arms.values()) + ([a for a in pg if a.id not in {x.id for x in LA}] if paired else [])
        n = common_prefix(joint)
        floor_check("ladder (paired prefix of %s)" % [a.id for a in joint], n, "ladder")
        if reasons:
            d.reasons = sorted(set(reasons))
            return d
        d.levels = np.stack([vals(a, n) for a in LA], axis=-1)
        d.res, d.ip, d.triples = c["ladder"]["resolutions"], c["ladder"]["production_index"], c["ladder"].get("triples")
        d.gates = [dict(g) for g in c["ladder"].get("gates", [])]
        d.fallback_levels = c["ladder"].get("fallback_levels")
        for g in d.gates:
            if g["type"] == "extra_level":
                d.gate_extra = vals(gate_arms[g["arm"]], n)
    if pg:
        n = d.levels.shape[0] if paired else common_prefix(pg)
        d.prod_fine, d.prod_fine_paired = vals(pf, n), paired
        d.prod_coarse = vals(pc, n) if pc is not None else None
    if wf is not None:
        n = common_prefix([wf] + ([wc] if wc is not None else []))
        d.wide_fine, d.wide_coarse = vals(wf, n), (vals(wc, n) if wc is not None else None)
    if h16 is not None:
        n = common_prefix([h16, h8])
        floor_check("h_max passes %s" % [h16.id, h8.id], n, "hmax")
        d.h16, d.h8 = vals(h16, n), vals(h8, n)
    if reasons:
        d.reasons = sorted(set(reasons))
    return d


def _check_complete(d):
    miss = []
    if not d.components:
        miss.append("no budget component configured")
    if d.prod is None:
        miss.append("production")
    if "E_h" in d.components and d.levels is None:
        miss.append("ladder")
    if ("E_W" in d.components or "E_WxH" in d.components) and (d.wide_fine is None or d.prod_fine is None):
        miss.append("width")
    if "E_WxH" in d.components and (d.wide_coarse is None or d.prod_coarse is None):
        miss.append("width 2x2")
    if "E_int" in d.components and (d.h16 is None or d.h8 is None):
        miss.append("hmax")
    return miss


def build_components(d, eh_cfg, B, rng):
    """List of mcstat component objects in the order E_h, E_W, E_WxH, E_int (all Student-studentized)."""
    out = []
    for name in d.components:
        if name == "E_h":
            lo, hi = eh_cfg["p_range"]
            out.append(eh_mod.EhBound(d.levels, d.res, d.ip, d.triples, lo, hi, eh_cfg["p_cap"], eh_cfg["safety"],
                                      eh_cfg["fallback_factor"], B=B, rng=rng, gates=d.gates, extra=d.gate_extra,
                                      gate_level=eh_cfg.get("gate_level", 0.90), fallback_levels=d.fallback_levels))
        elif name == "E_W":
            out.append(e_w(d.wide_fine, d.prod_fine, method="student"))
        elif name == "E_WxH":
            out.append(e_interaction(d.wide_fine, d.wide_coarse, d.prod_fine, d.prod_coarse, method="student"))
        elif name == "E_int":
            out.append(e_int(d.h16, d.h8, method="student"))
    return out


class RobustTest:
    """Robust-sign test of one (amplitude, estimator, basis).  Vectorized over leading replicate axes of the design arrays."""

    def __init__(self, design, null, cfg, label, lattice=None, sampling=None, B_sampling=None, B_budget=None, rng=None):
        self.design, self.null, self.label = design, null, label
        self.testable = design.testable and not _check_complete(design)
        self.missing = list(design.reasons) + [m for m in _check_complete(design)]
        self.p_cache = None
        self.rtol = float(cfg.get("tie_rtol", 1e-9))
        if not self.testable:
            return
        bs = cfg["bootstrap"]
        rng = rng if rng is not None else seq_rng(bs["seed"], "robust", label)
        method = sampling or cfg["sampling_method"]
        Bs = B_sampling or bs["B_sampling"]
        self.sampling = StudentMean(design.prod) if method == "student" else BootstrapTMean(design.prod, B=Bs, rng=rng)
        self.components = build_components(design, cfg["eh"], B_budget or bs["B_budget"], rng)
        self.rl = RobustLimit(self.sampling, self.components, lattice)
        self.cfg = cfg

    def p_value(self):
        """(p, never_positive): smallest lattice level with RL > null (1.0 if none)."""
        if not self.testable:
            return 1.0, True
        if self.p_cache is None:
            RL = self.rl.lower(self.rl.lattice)
            pos = exceeds(RL, self.null, self.rtol)
            first = np.argmax(pos, axis=-1)
            any_pos = pos.any(axis=-1)
            self.p_cache = (np.where(any_pos, self.rl.lattice[first], 1.0), ~any_pos)
        return self.p_cache

    def rl_at(self, a):
        return self.rl.lower(a)

    def reject(self, a):
        """Exact decision at level ``a``: RL(a) > null (not read off the lattice)."""
        return bool(self.testable and np.all(exceeds(self.rl.lower(a), self.null, self.rtol)))

    def upper_bound(self, tail):
        """Bound of the class 'unresolved (bounded)': U(tail) + sum_k B_k(tail / K)."""
        if not self.testable:
            return np.inf
        return self.sampling.upper(tail) + self.rl.component_bounds(2.0 * tail).sum(axis=-1)

    def report(self, levels):
        rep = self.rl.report(np.asarray(levels, dtype=float))
        return rep


class FailedTest:
    """Stand-in for a ``RobustTest`` whose construction or evaluation raised: untestable, ``p = 1``, never rejects."""
    design = None

    def __init__(self, label, null):
        self.label, self.null, self.testable, self.missing = label, null, False, ["error"]

    def p_value(self):
        return 1.0, True

    def reject(self, a):
        return False

    def upper_bound(self, tail):
        return np.inf


def record_error(sink, hid, exc, status="error"):
    """Per-hypothesis isolation: append ``{id, status, type, message, traceback}`` to ``sink`` (a list, or None) and print ``HYPOTHESIS_ERROR``
    on stderr.  The hypothesis is then untestable (``p = 1``); the caller continues with the rest of the family."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print("HYPOTHESIS_ERROR %s: %s: %s" % (hid, type(exc).__name__, exc), file=sys.stderr)
    print(tb, file=sys.stderr)
    if sink is not None:
        sink.append({"id": hid, "status": status, "type": type(exc).__name__, "message": str(exc), "traceback": tb})


# ----------------------------------------------------------------------------------------------------------------------
# bias-corrected estimate (class "numerical")
# ----------------------------------------------------------------------------------------------------------------------
def bias_corrected(d, eh_bound, p_order, B, rng, ucl_tail):
    """Production mean + Richardson correction + width correction, with the seed-bootstrap upper limit (module docstring).

    ``p_order``: the order of the Richardson correction (H16: ``p_use`` if the order branch is in range, else 1), held fixed in the
    resamples.  Returns dict(mean_prod, richardson, width, estimate, ucl, p).  The production mean is the ladder's production column
    when a ladder exists (aligned seeds), else the production arm.
    """
    ladder = d.levels is not None and "E_h" in d.components
    width = "E_W" in d.components and d.wide_fine is not None
    if ladder:
        Lv = d.levels
        n = Lv.shape[0]
        WL = boot_counts(n, B, rng) / n  # (B, n)
        ms = Lv.T @ WL.T  # (K, B)
        prod_s = ms[d.ip]
        rich_hat, rich_s = float(eh_bound.correction(Lv.mean(axis=0), p_order)), eh_bound.correction(ms.T, p_order)
        prod_hat = float(Lv[:, d.ip].mean())
    else:
        n = d.prod.size
        WL = boot_counts(n, B, rng) / n
        prod_s, prod_hat = d.prod @ WL.T, float(d.prod.mean())
        rich_hat, rich_s = 0.0, 0.0
    w_hat, w_s = 0.0, 0.0
    if width:
        nw = d.wide_fine.size
        Ww = boot_counts(nw, B, rng) / nw
        if d.prod_fine_paired:
            pf_hat, pf_s = float(d.prod_fine.mean()), d.prod_fine @ WL.T
        else:
            npf = d.prod_fine.size
            pf_hat, pf_s = float(d.prod_fine.mean()), d.prod_fine @ (boot_counts(npf, B, rng) / npf).T
        w_hat, w_s = float(d.wide_fine.mean()) - pf_hat, d.wide_fine @ Ww.T - pf_s
    star = np.sort(prod_s + rich_s + w_s)
    return {"mean_prod": prod_hat, "richardson": rich_hat, "width": w_hat, "estimate": prod_hat + rich_hat + w_hat, "p": p_order,
            "ucl": float(ucb_sorted(star, ucl_tail)), "ucl_tail": ucl_tail, "B": B}


def classify_estimator(test, cfg, label):
    """Non-Holm part of the class of one (estimator, basis): dict(numerical, bounded, bound, interval, corrected)."""
    out = {"testable": test.testable, "numerical": False, "bounded": False, "bound": float("inf"), "reasons": test.missing}
    if not test.testable:
        return out
    lv, d, bs = cfg["levels"], test.design, cfg["bootstrap"]
    out["bound"] = float(test.upper_bound(lv["ucl"]))
    out["bounded"] = bool(np.isfinite(out["bound"]))
    iv = bootstrap_t_interval(d.prod, lv["interval"], "two", bs["B_sampling"], rng=seq_rng(bs["seed"], "interval", label))
    out["interval"] = [float(iv.lo), float(iv.hi)]
    ehb = next((c for c in test.components if getattr(c, "name", "") == "E_h"), None)
    p_order, order_used = 1.0, False
    if ehb is not None:  # H16: p = p_use (the E_h bound at the class's component tail ucl / K) if in range, else 1
        det = ehb.details(lv["ucl"] / len(test.components))
        order_used = bool(det["order_branch_used"])
        p_order = float(det["p_use"]) if order_used else 1.0
    corr = bias_corrected(d, ehb, p_order, bs["B_ucl"], seq_rng(bs["seed"], "ucl", label), lv["ucl"])
    corr["order_branch_in_range"] = order_used
    out["corrected"] = corr
    # production mean of the comparison = the aligned one used by the correction
    out["numerical"] = bool(bool(exceeds(iv.lo, test.null, test.rtol)) and corr["mean_prod"] > 0.0
                            and corr["ucl"] < cfg["numerical_fraction"] * corr["mean_prod"])
    return out


def classify_amplitude(holm_rejected, per_est):
    """(class, note).  ``per_est``: {estimator: classify_estimator dict}.  Precedence: physical, numerical, bounded, unresolved."""
    if holm_rejected:
        return CLASS_PHYS, ""
    cls = {}
    for e, c in per_est.items():
        cls[e] = CLASS_NUM if c["numerical"] else CLASS_BOUNDED if c["bounded"] else CLASS_UNRES
    if len(set(cls.values())) > 1:
        return CLASS_UNRES, "estimator discordance: " + ", ".join("%s=%s" % kv for kv in sorted(cls.items()))
    return next(iter(cls.values())), ""
