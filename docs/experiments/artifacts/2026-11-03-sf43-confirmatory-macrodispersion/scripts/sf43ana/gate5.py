"""Gate 5 of SF-43 (decision G5.5 with the Fieller form of pre-activation decision H5): paired ratio-of-means equivalence.

Estimand ``mu_L / mu_P`` (ratio of arithmetic means of the plane-based ``alpha_LP`` of the consumer twin and of Par2 on the same seeds)
within ``[1/m, m]``, ``m = 1.10``.  FIELLER FORM: the two one-sided hypotheses are one-sample tests on the per-seed differences

    D_hi,r = L_r - m P_r      H01: mean(D_hi) >= 0   (ratio >= m)      rejected for small  D_hi  (p_hi)
    D_lo,r = L_r - P_r / m    H02: mean(D_lo) <= 0   (ratio <= 1/m)    rejected for large  D_lo  (p_lo)

each at the full (Holm) level (TOST / intersection-union: ``p_tost = max(p_hi, p_lo)``).  At a general ``theta`` the tests use
``D(theta) = L - theta P``; the ``level`` confidence interval of the ratio is obtained by inverting them (Fieller), the one-sided limits
being the smallest ``theta`` at which ``H(theta): ratio >= theta`` is rejected at tail ``(1 - level) / 2`` (upper limit) and the largest at
which ``ratio <= theta`` is rejected (lower limit); a geometric scan from ``theta_hat`` (factor 1.02, span 1e3) brackets the first
rejection, then a bisection on the monotone predicate finds it (an unbounded set gives ``inf`` / ``0``).

Procedures (the pre-registered order of the selection rule), for a sample ``x`` of ``D`` of size ``n``, ``T = mean / (sd / sqrt n)``:

* ``hall``       Hall (1992, JRSS-B 54:221) skewness-corrected statistic ``h(T) = T + c T^2 + c^2 T^3 / 3 + gamma / (6 sqrt n)`` with
                 ``c = gamma / (3 sqrt n)`` and ``gamma = m3 / s^3`` (``m3`` the mean cubed deviation, ``s`` the ddof = 1 sd of the same
                 studentization).  ``h = ((1 + c T)^3 - 1) / (3 c) + gamma / (6 sqrt n)`` is strictly increasing in ``T``, hence
                 monotonically invertible; it is compared with the Student ``t_{n-1}`` distribution (``ref='t'``, default: it reduces to
                 the Student test when ``gamma = 0``; ``ref='normal'`` is the original normal reference).  ``p_hi = F_t(h)``,
                 ``p_lo = 1 - F_t(h)``.
* ``boott``      bootstrap-t: resample the seed pairs jointly, ``T*_b = (mean*_b - mean) / se*_b`` with ``se*_b`` the ddof = 1 standard
                 error of the resample; rank p-values ``p_hi = (1 + #{T* <= T}) / (B + 1)``, ``p_lo = (1 + #{T* >= T}) / (B + 1)``
                 (``p <= tail`` iff ``T`` lies beyond the rank-``floor(tail (B + 1))`` order statistic: the rank rule of
                 ``mcstat._util``).  ``B = DEFAULT_B = 33599``: ``B + 1 = 33600`` is a multiple of ``1/a`` for every Holm level
                 ``a = 0.05 / j`` (``j <= 8``) and of every half level (Hall 1986: exact quantile ranks).
* ``boott_half`` the bootstrap-t read at HALF the level: reject at Holm level ``a`` iff ``p_boott <= a / 2``, i.e. ``p = min(1, 2 p_boott)``.
* ``student``    descriptive only (never selected; known not to qualify, SF-42 Finding (j)).

SELECTION (G5.3, H5, H15).  The coverage summary of the harness ``sf43cov`` is read (``qualification``): a procedure is qualified at a
grid size ``n_g`` iff, for each of ``normal``, ``stress``, ``stress-``, both sides (``hi``, ``lo``) and every tail used (the Holm levels
``fwer / (m - j)`` for ``hall`` and ``boott``; their halves for ``boott_half``), the cell passed (95 % one-sided Clopper-Pearson UCL of the
miss rate at most 1.25 x the tail, at least 175 expected misses).  The procedure used at realized ``n`` is the first of
(``hall``, ``boott``, ``boott_half``) qualified at the LARGEST grid ``n_g <= n``; if none is, Gate 5 at that amplitude is descriptive and
its p-value is 1 (the hypothesis stays in the Holm family, H4).

Reported alongside (descriptive, outside the family): the Student Fieller form, the SF-42 delta-method TOST (``mcstat.gate5``), achieved
power (the analytic Student delta-method power of ``mcstat.gate5.tost_power`` at the Holm worst level ``fwer / m``, with the observed CVs
and pairing correlation: an approximation to the power of the selected procedure, not its exact value), and the SF-39-style unmatched
comparison (ratio of means of independent samples, delta-method log-ratio interval).

``run(prereg, data)`` -- ``prereg["gate5"]``::

    family_m (8), fwer (0.05), margin_ratio (1.10), level (0.90), B (33599), seed (int), key ("alpha_LP"), ref ("t"),
    coverage_summary (path of the summary.json of sf43cov; or ``data["coverage_summary"]``, an already loaded dict),
    amplitudes: {amp: {"consumer": arm, "par2": arm, "unmatched": {"consumer": arm, "par2": arm} (optional)}}

``data``::

    {"arms": {arm: {"seed": int array (n,), "cols": {name: float array (n,)}}}}     (NaN allowed; pairs with a NaN are dropped, counted)

Returns ``{"amplitudes": {amp: {...}}, "pvalues": {amp: p}, "holm_order": [amp, ...], "selection": {...}, "config": {...}}``.
"""
from __future__ import annotations

import json
import math
import zlib

from sf43cov import mcpath  # noqa: F401  (puts the SF-42 mcstat package on sys.path)

import numpy as np
from scipy import stats as _st

from mcstat import gate5 as _mg5
from mcstat._util import bootstrap_means, get_rng

MARGIN_RATIO = 1.10
DEFAULT_B = 33599
PROCS = ("hall", "boott", "boott_half")  # the pre-registered order
DISTS = ("normal", "stress", "stress-")
COVERAGE_LABEL = "fieller"
GRID_FACTOR = 1.02
GRID_SPAN = 1e3


def holm_levels(m=8, fwer=0.05):
    """``fwer / m, fwer / (m - 1), ..., fwer`` (increasing): the levels of the Holm step-down for ``m`` tests."""
    return np.array([fwer / (m - j) for j in range(m)])


def jsonable(o):
    """Recursively converts numpy scalars / arrays, tuples as keys and NaN-safe floats into JSON-serializable values."""
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def arm_series(data, arm, key):
    """``(seed, values)`` of column ``key`` of ``arm`` (values float, shape (n,) or (n, K)); KeyError with a clear message if absent."""
    try:
        a = data["arms"][arm]
    except KeyError:
        raise KeyError("arm %r is not in data['arms'] (have %s)" % (arm, sorted(data.get("arms", {})))) from None
    if key not in a["cols"]:
        raise KeyError("arm %r has no column %r (have %s)" % (arm, key, sorted(a["cols"])))
    seed = np.asarray(a["seed"])
    v = np.asarray(a["cols"][key], dtype=float)
    if v.shape[0] != seed.shape[0]:
        raise ValueError("arm %r column %r is not aligned with its seeds" % (arm, key))
    return seed, v


def pair_by_seed(sa, va, sb, vb):
    """Values of two seed-indexed vectors on the common seeds (sorted by seed) with both finite.  Returns ``(seeds, a, b, info)``."""
    if len(set(sa.tolist())) != len(sa) or len(set(sb.tolist())) != len(sb):
        raise ValueError("duplicate seeds in an arm")
    ia = {int(s): i for i, s in enumerate(sa)}
    ib = {int(s): i for i, s in enumerate(sb)}
    common = sorted(set(ia) & set(ib))
    a = np.array([va[ia[s]] for s in common], dtype=float)
    b = np.array([vb[ib[s]] for s in common], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    info = {"n_common": len(common), "n_only_a": len(ia) - len(common), "n_only_b": len(ib) - len(common),
            "n_nonfinite_dropped": int((~ok).sum())}
    return np.array(common, dtype=np.int64)[ok], a[ok], b[ok], info


# ----------------------------------------------------------------------------------------------------------------------
# One-sample procedures on a sample x of D (arrays (..., n); used by the coverage harness and by the Fieller class)
# ----------------------------------------------------------------------------------------------------------------------
def _studentized(x):
    n = x.shape[-1]
    mean = x.mean(axis=-1)
    sd = x.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0.0, mean * math.sqrt(n) / sd, np.where(mean == 0.0, 0.0, np.sign(mean) * np.inf))
    return t, mean, sd


def hall_statistic(t, gamma, n):
    """Hall (1992): ``T + c T^2 + c^2 T^3 / 3 + gamma / (6 sqrt n)``, ``c = gamma / (3 sqrt n)`` (arrays broadcast)."""
    c = gamma / (3.0 * math.sqrt(n))
    return t + c * t * t + c * c * t ** 3 / 3.0 + gamma / (6.0 * math.sqrt(n))


def sample_skewness(x):
    """``m3 / s^3``: mean cubed deviation over the cube of the ddof = 1 sd (the studentization of ``T``); 0 for constant data."""
    xc = x - x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(sd > 0.0, (xc ** 3).mean(axis=-1) / sd ** 3, 0.0)
    return g


def _ref_cdf_sf(h, n, ref):
    if ref == "t":
        return _st.t.cdf(h, n - 1), _st.t.sf(h, n - 1)
    if ref == "normal":
        return _st.norm.cdf(h), _st.norm.sf(h)
    raise ValueError("ref must be 't' or 'normal', got %r" % (ref,))


def boot_tstar(x, B, rng, share_draws=True):
    """Unsorted ``T*`` (..., B) of the bootstrap-t of the mean of ``x`` (..., n): the formula of ``mcstat.intervals.bootstrap_t``
    (resample mean of the centred values over the resample ddof = 1 standard error; a resample with zero variance gets ``0`` or
    ``+-inf``) without the sort, which a rank p-value does not need."""
    n = x.shape[-1]
    xc = x - x.mean(axis=-1, keepdims=True)
    m, s2 = bootstrap_means((xc, xc * xc), B, rng, share_draws)
    var = np.maximum(n / (n - 1.0) * (s2 - m * m), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 0.0, m * math.sqrt(n) / np.sqrt(var), np.where(m == 0.0, 0.0, np.sign(m) * np.inf))


def rank_p(tstar, t):
    """``(p_hi, p_lo)``: ``(1 + #{T* <= t}) / (B + 1)`` and ``(1 + #{T* >= t}) / (B + 1)``."""
    B = tstar.shape[-1]
    tt = np.asarray(t)[..., None]
    return (1.0 + (tstar <= tt).sum(axis=-1)) / (B + 1.0), (1.0 + (tstar >= tt).sum(axis=-1)) / (B + 1.0)


def onesample_p(x, proc, B=DEFAULT_B, rng=None, share_draws=True, ref="t"):
    """One-sided p-values ``(p_hi, p_lo)`` of the mean of ``x`` (..., n): H0 mean >= 0 against smaller (``p_hi``) and H0 mean <= 0
    against larger (``p_lo``), by ``proc`` in ``'student'``, ``'hall'``, ``'boott'`` (``'boott_half'`` is ``min(1, 2 p)`` of it)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    t, _, _ = _studentized(x)
    if proc == "student":
        return _st.t.cdf(t, n - 1), _st.t.sf(t, n - 1)
    if proc == "hall":
        return _ref_cdf_sf(hall_statistic(t, sample_skewness(x), n), n, ref)
    if proc == "boott":
        return rank_p(boot_tstar(x, B, get_rng(rng), share_draws), t)
    raise ValueError("unknown procedure %r" % (proc,))


# ----------------------------------------------------------------------------------------------------------------------
# Fieller form on paired data
# ----------------------------------------------------------------------------------------------------------------------
class Fieller:
    """Paired positive data ``L``, ``P`` (1-D, aligned): one-sided p-values of ``H(theta)`` for any ``theta`` and procedure.

    ``p(proc, theta) -> (p_hi, p_lo)``: ``p_hi`` tests ``ratio >= theta`` (small ``D(theta) = L - theta P``), ``p_lo`` tests
    ``ratio <= theta``.  The bootstrap (``boott``, ``boott_half``) draws the seed-pair resamples once (``rng``) and evaluates
    ``T*(theta)`` from the resample means and second moments of the centred ``L``, ``P`` (exactly the bootstrap-t of the mean of
    ``D(theta)``: tested against ``onesample_p``).
    """

    def __init__(self, L, P, B=DEFAULT_B, rng=None, ref="t"):
        self.L, self.P = np.asarray(L, dtype=float), np.asarray(P, dtype=float)
        if self.L.shape != self.P.shape or self.L.ndim != 1:
            raise ValueError("L and P must be aligned 1-D arrays")
        self.n = self.L.size
        if self.n < 3:
            raise ValueError("Gate 5 needs at least 3 paired seeds")
        self.mL, self.mP = float(self.L.mean()), float(self.P.mean())
        if not (self.mL > 0 and self.mP > 0):
            raise ValueError("the ratio of means needs positive means")
        self.theta_hat = self.mL / self.mP
        self.Lc, self.Pc = self.L - self.mL, self.P - self.mP
        self.B, self.rng, self.ref = B, rng, ref
        self._boot = None

    def d_stats(self, theta):
        """``(mean, sd, gamma)`` of ``D(theta)`` (ddof = 1 sd; ``gamma`` as ``sample_skewness``)."""
        dc = self.Lc - theta * self.Pc
        mean = self.mL - theta * self.mP
        sd = float(np.sqrt((dc * dc).sum() / (self.n - 1.0)))
        gamma = float((dc ** 3).mean() / sd ** 3) if sd > 0 else 0.0
        return mean, sd, gamma

    def _tstar(self, theta):
        if self._boot is None:
            if self.rng is None:
                raise ValueError("bootstrap procedures need rng= (a numpy Generator): bootstrap work must be reproducible")
            Lc, Pc = self.Lc, self.Pc
            self._boot = bootstrap_means((Lc, Pc, Lc * Lc, Lc * Pc, Pc * Pc), self.B, self.rng, share_draws=True)
        a, b, sll, slp, spp = self._boot
        num = a - theta * b
        var = np.maximum(self.n / (self.n - 1.0) * (sll - 2.0 * theta * slp + theta * theta * spp - num * num), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0.0, num * math.sqrt(self.n) / np.sqrt(var), np.where(num == 0.0, 0.0, np.sign(num) * np.inf))

    def p(self, proc, theta):
        mean, sd, gamma = self.d_stats(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = mean * math.sqrt(self.n) / sd if sd > 0 else (0.0 if mean == 0 else math.copysign(math.inf, mean))
        if proc == "student":
            return float(_st.t.cdf(t, self.n - 1)), float(_st.t.sf(t, self.n - 1))
        if proc == "hall":
            ph, pl = _ref_cdf_sf(hall_statistic(t, gamma, self.n), self.n, self.ref)
            return float(ph), float(pl)
        if proc in ("boott", "boott_half"):
            ph, pl = rank_p(self._tstar(theta), t)
            ph, pl = float(ph), float(pl)
            return (min(1.0, 2.0 * ph), min(1.0, 2.0 * pl)) if proc == "boott_half" else (ph, pl)
        raise ValueError("unknown procedure %r" % (proc,))

    def limits(self, proc, tail):
        """``(lower, upper)`` one-sided limits of the ratio at ``tail`` each (a ``1 - 2 tail`` interval) by inverting the tests of
        ``proc`` (``boott_half`` is the bootstrap-t at tail ``tail / 2``): see the module docstring (scan + bisection)."""
        th = self.theta_hat

        def rej_hi(x):
            return self.p(proc, x)[0] <= tail

        def rej_lo(x):
            return self.p(proc, x)[1] <= tail

        return (_invert(rej_lo, th, -1), _invert(rej_hi, th, +1))


def _invert(rej, th, direction):
    """First ``theta`` away from ``th`` (``direction`` = +1 up, -1 down) where the monotone predicate ``rej`` turns True (bisection to
    1e-13 relative); ``inf`` (up) / ``0.0`` (down) if it does not within the span; ``th`` itself if it is already True there."""
    if rej(th):
        return th
    prev = th
    k_max = int(math.ceil(math.log(GRID_SPAN) / math.log(GRID_FACTOR)))
    for k in range(1, k_max + 1):
        cur = th * GRID_FACTOR ** (direction * k)
        if rej(cur):
            a, b = prev, cur  # rej(a) is False, rej(b) True; bisect on the monotone predicate
            for _ in range(100):
                mid = 0.5 * (a + b)
                if mid == a or mid == b:
                    break
                if rej(mid):
                    b = mid
                else:
                    a = mid
            return b
        prev = cur
    return math.inf if direction > 0 else 0.0


def verdict(lo, hi, margin_ratio=MARGIN_RATIO):
    """G5.5: ``agree`` (interval inside ``[1/m, m]``), ``disagree`` (entirely outside), else ``inconclusive`` (ratio scale)."""
    if lo >= 1.0 / margin_ratio and hi <= margin_ratio:
        return "agree"
    if lo > margin_ratio or hi < 1.0 / margin_ratio:
        return "disagree"
    return "inconclusive"


def p_tost(F, proc, margin_ratio=MARGIN_RATIO):
    """``(p_hi(m), p_lo(1/m), max)``: the two Fieller one-sided p-values at the margins and the TOST p-value."""
    ph = F.p(proc, margin_ratio)[0]
    pl = F.p(proc, 1.0 / margin_ratio)[1]
    return ph, pl, max(ph, pl)


# ----------------------------------------------------------------------------------------------------------------------
# Qualification and selection from the coverage summary of sf43cov
# ----------------------------------------------------------------------------------------------------------------------
def load_coverage_summary(path):
    with open(path) as f:
        return json.load(f)


def _find_entry(entries, tail):
    for e in entries or ():
        if abs(e["tail"] - tail) <= 1e-9 * tail:
            return e
    return None


def qualification(summary, m=8, fwer=0.05, dists=DISTS, label=COVERAGE_LABEL):
    """``{proc: {n_grid: {"qualified": bool, "max_ucl_over_tail": float, "problems": [str, ...]}}}`` from a coverage summary
    (cells with rows ``<proc>/<side>``; see the module docstring).  Unjudged tails, missing cells or tails count as NOT qualified."""
    main = holm_levels(m, fwer)
    cells = [c for c in summary["cells"] if c["label"] == label]
    ns = sorted({c["n"] for c in cells if not c.get("report_only")})
    out = {p: {} for p in PROCS}
    for n in ns:
        for proc in PROCS:
            row = "boott" if proc == "boott_half" else proc
            used = main / 2.0 if proc == "boott_half" else main
            problems, worst = [], 0.0
            for d in dists:
                cell = next((c for c in cells if c["n"] == n and c["dist"] == d), None)
                if cell is None:
                    problems.append("%s: cell missing or incomplete" % d)
                    continue
                for side in ("hi", "lo"):
                    ents = cell["rows"].get("%s/%s" % (row, side))
                    for t in used:
                        e = _find_entry(ents, t)
                        if e is None:
                            problems.append("%s %s tail %.4g: not simulated" % (d, side, t))
                            continue
                        worst = max(worst, e["ucl_over_tail"])
                        if e["pass"] is not True:
                            problems.append("%s %s tail %.4g: %s (UCL/tail %.3f)" %
                                            (d, side, t, "not judged" if e["pass"] is None else "fails", e["ucl_over_tail"]))
            out[proc][n] = {"qualified": not problems, "max_ucl_over_tail": worst, "problems": problems}
    return out


def select_procedure(summary, n_realized, m=8, fwer=0.05, dists=DISTS):
    """The first of ``PROCS`` qualified at the largest grid size ``<= n_realized`` (see the module docstring).

    Returns ``{"procedure": name | None, "grid_n": int | None, "n": n_realized, "table": {proc: qualified at grid_n},
    "problems": {proc: [...]}}``."""
    q = qualification(summary, m, fwer, dists)
    ns = sorted(q[PROCS[0]])
    below = [g for g in ns if g <= n_realized]
    if not below:
        return {"procedure": None, "grid_n": None, "n": n_realized, "table": {}, "problems": {},
                "reason": "no coverage grid size <= %d (grid %s)" % (n_realized, ns)}
    g = below[-1]
    chosen = next((p for p in PROCS if q[p][g]["qualified"]), None)
    return {"procedure": chosen, "grid_n": g, "n": n_realized, "table": {p: q[p][g]["qualified"] for p in PROCS},
            "problems": {p: q[p][g]["problems"][:6] for p in PROCS if not q[p][g]["qualified"]}}


# ----------------------------------------------------------------------------------------------------------------------
# Descriptive companions
# ----------------------------------------------------------------------------------------------------------------------
def unmatched(L, P, level=0.90, margin_ratio=MARGIN_RATIO):
    """SF-39-style comparison for continuity: independent samples ``L``, ``P`` (any sizes), ratio of means, delta-method log-ratio
    interval ``log(mL / mP) -/+ t_{(1+level)/2, nu} sqrt(cv_L^2 / n_L + cv_P^2 / n_P)`` with the Satterthwaite ``nu`` of the two
    squared standard errors of the means."""
    L, P = np.asarray(L, dtype=float), np.asarray(P, dtype=float)
    nL, nP = L.size, P.size
    mL, mP = L.mean(), P.mean()
    if nL < 2 or nP < 2 or not (mL > 0 and mP > 0):
        return {"ratio": float("nan"), "lo": float("nan"), "hi": float("nan"), "verdict": "undefined", "n_L": nL, "n_P": nP}
    vL, vP = (L.std(ddof=1) / mL) ** 2 / nL, (P.std(ddof=1) / mP) ** 2 / nP
    se = math.sqrt(vL + vP)
    nu = (vL + vP) ** 2 / (vL ** 2 / (nL - 1) + vP ** 2 / (nP - 1)) if se > 0 else float(nL + nP - 2)
    q = float(_st.t.isf(0.5 * (1.0 - level), nu))
    th = math.log(mL / mP)
    lo, hi = math.exp(th - q * se), math.exp(th + q * se)
    return {"ratio": math.exp(th), "lo": lo, "hi": hi, "se_log": se, "dof": nu, "verdict": verdict(lo, hi, margin_ratio),
            "n_L": nL, "n_P": nP, "mean_L": float(mL), "mean_P": float(mP)}


def achieved_power(n, cv_L, cv_P, corr, alpha, ratios=(1.0, 0.95), margin_ratio=MARGIN_RATIO):
    """Analytic Student delta-method TOST power (``mcstat.gate5.gate5_power``) at the given true ratios, at level ``alpha`` per side."""
    out = {"alpha": alpha}
    for r in ratios:
        try:
            out["ratio_%g" % r] = float(_mg5.gate5_power(n, cv_L, cv_P, corr, alpha=alpha, ratio=r, margin_ratio=margin_ratio))
        except ValueError as e:  # inconsistent CVs / correlation, n < 3
            out["ratio_%g" % r] = None
            out["note"] = str(e)
    return out


def _delta_method(L, P, level, margin_ratio):
    r = _mg5.gate5_paired(L, P, level=level, margin_ratio=margin_ratio)
    return {"ratio": float(r.ratio), "lo": float(r.ratio_lo), "hi": float(r.ratio_hi), "p_tost": float(r.p_tost),
            "verdict": str(r.verdict), "se_log": float(r.se), "dof": int(r.dof)}


# ----------------------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------------------
def _amplitude(cfg, amp, spec, data, summary, rng_seed):
    m, fwer, margin, level = cfg["family_m"], cfg["fwer"], cfg["margin_ratio"], cfg["level"]
    key = cfg["key"]
    sc, L_all = arm_series(data, spec["consumer"], key)
    sp, P_all = arm_series(data, spec["par2"], key)
    seeds, L, P, info = pair_by_seed(sc, L_all, sp, P_all)
    n = int(L.size)
    out = {"amplitude": amp, "arms": {"consumer": spec["consumer"], "par2": spec["par2"]}, "key": key, "n": n, "pairing": info}
    pos = n >= 3 and L.mean() > 0 and P.mean() > 0
    if not pos:
        out.update(procedure=None, p_tost=1.0, verdict="undefined", note="fewer than 3 pairs or a non-positive mean: Gate 5 descriptive, p = 1")
        return out
    corr = float(np.corrcoef(L, P)[0, 1])
    cv_L, cv_P = float(L.std(ddof=1) / L.mean()), float(P.std(ddof=1) / P.mean())
    out.update(mean_L=float(L.mean()), mean_P=float(P.mean()), ratio=float(L.mean() / P.mean()), cv_L=cv_L, cv_P=cv_P, corr=corr)
    sel = select_procedure(summary, n, m, fwer) if summary is not None else \
        {"procedure": None, "grid_n": None, "n": n, "table": {}, "problems": {}, "reason": "no coverage summary supplied"}
    out["selection"] = sel
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], zlib.crc32(amp.encode())]))
    F = Fieller(L, P, B=cfg["B"], rng=rng, ref=cfg["ref"])
    tail = 0.5 * (1.0 - level)
    by = {}
    for proc in ("student",) + PROCS:
        ph, pl, pt = p_tost(F, proc, margin)
        by[proc] = {"p_hi": ph, "p_lo": pl, "p_tost": pt}
    out["p_by_procedure"] = by
    chosen = sel["procedure"]
    out["procedure"] = chosen
    out["p_tost"] = by[chosen]["p_tost"] if chosen else 1.0
    desc_proc = chosen if chosen else "student"
    lo_t, hi_t = F.limits(desc_proc, tail)
    out["ci"] = {"level": level, "procedure": desc_proc, "lo": lo_t, "hi": hi_t, "descriptive": chosen is None,
                 "note": "bootstrap-t at half the level: tails %.4g each side (a %.1f %% interval), conservative" % (tail / 2, 100 * (1 - tail))
                 if chosen == "boott_half" else None}
    out["verdict"] = verdict(lo_t, hi_t, margin)
    if chosen is None:
        out["note"] = "no Fieller-form procedure is qualified at n = %d (grid %s): Gate 5 descriptive, its p-value is 1" % (n, sel.get("grid_n"))
    sl, su = F.limits("student", tail)
    out["student_fieller"] = {"lo": sl, "hi": su, "p_tost": by["student"]["p_tost"], "verdict": verdict(sl, su, margin)}
    out["delta_method"] = _delta_method(L, P, level, margin)
    out["power"] = achieved_power(n, cv_L, cv_P, corr, fwer / m, ratios=(1.0, 0.95, round(float(L.mean() / P.mean()), 6)), margin_ratio=margin)
    um = spec.get("unmatched")
    if um:
        _, Lu = arm_series(data, um["consumer"], key)
        _, Pu = arm_series(data, um["par2"], key)
        out["unmatched"] = dict(unmatched(Lu[np.isfinite(Lu)], Pu[np.isfinite(Pu)], level, margin), arms=um)
    else:
        out["unmatched"] = dict(unmatched(*(np.asarray(v, float)[np.isfinite(v)] for v in (L_all, P_all)), level, margin),
                                arms="the paired arms, pairing ignored (all finite values of each)")
    return out


def config(prereg):
    g = dict(prereg["gate5"])
    return {"family_m": int(g.get("family_m", 8)), "fwer": float(g.get("fwer", 0.05)), "margin_ratio": float(g.get("margin_ratio", MARGIN_RATIO)),
            "level": float(g.get("level", 0.90)), "B": int(g.get("B", DEFAULT_B)), "seed": int(g.get("seed", 20261103)),
            "key": g.get("key", "alpha_LP"), "ref": g.get("ref", "t")}


def run(prereg, data):
    """Gate 5 at every amplitude of ``prereg['gate5']['amplitudes']`` (see the module docstring)."""
    g = prereg["gate5"]
    cfg = config(prereg)
    summary = data.get("coverage_summary")
    if summary is None and g.get("coverage_summary"):
        summary = load_coverage_summary(g["coverage_summary"])
    amps = {}
    for amp, spec in g["amplitudes"].items():
        amps[amp] = _amplitude(cfg, amp, spec, data, summary, cfg["seed"])
    return jsonable({"amplitudes": amps, "pvalues": {a: r["p_tost"] for a, r in amps.items()}, "holm_order": list(amps),
                     "config": cfg, "coverage_plan_hash": None if summary is None else summary.get("plan_hash")})
