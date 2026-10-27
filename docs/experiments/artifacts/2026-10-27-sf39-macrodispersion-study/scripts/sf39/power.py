"""SF-39 sample-size rules (contract C-NR; preregistration section 6, sections 1 and 4; F9).  Pure functions, no I/O.

Margins (section 4): transverse ``max(0.05|a|, 1.25e-3)``.  Ladder power: exact paired TOST at true difference 0.
"""
import math

import numpy as np
from scipy import integrate, stats

N_MIN, N_MAX, POWER_TARGET = 20, 500, 0.80
AMPLITUDES = ("s1", "s225", "s4")
ESTIMATORS = ("alpha_TAM", "alpha_TP")
LADDER_KEYS = ("N_narrow", "N_twin", "N_hmax", "N_R")
# narrow rungs (1/2, 1, 2 x production lambda/h) per amplitude, preregistration section 1
NARROW_ARMS = {"s1": ("s1_n05", "s1_n10", "s1_n20"), "s225": ("s225_n08", "s225_n16", "s225_n32"),
               "s4": ("s4_n16", "s4_n32", "s4_n64")}
WIDTH_ARMS = {"s4_w24": 30, "s4_w12": 30}  # F10; never scaled
REF_FLOOR = 30  # s4_tw and s4_n16 run max(N, 30) (R25)


def s_upper(s, n):
    """Upper one-sided 80 % confidence limit of an sd estimated from n samples (chi-square, n-1 d.o.f.)."""
    return s * math.sqrt((n - 1) / stats.chi2.ppf(0.20, n - 1))


def margin_t(a):
    """Transverse equivalence margin at ensemble mean ``a``."""
    return max(0.05 * abs(a), 1.25e-3)


def tost_power(N, margin, sigma):
    """P(|Dbar| + t_{0.95,N-1} S/sqrt(N) <= margin), Dbar~N(0,sigma^2/N), (N-1)S^2/sigma^2~chi2_{N-1}, independent.

    Quadrature over W=(N-1)S^2/sigma^2 of [2 Phi((margin - t S/sqrt N) sqrt N/sigma) - 1]_+ chi2.pdf(W); the range is clipped to
    where the bracket is positive (W < w*) and to the chi2 1-1e-13 quantile (an unclipped quad can return 0 for large margin/sigma).
    """
    if margin <= 0:
        return 0.0
    if sigma <= 0:
        return 1.0
    df = N - 1
    t = stats.t.ppf(0.95, df)
    rn = math.sqrt(N)
    wstar = df * (margin * rn / (t * sigma)) ** 2
    hi = min(wstar, stats.chi2.ppf(1 - 1e-13, df))

    def f(w):
        s = sigma * math.sqrt(w / df)
        return (2.0 * stats.norm.cdf((margin - t * s / rn) * rn / sigma) - 1.0) * stats.chi2.pdf(w, df)

    return min(1.0, integrate.quad(f, 0.0, hi, limit=400, epsabs=1e-14, epsrel=1e-11)[0])


def ladder_n(diff_sd, n_pilot, margin):
    """Smallest N in [20, 500] with tost_power(N, margin, s_upper(diff_sd, n_pilot)) >= 0.80; (500, True) if not reached."""
    su = s_upper(diff_sd, n_pilot)
    if tost_power(N_MAX, margin, su) < POWER_TARGET:
        return N_MAX, True
    lo, hi = N_MIN - 1, N_MAX  # invariant: power(hi) >= target; power is monotone in N (checked in the tests)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if tost_power(mid, margin, su) >= POWER_TARGET else (mid, hi)
    return hi, False


def nr_b(sd_alpha, n_pilot, alpha_pilot_mean):
    """Smallest N in [20, 500] with t_{0.975,N-1} s_U/sqrt(N) <= max(0.1|alpha_pilot_mean|, 2.5e-3); (500, True) if not reached."""
    su = s_upper(sd_alpha, n_pilot)
    target = max(0.1 * abs(alpha_pilot_mean), 2.5e-3)
    for N in range(N_MIN, N_MAX + 1):
        if stats.t.ppf(0.975, N - 1) * su / math.sqrt(N) <= target:
            return N, False
    return N_MAX, True


def derive(n):
    """Add N_A, N_run, par2_NR to a dict holding N_narrow, N_twin, N_hmax, N_R (section 1, Production counts)."""
    out = {k: n[k] for k in LADDER_KEYS}
    out["N_A"] = max(N_MIN, n["N_hmax"])
    out["N_run"] = max(n["N_R"], out["N_A"], n["N_twin"])
    out["par2_NR"] = n["N_R"]
    return out


def _sd(x):
    return float(np.std(np.asarray(x, dtype=float), ddof=1))


def amplitude_ns(data):
    """data = {estimator: {D_f, D_tw, d16_8, prod: arrays; mean_2x, mean_h16: floats}} for both estimators of one amplitude.

    Returns (dict with N_narrow, N_twin, N_hmax, N_R, N_A, N_run, par2_NR; sorted list of keys at the 500 cap without reaching
    their target).  Each estimator's n is the length of its own pilot array.
    """
    best = dict.fromkeys(LADDER_KEYS, 0)
    capped = set()

    def take(key, res):
        best[key] = max(best[key], res[0])
        if res[1]:
            capped.add(key)

    for est in ESTIMATORS:
        d = data[est]
        prod_mean = float(np.mean(d["prod"]))
        m_h = min(margin_t(d["mean_2x"]), margin_t(prod_mean))
        take("N_narrow", ladder_n(_sd(d["D_f"]), len(d["D_f"]), m_h))
        take("N_twin", ladder_n(_sd(d["D_tw"]), len(d["D_tw"]), 2 ** 1.5 * m_h))
        take("N_hmax", ladder_n(_sd(d["d16_8"]), len(d["d16_8"]), margin_t(d["mean_h16"])))
        take("N_R", nr_b(_sd(d["prod"]), len(d["prod"]), prod_mean))
    out = derive(best)
    # a capped component is 500, the largest possible value, so it sets every derived value it feeds
    if "N_hmax" in capped:
        capped.add("N_A")
    if capped & {"N_R", "N_hmax", "N_twin"}:
        capped.add("N_run")
    if "N_R" in capped:
        capped.add("par2_NR")
    return out, sorted(capped)


def arm_counts(n, amp):
    """Total realizations per arm for one amplitude from its N dict (pilot realizations included)."""
    c = {f"{amp}_A": n["N_A"], f"{amp}_B": n["N_run"] - n["N_A"], f"{amp}_tw": n["N_twin"], f"{amp}_par2": n["par2_NR"]}
    for arm in NARROW_ARMS[amp]:
        c[arm] = n["N_narrow"]
    if amp == "s4":
        c["s4_tw"] = max(n["N_twin"], REF_FLOOR)
        c["s4_n16"] = max(n["N_narrow"], REF_FLOOR)
        c.update(WIDTH_ARMS)
    return c


def projected_gpu_h(N, costs, done):
    """sum over arms of (count - done)^+ * GPU-s per realization / 3600.  costs: arm -> GPU-s ('<c>_A' and '<c>_B' distinct)."""
    tot = 0.0
    for amp in N:
        for arm, cnt in arm_counts(N[amp], amp).items():
            tot += max(0, cnt - done.get(arm, 0)) * costs[arm]
    return tot / 3600.0


def _scale(n, k):
    """max(20, ceil(k/100 N)) in integer arithmetic (no float ceil artefacts), then recompute N_A, N_run."""
    return derive({key: max(N_MIN, -(-k * n[key] // 100)) for key in LADDER_KEYS})


def allocate(N, costs, done, budget_gpu_h):
    """Budget rule (section 6).  N: {amp: dict with N_narrow, N_twin, N_hmax, N_R}; done: arm -> realizations already run.

    Over budget: (1) the s4 ladders (narrow, twin, h_max) go to 20; (2) the largest f in {1.00, ..., 0.01} scaling every remaining N
    (N_R, ladders, Par2) as max(20, ceil(f N)).  Width arms and the 30 s4 reference realizations never scale.
    Returns dict(N, f, s4_ladders_to_20, projected_gpu_h, projected_gpu_h_unscaled, owner); owner=True when even f=0.01 does not fit.
    """
    cur = {a: derive(N[a]) for a in N}
    p0 = projected_gpu_h(cur, costs, done)
    res = dict(N=cur, f=1.0, s4_ladders_to_20=False, projected_gpu_h=p0, projected_gpu_h_unscaled=p0, owner=False)
    if p0 <= budget_gpu_h:
        return res
    if "s4" in cur:
        cur["s4"] = derive({**cur["s4"], "N_narrow": N_MIN, "N_twin": N_MIN, "N_hmax": N_MIN})
    res.update(N=cur, s4_ladders_to_20=True, projected_gpu_h=projected_gpu_h(cur, costs, done))
    if res["projected_gpu_h"] <= budget_gpu_h:
        return res
    for k in range(100, 0, -1):
        sc = {a: _scale(cur[a], k) for a in cur}
        p = projected_gpu_h(sc, costs, done)
        res.update(N=sc, f=k / 100, projected_gpu_h=p)
        if p <= budget_gpu_h:
            return res
    res["owner"] = True
    return res
