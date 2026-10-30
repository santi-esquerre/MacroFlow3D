"""Blinded variance re-estimation (decision G5.2): "One blinded variance re-estimation is allowed (variance only, upward only, at most
2x, at about 25 % of N; the script outputs standard deviations, never means)".

Contract.  The functions take per-arm per-realization values (the interim data) and return ONLY standard deviations, the numbers of
seeds, and the re-sized ``N`` computed from the standard deviations and from quantities fixed in advance (the original ``N``, the
pre-registered absolute half-width targets or planning means).  They never return, store or print a mean, a quantile or any other
function of the location of the data: every returned number is invariant under adding a constant to an arm (tested by shifting
the inputs; the Gate 5 variant additionally returns the correlation, which is also location-free).  Nothing is printed.

Re-sizing rule: ``N_new = min(max(N_required, N_orig), floor(2 N_orig))`` (upward only, at most doubling), with ``N_required`` from the
SD inflated to its 80 % upper confidence bound (``sizing.n_absolute_target``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gate5 import gate5_sample_size
from .sizing import CONF80, clamp_resize, corr_lower, n_absolute_target, sd_upper


def _clean(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size < 4 or not np.all(np.isfinite(a)):
        raise ValueError("arm %r: need a finite 1-D array with at least 4 values" % (name,))
    return a


@dataclass(frozen=True)
class BlindedResult:
    """Only standard deviations and sample sizes (no field holds a location statistic)."""
    sd: dict = field(default_factory=dict)
    n_obs: dict = field(default_factory=dict)
    n_required: dict = field(default_factory=dict)
    n_new: dict = field(default_factory=dict)


def blinded_sds(arms):
    """Sample standard deviations (ddof = 1) of per-arm values (a paired contrast is passed as its per-seed differences)."""
    return {k: float(_clean(v, k).std(ddof=1)) for k, v in arms.items()}


def blinded_reestimate(arms, n_orig, half_widths, level=0.95, conf=CONF80, max_factor=2.0):
    """Blinded re-sizing of arms with absolute half-width targets.

    ``arms``: ``{name: 1-D per-realization values}``; ``n_orig``: int or ``{name: N}``; ``half_widths``: ``{name: pre-registered
    absolute half-width}`` (arms without a target get their SD only); ``level``: two-sided confidence level of the half-width;
    ``conf``: confidence of the SD upper bound (80 %).  Returns a ``BlindedResult``.
    """
    sd, n_obs, req, new = {}, {}, {}, {}
    for k, v in arms.items():
        a = _clean(v, k)
        sd[k] = float(a.std(ddof=1))
        n_obs[k] = int(a.size)
        if k in half_widths:
            n0 = n_orig[k] if isinstance(n_orig, dict) else n_orig
            req[k] = int(n_absolute_target(sd[k], a.size, half_widths[k], level, conf))
            new[k] = clamp_resize(req[k], n0, max_factor)
    return BlindedResult(sd, n_obs, req, new)


@dataclass(frozen=True)
class BlindedGate5Result:
    """Standard deviations, the (location-free) pairing correlation and the re-sized number of paired seeds."""
    sd_L: float
    sd_P: float
    corr: float
    n_obs: int
    n_required: int
    n_new: int


def blinded_gate5(L, P, plan_mean_L, plan_mean_P, alpha, n_orig, calibrated_corr=True, conf=CONF80, max_factor=2.0):
    """Blinded re-sizing of the Gate 5 pairs (decision G5.5 sizing, G5.2 re-estimation).

    ``L``, ``P``: interim paired values (aligned by seed).  The interim means are NOT used: the coefficients of variation come
    from the SDs (80 % upper bounds) over the PRE-REGISTERED planning means ``plan_mean_*``; the correlation enters at its 80 %
    lower bound (``calibrated_corr=False`` uses 0).  Power 0.8 at ratio 1 at the one-sided level ``alpha`` (Holm worst level).
    """
    a, b = _clean(L, "L"), _clean(P, "P")
    if a.shape != b.shape:
        raise ValueError("L and P must be aligned by seed")
    n = a.size
    sd_L, sd_P = float(a.std(ddof=1)), float(b.std(ddof=1))
    corr = float(np.corrcoef(a, b)[0, 1])
    cv_L = float(sd_upper(sd_L, n, conf)) / plan_mean_L
    cv_P = float(sd_upper(sd_P, n, conf)) / plan_mean_P
    rho = float(corr_lower(corr, n, conf)) if calibrated_corr else 0.0
    n_req, _ = gate5_sample_size(cv_L, cv_P, rho, alpha)
    return BlindedGate5Result(sd_L, sd_P, corr, n, int(n_req), clamp_resize(n_req, n_orig, max_factor))
