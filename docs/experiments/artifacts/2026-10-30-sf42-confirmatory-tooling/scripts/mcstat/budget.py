"""Error budget and robust lower limit of decision G5.4 (docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md).

For a transverse estimator ``e`` and level ``a``::

    RL_e(a) = L_e(a / 2) - sum_{k=1..K} B_k(a / (2 K))

``L_e`` is the one-sided sampling lower limit of the production mean (a *sampling* object: ``StudentMean``,
``BootstrapTMean``); ``B_k(t)`` is the one-sided upper confidence bound, at tail ``t``, of ``|bias_k|`` for each of the ``K`` budget
components (a *component* object: ``ContrastBound`` for ``E_int`` / ``E_W`` / the width x lambda/h interaction, ``LadderBound`` for
``E_h``).  Union bound: ``P(RL_e(a) > alpha_e) <= a`` under any dependence between the pieces, conditional on the stated transfer
assumptions.  The p-value is ``p_e = inf{a : RL_e(a) > 0}`` (``RobustLimit.p_value``); the amplitude p-value
``max(p_TP, p_TAP)`` (intersection-union) is ``intersection_union``.

Bootstrap over seeds
--------------------
Every component draws ``B`` resamples of SEEDS with replacement.  Arms that share seeds (a grid ladder, the ``h_max`` passes, the two
resolutions inside one width) share the resampled indices ("pairing group"); arms that do not (the width arms: the field depends on
the box) are resampled independently.  The component's statistic is recomputed on every resample from the arm means.  Upper
confidence bounds of ``|Delta|`` are the ``(1 - tau)`` percentile of ``|Delta*|`` (rank rule of ``_util``); for ``Delta_hat ~ N(Delta,
s^2)`` and ``Delta*`` drawn around ``Delta_hat`` this percentile bound has coverage ``>= 1 - tau`` for every ``Delta`` (the failure
event implies a standard-normal tail below ``tau``); for finite ``n`` and skewed data it is an approximation, which the coverage
simulation of SF-42 measures for the whole robust-sign pipeline.  A component built from two estimated inputs (``E_h``: the difference and the order) takes each input at tail
``t / 2``.

Studentized alternatives (``method=``, SF-42 N6c)
-----------------------------------------------
The percentile bound above is NOT studentized: on normal data at ``n = 30`` the 95 % upper limit of its miss rate is 1.4-2.9 times the
tail at the tails of the budget (coverage simulation of N6b, repeated in N6c; the miss rate itself is 1.4-2.7 times the tail; analytic:
``P(t_{n-1} < -z sqrt((n-1)/n))``).  Every contrast bound and ``LadderBound`` therefore
take ``method``: ``'percentile'`` (default; bit-for-bit the original), ``'student'`` or ``'bootstrap-t'``, which replace the percentile
bound of ``|Delta|`` by ``U(t) = max(UCL_t, -LCL_t)`` built from studentized one-sided limits of ``Delta`` (``intervals.abs_upper_bound``:
validity argument, Welch-Satterthwaite standard error over the pairing groups, rank rule of the bootstrap-t).  ``E_h`` keeps the
order lower limit ``p_L`` as the percentile of the bootstrap orders in every method (``rng=`` / ``seed=`` is therefore still
needed by ``LadderBound``; ``ContrastBound`` needs it only for ``'percentile'`` and ``'bootstrap-t'``).  The fallback range of
``E_h``, ``UCB(max_ij |m_i - m_j|)``, becomes ``max_ij U_ij(t / P)`` over the ``P = K (K - 1) / 2`` pairs of levels (per-seed pair
differences, studentized as above), a Bonferroni split that is valid under any dependence between the pairs.  (Remark: because the
true range is attained at one fixed, unknown pair, ``max_ij U_ij(t)`` is already a valid bound; ``t / P`` is the more conservative
choice that was specified.)

Monotone bounds
---------------
``B_k(t)`` is a bound at confidence level ``1 - t``: it must be NON-INCREASING in the tail ``t`` (a more confident bound is never
smaller).  Raw bounds are monotone except where ``E_h`` switches between its branches (the branch depends on the lower limit of
the observed order, which depends on the tail).  ``RobustLimit`` therefore enforces
``B~(t) = max(B(t), max over lattice tails t' >= t of B(t'))``: a larger bound is always valid, and this one is non-increasing in
``t`` on the lattice (guaranteed by construction; between lattice points the bound is ``max(raw(t), suffix maximum)``, monotone wherever the
raw bound is; tested on a fine grid).  With a sampling limit non-decreasing in ``a`` this makes ``RL(a)`` non-decreasing, so
``{a : RL(a) > 0}`` is an up-set and the p-value is a monotone search.

p-value precision
-----------------
``p_e`` is searched on a log lattice of ``a`` (default 200 points per decade between 1e-8 and 1) and returned as the smallest
lattice value with ``RL > 0``: an upper bound of the infimum within a factor of ``10**(1/200) = 1.0116``.  If no lattice value
gives ``RL > 0`` the p-value is ``1.0`` (``never_positive=True``).  If the smallest such value is the first one at which the bounds
are resolvable by ``B`` resamples (rank rule), the true infimum may lie lower: ``resolution_limited=True``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._util import as_levels, as_tails, bootstrap_means, get_rng, lcb_sorted, ucb_sorted
from .intervals import (abs_upper_bound, boot_tstar, bootstrap_t, bootstrap_t_limit, check_bound_method, student_limit,
                        tstar_from_moments, welch_groups)

LATTICE_PER_DECADE = 200
LATTICE_A_MIN = 1e-8


def default_lattice(per_decade=LATTICE_PER_DECADE, a_min=LATTICE_A_MIN):
    """Log-spaced levels ``a`` from ``a_min`` to exactly 1 (inclusive), ``per_decade`` points per decade."""
    ndec = -np.log10(a_min)
    n = int(round(ndec * per_decade)) + 1
    a = 10.0 ** np.linspace(np.log10(a_min), 0.0, n)
    a[-1] = 1.0
    return a


# ----------------------------------------------------------------------------------------------------------------------
# Sampling limits L_e(tail) of the production mean
# ----------------------------------------------------------------------------------------------------------------------
class StudentMean:
    """Student-t sampling limits of the mean of ``x`` (..., n)."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        self.mean = self.x.mean(axis=-1)

    def lower(self, tails):
        return student_limit(self.x, tails, "lower")

    def upper(self, tails):
        return student_limit(self.x, tails, "upper")


class BootstrapTMean:
    """Bootstrap-t sampling limits of the mean of ``x`` (..., n) (``intervals.bootstrap_t``; ``B`` must resolve the tails used)."""

    def __init__(self, x, B=4000, rng=None, seed=None, share_draws=False):
        self.x = np.asarray(x, dtype=float)
        self.mean = self.x.mean(axis=-1)
        self.bt = bootstrap_t(self.x, B, rng, seed, share_draws)

    def lower(self, tails):
        return bootstrap_t_limit(self.bt, tails, "lower")

    def upper(self, tails):
        return bootstrap_t_limit(self.bt, tails, "upper")


# ----------------------------------------------------------------------------------------------------------------------
# Budget components: B_k(t) = upper confidence bound at tail t of |bias_k|
# ----------------------------------------------------------------------------------------------------------------------
class ContrastBound:
    """Bound of ``factor * |sum_j c_j mean(arm_j)|`` (E_int, E_W, interaction) by a joint seed bootstrap.

    ``terms``: list of ``(values, group, coeff)``; ``values`` (..., n_group) per-seed values of one arm; arms with the same
    ``group`` label must have the same ``n`` and the same seed order (they share the resampled indices); different groups are
    resampled independently.  ``method='percentile'`` (default): ``bound(t) = factor * (1 - t)-percentile of |sum_j c_j mean*_j|``.
    ``method='student'`` / ``'bootstrap-t'``: ``bound(t) = factor * U(t)``, the studentized upper bound of ``|Delta|`` of
    ``intervals.abs_upper_bound`` (module docstring); ``'student'`` needs no random numbers.
    """

    def __init__(self, name, terms, factor=1.0, B=2000, rng=None, seed=None, share_draws=False, method="percentile"):
        check_bound_method(method)
        rng = get_rng(rng, seed) if method != "student" else None
        self.name = name
        self.factor = float(factor)
        self.B = B
        self.method = method
        combined = {}  # group -> per-seed combination sum_j c_j v_j: arms of a group share the draws and the mean is linear
        for values, group, coeff in terms:
            v = coeff * np.asarray(values, dtype=float)
            if group in combined:
                if combined[group].shape[-1] != v.shape[-1]:
                    raise ValueError("arms of pairing group %r must have the same number of seeds" % (group,))
                combined[group] = combined[group] + v
            else:
                combined[group] = v
        if method == "percentile":
            est = 0.0
            acc = 0.0
            for u in combined.values():
                est = est + u.mean(axis=-1)
                acc = acc + bootstrap_means((u,), B, rng, share_draws)[0]
            self.estimate = self.factor * np.abs(est)
            self._sorted = np.sort(self.factor * np.abs(acc), axis=-1)
            return
        groups = list(combined.values())
        self._delta, self._se, self._nu = welch_groups(groups)
        self._tstar = boot_tstar(groups, B, rng, share_draws) if method == "bootstrap-t" else None
        self.estimate = self.factor * np.abs(self._delta)

    def bound(self, tails):
        if self.method == "percentile":
            return ucb_sorted(self._sorted, tails)
        return self.factor * abs_upper_bound(self._delta, self._se, tails, self._nu, self._tstar)

    def details(self, tails):
        d = {"estimate": self.estimate, "bound": self.bound(tails), "method": self.method}
        if self.method != "percentile":
            d.update({"se": self.factor * self._se, "nu": self._nu})
        return d


def e_int(alpha_h16, alpha_h8, factor=2.0, B=2000, rng=None, seed=None, share_draws=False, method="percentile"):
    """Integration bound (G5.4): ``E_int = 2 UCB|alpha(h/16) - alpha(h/8)|`` from the paired ``h_max`` passes (order 1 assumed)."""
    return ContrastBound("E_int", [(alpha_h16, "passes", 1.0), (alpha_h8, "passes", -1.0)], factor, B, rng, seed, share_draws, method)


def e_w(alpha_wide, alpha_prod_width, B=2000, rng=None, seed=None, share_draws=False, method="percentile"):
    """Width bound (G5.4): ``E_W = UCB|alpha(W*) - alpha(W_prod)|``, the two widths UNPAIRED (the field depends on the box)."""
    return ContrastBound("E_W", [(alpha_wide, "wide", 1.0), (alpha_prod_width, "prod", -1.0)], 1.0, B, rng, seed, share_draws, method)


def e_interaction(wide_fine, wide_coarse, prod_fine, prod_coarse, B=2000, rng=None, seed=None, share_draws=False,
                  method="percentile"):
    """Width x lambda/h interaction bound for a measured 2x2 (G5.4 "where measured").

    Definition: ``I = [alpha(W*, h_f) - alpha(W*, h_c)] - [alpha(W_prod, h_f) - alpha(W_prod, h_c)]``, the change of the
    resolution effect between the two widths (equivalently the change of the width effect between the two resolutions); it bounds
    the error of transferring the width effect ``E_W`` measured at the cheap resolution ``h_c`` to the fine one ``h_f``.  Within a
    width both resolutions use the same seeds (shared indices); the two widths are independent groups.
    """
    return ContrastBound("E_WxH", [(wide_fine, "wide", 1.0), (wide_coarse, "wide", -1.0),
                                   (prod_fine, "prod", -1.0), (prod_coarse, "prod", 1.0)], 1.0, B, rng, seed, share_draws, method)


class LadderBound:
    """Discretization bound ``E_h`` of the production level from a ratio-``r`` grid ladder (G5.4).

    ``levels``: (..., n, K) per-seed values on the SAME seeds, ordered coarse -> fine (decreasing ``h``); ``resolutions``: (K,)
    increasing ``lambda/h`` (ratio ``r = res[i+1] / res[i]``); ``ip``: index of the production level.  Means ``m_i`` (resample
    means ``m*_i``), consecutive differences ``d_i = m_{i+1} - m_i``.

    * Observed order ``p``: a REAL order exists iff the ladder has ``>= 3`` levels, constant ratio ``r``, and all ``d_i`` are
      nonzero with one sign (monotone); then ``p = -slope(log|d_i| against i) / ln r`` (for 3 levels ``log(|d_0|/|d_1|)/ln r``).
      Non-monotone or oscillating ladders have no real order.
    * ``p_L(tau)``: the ``tau``-quantile (lower limit) of the bootstrap orders, where resamples without a real order enter as
      ``-inf`` (they count as outside the pre-registered range ``[p_min, p_max]``).
    * If ``p_min <= p_L <= p_max`` (each input of this branch at tail ``t / 2``): finer level exists (``ip < K - 1``),
      ``E_h = safety * UCB|m_{ip+1} - m_{ip}| * r^p_L / (r^p_L - 1)``; production finest (``ip = K - 1``),
      ``E_h = safety * UCB|m_{ip} - m_{ip-1}| / (r^p_L - 1)`` (asymptotic range assumed).  With ``e(h) = C h^p``: the first is
      ``|e(h_p)|`` from the next-finer difference, the second ``|e(h_p)|`` from the next-coarser one.
    * Otherwise (also ``K < 3``, unequal ratios, non-monotone, oscillating, order outside the range):
      ``E_h = fallback_factor * UCB(max_ij |m_i - m_j|) / (r_min - 1)`` (single input, full tail ``t``).

    ``method`` selects the upper bounds of the differences (``|m_{ip+1} - m_{ip}|`` and the range); ``'percentile'`` (default) is
    the original percentile bootstrap of the means.  ``'student'`` / ``'bootstrap-t'`` use the studentized ``U(t)`` of
    ``intervals.abs_upper_bound`` on the per-seed difference of the two levels (a paired contrast of one group); the range
    becomes ``max_ij U_ij(t / P)`` over the ``P = K (K - 1) / 2`` pairs (Bonferroni).  The order lower limit ``p_L`` is the
    percentile of the bootstrap orders in all methods, so ``rng`` / ``seed`` is always required.  ``'bootstrap-t'`` resamples the
    seeds once for the levels and the pair differences (same indices).
    """

    def __init__(self, levels, resolutions, ip, p_min=0.5, p_max=4.0, safety=1.25, fallback_factor=3.0,
                 B=2000, rng=None, seed=None, share_draws=False, method="percentile"):
        self.name = "E_h"
        check_bound_method(method)
        self.method = method
        rng = get_rng(rng, seed)
        x = np.asarray(levels, dtype=float)
        res = np.asarray(resolutions, dtype=float)
        K = x.shape[-1]
        if res.shape != (K,) or K < 2 or not np.all(np.diff(res) > 0):
            raise ValueError("resolutions must be K increasing values, one per level (K >= 2)")
        if not 0 <= ip < K:
            raise ValueError("ip out of range")
        if not (0.0 < p_min <= p_max):
            raise ValueError("need 0 < p_min <= p_max")
        ratios = res[1:] / res[:-1]
        self.r_min = float(ratios.min())
        self.const_ratio = bool(np.allclose(ratios, ratios[0], rtol=1e-9))
        self.r = float(ratios[0])
        self.ip, self.K, self.p_min, self.p_max = ip, K, p_min, p_max
        self.safety, self.fallback_factor, self.B = safety, fallback_factor, B
        m_hat = x.mean(axis=-2)  # (..., K)
        if method != "percentile":
            iu, ju = np.triu_indices(K, 1)
            u = np.moveaxis(x[..., ju] - x[..., iu], -1, -2)  # (..., P, n): per-seed difference m_j - m_i of every pair i < j
        pair_tstar = None
        if method == "bootstrap-t":  # one resampling for the level means and the moments of every pair difference
            uc = u - u.mean(axis=-1, keepdims=True)
            stack = np.concatenate([np.moveaxis(x, -1, -2), uc, uc * uc], axis=-2)  # (..., K + 2P, n)
            boot = bootstrap_means((stack,), B, rng, share_draws, inner_ndim=1)[0]  # (..., K + 2P, B)
            P = iu.size
            m_star = np.moveaxis(boot[..., :K, :], -2, -1)  # (..., B, K)
            pair_tstar = tstar_from_moments(boot[..., K:K + P, :], boot[..., K + P:, :], x.shape[-2])
        else:
            m_star = np.moveaxis(bootstrap_means((np.moveaxis(x, -1, -2),), B, rng, share_draws, inner_ndim=1)[0], -2, -1)  # (..., B, K)
        self.order_hat = self._order(m_hat)
        p_star = self._order(m_star)
        p_star = np.where(np.isnan(p_star), -np.inf, p_star)
        self._p_sorted = np.sort(p_star, axis=-1)
        j = ip + 1 if ip < K - 1 else ip - 1
        self._finer = ip < K - 1
        self.delta_hat = np.abs(m_hat[..., j] - m_hat[..., ip])
        self.range_hat = m_hat.max(axis=-1) - m_hat.min(axis=-1)
        if method == "percentile":
            self._delta_sorted = np.sort(np.abs(m_star[..., j] - m_star[..., ip]), axis=-1)
            self._range_sorted = np.sort(m_star.max(axis=-1) - m_star.min(axis=-1), axis=-1)
            return
        self._P = iu.size
        self._pd, self._pse, self._pnu = welch_groups([u])  # (..., P) each: estimate, standard error, degrees of freedom
        self._pt = pair_tstar
        a = int(np.flatnonzero((iu == min(ip, j)) & (ju == max(ip, j)))[0])  # the pair (ip, ip +- 1) of the adjacent difference
        self._adj = (self._pd[..., a], self._pse[..., a], self._pnu[..., a], None if pair_tstar is None else pair_tstar[..., a, :])

    def _order(self, m):
        """Observed order from level means ``m`` (..., K); NaN when there is no real order."""
        if self.K < 3 or not self.const_ratio:
            return np.full(m.shape[:-1], np.nan)
        d = np.diff(m, axis=-1)
        same = np.all(d > 0, axis=-1) | np.all(d < 0, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            logd = np.log(np.abs(d))
        i = np.arange(d.shape[-1], dtype=float)
        ic = i - i.mean()
        slope = (logd - logd.mean(axis=-1, keepdims=True)) @ ic / (ic @ ic)
        p = -slope / np.log(self.r)
        return np.where(same & np.isfinite(p), p, np.nan)

    def _parts(self, tails):
        tau, scalar = as_tails(tails)
        half = 0.5 * tau
        p_L = lcb_sorted(self._p_sorted, half)
        if self.method == "percentile":
            dU = ucb_sorted(self._delta_sorted, half)
            rU = ucb_sorted(self._range_sorted, tau)
        else:  # studentized: adjacent difference at t / 2; range = max over the P pairs at t / P (Bonferroni)
            dU = abs_upper_bound(*self._adj[:2], half, self._adj[2], self._adj[3])
            rU = abs_upper_bound(self._pd, self._pse, tau / self._P, self._pnu, self._pt).max(axis=-2)
        in_range = np.isfinite(p_L) & (p_L >= self.p_min) & (p_L <= self.p_max)
        p_use = np.where(in_range, p_L, self.p_min)
        rp = self.r ** p_use
        with np.errstate(invalid="ignore", divide="ignore"):
            if self._finer:
                order_branch = self.safety * dU * rp / (rp - 1.0)
            else:
                order_branch = self.safety * dU / (rp - 1.0)
            fallback = self.fallback_factor * rU / (self.r_min - 1.0)
        bound = np.where(in_range, order_branch, fallback)
        return scalar, bound, p_L, in_range, order_branch, fallback

    def bound(self, tails):
        scalar, bound = self._parts(tails)[:2]
        return bound[..., 0] if scalar else bound

    def details(self, tails):
        """Intermediate quantities for reporting (per tail): branch used, ``p_L``, both candidate bounds."""
        scalar, bound, p_L, in_range, order_branch, fallback = self._parts(tails)
        sq = (lambda v: v[..., 0]) if scalar else (lambda v: v)
        return {"method": self.method, "order_hat": self.order_hat, "delta_hat": self.delta_hat, "range_hat": self.range_hat,
                "p_L": sq(p_L), "order_branch_used": sq(in_range), "bound": sq(bound),
                "order_branch_value": sq(order_branch), "fallback_value": sq(fallback)}


# ----------------------------------------------------------------------------------------------------------------------
# Robust lower limit and p-value
# ----------------------------------------------------------------------------------------------------------------------
class PValue(NamedTuple):
    p: np.ndarray
    resolution_limited: np.ndarray
    never_positive: np.ndarray


class RobustLimit:
    """``RL(a) = L(a/2) - sum_k B~_k(a / (2K))`` with monotone-enforced bounds ``B~_k`` (see module docstring).

    ``sampling``: object with ``lower(tails)`` (``StudentMean`` / ``BootstrapTMean``); ``components``: list of objects with
    ``bound(tails)`` (``ContrastBound`` / ``LadderBound`` / anything with that method returning shape ``batch + (T,)``).
    Everything is vectorized over the leading replicate axes of the underlying data.  Memory: a component holds ``R * B`` floats
    (``R * B * K`` while a ladder is built), so build the components of at most a few hundred to a few thousand replicates per call.
    """

    def __init__(self, sampling, components, lattice=None):
        self.sampling = sampling
        self.components = list(components)
        if not self.components:
            raise ValueError("a robust limit needs at least one budget component (K >= 1)")
        self.K = len(self.components)
        self.lattice = default_lattice() if lattice is None else np.asarray(lattice, dtype=float)
        if np.any(np.diff(self.lattice) <= 0) or self.lattice[-1] != 1.0:
            raise ValueError("the lattice must be increasing and end at 1.0")
        self._suffix = None

    def _lattice_suffix(self):
        """Per component: suffix maximum over the lattice of the raw bounds (non-increasing in the tail)."""
        if self._suffix is None:
            t = self.lattice / (2.0 * self.K)
            self._suffix = []
            for c in self.components:
                raw = np.asarray(c.bound(t))
                self._suffix.append(np.maximum.accumulate(raw[..., ::-1], axis=-1)[..., ::-1])
        return self._suffix

    def component_bounds(self, a):
        """Enforced bounds ``B~_k(a / (2K))``: array (..., K, A) for levels ``a`` (scalar -> A dropped)."""
        a_arr, scalar = as_levels(a)
        suf = self._lattice_suffix()
        idx = np.minimum(np.searchsorted(self.lattice, a_arr, side="left"), self.lattice.size - 1)
        outs = []
        for c, s in zip(self.components, suf):
            raw = np.asarray(c.bound(a_arr / (2.0 * self.K)))
            outs.append(np.maximum(raw, s[..., idx]))
        out = np.stack(outs, axis=-2)
        return out[..., 0] if scalar else out

    def lower(self, a):
        """``RL(a)``: shape batch + (A,) (scalar ``a`` -> no A axis)."""
        a_arr, scalar = as_levels(a)
        L = np.asarray(self.sampling.lower(a_arr / 2.0))
        with np.errstate(invalid="ignore"):
            out = L - self.component_bounds(a_arr).sum(axis=-2)
        return out[..., 0] if scalar else out

    def p_value(self):
        """``p = inf{a : RL(a) > 0}`` on the lattice (see module docstring for the precision statement)."""
        a = self.lattice
        suf = self._lattice_suffix()
        L = np.asarray(self.sampling.lower(a / 2.0))
        with np.errstate(invalid="ignore"):
            RL = L - sum(suf)
        pos = RL > 0.0
        any_pos = pos.any(axis=-1)
        first = np.argmax(pos, axis=-1)
        finite = np.isfinite(RL)
        first_fin = np.argmax(finite, axis=-1)
        p = np.where(any_pos, a[first], 1.0)
        limited = any_pos & (first == first_fin) & (first > 0)
        return PValue(p, limited, ~any_pos)

    def report(self, a):
        """All intermediate quantities at level(s) ``a``: sampling limit, raw and enforced bounds per component, ``RL``."""
        a_arr, scalar = as_levels(a)
        t = a_arr / (2.0 * self.K)
        enforced = self.component_bounds(a_arr)
        rep = {"a": a_arr, "K": self.K, "L_a_over_2": np.asarray(self.sampling.lower(a_arr / 2.0)),
               "components": {}, "RL": self.lower(a_arr)}
        for k, c in enumerate(self.components):
            entry = {"raw": np.asarray(c.bound(t)), "enforced": enforced[..., k, :]}
            if hasattr(c, "details"):
                entry["details"] = c.details(t)
            rep["components"][getattr(c, "name", "component%d" % k)] = entry
        return rep


def intersection_union(*p_values):
    """Amplitude p-value ``max(p_TP, p_TAP)`` (intersection-union test: all null hypotheses must be rejected)."""
    return np.maximum.reduce([np.asarray(p, dtype=float) for p in p_values])


def is_non_increasing_in_tail(bound_fn, tails):
    """Verification helper: is ``bound_fn(tails)`` non-increasing along increasing ``tails`` (all replicates)?"""
    v = np.asarray(bound_fn(np.sort(np.asarray(tails, dtype=float))))
    with np.errstate(invalid="ignore"):  # inf - inf where the tail is unresolvable by B resamples
        d = np.diff(v, axis=-1)
    d = d[~np.isnan(d)]
    return bool(np.all(d <= 0.0))
