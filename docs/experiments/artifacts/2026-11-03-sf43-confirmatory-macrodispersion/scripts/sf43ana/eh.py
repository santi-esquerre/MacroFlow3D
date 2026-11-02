"""Discretization bound ``E_h`` of the SF-43 pre-registration (H6): ``mcstat.LadderBound`` extended to several constant-ratio
triples, a per-triple order range test, an order cap, a second difference at the finest level and a gate on the asymptotic range.
``mcstat`` is not modified; its helpers are reused.

Definitions (``m_i`` the means of the per-seed ladder values on COMMON seeds, ordered coarse -> fine; ``r`` the ratio of the
production level to its neighbour; each studentized upper bound ``U`` is ``mcstat.intervals.abs_upper_bound`` with the Student
option on the per-seed difference of the two levels; this is the only method SF-42's coverage simulation qualified for the
contrasts):

* order of a triple ``(i, j, k)`` of constant ratio ``rt``: ``p = log(|m_j - m_i| / |m_k - m_j|) / log(rt)`` when both differences
  are nonzero with one sign, else no real order (non-monotone / oscillating -> outside the range);
* ``p_L(tau)`` of a triple: lower ``tau/2`` quantile of its bootstrap orders (resamples without a real order count as ``-inf``);
* ORDER RANGE TEST PER TRIPLE: every triple's ``p_L`` must lie in ``[p_min, p_max]`` (default ``[0.5, 2.1]``; a triple without a real
  order, ``p_L = -inf``, is outside); ``p_use = min(p_L over the triples, p_cap)`` (default cap 2);
* finer level exists (``ip < K - 1``):  ``E_h = safety * U(m_{ip+1} - m_ip) * r^p_use / (r^p_use - 1)``;
* production finest:  ``E_h = safety * max(U(m_ip - m_{ip-1}), r^-p_use U(m_{ip-1} - m_{ip-2})) / (r^p_use - 1)`` (the second
  term only when ``ip >= 2``; Giles 2015): under the power-law error model both terms estimate ``|e(h_p)|`` and either one valid
  suffices for the bound to hold, so the maximum is conservative; each ``U`` is taken at tail ``t/2`` and ``p_L`` at ``t/2``;
* otherwise (``p_L`` outside the range, fewer than 3 levels, no constant-ratio triple, non-monotone):
  ``E_h = fallback_factor * max over pairs of the FALLBACK LEVELS of U_ij(t / P) / (r_min - 1)``, ``P`` the number of those pairs
  (as ``LadderBound``); the fallback levels default to all levels and are configurable (H6: the ratio-2 levels).
* GATE ON THE ASYMPTOTIC RANGE (order branch only; H6 / numerics review B2).  Each gate is a paired seed bootstrap of the ladder
  means (the resamples of the ladder are used, with the ``extra`` level when a gate needs it) and compares the observed difference
  with 0 at the central ``gate_level`` (default 0.90): the gate passes iff at least ``(1 - gate_level) / 2`` of the resamples have
  ``diff <= 0`` AND at least that share have ``diff >= 0`` (resamples without a real order, or with a failing fit, are counted in
  neither, which can only make the gate fail), and the observed difference exists.  Types:
    ``extra_level``    the level ``resolution`` (a ``(n,)`` array ``extra``, same seeds) against the error law ``a_inf + C res^-p`` fitted
                       on the given ``triple`` (three means, constant ratio, ``p > 0``): ``diff = m_extra - a_fit(resolution)``;
    ``orders_agree``   the orders of the given ``triples`` (two or more): ``diff = p_a - p_b`` (first two).
  If a gate fails, ``E_h = max(order branch, fallback)``.

Because ``p_use <= p_L`` of the ladder's triple, the per-triple test only narrows the order branch, the finest branch only adds a
maximum and a failed gate takes the maximum with the fallback, the bound is at least ``LadderBound``'s (same range, same bootstrap
draws, ``method='student'``) for a three-level ladder in every branch; a unit test checks it on random ladders.

The signed Richardson correction used for the class ``numerical`` is ``correction(means, p)``: finer branch
``(m_{ip+1} - m_ip) r^p/(r^p - 1)``, finest ``(m_ip - m_{ip-1})/(r^p - 1)``, with ``p`` = ``p_use`` (``details(tail)['p_use']``) if the
order branch is in range, else 1.
"""
import warnings

import numpy as np

from mcstat._util import as_tails, bootstrap_means, get_rng, lcb_sorted
from mcstat.intervals import abs_upper_bound, welch_groups


def _fit_extra(m, triple, res, res_extra, m_extra):
    """diff = m_extra - a_fit(res_extra) with ``a_fit = a_inf + C res^-p`` through the three means of ``triple`` (NaN if no fit)."""
    i, j, k = triple
    d0, d1 = m[..., j] - m[..., i], m[..., k] - m[..., j]
    rt = res[j] / res[i]
    same = ((d0 > 0) & (d1 > 0)) | ((d0 < 0) & (d1 < 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (np.log(np.abs(d0)) - np.log(np.abs(d1))) / np.log(rt)
        good = same & np.isfinite(p) & (p > 0.0)
        c = d0 / (res[j] ** -p - res[i] ** -p)
        a_inf = m[..., i] - c * res[i] ** -p
        fit = a_inf + c * res_extra ** -p
        return np.where(good & np.isfinite(fit), m_extra - fit, np.nan)


class EhBound:
    name = "E_h"
    method = "student"

    def __init__(self, levels, resolutions, ip, triples=None, p_min=0.5, p_max=2.1, p_cap=2.0, safety=1.25,
                 fallback_factor=3.0, B=2000, rng=None, seed=None, share_draws=False, gates=(), extra=None, gate_level=0.90,
                 fallback_levels=None):
        rng = get_rng(rng, seed)
        x = np.asarray(levels, dtype=float)
        res = np.asarray(resolutions, dtype=float)
        K = x.shape[-1]
        if res.shape != (K,) or K < 2 or not np.all(np.diff(res) > 0):
            raise ValueError("resolutions must be K increasing values, one per level (K >= 2)")
        if not 0 <= ip < K:
            raise ValueError("ip out of range")
        if not (0.0 < p_min <= p_max) or p_cap <= 1.0:
            raise ValueError("need 0 < p_min <= p_max and p_cap > 1")
        ratios = res[1:] / res[:-1]
        if triples is None:
            triples = [(i, i + 1, i + 2) for i in range(K - 2) if np.isclose(ratios[i], ratios[i + 1], rtol=1e-9)]
        self.triples = [tuple(int(v) for v in t) for t in triples]
        self.triple_r = []
        for i, j, k in self.triples:
            if not (0 <= i < j < k < K) or not np.isclose(res[j] / res[i], res[k] / res[j], rtol=1e-9):
                raise ValueError("triple %r does not have a constant ratio" % ((i, j, k),))
            self.triple_r.append(float(res[j] / res[i]))
        fl = list(range(K)) if fallback_levels is None else sorted(int(i) for i in fallback_levels)
        if len(fl) < 2 or not set(fl) <= set(range(K)):
            raise ValueError("fallback_levels must be at least two valid level indices")
        self.fallback_levels = fl
        self.finer = ip < K - 1
        self.r = float(res[ip + 1] / res[ip]) if self.finer else float(res[ip] / res[ip - 1])
        self.r_min = float((res[fl][1:] / res[fl][:-1]).min())
        self.K, self.ip, self.p_min, self.p_max, self.p_cap = K, ip, p_min, p_max, p_cap
        self.safety, self.fallback_factor, self.B = safety, fallback_factor, B
        self.resolutions = res
        self.gates = [dict(g) for g in gates]
        self.gate_level = float(gate_level)
        need_extra = any(g["type"] == "extra_level" for g in self.gates)
        if need_extra:
            if extra is None:
                raise ValueError("an extra_level gate needs the 'extra' per-seed values")
            ex = np.asarray(extra, dtype=float)
            if ex.shape != x.shape[:-1]:
                raise ValueError("extra must have the shape of the ladder without the level axis")
            x_all = np.concatenate([x, ex[..., None]], axis=-1)
        else:
            ex, x_all = None, x
        m_hat_all = x_all.mean(axis=-2)
        m_hat = m_hat_all[..., :K]
        self.means = m_hat
        boot = np.moveaxis(bootstrap_means((np.moveaxis(x_all, -1, -2),), B, rng, share_draws, inner_ndim=1)[0], -2, -1)  # (..., B, K[+1])
        m_star = boot[..., :K]
        self.order_hat = self._point_order(m_hat)
        self._p_sorted = [np.sort(np.where(np.isnan(p), -np.inf, p), axis=-1) for p in np.moveaxis(self._triple_orders(m_star), -1, 0)]
        self.gate_ok, self.gate_info = self._evaluate_gates(m_hat_all, boot)
        iu, ju = np.triu_indices(K, 1)
        keep = np.isin(iu, fl) & np.isin(ju, fl)
        u = np.moveaxis(x[..., ju[keep]] - x[..., iu[keep]], -1, -2)  # (..., P, n): per-seed difference of every fallback pair i < j
        self._P = int(keep.sum())
        self._pd, self._pse, self._pnu = welch_groups([u])

        def full_pair(a, b):  # per-seed difference of any two levels (also outside the fallback levels) for the order branch
            return x[..., max(a, b)] - x[..., min(a, b)]

        j = ip + 1 if self.finer else ip - 1
        self._adj = welch_groups([full_pair(ip, j)[..., None, :]])
        self._second = welch_groups([full_pair(ip - 2, ip - 1)[..., None, :]]) if (not self.finer and ip >= 2) else None
        self.delta_hat = np.abs(m_hat[..., j] - m_hat[..., ip])

    # -- orders --------------------------------------------------------------------------------------------------
    def _triple_orders(self, m):
        """(..., T) observed order of every triple from level means ``m`` (..., K); NaN where there is no real order."""
        cols = []
        for (i, j, k), rt in zip(self.triples, self.triple_r):
            d0, d1 = m[..., j] - m[..., i], m[..., k] - m[..., j]
            same = ((d0 > 0) & (d1 > 0)) | ((d0 < 0) & (d1 < 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                p = (np.log(np.abs(d0)) - np.log(np.abs(d1))) / np.log(rt)
            cols.append(np.where(same & np.isfinite(p), p, np.nan))
        if not cols:
            return np.full(m.shape[:-1] + (0,), np.nan)
        return np.stack(cols, axis=-1)

    def _point_order(self, m):
        P = self._triple_orders(m)
        if P.shape[-1] == 0:
            return np.full(m.shape[:-1], np.nan)
        return np.where(np.isnan(P).any(axis=-1), np.nan, P.min(axis=-1))

    # -- gates ---------------------------------------------------------------------------------------------------
    def _gate_diff(self, g, m):
        """(...,) observed gate difference from the level means ``m`` (..., K[+1]); NaN where it does not exist."""
        K = self.K
        if g["type"] == "extra_level":
            return _fit_extra(m[..., :K], tuple(g["triple"]), self.resolutions, float(g["resolution"]), m[..., K])
        trs = [tuple(t) for t in g["triples"]]
        orders = []
        for t in trs[:2]:
            ix = self.triples.index(t) if t in self.triples else None
            if ix is None:
                rt = self.resolutions[t[1]] / self.resolutions[t[0]]
                if not np.isclose(self.resolutions[t[2]] / self.resolutions[t[1]], rt, rtol=1e-9):
                    raise ValueError("gate triple %r has no constant ratio" % (t,))
                d0, d1 = m[..., t[1]] - m[..., t[0]], m[..., t[2]] - m[..., t[1]]
                same = ((d0 > 0) & (d1 > 0)) | ((d0 < 0) & (d1 < 0))
                with np.errstate(divide="ignore", invalid="ignore"):
                    p = (np.log(np.abs(d0)) - np.log(np.abs(d1))) / np.log(rt)
                orders.append(np.where(same & np.isfinite(p), p, np.nan))
            else:
                orders.append(self._triple_orders(m)[..., ix])
        return orders[0] - orders[1]

    def _evaluate_gates(self, m_hat_all, boot):
        """-> (ok (...,) bool, info list).  All gates must pass."""
        shape = m_hat_all.shape[:-1]
        ok = np.ones(shape, dtype=bool)
        info = []
        a = 0.5 * (1.0 - self.gate_level)
        for g in self.gates:
            d_hat = self._gate_diff(g, m_hat_all)
            d_star = self._gate_diff(g, boot)  # (..., B)
            with np.errstate(invalid="ignore"):
                below = (d_star <= 0.0).mean(axis=-1)
                above = (d_star >= 0.0).mean(axis=-1)
            passed = np.isfinite(d_hat) & (below >= a) & (above >= a)
            ok &= passed
            fin = np.isfinite(d_star)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices: no interval
                lo, hi = np.nanquantile(np.where(fin, d_star, np.nan), [a, 1.0 - a], axis=-1)
            info.append({"gate": dict(g), "diff": d_hat, "share_le0": below, "share_ge0": above, "passed": passed,
                         "share_resamples_without_value": 1.0 - fin.mean(axis=-1), "interval": [lo, hi]})
        return ok, info

    def correction(self, means, p):
        """Signed Richardson correction ``alpha* - alpha(h_p)`` from level means (..., K) and the order ``p`` (see module docstring):
        ``p`` outside ``[p_min, p_max]`` or not finite is replaced by 1."""
        m = np.asarray(means, dtype=float)
        p = float(p)
        p = p if (np.isfinite(p) and self.p_min <= p <= self.p_max) else 1.0
        rp = self.r ** p
        if self.finer:
            return (m[..., self.ip + 1] - m[..., self.ip]) * rp / (rp - 1.0)
        return (m[..., self.ip] - m[..., self.ip - 1]) / (rp - 1.0)

    # -- bound ---------------------------------------------------------------------------------------------------
    def _parts(self, tails):
        tau, scalar = as_tails(tails)
        half = 0.5 * tau
        if self._p_sorted:
            lcbs = [lcb_sorted(s, half) for s in self._p_sorted]
            p_lo, p_hi = np.minimum.reduce(lcbs), np.maximum.reduce(lcbs)
        else:
            p_lo = p_hi = np.full(self._pd.shape[:-1] + tau.shape, -np.inf)
        in_range = np.isfinite(p_lo) & (p_lo >= self.p_min) & (p_hi <= self.p_max)
        p_use = np.where(in_range, np.minimum(p_lo, self.p_cap), np.nan)
        pu = np.where(in_range, p_use, self.p_min)
        rp = self.r ** pu
        dU1 = abs_upper_bound(self._adj[0][..., 0], self._adj[1][..., 0], half, self._adj[2][..., 0])
        rU = abs_upper_bound(self._pd, self._pse, tau / self._P, self._pnu).max(axis=-2)
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.finer:
                order_branch = self.safety * dU1 * rp / (rp - 1.0)
            else:
                top = dU1
                if self._second is not None:
                    dU2 = abs_upper_bound(self._second[0][..., 0], self._second[1][..., 0], half, self._second[2][..., 0])
                    top = np.maximum(dU1, dU2 / rp)
                order_branch = self.safety * top / (rp - 1.0)
            fallback = self.fallback_factor * rU / (self.r_min - 1.0)
        gated = np.where(self.gate_ok[..., None], order_branch, np.maximum(order_branch, fallback))
        return scalar, np.where(in_range, gated, fallback), p_lo, in_range, order_branch, fallback, p_use

    def bound(self, tails):
        scalar, bound = self._parts(tails)[:2]
        return bound[..., 0] if scalar else bound

    def details(self, tails):
        scalar, bound, p_L, in_range, order_branch, fallback, p_use = self._parts(tails)
        sq = (lambda v: v[..., 0]) if scalar else (lambda v: v)
        return {"method": self.method, "order_hat": self.order_hat, "triples": self.triples, "delta_hat": self.delta_hat,
                "p_L": sq(p_L), "p_use": sq(p_use), "order_branch_used": sq(in_range), "bound": sq(bound),
                "order_branch_value": sq(order_branch), "fallback_value": sq(fallback), "gate_ok": self.gate_ok,
                "gates": self.gate_info}
