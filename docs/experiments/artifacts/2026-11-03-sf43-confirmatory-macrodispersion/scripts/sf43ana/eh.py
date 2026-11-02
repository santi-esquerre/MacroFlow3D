"""Discretization bound ``E_h`` of the SF-43 pre-registration (H6): ``mcstat.LadderBound`` extended to several constant-ratio
triples, an order cap and a second difference at the finest level.  ``mcstat`` is not modified; its helpers are reused.

Definitions (``m_i`` the means of the per-seed ladder values on COMMON seeds, ordered coarse -> fine; ``r`` the ratio of the
production level to its neighbour; each studentized upper bound ``U`` is ``mcstat.intervals.abs_upper_bound`` with the Student
option on the per-seed difference of the two levels; this is the only method SF-42's coverage simulation qualified for the
contrasts):

* order of a triple ``(i, j, k)`` of constant ratio ``rt``: ``p = log(|m_j - m_i| / |m_k - m_j|) / log(rt)`` when both differences
  are nonzero with one sign, else no real order (non-monotone / oscillating -> outside the range);
* ``p_L(tau)`` of a triple: lower ``tau/2`` quantile of its bootstrap orders (resamples without a real order count as ``-inf``);
  ``p_L = min over triples`` (no triple -> ``-inf``);
* in range iff ``p_min <= p_L <= p_max`` (default ``[0.5, 2.1]``); ``p_use = min(p_L, p_cap)`` (default cap 2);
* finer level exists (``ip < K - 1``):  ``E_h = safety * U(m_{ip+1} - m_ip) * r^p_use / (r^p_use - 1)``;
* production finest:  ``E_h = safety * max(U(m_ip - m_{ip-1}), r^-p_use U(m_{ip-1} - m_{ip-2})) / (r^p_use - 1)`` (the second
  term only when ``ip >= 2``); under the power-law error model both terms estimate ``|e(h_p)|`` and either one valid suffices
  for the bound to hold, so the maximum is conservative; each ``U`` is taken at tail ``t/2`` and ``p_L`` at ``t/2``;
* otherwise (``p_L`` outside the range, fewer than 3 levels, no constant-ratio triple, non-monotone):
  ``E_h = fallback_factor * max over pairs of U_ij(t / P) / (r_min - 1)``, ``P = K (K - 1) / 2`` (exactly as ``LadderBound``).

Because ``p_use <= p_L`` and the finest branch only adds a maximum, the bound is at least ``LadderBound``'s (same range, same
bootstrap draws, ``method='student'``) in every branch; a unit test checks it on random ladders.

The signed Richardson correction used for the class ``numerical`` is ``correction(means)``: finer branch
``(m_{ip+1} - m_ip) r^p/(r^p - 1)``, finest ``(m_ip - m_{ip-1})/(r^p - 1)``, with ``p`` the POINT order (min over triples,
capped at ``p_cap``) if it lies in the range, else 1.
"""
import numpy as np

from mcstat._util import as_tails, bootstrap_means, get_rng, lcb_sorted
from mcstat.intervals import abs_upper_bound, welch_groups


class EhBound:
    name = "E_h"
    method = "student"

    def __init__(self, levels, resolutions, ip, triples=None, p_min=0.5, p_max=2.1, p_cap=2.0, safety=1.25,
                 fallback_factor=3.0, B=2000, rng=None, seed=None, share_draws=False):
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
        self.finer = ip < K - 1
        self.r = float(res[ip + 1] / res[ip]) if self.finer else float(res[ip] / res[ip - 1])
        self.r_min = float(ratios.min())
        self.K, self.ip, self.p_min, self.p_max, self.p_cap = K, ip, p_min, p_max, p_cap
        self.safety, self.fallback_factor, self.B = safety, fallback_factor, B
        self.resolutions = res
        m_hat = x.mean(axis=-2)
        self.means = m_hat
        m_star = np.moveaxis(bootstrap_means((np.moveaxis(x, -1, -2),), B, rng, share_draws, inner_ndim=1)[0], -2, -1)  # (..., B, K)
        self.order_hat = self._point_order(m_hat)
        self._p_sorted = [np.sort(np.where(np.isnan(p), -np.inf, p), axis=-1) for p in np.moveaxis(self._triple_orders(m_star), -1, 0)]
        iu, ju = np.triu_indices(K, 1)
        u = np.moveaxis(x[..., ju] - x[..., iu], -1, -2)  # (..., P, n): per-seed difference m_j - m_i of every pair i < j
        self._P = iu.size
        self._pd, self._pse, self._pnu = welch_groups([u])

        def pair(a, b):
            return int(np.flatnonzero((iu == min(a, b)) & (ju == max(a, b)))[0])

        j = ip + 1 if self.finer else ip - 1
        self._adj = pair(ip, j)
        self._second = pair(ip - 2, ip - 1) if (not self.finer and ip >= 2) else None
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

    def correction(self, means):
        """Signed Richardson correction ``alpha* - alpha(h_p)`` from level means (..., K) (see module docstring)."""
        m = np.asarray(means, dtype=float)
        ph = self._point_order(m)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(ph) & (ph >= self.p_min) & (ph <= self.p_max)
        p = np.where(ok, np.minimum(ph, self.p_cap), 1.0)
        rp = self.r ** p
        if self.finer:
            return (m[..., self.ip + 1] - m[..., self.ip]) * rp / (rp - 1.0)
        return (m[..., self.ip] - m[..., self.ip - 1]) / (rp - 1.0)

    # -- bound ---------------------------------------------------------------------------------------------------
    def _parts(self, tails):
        tau, scalar = as_tails(tails)
        half = 0.5 * tau
        if self._p_sorted:
            p_L = np.minimum.reduce([lcb_sorted(s, half) for s in self._p_sorted])
        else:
            p_L = np.full(self._pd.shape[:-1] + tau.shape, -np.inf)
        a = self._adj
        dU1 = abs_upper_bound(self._pd[..., a], self._pse[..., a], half, self._pnu[..., a])
        rU = abs_upper_bound(self._pd, self._pse, tau / self._P, self._pnu).max(axis=-2)
        in_range = np.isfinite(p_L) & (p_L >= self.p_min) & (p_L <= self.p_max)
        p_use = np.where(in_range, np.minimum(p_L, self.p_cap), self.p_min)
        rp = self.r ** p_use
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.finer:
                order_branch = self.safety * dU1 * rp / (rp - 1.0)
            else:
                top = dU1
                if self._second is not None:
                    b = self._second
                    dU2 = abs_upper_bound(self._pd[..., b], self._pse[..., b], half, self._pnu[..., b])
                    top = np.maximum(dU1, dU2 / rp)
                order_branch = self.safety * top / (rp - 1.0)
            fallback = self.fallback_factor * rU / (self.r_min - 1.0)
        return scalar, np.where(in_range, order_branch, fallback), p_L, in_range, order_branch, fallback

    def bound(self, tails):
        scalar, bound = self._parts(tails)[:2]
        return bound[..., 0] if scalar else bound

    def details(self, tails):
        scalar, bound, p_L, in_range, order_branch, fallback = self._parts(tails)
        sq = (lambda v: v[..., 0]) if scalar else (lambda v: v)
        return {"method": self.method, "order_hat": self.order_hat, "triples": self.triples, "delta_hat": self.delta_hat,
                "p_L": sq(p_L), "order_branch_used": sq(in_range), "bound": sq(bound),
                "order_branch_value": sq(order_branch), "fallback_value": sq(fallback)}
