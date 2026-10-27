"""SF-39 Lagrangian reducer (contract C-REDUCE): preregistration section 3 from the `lagrangian_ensemble` files.

Every weighted mean/variance is the BIASED weighted form with the weights renormalized over the selected particle set:
`m = sum(w x)/sum(w)`, `V = sum(w (x-m)^2)/sum(w)`.  A record is present iff finite (records are prefix-complete).  An empty set
or a zero total weight raises `ReduceError`; nothing is clipped.  alpha is in units of lambda (= ell, box units).

Window estimator (per direction `e`, unit vector; variances are linear-interpolated, never positions):
  `W = [t_lo, t_hi] = [0.75, 1] tau_b* ell / U`; `j_hi` = first snapshot with `t_j >= t_hi` (`WindowError` if none, or if
  `t_lo < t_1`); set `S_r` = particles with a (finite) record at `j_hi`, used at both endpoints;
  `V_e(t)` = interpolation of `V_e(t_j)` between the bracketing snapshots; `alpha_e = [V_e(t_hi)-V_e(t_lo)]/(2 ell U (t_hi-t_lo))`.
  `V_e = e^T C e` with `C` the weighted covariance of the (x1,x2,x3) records (identical to the variance of the projections).
Plane estimator: `T_r` = particles with records at X_a and X_b; `sigma_T^2(X) = (var x2 + var x3)/2` over `T_r`;
  `alpha_TP = [sigma_T^2(X_b) - sigma_T^2(X_a)] / (2 ell (X_b - X_a))`.  `X_a = x0 + L/2`, `X_b = L1 - 5 ell`, `X_q = x0 + L/4`,
  `L = L1 - 10 ell`; planes are found by nearest value within `1e-12 max(1,|x|)` (`MissingData` if absent).

Methods (weight/set modes; `method` column of the long output):
  flux     consumer flux weights `weights.npy` over S_r (resp. T_r)
  unif     uniform weights over the same set (twin rule)
  expI     flux * exp(-I) with I at each evaluated snapshot (`times_I`) / plane (`planes_I`), renormalized per evaluation
  full     (window only) every traced particle: outlet exit frozen at x1=L1, inlet exit at x1=0, last recorded (x2,x3)
           (seed (x2,x3) if no record exists); particles with another non-ok status and no record are unplaced (counted)
  pollock  Pollock flux weights `pollock_weights.npy` over the Pollock records
  ixc/ixp  intersection set (records in the consumer AND Pollock arrays), uniform weights, consumer / Pollock positions

Columns of one method dict (`win.*` window estimator, `tp.*` plane estimator; NaN where not applicable):
  win.n            |S| (set size)
  win.excl         excluded fraction of the method's reference weight (flux, expI, full, pollock: 1 - sum w[S]/sum w; unif, ixc,
                   ixp: 1 - |S|/n_traced, n_traced = consumer particles with weight > 0)
  win.alpha_e1/_e2/_e3   fixed axes; alpha_LAM = alpha_e1
  win.alpha_Tfixed (alpha_e2+alpha_e3)/2
  win.alpha_eL, win.alpha_eT1, win.alpha_eT2   aligned frame (E4): e_L = q/|q| (q = flow_stats interior mean_u,v,w), e_T1/e_T2 =
                   Gram-Schmidt of e2, e3 against e_L; win.alpha_TAM = (alpha_eT1+alpha_eT2)/2 (order independent)
  win.dTfixed_minus_TAM   alpha_Tfixed - alpha_TAM
  win.alpha_eD, win.alpha_TD, win.drift_tilt   plume-drift frame: e_D = unit(mean(t_hi)-mean(t_lo)) (interpolated means over S)
  full only: win.n_exit_out, win.n_exit_in (frozen), win.n_seed_fallback (exit before the first snapshot), win.n_unplaced
  tp.n             |T|;  tp.excl  excluded fraction (as win.excl);  tp.sigT2_Xa, tp.sigT2_Xb;  tp.alpha_TP
  tp.n_noncross_Xa, tp.n_noncross_Xb   (flux/unif/expI) traced particles without a record at that plane
  flux only: tp.alpha_TP_q (X_q -> X_a over its own set), tp.delta_half = alpha_TP[X_q,X_a] - alpha_TP[X_a,X_b] (F7)

Columns of the per-realization common dict (`flatten` prefixes methods as `<method>.<column>`):
  r, seed, qbar, applied_scale, raw_variance, final_variance, ell, L1, x0, Xa, Xb, Xq, U, tau_b_star, t_lo, t_hi, j_hi, t_jhi,
  tb (t_b,r; NaN if X_b absent), tilt (atan2(|e_L x e1|, e_L.e1)), x_nonpositive_fraction, speed_p0, speed_p0_1, speed_p1,
  speed_p50, plane_hist_k / time_hist_k (k=0..8, status histograms of the pass arrays), exit_inlet_jhi, exit_outlet_jhi
  (exits_time.csv row j_hi), ev_p50/p90/p99/p99_9/p100 (percentiles of |1-exp(-I)| at X_b over the records at X_b),
  S_r = sigma_T^2(X_b)/(ell (X_b-X_a)) (flux, T_r), split.<est>.A / split.<est>.B (est in alpha_TAM, alpha_TP, alpha_LAM; flux;
  halves by (m2+m3) parity of the lattice index p = m2 M + m3), skipped (';'-joined reasons for tolerated missing data).

Smoke cross-check: `crosscheck_pass/run` compare `sigma_series_rows` and the per-snapshot moments with the runner's
moments_planes.csv / moments_time.csv (flux) and twin/moments_*.csv (uniform): `contributing`/`active` equal, mean/var within
|a-b| <= 1e-9 max(|a|,|b|) + 1e-15, x1 within 1e-12; `max_rel` = max |a-b|/max(|a|,|b|,1e-6).  Any FAIL -> smoke exit 1.

Long tables: `sigma_series_rows` (x1, n_contrib, n_noncross, wfrac, mean/var x2,x3, sigma_T2: each plane over its own contributors,
flux weights) and `rho_rows` (F8: pair, plane, n, sigma_T, rho = sqrt(sum w |dx_T|^2/sum w)/sigma_T, rho_c = centred per-component
variant sqrt(sum w[(d2-d2bar)^2+(d3-d3bar)^2]/(2 sum w))/sigma_T, S_r, rho_S = rho S_r; weights and sigma_T from the first
(h/8) pass over the particles with records in both passes).

CLI: `python3 -m sf39.reduce {tb,alpha,smoke,signtable} ...` (see `main`).
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

from . import io as sio
from .io import (MissingData, ReduceError, WindowError, Realization, Pass, parse_range,
                 STATUS_EXIT_INLET, STATUS_EXIT_OUTLET)

WINDOW_FRACTIONS = (0.75, 1.0)
WIN_MODES = ("flux", "unif", "expI", "full", "pollock", "ixc", "ixp")
TP_MODES = ("flux", "unif", "expI", "pollock", "ixc", "ixp")
SPLIT_EST = ("alpha_TAM", "alpha_TP", "alpha_LAM")
EV_PCT = (50, 90, 99, 99.9, 100)
EV_NAMES = ("ev_p50", "ev_p90", "ev_p99", "ev_p99_9", "ev_p100")
RHO_PAIRS = ((8, 16), (16, 32), (8, 32))
NAN = float("nan")


# ------------------------------------------------------------------------------------------------
# weighted statistics
# ------------------------------------------------------------------------------------------------
def _check_w(w):
    w = np.asarray(w, dtype=float)
    if w.size == 0 or not np.all(np.isfinite(w)) or not (w.sum() > 0.0):
        raise ReduceError("empty particle set or non-positive/non-finite total weight")
    return w


def wvar(x, w):
    """Biased weighted variance of a vector, weights renormalized."""
    w = _check_w(w)
    x = np.asarray(x, dtype=float)
    m = np.sum(w * x) / w.sum()
    return float(np.sum(w * (x - m) ** 2) / w.sum())


def wmoments3(X, w):
    """Weighted mean (3,) and biased covariance (3,3) of the rows of X (n,3)."""
    w = _check_w(w)
    X = np.asarray(X, dtype=float)
    if not np.all(np.isfinite(X)):
        raise ReduceError("non-finite record inside the evaluated set (prefix-completeness violated)")
    sw = w.sum()
    m = (w[:, None] * X).sum(axis=0) / sw
    D = X - m
    return m, (D * w[:, None]).T @ D / sw


def _fin(a):
    """Rows whose trailing-axis entries are all finite."""
    return np.isfinite(np.asarray(a)).all(axis=-1)


# ------------------------------------------------------------------------------------------------
# window geometry and frames
# ------------------------------------------------------------------------------------------------
def window_edges(times, tau_b_star, ell, U):
    """(t_lo, t_hi, j_hi); WindowError if no snapshot >= t_hi or t_lo < t_1."""
    times = np.asarray(times, dtype=float)
    t_lo = WINDOW_FRACTIONS[0] * tau_b_star * ell / U
    t_hi = WINDOW_FRACTIONS[1] * tau_b_star * ell / U
    j_hi = int(np.searchsorted(times, t_hi, side="left"))
    if j_hi >= len(times):
        raise WindowError("no snapshot >= t_hi=%.17g (last snapshot %.17g)" % (t_hi, times[-1]))
    if t_lo < times[0]:
        raise WindowError("t_lo=%.17g < t_1=%.17g" % (t_lo, times[0]))
    return t_lo, t_hi, j_hi


def _bracket(times, t):
    """(ja, jb, f): t = times[ja] + f (times[jb]-times[ja]); ja==jb, f=0 on an exact snapshot."""
    jb = int(np.searchsorted(times, t, side="left"))
    if jb >= len(times) or (jb == 0 and times[0] != t):
        raise WindowError("time %.17g outside the recorded snapshots" % t)
    if times[jb] == t:
        return jb, jb, 0.0
    return jb - 1, jb, (t - times[jb - 1]) / (times[jb] - times[jb - 1])


def window_moments(times, t_lo, t_hi, pos, wt):
    """Interpolated (mean, cov) at t_lo and t_hi; pos(j) -> (n,3), wt(j) -> (n,) of the fixed set."""
    cache = {}

    def snap(j):
        if j not in cache:
            cache[j] = wmoments3(pos(j), wt(j))
        return cache[j]

    def at(t):
        ja, jb, f = _bracket(times, t)
        ma, Ca = snap(ja)
        if f == 0.0:
            return ma, Ca
        mb, Cb = snap(jb)
        return ma + f * (mb - ma), Ca + f * (Cb - Ca)

    m_lo, C_lo = at(t_lo)
    m_hi, C_hi = at(t_hi)
    return m_lo, C_lo, m_hi, C_hi


def _unit(v, what):
    n = float(np.linalg.norm(v))
    if not (n > 0.0) or not math.isfinite(n):
        raise ReduceError("degenerate %s (zero or non-finite norm)" % what)
    return np.asarray(v, dtype=float) / n


def frame_from_longitudinal(eL):
    """(e_T1, e_T2): Gram-Schmidt of e2, e3 against the unit vector eL."""
    e2, e3 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    t1 = _unit(e2 - (e2 @ eL) * eL, "transverse frame (e_L parallel to e2)")
    t2 = _unit(e3 - (e3 @ eL) * eL - (e3 @ t1) * t1, "transverse frame (e_L in span(e2,e3) degenerate)")
    return t1, t2


def aligned_frame(q):
    eL = _unit(np.asarray(q, dtype=float), "interior mean velocity")
    return (eL,) + frame_from_longitudinal(eL)


def tilt_angle(eL):
    e1 = np.array([1.0, 0.0, 0.0])
    return math.atan2(float(np.linalg.norm(np.cross(eL, e1))), float(eL @ e1))


def window_alphas(m_lo, C_lo, m_hi, C_hi, q, ell, U, dt):
    """All window alphas of one evaluation (fixed, aligned, plume-drift frames)."""
    dC = C_hi - C_lo
    den = 2.0 * ell * U * dt

    def a(e):
        return float(e @ dC @ e) / den

    out = {}
    E = np.eye(3)
    out["alpha_e1"], out["alpha_e2"], out["alpha_e3"] = a(E[0]), a(E[1]), a(E[2])
    out["alpha_Tfixed"] = (out["alpha_e2"] + out["alpha_e3"]) / 2.0
    eL, t1, t2 = aligned_frame(q)
    out["alpha_eL"], out["alpha_eT1"], out["alpha_eT2"] = a(eL), a(t1), a(t2)
    out["alpha_TAM"] = (out["alpha_eT1"] + out["alpha_eT2"]) / 2.0
    out["dTfixed_minus_TAM"] = out["alpha_Tfixed"] - out["alpha_TAM"]
    eD = _unit(m_hi - m_lo, "plume-drift direction")
    d1, d2 = frame_from_longitudinal(eD)
    out["alpha_eD"] = a(eD)
    out["alpha_TD"] = (a(d1) + a(d2)) / 2.0
    out["drift_tilt"] = tilt_angle(eD)
    return out


# ------------------------------------------------------------------------------------------------
# window populations
# ------------------------------------------------------------------------------------------------
def _all(member, n):
    return np.ones(n, dtype=bool) if member is None else np.asarray(member, dtype=bool)


def _excl_flux(w_all, idx):
    return 1.0 - float(np.asarray(w_all)[idx].sum()) / float(np.asarray(w_all).sum())


def _full_positions(P, idx, L1):
    """pos(j) with exits frozen (outlet x1=L1, inlet x1=0, last recorded (x2,x3))."""
    tx, st, seeds = P.times_x, np.asarray(P.status_time), np.asarray(P.seeds)

    def pos(j):
        X = np.array(tx[idx, j, :], dtype=float)
        bad = ~_fin(X)
        for k in np.flatnonzero(bad):
            i = idx[k]
            s = int(st[i])
            if s not in (STATUS_EXIT_INLET, STATUS_EXIT_OUTLET):
                raise ReduceError("particle %d: no record at snapshot %d and status %d is not an exit" % (i, j, s))
            fr = _fin(np.asarray(tx[i, :, :]))
            n = int(fr.sum())
            if not fr[:n].all():
                raise ReduceError("particle %d: records are not prefix-complete" % i)
            x23 = np.asarray(tx[i, n - 1, 1:3]) if n > 0 else seeds[i, 1:3]
            X[k] = (L1 if s == STATUS_EXIT_OUTLET else 0.0, x23[0], x23[1])
        return X

    return pos


def window_population(real, mode, j_hi, member=None):
    """dict(idx, pos, wt, excl, extra) for a window method (see module docstring)."""
    P = real.pass_
    w = np.asarray(P.weights, dtype=float)
    Np = len(w)
    mem = _all(member, Np)
    tx = P.times_x
    traced = w > 0.0
    n_traced = int((traced & mem).sum())
    fin = _fin(np.asarray(tx[:, j_hi, :]))
    extra = {}
    if mode in ("flux", "unif", "expI"):
        idx = np.flatnonzero(fin & mem)
        pos = lambda j: np.asarray(tx[idx, j, :], dtype=float)
        if mode == "unif":
            wt, excl = (lambda j: np.ones(len(idx))), 1.0 - len(idx) / n_traced
        elif mode == "flux":
            wt, excl = (lambda j: w[idx]), _excl_flux(np.where(mem, w, 0.0), idx)
        else:
            wt, excl = (lambda j: w[idx] * np.exp(-np.asarray(P.times_I[idx, j], dtype=float))), _excl_flux(np.where(mem, w, 0.0), idx)
    elif mode == "full":
        st = np.asarray(P.status_time)
        exit_ = (st == STATUS_EXIT_INLET) | (st == STATUS_EXIT_OUTLET)
        placed = (fin | exit_) & traced & mem
        idx = np.flatnonzero(placed)
        pos = _full_positions(P, idx, real.L1)
        wt, excl = (lambda j: w[idx]), _excl_flux(np.where(mem, w, 0.0), idx)
        frozen = placed & ~fin
        n_seed = 0
        for i in np.flatnonzero(frozen):
            if not np.isfinite(np.asarray(tx[i, 0, :])).all():
                n_seed += 1
        extra = {"n_exit_out": int((frozen & (st == STATUS_EXIT_OUTLET)).sum()),
                 "n_exit_in": int((frozen & (st == STATUS_EXIT_INLET)).sum()),
                 "n_seed_fallback": n_seed,
                 "n_unplaced": int((traced & mem & ~placed).sum())}
    else:
        Q = real.pollock
        if Q is None:
            raise MissingData("no Pollock bridge in %s" % real.rdir)
        qfin = _fin(np.asarray(Q.times_x[:, j_hi, :]))
        if mode == "pollock":
            qw = np.asarray(Q.weights, dtype=float)
            idx = np.flatnonzero(qfin & mem)
            pos = lambda j: np.asarray(Q.times_x[idx, j, :], dtype=float)
            wt, excl = (lambda j: qw[idx]), _excl_flux(np.where(mem, qw, 0.0), idx)
        else:
            idx = np.flatnonzero(fin & qfin & mem)
            src = tx if mode == "ixc" else Q.times_x
            pos = lambda j: np.asarray(src[idx, j, :], dtype=float)
            wt, excl = (lambda j: np.ones(len(idx))), 1.0 - len(idx) / n_traced
    if len(idx) == 0:
        raise ReduceError("empty particle set for method %s" % mode)
    return {"idx": idx, "pos": pos, "wt": wt, "excl": excl, "extra": extra}


def window_method(real, mode, tau_b_star, U, member=None):
    """`win.*` dict of one method."""
    times = real.pass_.times
    t_lo, t_hi, j_hi = window_edges(times, tau_b_star, real.ell, U)
    pop = window_population(real, mode, j_hi, member)
    m_lo, C_lo, m_hi, C_hi = window_moments(times, t_lo, t_hi, pop["pos"], pop["wt"])
    q = (real.flow["interior"]["mean_u"], real.flow["interior"]["mean_v"], real.flow["interior"]["mean_w"])
    out = {"win.n": len(pop["idx"]), "win.excl": pop["excl"]}
    out.update({"win." + k: v for k, v in window_alphas(m_lo, C_lo, m_hi, C_hi, q, real.ell, U, t_hi - t_lo).items()})
    out.update({"win." + k: v for k, v in pop["extra"].items()})
    return out


# ------------------------------------------------------------------------------------------------
# plane estimator
# ------------------------------------------------------------------------------------------------
def sigma_T2(x23, w):
    """(var x2 + var x3)/2, biased weighted, renormalized."""
    x23 = np.asarray(x23, dtype=float)
    return (wvar(x23[:, 0], w) + wvar(x23[:, 1], w)) / 2.0


def tp_value(real, mode, ia, ib, xa, xb, member=None):
    """`tp.*` dict (without the flux-only extras) of one method for the two planes ia < ib."""
    P = real.pass_
    w = np.asarray(P.weights, dtype=float)
    mem = _all(member, len(w))
    traced = w > 0.0
    n_traced = int((traced & mem).sum())
    fa, fb = _fin(np.asarray(P.planes_x23[:, ia, :])), _fin(np.asarray(P.planes_x23[:, ib, :]))
    out = {}
    if mode in ("flux", "unif", "expI"):
        idx = np.flatnonzero(fa & fb & mem)
        X = P.planes_x23
        xa_, xb_ = np.asarray(X[idx, ia, :], dtype=float), np.asarray(X[idx, ib, :], dtype=float)
        if mode == "flux":
            wa = wb = w[idx]
        elif mode == "unif":
            wa = wb = np.ones(len(idx))
        else:
            wa = w[idx] * np.exp(-np.asarray(P.planes_I[idx, ia], dtype=float))
            wb = w[idx] * np.exp(-np.asarray(P.planes_I[idx, ib], dtype=float))
        excl = (1.0 - len(idx) / n_traced) if mode == "unif" else _excl_flux(np.where(mem, w, 0.0), idx)
        out["tp.n_noncross_Xa"] = int((traced & mem & ~fa).sum())
        out["tp.n_noncross_Xb"] = int((traced & mem & ~fb).sum())
    else:
        Q = real.pollock
        if Q is None:
            raise MissingData("no Pollock bridge in %s" % real.rdir)
        qa, qb = _fin(np.asarray(Q.planes_x23[:, ia, :])), _fin(np.asarray(Q.planes_x23[:, ib, :]))
        if mode == "pollock":
            qw = np.asarray(Q.weights, dtype=float)
            idx = np.flatnonzero(qa & qb & mem)
            xa_, xb_ = np.asarray(Q.planes_x23[idx, ia, :], dtype=float), np.asarray(Q.planes_x23[idx, ib, :], dtype=float)
            wa = wb = qw[idx]
            excl = _excl_flux(np.where(mem, qw, 0.0), idx)
        else:
            idx = np.flatnonzero(fa & fb & qa & qb & mem)
            src = P.planes_x23 if mode == "ixc" else Q.planes_x23
            xa_, xb_ = np.asarray(src[idx, ia, :], dtype=float), np.asarray(src[idx, ib, :], dtype=float)
            wa = wb = np.ones(len(idx))
            excl = 1.0 - len(idx) / n_traced
    if len(idx) == 0:
        raise ReduceError("empty plane set for method %s" % mode)
    sa, sb = sigma_T2(xa_, wa), sigma_T2(xb_, wb)
    out.update({"tp.n": len(idx), "tp.excl": excl, "tp.sigT2_Xa": sa, "tp.sigT2_Xb": sb,
                "tp.alpha_TP": (sb - sa) / (2.0 * real.ell * (xb - xa))})
    return out


def tp_method(real, mode, kp, member=None):
    """`tp.*` dict of one method; kp = key_planes() of the realization (Xa, Xb, Xq with indices)."""
    (xa, ia), (xb, ib) = kp["Xa"], kp["Xb"]
    out = tp_value(real, mode, ia, ib, xa, xb, member)
    if mode == "flux":
        xq, iq = kp["Xq"]
        if iq is not None:
            first = tp_value(real, "flux", iq, ia, xq, xa, member)["tp.alpha_TP"]
            out["tp.alpha_TP_q"] = first
            out["tp.delta_half"] = first - out["tp.alpha_TP"]
    return out


def breakthrough_time(P, ib):
    """t_b,r = min planes_tau[i, X_b] over status_plane == 0 and a record at X_b."""
    ok = (np.asarray(P.status_plane) == 0) & np.isfinite(np.asarray(P.planes_tau[:, ib]))
    if not ok.any():
        raise ReduceError("no particle with status_plane == 0 and a record at X_b")
    return float(np.min(np.asarray(P.planes_tau[:, ib])[ok]))


def ev_percentiles(P, ib):
    """Percentiles (50, 90, 99, 99.9, 100) of |1 - exp(-I)| at X_b over the records at X_b (unweighted)."""
    fb = _fin(np.asarray(P.planes_x23[:, ib, :]))
    I = np.asarray(P.planes_I[:, ib], dtype=float)[fb]
    if I.size == 0 or not np.all(np.isfinite(I)):
        raise ReduceError("no contributors or non-finite I at X_b")
    return dict(zip(EV_NAMES, (float(v) for v in np.percentile(np.abs(1.0 - np.exp(-I)), EV_PCT))))


def half_masks(P):
    """(A, B) boolean masks: (m2+m3) even / odd with lattice index p = m2 M + m3."""
    Np, M = len(P.weights), P.M
    if Np != M * M:
        raise ReduceError("Np=%d is not M^2 (M=%d): cannot form the lattice halves" % (Np, M))
    p = np.arange(Np)
    even = ((p // M + p % M) % 2) == 0
    return even, ~even


# ------------------------------------------------------------------------------------------------
# long tables
# ------------------------------------------------------------------------------------------------
def sigma_series_rows(P, uniform=False):
    """One row per plane: each plane over its own contributors, consumer flux weights (uniform=True: uniform weights)."""
    w = np.ones(len(P.weights)) if uniform else np.asarray(P.weights, dtype=float)
    traced = w > 0.0
    rows = []
    for p, x1 in enumerate(P.planes):
        X = np.asarray(P.planes_x23[:, p, :], dtype=float)
        f = _fin(X)
        n = int(f.sum())
        row = {"plane": p, "x1": float(x1), "n_contrib": n, "n_noncross": int((traced & ~f).sum()),
               "wfrac": float(w[f].sum() / w.sum()) if w.sum() > 0 else NAN,
               "mean_x2": NAN, "mean_x3": NAN, "var_x2": NAN, "var_x3": NAN, "sigma_T2": NAN}
        if n > 0 and w[f].sum() > 0:
            ww = w[f]
            row.update(mean_x2=float(np.sum(ww * X[f, 0]) / ww.sum()), mean_x3=float(np.sum(ww * X[f, 1]) / ww.sum()),
                       var_x2=wvar(X[f, 0], ww), var_x3=wvar(X[f, 1], ww))
            row["sigma_T2"] = (row["var_x2"] + row["var_x3"]) / 2.0
        rows.append(row)
    return rows


def rho_rows(rdir, divs=(8, 16, 32), pairs=RHO_PAIRS):
    """F8 rho(X) per plane for each pass pair of one realization (directory A); see the module docstring."""
    reals = {d: Realization(rdir, d) for d in divs}
    base = reals[divs[0]]
    P0 = base.pass_
    for d in divs:
        if reals[d].pass_.planes.shape != P0.planes.shape or np.any(reals[d].pass_.planes != P0.planes):
            raise ReduceError("planes differ between passes in %s" % rdir)
    w = np.asarray(P0.weights, dtype=float)
    kp = base.key_planes()
    xa, ia = kp["Xa"]
    xb, ib = kp["Xb"]
    fa, fb = _fin(np.asarray(P0.planes_x23[:, ia, :])), _fin(np.asarray(P0.planes_x23[:, ib, :]))
    T = fa & fb
    iT = np.flatnonzero(T)
    S_r = sigma_T2(np.asarray(P0.planes_x23[iT, ib, :], dtype=float), w[iT]) / (base.ell * (xb - xa))
    rows = []
    for (da, db) in pairs:
        A, B = np.asarray(reals[da].pass_.planes_x23), np.asarray(reals[db].pass_.planes_x23)
        for p, x1 in enumerate(P0.planes):
            f = _fin(np.asarray(A[:, p, :])) & _fin(np.asarray(B[:, p, :]))
            n = int(f.sum())
            row = {"pair": "%d-%d" % (da, db), "plane": p, "x1": float(x1), "n": n, "sigma_T": NAN, "rho": NAN,
                   "rho_c": NAN, "S_r": S_r, "rho_S": NAN}
            if n > 0 and w[f].sum() > 0:
                ww = w[f]
                # sigma_T from the first (h/8) pass over the same particles
                sT = math.sqrt(sigma_T2(np.asarray(P0.planes_x23[np.flatnonzero(f), p, :], dtype=float), ww))
                d = np.asarray(B[np.flatnonzero(f), p, :], dtype=float) - np.asarray(A[np.flatnonzero(f), p, :], dtype=float)
                if sT > 0.0:
                    d2, d3 = d[:, 0], d[:, 1]
                    sw = ww.sum()
                    c2, c3 = np.sum(ww * d2) / sw, np.sum(ww * d3) / sw
                    row["sigma_T"] = sT
                    row["rho"] = math.sqrt(np.sum(ww * (d2 ** 2 + d3 ** 2)) / sw) / sT
                    row["rho_c"] = math.sqrt(np.sum(ww * ((d2 - c2) ** 2 + (d3 - c3) ** 2)) / (2.0 * sw)) / sT
                    row["rho_S"] = row["rho"] * S_r
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------
# per-realization reducer
# ------------------------------------------------------------------------------------------------
def reduce_methods(real, tau_b_star, U, *, tolerate_missing=False, r=None):
    """(common dict, {method: dict}) for one realization/pass.  Raises unless `tolerate_missing` turns MissingData into
    the `skipped` column (a window/plane estimator that cannot run is then absent, never zero)."""
    P = real.pass_
    skipped = []
    ell, L1, x0 = real.ell, real.L1, real.x0
    kp = real.key_planes(tolerate_missing)
    (xa, ia), (xb, ib), (xq, iq) = kp["Xa"], kp["Xb"], kp["Xq"]
    for k, i in (("Xa", ia), ("Xb", ib), ("Xq", iq)):
        if i is None:
            skipped.append("plane_" + k)
    t_lo, t_hi, j_hi = window_edges(P.times, tau_b_star, ell, U)
    q = (real.flow["interior"]["mean_u"], real.flow["interior"]["mean_v"], real.flow["interior"]["mean_w"])
    fs = real.flow
    common = {"r": r if r is not None else -1, "seed": real.seed, "qbar": real.qbar,
              "applied_scale": real.field["applied_scale"], "raw_variance": real.field["raw_variance"],
              "final_variance": real.field["final_variance"], "ell": ell, "L1": L1, "x0": x0,
              "Xa": xa, "Xb": xb, "Xq": xq, "U": U, "tau_b_star": tau_b_star, "t_lo": t_lo, "t_hi": t_hi,
              "j_hi": j_hi, "t_jhi": float(P.times[j_hi]),
              "tb": breakthrough_time(P, ib) if ib is not None else NAN,
              "tilt": tilt_angle(aligned_frame(q)[0]),
              "x_nonpositive_fraction": fs["x_nonpositive_fraction"]}
    sp = fs["speed_over_qbar_percentiles"]
    common.update(speed_p0=sp["p0"], speed_p0_1=sp["p0.1"], speed_p1=sp["p1"], speed_p50=sp["p50"])
    for which in ("plane", "time"):
        for k, c in enumerate(P.status_hist(which)):
            common["%s_hist_%d" % (which, k)] = int(c)
    common["exit_inlet_jhi"], common["exit_outlet_jhi"] = P.exits_time_row(j_hi)
    methods = {}
    pollock_missing = real.pollock is None
    if pollock_missing:
        if not tolerate_missing:
            raise MissingData("no Pollock bridge in %s" % real.rdir)
        skipped.append("pollock")
    for mode in WIN_MODES:
        if pollock_missing and mode in ("pollock", "ixc", "ixp"):
            continue
        methods.setdefault(mode, {}).update(window_method(real, mode, tau_b_star, U))
    if ia is not None and ib is not None:
        for mode in TP_MODES:
            if pollock_missing and mode in ("pollock", "ixc", "ixp"):
                continue
            methods.setdefault(mode, {}).update(tp_method(real, mode, kp))
        fl = methods["flux"]
        common["S_r"] = fl["tp.sigT2_Xb"] / (ell * (xb - xa))
    else:
        common["S_r"] = NAN
    if ib is not None:
        common.update(ev_percentiles(P, ib))
    A, B = half_masks(P)
    for name, mem in (("A", A), ("B", B)):
        win = window_method(real, "flux", tau_b_star, U, mem)
        vals = {"alpha_TAM": win["win.alpha_TAM"], "alpha_LAM": win["win.alpha_e1"]}
        if ia is not None and ib is not None:
            vals["alpha_TP"] = tp_value(real, "flux", ia, ib, xa, xb, mem)["tp.alpha_TP"]
        for est in SPLIT_EST:
            common["split.%s.%s" % (est, name)] = vals.get(est, NAN)
    common["skipped"] = ";".join(skipped)
    return common, methods


def flatten(common, methods):
    out = dict(common)
    for m, d in methods.items():
        out.update({"%s.%s" % (m, k): v for k, v in d.items()})
    return out


def long_rows(common, methods, keep=("r", "seed", "qbar", "U", "tau_b_star", "t_lo", "t_hi", "j_hi", "tilt")):
    """One row per method (columns: `keep`, method, then the `win.*` / `tp.*` columns)."""
    return [dict({k: common[k] for k in keep}, method=m, **d) for m, d in methods.items()]


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.17g" % v
    return str(v)


def write_csv(path, rows, columns=None):
    """CSV of dict rows (union of keys in first-seen order unless `columns`); floats as %.17g; path None -> stdout."""
    if columns is None:
        columns = []
        for row in rows:
            columns += [k for k in row if k not in columns]
    f = open(path, "w", newline="") if path else sys.stdout
    try:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(columns)
        for row in rows:
            wr.writerow([_fmt(row.get(k)) for k in columns])
    finally:
        if path:
            f.close()


# ------------------------------------------------------------------------------------------------
# sign table (preregistration RR3 (iv); review section 5.4)
# ------------------------------------------------------------------------------------------------
SIGN_N_PLANES = 14
SIGN_EXPECTED = {(1, 6101): 1, (1, 6102): 1, (1, 6103): 14, (2, 6101): 3, (2, 6102): 0, (2, 6103): 5}
SIGN_GRIDS = (32, 64, 128)  # coarse g; lambda/h = g/2 (16, 32, 64); reference g256 (lambda/h 128)


def _sigma_run(evladder, eps, seed, g):
    d = os.path.join(evladder, "F_e%d_s%d_g%d" % (eps, seed, g), "realizations", "r_0000")
    P = Pass(sio.pass_root(d))
    rows = sigma_series_rows(P)
    return np.array([r["x1"] for r in rows]), np.array([r["sigma_T2"] for r in rows])


def signtable(evladder, out=None):
    """Counts of planes where sigma_T^2 (coarse) < sigma_T^2 (g256 reference); True iff equal to SIGN_EXPECTED."""
    out = out or sys.stdout
    ok = True
    for eps in (1, 2):
        for seed in (6101, 6102, 6103):
            x_ref, s_ref = _sigma_run(evladder, eps, seed, 256)
            for g in SIGN_GRIDS:
                x, s = _sigma_run(evladder, eps, seed, g)
                if x.shape != x_ref.shape or np.any(np.abs(x - x_ref) > 1e-12 * np.maximum(1.0, np.abs(x_ref))):
                    raise ReduceError("plane lists differ for eps=%d seed=%d g=%d" % (eps, seed, g))
                neg = int(np.sum(s < s_ref))
                rel = (s - s_ref) / s_ref
                exp = SIGN_EXPECTED[(eps, seed)]
                good = neg == exp and len(x) == SIGN_N_PLANES
                ok = ok and good
                print("eps=%d seed=%d lh=%3d sigmaT2<ref %2d/%d rel first/last %+.2e/%+.2e expected %2d/%d %s"
                      % (eps, seed, g // 2, neg, len(x), rel[0], rel[-1], exp, SIGN_N_PLANES, "OK" if good else "MISMATCH"), file=out)
    print("SIGNTABLE %s" % ("EQUAL" if ok else "DIFFERENT"), file=out)
    return ok


# ------------------------------------------------------------------------------------------------
# cross-check of the reducer's moments against the runner's own C++ moments (smoke, RR3 (iv))
# ------------------------------------------------------------------------------------------------
XC_RTOL, XC_ATOL, XC_FLOOR = 1e-9, 1e-15, 1e-6  # |a-b| <= RTOL max(|a|,|b|) + ATOL; max_rel scale floor ATOL/RTOL


def _near(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= XC_RTOL * max(abs(a), abs(b)) + XC_ATOL


def _rel(a, b):
    if math.isnan(a) or math.isnan(b):
        return 0.0 if (math.isnan(a) and math.isnan(b)) else math.inf
    return abs(a - b) / max(abs(a), abs(b), XC_FLOOR)


def _mean_var(x, w):
    """Biased weighted mean and variance (NaN, NaN if empty or sum w <= 0: the runner's rule)."""
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    if x.size == 0 or not (w.sum() > 0.0):
        return NAN, NAN
    m = float(np.sum(w * x) / w.sum())
    return m, float(np.sum(w * (x - m) ** 2) / w.sum())


def _compare(pairs):
    """(ok, max_rel) of a list of (mine, runner) float pairs."""
    return all(_near(a, b) for a, b in pairs), max([_rel(a, b) for a, b in pairs] + [0.0])


def crosscheck_planes(P, csv_path, uniform=False):
    """(n_bad_rows, max_rel): sigma_series_rows vs the runner's moments_planes.csv (x1, contributing, mean/var x2, x3)."""
    mine, ref = sigma_series_rows(P, uniform), sio.read_csv_rows(csv_path)
    if len(mine) != len(ref):
        return abs(len(mine) - len(ref)) or 1, math.inf
    bad, mx = 0, 0.0
    for m, r in zip(mine, ref):
        ok, rel = _compare([(m[k], r[k]) for k in ("mean_x2", "var_x2", "mean_x3", "var_x3")])
        ok = ok and int(r["contributing"]) == m["n_contrib"] and abs(r["x1"] - m["x1"]) <= 1e-12 * max(1.0, abs(m["x1"]))
        bad += not ok
        mx = max(mx, rel)
    return bad, mx


def crosscheck_times(P, csv_path, uniform=False):
    """(n_bad_rows, max_rel): per-snapshot weighted mean/biased variance of x1,x2,x3 over the records at j vs moments_time.csv."""
    ref = sio.read_csv_rows(csv_path)
    if len(ref) != len(P.times):
        return abs(len(ref) - len(P.times)) or 1, math.inf
    w = np.ones(len(P.weights)) if uniform else np.asarray(P.weights, dtype=float)
    bad, mx = 0, 0.0
    for j, r in enumerate(ref):
        X = np.asarray(P.times_x[:, j, :], dtype=float)
        f = _fin(X)
        pairs = []
        for k, (mk, vk) in enumerate((("mean_x", "var_x"), ("mean_y", "var_y"), ("mean_z", "var_z"))):
            m, v = _mean_var(X[f, k], w[f])
            pairs += [(m, r[mk]), (v, r[vk])]
        ok, rel = _compare(pairs)
        ok = ok and int(r["active"]) == int(f.sum()) and abs(r["t"] - P.times[j]) <= 1e-12 * max(1.0, abs(P.times[j]))
        bad += not ok
        mx = max(mx, rel)
    return bad, mx


def crosscheck_pass(root):
    """{'planes': (n_bad, max_rel), 'times': (...), 'twin': (...) or None} for one pass root (twin: uniform weights, absent
    twin/ directory -> None).  Missing moments CSVs raise MissingData."""
    P = Pass(root)
    out = {"planes": crosscheck_planes(P, os.path.join(root, "moments_planes.csv")),
           "times": crosscheck_times(P, os.path.join(root, "moments_time.csv")), "twin": None}
    if os.path.isdir(os.path.join(root, "twin")):
        a = crosscheck_planes(P, os.path.join(root, "twin", "moments_planes.csv"), uniform=True)
        b = crosscheck_times(P, os.path.join(root, "twin", "moments_time.csv"), uniform=True)
        out["twin"] = (a[0] + b[0], max(a[1], b[1]))
    return out


def crosscheck_run(rdirs):
    """Aggregate crosscheck_pass over every pass of every realization directory; missing CSVs count as 1 bad."""
    agg = {"planes": [0, 0.0], "times": [0, 0.0], "twin": None}
    for rdir in rdirs:
        for root in sio.pass_roots(rdir):
            try:
                res = crosscheck_pass(root)
            except MissingData:
                agg["planes"][0] += 1
                agg["times"][0] += 1
                continue
            for k, v in res.items():
                if v is None:
                    continue
                if agg[k] is None:
                    agg[k] = [0, 0.0]
                agg[k][0] += v[0]
                agg[k][1] = max(agg[k][1], v[1])
    return agg


def crosscheck_line(name, agg):
    st = lambda v: "absent" if v is None else ("ok" if v[0] == 0 else "FAIL %d" % v[0])
    mx = max([v[1] for v in agg.values() if v is not None] + [0.0])
    return "CROSSCHECK run=%s planes=%s times=%s twin=%s max_rel=%.3g" % (name, st(agg["planes"]), st(agg["times"]), st(agg["twin"]), mx)


def crosscheck_ok(agg):
    return all(v is None or v[0] == 0 for v in agg.values())


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _iter_realizations(out_dir, subset, div):
    warn = sys.stderr
    for r, path, done in sio.list_realizations(out_dir, subset):
        if not done:
            print("NOT_DONE (skipped, reported): %s" % path, file=warn)
            continue
        yield r, Realization(path, div)


def cmd_tb(a):
    rows = []
    for r, real in _iter_realizations(a.out_dir, None, a.pass_div):
        xb, ib = real.key_planes()["Xb"]
        P = real.pass_
        ok = (np.asarray(P.status_plane) == 0) & np.isfinite(np.asarray(P.planes_tau[:, ib]))
        rows.append({"r": r, "seed": real.seed, "qbar": real.qbar, "X_b": xb, "n_ok": int(ok.sum()),
                     "t_b": breakthrough_time(P, ib)})
    write_csv(a.dest, rows, ["r", "seed", "qbar", "X_b", "n_ok", "t_b"])
    return 0


def cmd_alpha(a):
    rows = []
    for r, real in _iter_realizations(a.out_dir, parse_range(a.realizations) if a.realizations else None, a.pass_div):
        if abs(real.ell - a.ell_run) > 1e-12 * abs(a.ell_run):
            raise ReduceError("--ell-run %.17g differs from field.json ell %.17g (r=%d)" % (a.ell_run, real.ell, r))
        common, methods = reduce_methods(real, a.tau_b_star, a.U, r=r)
        rows += long_rows(common, methods)
    write_csv(a.dest, rows)
    return 0


def cmd_signtable(a):
    return 0 if signtable(a.evladder_dir) else 1


def cmd_smoke(a):
    print("SMOKE on existing SF-41 outputs: NOT EVIDENCE (script check only; missing planes/twin tolerated and reported)")
    runs = [os.path.join(a.campaign_dir, "prodgrid", "E%d" % k) for k in (1, 2, 3, 4)]
    ev = os.path.join(a.campaign_dir, "evladder")
    runs += [os.path.join(ev, n) for n in sorted(os.listdir(ev)) if n.startswith("F_") and os.path.isdir(os.path.join(ev, n))] \
        if os.path.isdir(ev) else []
    n_run = n_fail = 0
    for run in runs:
        name = os.path.relpath(run, a.campaign_dir)
        try:
            reals = [(r, Realization(p, None)) for r, p, done in sio.list_realizations(run) if done]
        except MissingData as e:
            print("%-28s ABSENT (%s)" % (name, e))
            continue
        if not reals:
            print("%-28s no completed realization" % name)
            continue
        n_run += 1
        notes = []
        try:
            xbs = [rl.key_planes(True)["Xb"][1] for _, rl in reals]
            tbs = [breakthrough_time(rl.pass_, ib) for (_, rl), ib in zip(reals, xbs) if ib is not None]
            for _, rl in reals:
                sigma_series_rows(rl.pass_)
            notes.append("sigma_series ok")
            if not tbs:
                notes.append("window skipped: no X_b plane")
            else:
                U = float(np.mean([rl.qbar for _, rl in reals]))
                tau = U * min(tbs) / reals[0][1].ell
                nred = 0
                for r, rl in reals:
                    try:
                        common, methods = reduce_methods(rl, tau, U, tolerate_missing=True, r=r)
                        nred += 1
                        if common["skipped"]:
                            notes.append("r%d skipped[%s]" % (r, common["skipped"]))
                    except (MissingData, WindowError) as e:
                        notes.append("r%d not applicable: %s" % (r, e))
                notes.append("reduced %d/%d (tau_b*=%.4g from t_b,min=%.4g)" % (nred, len(reals), tau, min(tbs)))
        except Exception as e:  # a script defect: everything that can run must run
            n_fail += 1
            notes.append("FAILED %s: %s" % (type(e).__name__, e))
        print("%-28s %s" % (name, "; ".join(notes)))
        try:
            agg = crosscheck_run([p for _, p, done in sio.list_realizations(run) if done])
            print(crosscheck_line(name, agg))
            n_fail += not crosscheck_ok(agg)
        except Exception as e:
            n_fail += 1
            print("CROSSCHECK run=%s FAILED %s: %s" % (name, type(e).__name__, e))
    print("SMOKE runs=%d failed=%d (NOT EVIDENCE)" % (n_run, n_fail))
    return 0 if n_run > 0 and n_fail == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf39.reduce", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("tb", help="t_b,r and qbar per realization (CSV)")
    s.add_argument("out_dir")
    s.add_argument("--pass-div", type=int, default=8)
    s.add_argument("--dest", default=None)
    s.set_defaults(fn=cmd_tb)
    s = sub.add_parser("alpha", help="window + plane estimators, one row per realization and method")
    s.add_argument("out_dir")
    s.add_argument("--tau-b-star", type=float, required=True)
    s.add_argument("--ell-run", type=float, required=True)
    s.add_argument("--U", type=float, required=True)
    s.add_argument("--dest", required=True)
    s.add_argument("--pass-div", type=int, default=8)
    s.add_argument("--realizations", default=None, help="e.g. 0-19 or 0-4,7")
    s.set_defaults(fn=cmd_alpha)
    s = sub.add_parser("smoke", help="every applicable reduction on SF-41 outputs (NOT EVIDENCE)")
    s.add_argument("campaign_dir")
    s.set_defaults(fn=cmd_smoke)
    s = sub.add_parser("signtable", help="reproduce the sign table of the pre-activation review 5.4")
    s.add_argument("evladder_dir")
    s.set_defaults(fn=cmd_signtable)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
