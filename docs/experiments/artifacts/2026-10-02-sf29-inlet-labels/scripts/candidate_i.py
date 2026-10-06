#!/usr/bin/env python3
"""SF-29 N2: candidate (i) -- non-divergence form of the same-index equation (14) on the slab with inlet labels.

Grid (shared with N1, see metrics.py / cases.py): vertices x1 = j h (j = 0..N), x2, x3 = m h periodic (m = 0..N-1),
h = 1/N.  Unknowns psi1 = x2 + u1, psi2 = x3 + u2 on planes j = 1..N (u_i periodic in x2, x3); plane j = 0 holds
the inlet labels psi0 of cases.load_case (Dirichlet).  Only the periodic parts u_i are differenced; the affine
gradients e2, e3 are added exactly (as metrics.label_gradients does).

Operator (non-divergence form, same-index pairing, decision 2026-09-30):
    L_i psi_i = lap_h psi_i - grad(ln k) . grad_h psi_i        (grad ln k ANALYTIC at the vertices)
    B   = H(psi2) grad psi1 - H(psi1) grad psi2,  c = grad_h psi1 x grad_h psi2
    S_i = ((B x grad_h psi_i) . c) / |c|^2                       (NO regularization of |c|^2)
    F_i = -q (L_i psi_i - S_i),  q = 1/k
Second-order centered stencils on planes 1..N-1 (d1, d11, mixed d1j = centered d1 of the centered dj; in-plane
d22, d33, d23 and d2, d3 centered periodic).

Outlet plane j = N, two variants:
  i0 (spec-literal control, prediction P1): equation rows also at j = N with second-order one-sided x1 stencils
     d1 = (3u_N - 4u_{N-1} + u_{N-2})/(2h), d11 = (2u_N - 5u_{N-1} + 4u_{N-2} - u_{N-3})/h^2,
     d1j = one-sided d1 of the centered dj; no boundary condition.
  i1 (orchestrator's closure, prediction P3 / deviation D-2): equation rows on planes 1..N-1 only; at j = N the two
     rows per vertex impose (grad_h psi1 x grad_h psi2) x e1 = vperp_in x e1, i.e. c2 = v2_in, c3 = v3_in, with
     vperp_in the tangential Darcy velocity on the INLET face at the same (x2, x3) (identically 0 for `_ch` fields,
     where this is the Neumann condition d1 psi_i = 0); d1 by the one-sided stencil above, in-plane centered.

Residual norms (SF-26 probe convention):
    r_F   = sqrt((RMS F1^2 + RMS F2^2)/2) / q_rms        over the equation rows only
    r_out = RMS|c_perp(x1=1) - vperp_in| / v_rms          (i1 outlet rows, reported separately)
Newton system (one vector E, row index = unknown index (field, plane j = 1..N, m2, m3)):
    equation rows   E = F_i                                 (i0: all planes; i1: planes 1..N-1)
    i1 outlet rows  E_1 = (2 q/h)(v2_in - c2),  E_2 = (2 q/h)(v3_in - c3)
The outlet-row scale 2q/h is the ghost-point Neumann row scale of -q lap (it makes the linearization at k = 1 of the
outlet row equal to (2q/h) d1 u_i); it changes neither the solution nor the Newton step (exact solve), only GMRES
conditioning and the relative singular values of the dense spectrum (stated next to every spectrum).

Nonlinear method: damped Newton (backtracking on the merit sqrt(r_F^2 + r_out^2) = r_F for i0), Jacobian by colored
central finite differences of E (sparsity: 3x3x3 stencil, outlet rows reach j = N-3, both fields; colors
(field, j mod 4, m2 mod p, m3 mod p) with p the smallest divisor >= 3 of N), assembled as scipy.sparse CSR.
Linear solve: splu for N <= --direct-max (default 32 since C-i4; was 24) (fallback lsmr min-norm least squares if splu fails or its step residual is
> 1e-8, reported); N >= 32: right-preconditioned restarted GMRES (CGS2, tolerance --lin-tol relative on the
true residual; restart --restart, default 300: restart 50 stagnated at 9e-2 on the
second Newton step of gauss:0.25:32:i1) with preconditioner = exact inverse of the k = 1 linearization of THIS discretization per transverse
Fourier mode (`--prec lin0`, the coupled 2x2 operator (d11+d22)u1 + d23 u2, (d11+d33)u2 + d23 u1 with the same
inlet/outlet row structure; FFT in x2, x3, dense per-mode 2N x 2N inverse in x1) or the per-field slab Laplacian
(`--prec lap`, the operator named in the task).  Modes that are exactly singular (i0) use the pseudo-inverse.
Amplitude continuation eps 0.25 -> 0.5 -> 1.0 up to the target (warm start; the first start from u = 0, which is
the exact solution at eps = 0).  A failed stage (line-search failure / stagnation / maxit with merit > STAGE_OK) is
retried at the midpoint amplitude from the last converged one (at most --bisect times); the path actually taken is
printed (PATH line).

Usage (run from this directory):
  python3 candidate_i.py <field>:<eps>:<N>:<variant> ...          variant in {i0, i1}
  python3 candidate_i.py --jactest gauss:0.25:12:i1                Jacobian action vs FD (3 step sizes)
  python3 candidate_i.py --consistency gauss:0.25 gauss_ch:0.25    residual at the oracle labels, N = 16/32/48
  python3 candidate_i.py --spectrum 12 gauss:0.25:12:i0 ...        dense FD Jacobian SVD at the converged state
  python3 candidate_i.py --k1check 12                               k = 1 control: exact nulls of i0/i1, preconditioner
Options: --maxit K (Newton iterations per amplitude, default 40)  --tol T (default 1e-13)
         --no-continuation  --out DIR (default ../raw)  --prec lin0|lap  --lin-tol T (default 1e-13)
         --direct-max K (splu for N <= K, GMRES above; default 32)  --restart K (GMRES restart, default 300)
         --reuse-lu K (C-i4: reuse the last LU as GMRES preconditioner for up to K Newton steps; default 0)
         --bisect K (continuation bisections, default 4)  --lm K (i0 Levenberg-Marquardt fallback its, N <= 16,
         default 30; 0 disables)  --grids 16,32,48 (for --consistency)
         --init zero|inlet|oracle (default zero; oracle = DIAGNOSTIC start at the oracle labels)
Every result prints the shared CASE line (metrics.print_case, cand=i0/i1) and the oracle-FD ceiling line
(cand=oracle_fd: metrics.fd_metrics of the oracle labels on the same grid, deviation D-5).

C-i4 additions:
  --order 4 (variant i1 only, cand=i1o4): the same equations discretized at 4th order (class Stencils4, derivs4,
     Pattern4, 4th-order ModePrec; see the README section "Candidate (i): 4th-order variant"); CASE lines
     cand=i1o4 (4th-order reconstruction, ceiling cand=oracle_fd4) and cand=i1o4_fd2 (2nd-order reconstruction,
     ceiling cand=oracle_fd).
  python3 candidate_i.py --selfcheck --order 4        stencil exactness on polynomials, observed orders
  --save DIR       DIR/<field>_<eps>_<N>_<cand>.npz after every solve (u1, u2 on planes 0..N, status, history,
                   metrics)
  python3 candidate_i.py --remetric FILE...           CASE lines of saved solutions with the current metrics
  Every direct factorization prints a LINEAR line (t_fact, t_solve, nnz(L+U), fill, mem estimate, peak RSS).
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import cases as C
import metrics as M

_HERE = os.path.dirname(os.path.abspath(__file__))
EPS_LADDER = (0.25, 0.5, 1.0)
STAGE_OK = 1e-9          # an intermediate continuation stage is a usable warm start below this merit


# ------------------------------------------------------------------------------------------------
# case parsing and loading
# ------------------------------------------------------------------------------------------------
def parse_cand_spec(spec, order=2):
    """'field:eps:N:variant' -> (field, eps, N, variant, order).  N and variant optional (defaults: 16, i1);
    variant 'i1o4' is the same as 'i1' with --order 4."""
    parts = spec.split(":")
    variant = "i1"
    if parts[-1] in ("i0", "i1", "i1o4"):
        variant = parts[-1]; parts = parts[:-1]
    if variant == "i1o4":
        variant = "i1"; order = 4
    if order == 4 and variant != "i1":
        raise SystemExit("--order 4 is implemented for variant i1 only (i1o4); got %s" % spec)
    field, eps, N = C.parse_spec(":".join(parts))
    return field, eps, (16 if N is None else N), variant, order


def cand_name(variant, order):
    return variant + ("o4" if order == 4 else "")


def light_case(field, eps, N, nphi=None):
    """Data needed by the residual only (no oracle): k, grad ln k, inlet labels, inlet v_perp, v_rms.
    Built from the N1 primitives (cases.build_reference, InletLabels.labels, DarcyReference.velocity);
    used for the intermediate continuation amplitudes, whose oracle is not needed."""
    ref, inl = C.build_reference(field, eps, nphi=nphi, verbose=False)
    x, Y, Z = C.vertex_coords(N)
    X1v = np.broadcast_to((np.arange(N + 1) / float(N))[:, None, None], (N + 1, N, N))
    X2v = np.broadcast_to(x[None, :, None], (N + 1, N, N))
    X3v = np.broadcast_to(x[None, None, :], (N + 1, N, N))
    lnk = ref.lk.lnk(X1v, X2v, X3v)
    p1, p2 = inl.labels(Y, Z)
    vin = ref.velocity(0.0, Y, Z)
    vD = C.vertex_velocity(ref, N)
    return {"k": np.exp(lnk), "lnk": lnk, "grad_lnk": tuple(ref.lk.grad_lnk(X1v, X2v, X3v)),
            "psi0": (p1.reshape(N, N), p2.reshape(N, N)),
            "vperp_in": (vin[:, 1].reshape(N, N), vin[:, 2].reshape(N, N)),
            "vD": (vD[0], vD[1], vD[2]), "meta": {"field": field, "eps": float(eps), "N": int(N)}}


class Ctx(object):
    """Discretization context: everything the residual needs for one (case, N, variant)."""

    def __init__(self, case, variant, order=2):
        k = case["k"]; N = k.shape[1]
        self.N = N; self.h = 1.0 / N; self.variant = variant; self.order = order
        self.cand = cand_name(variant, order)
        if order == 4:
            if variant != "i1":
                raise ValueError("order 4 is implemented for variant i1 only")
            self.st4 = Stencils4(N)
        self.q = 1.0 / k[1:]                                   # planes 1..N
        self.glnk = tuple(g[1:] for g in case["grad_lnk"])
        x = np.arange(N) / float(N)
        self.u0 = (case["psi0"][0] - x[:, None], case["psi0"][1] - x[None, :])   # periodic parts on the inlet
        self.vperp = case["vperp_in"]
        self.v_rms = M.rms_vec(case["vD"])
        neq = N if variant == "i0" else N - 1
        self.neq = neq
        self.q_rms = float(np.sqrt(np.mean(self.q[:neq] ** 2)))
        self.n = N ** 3
        # row scale used by the preconditioner: the Newton rows divided by these give the k = 1 linear rows
        s = np.empty((2, N, N, N))
        s[0] = self.q; s[1] = self.q
        self.rowscale = s


# ------------------------------------------------------------------------------------------------
# discrete operators
# ------------------------------------------------------------------------------------------------
def _rp(a, s, ax):
    """_rp(a, 1, ax)[i] = a[i+1] (periodic)."""
    return np.roll(a, -s, axis=ax)


def _d1op(A, h):
    """x1-derivative from planes 0..N to planes 1..N: centered on 1..N-1, one-sided second order at N."""
    N = A.shape[0] - 1
    out = np.empty((N,) + A.shape[1:])
    out[:-1] = (A[2:] - A[:-2]) / (2 * h)
    out[-1] = (3 * A[N] - 4 * A[N - 1] + A[N - 2]) / (2 * h)
    return out


def _d11op(A, h):
    N = A.shape[0] - 1
    out = np.empty((N,) + A.shape[1:])
    out[:-1] = (A[2:] - 2 * A[1:-1] + A[:-2]) / (h * h)
    out[-1] = (2 * A[N] - 5 * A[N - 1] + 4 * A[N - 2] - A[N - 3]) / (h * h)
    return out


def derivs(U, h):
    """Gradient (3) and Hessian (dict) of a periodic part U (planes 0..N) on planes 1..N."""
    D2 = (_rp(U, 1, 1) - _rp(U, -1, 1)) / (2 * h)
    D3 = (_rp(U, 1, 2) - _rp(U, -1, 2)) / (2 * h)
    V = U[1:]
    g = (_d1op(U, h), D2[1:], D3[1:])
    H = {(0, 0): _d11op(U, h),
         (1, 1): (_rp(V, 1, 1) - 2 * V + _rp(V, -1, 1)) / (h * h),
         (2, 2): (_rp(V, 1, 2) - 2 * V + _rp(V, -1, 2)) / (h * h),
         (0, 1): _d1op(D2, h), (0, 2): _d1op(D3, h),
         (1, 2): (_rp(D2[1:], 1, 2) - _rp(D2[1:], -1, 2)) / (2 * h)}
    for (a, b) in list(H.keys()):
        H[(b, a)] = H[(a, b)]
    return g, H


class Stencils4(object):
    """4th-order x1 stencils from planes 0..N to the unknown planes 1..N (C-i4, variant i1o4).

    Planes 2..N-2: centered (offsets -2..2).  Planes 1 and N-1: skewed (d1: 5 points 0..4 / N-4..N; d11: 6 points
    0..5 / N-5..N).  Plane N: one-sided (d1: 5 points N-4..N = (25, -48, 36, -16, 3)/(12h); d11: 6 points
    N-5..N, used only by the equation pieces that the outlet rows overwrite).  Weights from metrics.fd_weights
    (Fornberg), stored times h^deriv."""

    def __init__(self, N):
        if N < 8:
            raise ValueError("order 4 needs N >= 8")
        self.N = N
        b1, self.c1 = M.x1_stencils4(N, 1)
        b2, self.c2 = M.x1_stencils4(N, 2)
        self.b1 = {p: b1[p] for p in (1, N - 1, N)}
        self.b2 = {p: b2[p] for p in (1, N - 1, N)}

    def stencil(self, p, deriv):
        """(first plane, weights) of the x1 stencil at plane p (1..N)."""
        b = self.b1 if deriv == 1 else self.b2
        if p in b:
            return b[p]
        return p - 2, (self.c1 if deriv == 1 else self.c2)

    def apply(self, A, h, deriv):
        """x1-derivative of A (planes 0..N, axis 0) on planes 1..N."""
        N = self.N
        cw = self.c1 if deriv == 1 else self.c2
        b = self.b1 if deriv == 1 else self.b2
        out = np.empty((N,) + A.shape[1:])
        out[1:N - 2] = sum(cw[k] * A[k:N - 3 + k] for k in range(5))       # planes 2..N-2
        for p, (s, w) in b.items():
            out[p - 1] = sum(w[k] * A[s + k] for k in range(w.size))
        return out / h ** deriv


def _dp4(a, ax, h):
    return (-_rp(a, 2, ax) + 8 * _rp(a, 1, ax) - 8 * _rp(a, -1, ax) + _rp(a, -2, ax)) / (12 * h)


def _dpp4(a, ax, h):
    return (-_rp(a, 2, ax) + 16 * _rp(a, 1, ax) - 30 * a + 16 * _rp(a, -1, ax) - _rp(a, -2, ax)) / (12 * h * h)


def derivs4(U, h, st):
    """4th-order gradient and Hessian of a periodic part U (planes 0..N) on planes 1..N: centered 4th order in
    x2, x3; x1 by Stencils4; mixed d1j = x1 stencil of the 4th-order dj, d23 = d3(d2)."""
    D2 = _dp4(U, 1, h)
    D3 = _dp4(U, 2, h)
    V = U[1:]
    g = (st.apply(U, h, 1), D2[1:], D3[1:])
    H = {(0, 0): st.apply(U, h, 2),
         (1, 1): _dpp4(V, 1, h), (2, 2): _dpp4(V, 2, h),
         (0, 1): st.apply(D2, h, 1), (0, 2): st.apply(D3, h, 1),
         (1, 2): _dp4(D2[1:], 2, h)}
    for (a, b) in list(H.keys()):
        H[(b, a)] = H[(a, b)]
    return g, H


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _Hv(H, g):
    return tuple(H[(i, 0)] * g[0] + H[(i, 1)] * g[1] + H[(i, 2)] * g[2] for i in range(3))


def full_planes(uvec, ctx):
    N = ctx.N
    u = uvec.reshape(2, N, N, N)
    U1 = np.concatenate([ctx.u0[0][None], u[0]], axis=0)
    U2 = np.concatenate([ctx.u0[1][None], u[1]], axis=0)
    return U1, U2


def pieces(U1, U2, ctx):
    """All residual pieces on planes 1..N (arrays (N, N, N))."""
    h = ctx.h
    if ctx.order == 4:
        d1, H1 = derivs4(U1, h, ctx.st4)
        d2, H2 = derivs4(U2, h, ctx.st4)
    else:
        d1, H1 = derivs(U1, h)
        d2, H2 = derivs(U2, h)
    g1 = (d1[0], 1.0 + d1[1], d1[2])
    g2 = (d2[0], d2[1], 1.0 + d2[2])
    B = tuple(a - b for a, b in zip(_Hv(H2, g1), _Hv(H1, g2)))
    c = _cross(g1, g2)
    cc = _dot(c, c)
    S1 = _dot(_cross(B, g1), c) / cc
    S2 = _dot(_cross(B, g2), c) / cc
    L1 = H1[(0, 0)] + H1[(1, 1)] + H1[(2, 2)] - _dot(ctx.glnk, g1)
    L2 = H2[(0, 0)] + H2[(1, 1)] + H2[(2, 2)] - _dot(ctx.glnk, g2)
    F1 = -ctx.q * (L1 - S1)
    F2 = -ctx.q * (L2 - S2)
    return {"F1": F1, "F2": F2, "c": c, "cc": cc}


def system(uvec, ctx, want_parts=False):
    """Newton residual vector E (same layout as uvec) and the norms (r_F, r_out)."""
    U1, U2 = full_planes(uvec, ctx)
    p = pieces(U1, U2, ctx)
    N = ctx.N; neq = ctx.neq
    E = np.empty((2, N, N, N))
    E[0] = p["F1"]; E[1] = p["F2"]
    F1e = p["F1"][:neq]; F2e = p["F2"][:neq]
    r_F = float(np.sqrt((np.mean(F1e ** 2) + np.mean(F2e ** 2)) / 2.0)) / ctx.q_rms
    r_out = 0.0
    if ctx.variant == "i1":
        c = p["c"]
        d2 = c[1][-1] - ctx.vperp[0]
        d3 = c[2][-1] - ctx.vperp[1]
        sc = 2.0 * ctx.q[-1] / ctx.h
        E[0][-1] = -sc * d2
        E[1][-1] = -sc * d3
        r_out = float(np.sqrt(np.mean(d2 ** 2 + d3 ** 2))) / ctx.v_rms
    if want_parts:
        return E.ravel(), r_F, r_out, p
    return E.ravel(), r_F, r_out


def merit(r_F, r_out):
    return float(np.sqrt(r_F ** 2 + r_out ** 2))


# ------------------------------------------------------------------------------------------------
# colored finite-difference sparse Jacobian
# ------------------------------------------------------------------------------------------------
def _period(N):
    for p in range(3, N + 1):
        if N % p == 0:
            return p
    return N


class Pattern(object):
    """Sparsity pattern (row, col) and the coloring of the columns."""

    def __init__(self, N):
        self.N = N
        n = N ** 3
        self.n = n
        idx = np.arange(2 * n, dtype=np.int64)
        f = idx // n; rem = idx % n
        jj = rem // (N * N) + 1                      # plane 1..N
        m2 = (rem // N) % N; m3 = rem % N
        rows = []; cols = []
        offs = [(fr, dj, a, b) for fr in (0, 1) for dj in (-1, 0, 1) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        for (fr, dj, a, b) in offs:
            jr = jj + dj
            ok = (jr >= 1) & (jr <= N)
            r = fr * n + (jr - 1) * N * N + ((m2 + a) % N) * N + (m3 + b) % N
            rows.append(r[ok]); cols.append(idx[ok])
        # outlet rows (plane N) depend on planes N-3..N
        sel = jj >= N - 3
        for fr in (0, 1):
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    r = fr * n + (N - 1) * N * N + ((m2[sel] + a) % N) * N + (m3[sel] + b) % N
                    rows.append(r); cols.append(idx[sel])
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        key = np.unique(rows * (2 * n) + cols)
        self.rows = key // (2 * n); self.cols = key % (2 * n)
        p = _period(N); self.p = p
        self.color = (f * 4 + (jj - 1) % 4) * p * p + (m2 % p) * p + (m3 % p)
        self.ncolor = int(self.color.max()) + 1
        # entries grouped by the color of their column
        cc = self.color[self.cols]
        order = np.argsort(cc, kind="stable")
        self.rows = self.rows[order]; self.cols = self.cols[order]
        self.bounds = np.searchsorted(cc[order], np.arange(self.ncolor + 1))
        # coloring check: inside one color every row appears at most once
        for g in range(self.ncolor):
            r = self.rows[self.bounds[g]:self.bounds[g + 1]]
            if r.size != np.unique(r).size:
                raise RuntimeError("coloring conflict in color %d (N=%d p=%d)" % (g, N, p))


def _period_min(N, pmin):
    for p in range(pmin, N + 1):
        if N % p == 0:
            return p
    return N


class Pattern4(object):
    """Sparsity pattern and column coloring of the 4th-order i1 system (C-i4).

    Equation row at plane jr (1..N-1), vertex (m2, m3), either field: columns of both fields on the union of the
    x1 stencil planes of d1 and d11 at jr (unknown planes >= 1); on plane jr itself the 5x5 in-plane box (d22, d33,
    d23 = d3 d2, d2, d3), on the other planes the radius-2 in-plane cross (d1, d11 at the vertex; d1j = x1 stencil
    of the in-plane dj).  Outlet rows (plane N): the radius-2 cross on plane N (in-plane d2, d3 of c) and the
    vertex on planes N-4..N-1 (one-sided d1).  Colors (field, (j-1) mod 6, m2 mod p, m3 mod p), p the smallest
    divisor >= 5 of N (two columns of one row differ by <= 5 planes and <= 4 in-plane steps); the column-collision
    check of Pattern is kept."""

    def __init__(self, N):
        self.N = N
        n = N ** 3
        self.n = n
        st = Stencils4(N)
        box = [(a, b) for a in range(-2, 3) for b in range(-2, 3)]
        cross = [(a, 0) for a in range(-2, 3)] + [(0, b) for b in (-2, -1, 1, 2)]
        deps = {}
        for jr in range(1, N):
            planes = set()
            for deriv in (1, 2):
                s, w = st.stencil(jr, deriv)
                planes |= set(range(s, s + w.size))
            planes.add(jr)
            deps[jr] = [(jc, box if jc == jr else cross) for jc in sorted(planes) if jc >= 1]
        s, w = st.stencil(N, 1)
        deps[N] = [(jc, cross if jc == N else [(0, 0)]) for jc in range(max(s, 1), N + 1)]
        self.deps = deps
        m2, m3 = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        m2 = m2.ravel(); m3 = m3.ravel()
        rows = []; cols = []
        for jr, dl in deps.items():
            rin = (jr - 1) * N * N + m2 * N + m3
            for (jc, offs) in dl:
                for (a, b) in offs:
                    cin = (jc - 1) * N * N + ((m2 + a) % N) * N + (m3 + b) % N
                    for fr in (0, 1):
                        for fc in (0, 1):
                            rows.append(fr * n + rin); cols.append(fc * n + cin)
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        key = np.unique(rows.astype(np.int64) * (2 * n) + cols)
        self.rows = key // (2 * n); self.cols = key % (2 * n)
        p = _period_min(N, 5); self.p = p; self.p1 = 6
        idx = np.arange(2 * n, dtype=np.int64)
        f = idx // n; rem = idx % n
        jj = rem // (N * N) + 1
        cm2 = (rem // N) % N; cm3 = rem % N
        self.color = (f * self.p1 + (jj - 1) % self.p1) * p * p + (cm2 % p) * p + (cm3 % p)
        self.ncolor = int(self.color.max()) + 1
        cc = self.color[self.cols]
        order = np.argsort(cc, kind="stable")
        self.rows = self.rows[order]; self.cols = self.cols[order]
        self.bounds = np.searchsorted(cc[order], np.arange(self.ncolor + 1))
        for g in range(self.ncolor):
            r = self.rows[self.bounds[g]:self.bounds[g + 1]]
            if r.size != np.unique(r).size:
                raise RuntimeError("coloring conflict in color %d (N=%d p=%d, order 4)" % (g, N, p))


def make_pattern(N, order):
    return Pattern(N) if order == 2 else Pattern4(N)


def fd_delta(uvec):
    return 1e-6 * (1.0 + float(np.max(np.abs(uvec))))


def jacobian_colored(uvec, ctx, pat):
    delta = fd_delta(uvec)
    vals = np.empty(pat.rows.size)
    for g in range(pat.ncolor):
        cols = np.nonzero(pat.color == g)[0]
        up = uvec.copy(); up[cols] += delta
        um = uvec.copy(); um[cols] -= delta
        d = (system(up, ctx)[0] - system(um, ctx)[0]) / (2 * delta)
        sl = slice(pat.bounds[g], pat.bounds[g + 1])
        vals[sl] = d[pat.rows[sl]]
    J = sp.csr_matrix((vals, (pat.rows, pat.cols)), shape=(2 * pat.n, 2 * pat.n))
    return J


def jacobian_dense(uvec, ctx):
    delta = fd_delta(uvec)
    nn = uvec.size
    J = np.empty((nn, nn))
    for col in range(nn):
        up = uvec.copy(); up[col] += delta
        um = uvec.copy(); um[col] -= delta
        J[:, col] = (system(up, ctx)[0] - system(um, ctx)[0]) / (2 * delta)
    return J


# ------------------------------------------------------------------------------------------------
# preconditioner: per transverse Fourier mode inverse of the k = 1 linearization (or slab Laplacian)
# ------------------------------------------------------------------------------------------------
class ModePrec(object):
    def __init__(self, N, variant, kind="lin0", order=2):
        self.N = N; self.kind = kind; self.variant = variant; self.order = order
        h = 1.0 / N
        th = 2 * np.pi * np.arange(N) / N
        th3 = th[: N // 2 + 1]
        T2, T3 = np.meshgrid(th, th3, indexing="ij")       # (N, N//2+1)
        if order == 2:
            a2 = (2 * np.cos(T2) - 2) / h ** 2
            a3 = (2 * np.cos(T3) - 2) / h ** 2
            b = -np.sin(T2) * np.sin(T3) / h ** 2
        else:                                               # symbols of the 4th-order d22, d33, d23 = d3 d2
            a2 = (-2 * np.cos(2 * T2) + 32 * np.cos(T2) - 30) / (12 * h ** 2)
            a3 = (-2 * np.cos(2 * T3) + 32 * np.cos(T3) - 30) / (12 * h ** 2)
            s2 = (8 * np.sin(T2) - np.sin(2 * T2)) / (6 * h)
            s3 = (8 * np.sin(T3) - np.sin(2 * T3)) / (6 * h)
            b = -s2 * s3
        nm = a2.size
        a2 = a2.ravel(); a3 = a3.ravel(); b = b.ravel()
        D11 = np.zeros((N, N))                              # unknown planes 1..N -> rows 1..N (u_0 = 0)
        if order == 2:
            for r in range(N - 1):                          # row plane r+1
                D11[r, r] = -2.0
                D11[r, r + 1] = 1.0
                if r >= 1:
                    D11[r, r - 1] = 1.0
        else:
            st = Stencils4(N)
            for r in range(N - 1):
                s, w = st.stencil(r + 1, 2)
                for kk in range(w.size):
                    if s + kk >= 1:
                        D11[r, s + kk - 1] += w[kk]
        outlet = np.zeros(N)
        if variant == "i0":                                 # one-sided d11 at the outlet
            for c, w in ((N, 2.0), (N - 1, -5.0), (N - 2, 4.0), (N - 3, -1.0)):
                if c >= 1:
                    outlet[c - 1] = w
        elif order == 2:                                    # (2/h) d1 one-sided = (3u_N - 4u_{N-1} + u_{N-2})/h^2
            for c, w in ((N, 3.0), (N - 1, -4.0), (N - 2, 1.0)):
                if c >= 1:
                    outlet[c - 1] = w
        else:                                               # (2/h) d1 one-sided 4th order (25,-48,36,-16,3)/(12h)
            s, w = Stencils4(N).stencil(N, 1)
            for kk in range(w.size):
                outlet[s + kk - 1] = 2.0 * w[kk]
        D11[N - 1] = outlet
        D11 /= h * h
        I = np.eye(N)
        A = np.zeros((nm, 2 * N, 2 * N))
        eqrow = np.ones(N); eqrow[N - 1] = 1.0 if variant == "i0" else 0.0
        Eq = np.diag(eqrow)
        if variant == "i0":
            base11 = -D11
        else:
            base11 = -D11.copy(); base11[N - 1] = D11[N - 1]   # outlet row: +(2/h) d1 u
        if kind == "lin0":
            A[:, :N, :N] = base11[None] - (a2[:, None, None] * Eq[None])
            A[:, N:, N:] = base11[None] - (a3[:, None, None] * Eq[None])
            A[:, :N, N:] = -(b[:, None, None] * Eq[None])
            A[:, N:, :N] = -(b[:, None, None] * Eq[None])
        elif kind == "lap":
            A[:, :N, :N] = base11[None] - ((a2 + a3)[:, None, None] * Eq[None])
            A[:, N:, N:] = base11[None] - ((a2 + a3)[:, None, None] * Eq[None])
        else:
            raise ValueError(kind)
        t0 = time.time()
        if variant == "i0":
            self.Minv = np.linalg.pinv(A, rcond=1e-12)
        else:
            self.Minv = np.linalg.inv(A)
        self.t_setup = time.time() - t0
        self.nm = nm

    def apply(self, r, rowscale):
        N = self.N
        x = (r.reshape(2, N, N, N) / rowscale)
        Xh = np.fft.rfftn(x, axes=(2, 3))                  # (2, N, N, N//2+1)
        V = Xh.transpose(2, 3, 0, 1).reshape(self.nm, 2 * N)
        Z = np.matmul(self.Minv, V.real[..., None])[..., 0] + 1j * np.matmul(self.Minv, V.imag[..., None])[..., 0]
        Zh = Z.reshape(N, N // 2 + 1, 2, N).transpose(2, 3, 0, 1)
        z = np.fft.irfftn(Zh, s=(N, N), axes=(2, 3))
        return z.ravel()


def gmres_right(Aop, b, Pinv, tol=1e-13, restart=300, maxiter=6000):
    """Right-preconditioned restarted GMRES; stops on the TRUE relative residual ||b - A x|| / ||b||."""
    nb = np.linalg.norm(b)
    x = np.zeros_like(b)
    if nb == 0:
        return x, 0.0, 0
    its = 0
    r = b.copy()
    beta = np.linalg.norm(r)
    hist = []
    while its < maxiter:
        V = np.zeros((restart + 1, b.size)); Hm = np.zeros((restart + 1, restart))
        V[0] = r / beta
        g = np.zeros(restart + 1); g[0] = beta
        cs = np.zeros(restart); sn = np.zeros(restart)
        k_used = 0
        for k in range(restart):
            w = Aop(Pinv(V[k]))
            for _ in range(2):                              # classical Gram-Schmidt, twice (CGS2)
                t = V[:k + 1] @ w
                Hm[:k + 1, k] += t
                w = w - V[:k + 1].T @ t
            Hm[k + 1, k] = np.linalg.norm(w)
            if Hm[k + 1, k] > 0:
                V[k + 1] = w / Hm[k + 1, k]
            for i in range(k):
                t = cs[i] * Hm[i, k] + sn[i] * Hm[i + 1, k]
                Hm[i + 1, k] = -sn[i] * Hm[i, k] + cs[i] * Hm[i + 1, k]; Hm[i, k] = t
            den = np.hypot(Hm[k, k], Hm[k + 1, k])
            cs[k] = Hm[k, k] / den; sn[k] = Hm[k + 1, k] / den
            Hm[k, k] = den; Hm[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k]
            its += 1; k_used = k + 1
            if abs(g[k + 1]) / nb < 0.1 * tol or Hm[k, k] == 0:
                break
        y = np.linalg.solve(np.triu(Hm[:k_used, :k_used]), g[:k_used])
        x = x + Pinv(V[:k_used].T @ y)         # right preconditioning: x = P^-1 (V y)
        r = b - Aop(x)
        beta = np.linalg.norm(r)
        hist.append(beta / nb)
        if beta / nb <= tol:
            break
        if len(hist) >= 3 and hist[-1] > 0.9 * hist[-3]:   # restart stagnation
            break
    return x, float(beta / nb), its


# ------------------------------------------------------------------------------------------------
# Newton with continuation
# ------------------------------------------------------------------------------------------------
def _maxrss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # Linux: KiB
    except Exception:                                                        # pragma: no cover
        return float("nan")


def linear_solve(J, rhs, ctx, prec, opts, log, lu_state=None):
    """Newton linear solve.  N <= --direct-max: sparse direct splu (factorization time, solve time, nnz(L+U)
    and peak RSS logged as a LINEAR line); with --reuse-lu K > 0 the last LU is reused as the right
    preconditioner of GMRES for up to K subsequent Newton steps (refactor when the frozen-LU GMRES misses
    --lin-tol or K is exhausted).  N > --direct-max: GMRES + per-mode preconditioner (unchanged)."""
    N = ctx.N
    t0 = time.time()
    if lu_state is None:
        lu_state = {}
    if N <= opts["direct_max"]:
        K = opts.get("reuse_lu", 0)
        if K > 0 and lu_state.get("lu") is not None and lu_state.get("age", 0) < K:
            lu = lu_state["lu"]
            lu_state["age"] = lu_state.get("age", 0) + 1
            y, rel, its = gmres_right(lambda v: J @ v, rhs, lu.solve, tol=opts["lin_tol"], restart=opts["restart"],
                                      maxiter=opts["restart"])
            tl = time.time() - t0
            log("    LINEAR frozen-LU GMRES (LU age %d/%d) its=%d rel=%.1e t=%.2fs%s"
                % (lu_state["age"], K, its, rel, tl, "" if rel <= opts["lin_tol"] else " -> missed lin-tol, refactor"))
            if rel <= opts["lin_tol"] and np.all(np.isfinite(y)):
                lu_state["last_t_fact"] = 0.0
                return y, "gmres+frozenLU", rel, its, tl
            t0 = time.time()
        try:
            Jc = J.tocsc()
            tf0 = time.time()
            lu = spla.splu(Jc)
            t_fact = time.time() - tf0
            ts0 = time.time()
            dx = lu.solve(rhs)
            t_solve = time.time() - ts0
            nlu = int(lu.L.nnz + lu.U.nnz)
            res = np.linalg.norm(J @ dx - rhs) / np.linalg.norm(rhs)
            log("    LINEAR splu n=%d nnz(J)=%d t_fact=%.2fs t_solve=%.3fs nnz(L+U)=%d fill=%.1f mem(L+U)~%.0fMB "
                "maxrss=%.0fMB res=%.1e" % (J.shape[0], J.nnz, t_fact, t_solve, nlu, nlu / float(J.nnz),
                                            nlu * 12 / 2.0 ** 20, _maxrss_mb(), res))
            lu_state["lu"] = lu; lu_state["age"] = 0; lu_state["last_t_fact"] = t_fact
            if np.isfinite(res) and res <= 1e-8 and np.all(np.isfinite(dx)):
                return dx, "splu", res, 0, time.time() - t0
            log("    splu step residual %.1e (or non-finite) -> lsmr min-norm least squares" % res)
        except (RuntimeError, MemoryError) as exc:
            log("    splu failed (%s: %s) -> lsmr min-norm least squares" % (type(exc).__name__, exc))
            lu_state["lu"] = None
        out = spla.lsmr(J, rhs, atol=1e-15, btol=1e-15, conlim=1e16, maxiter=20 * rhs.size)
        dx = out[0]
        res = np.linalg.norm(J @ dx - rhs) / np.linalg.norm(rhs)
        return dx, "lsmr(istop=%d)" % out[1], res, int(out[2]), time.time() - t0
    Aop = lambda v: J @ v
    Pinv = lambda v: prec.apply(v, ctx.rowscale)
    y, rel, its = gmres_right(Aop, rhs, Pinv, tol=opts["lin_tol"], restart=opts["restart"])
    return y, "gmres+%s" % prec.kind, rel, its, time.time() - t0


def newton(uvec, ctx, opts, log, label=""):
    pat = opts["pattern"](ctx.N, ctx.order)
    prec = None
    lu_state = {}
    if ctx.N > opts["direct_max"]:
        key = (ctx.N, ctx.variant, opts["prec"], ctx.order)
        prec = opts["prec_cache"].get(key)
        if prec is None:
            prec = ModePrec(ctx.N, ctx.variant, opts["prec"], ctx.order)
            opts["prec_cache"][key] = prec
            log("  preconditioner %s/%s N=%d modes=%d setup %.1fs" % (opts["prec"], ctx.variant, ctx.N, prec.nm,
                                                                       prec.t_setup))
    E, r_F, r_out = system(uvec, ctx)
    m = merit(r_F, r_out)
    hist = [(r_F, r_out)]
    log("  NEWTON %s it=%2d r_F=%.3e r_out=%.3e" % (label, 0, r_F, r_out))
    status = "maxit"
    its = 0
    for it in range(1, opts["maxit"] + 1):
        if r_F <= opts["tol"] and r_out <= opts["tol"]:
            status = "converged"; break
        t0 = time.time()
        J = jacobian_colored(uvec, ctx, pat)
        tj = time.time() - t0
        dx, how, lres, lits, tl = linear_solve(J, -E, ctx, prec, opts, log, lu_state)
        lam = 1.0; accepted = False
        while lam >= 1.0 / 1024:
            un = uvec + lam * dx
            En, rFn, ron = system(un, ctx)
            mn = merit(rFn, ron)
            if np.isfinite(mn) and mn < (1 - 1e-4 * lam) * m:
                accepted = True; break
            lam *= 0.5
        its = it
        if not accepted:
            log("  NEWTON %s it=%2d line search failed (min lambda 1/1024): merit %.3e -> %.3e  [%s rel=%.1e lits=%d "
                "|dx|max=%.2e |dx|2=%.2e]" % (label, it, m, mn, how, lres, lits, np.abs(dx).max(), np.linalg.norm(dx)))
            status = "linesearch-fail"; break
        uvec, E, r_F, r_out, m = un, En, rFn, ron, mn
        hist.append((r_F, r_out))
        log("  NEWTON %s it=%2d r_F=%.3e r_out=%.3e lambda=%.4g |dx|max=%.2e lin=%s rel=%.1e lits=%d "
            "t_jac=%.1fs t_lin=%.1fs |dx|2=%.2e t_fact=%.2fs" % (label, it, r_F, r_out, lam, np.abs(dx).max(), how,
                                                                lres, lits, tj, tl, np.linalg.norm(dx),
                                                                lu_state.get("last_t_fact", 0.0)))
        if len(hist) >= 5:
            mm = [merit(*x) for x in hist[-5:]]
            if mm[-1] > 0.5 * mm[0] and not (r_F <= opts["tol"] and r_out <= opts["tol"]):
                status = "stagnation"; break
    if r_F <= opts["tol"] and r_out <= opts["tol"]:
        status = "converged"
    return uvec, {"status": status, "its": its, "hist": hist, "r_F": r_F, "r_out": r_out}


def levenberg_marquardt(uvec, ctx, opts, log, label=""):
    """Regularized fallback (i0, N <= 16): Levenberg-Marquardt on ||E||^2 with the colored FD Jacobian, dense
    normal equations (J^T J + mu tr(J^T J)/n I) dx = -J^T E.  Used only after Newton fails; it distinguishes an
    ill-conditioned but solvable system (LM reaches the tolerance) from a drift along a near-null valley (r_F
    decreases slowly while the labels move away)."""
    pat = opts["pattern"](ctx.N, ctx.order)
    E, r_F, r_out = system(uvec, ctx)
    mu = 1e-3
    hist = [(r_F, r_out)]
    status = "lm-maxit"
    for it in range(1, opts["lm"] + 1):
        t0 = time.time()
        J = jacobian_colored(uvec, ctx, pat).toarray()
        JtJ = J.T @ J; g = J.T @ E; sc = np.trace(JtJ) / JtJ.shape[0]
        nE = np.linalg.norm(E); ok = False
        while mu <= 1e8:
            dx = -np.linalg.solve(JtJ + mu * sc * np.eye(JtJ.shape[0]), g)
            En, rFn, ron = system(uvec + dx, ctx)
            if np.isfinite(rFn) and np.linalg.norm(En) < nE:
                ok = True; break
            mu *= 10.0
        if not ok:
            status = "lm-no-descent"; break
        uvec = uvec + dx; E, r_F, r_out = En, rFn, ron; mu = max(mu / 10.0, 1e-14)
        hist.append((r_F, r_out))
        log("  LM %s it=%2d r_F=%.3e r_out=%.3e mu=%.1e |dx|max=%.2e t=%.1fs" % (label, it, r_F, r_out, mu,
                                                                              np.abs(dx).max(), time.time() - t0))
        if r_F <= opts["tol"] and r_out <= opts["tol"]:
            status = "lm-converged"; break
    return uvec, {"status": status, "its": len(hist) - 1, "hist": hist, "r_F": r_F, "r_out": r_out}


def amplitude_path(eps, continuation):
    if not continuation:
        return [eps]
    return [e for e in EPS_LADDER if e < eps - 1e-12] + [eps]


def _nphi_light(field, e):
    """Reference resolution for an amplitude of the continuation path: the cases.NPHI entry if present, else that
    of the smallest tabulated ladder amplitude >= e (intermediate bisection amplitudes are warm starts only)."""
    try:
        return C.nphi_for(field, e)
    except KeyError:
        for a in EPS_LADDER:
            if a >= e - 1e-12:
                return C.nphi_for(field, a)
        return C.nphi_for(field, EPS_LADDER[-1])


def _start_vector(ctx, opts, case, log):
    N = ctx.N
    if opts["init"] == "oracle":                 # DIAGNOSTIC ONLY: start at the oracle labels (not a solver run)
        X2, X3 = M.affine(N)
        log("DIAGNOSTIC --init oracle: Newton started at the oracle labels (locates the discrete solution near the "
            "Darcy labels; not a solver result)")
        return np.concatenate([(case["psi_or"][0] - X2)[1:].ravel(), (case["psi_or"][1] - X3)[1:].ravel()])
    if opts["init"] == "inlet":
        return np.concatenate([np.broadcast_to(ctx.u0[0], (N, N, N)).ravel(),
                               np.broadcast_to(ctx.u0[1], (N, N, N)).ravel()])
    return np.zeros(2 * N ** 3)


def _log(msg):
    print(msg, flush=True)


def solve_case(field, eps, N, variant, opts, case=None, log=_log):
    """Damped Newton with amplitude continuation 0.25 -> 0.5 -> 1.0 (up to eps), warm starts, first start u = 0.
    If a stage fails (line-search failure, stagnation or maxit above STAGE_OK), the stage is retried from the last
    converged amplitude at the midpoint amplitude (bisection of the continuation step, at most opts['bisect']
    times in total); every stage is printed, so the path actually taken is part of the record.
    The discretization order is opts['order'] (2: i0/i1; 4: i1o4)."""
    t0 = time.time()
    order = opts["order"]
    cand = cand_name(variant, order)
    if case is None:
        case = C.load_case(field, eps, N, verbose=False)
    if not opts["continuation"] or opts["init"] == "oracle":
        ctx = Ctx(case, variant, order)
        uvec = _start_vector(ctx, opts, case, log)
        log("STAGE field=%s eps=%g N=%d cand=%s (no continuation, init=%s)" % (field, eps, N, cand, opts["init"]))
        uvec, info = newton(uvec, ctx, opts, log, label="%s:%g:%d:%s" % (field, eps, N, cand))
        info["t"] = time.time() - t0; info["path"] = "%g" % eps
        return case, ctx, uvec, info
    todo = amplitude_path(eps, True)
    e_conv = 0.0; u_conv = None; nbis = 0; taken = []
    ctx = None; uvec = None; info = None
    while todo:
        e = todo[0]
        cs = case if abs(e - eps) < 1e-12 else light_case(field, e, N, nphi=_nphi_light(field, e))
        ctx = Ctx(cs, variant, order)
        u_start = _start_vector(ctx, opts, case, log) if u_conv is None else u_conv
        log("STAGE field=%s eps=%g N=%d cand=%s (from eps=%g, %s)" % (field, e, N, cand, e_conv,
                                                                       "u=0" if u_conv is None else "warm start"))
        uvec, info = newton(u_start, ctx, opts, log, label="%s:%g:%d:%s" % (field, e, N, cand))
        ok = info["status"] == "converged" or merit(info["r_F"], info["r_out"]) <= STAGE_OK
        log("  STAGE_END field=%s eps=%g N=%d cand=%s status=%s its=%d r_F=%.3e r_out=%.3e -> %s"
            % (field, e, N, cand, info["status"], info["its"], info["r_F"], info["r_out"],
               "accepted" if ok else "FAILED"))
        taken.append("%g%s" % (e, "" if ok else "(fail)"))
        if ok:
            e_conv = e; u_conv = uvec; todo.pop(0)
        elif nbis < opts["bisect"]:
            nbis += 1
            todo.insert(0, 0.5 * (e_conv + e))
            log("  CONTINUATION bisection %d/%d: retry from eps=%g at eps=%g" % (nbis, opts["bisect"], e_conv,
                                                                               todo[0]))
        else:
            log("  CONTINUATION gave up after %d bisections; reporting the failed state at eps=%g" % (nbis, e))
            if abs(e - eps) > 1e-12:     # report at the target: one more attempt from the last converged state
                ctx = Ctx(case, variant, order)
                log("STAGE field=%s eps=%g N=%d cand=%s (final attempt from eps=%g)" % (field, eps, N, cand,
                                                                                       e_conv))
                uvec, info = newton(u_conv if u_conv is not None else _start_vector(ctx, opts, case, log), ctx,
                                    opts, log, label="%s:%g:%d:%s" % (field, eps, N, cand))
                taken.append("%g(final)" % eps)
            break
    if variant == "i0" and info["status"] != "converged" and opts["lm"] > 0 and N <= 16 and ctx is not None:
        log("FALLBACK field=%s eps=%g N=%d cand=%s Newton status=%s -> Levenberg-Marquardt (%d its max) from the "
            "final Newton state" % (field, eps, N, cand, info["status"], opts["lm"]))
        uvec, lminfo = levenberg_marquardt(uvec, ctx, opts, log, label="%s:%g:%d:%s" % (field, eps, N, cand))
        info = {"status": "newton:%s/%s" % (info["status"], lminfo["status"]), "its": info["its"] + lminfo["its"],
                "hist": info["hist"] + lminfo["hist"][1:], "r_F": lminfo["r_F"], "r_out": lminfo["r_out"]}
        taken.append("LM")
    info["t"] = time.time() - t0
    info["path"] = "->".join(taken)
    log("PATH field=%s eps=%g N=%d cand=%s continuation path: %s" % (field, eps, N, cand, info["path"]))
    return case, ctx, uvec, info


def labels_from(uvec, ctx):
    U1, U2 = full_planes(uvec, ctx)
    X2, X3 = M.affine(ctx.N)
    return X2 + U1, X3 + U2


def case_lines(field, eps, N, variant, order, psi1, psi2, case, info):
    """The CASE lines of one solution: order 2 -> cand=<variant> + ceiling cand=oracle_fd; order 4 -> cand=i1o4
    (4th-order reconstruction) + ceiling cand=oracle_fd4, then cand=i1o4_fd2 (2nd-order reconstruction of the same
    labels) + ceiling cand=oracle_fd.  Returns {cand name: metrics dict}."""
    vD = case["vD"]; psi_or = case["psi_or"]
    cand = cand_name(variant, order)
    out = {}
    m = M.fd_metrics(psi1, psi2, vD, psi_or, order=order)
    M.print_case(field, eps, N, cand, m, r_F=info["r_F"], its=info["its"], t=info["t"])
    mo = M.fd_metrics(psi_or[0], psi_or[1], vD, psi_or, order=order)
    M.print_case(field, eps, N, "oracle_fd" if order == 2 else "oracle_fd4", mo)
    out[cand] = m; out["oracle_fd" if order == 2 else "oracle_fd4"] = mo
    if order == 4:
        m2 = M.fd_metrics(psi1, psi2, vD, psi_or, order=2)
        M.print_case(field, eps, N, cand + "_fd2", m2, r_F=info["r_F"], its=info["its"], t=info["t"])
        mo2 = M.fd_metrics(psi_or[0], psi_or[1], vD, psi_or, order=2)
        M.print_case(field, eps, N, "oracle_fd", mo2)
        out[cand + "_fd2"] = m2; out["oracle_fd"] = mo2
    return out


def report(field, eps, N, variant, case, ctx, uvec, info):
    psi1, psi2 = labels_from(uvec, ctx)
    vD = case["vD"]; psi_or = case["psi_or"]
    order = ctx.order; cand = ctx.cand
    ms = case_lines(field, eps, N, variant, order, psi1, psi2, case, info)
    m = ms[cand]
    # outlet rows of i1 evaluated for any labels, and inlet oblique-condition defects (P4); gradients of the
    # candidate's own order
    ex = {}
    for name, (p1, p2) in (("cand", (psi1, psi2)), ("oracle", psi_or)):
        g1, g2 = M.label_gradients(p1, p2, order)
        c = M.cross(g1, g2)
        vp = ctx.vperp
        out = float(np.sqrt(np.mean((c[1][N] - vp[0]) ** 2 + (c[2][N] - vp[1]) ** 2))) / ctx.v_rms
        in0 = float(np.sqrt(np.mean((c[1][0] - vp[0]) ** 2 + (c[2][0] - vp[1]) ** 2))) / ctx.v_rms
        in1 = float(np.sqrt(np.mean((c[1][1] - vp[0]) ** 2 + (c[2][1] - vp[1]) ** 2))) / ctx.v_rms
        in1v = float(np.sqrt(np.mean((c[1][1] - vD[1][1]) ** 2 + (c[2][1] - vD[2][1]) ** 2))) / ctx.v_rms
        ex[name] = (out, in0, in1, in1v)
    print("EXTRA field=%s eps=%g N=%d cand=%s status=%s r_F=%.3e r_out=%.3e | outlet c_perp-vperp_in: cand=%.3e "
          "oracle_fd=%.3e | inlet oblique defect (c x e1 - vperp_in x e1)/v_rms: j=0(one-sided) cand=%.3e oracle_fd=%.3e; "
          "j=1 cand=%.3e oracle_fd=%.3e; j=1 vs vD(j=1) cand=%.3e oracle_fd=%.3e | min|c|/v_rms=%.4f min|vD|/v_rms=%.4f "
          "e_i1=%.3e e_i2=%.3e | t=%.1fs | a_psi1=%.3e a_psi2=%.3e (FD order %d)"
          % (field, eps, N, cand, info["status"], info["r_F"], info["r_out"], ex["cand"][0], ex["oracle"][0],
             ex["cand"][1], ex["oracle"][1], ex["cand"][2], ex["oracle"][2], ex["cand"][3], ex["oracle"][3],
             m["min_c"] / m["v_rms"], m["vD_min"] / m["v_rms"], m["e_i1"], m["e_i2"], info["t"], m["a_psi1"],
             m["a_psi2"], order), flush=True)
    print("HISTORY field=%s eps=%g N=%d cand=%s r_F: %s" % (field, eps, N, cand,
                                                             " ".join("%.2e" % x[0] for x in info["hist"])), flush=True)
    if variant == "i1":
        print("HISTORY field=%s eps=%g N=%d cand=%s r_out: %s" % (field, eps, N, cand,
                                                                   " ".join("%.2e" % x[1] for x in info["hist"])),
              flush=True)
    return ms


# ------------------------------------------------------------------------------------------------
# saved solutions (--save DIR / --remetric FILE...)
# ------------------------------------------------------------------------------------------------
def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _jsonable(v)
        elif isinstance(v, (tuple, list)):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def save_solution(outdir, field, eps, N, variant, ctx, uvec, info, ms, opts):
    """DIR/<field>_<eps>_<N>_<cand>.npz: u1, u2 on planes 0..N (inlet plane included), status, residual history,
    the metrics dicts of the CASE lines (JSON), and the solver options."""
    import json
    os.makedirs(outdir, exist_ok=True)
    U1, U2 = full_planes(uvec, ctx)
    path = os.path.join(outdir, "%s_%g_%d_%s.npz" % (field, eps, N, ctx.cand))
    np.savez(path, u1=U1, u2=U2, field=field, eps=float(eps), N=int(N), variant=variant, order=int(ctx.order),
             cand=ctx.cand, status=str(info["status"]), its=int(info["its"]), r_F=float(info["r_F"]),
             r_out=float(info["r_out"]), t=float(info["t"]), path=str(info.get("path", "")),
             hist=np.array(info["hist"], dtype=float), metrics_json=json.dumps(_jsonable(ms)),
             opts_json=json.dumps({k: v for k, v in opts.items() if k not in ("prec_cache", "pattern")}),
             numpy=np.__version__, python=sys.version.split()[0])
    print("SAVED %s status=%s (%.0f kB)" % (os.path.relpath(path, _HERE), info["status"],
                                             os.path.getsize(path) / 1024.0), flush=True)
    return path


def remetric(path):
    """Recompute the CASE lines of a saved solution with the current metrics (no re-solve)."""
    d = np.load(path, allow_pickle=False)
    field = str(d["field"]); eps = float(d["eps"]); N = int(d["N"]); variant = str(d["variant"])
    order = int(d["order"])
    info = {"r_F": float(d["r_F"]), "its": int(d["its"]), "t": float(d["t"])}
    case = C.load_case(field, eps, N, verbose=False)
    X2, X3 = M.affine(N)
    psi1 = X2 + d["u1"]; psi2 = X3 + d["u2"]
    print("REMETRIC file=%s field=%s eps=%g N=%d cand=%s status=%s path=%s r_out=%.3e"
          % (os.path.relpath(path, _HERE), field, eps, N, str(d["cand"]), str(d["status"]), str(d["path"]),
             float(d["r_out"])), flush=True)
    return case_lines(field, eps, N, variant, order, psi1, psi2, case, info)


# ------------------------------------------------------------------------------------------------
# modes
# ------------------------------------------------------------------------------------------------
def spectrum(field, eps, N, variant, ctx, uvec, outdir, state="converged"):
    cand = ctx.cand
    t0 = time.time()
    J = jacobian_dense(uvec, ctx)
    tj = time.time() - t0
    s = np.linalg.svd(J, compute_uv=False)
    s = np.sort(s)
    rel = s / s[-1]
    t1 = time.time() - t0
    cnt = {thr: int((rel < thr).sum()) for thr in (1e-2, 1e-3, 1e-4, 1e-6, 1e-10)}
    low = rel[rel < 1e-2]
    gap = (float("nan"), -1)
    if low.size >= 1:
        nxt = rel[: low.size + 1]                           # include the first value >= 1e-2
        ratios = nxt[1:] / np.maximum(nxt[:-1], 1e-300)
        k = int(np.argmax(ratios))
        gap = (float(ratios[k]), k + 1)
    print("SPECTRUM field=%s eps=%g N=%d cand=%s state=%s rows: F_i=-q(L_i-S_i)%s | size=%d smax=%.3e t_jac=%.1fs "
          "t_svd=%.1fs" % (field, eps, N, cand, state, "; outlet (2q/h)(vperp_in-c_perp)" if variant == "i1" else "",
                           J.shape[0], s[-1], tj, t1 - tj), flush=True)
    print("SPECTRUM field=%s eps=%g N=%d cand=%s state=%s smallest12 rel sv: %s"
          % (field, eps, N, cand, state, " ".join("%.2e" % x for x in rel[:12])), flush=True)
    print("SPECTRUM field=%s eps=%g N=%d cand=%s state=%s counts <1e-2:%d <1e-3:%d <1e-4:%d <1e-6:%d <1e-10:%d | largest "
          "consecutive gap ratio among values <1e-2 (incl. the next value): %.3e after the %d smallest (%.2e -> %.2e) "
          "| 2N=%d"
          % (field, eps, N, cand, state, cnt[1e-2], cnt[1e-3], cnt[1e-4], cnt[1e-6], cnt[1e-10], gap[0], gap[1],
             rel[gap[1] - 1] if gap[1] > 0 else float("nan"), rel[gap[1]] if gap[1] > 0 else float("nan"), 2 * N),
          flush=True)
    os.makedirs(outdir, exist_ok=True)
    suffix = "" if state == "converged" else "_" + state
    path = os.path.join(outdir, "spectrum_cand_i_%s_%s_%g_%d%s.txt" % (cand, field, eps, N, suffix))
    hdr = ("SF-29 %s candidate (i) %s field=%s eps=%g N=%d: sorted relative singular values of the dense central-FD "
           "Jacobian at the %s state (rows F_i=-q(L_i-S_i)%s; smax=%.6e)"
           % ("C-i4" if ctx.order == 4 else "N2", cand, field, eps, N, state,
              "; outlet rows (2q/h)(vperp_in-c_perp)" if variant == "i1" else "", s[-1]))
    np.savetxt(path, rel, fmt="%.6e", header=hdr)
    print("SPECTRUM saved %s" % os.path.relpath(path, _HERE), flush=True)
    # cross-check of the colored sparse Jacobian against the dense columns
    Js = jacobian_colored(uvec, ctx, make_pattern(N, ctx.order)).toarray()
    print("SPECTRUM cross-check colored-sparse vs dense FD Jacobian: max|diff|/max|J| = %.2e"
          % (np.abs(Js - J).max() / np.abs(J).max()), flush=True)
    return rel


def jactest(field, eps, N, variant, opts):
    order = opts["order"]
    case = C.load_case(field, eps, N, verbose=False)
    ctx = Ctx(case, variant, order)
    cand = ctx.cand
    X2, X3 = M.affine(N)
    uor = np.concatenate([(case["psi_or"][0] - X2)[1:].ravel(), (case["psi_or"][1] - X3)[1:].ravel()])
    rng = np.random.default_rng(29)
    u = uor + 1e-2 * float(np.sqrt(np.mean(uor ** 2))) * rng.standard_normal(uor.size)
    E, r_F, r_out = system(u, ctx)
    print("JACTEST field=%s eps=%g N=%d cand=%s state = oracle + 1e-2 rms noise: r_F=%.3e r_out=%.3e"
          % (field, eps, N, cand, r_F, r_out), flush=True)
    t0 = time.time()
    pat = make_pattern(N, order)
    J = jacobian_colored(u, ctx, pat)
    print("JACTEST colored Jacobian: nnz=%d colors=%d (p=%d) delta=%.2e t=%.2fs"
          % (J.nnz, pat.ncolor, pat.p, fd_delta(u), time.time() - t0), flush=True)
    v = rng.standard_normal(u.size)
    v *= float(np.sqrt(np.mean(u ** 2))) / float(np.sqrt(np.mean(v ** 2)))
    Jv = J @ v
    worst = 0.0
    for t in (1e-3, 1e-4, 1e-5):
        fd = (system(u + t * v, ctx)[0] - system(u - t * v, ctx)[0]) / (2 * t)
        err = np.linalg.norm(Jv - fd) / np.linalg.norm(fd)
        worst = max(worst, err)
        print("JACTEST field=%s eps=%g N=%d cand=%s step=%.0e  ||Jv - FD||/||FD|| = %.3e" % (field, eps, N, cand,
                                                                                         t, err), flush=True)
    # directions restricted to the outlet planes (one-sided stencils / outlet rows) and, for order 4, to the
    # inlet-side planes 1..6 (skewed stencils of plane 1)
    nb = 4 if order == 2 else 6
    dirs = [("outlet-planes", slice(N - nb, N))]
    if order == 4:
        dirs.append(("inlet-planes", slice(0, nb)))
    for dname, sl in dirs:
        vo = np.zeros_like(v).reshape(2, N, N, N); vo[:, sl] = v.reshape(2, N, N, N)[:, sl]
        vo = vo.ravel(); Jvo = J @ vo
        for t in (1e-3, 1e-4, 1e-5):
            fd = (system(u + t * vo, ctx)[0] - system(u - t * vo, ctx)[0]) / (2 * t)
            err = np.linalg.norm(Jvo - fd) / np.linalg.norm(fd)
            worst = max(worst, err)
            print("JACTEST field=%s eps=%g N=%d cand=%s step=%.0e %s direction ||Jv - FD||/||FD|| = %.3e"
                  % (field, eps, N, cand, t, dname, err), flush=True)
    print("JACTEST field=%s eps=%g N=%d cand=%s worst relative error %.3e -> %s (threshold 1e-6)"
          % (field, eps, N, cand, worst, "PASS" if worst <= 1e-6 else "FAIL"), flush=True)
    return worst


def k1check(N, outdir, order=2):
    """k = 1 (uniform flow, affine labels, u = 0) control of THIS discretization: dense FD Jacobian SVD (exact nulls
    of i0 = the 2N shear modes of the UNDERSTAND record section 8; none for i1) and exactness of the per-mode
    preconditioner (P^-1 J x = x for i1; i0 is singular -> pseudo-inverse, P^-1 J x = projection of x).
    order 4: variant i1 only (i1o4)."""
    x = np.arange(N) / float(N)
    z = np.zeros((N + 1, N, N))
    case = {"k": np.ones((N + 1, N, N)), "grad_lnk": (z, z, z),
            "psi0": (np.broadcast_to(x[:, None], (N, N)).copy(), np.broadcast_to(x[None, :], (N, N)).copy()),
            "vperp_in": (np.zeros((N, N)), np.zeros((N, N))), "vD": (np.ones((N + 1, N, N)), z, z)}
    rng = np.random.default_rng(1)
    for variant in (("i0", "i1") if order == 2 else ("i1",)):
        ctx = Ctx(case, variant, order)
        cand = ctx.cand
        u0 = np.zeros(2 * N ** 3)
        J = jacobian_colored(u0, ctx, make_pattern(N, order)).toarray()
        Jd = jacobian_dense(u0, ctx)
        s = np.sort(np.linalg.svd(Jd, compute_uv=False)); rel = s / s[-1]
        X = rng.standard_normal(2 * N ** 3)
        out = []
        for kind in ("lin0", "lap"):
            P = ModePrec(N, variant, kind, order)
            out.append(np.linalg.norm(P.apply(J @ X, ctx.rowscale) - X) / np.linalg.norm(X))
        nz = int((rel < 1e-10).sum())
        print("K1CHECK N=%d cand=%s r_F(u=0)=%.1e | dense J: exact nulls (<1e-10)=%d (2N=%d) next rel sv=%.2e "
              "<1e-3:%d <1e-2:%d min=%.2e | colored vs dense max|diff|=%.1e | |P^-1 J x - x|/|x|: lin0=%.1e lap=%.1e"
              % (N, cand, system(u0, ctx)[1], nz, 2 * N, rel[nz], int((rel < 1e-3).sum()), int((rel < 1e-2).sum()),
                 rel[0], np.abs(J - Jd).max(), out[0], out[1]), flush=True)
        print("K1CHECK N=%d cand=%s smallest %d rel sv beyond the exact nulls: %s"
              % (N, cand, 4 * N, " ".join("%.2e" % v for v in rel[nz:nz + 4 * N])), flush=True)
        path = os.path.join(outdir, "spectrum_cand_i_%s_uniform_k1_%d.txt" % (cand, N))
        np.savetxt(path, rel, fmt="%.6e", header="SF-29 %s candidate (i) %s k = 1 control (u = 0, affine labels) N=%d: "
                   "sorted relative singular values of the dense central-FD Jacobian (smax=%.6e)"
                   % ("C-i4" if order == 4 else "N2", cand, N, s[-1]))
        print("K1CHECK saved %s" % os.path.relpath(path, _HERE), flush=True)


def consistency(field, eps, grids, order=2):
    if order == 4:
        return consistency4(field, eps, grids)
    rows = []
    for N in grids:
        t0 = time.time()
        case = C.load_case(field, eps, N, verbose=False)
        X2, X3 = M.affine(N)
        uor = np.concatenate([(case["psi_or"][0] - X2)[1:].ravel(), (case["psi_or"][1] - X3)[1:].ravel()])
        r = {}
        for variant in ("i1", "i0"):
            ctx = Ctx(case, variant)
            E, r_F, r_out, p = system(uor, ctx, want_parts=True)
            r[variant] = (r_F, r_out)
            if variant == "i1":
                # interior equation rows only (planes 1..N-1) and the outlet-plane equation rows (i0 rows at j = N)
                Fo = np.sqrt((np.mean(p["F1"][-1] ** 2) + np.mean(p["F2"][-1] ** 2)) / 2.0) / ctx.q_rms
                r["i0_outlet_rows"] = float(Fo)
        rows.append((N, r))
        print("CONSISTENCY field=%s eps=%g N=%d at the oracle labels: i1 r_F(planes 1..N-1)=%.3e r_out(outlet c_perp "
              "rows)=%.3e | i0 r_F(planes 1..N incl. one-sided outlet rows)=%.3e (outlet-plane rows alone %.3e) | "
              "t=%.1fs" % (field, eps, N, r["i1"][0], r["i1"][1], r["i0"][0], r["i0_outlet_rows"], time.time() - t0),
              flush=True)
    Ns = [x[0] for x in rows]
    for name, getter in (("i1 r_F", lambda r: r["i1"][0]), ("i1 r_out", lambda r: r["i1"][1]),
                         ("i0 r_F", lambda r: r["i0"][0]), ("i0 outlet rows", lambda r: r["i0_outlet_rows"])):
        vals = [getter(x[1]) for x in rows]
        print("CONSISTENCY_ORDER field=%s eps=%g %s: %s | orders %s" % (
            field, eps, name, " ".join("%.3e" % v for v in vals),
            " ".join("%.2f" % o for o in M.orders(vals, Ns))), flush=True)


def consistency4(field, eps, grids):
    """i1o4 residual at the oracle labels: all equation rows (planes 1..N-1, r_F), the outlet rows (r_out), and
    separately the rows with skewed x1 stencils (planes 1 and N-1) and the centered rows (planes 2..N-2); same
    normalization as r_F (q_rms of the equation rows)."""
    rows = []
    for N in grids:
        t0 = time.time()
        case = C.load_case(field, eps, N, verbose=False)
        X2, X3 = M.affine(N)
        uor = np.concatenate([(case["psi_or"][0] - X2)[1:].ravel(), (case["psi_or"][1] - X3)[1:].ravel()])
        ctx = Ctx(case, "i1", 4)
        E, r_F, r_out, p = system(uor, ctx, want_parts=True)

        def part(sl):
            return float(np.sqrt((np.mean(p["F1"][sl] ** 2) + np.mean(p["F2"][sl] ** 2)) / 2.0)) / ctx.q_rms
        r = {"r_F": r_F, "r_out": r_out, "p1": part(slice(0, 1)), "pNm1": part(slice(N - 2, N - 1)),
             "int": part(slice(1, N - 2))}
        rows.append((N, r))
        print("CONSISTENCY4 field=%s eps=%g N=%d at the oracle labels (i1o4): r_F(planes 1..N-1)=%.3e r_out(outlet "
              "c_perp rows)=%.3e | centered rows (planes 2..N-2)=%.3e | skewed rows plane 1=%.3e plane N-1=%.3e | "
              "t=%.1fs" % (field, eps, N, r_F, r_out, r["int"], r["p1"], r["pNm1"], time.time() - t0), flush=True)
    Ns = [x[0] for x in rows]
    for name, key in (("i1o4 r_F", "r_F"), ("i1o4 r_out", "r_out"), ("i1o4 centered rows", "int"),
                      ("i1o4 plane-1 rows", "p1"), ("i1o4 plane-(N-1) rows", "pNm1")):
        vals = [x[1][key] for x in rows]
        print("CONSISTENCY_ORDER field=%s eps=%g %s: %s | orders %s" % (
            field, eps, name, " ".join("%.3e" % v for v in vals),
            " ".join("%.2f" % o for o in M.orders(vals, Ns))), flush=True)


def selfcheck(order):
    """Stencil exactness on polynomials (order 4: every interior, skewed and one-sided stencil of Stencils4, the
    in-plane centered stencils, and the 4th-order metrics stencils) and observed order on a smooth function."""
    ok = True

    def chk(name, val, thr):
        nonlocal ok
        good = bool(val <= thr)
        ok = ok and good
        print("SELFCHECK %-72s %.2e <= %.0e  %s" % (name, val, thr, "PASS" if good else "FAIL"), flush=True)
    if order != 4:
        print("SELFCHECK only --order 4 has a stencil self-check", flush=True)
        return True
    N = 16; h = 1.0 / N
    st = Stencils4(N)
    x = np.arange(N + 1) * h
    for p in sorted(set([1, 2, N - 2, N - 1, N])):
        for deriv in (1, 2):
            s, w = st.stencil(p, deriv)
            maxdeg = w.size - 1
            print("SELFCHECK stencil plane %2d d%s: planes %d..%d weights*12 = %s" % (
                p, "1" if deriv == 1 else "11", s, s + w.size - 1, " ".join("%+.6g" % (12 * x_) for x_ in w)),
                flush=True)
            err = 0.0
            for deg in range(maxdeg + 1):
                f = x ** deg
                exact = 0.0 if deg < deriv else (deg * x[p] ** (deg - 1) if deriv == 1 else
                                                  deg * (deg - 1) * x[p] ** (deg - 2))
                val = sum(w[k] * f[s + k] for k in range(w.size)) / h ** deriv
                err = max(err, abs(val - exact) / max(1.0, abs(exact)))
            chk("plane %d d%s exact on degree 0..%d" % (p, "1" if deriv == 1 else "11", maxdeg), err, 1e-9)
            f = x ** (maxdeg + 1)
            exact = ((maxdeg + 1) * x[p] ** maxdeg if deriv == 1 else (maxdeg + 1) * maxdeg * x[p] ** (maxdeg - 1))
            val = sum(w[k] * f[s + k] for k in range(w.size)) / h ** deriv
            print("SELFCHECK   (informative: degree %d error %.2e; nonzero unless the stencil is symmetric)"
                  % (maxdeg + 1, abs(val - exact)), flush=True)
    # operator-level checks: Stencils4.apply and derivs4 on polynomials in x1 and trigonometric in x2, x3
    xx = np.arange(N) * h
    X1 = x[:, None, None]; X2 = xx[None, :, None]; X3 = xx[None, None, :]
    for deg in (0, 1, 2, 3, 4):
        U = (X1 ** deg) * np.ones((1, N, N))
        g, H = derivs4(U, h, st)
        e1 = deg * X1[1:] ** (deg - 1) if deg >= 1 else 0.0 * X1[1:]
        e11 = deg * (deg - 1) * X1[1:] ** (deg - 2) if deg >= 2 else 0.0 * X1[1:]
        chk("derivs4: d1, d11 of x1^%d on planes 1..N (max abs err)" % deg,
            max(np.abs(g[0] - e1).max(), np.abs(H[(0, 0)] - e11).max()), 1e-8)
    # in-plane: 4th-order periodic stencils exact on the trigonometric modes only to O(h^4): observed order
    errs = []
    for NN in (16, 32, 64):
        hh = 1.0 / NN
        xs = np.arange(NN) * hh
        xs1 = np.arange(NN + 1) * hh
        U = np.sin(2 * np.pi * xs1[:, None, None]) * np.cos(2 * np.pi * xs[None, :, None]) * \
            np.sin(2 * np.pi * xs[None, None, :])
        g, H = derivs4(U, hh, Stencils4(NN))
        A = 2 * np.pi
        Xa = xs1[1:, None, None]; Ya = xs[None, :, None]; Za = xs[None, None, :]
        ex = {"d1": A * np.cos(A * Xa) * np.cos(A * Ya) * np.sin(A * Za),
              "d2": -A * np.sin(A * Xa) * np.sin(A * Ya) * np.sin(A * Za),
              "d23": -A * A * np.sin(A * Xa) * np.sin(A * Ya) * np.cos(A * Za),
              "d12": -A * A * np.cos(A * Xa) * np.sin(A * Ya) * np.sin(A * Za),
              "d11": -A * A * np.sin(A * Xa) * np.cos(A * Ya) * np.sin(A * Za),
              "d33": -A * A * np.sin(A * Xa) * np.cos(A * Ya) * np.sin(A * Za)}
        got = {"d1": g[0], "d2": g[1], "d23": H[(1, 2)], "d12": H[(0, 1)], "d11": H[(0, 0)], "d33": H[(2, 2)]}
        errs.append({k: float(np.abs(got[k] - ex[k]).max()) for k in ex})
    for k in errs[0]:
        e = [r[k] for r in errs]
        o = M.orders(e, [16, 32, 64])
        print("SELFCHECK derivs4 %s on sin(2pi x1)cos(2pi x2)sin(2pi x3): max err %s | orders %s"
              % (k, " ".join("%.2e" % v for v in e), " ".join("%.2f" % v for v in o)), flush=True)
        chk("derivs4 %s observed order (last pair) >= 3.8" % k, max(0.0, 3.8 - o[-1]), 0.0)
    # metrics.d1_fd4 (planes 0..N incl. the one-sided inlet stencil)
    bnd, _ = M.x1_stencils4(N, 1)
    s, w = bnd[0]
    print("SELFCHECK metrics stencil plane  0 d1: planes %d..%d weights*12 = %s" % (
        s, s + w.size - 1, " ".join("%+.6g" % (12 * x_) for x_ in w)), flush=True)
    for deg in range(5):
        u = x ** deg
        chk("metrics.d1_fd4 exact on x1^%d (all planes 0..N)" % deg,
            np.abs(M.d1_fd4(u, h) - (deg * x ** (deg - 1) if deg else 0.0)).max(), 1e-9)
    print("SELFCHECK %s" % ("ALL PASS" if ok else "FAILURES"), flush=True)
    return ok


# ------------------------------------------------------------------------------------------------
def main(argv):
    opts = {"maxit": 40, "tol": 1e-13, "continuation": True, "out": os.path.normpath(os.path.join(_HERE, "..", "raw")),
            "prec": "lin0", "lin_tol": 1e-13, "init": "zero", "prec_cache": {}, "bisect": 4, "lm": 30,
            "direct_max": 32, "restart": 300, "order": 2, "reuse_lu": 0, "save": None}
    pat_cache = {}

    def pattern(N, order=2):
        if (N, order) not in pat_cache:
            pat_cache[(N, order)] = make_pattern(N, order)
        return pat_cache[(N, order)]
    opts["pattern"] = pattern
    mode = "solve"; spec_M = None; grids = [16, 32, 48]
    specs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--maxit":
            opts["maxit"] = int(argv[i + 1]); i += 2
        elif a == "--tol":
            opts["tol"] = float(argv[i + 1]); i += 2
        elif a == "--lin-tol":
            opts["lin_tol"] = float(argv[i + 1]); i += 2
        elif a == "--prec":
            opts["prec"] = argv[i + 1]; i += 2
        elif a == "--restart":
            opts["restart"] = int(argv[i + 1]); i += 2
        elif a == "--direct-max":
            opts["direct_max"] = int(argv[i + 1]); i += 2
        elif a == "--reuse-lu":
            opts["reuse_lu"] = int(argv[i + 1]); i += 2
        elif a == "--order":
            opts["order"] = int(argv[i + 1]); i += 2
            if opts["order"] not in (2, 4):
                raise SystemExit("--order must be 2 or 4")
        elif a == "--save":
            opts["save"] = os.path.abspath(argv[i + 1]); i += 2
        elif a == "--lm":
            opts["lm"] = int(argv[i + 1]); i += 2
        elif a == "--bisect":
            opts["bisect"] = int(argv[i + 1]); i += 2
        elif a == "--init":
            opts["init"] = argv[i + 1]; i += 2
        elif a == "--out":
            opts["out"] = os.path.abspath(argv[i + 1]); i += 2
        elif a == "--continuation":
            opts["continuation"] = True; i += 1
        elif a == "--no-continuation":
            opts["continuation"] = False; i += 1
        elif a == "--grids":
            grids = [int(x) for x in argv[i + 1].split(",")]; i += 2
        elif a == "--spectrum":
            mode = "spectrum"; spec_M = int(argv[i + 1]); i += 2
        elif a == "--jactest":
            mode = "jactest"; i += 1
        elif a == "--consistency":
            mode = "consistency"; i += 1
        elif a == "--k1check":
            mode = "k1check"; spec_M = int(argv[i + 1]); i += 2
        elif a == "--selfcheck":
            mode = "selfcheck"; i += 1
        elif a == "--remetric":
            mode = "remetric"; i += 1
        elif a.startswith("--"):
            raise SystemExit("unknown option %s" % a)
        else:
            specs.append(a); i += 1
    print("candidate_i.py mode=%s opts=%s numpy=%s scipy=%s python=%s" % (
        mode, {k: v for k, v in opts.items() if k not in ("prec_cache", "pattern")}, np.__version__,
        __import__("scipy").__version__, sys.version.split()[0]), flush=True)
    if mode == "selfcheck":
        return 0 if selfcheck(opts["order"]) else 1
    if mode == "remetric":
        for s in specs:
            remetric(s)
        return 0
    if mode == "k1check":
        k1check(spec_M, opts["out"], opts["order"])
        return 0
    if mode == "consistency":
        for s in specs:
            field, eps, _ = C.parse_spec(s if s.count(":") >= 1 else s)[:3]
            consistency(field, eps, grids, opts["order"])
        return 0
    worst = 0.0
    base_order = opts["order"]
    for s in specs:
        field, eps, N, variant, order = parse_cand_spec(s, base_order)
        opts["order"] = order
        if mode == "jactest":
            worst = max(worst, jactest(field, eps, N, variant, opts))
            continue
        if mode == "spectrum" and N != spec_M:
            print("NOTE --spectrum %d: case N=%d overridden to %d" % (spec_M, N, spec_M)); N = spec_M
        case, ctx, uvec, info = solve_case(field, eps, N, variant, opts)
        ms = report(field, eps, N, variant, case, ctx, uvec, info)
        if opts["save"]:
            save_solution(opts["save"], field, eps, N, variant, ctx, uvec, info, ms, opts)
        if mode == "spectrum":
            if info["status"] == "converged":
                spectrum(field, eps, N, variant, ctx, uvec, opts["out"], state="converged")
            else:
                print("SPECTRUM NOTE field=%s eps=%g N=%d cand=%s: no converged state (status %s); the spectrum is "
                      "computed at the final iterate AND at the oracle labels (both labelled)"
                      % (field, eps, N, ctx.cand, info["status"]), flush=True)
                spectrum(field, eps, N, variant, ctx, uvec, opts["out"], state="final_iterate")
                X2, X3 = M.affine(N)
                uor = np.concatenate([(case["psi_or"][0] - X2)[1:].ravel(), (case["psi_or"][1] - X3)[1:].ravel()])
                spectrum(field, eps, N, variant, ctx, uor, opts["out"], state="oracle")
    return 1 if worst > 1e-6 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
