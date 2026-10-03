#!/usr/bin/env python3
"""SF-29 N3 -- candidate (ii): dissipation-energy fit with discretely solenoidal (mimetic) face fluxes.

Usage (run from this directory):
  python3 candidate_ii.py field:eps:N [...]                 solve + metrics (CASE lines cand=ii and cand=oracle_mim)
  python3 candidate_ii.py --free-outlet field:eps:N [...]   periodic cases WITHOUT the outlet flux constraint (cand=ii_free)
  python3 candidate_ii.py --gradtest field:eps:N [...]      div_h curl_h = 0, exact gradient / constraint Jacobian vs FD,
                                                            colored FD Hessian vs dense columns, mode preconditioner
  python3 candidate_ii.py --consistency field:eps [...]     |grad E_h| and constraint residual AT THE ORACLE LABELS,
                                                            N in --grids (default 16,32,48), observed orders
  python3 candidate_ii.py --spectrum M field:eps[:M] [...]  solve at N = M, dense Hessian (reduced Hessian + KKT if
                                                            constrained) relative singular values
Options: --out DIR (spectrum files, default ../raw), --solver auto|direct|gmres, --grids 16,32,48,
         --no-continuation, --tol 1e-13, --maxit 60 (final stage; intermediate stages maxit/2),
         --hessian fd|gn (fd: colored central FD Hessian, the reference method; gn: Gauss-Newton J^T W J, cand=..._gn),
         --start inlet|zero|oracle (initial guess; oracle = diagnostic local-convergence start, cand=..._from_oracle)

Formulation (UNDERSTAND record section 2.2).  E[psi1, psi2] = 1/2 int q |grad psi1 x grad psi2|^2, q = 1/k.
Its Euler-Lagrange equations are equation (14) (same-index pairing); the natural condition at a free face
x1 = 1 is c x e1 = 0 (constant head).  Constant-head cases (`*_ch`): inlet Dirichlet labels, outlet free.
Periodic-flow cases: inlet Dirichlet labels + outlet flux constraint c1(1, x2, x3) = f1_in (Kelvin principle
with the Neumann data of the periodic flow on both faces; deviation D-3).

Grid.  Vertices x1 = j h (j = 0..N, axis 0), x2 = m2 h, x3 = m3 h (m = 0..N-1, periodic), h = 1/N.
psi1 = x2 + U1, psi2 = x3 + U2, U_i periodic in (x2, x3), arrays (N+1, N, N).  Plane j = 0 holds the inlet
labels (U_i^0 = psi0_i - affine_i, Dirichlet); the unknowns are U1, U2 on planes 1..N, flattened as
x = [U1[1:].ravel(), U2[1:].ravel()], index(a, j, m2, m3) = a N^3 + (j-1) N^2 + m2 N + m3.

Mimetic (Whitney) flux.  Edge field a = (average of psi1 at the two edge endpoints) x (edge difference of psi2)/h
on every edge; face flux c_f = (circulation of a around the 4 edges of face f) / h^2, i.e. c_f = curl_h a, so that
div_h c_f = div_h curl_h a = 0 EXACTLY on every cell (verified in --gradtest).  For a face with corners
P00, P10, P11, P01 (counterclockwise in its (a, b) axes, normal n = e_a x e_b) the trapezoidal circulation
reduces exactly to the diagonal form

    c_f = [ (psi1(P11) - psi1(P00)) (psi2(P01) - psi2(P10)) - (psi1(P01) - psi1(P10)) (psi2(P11) - psi2(P00)) ] / (2 h^2)

which involves label DIFFERENCES only (bilinear, antisymmetric in psi1 <-> psi2).  Affine parts are handled
analytically: a difference of psi1 = x2 + U1 between corners P, Q is (U1(P) - U1(Q)) + h (d2(P) - d2(Q)), with
d2 the integer x2-offset of the corner (the same with x3 for psi2), so every stored array is periodic and the
label jumps never enter.  By bilinearity c_f = e1-piece + (e2 x G_h U2) + (G_h U1 x e3) + curl_h(avg(U1) G_h U2)
(the pieces of the task statement): psi1 = x2, psi2 = x3 gives exactly c = e1 (unit flux on x1-faces, 0 on the
others); x2 with U2 gives the average of the two x3-edge differences of U2 on x1-faces, etc.  Face families
(the layout of cases.load_case()['vD_faces']):
    f1[j, m2, m3]  x1-face in plane x1 = j h, (a, b) = (x2, x3), P00 = (j, m2, m3), P10 = +e2, P11 = +e2+e3, P01 = +e3
    f2[j, m2, m3]  x2-face in plane x2 = m2 h, (a, b) = (x3, x1), P00 = (j, m2, m3), P10 = +e3, P11 = +e3+e1, P01 = +e1
    f3[j, m2, m3]  x3-face in plane x3 = m3 h, (a, b) = (x1, x2), P00 = (j, m2, m3), P10 = +e1, P11 = +e1+e2, P01 = +e2
The collocated pointwise grad psi1 x grad psi2 (2026-10-02 note section 4, unsound) is NOT used.

Energy quadrature.  E_h = 1/2 h^3 sum_cells q_cell sum_dir (1/2)(c_{f-}^2 + c_{f+}^2): per cell, per direction the
mean of the squared fluxes of its two opposite faces (a trapezoidal rule in the normal direction), q_cell = 1/k at
the cell center with analytic k (reference.LogConductivity of the case's ref.lk).  Equivalently
E_h = 1/2 h^3 sum_faces omega_f c_f^2, omega_f = 1/2 sum of q over the (one or two) cells adjacent to f.

Exact discrete gradient (adjoint chain).  dE_h/dc_f = h^3 omega_f c_f =: W_f; dE_h/dpsi(v) = sum_f W_f dc_f/dpsi(v),
with dc_f/dpsi1(P11) = (psi2(P01) - psi2(P10))/(2h^2), etc. (scatter by np.roll; adjoint of the corner gathers).

Outlet constraint (periodic cases).  C(u) = c1[N] - t on the N^2 outlet x1-faces, t = f1_in - mean(f1_in) + 1,
f1_in = vD_faces['f1'][0].  sum_faces c1[N] = N^2 IDENTICALLY (the periodic parts telescope), so one constraint
is redundant and f1_in must have mean exactly 1 to be feasible: the mean of f1_in (1 - O(1e-10), reference /
GL3 level, printed) is replaced by 1.  Lagrange multipliers with a bordered KKT system:
    [ H_L   Jt^T  0 ] [dx  ]     [ grad E_h + Jt^T lam ]
    [ Jt    0     1 ] [dlam] = - [ Ct + mu 1           ],   Jt = h^2 dC/dx, Ct = h^2 C (scaling),
    [ 0     1^T   0 ] [dmu ]     [ sum lam             ]
(sum lam = 0 fixes the multiplier null direction 1; mu absorbs the redundant row and is 0 at the solution).
H_L is the Hessian of the Lagrangian (the constraint is bilinear in the outlet labels).

Nonlinear method.  Newton on the stationarity (KKT) system; Hessian assembled by colored central finite
differences of the exact gradient (delta = 1e-6; the coupling stencil of a vertex is the 19-point set of vertices
sharing a face, coloring by ((j-1) mod 3, m2 mod p, m3 mod p), p >= 3 the smallest divisor of N, 2 fields,
symmetrized).  Linear solve: splu (N <= 24) or GMRES (N >= 32) preconditioned by the EXACT inverse of the
constant-coefficient problem (q = mean q, affine labels): transversally translation-invariant, so per transverse
Fourier mode a 2x2-block tridiagonal system in x1 (block Thomas, vectorized over modes) bordered by the one outlet
multiplier of that mode (shifted by max(nu, 1e-6) x mean diag: the uniform-state Hessian is singular on the
hourglass modes, see below).  Globalization: Levenberg-Marquardt shift nu of the Hessian block (adapted) and
backtracking; merit E_h (Armijo, descent required) for the unconstrained problem while g_rel > 1e-6, |F|
otherwise.  Amplitude continuation s = 0.25 -> 0.5 -> 1.0 (homotopy: ln k -> s ln k, inlet periodic parts ->
s U^0, outlet target -> 1 + s (t - 1)), warm starts, initial guess = constant x1-extension of the inlet labels;
intermediate stages to g_rel <= 1e-8, the final stage to g_rel <= --tol (1e-13), or stop on stagnation
(< 2x reduction of g_rel over 15 iterations, or over 5 iterations below 1e-10), or maxit.

    g_rel = |grad_x L(x)| / |grad E_h(x0)|,  x0 = unknowns 0 (inlet labels in place, s = 1);  r_F := g_rel.

Discrete Kelvin identity (diagnostic, KELVIN line).  For every label pair with the same inlet flux (and outlet
flux, constrained case) E_h(u) = E_K + 1/2 sum_f w_f (c_f(u) - c_K,f)^2 EXACTLY, with c_K the two-point-flux
(harmonic face mean of k) Darcy flux minimizing E_h over ALL discretely solenoidal face fluxes (kelvin_tpfa;
verified in --gradtest).  So minimizing E_h over labels = least-squares fit of the mimetic flux to c_K; any exact
minimizer has c = c_K and e_v = e_v(c_K).

Hourglass deficiency of the prescribed mimetic flux (finding of N3; --gradtest prints it).  At the uniform state
(affine labels) the linearized face fluxes only see the x3-AVERAGE of the x2-edge differences of U1 (and the
analogous averages), so U1 = (-1)^m3 f(j, m2) and U2 = (-1)^m2 g(j, m3) change no face flux to first order:
2 N^2 exact null modes of the Hessian (288 at 12^3), and the linearized outlet-flux row of the (N/2, N/2)
checkerboard vanishes.  Off the uniform state they are lifted only by O(|grad u|) couplings.

Metrics (shared CASE format, cand=ii / ii_free): e_v = RMS over ALL faces of the three families of
(c_f - vD_faces) / RMS(vD_faces) (face quantities on both sides); e_psi, e_i from metrics.fd_metrics on the vertex
labels; e_div = RMS(div_h c_f)/RMS(vD_faces) from the mimetic fluxes (overrides the FD value); min_c and the
percentiles of |c| from the cell-averaged face fluxes.  Next to every candidate line the ceiling line
cand=oracle_mim: the same mimetic face flux evaluated on the exact oracle labels (deviation D-5).
No regularization of |c| anywhere (the energy has no denominator).
"""
import inspect
import itertools
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import cases as C
import metrics as M

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.normpath(os.path.join(_HERE, "..", "raw"))

# corner offsets (d1, d2, d3) of P00, P10, P11, P01 per face family (see module docstring)
FAMILIES = {
    1: ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)),
    2: ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)),
    3: ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
}
FD_DELTA = 1e-6
STAGES = (0.25, 0.5, 1.0)
STAGE_TOL = 1e-8
OFFSETS19 = [o for o in itertools.product((-1, 0, 1), repeat=3) if sum(1 for v in o if v) <= 2]


def _hdr(msg):
    print("=" * 8 + " " + msg, flush=True)


# ------------------------------------------------------------------------------------------------
# mimetic flux, its partials and the adjoint scatter
def _gather(V, fam, d, N):
    A = V if fam == 1 else V[d[0]:d[0] + N]
    if d[1] or d[2]:
        A = np.roll(A, (-d[1], -d[2]), axis=(1, 2))
    return A


def _scatter(G, fam, d, out, N):
    if d[1] or d[2]:
        G = np.roll(G, (d[1], d[2]), axis=(1, 2))
    if fam == 1:
        out += G
    else:
        out[d[0]:d[0] + N] += G


def face_terms(U1, U2, fam, h):
    """Mimetic flux of one family and the four diagonal differences (A, B, Cd, D)."""
    N = U1.shape[1]
    P = FAMILIES[fam]
    u1 = [_gather(U1, fam, d, N) for d in P]
    u2 = [_gather(U2, fam, d, N) for d in P]
    # differences with the exact affine parts (offsets x h)
    A = u1[2] - u1[0] + h * (P[2][1] - P[0][1])        # psi1(P11) - psi1(P00)
    B = u2[3] - u2[1] + h * (P[3][2] - P[1][2])        # psi2(P01) - psi2(P10)
    Cd = u1[3] - u1[1] + h * (P[3][1] - P[1][1])       # psi1(P01) - psi1(P10)
    D = u2[2] - u2[0] + h * (P[2][2] - P[0][2])        # psi2(P11) - psi2(P00)
    c = (A * B - Cd * D) / (2.0 * h * h)
    return c, (A, B, Cd, D)


def mimetic_fluxes(U1, U2, h):
    return {"f%d" % f: face_terms(U1, U2, f, h)[0] for f in (1, 2, 3)}


def adjoint(U1, U2, W, h):
    """Gradient of sum_f W_f c_f with respect to the full vertex arrays U1, U2 (adjoint chain)."""
    N = U1.shape[1]
    G1 = np.zeros_like(U1); G2 = np.zeros_like(U2)
    s = 1.0 / (2.0 * h * h)
    for fam in (1, 2, 3):
        Wf = W.get(fam)
        if Wf is None:
            continue
        _, (A, B, Cd, D) = face_terms(U1, U2, fam, h)
        P = FAMILIES[fam]
        # dc/dpsi1: P00 -B, P10 +D, P11 +B, P01 -D ; dc/dpsi2: P00 +Cd, P10 -A, P11 -Cd, P01 +A
        WB = Wf * B * s; WD = Wf * D * s; WA = Wf * A * s; WC = Wf * Cd * s
        _scatter(-WB, fam, P[0], G1, N); _scatter(WD, fam, P[1], G1, N)
        _scatter(WB, fam, P[2], G1, N); _scatter(-WD, fam, P[3], G1, N)
        _scatter(WC, fam, P[0], G2, N); _scatter(-WA, fam, P[1], G2, N)
        _scatter(-WC, fam, P[2], G2, N); _scatter(WA, fam, P[3], G2, N)
    return G1, G2


def face_divergence(fl, N):
    return C.face_divergence(fl, N)


def all_faces(fl):
    return np.concatenate([fl["f1"].ravel(), fl["f2"].ravel(), fl["f3"].ravel()])


def cell_flux_norm(fl):
    f1, f2, f3 = fl["f1"], fl["f2"], fl["f3"]
    a1 = 0.5 * (f1[1:] + f1[:-1])
    a2 = 0.5 * (f2 + np.roll(f2, -1, axis=1))
    a3 = 0.5 * (f3 + np.roll(f3, -1, axis=2))
    return np.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)


# ------------------------------------------------------------------------------------------------
class EnergyProblem:
    """Discrete energy, gradient, outlet constraint for one case at homotopy stage s."""

    def __init__(self, case, constrained, s=1.0, qconst=None, zero_inlet=False):
        meta = case["meta"]
        self.case = case
        self.N = N = int(meta["N"]); self.h = h = 1.0 / N
        self.N3 = N ** 3; self.n = 2 * N ** 3
        self.constrained = bool(constrained)
        self.s = float(s)
        xc = (np.arange(N) + 0.5) * h
        X1, X2, X3 = np.meshgrid(xc, xc, xc, indexing="ij")
        self.lnk_cell = case["ref"].lk.lnk(X1, X2, X3)
        q = np.exp(-self.s * self.lnk_cell) if qconst is None else np.full((N, N, N), float(qconst))
        self.q = q
        w1 = np.zeros((N + 1, N, N)); w1[:N] += 0.5 * q; w1[1:] += 0.5 * q
        self.omega = {1: w1, 2: 0.5 * (q + np.roll(q, 1, axis=1)), 3: 0.5 * (q + np.roll(q, 1, axis=2))}
        a2, a3 = M.affine(N)
        if zero_inlet:
            self.U0 = (np.zeros((N, N)), np.zeros((N, N)))
        else:
            p0 = case["psi0"]
            self.U0 = (self.s * (p0[0] - a2[0]), self.s * (p0[1] - a3[0]))
        f1in = case["vD_faces"]["f1"][0]
        self.f1in = f1in
        self.f1in_mean_defect = float(f1in.mean() - 1.0)
        t = f1in - f1in.mean() + 1.0
        self.t_full = t
        self.t = 1.0 + self.s * (t - 1.0)
        self._J = None

    # layout
    def full(self, x):
        N = self.N
        U1 = np.empty((N + 1, N, N)); U2 = np.empty((N + 1, N, N))
        U1[0] = self.U0[0]; U2[0] = self.U0[1]
        U1[1:] = x[:self.N3].reshape(N, N, N); U2[1:] = x[self.N3:].reshape(N, N, N)
        return U1, U2

    def pack(self, U1, U2):
        return np.concatenate([U1[1:].ravel(), U2[1:].ravel()])

    def idx(self, a, j, m2, m3):
        N = self.N
        return a * self.N3 + (j - 1) * N * N + m2 * N + m3

    # energy and gradient
    def fluxes(self, x):
        U1, U2 = self.full(x)
        return mimetic_fluxes(U1, U2, self.h)

    def energy(self, x):
        fl = self.fluxes(x)
        h3 = self.h ** 3
        return 0.5 * h3 * sum(float(np.sum(self.omega[f] * fl["f%d" % f] ** 2)) for f in (1, 2, 3))

    def grad(self, x):
        U1, U2 = self.full(x)
        h3 = self.h ** 3
        W = {f: h3 * self.omega[f] * face_terms(U1, U2, f, self.h)[0] for f in (1, 2, 3)}
        G1, G2 = adjoint(U1, U2, W, self.h)
        return np.concatenate([G1[1:].ravel(), G2[1:].ravel()])

    # constraint (raw, unscaled): c1 on the outlet x1-faces minus the target
    def cons(self, x):
        U1, U2 = self.full(x)
        return (face_terms(U1, U2, 1, self.h)[0][self.N] - self.t).ravel()

    def consT(self, x, lam):
        """(dC/dx)^T lam (raw constraint)."""
        U1, U2 = self.full(x)
        N = self.N
        W1 = np.zeros((N + 1, N, N)); W1[N] = lam.reshape(N, N)
        G1, G2 = adjoint(U1, U2, {1: W1}, self.h)
        return np.concatenate([G1[1:].ravel(), G2[1:].ravel()])

    def flux_jac(self, x):
        """Exact sparse dc_f/dx for all faces (rows: f1 (N+1)N^2, then f2 N^3, then f3 N^3) and the energy
        weights w_f = h^3 omega_f, so that grad E_h = J^T (w c) and the Gauss-Newton Hessian is J^T diag(w) J."""
        U1, U2 = self.full(x)
        N, h = self.N, self.h
        s = 1.0 / (2.0 * h * h)
        rows, cols, vals, wts = [], [], [], []
        off = 0
        for fam in (1, 2, 3):
            _, (A, B, Cd, D) = face_terms(U1, U2, fam, h)
            P = FAMILIES[fam]
            nj = N + 1 if fam == 1 else N
            Jf, M2, M3 = np.meshgrid(np.arange(nj), np.arange(N), np.arange(N), indexing="ij")
            row = off + (Jf * N * N + M2 * N + M3)
            parts = [(0, P[0], -B), (0, P[1], D), (0, P[2], B), (0, P[3], -D),
                     (1, P[0], Cd), (1, P[1], -A), (1, P[2], -Cd), (1, P[3], A)]
            for a, d, v in parts:
                jv = Jf + d[0]
                ok = jv >= 1                                   # plane 0 is Dirichlet data
                rows.append(row[ok])
                cols.append(self.idx(a, jv[ok], (M2[ok] + d[1]) % N, (M3[ok] + d[2]) % N))
                vals.append((v * s)[ok])
            wts.append((h ** 3 * self.omega[fam]).ravel())
            off += nj * N * N
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(off, self.n))
        return J, np.concatenate(wts)

    def cons_jac(self, x):
        """Exact sparse dC/dx (N^2 x n): rows = outlet x1-faces (m2 N + m3)."""
        U1, U2 = self.full(x)
        N, h = self.N, self.h
        s = 1.0 / (2.0 * h * h)
        _, (A, B, Cd, D) = face_terms(U1, U2, 1, h)
        A, B, Cd, D = A[N], B[N], Cd[N], D[N]
        P = FAMILIES[1]
        M2, M3 = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        row = (M2 * N + M3).ravel()
        parts = [(0, P[0], -B), (0, P[1], D), (0, P[2], B), (0, P[3], -D),
                 (1, P[0], Cd), (1, P[1], -A), (1, P[2], -Cd), (1, P[3], A)]
        rows, cols, vals = [], [], []
        for a, d, v in parts:
            rows.append(row)
            cols.append(self.idx(a, N, (M2 + d[1]) % N, (M3 + d[2]) % N).ravel())
            vals.append((v * s).ravel())
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N * N, self.n))

    # Lagrangian gradient (scaled multipliers: Jt = h^2 dC/dx)
    def gradL(self, x, lam):
        g = self.grad(x)
        if self.constrained and lam is not None:
            g = g + self.h ** 2 * self.consT(x, lam)
        return g


# ------------------------------------------------------------------------------------------------
# Hessian by colored central finite differences of the gradient
def _period(N):
    for p in range(3, N + 1):
        if N % p == 0:
            return p
    return N


def hessian_fd(prob, gfun, x, delta=FD_DELTA):
    N = prob.N
    p = _period(N)
    J, M2, M3 = np.meshgrid(np.arange(1, N + 1), np.arange(N), np.arange(N), indexing="ij")
    J = J.ravel(); M2 = M2.ravel(); M3 = M3.ravel()
    rows, cols, vals = [], [], []
    ngrad = 0
    for a in (0, 1):
        for r1 in range(3):
            for r2 in range(p):
                for r3 in range(p):
                    mask = ((J - 1) % 3 == r1) & (M2 % p == r2) & (M3 % p == r3)
                    cj, cm2, cm3 = J[mask], M2[mask], M3[mask]
                    if cj.size == 0:
                        continue
                    cidx = prob.idx(a, cj, cm2, cm3)
                    e = np.zeros(prob.n); e[cidx] = delta
                    Dg = (gfun(x + e) - gfun(x - e)) / (2.0 * delta)
                    ngrad += 2
                    for o in OFFSETS19:
                        rj = cj + o[0]
                        ok = (rj >= 1) & (rj <= N)
                        if not ok.any():
                            continue
                        rm2 = (cm2[ok] + o[1]) % N; rm3 = (cm3[ok] + o[2]) % N
                        for b in (0, 1):
                            ridx = prob.idx(b, rj[ok], rm2, rm3)
                            rows.append(ridx); cols.append(cidx[ok]); vals.append(Dg[ridx])
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(prob.n, prob.n))
    H = 0.5 * (H + H.T)
    return H.tocsr(), ngrad


def hessian_dense(gfun, x, delta=FD_DELTA):
    n = x.size
    H = np.empty((n, n))
    e = np.zeros(n)
    for i in range(n):
        e[i] = delta
        H[:, i] = (gfun(x + e) - gfun(x - e)) / (2.0 * delta)
        e[i] = 0.0
    return 0.5 * (H + H.T)


# ------------------------------------------------------------------------------------------------
# exact inverse of the constant-coefficient problem per transverse Fourier mode (preconditioner)
class ModePreconditioner:
    """Exact inverse of the constant-coefficient (q = qbar, affine labels) Hessian/KKT, shifted by
    `shift_rel` x mean(diag): the uniform-state Hessian is singular on the 2 N^2 hourglass modes (README,
    findings of N3); the shift only affects the preconditioner, never the operator."""

    def __init__(self, case, constrained, qbar, shift_rel=1e-6):
        prob = EnergyProblem(case, constrained, s=1.0, qconst=qbar, zero_inlet=True)
        N = prob.N; self.N = N; self.constrained = constrained; self.n = prob.n
        x0 = np.zeros(prob.n)
        nm = N * N
        Dg = np.zeros((nm, N, 2, 2), complex)
        Lo = np.zeros((nm, max(N - 1, 1), 2, 2), complex)      # B[j+1, j]
        Up = np.zeros((nm, max(N - 1, 1), 2, 2), complex)      # B[j, j+1]
        for a in (0, 1):
            for j in range(1, N + 1):
                e = np.zeros(prob.n); e[prob.idx(a, j, 0, 0)] = FD_DELTA
                col = (prob.grad(x0 + e) - prob.grad(x0 - e)) / (2 * FD_DELTA)
                col = col.reshape(2, N, N, N)                     # [b, plane, m2, m3]
                jj = j - 1
                for b in (0, 1):
                    Dg[:, jj, b, a] = np.fft.fft2(col[b, jj]).ravel()
                    if jj + 1 < N:
                        Lo[:, jj, b, a] = np.fft.fft2(col[b, jj + 1]).ravel()
                    if jj - 1 >= 0:
                        Up[:, jj - 1, b, a] = np.fft.fft2(col[b, jj - 1]).ravel()
        self.Up = Up; self.Lo = Lo
        dmean = float(np.mean(np.real(np.einsum("mjaa->mja", Dg))))
        self.shift = shift_rel * dmean
        Dg = Dg + self.shift * np.eye(2)[None, None]
        # block Thomas factorization: S_j = D_j - Lo_{j-1} S_{j-1}^{-1} Up_{j-1}
        Sinv = np.zeros_like(Dg)
        S = Dg[:, 0]
        Sinv[:, 0] = np.linalg.inv(S)
        for jj in range(1, N):
            S = Dg[:, jj] - Lo[:, jj - 1] @ Sinv[:, jj - 1] @ Up[:, jj - 1]
            Sinv[:, jj] = np.linalg.inv(S)
        self.Sinv = Sinv
        if constrained:
            Jt = prob.h ** 2 * prob.cons_jac(x0)
            row = Jt.getrow(0).toarray().ravel().reshape(2, N, N, N)[:, N - 1]   # outlet plane, [a, m2, m3]
            s = np.stack([N * N * np.fft.ifft2(row[a]).ravel() for a in (0, 1)], axis=1)   # (nm, 2)
            self.s = s
            rhs = np.zeros((nm, N, 2), complex); rhs[:, N - 1] = np.conj(s)
            self.z = self._bsolve(rhs)
            self.sigma = np.einsum("ma,ma->m", s, self.z[:, N - 1])
            # modes whose linearized outlet-flux row vanishes: xi = 0 (sum c1 = N^2 identically) and the
            # (N/2, N/2) checkerboard (the hourglass deficiency of the mimetic flux, README): no multiplier
            self.dead = np.abs(self.sigma) < 1e-12 * np.abs(self.sigma).max()
            self.dead[0] = True
            self.sigma[self.dead] = 1.0

    def _bsolve(self, R):
        N = self.N
        y = np.empty_like(R)
        y[:, 0] = R[:, 0]
        for jj in range(1, N):
            y[:, jj] = R[:, jj] - np.einsum("mab,mb->ma", self.Lo[:, jj - 1] @ self.Sinv[:, jj - 1], y[:, jj - 1])
        x = np.empty_like(R)
        x[:, N - 1] = np.einsum("mab,mb->ma", self.Sinv[:, N - 1], y[:, N - 1])
        for jj in range(N - 2, -1, -1):
            x[:, jj] = np.einsum("mab,mb->ma", self.Sinv[:, jj],
                                 y[:, jj] - np.einsum("mab,mb->ma", self.Up[:, jj], x[:, jj + 1]))
        return x

    def apply(self, r):
        N = self.N; nm = N * N; n = self.n
        r1 = r[:n].reshape(2, N, N, N)
        R1 = np.fft.fft2(r1, axes=(2, 3)).reshape(2, N, nm).transpose(2, 1, 0)   # (nm, plane, field)
        if not self.constrained:
            X = self._bsolve(R1)
            return np.fft.ifft2(X.transpose(2, 1, 0).reshape(2, N, N, N), axes=(2, 3)).real.ravel()
        r2 = r[n:n + nm].reshape(N, N); r3 = r[n + nm]
        R2 = np.fft.fft2(r2).ravel()
        Y = self._bsolve(R1)
        lam = (np.einsum("ma,ma->m", self.s, Y[:, N - 1]) - R2) / self.sigma
        lam[self.dead] = 0.0
        X = Y - self.z * lam[:, None, None]
        X[self.dead] = Y[self.dead]                                 # s = 0: no coupling on dead modes
        lam[0] = r3                                                 # sum(lam) = r3 (unnormalized FFT)
        mu = R2[0].real / nm
        x = np.fft.ifft2(X.transpose(2, 1, 0).reshape(2, N, N, N), axes=(2, 3)).real.ravel()
        lr = np.fft.ifft2(lam.reshape(N, N)).real.ravel()
        return np.concatenate([x, lr, [mu]])


def _gmres(A, b, Mop, rtol, restart=60, maxiter=40):
    its = [0]

    def cb(_):
        its[0] += 1
    kw = dict(restart=restart, maxiter=maxiter, M=Mop, atol=0.0, callback=cb, callback_type="pr_norm")
    if "rtol" in inspect.signature(spla.gmres).parameters:
        kw["rtol"] = rtol
    else:
        kw["tol"] = rtol
    x, info = spla.gmres(A, b, **kw)
    return x, info, its[0]


# ------------------------------------------------------------------------------------------------
# Newton on the stationarity / KKT system
def kkt_residual(prob, x, lam, mu):
    g = prob.gradL(x, lam)
    if not prob.constrained:
        return g, g, None
    h2 = prob.h ** 2
    c = prob.cons(x)
    F2 = h2 * c + mu
    F3 = np.array([lam.sum()])
    return np.concatenate([g, F2, F3]), g, c


def kkt_matrix(prob, H, x):
    if not prob.constrained:
        return H
    nm = prob.N ** 2
    Jt = (prob.h ** 2) * prob.cons_jac(x)
    one = sp.csr_matrix(np.ones((nm, 1)))
    return sp.bmat([[H, Jt.T, None], [Jt, None, one], [None, one.T, None]], format="csc")


def _solve_lin(K, rhs, solver, prec):
    if solver == "direct":
        return spla.splu(K.tocsc()).solve(rhs), "splu"
    Mop = spla.LinearOperator(K.shape, matvec=prec.apply)
    d, info, nit = _gmres(K, rhs, Mop, rtol=1e-10)
    lres = np.linalg.norm(K @ d - rhs) / np.linalg.norm(rhs)
    return d, "gmres its=%d info=%d linres=%.1e" % (nit, info, lres)


def newton_stage(prob, x, lam, mu, gref, tol, ctol, solver, case, maxit=40, label="", hessian="fd"):
    """Newton on the stationarity / KKT system, globalized by a Levenberg-Marquardt shift of the Hessian
    block (H + nu dmean I, nu adapted: /10 after a full step, x10 after a rejected trial; nu -> 0 gives the
    pure Newton step) and backtracking (<= 3 halvings per trial).  Merit: E_h with Armijo while
    g_rel > 1e-6 in the unconstrained case (descent direction required: avoids saddles), |F| otherwise
    (and always for the KKT system)."""
    hist = []
    F, g, c = kkt_residual(prob, x, lam, mu)
    nF = np.linalg.norm(F)
    E = prob.energy(x)
    its = 0
    status = "maxit"
    nu = 1e-4
    n = prob.n; nm = prob.N ** 2
    for it in range(maxit + 1):
        grel = np.linalg.norm(g) / gref
        crel = (np.abs(c).max() / np.abs(prob.t).max()) if c is not None else 0.0
        hist.append((it, grel, crel, E))
        print("  NEWTON %s it=%2d g_rel=%.3e c_rel=%.3e E=%.15e |F|=%.3e nu=%.1e" % (label, it, grel, crel, E, nF, nu),
              flush=True)
        if grel <= tol and crel <= ctol:
            status = "converged"
            break
        if it == maxit:
            break
        if len(hist) >= 6 and hist[-1][1] > 0.5 * hist[-6][1] and grel < 1e-10:
            status = "stagnation(roundoff)"
            break
        if len(hist) >= 16 and hist[-1][1] > 0.5 * hist[-16][1]:
            status = "stagnation(plateau: < 2x reduction of g_rel in 15 iterations)"
            break
        t0 = time.time()
        if hessian == "gn":
            # Gauss-Newton: E_h = E_K + 1/2 |c(u) - c_K|_W^2 exactly (Kelvin), so J^T W J is the Hessian at a
            # zero-residual minimizer; the (bilinear) constraint curvature lam . d2C is added exactly.
            Jf, w = prob.flux_jac(x)
            H = (Jf.T @ sp.diags(w) @ Jf).tocsr()
            ngr = 0
            if prob.constrained:
                Hc, ngr = hessian_fd(prob, lambda y: prob.h ** 2 * prob.consT(y, lam), x)
                H = (H + Hc).tocsr()
        else:
            gfun = (lambda y: prob.gradL(y, lam)) if prob.constrained else prob.grad
            H, ngr = hessian_fd(prob, gfun, x)
        dmean = float(np.mean(np.abs(H.diagonal())))
        t1 = time.time()
        accepted = False
        use_E = (not prob.constrained) and grel > 1e-6
        trial = 0
        for trial in range(12):
            Hs = (H + (nu * dmean) * sp.identity(n, format="csr")) if nu > 0 else H
            K = kkt_matrix(prob, Hs, x)
            prec = ModePreconditioner(case, prob.constrained, float(prob.q.mean()),
                                      shift_rel=max(nu, 1e-6)) if solver == "gmres" else None
            d, lin = _solve_lin(K, -F, solver, prec)
            gd = float(g @ d[:n])
            alpha = 1.0
            for _ in range(4):
                xn = x + alpha * d[:n]
                ln = lam + alpha * d[n:n + nm] if prob.constrained else lam
                mn = mu + alpha * d[n + nm] if prob.constrained else mu
                Fn, gn, cn = kkt_residual(prob, xn, ln, mn)
                nFn = np.linalg.norm(Fn)
                En = prob.energy(xn)
                if use_E:
                    ok = gd < 0 and En <= E + 1e-4 * alpha * gd
                else:
                    ok = nFn <= (1 - 1e-4 * alpha) * nF
                if ok:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                break
            nu = 1e-8 if nu == 0 else min(10 * nu, 1e8)
        if not accepted:
            status = "linesearch_failed"
            break
        t2 = time.time()
        print("    step alpha=%.3g |d|=%.2e nu=%.1e trials=%d hess(ngrad=%d nnz=%d)=%.2fs solve[%s]=%.2fs"
              % (alpha, np.linalg.norm(d[:n]), nu, trial + 1, ngr, H.nnz, t1 - t0, lin, t2 - t1), flush=True)
        if alpha == 1.0:
            nu = 0.0 if nu <= 1e-10 else nu / 10.0
        x, lam, mu = xn, ln, mn
        F, g, c, nF, E = Fn, gn, cn, nFn, En
        its += 1
    return x, lam, mu, its, status, hist


def initial_guess(prob, case, start):
    N = prob.N
    if start == "zero":
        return np.zeros(prob.n)
    if start == "oracle":
        a2, a3 = M.affine(N)
        return prob.pack(case["psi_or"][0] - a2, case["psi_or"][1] - a3)
    # "inlet" (default): constant extension in x1 of the (stage-scaled) inlet periodic parts
    U1 = np.broadcast_to(prob.U0[0], (N + 1, N, N)).copy(); U2 = np.broadcast_to(prob.U0[1], (N + 1, N, N)).copy()
    return prob.pack(U1, U2)


def solve_case(case, constrained, solver="auto", tol=1e-13, continuation=True, start="inlet", maxit=60, hessian="fd"):
    N = int(case["meta"]["N"])
    if solver == "auto":
        solver = "direct" if N <= 24 else "gmres"
    stages = STAGES if (continuation and start != "oracle") else (1.0,)
    prob1 = EnergyProblem(case, constrained, s=1.0)
    gref = np.linalg.norm(prob1.grad(np.zeros(prob1.n)))
    lam = np.zeros(N * N) if constrained else None
    mu = 0.0
    tot = 0; history = []
    t0 = time.time()
    status = None
    x = None
    for s in stages:
        prob = prob1 if s == 1.0 else EnergyProblem(case, constrained, s=s)
        if x is None:
            x = initial_guess(prob, case, start)
        stol = tol if s == 1.0 else STAGE_TOL
        x, lam, mu, its, status, hist = newton_stage(prob, x, lam, mu, gref, stol, 1e-12 if s == 1.0 else 1e-8,
                                                     solver, case, maxit=maxit if s == 1.0 else maxit // 2,
                                                     label="s=%.2f" % s, hessian=hessian)
        tot += its; history.append((s, its, status, hist))
    return prob1, x, lam, mu, tot, status, history, gref, time.time() - t0, solver


# ------------------------------------------------------------------------------------------------
# discrete Kelvin reference (diagnostic): the minimizer of E_h over ALL discretely solenoidal face fluxes
def kelvin_tpfa(prob, inlet_flux, outlet_flux=None):
    """min 1/2 h^3 sum omega c^2 s.t. div_h c = 0 per cell, c1[0] = inlet_flux, c1[N] = outlet_flux (or free).
    Stationarity: c_f = (p_L - p_R) / (h omega_f) (two-point flux with the harmonic face mean of k); free
    outlet <-> p = 0 outside.  Returns the face fluxes and their E_h.  Any label pair has E_h >= this value
    (its mimetic fluxes are discretely solenoidal with the same inlet flux), with equality iff c_f = these."""
    N, h = prob.N, prob.h
    om = prob.omega
    nc = N ** 3
    cid = np.arange(nc).reshape(N, N, N)
    rows, cols, vals = [], [], []
    b = np.zeros(nc)

    def add(i, j, v):
        rows.append(i.ravel()); cols.append(j.ravel()); vals.append(np.broadcast_to(v, i.shape).ravel())
    T1 = 1.0 / (h * om[1])
    L = cid[:-1]; R = cid[1:]; T = T1[1:N]                 # interior x1-faces j = 1..N-1
    add(L, L, T); add(L, R, -T); add(R, R, T); add(R, L, -T)
    for fam, ax in ((2, 1), (3, 2)):                       # face m between cells m-1 (L) and m (R)
        T = 1.0 / (h * om[fam])
        R = cid; L = np.roll(cid, 1, axis=ax)
        add(L, L, T); add(L, R, -T); add(R, R, T); add(R, L, -T)
    b[cid[0].ravel()] += inlet_flux.ravel()
    if outlet_flux is None:
        add(cid[N - 1], cid[N - 1], T1[N])
    else:
        b[cid[N - 1].ravel()] -= outlet_flux.ravel()
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nc, nc))
    if outlet_flux is not None:                            # pure Neumann: pin one pressure value
        A = A.tolil(); A[0, :] = 0.0; A[0, 0] = 1.0; A = A.tocsr(); b[0] = 0.0
    p = spla.spsolve(A.tocsc(), b).reshape(N, N, N)
    f1 = np.empty((N + 1, N, N)); f1[0] = inlet_flux
    f1[1:N] = (p[:-1] - p[1:]) * T1[1:N]
    f1[N] = p[N - 1] * T1[N] if outlet_flux is None else outlet_flux
    f2 = (np.roll(p, 1, axis=1) - p) / (h * om[2])
    f3 = (np.roll(p, 1, axis=2) - p) / (h * om[3])
    fl = {"f1": f1, "f2": f2, "f3": f3}
    E = 0.5 * h ** 3 * sum(float(np.sum(om[f] * fl["f%d" % f] ** 2)) for f in (1, 2, 3))
    return fl, E


# ------------------------------------------------------------------------------------------------
def candidate_metrics(case, U1, U2, fl):
    N = int(case["meta"]["N"])
    vf = case["vD_faces"]
    a2, a3 = M.affine(N)
    m = M.fd_metrics(a2 + U1, a3 + U2, case["vD"], case["psi_or"])
    vref = all_faces(vf)
    m["e_v"] = M.rms(all_faces(fl) - vref) / M.rms(vref)
    m["e_v_fam"] = tuple(M.rms(fl[k] - vf[k]) / M.rms(vref) for k in ("f1", "f2", "f3"))
    m["e_div"] = M.rms(face_divergence(fl, N)) / M.rms(vref)
    m["div_max"] = float(np.abs(face_divergence(fl, N)).max())
    cn = cell_flux_norm(fl)
    m["min_c"] = float(cn.min())
    m["p0.1"], m["p1"], m["p5"], m["p50"] = (float(v) for v in np.percentile(cn, [0.1, 1.0, 5.0, 50.0]))
    # natural-condition defect at the outlet: c_perp extrapolated (2nd order) from the last two cell layers
    h = 1.0 / N
    cout = [1.5 * fl[k][N - 1] - 0.5 * fl[k][N - 2] for k in ("f2", "f3")]
    vout = [1.5 * vf[k][N - 1] - 0.5 * vf[k][N - 2] for k in ("f2", "f3")]
    m["nat_defect"] = float(np.sqrt(np.mean(cout[0] ** 2 + cout[1] ** 2))) / M.rms(vref)
    m["nat_defect_vD"] = float(np.sqrt(np.mean(vout[0] ** 2 + vout[1] ** 2))) / M.rms(vref)
    m["cperp_out_vs_vD"] = float(np.sqrt(np.mean((cout[0] - vout[0]) ** 2 + (cout[1] - vout[1]) ** 2))) / M.rms(vref)
    m["c1_out_vs_f1in"] = M.rms(fl["f1"][N] - vf["f1"][0]) / M.rms(vf["f1"][0])
    m["c1_out_vs_vD"] = M.rms(fl["f1"][N] - vf["f1"][N]) / M.rms(vf["f1"][N])
    return m


def _extra_line(tag, field, eps, N, m):
    print("%s field=%s eps=%g N=%d e_v(f1,f2,f3)=(%.3e,%.3e,%.3e) max|div_h c|=%.1e nat_defect|c_perp(out)|=%.3e "
          "(vD: %.3e) |c_perp(out)-vD_perp(out)|=%.3e |c1(out)-f1_in|=%.3e |c1(out)-vD_f1(out)|=%.3e"
          % (tag, field, eps, N, m["e_v_fam"][0], m["e_v_fam"][1], m["e_v_fam"][2], m["div_max"], m["nat_defect"],
             m["nat_defect_vD"], m["cperp_out_vs_vD"], m["c1_out_vs_f1in"], m["c1_out_vs_vD"]), flush=True)


def hourglass_content(U1, U2):
    """RMS of the hourglass components: U1 at the x3-Nyquist mode, U2 at the x2-Nyquist mode (planes 1..N)."""
    N = U1.shape[1]
    s = (-1.0) ** np.arange(N)
    return (float(np.sqrt(np.mean((U1[1:] * s[None, None, :]).mean(axis=2) ** 2))),
            float(np.sqrt(np.mean((U2[1:] * s[None, :, None]).mean(axis=1) ** 2))))


def run_case(field, eps, N, free_outlet=False, solver="auto", tol=1e-13, continuation=True, spectrum_out=None,
             start="inlet", maxit=60, hessian="fd"):
    _hdr("candidate (ii) %s:%g:%d %s start=%s" % (field, eps, N, "free-outlet" if free_outlet else "", start))
    tl = time.time()
    case = C.load_case(field, eps, N, verbose=False)
    tl = time.time() - tl
    ch = field.endswith("_ch")
    constrained = (not ch) and (not free_outlet)
    cand = "ii_free" if (free_outlet and not ch) else "ii"
    if start == "oracle":
        cand += "_from_oracle"
    if hessian == "gn":
        cand += "_gn"
    prob, x, lam, mu, its, status, hist, gref, tsol, solver = solve_case(case, constrained, solver, tol, continuation,
                                                                         start=start, maxit=maxit, hessian=hessian)
    U1, U2 = prob.full(x)
    fl = mimetic_fluxes(U1, U2, prob.h)
    m = candidate_metrics(case, U1, U2, fl)
    grel = np.linalg.norm(prob.gradL(x, lam)) / gref
    crel = (np.abs(prob.cons(x)).max() / np.abs(prob.t).max()) if constrained else float("nan")
    a2, a3 = M.affine(N)
    Uor = (case["psi_or"][0] - a2, case["psi_or"][1] - a3)
    xor = prob.pack(*Uor)
    E, Eor = prob.energy(x), prob.energy(xor)
    print("SOLVE field=%s eps=%g N=%d cand=%s constrained=%s solver=%s status=%s its=%d g_rel=%.3e c_rel(max)=%.3e "
          "f1_in mean-1=%+.1e mu=%.1e |lam|=%.3e E_min=%.15e E(oracle)=%.15e E_min-E(oracle)=%+.3e t_load=%.1fs t_solve=%.1fs"
          % (field, eps, N, cand, constrained, solver, status, its, grel, crel, prob.f1in_mean_defect, mu,
             np.linalg.norm(lam) if lam is not None else 0.0, E, Eor, E - Eor, tl, tsol), flush=True)
    hstr = " ".join("%.1e" % h[1] for st in hist for h in st[3])
    print("HISTORY field=%s eps=%g N=%d cand=%s g_rel: %s" % (field, eps, N, cand, hstr), flush=True)
    _extra_line("DIAG", field, eps, N, m)
    print("VDREF field=%s eps=%g N=%d |vD| p0.1/1/5/50=%.4f/%.4f/%.4f/%.4f min|vD|=%.4f"
          % (field, eps, N, m["vD_p"][0], m["vD_p"][1], m["vD_p"][2], m["vD_p"][3], m["vD_min"]), flush=True)
    M.print_case(field, eps, N, cand, m, r_F=grel, its=its, t=tsol)
    # oracle-FD ceiling (D-5): the same mimetic flux on the exact oracle labels
    flo = mimetic_fluxes(Uor[0], Uor[1], prob.h)
    mo = candidate_metrics(case, Uor[0], Uor[1], flo)
    mo["e_psi"] = 0.0
    _extra_line("DIAG_ORACLE", field, eps, N, mo)
    M.print_case(field, eps, N, "oracle_mim", mo, r_F=None, its=None, t=0.0)
    # discrete Kelvin bound (diagnostic): best discretely solenoidal flux with the same inlet flux (and outlet)
    flk, Ek = kelvin_tpfa(prob, fl["f1"][0], prob.t if constrained else None)
    vref = all_faces(case["vD_faces"])
    hg = hourglass_content(U1, U2); hgo = hourglass_content(*Uor)
    print("KELVIN field=%s eps=%g N=%d cand=%s E_kelvin=%.15e E_min-E_kelvin=%+.3e E(oracle)-E_kelvin=%+.3e "
          "e_v(kelvin flux vs vD_faces)=%.3e |c_min - c_kelvin|/rms(vD)=%.3e |c_oracle - c_kelvin|/rms(vD)=%.3e "
          "hourglass rms (U1 x3-Nyq, U2 x2-Nyq): candidate=(%.2e,%.2e) oracle=(%.2e,%.2e)"
          % (field, eps, N, cand, Ek, E - Ek, Eor - Ek, M.rms(all_faces(flk) - vref) / M.rms(vref),
             M.rms(all_faces(fl) - all_faces(flk)) / M.rms(vref), M.rms(all_faces(flo) - all_faces(flk)) / M.rms(vref),
             hg[0], hg[1], hgo[0], hgo[1]), flush=True)
    if spectrum_out is not None:
        spectrum(prob, x, lam, field, eps, N, spectrum_out, tag=("_gn" if hessian == "gn" else ""))
    return m, mo


# ------------------------------------------------------------------------------------------------
def spec_stats(s):
    s = np.sort(np.abs(np.asarray(s)))
    r = s / s.max()
    counts = {t: int(np.sum(r < t)) for t in (1e-2, 1e-3, 1e-4, 1e-6, 1e-10)}
    best = (1.0, -1)
    for i in range(len(r) - 1):
        if r[i] < 1e-2:
            g = r[i + 1] / r[i] if r[i] > 0 else np.inf
            if g > best[0]:
                best = (g, i + 1)
    return r, counts, best


def _print_spec(tag, field, eps, N, s, path):
    r, counts, best = spec_stats(s)
    print("SPECTRUM %s field=%s eps=%g N=%d n=%d smallest12: %s" % (tag, field, eps, N, r.size,
                                                                   " ".join("%.2e" % v for v in r[:12])), flush=True)
    print("SPECTRUM %s field=%s eps=%g N=%d counts <1e-2:%d <1e-3:%d <1e-4:%d <1e-6:%d <1e-10:%d | largest gap below 1e-2: "
          "ratio=%.2f after %d values (r=%.2e -> %.2e)"
          % (tag, field, eps, N, counts[1e-2], counts[1e-3], counts[1e-4], counts[1e-6], counts[1e-10], best[0], best[1],
             r[best[1] - 1] if best[1] > 0 else np.nan, r[best[1]] if best[1] > 0 else np.nan), flush=True)
    np.savetxt(path, r, header="sorted relative singular values (%s) field=%s eps=%g N=%d n=%d" % (tag, field, eps, N, r.size))
    print("SPECTRUM %s saved %s" % (tag, os.path.relpath(path, _HERE)), flush=True)


def spectrum(prob, x, lam, field, eps, N, outdir, tag=""):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    gfun = (lambda y: prob.gradL(y, lam)) if prob.constrained else prob.grad
    H = hessian_dense(gfun, x)
    ev = np.linalg.eigvalsh(H)
    print("SPECTRUM field=%s eps=%g N=%d dense Hessian%s n=%d: eig min=%.3e max=%.3e negative=%d (t=%.1fs)"
          % (field, eps, N, " of the Lagrangian" if prob.constrained else "", H.shape[0], ev.min(), ev.max(),
             int(np.sum(ev < 0)), time.time() - t0), flush=True)
    base = os.path.join(outdir, "spectrum_cand_ii%s_%s_%g_%d" % (tag, field, eps, N))
    if not prob.constrained:
        _print_spec("hessian", field, eps, N, np.linalg.svd(H, compute_uv=False), base + ".txt")
        return
    Jt = (prob.h ** 2 * prob.cons_jac(x)).toarray()
    U, sJ, Vt = np.linalg.svd(Jt, full_matrices=True)
    rank = int(np.sum(sJ > 1e-12 * sJ.max()))
    print("SPECTRUM field=%s eps=%g N=%d constraint Jacobian %dx%d rank=%d (sigma: max=%.3e min_kept=%.3e dropped=%s)"
          % (field, eps, N, Jt.shape[0], Jt.shape[1], rank, sJ.max(), sJ[rank - 1],
             " ".join("%.1e" % v for v in sJ[rank:])), flush=True)
    Z = Vt[rank:].T
    Hr = Z.T @ H @ Z
    evr = np.linalg.eigvalsh(0.5 * (Hr + Hr.T))
    print("SPECTRUM field=%s eps=%g N=%d reduced Hessian Z^T H_L Z n=%d: eig min=%.3e max=%.3e negative=%d"
          % (field, eps, N, Hr.shape[0], evr.min(), evr.max(), int(np.sum(evr < 0))), flush=True)
    _print_spec("reduced_hessian", field, eps, N, np.linalg.svd(Hr, compute_uv=False), base + ".txt")
    Jr = U[:, :rank].T @ Jt
    nr = rank
    K = np.zeros((H.shape[0] + nr, H.shape[0] + nr))
    K[:H.shape[0], :H.shape[0]] = H; K[:H.shape[0], H.shape[0]:] = Jr.T; K[H.shape[0]:, :H.shape[0]] = Jr
    _print_spec("kkt", field, eps, N, np.linalg.svd(K, compute_uv=False), base + "_kkt.txt")
    _print_spec("hessian_L_full", field, eps, N, np.linalg.svd(H, compute_uv=False), base + "_hL.txt")


# ------------------------------------------------------------------------------------------------
def consistency(field, eps, grids, free_outlet=False):
    _hdr("consistency at the oracle labels %s:%g grids=%s" % (field, eps, grids))
    ch = field.endswith("_ch")
    constrained = (not ch) and (not free_outlet)
    rows = []
    for N in grids:
        case = C.load_case(field, eps, N, verbose=False)
        prob = EnergyProblem(case, constrained, s=1.0)
        a2, a3 = M.affine(N)
        xor = prob.pack(case["psi_or"][0] - a2, case["psi_or"][1] - a3)
        gref = np.linalg.norm(prob.grad(np.zeros(prob.n)))
        g = prob.grad(xor)
        nm = N * N
        if constrained:
            # least-squares multipliers on the outlet plane: min |g + Jt^T lam|
            Jt = (prob.h ** 2 * prob.cons_jac(xor)).tocsc()
            cols = np.concatenate([prob.idx(0, N, np.arange(N)[:, None], np.arange(N)[None, :]).ravel(),
                                   prob.idx(1, N, np.arange(N)[:, None], np.arange(N)[None, :]).ravel()])
            Jo = Jt[:, cols].toarray()
            lam, *_ = np.linalg.lstsq(Jo.T, -g[cols], rcond=None)
            g = g + Jt.T @ lam
            cres = prob.cons(xor)
            c_rel = M.rms(cres) / M.rms(prob.t)
            c1o = face_terms(*prob.full(xor), 1, prob.h)[0][N]
            c_raw = M.rms(c1o - case["vD_faces"]["f1"][0]) / M.rms(case["vD_faces"]["f1"][0])
        else:
            c_rel = c_raw = float("nan")
        G = g.reshape(2, N, N, N)
        g_int = np.linalg.norm(G[:, :N - 1]) / gref
        g_out = np.linalg.norm(G[:, N - 1]) / gref
        g_tot = np.linalg.norm(g) / gref
        # the flux every exact minimizer of E_h must have (Kelvin identity): two-point-flux Darcy, same inlet flux
        flo = mimetic_fluxes(*prob.full(xor), prob.h)
        flk, _ = kelvin_tpfa(prob, flo["f1"][0], prob.t if constrained else None)
        vref = all_faces(case["vD_faces"])
        ev_k = M.rms(all_faces(flk) - vref) / M.rms(vref)
        ev_o = M.rms(all_faces(flo) - vref) / M.rms(vref)
        rows.append((N, g_tot, g_int, g_out, c_rel, ev_k, ev_o))
        print("CONSIST field=%s eps=%g N=%d constrained=%s g_rel(oracle)=%.3e interior(planes 1..N-1)=%.3e "
              "outlet=%.3e constraint rms rel=%.3e (vs raw f1_in: %.3e) |grad E(0)|=%.3e | e_v(Kelvin/TPFA flux)=%.3e "
              "e_v(oracle_mim)=%.3e"
              % (field, eps, N, constrained, g_tot, g_int, g_out, c_rel, c_raw, gref, ev_k, ev_o), flush=True)
    Ns = [r[0] for r in rows]
    for k, name in ((1, "g_rel"), (2, "g_interior"), (3, "g_outlet"), (4, "constraint"), (5, "e_v_kelvin"),
                    (6, "e_v_oracle_mim")):
        e = [r[k] for r in rows]
        if all(np.isfinite(e)) and all(v > 0 for v in e):
            print("CONSIST_ORDER field=%s eps=%g %s: %s orders %s" % (field, eps, name, " ".join("%.3e" % v for v in e),
                                                                      " ".join("%.2f" % o for o in M.orders(e, Ns))),
                  flush=True)


# ------------------------------------------------------------------------------------------------
def gradtest(field, eps, N, seed=7):
    _hdr("gradtest %s:%g:%d" % (field, eps, N))
    case = C.load_case(field, eps, N, verbose=False)
    rng = np.random.default_rng(seed)
    h = 1.0 / N
    ok = True

    def check(name, val, thr):
        nonlocal ok
        good = val <= thr
        ok &= good
        print("GRADTEST %-62s %.3e  (<= %.0e) %s" % (name, val, thr, "PASS" if good else "FAIL"), flush=True)

    # (a) affine labels: c = e1 exactly
    Z = np.zeros((N + 1, N, N))
    fl = mimetic_fluxes(Z, Z, h)
    check("affine labels: max|c1 - 1|, max|c2|, max|c3|",
          max(np.abs(fl["f1"] - 1).max(), np.abs(fl["f2"]).max(), np.abs(fl["f3"]).max()), 1e-13)
    # (b) div_h c_f = 0 for random labels (O(1) periodic parts)
    for amp in (0.1, 1.0):
        U1 = amp * rng.standard_normal((N + 1, N, N)); U2 = amp * rng.standard_normal((N + 1, N, N))
        fl = mimetic_fluxes(U1, U2, h)
        dv = face_divergence(fl, N)
        scale = max(np.abs(all_faces(fl)).max() / h, 1.0)
        check("random labels amp=%g: max|div_h c_f| / (max|c_f|/h)" % amp, np.abs(dv).max() / scale, 1e-14)
    for constrained in ((False, True) if not field.endswith("_ch") else (False,)):
        prob = EnergyProblem(case, constrained)
        x = 0.05 * rng.standard_normal(prob.n)
        d = rng.standard_normal(prob.n)
        g = prob.grad(x)
        gd = float(g @ d)
        Jf, w = prob.flux_jac(x)
        fl = prob.fluxes(x)
        check("grad E_h vs J^T (w c) with the exact sparse flux Jacobian", np.abs(Jf.T @ (w * all_faces(fl)) - g).max()
              / np.abs(g).max(), 1e-12)
        fd = (all_faces(prob.fluxes(x + 1e-6 * d)) - all_faces(prob.fluxes(x - 1e-6 * d))) / 2e-6
        check("flux Jacobian J d vs central FD of c_f", np.abs(Jf @ d - fd).max() / np.abs(fd).max(), 1e-8)
        if not constrained:
            flk, Ek = kelvin_tpfa(prob, fl["f1"][0])
            dk = all_faces(fl) - all_faces(flk)
            check("Kelvin identity E_h(u) - E_K = 1/2 |c(u) - c_K|_W^2 (random u, free outlet)",
                  abs(prob.energy(x) - Ek - 0.5 * float(np.sum(w * dk ** 2))) / prob.energy(x), 1e-12)
            check("TPFA Kelvin flux: max|div_h c_K| h / max|c_K|",
                  np.abs(face_divergence(flk, N)).max() * h / np.abs(all_faces(flk)).max(), 1e-12)
        for st in (1e-5, 1e-6, 1e-7):
            fd = (prob.energy(x + st * d) - prob.energy(x - st * d)) / (2 * st)
            check("dE/dx . d vs central FD of E_h (step %.0e)" % st, abs(fd - gd) / abs(gd), 1e-6)
        if constrained:
            lam = rng.standard_normal(N * N)
            jd = float(prob.consT(x, lam) @ d)
            for st in (1e-5, 1e-6, 1e-7):
                fd = (lam @ prob.cons(x + st * d) - lam @ prob.cons(x - st * d)) / (2 * st)
                check("(dC/dx)^T lam . d vs central FD of lam.C (step %.0e)" % st, abs(fd - jd) / abs(jd), 1e-6)
            Jx = prob.cons_jac(x)
            check("sparse dC/dx vs adjoint (dC/dx)^T lam", np.abs(Jx.T @ lam - prob.consT(x, lam)).max()
                  / np.abs(prob.consT(x, lam)).max(), 1e-13)
            check("sum over outlet faces of dC/dx (identically 0)", np.abs(np.asarray(Jx.sum(axis=0))).max()
                  / np.abs(Jx).max(), 1e-13)
        # colored FD Hessian vs dense FD columns (Hessian-vector products on random vectors)
        lamv = rng.standard_normal(N * N) if constrained else None
        gfun = (lambda y: prob.gradL(y, lamv)) if constrained else prob.grad
        H, ngr = hessian_fd(prob, gfun, x)
        for _ in range(2):
            v = rng.standard_normal(prob.n)
            Hv = (gfun(x + 1e-5 * v) - gfun(x - 1e-5 * v)) / 2e-5
            check("colored FD Hessian . v vs directional FD (constrained=%s)" % constrained,
                  np.linalg.norm(H @ v - Hv) / np.linalg.norm(Hv), 1e-6)
        # mode preconditioner = exact inverse of the constant-coefficient KKT/Hessian at the affine state
        qbar = float(prob.q.mean())
        P = ModePreconditioner(case, constrained, qbar)
        p0 = EnergyProblem(case, constrained, qconst=qbar, zero_inlet=True)
        x0 = np.zeros(p0.n)
        H0, _ = hessian_fd(p0, p0.grad, x0)
        ev0 = np.linalg.eigvalsh(H0.toarray()) if p0.n <= 4000 else None
        if ev0 is not None:
            r0 = np.abs(ev0) / np.abs(ev0).max()
            print("GRADTEST info: uniform-state Hessian (q = mean q, affine labels) n=%d: %d relative eigenvalues < 1e-10 "
                  "(2 N^2 = %d hourglass modes), next = %.2e, negative = %d"
                  % (p0.n, int(np.sum(r0 < 1e-10)), 2 * N * N, np.sort(r0)[int(np.sum(r0 < 1e-10))],
                     int(np.sum(ev0 < -1e-10 * np.abs(ev0).max()))), flush=True)
        # the preconditioner inverts the SHIFTED constant-coefficient operator exactly
        K0 = kkt_matrix(p0, H0 + P.shift * sp.identity(p0.n, format="csr"), x0)
        r = rng.standard_normal(K0.shape[0])
        if constrained:
            print("GRADTEST info: linearized outlet-flux rows vanishing at the uniform state (Fourier modes): %d "
                  "(xi = 0 and the (N/2, N/2) checkerboard expected)" % int(np.sum(P.dead)), flush=True)
            # the operator is singular on the dead modes: test on a right-hand side without them
            R2 = np.fft.fft2(r[p0.n:p0.n + N * N].reshape(N, N)).ravel(); R2[P.dead] = 0.0
            r[p0.n:p0.n + N * N] = np.fft.ifft2(R2.reshape(N, N)).real.ravel()
            r[-1] = 0.3
        y = P.apply(r)
        check("mode preconditioner: |(K0 + shift) P^-1 r - r| / |r| (constrained=%s)" % constrained,
              np.linalg.norm(K0 @ y - r) / np.linalg.norm(r), 1e-6)
    # oracle labels: mimetic flux close to the face fluxes
    a2, a3 = M.affine(N)
    Uor = (case["psi_or"][0] - a2, case["psi_or"][1] - a3)
    flo = mimetic_fluxes(Uor[0], Uor[1], h)
    vref = all_faces(case["vD_faces"])
    print("GRADTEST info: e_v(mimetic flux of the oracle labels vs vD_faces) = %.3e at N=%d"
          % (M.rms(all_faces(flo) - vref) / M.rms(vref), N), flush=True)
    print("GRADTEST %s" % ("ALL PASS" if ok else "SOME FAILED"), flush=True)
    return ok


# ------------------------------------------------------------------------------------------------
def main(argv):
    opts = {"spectrum": None, "consistency": False, "free": False, "gradtest": False, "out": RAW_DIR,
            "solver": "auto", "grids": (16, 32, 48), "cont": True, "tol": 1e-13, "start": "inlet", "maxit": 60, "hessian": "fd"}
    specs = []
    it = iter(argv)
    for a in it:
        if a == "--spectrum":
            opts["spectrum"] = int(next(it))
        elif a == "--consistency":
            opts["consistency"] = True
        elif a == "--free-outlet":
            opts["free"] = True
        elif a == "--gradtest":
            opts["gradtest"] = True
        elif a == "--out":
            opts["out"] = os.path.abspath(next(it))
        elif a == "--solver":
            opts["solver"] = next(it)
        elif a == "--grids":
            opts["grids"] = tuple(int(v) for v in next(it).split(","))
        elif a == "--no-continuation":
            opts["cont"] = False
        elif a == "--start":
            opts["start"] = next(it)
        elif a == "--hessian":
            opts["hessian"] = next(it)
        elif a == "--maxit":
            opts["maxit"] = int(next(it))
        elif a == "--tol":
            opts["tol"] = float(next(it))
        elif a.startswith("--"):
            print(__doc__); return 2
        else:
            specs.append(a)
    if not specs:
        print(__doc__); return 2
    print("candidate_ii.py python=%s numpy=%s scipy=%s argv=%s" % (sys.version.split()[0], np.__version__,
                                                                 __import__("scipy").__version__, " ".join(argv)),
          flush=True)
    rc = 0
    for spec in specs:
        field, eps, N = C.parse_spec(spec)
        if opts["consistency"]:
            consistency(field, eps, opts["grids"], opts["free"])
            continue
        if opts["spectrum"] is not None:
            if N is not None and N != opts["spectrum"]:
                print("note: --spectrum %d overrides N=%d of %s" % (opts["spectrum"], N, spec))
            N = opts["spectrum"]
        if N is None:
            print("case spec needs N: %s" % spec); return 2
        if opts["gradtest"]:
            rc |= 0 if gradtest(field, eps, N) else 1
            continue
        run_case(field, eps, N, free_outlet=opts["free"], solver=opts["solver"], tol=opts["tol"],
                 continuation=opts["cont"], spectrum_out=(opts["out"] if opts["spectrum"] is not None else None),
                 start=opts["start"], maxit=opts["maxit"], hessian=opts["hessian"])
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
