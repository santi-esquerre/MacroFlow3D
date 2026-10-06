#!/usr/bin/env python3
"""SF-29 N3 + C-ii -- candidate (ii): dissipation-energy fit with discretely solenoidal (mimetic) face fluxes.

C-ii (corrective node) changed two things relative to N3, both prescribed by the N3 audit:
  * default energy --energy q1: the energy of the POINTWISE in-cell product of the trilinear (Q1) label
    interpolants (section "Q1 energy" below), which removes the 2 N^2 hourglass kernel of the N3 form; the N3
    edge-averaged Whitney energy is kept as --energy whitney (control, cand=ii_whitney);
  * D-3 completion: in periodic-flow cases the two scalar rows "mean transverse flux = 0" are added to the outlet
    flux constraint (section "D-3 rows"); --no-meanflux restores the N3 constraint set (cand=..._nomf).

Usage (run from this directory):
  python3 candidate_ii.py field:eps:N [...]                 solve + metrics (CASE lines cand=ii and cand=oracle_mim)
  python3 candidate_ii.py --free-outlet field:eps:N [...]   periodic cases WITHOUT the outlet constraints (cand=ii_free)
  python3 candidate_ii.py --gradtest field:eps:N [...]      Whitney div_h curl_h = 0; Q1 checks (a) pointwise div c,
                                                            (b) face averages = Whitney c_f, (c) exact gradient vs FD,
                                                            (d) k = 1 Hessian kernel of both forms + checkerboard
                                                            curvature; constraint Jacobians vs FD, colored FD Hessian
                                                            vs dense columns, mode preconditioner
  python3 candidate_ii.py --consistency field:eps [...]     |grad L| with least-squares multipliers and ALL constraint
                                                            residuals AT THE ORACLE LABELS, N in --grids, orders
  python3 candidate_ii.py --spectrum M field:eps[:M] [...]  solve at N = M, dense Hessian (reduced Hessian + KKT if
                                                            constrained) relative singular values
Options: --energy q1|whitney (default q1), --q center|q1 (Q1 energy: q at the cell center (default) or the
         trilinear interpolant of the vertex q, cand=ii_qq1), --no-meanflux, --out DIR (spectrum files, default
         ../raw), --solver auto|direct|gmres (auto: splu for N <= 32), --grids 16,32,48, --no-continuation,
         --tol 1e-13, --maxit 60 (final stage; intermediate stages maxit/2),
         --hessian fd|gn (fd: colored central FD Hessian, the reference method; gn: Gauss-Newton J^T W J, Whitney only,
         cand=..._gn), --start inlet|zero|oracle (initial guess; oracle = diagnostic local-convergence start,
         cand=..._from_oracle)

Formulation (UNDERSTAND record section 2.2).  E[psi1, psi2] = 1/2 int q |grad psi1 x grad psi2|^2, q = 1/k.
Its Euler-Lagrange equations are equation (14) (same-index pairing); the natural condition at a free face
x1 = 1 is c x e1 = 0 (constant head).  Constant-head cases (`*_ch`): inlet Dirichlet labels, outlet free (no
constraint).  Periodic-flow cases: inlet Dirichlet labels + outlet flux constraint c1(1, x2, x3) = f1_in + the two
D-3 rows (Kelvin principle with the Neumann data of the periodic flow on both faces and zero mean transverse flux;
deviation D-3).

Q1 energy (default).  In every cell the labels are the trilinear (Q1) interpolants of the 8 vertex values,
psi1^h = x2 + U1^h, psi2^h = x3 + U2^h; x2 and x3 are themselves trilinear, so the affine parts are exact (corner
value = periodic part + h d2 for psi1, + h d3 for psi2, d the corner offset; the label jump across the periodic
boundary never enters).  c(x) = grad psi1^h x grad psi2^h pointwise inside the cell.  This field is
  - exactly divergence-free inside every cell (c = curl(psi1^h grad psi2^h) of a polynomial),
  - normal-continuous across faces (c . n involves only tangential derivatives of the traces, and the trace of a Q1
    function on a face depends only on the 4 face vertices),
  - with face averages of c . n EXACTLY equal to the Whitney c_f below (the face integral of the bilinear
    Jacobian det d(psi1, psi2)/d(a, b) is the boundary circulation of psi1 d psi2 with psi linear along the edges =
    the trapezoidal circulation), so vD_faces comparisons, metrics and the outlet flux constraint are unchanged.
  It is NOT the collocated pointwise product at the VERTICES (2026-10-02 note section 4, unsound: not discretely
  solenoidal); that object is not used anywhere.
  E_h = 1/2 sum_cells q_cell int_cell |c(x)|^2 dx with the tensor 3x3x3 Gauss-Legendre rule per cell.  Polynomial
  degree per direction (x1, x2, x3): d1 psi ~ (0,1,1), d2 psi ~ (1,0,1), d3 psi ~ (1,1,0), so c1 ~ (2,1,1),
  c2 ~ (1,2,1), c3 ~ (1,1,2) and |c|^2 has degree <= 4 in each direction; 3-point Gauss is exact to degree 5 per
  direction, so the rule integrates |c|^2 EXACTLY (also with --q q1, degree 5).  q_cell = 1/k at the cell center
  (analytic k), or (--q q1) the trilinear interpolant of q at the vertices.  Exact gradient = adjoint of the
  Gauss-point chain: dE = sum_p W_p (dg1 . (g2 x c) + dg2 . (c x g1)), W_p = h^3 w_p q_p, dg_i = D du_i / h.
  At k = 1, u = 0 the Hessian is the Q1 discretization of int (d2 u1 + d3 u2)^2 + (d1 u1)^2 + (d1 u2)^2, positive
  definite with inlet Dirichlet data (--gradtest (d)); the checkerboard directions of the Whitney kernel have
  O(1/h^2) curvature.  Coupling stencil of a vertex: the 27 vertices of its 8 cells (same coloring as before).

D-3 rows (periodic-flow cases; constrained).  Cm = (h^2 sum_{j, m3} c2[j, 0, m3], h^2 sum_{j, m2} c3[j, m2, 0]) = 0:
the total flux through the x2-plane m2 = MF_PLANE = 0 and the x3-plane m3 = 0 (relative to the unit mean
x1-flux).  For a discretely solenoidal field the plane fluxes satisfy S2(m+1) - S2(m) = -h^2 sum_m3 (c1[N, m, :] -
c1[0, m, :]) (cell balance summed over the slab row m), so they are plane-independent iff the outlet and inlet
x1-fluxes have equal row sums; here c1[N] = t (constraint, from the reference faces) and c1[0] = the mimetic flux
of the inlet labels, which differ at O(h^2): the plane fluxes are plane-dependent at that level, and the rows are
imposed on plane 0 (MEANFLUX lines report the spread over planes, the identity defect and the row mismatch).  The
rows are additional Lagrange-multiplier rows of the bordered KKT system (multipliers eta = head jumps across the
periodic boundary, i.e. minus the mean transverse head gradient).

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

Whitney energy (--energy whitney, the N3 form; control).  E_h = 1/2 h^3 sum_cells q_cell sum_dir (1/2)(c_{f-}^2 + c_{f+}^2): per cell, per direction the
mean of the squared fluxes of its two opposite faces (a trapezoidal rule in the normal direction), q_cell = 1/k at
the cell center with analytic k (reference.LogConductivity of the case's ref.lk).  Equivalently
E_h = 1/2 h^3 sum_faces omega_f c_f^2, omega_f = 1/2 sum of q over the (one or two) cells adjacent to f.

Exact discrete gradient of the Whitney energy (adjoint chain).  dE_h/dc_f = h^3 omega_f c_f =: W_f; dE_h/dpsi(v) = sum_f W_f dc_f/dpsi(v),
with dc_f/dpsi1(P11) = (psi2(P01) - psi2(P10))/(2h^2), etc. (scatter by np.roll; adjoint of the corner gathers).

Outlet constraint (periodic cases).  C(u) = c1[N] - t on the N^2 outlet x1-faces, t = f1_in - mean(f1_in) + 1,
f1_in = vD_faces['f1'][0].  sum_faces c1[N] = N^2 IDENTICALLY (the periodic parts telescope), so one constraint
is redundant and f1_in must have mean exactly 1 to be feasible: the mean of f1_in (1 - O(1e-10), reference /
GL3 level, printed) is replaced by 1.  Lagrange multipliers with a bordered KKT system (Jm: the two D-3 rows):
    [ H_L   Jt^T  0   Jm^T ] [dx  ]     [ grad E_h + Jt^T lam + Jm^T eta ]
    [ Jt    0     1   0    ] [dlam] = - [ Ct + mu 1                      ],   Jt = h^2 dC/dx, Ct = h^2 C (scaling),
    [ 0     1^T   0   0    ] [dmu ]     [ sum lam                        ]
    [ Jm    0     0   0    ] [deta]     [ Cm                             ]
(sum lam = 0 fixes the multiplier null direction 1; mu absorbs the redundant row and is 0 at the solution).
H_L is the Hessian of the Lagrangian (the constraints are bilinear in the labels).  Multiplier vector
y = [lam, mu, eta]; c_rel = max(max|C| / max|t|, max|Cm|).

Nonlinear method.  Newton on the stationarity (KKT) system; Hessian assembled by colored central finite
differences of the exact gradient (delta = 1e-6; the coupling stencil of a vertex is the 27-point set of vertices
sharing a cell (Q1) or the 19-point set sharing a face (Whitney), coloring by ((j-1) mod 3, m2 mod p, m3 mod p),
p >= 3 the smallest divisor of N, 2 fields, symmetrized; same-colored columns are >= 3 apart in every direction,
so their 27-point neighbourhoods are disjoint).  Linear solve: splu (N <= 32) or GMRES (N > 32) preconditioned by
the EXACT inverse of the
constant-coefficient problem (q = mean q, affine labels): transversally translation-invariant, so per transverse
Fourier mode a 2x2-block tridiagonal system in x1 (block Thomas, vectorized over modes) bordered by the one outlet
multiplier of that mode (shifted by max(nu, 1e-6) x mean diag: the uniform-state Whitney Hessian is singular on
the hourglass modes, see below); the two D-3 multipliers are passed through (identity block; a rank-2 bordering
left to GMRES).  Globalization: Levenberg-Marquardt shift nu of the Hessian block (adapted) and
backtracking; merit E_h (Armijo, descent required) for the unconstrained problem while g_rel > 1e-6, |F|
otherwise.  Amplitude continuation s = 0.25 -> 0.5 -> 1.0 (homotopy: ln k -> s ln k, inlet periodic parts ->
s U^0, outlet target -> 1 + s (t - 1)), warm starts, initial guess = constant x1-extension of the inlet labels;
intermediate stages to g_rel <= 1e-8, the final stage to g_rel <= --tol (1e-13), or stop on stagnation
(< 2x reduction of g_rel over 15 iterations, or over 5 iterations below 1e-10), or maxit.

    g_rel = |grad_x L(x)| / |grad E_h(x0)|,  x0 = unknowns 0 (inlet labels in place, s = 1);  r_F := g_rel.

Discrete Kelvin identity (diagnostic, KELVIN line; Whitney energy only).  For every label pair with the same inlet
flux (and outlet flux and D-3 rows, constrained case) E_h(u) = E_K + 1/2 sum_f w_f (c_f(u) - c_K,f)^2 EXACTLY, with c_K the two-point-flux
(harmonic face mean of k) Darcy flux minimizing E_h over ALL discretely solenoidal face fluxes (kelvin_tpfa;
verified in --gradtest).  So minimizing E_h over labels = least-squares fit of the mimetic flux to c_K; any exact
minimizer has c = c_K and e_v = e_v(c_K).  With the D-3 rows c_K carries two head jumps across the periodic
boundary (kelvin_tpfa(meanflux=True)).  For the Q1 energy the identity does NOT apply (its minimizer is a different,
Q1-based Darcy discretization); the KELVIN line then reports E(oracle) - E_min and the TPFA flux only for reference.

Hourglass deficiency of the Whitney energy (finding of N3, MAJOR-1; --gradtest (d) prints it for both forms).  At the uniform state
(affine labels) the linearized face fluxes only see the x3-AVERAGE of the x2-edge differences of U1 (and the
analogous averages), so U1 = (-1)^m3 f(j, m2) and U2 = (-1)^m2 g(j, m3) change no face flux to first order:
2 N^2 exact null modes of the Hessian (288 at 12^3), and the linearized outlet-flux row of the (N/2, N/2)
checkerboard vanishes.  Off the uniform state they are lifted only by O(|grad u|) couplings.  The Q1 energy sees
the in-cell variation of c and has no such kernel; the vanishing outlet-flux row of the (N/2, N/2) checkerboard is a
property of the Whitney outlet flux (constraint), independent of the energy form, and remains.

Metrics (shared CASE format; cand=ii (Q1) / ii_whitney, suffixes _qq1, _free, _nomf, _from_oracle, _gn): e_v = RMS over ALL faces of the three families of
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
OFFSETS27 = list(itertools.product((-1, 0, 1), repeat=3))
ENERGY_FORMS = ("q1", "whitney")
MF_PLANE = 0                                   # plane m2 = 0 (x2-faces) / m3 = 0 (x3-faces) of the D-3 rows

# Q1 (trilinear) cell element: corners d = (d1, d2, d3) in {0, 1}^3, local coordinates xi in [0, 1]^3
CORNERS = tuple(itertools.product((0, 1), repeat=3))
_GL3_X = 0.5 * (1.0 + np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]))
_GL3_W = np.array([5.0, 8.0, 5.0]) / 18.0


def q1_tables(xi):
    """Trilinear shape functions N_d and reference derivatives dN_d/dxi_a at local points xi (P, 3) in [0, 1]^3,
    corners in CORNERS order.  Returns Nv (P, 8), D (P, 8, 3)."""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    P = xi.shape[0]
    Nv = np.ones((P, 8)); D = np.ones((P, 8, 3))
    for k, d in enumerate(CORNERS):
        for a in range(3):
            fa = xi[:, a] if d[a] else 1.0 - xi[:, a]
            Nv[:, k] *= fa
            for b in range(3):
                D[:, k, b] *= (1.0 if d[a] else -1.0) if a == b else fa
    return Nv, D


# tensor 3x3x3 Gauss-Legendre rule on the unit cell (weights sum to 1; exact for degree <= 5 per direction)
GQ_PTS = np.array(list(itertools.product(_GL3_X, repeat=3)))
GQ_W = np.array([a * b * c for a, b, c in itertools.product(_GL3_W, repeat=3)])
GQ_N, GQ_D = q1_tables(GQ_PTS)


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
# Q1 in-cell product: c(x) = grad psi1^h x grad psi2^h of the trilinear label interpolants (cells j = 0..N-1)
def _cgather(V, d, N):
    """Corner d of every cell (j, m2, m3), j = 0..N-1, from a vertex array (N+1, N, N) (periodic in m2, m3)."""
    A = V[d[0]:d[0] + N]
    if d[1] or d[2]:
        A = np.roll(A, (-d[1], -d[2]), axis=(1, 2))
    return A


def _cscatter(G, d, out, N):
    """Adjoint of _cgather: add a cell array (N, N, N) into the vertex array (N+1, N, N)."""
    if d[1] or d[2]:
        G = np.roll(G, (d[1], d[2]), axis=(1, 2))
    out[d[0]:d[0] + N] += G


def q1_corner_values(U1, U2, h):
    """Corner values (8, N, N, N) of psi1 = x2 + U1 and psi2 = x3 + U2 per cell, up to the cell constants
    h m2 (psi1) and h m3 (psi2), which do not enter any gradient: the periodic part plus the exact affine offset
    h d2 (psi1) / h d3 (psi2) of the corner (the periodic wrap of the stored arrays never enters)."""
    N = U1.shape[1]
    u1 = np.stack([_cgather(U1, d, N) + h * d[1] for d in CORNERS])
    u2 = np.stack([_cgather(U2, d, N) + h * d[2] for d in CORNERS])
    return u1, u2


def _cross(a, b):
    """Cross product along axis 1 of (P, 3, ...) arrays."""
    return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                     a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], axis=1)


def q1_fields(U1, U2, h, D=GQ_D):
    """grad psi1^h, grad psi2^h and c = grad psi1^h x grad psi2^h at the local points of D (P, 8, 3) in every
    cell: arrays (P, 3, N, N, N)."""
    u1, u2 = q1_corner_values(U1, U2, h)
    g1 = np.tensordot(D, u1, axes=([1], [0])) / h
    g2 = np.tensordot(D, u2, axes=([1], [0])) / h
    return g1, g2, _cross(g1, g2)


def q1_c_at(U1, U2, h, xi):
    """c(x) at the local points xi (P, 3) of every cell: (P, 3, N, N, N)."""
    return q1_fields(U1, U2, h, q1_tables(xi)[1])[2]


def q1_energy(U1, U2, h, qg):
    """E_h = 1/2 sum_cells int_cell q |c|^2 with the 27-point Gauss rule; qg (27 or 1, N, N, N)."""
    _, _, c = q1_fields(U1, U2, h)
    wq = (h ** 3) * GQ_W[:, None, None, None] * qg
    return 0.5 * float(np.sum(wq * np.sum(c * c, axis=1)))


def q1_gradient(U1, U2, h, qg):
    """Exact gradient of q1_energy with respect to the full vertex arrays (adjoint of the Gauss-point chain):
    dE = sum_p W_p c . (dg1 x g2 + g1 x dg2) = sum_p W_p (dg1 . (g2 x c) + dg2 . (c x g1)),
    W_p = h^3 w_p q_p; dg_i = D du_i / h; scatter of the corner adjoints to the vertices."""
    N = U1.shape[1]
    g1, g2, c = q1_fields(U1, U2, h)
    W = ((h ** 3) * GQ_W[:, None, None, None] * qg)[:, None]
    a1 = W * _cross(g2, c)
    a2 = W * _cross(c, g1)
    du1 = np.tensordot(GQ_D, a1, axes=([0, 2], [0, 1])) / h          # (8, N, N, N)
    du2 = np.tensordot(GQ_D, a2, axes=([0, 2], [0, 1])) / h
    G1 = np.zeros_like(U1); G2 = np.zeros_like(U2)
    for k, d in enumerate(CORNERS):
        _cscatter(du1[k], d, G1, N)
        _cscatter(du2[k], d, G2, N)
    return G1, G2


# ------------------------------------------------------------------------------------------------
class EnergyProblem:
    """Discrete energy (form q1 or whitney), gradient, outlet constraint and the two D-3 mean-transverse-flux
    rows for one case at homotopy stage s.  Multiplier vector y = [lam (N^2), mu, eta (ne)], ne = 2 when the
    D-3 rows are active (constrained periodic case and meanflux), else 0."""

    def __init__(self, case, constrained, s=1.0, qconst=None, zero_inlet=False, form="q1", qmode="center",
                 meanflux=True):
        meta = case["meta"]
        self.case = case
        self.N = N = int(meta["N"]); self.h = h = 1.0 / N
        self.N3 = N ** 3; self.n = 2 * N ** 3
        self.constrained = bool(constrained)
        if form not in ENERGY_FORMS or qmode not in ("center", "q1"):
            raise ValueError("energy form %r / q mode %r" % (form, qmode))
        self.form = form; self.qmode = qmode
        self.ne = 2 if (self.constrained and meanflux) else 0
        self.nmult = (N * N + 1 + self.ne) if self.constrained else 0
        self.s = float(s)
        xc = (np.arange(N) + 0.5) * h
        X1, X2, X3 = np.meshgrid(xc, xc, xc, indexing="ij")
        self.lnk_cell = case["ref"].lk.lnk(X1, X2, X3)
        q = np.exp(-self.s * self.lnk_cell) if qconst is None else np.full((N, N, N), float(qconst))
        self.q = q
        w1 = np.zeros((N + 1, N, N)); w1[:N] += 0.5 * q; w1[1:] += 0.5 * q
        self.omega = {1: w1, 2: 0.5 * (q + np.roll(q, 1, axis=1)), 3: 0.5 * (q + np.roll(q, 1, axis=2))}
        # q at the 27 Gauss points of every cell (Q1 energy): the cell-center value, or (qmode q1) the trilinear
        # interpolant of the vertex values q = exp(-s ln k) (analytic ln k at the vertices)
        if qmode == "q1":
            qv = np.exp(-self.s * case["lnk"]) if qconst is None else np.full((N + 1, N, N), float(qconst))
            self.qg = np.tensordot(GQ_N, np.stack([_cgather(qv, d, N) for d in CORNERS]), axes=([1], [0]))
        else:
            self.qg = q[None]
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
        if self.form == "q1":
            U1, U2 = self.full(x)
            return q1_energy(U1, U2, self.h, self.qg)
        fl = self.fluxes(x)
        h3 = self.h ** 3
        return 0.5 * h3 * sum(float(np.sum(self.omega[f] * fl["f%d" % f] ** 2)) for f in (1, 2, 3))

    def grad(self, x):
        U1, U2 = self.full(x)
        if self.form == "q1":
            G1, G2 = q1_gradient(U1, U2, self.h, self.qg)
            return np.concatenate([G1[1:].ravel(), G2[1:].ravel()])
        h3 = self.h ** 3
        W = {f: h3 * self.omega[f] * face_terms(U1, U2, f, self.h)[0] for f in (1, 2, 3)}
        G1, G2 = adjoint(U1, U2, W, self.h)
        return np.concatenate([G1[1:].ravel(), G2[1:].ravel()])

    # D-3 completion: zero mean transverse flux, h^2 x (sum of c2 over the x2-faces of the plane m2 = MF_PLANE,
    # sum of c3 over the x3-faces of the plane m3 = MF_PLANE); scaled like Jt (plane flux integral)
    def plane_fluxes(self, x):
        """h^2 sum of c2 over every x2-plane m2 and of c3 over every x3-plane m3: arrays (N,), (N,)."""
        U1, U2 = self.full(x)
        h2 = self.h ** 2
        c2 = face_terms(U1, U2, 2, self.h)[0]; c3 = face_terms(U1, U2, 3, self.h)[0]
        return h2 * c2.sum(axis=(0, 2)), h2 * c3.sum(axis=(0, 1))

    def cons_mean(self, x):
        S2, S3 = self.plane_fluxes(x)
        return np.array([S2[MF_PLANE], S3[MF_PLANE]])

    def cons_mean_T(self, x, eta):
        U1, U2 = self.full(x)
        N, h2 = self.N, self.h ** 2
        W2 = np.zeros((N, N, N)); W2[:, MF_PLANE, :] = h2 * eta[0]
        W3 = np.zeros((N, N, N)); W3[:, :, MF_PLANE] = h2 * eta[1]
        G1, G2 = adjoint(U1, U2, {2: W2, 3: W3}, self.h)
        return np.concatenate([G1[1:].ravel(), G2[1:].ravel()])

    def cons_mean_jac(self, x):
        rows = [self.cons_mean_T(x, e) for e in (np.array([1.0, 0.0]), np.array([0.0, 1.0]))]
        return sp.csr_matrix(np.vstack(rows))

    def split(self, y):
        """Multiplier vector -> (lam, mu, eta)."""
        nm = self.N * self.N
        return y[:nm], float(y[nm]), y[nm + 1:nm + 1 + self.ne]

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

    # Lagrangian gradient (scaled multipliers: Jt = h^2 dC/dx; the D-3 rows are already scaled)
    def gradL(self, x, y):
        g = self.grad(x)
        if self.constrained and y is not None:
            lam, _, eta = self.split(y)
            g = g + self.h ** 2 * self.consT(x, lam)
            if self.ne:
                g = g + self.cons_mean_T(x, eta)
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
                    for o in (OFFSETS27 if prob.form == "q1" else OFFSETS19):
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

    def __init__(self, case, constrained, qbar, shift_rel=1e-6, form="q1", ne=0):
        prob = EnergyProblem(case, constrained, s=1.0, qconst=qbar, zero_inlet=True, form=form, meanflux=False)
        N = prob.N; self.N = N; self.constrained = constrained; self.n = prob.n
        self.ne = ne      # D-3 multiplier rows: passed through (identity block; a rank-2 bordering left to GMRES)
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
        return np.concatenate([x, lr, [mu], r[n + nm + 1:n + nm + 1 + self.ne]])


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
def kkt_residual(prob, x, y):
    """Stationarity / KKT residual F, the Lagrangian gradient g, the outlet constraint c (raw) and the D-3 rows cm
    (scaled plane fluxes) at (x, y)."""
    g = prob.gradL(x, y)
    if not prob.constrained:
        return g, g, None, None
    lam, mu, _ = prob.split(y)
    c = prob.cons(x)
    parts = [g, prob.h ** 2 * c + mu, np.array([lam.sum()])]
    cm = None
    if prob.ne:
        cm = prob.cons_mean(x)
        parts.append(cm)
    return np.concatenate(parts), g, c, cm


def kkt_matrix(prob, H, x):
    """Bordered KKT matrix [[H, Jt^T, 0, Jm^T], [Jt, 0, 1, 0], [0, 1^T, 0, 0], [Jm, 0, 0, 0]] (Jm: D-3 rows)."""
    if not prob.constrained:
        return H
    nm = prob.N ** 2
    Jt = (prob.h ** 2) * prob.cons_jac(x)
    one = sp.csr_matrix(np.ones((nm, 1)))
    if not prob.ne:
        return sp.bmat([[H, Jt.T, None], [Jt, None, one], [None, one.T, None]], format="csc")
    Jm = prob.cons_mean_jac(x)
    return sp.bmat([[H, Jt.T, None, Jm.T], [Jt, None, one, None], [None, one.T, None, None],
                    [Jm, None, None, None]], format="csc")


def constraint_rel(prob, c, cm):
    """max(max|C| / max|t|, max|D-3 rows|) (the D-3 rows are plane fluxes relative to the unit mean flux)."""
    if c is None:
        return 0.0
    r = np.abs(c).max() / np.abs(prob.t).max()
    return max(r, float(np.abs(cm).max())) if cm is not None else r


def _solve_lin(K, rhs, solver, prec):
    if solver == "direct":
        return spla.splu(K.tocsc()).solve(rhs), "splu"
    Mop = spla.LinearOperator(K.shape, matvec=prec.apply)
    d, info, nit = _gmres(K, rhs, Mop, rtol=1e-10)
    lres = np.linalg.norm(K @ d - rhs) / np.linalg.norm(rhs)
    return d, "gmres its=%d info=%d linres=%.1e" % (nit, info, lres)


def newton_stage(prob, x, y, gref, tol, ctol, solver, case, maxit=40, label="", hessian="fd"):
    """Newton on the stationarity / KKT system, globalized by a Levenberg-Marquardt shift of the Hessian
    block (H + nu dmean I, nu adapted: /10 after a full step, x10 after a rejected trial; nu -> 0 gives the
    pure Newton step) and backtracking (<= 3 halvings per trial).  Merit: E_h with Armijo while
    g_rel > 1e-6 in the unconstrained case (descent direction required: avoids saddles), |F| otherwise
    (and always for the KKT system).  y = multiplier vector [lam, mu, eta] (None if unconstrained)."""
    hist = []
    F, g, c, cm = kkt_residual(prob, x, y)
    nF = np.linalg.norm(F)
    E = prob.energy(x)
    its = 0
    status = "maxit"
    nu = 1e-4
    n = prob.n
    for it in range(maxit + 1):
        grel = np.linalg.norm(g) / gref
        crel = constraint_rel(prob, c, cm)
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
            if prob.form != "whitney":
                raise ValueError("--hessian gn is defined only for --energy whitney (Kelvin identity)")
            # Gauss-Newton: E_h = E_K + 1/2 |c(u) - c_K|_W^2 exactly (Kelvin), so J^T W J is the Hessian at a
            # zero-residual minimizer; the (bilinear) constraint curvature is added exactly.
            Jf, w = prob.flux_jac(x)
            H = (Jf.T @ sp.diags(w) @ Jf).tocsr()
            ngr = 0
            if prob.constrained:
                Hc, ngr = hessian_fd(prob, lambda z: prob.gradL(z, y) - prob.grad(z), x)
                H = (H + Hc).tocsr()
        else:
            gfun = (lambda z: prob.gradL(z, y)) if prob.constrained else prob.grad
            H, ngr = hessian_fd(prob, gfun, x)
        dmean = float(np.mean(np.abs(H.diagonal())))
        t1 = time.time()
        accepted = False
        use_E = (not prob.constrained) and grel > 1e-6
        trial = 0
        for trial in range(12):
            Hs = (H + (nu * dmean) * sp.identity(n, format="csr")) if nu > 0 else H
            K = kkt_matrix(prob, Hs, x)
            prec = ModePreconditioner(case, prob.constrained, float(prob.q.mean()), shift_rel=max(nu, 1e-6),
                                      form=prob.form, ne=prob.ne) if solver == "gmres" else None
            d, lin = _solve_lin(K, -F, solver, prec)
            gd = float(g @ d[:n])
            alpha = 1.0
            for _ in range(4):
                xn = x + alpha * d[:n]
                yn = (y + alpha * d[n:]) if prob.constrained else y
                Fn, gn, cn, cmn = kkt_residual(prob, xn, yn)
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
        x, y = xn, yn
        F, g, c, cm, nF, E = Fn, gn, cn, cmn, nFn, En
        its += 1
    return x, y, its, status, hist


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


def solve_case(case, constrained, solver="auto", tol=1e-13, continuation=True, start="inlet", maxit=60, hessian="fd",
               fopts=None):
    fopts = dict(fopts or {})
    N = int(case["meta"]["N"])
    if solver == "auto":
        solver = "direct" if N <= 32 else "gmres"
    stages = STAGES if (continuation and start != "oracle") else (1.0,)
    prob1 = EnergyProblem(case, constrained, s=1.0, **fopts)
    gref = np.linalg.norm(prob1.grad(np.zeros(prob1.n)))
    if gref == 0.0:
        # the inlet-extended affine state is already stationary (k = 1, uniform field): g_rel is then the
        # absolute gradient norm (stated in the output)
        print("note: |grad E_h(x0)| = 0 exactly (uniform state); g_rel reported as the absolute |grad L|", flush=True)
        gref = 1.0
    y = np.zeros(prob1.nmult) if constrained else None
    tot = 0; history = []
    t0 = time.time()
    status = None
    x = None
    for s in stages:
        prob = prob1 if s == 1.0 else EnergyProblem(case, constrained, s=s, **fopts)
        if x is None:
            x = initial_guess(prob, case, start)
        stol = tol if s == 1.0 else STAGE_TOL
        x, y, its, status, hist = newton_stage(prob, x, y, gref, stol, 1e-12 if s == 1.0 else 1e-8,
                                               solver, case, maxit=maxit if s == 1.0 else maxit // 2,
                                               label="s=%.2f" % s, hessian=hessian)
        tot += its; history.append((s, its, status, hist))
    return prob1, x, y, tot, status, history, gref, time.time() - t0, solver


# ------------------------------------------------------------------------------------------------
# discrete Kelvin reference (diagnostic): the minimizer of E_h over ALL discretely solenoidal face fluxes
def kelvin_tpfa(prob, inlet_flux, outlet_flux=None, meanflux=False):
    """min 1/2 h^3 sum omega c^2 s.t. div_h c = 0 per cell, c1[0] = inlet_flux, c1[N] = outlet_flux (or free),
    and (meanflux) zero total flux through the x2-faces of the plane m2 = MF_PLANE and the x3-faces of m3 = MF_PLANE.
    Stationarity: c_f = (p_L - p_R + J_a [f in the D-3 plane]) / (h omega_f) (two-point flux with the harmonic face
    mean of k; J_2, J_3 = head jumps across the periodic boundary, the multipliers of the D-3 rows); free outlet
    <-> p = 0 outside.  Returns the face fluxes and their Whitney E_h.  For the Whitney energy every feasible label
    pair has E_h >= this value with equality iff c_f = these (Kelvin identity); for the Q1 energy this is only the
    TPFA Darcy flux of the same data (a different discretization, no identity)."""
    N, h = prob.N, prob.h
    om = prob.omega
    nc = N ** 3
    nx = nc + (2 if meanflux else 0)
    cid = np.arange(nc).reshape(N, N, N)
    rows, cols, vals = [], [], []
    b = np.zeros(nx)

    def add(i, j, v):
        i = np.asarray(i); j = np.broadcast_to(j, i.shape)
        rows.append(i.ravel()); cols.append(j.ravel()); vals.append(np.broadcast_to(v, i.shape).ravel())
    T1 = 1.0 / (h * om[1])
    L = cid[:-1]; R = cid[1:]; T = T1[1:N]                 # interior x1-faces j = 1..N-1
    add(L, L, T); add(L, R, -T); add(R, R, T); add(R, L, -T)
    for fam, ax in ((2, 1), (3, 2)):                       # face m between cells m-1 (L) and m (R)
        T = 1.0 / (h * om[fam])
        R = cid; L = np.roll(cid, 1, axis=ax)
        add(L, L, T); add(L, R, -T); add(R, R, T); add(R, L, -T)
        if meanflux:
            jx = nc + fam - 2
            sel = (slice(None), MF_PLANE, slice(None)) if ax == 1 else (slice(None), slice(None), MF_PLANE)
            Ls, Rs, Ts = L[sel], R[sel], T[sel]
            add(Ls, jx, Ts); add(Rs, jx, -Ts)              # J column in the cell balances
            add(np.full(Ls.shape, jx), Ls, Ts); add(np.full(Rs.shape, jx), Rs, -Ts)   # sum of plane fluxes = 0
            add(np.array([jx]), jx, Ts.sum())
    b[cid[0].ravel()] += inlet_flux.ravel()
    if outlet_flux is None:
        add(cid[N - 1], cid[N - 1], T1[N])
    else:
        b[cid[N - 1].ravel()] -= outlet_flux.ravel()
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx, nx))
    if outlet_flux is not None:                            # pure Neumann: pin one pressure value
        A = A.tolil(); A[0, :] = 0.0; A[0, 0] = 1.0; A = A.tocsr(); b[0] = 0.0
    sol = spla.spsolve(A.tocsc(), b)
    p = sol[:nc].reshape(N, N, N)
    f1 = np.empty((N + 1, N, N)); f1[0] = inlet_flux
    f1[1:N] = (p[:-1] - p[1:]) * T1[1:N]
    f1[N] = p[N - 1] * T1[N] if outlet_flux is None else outlet_flux
    f2 = (np.roll(p, 1, axis=1) - p) / (h * om[2])
    f3 = (np.roll(p, 1, axis=2) - p) / (h * om[3])
    if meanflux:
        f2[:, MF_PLANE, :] += sol[nc] / (h * om[2][:, MF_PLANE, :])
        f3[:, :, MF_PLANE] += sol[nc + 1] / (h * om[3][:, :, MF_PLANE])
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


def cand_name(form, free_outlet, ch, meanflux, qmode, start="inlet", hessian="fd"):
    cand = "ii" if form == "q1" else "ii_whitney"
    if qmode == "q1" and form == "q1":
        cand += "_qq1"
    if free_outlet and not ch:
        cand += "_free"
    elif not ch and not meanflux:
        cand += "_nomf"
    if start == "oracle":
        cand += "_from_oracle"
    if hessian == "gn":
        cand += "_gn"
    return cand


def meanflux_line(tag, field, eps, N, prob, x, vf):
    """D-3 diagnostics: plane fluxes h^2 sum c2 / h^2 sum c3 per transverse plane (value at MF_PLANE, spread over
    planes, mean), the reference's, and the per-row inlet/outlet x1-flux mismatch that makes them plane-dependent:
    S2(m+1) - S2(m) = -h^2 sum_m3 (c1[N, m, :] - c1[0, m, :]) (cell balance summed over the slab row m)."""
    S2, S3 = prob.plane_fluxes(x)
    h2 = prob.h ** 2
    U1, U2 = prob.full(x)
    c1 = face_terms(U1, U2, 1, prob.h)[0]
    row2 = h2 * (c1[prob.N] - c1[0]).sum(axis=1); row3 = h2 * (c1[prob.N] - c1[0]).sum(axis=0)
    ident = max(np.abs(np.roll(S2, -1)[:-1] - S2[:-1] + row2[:-1]).max(),
                np.abs(np.roll(S3, -1)[:-1] - S3[:-1] + row3[:-1]).max())
    R2 = h2 * vf["f2"].sum(axis=(0, 2)); R3 = h2 * vf["f3"].sum(axis=(0, 1))
    print("MEANFLUX %s field=%s eps=%g N=%d active=%s plane=%d S2=%+.3e S3=%+.3e | spread over planes max|S-S[plane]|="
          "(%.2e,%.2e) mean over planes=(%+.3e,%+.3e) | row x1-flux mismatch max|h^2 sum(c1_out - c1_in)|=(%.2e,%.2e) "
          "balance identity defect=%.1e | vD_faces: S2=%+.2e S3=%+.2e spread=(%.1e,%.1e)"
          % (tag, field, eps, N, bool(prob.ne), MF_PLANE, S2[MF_PLANE], S3[MF_PLANE],
             np.abs(S2 - S2[MF_PLANE]).max(), np.abs(S3 - S3[MF_PLANE]).max(), S2.mean(), S3.mean(),
             np.abs(row2).max(), np.abs(row3).max(), ident, R2[MF_PLANE], R3[MF_PLANE],
             np.abs(R2 - R2[MF_PLANE]).max(), np.abs(R3 - R3[MF_PLANE]).max()), flush=True)


def run_case(field, eps, N, free_outlet=False, solver="auto", tol=1e-13, continuation=True, spectrum_out=None,
             start="inlet", maxit=60, hessian="fd", form="q1", qmode="center", meanflux=True):
    _hdr("candidate (ii) energy=%s q=%s %s:%g:%d %s%s start=%s"
         % (form, qmode, field, eps, N, "free-outlet" if free_outlet else "", "" if meanflux else " no-meanflux", start))
    tl = time.time()
    case = C.load_case(field, eps, N, verbose=False)
    tl = time.time() - tl
    ch = field.endswith("_ch")
    constrained = (not ch) and (not free_outlet)
    cand = cand_name(form, free_outlet, ch, meanflux, qmode, start, hessian)
    fopts = dict(form=form, qmode=qmode, meanflux=meanflux)
    prob, x, y, its, status, hist, gref, tsol, solver = solve_case(case, constrained, solver, tol, continuation,
                                                                   start=start, maxit=maxit, hessian=hessian,
                                                                   fopts=fopts)
    U1, U2 = prob.full(x)
    fl = mimetic_fluxes(U1, U2, prob.h)
    m = candidate_metrics(case, U1, U2, fl)
    grel = np.linalg.norm(prob.gradL(x, y)) / gref
    if constrained:
        lam, mu, eta = prob.split(y)
        cm = prob.cons_mean(x) if prob.ne else None
        crel = constraint_rel(prob, prob.cons(x), cm)
    else:
        lam, mu, eta, crel = None, 0.0, np.zeros(0), float("nan")
    a2, a3 = M.affine(N)
    Uor = (case["psi_or"][0] - a2, case["psi_or"][1] - a3)
    xor = prob.pack(*Uor)
    E, Eor = prob.energy(x), prob.energy(xor)
    print("SOLVE field=%s eps=%g N=%d cand=%s energy=%s q=%s constrained=%s meanflux_rows=%d solver=%s status=%s its=%d "
          "g_rel=%.3e c_rel(max)=%.3e f1_in mean-1=%+.1e mu=%.1e |lam|=%.3e eta=%s E_min=%.15e E(oracle)=%.15e "
          "E(oracle)-E_min=%+.3e t_load=%.1fs t_solve=%.1fs"
          % (field, eps, N, cand, form, qmode, constrained, prob.ne, solver, status, its, grel, crel,
             prob.f1in_mean_defect, mu, np.linalg.norm(lam) if lam is not None else 0.0,
             "(" + ",".join("%+.3e" % v for v in eta) + ")", E, Eor, Eor - E, tl, tsol), flush=True)
    hstr = " ".join("%.1e" % h[1] for st in hist for h in st[3])
    print("HISTORY field=%s eps=%g N=%d cand=%s g_rel: %s" % (field, eps, N, cand, hstr), flush=True)
    for st in hist:
        print("HISTORY_STAGE field=%s eps=%g N=%d cand=%s s=%.2f its=%d status=%s g_rel: %s"
              % (field, eps, N, cand, st[0], st[1], st[2], " ".join("%.1e" % h[1] for h in st[3])), flush=True)
    _extra_line("DIAG", field, eps, N, m)
    print("VDREF field=%s eps=%g N=%d |vD| p0.1/1/5/50=%.4f/%.4f/%.4f/%.4f min|vD|=%.4f"
          % (field, eps, N, m["vD_p"][0], m["vD_p"][1], m["vD_p"][2], m["vD_p"][3], m["vD_min"]), flush=True)
    if not ch:
        meanflux_line("candidate", field, eps, N, prob, x, case["vD_faces"])
        meanflux_line("oracle", field, eps, N, prob, xor, case["vD_faces"])
    M.print_case(field, eps, N, cand, m, r_F=grel, its=its, t=tsol)
    # oracle-FD ceiling (D-5): the same mimetic flux (= Q1 face average) on the exact oracle labels
    flo = mimetic_fluxes(Uor[0], Uor[1], prob.h)
    mo = candidate_metrics(case, Uor[0], Uor[1], flo)
    mo["e_psi"] = 0.0
    _extra_line("DIAG_ORACLE", field, eps, N, mo)
    M.print_case(field, eps, N, "oracle_mim", mo, r_F=None, its=None, t=0.0)
    # Kelvin-type diagnostics: E(oracle) - E_min (>= 0 iff the oracle labels are not below the computed minimum)
    # and the TPFA Darcy flux of the same data (exact Kelvin identity only for --energy whitney)
    flk, Ek = kelvin_tpfa(prob, fl["f1"][0], prob.t if constrained else None, meanflux=bool(prob.ne))
    vref = all_faces(case["vD_faces"])
    hg = hourglass_content(U1, U2); hgo = hourglass_content(*Uor)
    if form == "whitney":
        kel = "E_kelvin=%.15e E_min-E_kelvin=%+.3e E(oracle)-E_kelvin=%+.3e" % (Ek, E - Ek, Eor - Ek)
    else:
        kel = "E(oracle)-E_min=%+.3e (Q1 energy: the TPFA Kelvin identity does not apply; TPFA flux shown for reference)" \
              % (Eor - E)
    print("KELVIN field=%s eps=%g N=%d cand=%s %s e_v(TPFA kelvin flux vs vD_faces)=%.3e |c_min - c_tpfa|/rms(vD)=%.3e "
          "|c_oracle - c_tpfa|/rms(vD)=%.3e hourglass rms (U1 x3-Nyq, U2 x2-Nyq): candidate=(%.2e,%.2e) "
          "oracle=(%.2e,%.2e)"
          % (field, eps, N, cand, kel, M.rms(all_faces(flk) - vref) / M.rms(vref),
             M.rms(all_faces(fl) - all_faces(flk)) / M.rms(vref), M.rms(all_faces(flo) - all_faces(flk)) / M.rms(vref),
             hg[0], hg[1], hgo[0], hgo[1]), flush=True)
    if spectrum_out is not None:
        spectrum(prob, x, y, field, eps, N, spectrum_out, tag=spec_tag(form, qmode, meanflux, constrained, ch, hessian))
    return m, mo


def spec_tag(form, qmode, meanflux, constrained, ch, hessian="fd"):
    t = "_q1" if form == "q1" else "_q1_ctl_whitney"
    if form == "q1" and qmode == "q1":
        t += "_qq1"
    if not ch and not constrained:
        t += "_free"
    elif not ch and not meanflux:
        t += "_nomf"
    if hessian == "gn":
        t += "_gn"
    return t


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


def spectrum(prob, x, y, field, eps, N, outdir, tag=""):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    gfun = (lambda z: prob.gradL(z, y)) if prob.constrained else prob.grad
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
    if prob.ne:                                            # the two D-3 rows
        Jt = np.vstack([Jt, prob.cons_mean_jac(x).toarray()])
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
def consistency(field, eps, grids, free_outlet=False, fopts=None):
    fopts = dict(fopts or {})
    _hdr("consistency at the oracle labels %s:%g grids=%s energy=%s q=%s meanflux=%s"
         % (field, eps, grids, fopts.get("form", "q1"), fopts.get("qmode", "center"), fopts.get("meanflux", True)))
    ch = field.endswith("_ch")
    constrained = (not ch) and (not free_outlet)
    rows = []
    for N in grids:
        case = C.load_case(field, eps, N, verbose=False)
        prob = EnergyProblem(case, constrained, s=1.0, **fopts)
        a2, a3 = M.affine(N)
        xor = prob.pack(case["psi_or"][0] - a2, case["psi_or"][1] - a3)
        gref = np.linalg.norm(prob.grad(np.zeros(prob.n)))
        g = prob.grad(xor)
        mf = (float("nan"), float("nan"))
        if constrained:
            # least-squares multipliers of ALL constraint rows: min_z |g + A^T z|, A = [h^2 dC/dx; D-3 rows]
            # (normal equations; A A^T is (N^2 + ne) square, singular in the redundant outlet direction -> lstsq)
            A = (prob.h ** 2 * prob.cons_jac(xor)).tocsr()
            if prob.ne:
                A = sp.vstack([A, prob.cons_mean_jac(xor)]).tocsr()
            AAt = (A @ A.T).toarray()
            z, *_ = np.linalg.lstsq(AAt, -(A @ g), rcond=1e-13)
            g = g + A.T @ z
            cres = prob.cons(xor)
            c_rel = M.rms(cres) / M.rms(prob.t)
            c1o = face_terms(*prob.full(xor), 1, prob.h)[0][N]
            c_raw = M.rms(c1o - case["vD_faces"]["f1"][0]) / M.rms(case["vD_faces"]["f1"][0])
            if prob.ne:
                mf = tuple(float(v) for v in prob.cons_mean(xor))
        else:
            c_rel = c_raw = float("nan")
        G = g.reshape(2, N, N, N)
        g_int = np.linalg.norm(G[:, :N - 1]) / gref
        g_out = np.linalg.norm(G[:, N - 1]) / gref
        g_tot = np.linalg.norm(g) / gref
        # TPFA Darcy flux of the same data (for --energy whitney: the flux of every exact minimizer, Kelvin)
        flo = mimetic_fluxes(*prob.full(xor), prob.h)
        flk, _ = kelvin_tpfa(prob, flo["f1"][0], prob.t if constrained else None, meanflux=bool(prob.ne))
        vref = all_faces(case["vD_faces"])
        ev_k = M.rms(all_faces(flk) - vref) / M.rms(vref)
        ev_o = M.rms(all_faces(flo) - vref) / M.rms(vref)
        mfa = max(abs(mf[0]), abs(mf[1])) if constrained and prob.ne else float("nan")
        rows.append((N, g_tot, g_int, g_out, c_rel, mfa, ev_k, ev_o))
        print("CONSIST field=%s eps=%g N=%d energy=%s constrained=%s meanflux_rows=%d g_rel(oracle)=%.3e "
              "interior(planes 1..N-1)=%.3e outlet=%.3e constraint rms rel=%.3e (vs raw f1_in: %.3e) "
              "D-3 rows (S2,S3)=(%+.3e,%+.3e) |grad E(0)|=%.3e | e_v(TPFA flux)=%.3e e_v(oracle_mim)=%.3e"
              % (field, eps, N, prob.form, constrained, prob.ne, g_tot, g_int, g_out, c_rel, c_raw, mf[0], mf[1],
                 gref, ev_k, ev_o), flush=True)
        if not ch:
            meanflux_line("oracle", field, eps, N, prob, xor, case["vD_faces"])
    Ns = [r[0] for r in rows]
    for k, name in ((1, "g_rel"), (2, "g_interior"), (3, "g_outlet"), (4, "constraint"), (5, "meanflux_rows"),
                    (6, "e_v_tpfa"), (7, "e_v_oracle_mim")):
        e = [r[k] for r in rows]
        if all(np.isfinite(e)) and all(v > 0 for v in e):
            print("CONSIST_ORDER field=%s eps=%g %s: %s orders %s" % (field, eps, name, " ".join("%.3e" % v for v in e),
                                                                      " ".join("%.2f" % o for o in M.orders(e, Ns))),
                  flush=True)


# ------------------------------------------------------------------------------------------------
def _face_rule():
    """3 x 3 Gauss-Legendre rule on the unit face: local 2-D points (9, 2), weights (9,) summing to 1."""
    pts = np.array(list(itertools.product(_GL3_X, repeat=2)))
    w = np.array([a * b for a, b in itertools.product(_GL3_W, repeat=2)])
    return pts, w


def _face_pts(axis, xi_a, pts2):
    """Local 3-D points on the cell face xi_axis = xi_a; the two tangential coordinates from pts2 (in increasing
    axis order)."""
    t = [b for b in range(3) if b != axis]
    P = np.empty((pts2.shape[0], 3))
    P[:, axis] = xi_a; P[:, t[0]] = pts2[:, 0]; P[:, t[1]] = pts2[:, 1]
    return P


def q1_structure_checks(N, rng, check):
    """(a) pointwise div c = 0, (b) face averages of c . n = Whitney c_f and normal continuity, for the Q1 in-cell
    product; random labels of O(0.1) and O(1) periodic parts."""
    h = 1.0 / N
    Z = np.zeros((N + 1, N, N))
    _, _, c = q1_fields(Z, Z, h)
    e1 = np.zeros((1, 3, 1, 1, 1)); e1[0, 0] = 1.0
    check("Q1 affine labels: max|c - e1| at the 27 Gauss points of every cell", np.abs(c - e1).max(), 1e-13)
    fp, fw = _face_rule()
    for amp in (0.1, 1.0):
        U1 = amp * rng.standard_normal((N + 1, N, N)); U2 = amp * rng.standard_normal((N + 1, N, N))
        # (a) div c at 6 random interior points of every cell by central differences of step dl (local units):
        # c_a is a polynomial of degree 2 in x_a (c1 ~ (2,1,1), c2 ~ (1,2,1), c3 ~ (1,1,2)), so the central
        # difference of c_a along x_a is EXACT up to roundoff
        base = rng.uniform(0.2, 0.8, (6, 3)); dl = 0.1
        pts = []
        for pb in base:
            for ax in range(3):
                for sg in (1.0, -1.0):
                    q = pb.copy(); q[ax] += sg * dl; pts.append(q)
        cc = q1_c_at(U1, U2, h, np.array(pts)).reshape(6, 3, 2, 3, N, N, N)
        div = sum((cc[:, ax, 0, ax] - cc[:, ax, 1, ax]) / (2.0 * dl * h) for ax in range(3))
        cmax = np.abs(cc).max()
        check("(a) Q1 amp=%g: pointwise max|div c| / (max|c|/h), 6 random points x %d cells" % (amp, N ** 3),
              np.abs(div).max() / (cmax / h), 1e-13)
        # (b) face average of c . n (3x3 Gauss on the face; c . n is bilinear on the face) vs Whitney c_f
        fl = mimetic_fluxes(U1, U2, h)
        err = 0.0; cont = 0.0
        for ax in range(3):
            lo = q1_c_at(U1, U2, h, _face_pts(ax, 0.0, fp))[:, ax]           # (9, N, N, N) on the face xi_ax = 0
            hi = q1_c_at(U1, U2, h, _face_pts(ax, 1.0, fp))[:, ax]           # on the face xi_ax = 1
            avg_lo = np.tensordot(fw, lo, axes=(0, 0))
            if ax == 0:
                avg = np.concatenate([avg_lo, np.tensordot(fw, hi[:, N - 1:N], axes=(0, 0))], axis=0)
                cont = max(cont, np.abs(hi[:, :-1] - lo[:, 1:]).max())
            else:
                avg = avg_lo
                cont = max(cont, np.abs(hi - np.roll(lo, -1, axis=ax + 1)).max())
            err = max(err, np.abs(avg - fl["f%d" % (ax + 1)]).max())
        scale = np.abs(all_faces(fl)).max()
        check("(b) Q1 amp=%g: max|face average of c.n - Whitney c_f| / max|c_f| (all faces)" % amp, err / scale,
              1e-13)
        check("(b) Q1 amp=%g: normal continuity max|c.n(left cell) - c.n(right cell)| / max|c_f|" % amp,
              cont / scale, 1e-13)


def uniform_kernel_report(case, N, rng, verdict):
    """(d) k = 1, u = 0: dense FD Hessian of the energy (inlet Dirichlet, free outlet), count of relative
    eigenvalues below 1e-10 and the curvature along the two checkerboard (hourglass) directions of the Whitney form,
    for both energy forms."""
    out = {}
    n = 2 * N ** 3
    if n > 8000:
        print("GRADTEST info: N=%d too large for the dense uniform-state Hessian (n=%d); skipped" % (N, n), flush=True)
        return out
    f = rng.standard_normal((N, N)); g = rng.standard_normal((N, N))
    sgn = (-1.0) ** np.arange(N)
    T1 = np.zeros((N + 1, N, N)); T1[1:] = f[:, :, None] * sgn[None, None, :]      # U1 = (-1)^m3 f(j, m2)
    T2 = np.zeros((N + 1, N, N)); T2[1:] = g[:, None, :] * sgn[None, :, None]      # U2 = (-1)^m2 g(j, m3)
    Z = np.zeros((N + 1, N, N))
    for form in ENERGY_FORMS:
        p0 = EnergyProblem(case, False, qconst=1.0, zero_inlet=True, form=form)
        x0 = np.zeros(p0.n)
        t0 = time.time()
        H = hessian_dense(p0.grad, x0)
        ev = np.linalg.eigvalsh(H)
        r = np.sort(np.abs(ev) / np.abs(ev).max())
        nnull = int(np.sum(r < 1e-10))
        curv = []
        for t in (p0.pack(T1, Z), p0.pack(Z, T2)):
            Ht = (p0.grad(x0 + 1e-6 * t) - p0.grad(x0 - 1e-6 * t)) / 2e-6
            curv.append(float(t @ Ht) / float(t @ t))
        print("GRADTEST info: energy=%s k=1 u=0 N=%d dense Hessian n=%d (t=%.1fs): eig max=%.4e, relative eigenvalues "
              "< 1e-10: %d (2 N^2 = %d), < 1e-6: %d, < 1e-3: %d, negative: %d, smallest 8: %s"
              % (form, N, p0.n, time.time() - t0, np.abs(ev).max(), nnull, 2 * N * N, int(np.sum(r < 1e-6)),
                 int(np.sum(r < 1e-3)), int(np.sum(ev < -1e-10 * np.abs(ev).max())),
                 " ".join("%.2e" % v for v in r[:8])), flush=True)
        print("GRADTEST info: energy=%s checkerboard curvature t^T H t / |t|^2: t1 (U1 = (-1)^m3 f(j,m2)) = %.4e "
              "(rel %.2e), t2 (U2 = (-1)^m2 g(j,m3)) = %.4e (rel %.2e); smallest eigenvalue = %.4e"
              % (form, curv[0], curv[0] / np.abs(ev).max(), curv[1], curv[1] / np.abs(ev).max(), ev.min()), flush=True)
        out[form] = (nnull, curv, ev)
    nq, cq, evq = out["q1"]
    verdict("(d) Q1 k=1 Hessian: relative eigenvalues < 1e-10 (expected 0)", nq, nq == 0)
    verdict("(d) Q1 k=1 Hessian: smallest eigenvalue > 0", float(evq.min()), evq.min() > 0)
    verdict("(d) Q1 checkerboard curvature t1 > 0", cq[0], cq[0] > 0)
    verdict("(d) Q1 checkerboard curvature t2 > 0", cq[1], cq[1] > 0)
    nw, cw, _ = out["whitney"]
    print("GRADTEST info: control energy=whitney: %d null modes (2 N^2 = %d expected), checkerboard curvature "
          "(%.1e, %.1e) (0 expected: the hourglass kernel)" % (nw, 2 * N * N, cw[0], cw[1]), flush=True)
    return out


def gradtest(field, eps, N, seed=7, fopts=None):
    fopts = dict(fopts or {})
    form = fopts.get("form", "q1")
    _hdr("gradtest %s:%g:%d energy=%s q=%s meanflux=%s" % (field, eps, N, form, fopts.get("qmode", "center"),
                                                            fopts.get("meanflux", True)))
    case = C.load_case(field, eps, N, verbose=False)
    rng = np.random.default_rng(seed)
    h = 1.0 / N
    ok = True

    def check(name, val, thr):
        nonlocal ok
        good = val <= thr
        ok &= good
        print("GRADTEST %-62s %.3e  (<= %.0e) %s" % (name, val, thr, "PASS" if good else "FAIL"), flush=True)

    def verdict(name, val, good):
        nonlocal ok
        ok &= bool(good)
        print("GRADTEST %-62s %s  %s" % (name, ("%.3e" % val) if isinstance(val, float) else str(val),
                                         "PASS" if good else "FAIL"), flush=True)

    # Whitney face fluxes (used by the constraints and the metrics): affine labels, div_h c_f = 0
    Z = np.zeros((N + 1, N, N))
    fl = mimetic_fluxes(Z, Z, h)
    check("affine labels: max|c1 - 1|, max|c2|, max|c3|",
          max(np.abs(fl["f1"] - 1).max(), np.abs(fl["f2"]).max(), np.abs(fl["f3"]).max()), 1e-13)
    for amp in (0.1, 1.0):
        U1 = amp * rng.standard_normal((N + 1, N, N)); U2 = amp * rng.standard_normal((N + 1, N, N))
        fl = mimetic_fluxes(U1, U2, h)
        dv = face_divergence(fl, N)
        scale = max(np.abs(all_faces(fl)).max() / h, 1.0)
        check("random labels amp=%g: max|div_h c_f| / (max|c_f|/h)" % amp, np.abs(dv).max() / scale, 1e-14)
    # Q1 in-cell product: (a) pointwise div, (b) face averages = Whitney c_f, normal continuity
    q1_structure_checks(N, rng, check)
    for constrained in ((False, True) if not field.endswith("_ch") else (False,)):
        prob = EnergyProblem(case, constrained, **fopts)
        x = 0.05 * rng.standard_normal(prob.n)
        d = rng.standard_normal(prob.n)
        g = prob.grad(x)
        gd = float(g @ d)
        Jf, w = prob.flux_jac(x)
        fl = prob.fluxes(x)
        if prob.form == "whitney":
            check("grad E_h vs J^T (w c) with the exact sparse flux Jacobian",
                  np.abs(Jf.T @ (w * all_faces(fl)) - g).max() / np.abs(g).max(), 1e-12)
        fd = (all_faces(prob.fluxes(x + 1e-6 * d)) - all_faces(prob.fluxes(x - 1e-6 * d))) / 2e-6
        check("flux Jacobian J d vs central FD of c_f", np.abs(Jf @ d - fd).max() / np.abs(fd).max(), 1e-8)
        if not constrained:
            pw = EnergyProblem(case, False, form="whitney")
            flk, Ek = kelvin_tpfa(pw, fl["f1"][0])
            dk = all_faces(fl) - all_faces(flk)
            check("Whitney Kelvin identity E_h(u) - E_K = 1/2 |c(u) - c_K|_W^2 (random u, free outlet)",
                  abs(pw.energy(x) - Ek - 0.5 * float(np.sum(w * dk ** 2))) / pw.energy(x), 1e-12)
            check("TPFA Kelvin flux: max|div_h c_K| h / max|c_K|",
                  np.abs(face_divergence(flk, N)).max() * h / np.abs(all_faces(flk)).max(), 1e-12)
        for st in (1e-5, 1e-6, 1e-7):
            fd = (prob.energy(x + st * d) - prob.energy(x - st * d)) / (2 * st)
            check("(c) energy=%s dE/dx . d vs central FD of E_h (step %.0e)" % (prob.form, st), abs(fd - gd) / abs(gd),
                  1e-6)
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
            if prob.ne:
                eta = rng.standard_normal(2)
                jd = float(prob.cons_mean_T(x, eta) @ d)
                for st in (1e-5, 1e-6, 1e-7):
                    fd = (eta @ prob.cons_mean(x + st * d) - eta @ prob.cons_mean(x - st * d)) / (2 * st)
                    check("D-3 rows: (dCm/dx)^T eta . d vs central FD (step %.0e)" % st, abs(fd - jd) / abs(jd), 1e-6)
                Jm = prob.cons_mean_jac(x)
                check("D-3 rows: sparse Jacobian vs adjoint", np.abs(Jm.T @ eta - prob.cons_mean_T(x, eta)).max()
                      / np.abs(prob.cons_mean_T(x, eta)).max(), 1e-13)
                # TPFA with the D-3 jumps: solenoidal, zero plane fluxes
                flk, _ = kelvin_tpfa(prob, fl["f1"][0], prob.t, meanflux=True)
                check("TPFA flux with D-3 head jumps: max|div_h c| h / max|c|",
                      np.abs(face_divergence(flk, N)).max() * h / np.abs(all_faces(flk)).max(), 1e-12)
                check("TPFA flux with D-3 head jumps: |plane-%d fluxes| (x2, x3)" % MF_PLANE,
                      max(abs(h * h * flk["f2"][:, MF_PLANE, :].sum()), abs(h * h * flk["f3"][:, :, MF_PLANE].sum())),
                      1e-12)
        # colored FD Hessian vs dense FD columns (Hessian-vector products on random vectors)
        yv = rng.standard_normal(prob.nmult) if constrained else None
        gfun = (lambda z: prob.gradL(z, yv)) if constrained else prob.grad
        H, ngr = hessian_fd(prob, gfun, x)
        for _ in range(2):
            v = rng.standard_normal(prob.n)
            Hv = (gfun(x + 1e-5 * v) - gfun(x - 1e-5 * v)) / 2e-5
            check("colored FD Hessian . v vs directional FD (constrained=%s, %s stencil)"
                  % (constrained, "27-pt" if prob.form == "q1" else "19-pt"),
                  np.linalg.norm(H @ v - Hv) / np.linalg.norm(Hv), 1e-6)
        # mode preconditioner = exact inverse of the constant-coefficient KKT/Hessian at the affine state (without
        # the D-3 rows, which the preconditioner passes through)
        qbar = float(prob.q.mean())
        P = ModePreconditioner(case, constrained, qbar, form=prob.form, ne=0)
        p0 = EnergyProblem(case, constrained, qconst=qbar, zero_inlet=True, form=prob.form, meanflux=False)
        x0 = np.zeros(p0.n)
        H0, _ = hessian_fd(p0, p0.grad, x0)
        K0 = kkt_matrix(p0, H0 + P.shift * sp.identity(p0.n, format="csr"), x0)
        r = rng.standard_normal(K0.shape[0])
        if constrained:
            print("GRADTEST info: linearized outlet-flux rows vanishing at the uniform state (Fourier modes): %d "
                  "(xi = 0 and the (N/2, N/2) checkerboard expected: a property of the Whitney outlet flux, "
                  "independent of the energy form)" % int(np.sum(P.dead)), flush=True)
            R2 = np.fft.fft2(r[p0.n:p0.n + N * N].reshape(N, N)).ravel(); R2[P.dead] = 0.0
            r[p0.n:p0.n + N * N] = np.fft.ifft2(R2.reshape(N, N)).real.ravel()
            r[-1] = 0.3
        y = P.apply(r)
        check("mode preconditioner: |(K0 + shift) P^-1 r - r| / |r| (constrained=%s)" % constrained,
              np.linalg.norm(K0 @ y - r) / np.linalg.norm(r), 1e-6)
    # (d) the k = 1 kernel of both forms
    uniform_kernel_report(case, N, rng, verdict)
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
            "solver": "auto", "grids": (16, 32, 48), "cont": True, "tol": 1e-13, "start": "inlet", "maxit": 60, "hessian": "fd",
            "energy": "q1", "qmode": "center", "meanflux": True}
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
        elif a == "--energy":
            opts["energy"] = next(it)
        elif a == "--q":
            opts["qmode"] = next(it)
        elif a == "--no-meanflux":
            opts["meanflux"] = False
        elif a.startswith("--"):
            print(__doc__); return 2
        else:
            specs.append(a)
    if not specs:
        print(__doc__); return 2
    print("candidate_ii.py python=%s numpy=%s scipy=%s argv=%s" % (sys.version.split()[0], np.__version__,
                                                                 __import__("scipy").__version__, " ".join(argv)),
          flush=True)
    if opts["energy"] not in ENERGY_FORMS or opts["qmode"] not in ("center", "q1"):
        print(__doc__); return 2
    fopts = dict(form=opts["energy"], qmode=opts["qmode"], meanflux=opts["meanflux"])
    rc = 0
    for spec in specs:
        field, eps, N = C.parse_spec(spec)
        if opts["consistency"]:
            consistency(field, eps, opts["grids"], opts["free"], fopts=fopts)
            continue
        if opts["spectrum"] is not None:
            if N is not None and N != opts["spectrum"]:
                print("note: --spectrum %d overrides N=%d of %s" % (opts["spectrum"], N, spec))
            N = opts["spectrum"]
        if N is None:
            print("case spec needs N: %s" % spec); return 2
        if opts["gradtest"]:
            rc |= 0 if gradtest(field, eps, N, fopts=fopts) else 1
            continue
        run_case(field, eps, N, free_outlet=opts["free"], solver=opts["solver"], tol=opts["tol"],
                 continuation=opts["cont"], spectrum_out=(opts["out"] if opts["spectrum"] is not None else None),
                 start=opts["start"], maxit=opts["maxit"], hessian=opts["hessian"], form=fopts["form"],
                 qmode=fopts["qmode"], meanflux=fopts["meanflux"])
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
