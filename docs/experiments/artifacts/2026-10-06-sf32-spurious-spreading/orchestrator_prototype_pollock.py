#!/usr/bin/env python3
"""SF-32 orchestrator prototype (numpy, CPU; independent of any worker code).

Checks, before any worker result:
  1. Stokes face fluxes from edge line integrals of psi1 grad psi2 (composite
     4-point Gauss-Legendre on the spline knot intervals) are exact surface
     integrals of c . n and exactly divergence-free (contract 3.2, S1-S3).
  2. The Pollock semi-analytical cell tracker (contract 3.3) on those fluxes:
     uniform pair exact (P1), pair A exactness (P2), and the one-period return
     map variance versus Delta on an analytic 3-D pair (prediction P1).

Labels are ANALYTIC here (no spline), so the "exact" fluxes are Gauss
quadratures of smooth functions (converged to roundoff with enough nodes) and
the exactness check of the piecewise-polynomial argument is NOT exercised
here (that needs the real B-spline; the worker's test S2 does it). The
Pollock algebra and the scaling prediction are what this prototype tests.
"""
import sys
import numpy as np

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Analytic label pairs (same conventions as tests/streamline_tracker/analytic_pairs.hpp)
# ---------------------------------------------------------------------------
def labels(pair, x, y, z, e=0.05, a=0.1, b=0.08):
    """Return psi1, psi2, g1 (3,...), g2 (3,...) at unwrapped (x, y, z)."""
    zero = np.zeros_like(x)
    if pair == "U":
        s1, s2 = zero, zero
        g1 = np.stack([zero, zero, zero]); g2 = np.stack([zero, zero, zero])
    elif pair == "A":
        s1 = a * np.sin(TWO_PI * x); s2 = zero
        g1 = np.stack([a * TWO_PI * np.cos(TWO_PI * x), zero, zero])
        g2 = np.stack([zero, zero, zero])
    elif pair == "B":
        s1 = a * np.sin(TWO_PI * x); s2 = b * np.sin(TWO_PI * y)
        g1 = np.stack([a * TWO_PI * np.cos(TWO_PI * x), zero, zero])
        g2 = np.stack([zero, b * TWO_PI * np.cos(TWO_PI * y), zero])
    elif pair == "G":
        sx, cx = np.sin(TWO_PI * x), np.cos(TWO_PI * x)
        sy, cy = np.sin(TWO_PI * y), np.cos(TWO_PI * y)
        sz, cz = np.sin(TWO_PI * z), np.cos(TWO_PI * z)
        syz, cyz = np.sin(TWO_PI * (y + z)), np.cos(TWO_PI * (y + z))
        sxz, cxz = np.sin(TWO_PI * (x - z)), np.cos(TWO_PI * (x - z))
        s1 = e * (sx * cz + 0.5 * syz)
        g1 = np.stack([e * TWO_PI * cx * cz, e * 0.5 * TWO_PI * cyz,
                       e * (-TWO_PI * sx * sz + 0.5 * TWO_PI * cyz)])
        s2 = e * (cx * sy + 0.5 * cxz)
        g2 = np.stack([e * (-TWO_PI * sx * sy - 0.5 * TWO_PI * sxz), e * TWO_PI * cx * cy,
                       e * 0.5 * TWO_PI * sxz])
    else:
        raise ValueError(pair)
    psi1 = y + s1
    psi2 = z + s2
    g1[1] += 1.0
    g2[2] += 1.0
    return psi1, psi2, g1, g2


def velocity(pair, x, y, z, **kw):
    _, _, g1, g2 = labels(pair, x, y, z, **kw)
    return np.cross(g1, g2, axis=0)


# ---------------------------------------------------------------------------
# Stokes face fluxes: edge integrals of psi1 (grad psi2 . t)
# ---------------------------------------------------------------------------
GL_X, GL_W = np.polynomial.legendre.leggauss(8)  # 8 nodes per sub-interval (smooth analytic integrand here)


def fluct(pair, x, y, z, **kw):
    """Periodic parts s1, s2 and grad s2 (3,...) at (x, y, z)."""
    psi1, psi2, g1, g2 = labels(pair, x, y, z, **kw)
    s1 = psi1 - y
    s2 = psi2 - z
    gs2 = g2.copy(); gs2[2] -= 1.0
    return s1, s2, gs2


def edge_integrals(pair, n, axis, nsub, **kw):
    """Periodic edge array along `axis`: int_edge [ s1 (d s2/dx_axis + gbar2[axis]) - s2 gbar1[axis] ] dl,
    vectorized over all n^3 edges; composite GL on nsub sub-intervals."""
    D = 1.0 / n
    gbar1 = np.array([0.0, 1.0, 0.0]); gbar2 = np.array([0.0, 0.0, 1.0])
    idx = np.arange(n) * D
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    total = np.zeros((n, n, n))
    for sidx in range(nsub):
        a0 = sidx * D / nsub; a1 = (sidx + 1) * D / nsub
        mid, half = 0.5 * (a0 + a1), 0.5 * (a1 - a0)
        for node, wgt in zip(GL_X, GL_W):
            t = mid + half * node
            pts = [X.copy(), Y.copy(), Z.copy()]
            pts[axis] = pts[axis] + t
            s1, s2, gs2 = fluct(pair, *pts, **kw)
            total += half * wgt * (s1 * (gs2[axis] + gbar2[axis]) - s2 * gbar1[axis])
    return total


def face_fluxes(pair, n, nsub=4, **kw):
    """Periodic MAC face averages u, v, w on an n^3 grid (periodic decomposition, contract 3.2)."""
    D = 1.0 / n
    ex = edge_integrals(pair, n, 0, nsub, **kw)
    ey = edge_integrals(pair, n, 1, nsub, **kw)
    ez = edge_integrals(pair, n, 2, nsub, **kw)
    ip = lambda arr, s, ax: np.roll(arr, -s, axis=ax)  # arr[idx + s] periodic
    u = 1.0 + (ey + ip(ez, 1, 1) - ip(ey, 1, 2) - ez) / D**2   # (gbar1 x gbar2) = e_x
    v = 0.0 + (ez + ip(ex, 1, 2) - ip(ez, 1, 0) - ex) / D**2
    w = 0.0 + (ex + ip(ey, 1, 0) - ip(ex, 1, 1) - ey) / D**2
    return u, v, w


def divergence(u, v, w):
    return (np.roll(u, -1, 0) - u) + (np.roll(v, -1, 1) - v) + (np.roll(w, -1, 2) - w)


def surface_quadrature(pair, n, i, j, k, comp, nsub=4, **kw):
    """Direct 2-D Gauss quadrature of c . e_comp over the face of cell (i,j,k) at its lower comp-coordinate."""
    D = 1.0 / n
    axes = [a for a in range(3) if a != comp]
    base = np.array([i, j, k]) * D
    total = 0.0
    for s in range(nsub):
        for r in range(nsub):
            a0, a1 = s * D / nsub, (s + 1) * D / nsub
            b0, b1 = r * D / nsub, (r + 1) * D / nsub
            ta = 0.5 * (a0 + a1) + 0.5 * (a1 - a0) * GL_X
            tb = 0.5 * (b0 + b1) + 0.5 * (b1 - b0) * GL_X
            TA, TB = np.meshgrid(ta, tb, indexing="ij")
            WA, WB = np.meshgrid(GL_W, GL_W, indexing="ij")
            pts = [np.full_like(TA, base[0]), np.full_like(TA, base[1]), np.full_like(TA, base[2])]
            pts[axes[0]] = base[axes[0]] + TA
            pts[axes[1]] = base[axes[1]] + TB
            c = velocity(pair, *pts, **kw)
            total += 0.25 * (a1 - a0) * (b1 - b0) * np.sum(WA * WB * c[comp])
    return total / D**2


# ---------------------------------------------------------------------------
# Pollock semi-analytical tracker (contract 3.3)
# ---------------------------------------------------------------------------
INF = np.inf


def axis_exit(vm, vp, D, xr, vpart):
    """Exit time along one axis (well-conditioned form). vm, vp face velocities (lower, upper),
    xr in [0, D] relative position, vpart = vm + A xr the interpolated particle velocity.
    With d = distance to the candidate face in the direction of motion: vf = vpart + A d exactly
    for the linear interpolant, so t = log1p(A d / vpart) / A (-> d / vpart as A -> 0); the particle
    reaches the face iff 1 + A d / vpart > 0 (velocity does not vanish before the face)."""
    if vpart == 0.0:
        return INF, 0
    A = (vp - vm) / D
    if vpart > 0.0:
        d, side = D - xr, +1
    else:
        d, side = 0.0 - xr, -1
    if A == 0.0:
        return d / vpart, side
    z = A * d / vpart
    if not (1.0 + z > 0.0):
        return INF, 0
    return np.log1p(z) / A, side


def axis_move(vm, vp, D, xr, vpart, t):
    """Position after time t: xr + vpart expm1(A t) / A (-> xr + vpart t as A -> 0)."""
    A = (vp - vm) / D
    if A == 0.0:
        return xr + vpart * t
    return xr + vpart * np.expm1(A * t) / A


def pollock_period(u, v, w, n, x0, max_cells=10**7):
    """Track one particle from x0 (on the face x1 = 0) until unwrapped x1 reaches 1.

    Returns unwrapped position, clock, status, cells crossed. Position state:
    cell index (ci, cj, ck) with periodic wraps (wi, wj, wk), relative
    position r in [0, D]^3 inside the cell.
    """
    D = 1.0 / n
    F = (u, v, w)
    pos = np.array(x0, dtype=float)
    cell = np.floor(pos / D).astype(int) % n
    r = pos - np.floor(pos / D) * D
    wrap = np.floor(pos / D).astype(int) // n
    # a seed exactly on the face x1 = 0 belongs to cell 0 if u > 0 (moving into it)
    t = 0.0
    cells = 0
    while True:
        if cells >= max_cells:
            return None, t, "cap", cells
        vm = [F[a][tuple(cell)] for a in range(3)]
        vp = [F[a][tuple((cell + np.eye(3, dtype=int)[a]) % n)] for a in range(3)]
        vpart = [vm[a] + (vp[a] - vm[a]) / D * r[a] for a in range(3)]
        exits = [axis_exit(vm[a], vp[a], D, r[a], vpart[a]) for a in range(3)]
        te = min(e[0] for e in exits)
        if not np.isfinite(te):
            return None, t, "stagnation", cells
        # unwrapped x1 after this step
        x1_exit = (cell[0] + wrap[0] * n) * D + axis_move(vm[0], vp[0], D, r[0], vpart[0], te)
        target = 1.0
        if x1_exit >= target - 1e-15 and exits[0][0] == te and exits[0][1] == +1 and abs(x1_exit - target) < 1e-12:
            # lands exactly on the x-face x1 = 1
            newr = np.array([axis_move(vm[a], vp[a], D, r[a], vpart[a], te) for a in range(3)])
            newr[0] = D
            xu = (cell + wrap * n) * D + newr
            return xu, t + te, "ok", cells + 1
        if x1_exit > target:
            # final partial step inside this cell (A_x != 0 or linear)
            A = (vp[0] - vm[0]) / D
            rt = target - (cell[0] + wrap[0] * n) * D
            if A == 0.0:
                tp = (rt - r[0]) / vpart[0]
            else:
                tp = np.log1p(A * (rt - r[0]) / vpart[0]) / A
            newr = np.array([axis_move(vm[a], vp[a], D, r[a], vpart[a], tp) for a in range(3)])
            newr[0] = rt
            xu = (cell + wrap * n) * D + newr
            return xu, t + tp, "ok", cells
        # take the full cell step
        newr = np.array([axis_move(vm[a], vp[a], D, r[a], vpart[a], te) for a in range(3)])
        for a in range(3):
            if exits[a][0] == te:
                if exits[a][1] == +1:
                    newr[a] = 0.0
                    cell[a] += 1
                else:
                    newr[a] = D
                    cell[a] -= 1
                if cell[a] >= n:
                    cell[a] -= n; wrap[a] += 1
                if cell[a] < 0:
                    cell[a] += n; wrap[a] -= 1
        r = newr
        t += te
        cells += 1


def seeds(npart, seed=20261006):
    rng = np.random.default_rng(seed)
    return rng.random((npart, 2))


def main():
    np.set_printoptions(precision=4)
    print("== 1. Stokes face fluxes on pair G (e=0.05), n = 8 ==")
    n = 8
    u, v, w = face_fluxes("G", n)
    div = divergence(u, v, w)
    print(f"max|div| = {np.max(np.abs(div)):.3e}   max|u| = {np.max(np.abs(u)):.3f}  (S1: expect roundoff)")
    err = 0.0
    for (i, j, k) in [(0, 0, 0), (3, 5, 2), (7, 1, 6)]:
        for comp, arr in enumerate((u, v, w)):
            q = surface_quadrature("G", n, i, j, k, comp)
            err = max(err, abs(q - arr[i, j, k]))
    print(f"max |edge-loop flux - direct surface quadrature| = {err:.3e}  (S2: expect ~1e-14)")
    u2, v2, w2 = face_fluxes("G", n // 2)
    uc = 0.25 * (u[::2, ::2, ::2] + u[::2, 1::2, ::2] + u[::2, ::2, 1::2] + u[::2, 1::2, 1::2])
    print(f"max |coarse flux - mean of fine| = {np.max(np.abs(uc - u2)):.3e}  (S3)")
    uu, vv, ww = face_fluxes("U", 4)
    print(f"uniform: max|u-1| = {np.max(np.abs(uu-1)):.1e} max|v| = {np.max(np.abs(vv)):.1e} max|w| = {np.max(np.abs(ww)):.1e}  (S4)")
    uA, vA, wA = face_fluxes("A", 8, a=0.1)
    D = 1.0 / 8
    xi = np.arange(8) * D
    vA_exact = -0.1 * (np.sin(TWO_PI * (xi + D)) - np.sin(TWO_PI * xi)) / D
    print(f"pair A: max|u-1| = {np.max(np.abs(uA-1)):.1e}  max|v - exact cell mean| = {np.max(np.abs(vA[:,0,0]-vA_exact)):.1e}  max|w| = {np.max(np.abs(wA)):.1e}  (S5)")

    print("\n== 2. Pollock: uniform pair (P1) and pair A exactness (P2) ==")
    for (y0, z0) in [(0.2, 0.3), (0.61, 0.07)]:
        xu, tau, st, nc = pollock_period(uu, vv, ww, 4, (0.0, y0, z0))
        print(f"U: start ({y0},{z0}) -> xu = {xu}, tau = {tau!r}, status {st}, cells {nc}")
    n = 16
    uA, vA, wA = face_fluxes("A", n, a=0.1)
    S = seeds(64)
    maxerr = 0.0
    for (y0, z0) in S:
        xu, tau, st, nc = pollock_period(uA, vA, wA, n, (0.0, y0, z0))
        if xu is None:
            print('  pair A seed', y0, z0, 'status', st); continue
        maxerr = max(maxerr, abs(xu[1] - y0), abs(xu[2] - z0), abs(tau - 1.0))
    print(f"A (n=16): max |delta_x2|, |delta_x3|, |tau-1| over 64 seeds = {maxerr:.3e}  (P2: expect roundoff)")

    print("\n== 3. Pollock return map on pair G (e=0.05): variance vs Delta (prediction P1) ==")
    S = seeds(1024)
    results = {}
    for n in (8, 16, 32, 64):
        u, v, w = face_fluxes("G", n, nsub=2)
        d2, d3, taus, bad = [], [], [], 0
        for (y0, z0) in S:
            xu, tau, st, nc = pollock_period(u, v, w, n, (0.0, y0, z0))
            if st != "ok":
                bad += 1
                continue
            d2.append(xu[1] - y0); d3.append(xu[2] - z0); taus.append(tau)
        d2, d3 = np.array(d2), np.array(d3)
        results[n] = (d2.var(), d3.var())
        print(f"n={n:3d}: var(dx2) = {d2.var():.3e} var(dx3) = {d3.var():.3e} mean(dx2) = {d2.mean():+.2e} mean(dx3) = {d3.mean():+.2e} <tau> = {np.mean(taus):.6f} bad = {bad}")
    ns = np.array(sorted(results))
    for comp, name in ((0, "x2"), (1, "x3")):
        var = np.array([results[n][comp] for n in ns])
        slope = np.polyfit(np.log(1.0 / ns), np.log(var), 1)[0]
        two = [np.log(var[i] / var[i + 1]) / np.log(2.0) for i in range(len(ns) - 1)]
        print(f"variance({name}) ~ Delta^p: least-squares p = {slope:.3f}; two-level {np.array(two)}")


if __name__ == "__main__":
    main()
