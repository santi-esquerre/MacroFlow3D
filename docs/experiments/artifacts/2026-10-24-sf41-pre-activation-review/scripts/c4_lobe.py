"""C4 design emulation: inlet-exit control on a harmonic k = 1 field with a backflow lobe at the
inlet (numpy/scipy; not SF-41 evidence).

Field (G1 = 1): Phi = G1 x1 + f, f = a cos(2 pi x2) exp(-2 pi x1) (harmonic), v = grad Phi,
exact stream function psi = G1 x2 - a exp(-2 pi x1) sin(2 pi x2) (v1 = d psi/d x2, v2 = -d psi/d x1),
separatrix psi in Z through the stagnation point x1s = ln(2 pi a / G1)/(2 pi), x2 = 0.
The field is x3-independent, so everything is 2-D (an M x M seed lattice = M distinct x2 values).

Methods per seed (seeds at x1 = X0 = 0.1):
  A  analytic field, scipy DOP853 (rtol 1e-12, atol 1e-14), events x1 = 0 (exit_inlet) and x1 = L1;
  B  DP5(4) arclength emulation (tol 1e-8 absolute, positions only, h_max = h/8, h0 = 0.1 h_max, factor
     clamp(0.9 err^-0.2, 0.2, 5)) on the 2-D spline of f sampled at cell centres: B_nak = not-a-knot in x1
     (periodic in x2), B_per = periodic in x1 (diagnostic, the SF-28 closure).  Lock-step vectorised over
     seeds; exit_inlet when an accepted step ends at x1 < 0, exit_outlet at x1 >= L1; the plane pass is
     read off the same trajectory (status 'plane' iff x1 >= L1/2 is reached before an inlet exit; the
     Henon landing is NOT emulated: it does not change the status); the time pass at T is read off the
     passive tau of the same trajectory (exit if tau_exit <= T else active).  A seed with spline g1 <= 0
     is 'seed_backflow' (not traced, as in the tracker);
  C  Pollock RT0 emulation on the analytically integrated face fluxes (exact face averages of v; RT0
     velocity linear in its own coordinate per cell; analytic exit times via log1p/expm1; non-periodic in
     x1 with exit_inlet / exit_outlet, periodic in x2); seeds with u <= 0 are still traced.

Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 15 taskset -c 10-15 python3 -I c4_lobe.py
"""
import importlib.util
import math
import os

import numpy as np
from scipy.integrate import solve_ivp

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c_common", os.path.join(here, "c_common.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

TP = 2.0 * math.pi
G1 = 1.0
X0 = 0.1
TOL = 1e-8
IN, OUT, SB, ACT, BAD = "exit_inlet", "exit_outlet", "seed_backflow", "active", "other"
CAP = "cap_exceeded"


# ------------------------------------------------------------------------------------------------
# A: analytic field + DOP853
# ------------------------------------------------------------------------------------------------
def v_exact(a, x1, x2):
    e = np.exp(-TP * x1)
    return G1 - TP * a * np.cos(TP * x2) * e, -TP * a * np.sin(TP * x2) * e


def psi_exact(a, x1, x2):
    return G1 * x2 - a * np.exp(-TP * x1) * np.sin(TP * x2)


def method_A(a, L1, x2s):
    out, tex, drift = [], [], 0.0

    def vel(t, y):
        v1, v2 = v_exact(a, y[0], y[1])
        return [v1, v2]

    def ev_in(t, y):
        return y[0]
    ev_in.terminal, ev_in.direction = True, -1

    def ev_out(t, y):
        return y[0] - L1
    ev_out.terminal, ev_out.direction = True, 1

    for x2 in x2s:
        if v_exact(a, X0, x2)[0] <= 0.0:
            out.append(SB)
            tex.append(np.nan)
            continue
        s = solve_ivp(vel, [0.0, 200.0], [X0, x2], method="DOP853", rtol=1e-12, atol=1e-14,
                      events=[ev_in, ev_out])
        out.append(IN if len(s.t_events[0]) else (OUT if len(s.t_events[1]) else BAD))
        tex.append(s.t[-1])
        drift = max(drift, abs(psi_exact(a, s.y[0, -1], s.y[1, -1]) - psi_exact(a, X0, x2)))
    return out, np.array(tex), drift


# ------------------------------------------------------------------------------------------------
# B: DP5(4) lock-step on the 2-D spline
# ------------------------------------------------------------------------------------------------
def method_B(kind1, a, n1, ny, h, x2s, hmax, tol=TOL, max_iter=400000):
    L1 = n1 * h
    x1c = (np.arange(n1) + 0.5) * h
    x2c = (np.arange(ny) + 0.5) * h
    f = a * np.exp(-TP * x1c)[:, None] * np.cos(TP * x2c)[None, :]
    sp = cc.Spline([(kind1, n1, h), ("per", ny, h)], f)
    Xp = 0.5 * L1

    def rhs(t, yy):
        _, gr = sp.eval(yy[..., :2])
        g1, g2 = G1 + gr[:, 0], gr[:, 1]
        gn = np.sqrt(g1 * g1 + g2 * g2)
        assert np.all(np.isfinite(gn)) and np.all(gn > 0.0)
        return np.stack([g1 / gn, g2 / gn, 1.0 / gn], axis=-1)  # k = 1

    P = len(x2s)
    y = np.zeros((P, 3))
    y[:, 0] = X0
    y[:, 1] = np.asarray(x2s)
    status = np.array([ACT] * P, dtype=object)
    _, gr0 = sp.eval(y[:, :2])
    sb = (G1 + gr0[:, 0]) <= 0.0
    status[sb] = SB
    hcur = np.full(P, 0.1 * hmax)
    plane = np.zeros(P, dtype=bool)
    tau_exit = np.full(P, np.nan)
    nrej = 0
    sarc = np.zeros(P)
    nacc = np.zeros(P, dtype=np.int64)
    for it in range(max_iter):
        idx = np.nonzero(status == ACT)[0]
        if idx.size == 0:
            break
        yA = y[idx]
        hA = hcur[idx][:, None]
        f1 = rhs(0.0, yA)
        y5, _, err = cc.dp5_step(rhs, 0.0, 0.0, yA, f1, hA, 2, tol)
        fac = cc.step_factor(err)
        acc = err <= 1.0
        rej = idx[~acc]
        hcur[rej] = hcur[rej] * fac[~acc]
        nrej += rej.size
        assert np.all(hcur[rej] >= 1e-14)
        ia = idx[acc]
        y[ia] = y5[acc]
        sarc[ia] += hcur[ia]
        nacc[ia] += 1
        # caps (closed streamlines / collapsed steps of a spurious field; B_per only): 5 L1 of arclength or
        # 20000 accepted steps (a regular trajectory needs ~L1/h_max = 512 / 1024)
        status[ia[(sarc[ia] > 5.0 * L1) | (nacc[ia] > 20000)]] = CAP
        hcur[ia] = np.minimum(hmax, hcur[ia] * fac[acc])
        x1n = y[ia, 0]
        plane[ia[(x1n >= Xp)]] = True
        e_in = ia[x1n < 0.0]
        e_out = ia[x1n >= L1]
        status[e_in] = IN
        status[e_out] = OUT
        tau_exit[e_in] = y[e_in, 2]
        tau_exit[e_out] = y[e_out, 2]
    else:
        raise RuntimeError("max_iter")
    return list(status), plane, tau_exit, nrej


# ------------------------------------------------------------------------------------------------
# C: Pollock RT0 on analytically integrated face fluxes
# ------------------------------------------------------------------------------------------------
def pollock_fluxes(a, n1, ny, h):
    X = np.arange(n1 + 1) * h
    Y = np.arange(ny + 1) * h
    ex = np.exp(-TP * X)
    # u[i, j] = (1/h) int_{Y_j}^{Y_j+h} v1(X_i, y) dy ;  w[i, j] = (1/h) int_{X_i}^{X_i+h} v2(x, Y_j) dx
    u = G1 - a * ex[:, None] * (np.sin(TP * Y[1:]) - np.sin(TP * Y[:-1]))[None, :] / h
    w = -a * (ex[:-1] - ex[1:])[:, None] * np.sin(TP * Y)[None, :] / h
    div = (u[1:] - u[:-1]) / h + (w[:, 1:] - w[:, :-1]) / h
    return u, w, np.max(np.abs(div))


def _exit_time(vp, A, r, h):
    d = np.where(vp > 0.0, h - r, -r)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = A * d / vp
        ok = (vp != 0.0) & (1.0 + z > 0.0)
        t = np.where(A == 0.0, d / vp, np.log1p(z) / A)
    return np.where(ok, t, np.inf)


def _advance(vp, A, r, t):
    with np.errstate(divide="ignore", invalid="ignore"):
        rn = np.where(A == 0.0, r + vp * t, r + vp * np.expm1(A * t) / A)
    return np.where(vp == 0.0, r, rn)


def method_C(a, n1, ny, h, x2s, max_iter=200000):
    u, w, divmax = pollock_fluxes(a, n1, ny, h)
    P = len(x2s)
    x2m = np.mod(np.asarray(x2s), 1.0)
    ci = np.full(P, int(math.floor(X0 / h)))
    r1 = np.full(P, X0 - ci[0] * h)
    cj = np.floor(x2m / h).astype(int) % ny
    r2 = np.clip(x2m - cj * h, 0.0, h)
    status = np.array([ACT] * P, dtype=object)
    clock = np.zeros(P)
    clock_exit = np.full(P, np.nan)
    plane = np.zeros(P, dtype=bool)
    stag = 0
    for it in range(max_iter):
        idx = np.nonzero(status == ACT)[0]
        if idx.size == 0:
            break
        i, j, ra, rb = ci[idx], cj[idx], r1[idx], r2[idx]
        um, up = u[i, j], u[i + 1, j]
        wm, wp = w[i, j], w[i, j + 1]
        A1, A2 = (up - um) / h, (wp - wm) / h
        vp1 = np.where(ra == h, up, A1 * ra + um)
        vp2 = np.where(rb == h, wp, A2 * rb + wm)
        t1 = _exit_time(vp1, A1, ra, h)
        t2 = _exit_time(vp2, A2, rb, h)
        te = np.minimum(t1, t2)
        dead = ~np.isfinite(te)
        if dead.any():
            status[idx[dead]] = BAD
            stag += int(dead.sum())
        live = ~dead
        idx, i, j, ra, rb = idx[live], i[live], j[live], ra[live], rb[live]
        vp1, vp2, A1, A2, t1, t2, te = (vp1[live], vp2[live], A1[live], A2[live], t1[live], t2[live],
                                        te[live])
        na = np.clip(_advance(vp1, A1, ra, te), 0.0, h)
        nb = np.clip(_advance(vp2, A2, rb, te), 0.0, h)
        ex1 = t1 == te
        ex2 = t2 == te
        s1 = np.where(vp1 > 0.0, 1, -1)
        s2 = np.where(vp2 > 0.0, 1, -1)
        na = np.where(ex1, np.where(s1 > 0, 0.0, h), na)
        nb = np.where(ex2, np.where(s2 > 0, 0.0, h), nb)
        ci[idx] = i + np.where(ex1, s1, 0)
        cj[idx] = (j + np.where(ex2, s2, 0)) % ny
        r1[idx], r2[idx] = na, nb
        clock[idx] += te
        plane[idx[ci[idx] >= n1 // 2]] = True
        out = idx[ci[idx] >= n1]
        inn = idx[ci[idx] < 0]
        status[out] = OUT
        status[inn] = IN
        clock_exit[out] = clock[out]
        clock_exit[inn] = clock[inn]
    else:
        raise RuntimeError("max_iter")
    return list(status), plane, clock_exit, divmax, stag


# ------------------------------------------------------------------------------------------------
def passes(status, plane, texit, T):
    """plane-pass and time-pass statuses at time T read off one trajectory."""
    pp = [IN if s == IN else (SB if s == SB else ("plane" if p else s)) for s, p in zip(status, plane)]
    tp = []
    for s, te in zip(status, texit):
        if s in (SB, BAD):
            tp.append(s)
        elif te <= T:
            tp.append(s)
        else:
            tp.append(ACT)
    return pp, tp


def compare(a, L1, n1, x2s, label, with_per=False):
    """Run A, B_nak, B_per, C on seeds x2s; return dict of per-seed outcomes and margins."""
    h = 1.0 / 32.0
    A_, tA, drift = method_A(a, L1, x2s)
    Bn, plN, tN, rejN = method_B("nak", a, n1, 32, h, x2s, h / 8.0)
    Bp, plP, tP, rejP = method_B("per", a, n1, 32, h, x2s, h / 8.0) if with_per else (None, None, None, None)
    C_, plC, tC, divmax, stag = method_C(a, n1, 32, h, x2s)
    psi0 = psi_exact(a, X0, np.asarray(x2s))
    mg = np.abs(psi0 - np.round(psi0))
    v1 = v_exact(a, X0, np.asarray(x2s))[0]
    return dict(x2=np.asarray(x2s), A=A_, tA=tA, drift=drift, Bn=Bn, plN=plN, tN=tN, Bp=Bp, plP=plP, tP=tP,
                C=C_, plC=plC, tC=tC, divmax=divmax, stag=stag, mg=mg, v1=v1, rejN=rejN, rejP=rejP)


def methods(R):
    return [m for m in (("B_nak", "Bn"), ("B_per", "Bp"), ("C Pollock", "C")) if R[m[1]] is not None]


def counts(lst, mask):
    s = [l for l, m in zip(lst, mask) if m]
    return {k: s.count(k) for k in sorted(set(s))}


def report(a, L1, n1, M, delta, v1min, R, T):
    inc = (R["mg"] >= delta) & (R["v1"] >= v1min)
    n_inc = int(inc.sum())
    print(f"  a={a} grid {n1}x32x32 (L1={L1:g}): M={M}  total x2 seeds {len(R['x2'])}  included {n_inc} "
          f"(v1 >= {v1min}, |psi0-psi_sep| >= {delta}); excluded {len(R['x2']) - n_inc}")
    print(f"    min |psi0 - psi_sep| over included = {R['mg'][inc].min():.4e}   min v1 included = "
          f"{R['v1'][inc].min():.4e}")
    print(f"    DOP853 max |psi_end - psi0| = {R['drift']:.2e};  Pollock max |div_cell| = {R['divmax']:.2e}")
    names = [("A analytic", "A", None), ("B_nak", "Bn", "plN"), ("B_per", "Bp", "plP"), ("C Pollock", "C", "plC")]
    names = [n for n in names if R[n[1]] is not None]
    tex = {"A": R["tA"], "Bn": R["tN"], "Bp": R["tP"], "C": R["tC"]}
    for nm, key, pk in names:
        if pk is None:
            pp = [("plane" if s == OUT else s) for s in R[key]]
            tp = [s if (s == IN or te <= T) else ACT for s, te in zip(R[key], R["tA"])]
        else:
            pp, tp = passes(R[key], R[pk], tex[key], T)
        print(f"    {nm:11s} plane pass {counts(pp, inc)}   time pass(T={T}) {counts(tp, inc)}")
    # disagreements (final exit kind) against A over the included seeds
    for nm, key in methods(R):
        bad = [(R["x2"][k], R["mg"][k], R["v1"][k], R["A"][k], R[key][k]) for k in np.nonzero(inc)[0]
               if R[key][k] != R["A"][k]]
        print(f"    disagreements {nm} vs A (included seeds): {len(bad)}")
        for b in bad[:12]:
            print(f"        x2={b[0]:+.5f} margin={b[1]:.3e} v1={b[2]:.3e} A={b[3]} {nm}={b[4]}")
    tex_out = R["tA"][inc & (np.array(R["A"], dtype=object) == OUT)]
    print(f"    exit time (A) of included outlet seeds: max {tex_out.max():.4f}   (recommended T = "
          f"{T})")


def scan(a, L1, n1, Ms, label):
    """Fine lattice: for each method list ALL disagreements vs A, with their margins (the delta needed)."""
    x2s = (np.arange(Ms) + 0.5) / Ms - 0.5
    R = compare(a, L1, n1, x2s, label)
    valid = np.array([s != SB for s in R["A"]])
    print(f"  scan a={a} grid {n1}x32x32: Ms={Ms}; analytic v1<=0 at the seed: {int((~valid).sum())}; "
          f"A: {counts(R['A'], np.ones(Ms, bool))}")
    for nm, key in methods(R):
        bad = [k for k in range(Ms) if R[key][k] != R["A"][k] and R["A"][k] != SB]
        if bad:
            mgb = R["mg"][bad]
            v1b = R["v1"][bad]
            print(f"    {nm}: {len(bad)} disagreements vs A among v1>0 seeds; max psi-margin of a "
                  f"disagreeing seed {mgb.max():.3e} (x2 {R['x2'][bad[int(np.argmax(mgb))]]:+.4f}); "
                  f"max v1 of a disagreeing seed {v1b.max():.3e}")
            kinds = {}
            for k in bad:
                kinds[(R['A'][k], R[key][k])] = kinds.get((R['A'][k], R[key][k]), 0) + 1
            print(f"        kinds (A,{nm}): {kinds}")
        else:
            print(f"    {nm}: 0 disagreements vs A among v1>0 seeds")
    return R


def dense_scans(a, L1, n1):
    """Dense scans through (i) the separatrix branch psi0 = 0 near |x2| ~ 0.26 and (ii) the v1 = 0 curve at the
    seed: where do B_nak and C start to disagree with A?  Returns printed lines."""
    from scipy.optimize import brentq
    xs = brentq(lambda x: psi_exact(a, X0, x), 0.2, 0.31)
    xv = brentq(lambda x: v_exact(a, X0, x)[0], 0.05, 0.2)
    out = []
    for name, x2s in (("separatrix branch (psi0 = 0 at x2 = %+.6f)" % xs, xs + np.linspace(-0.01, 0.01, 401)),
                      ("v1 = 0 curve at the seed (x2 = %+.6f)" % xv, xv + np.linspace(0.0, 0.03, 401)[1:])):
        R = compare(a, L1, n1, x2s, "dense")
        out.append(f"  a={a} grid {n1}x32x32, dense scan through the {name}: {len(x2s)} seeds, "
                   f"A {counts(R['A'], np.ones(len(x2s), bool))}")
        for nm, key in methods(R):
            bad = [k for k in range(len(x2s)) if R[key][k] != R["A"][k]]
            if bad:
                out.append(f"    {nm}: {len(bad)} disagreements; |psi0-psi_sep| of disagreeing seeds in "
                           f"[{R['mg'][bad].min():.3e}, {R['mg'][bad].max():.3e}], v1 in [{R['v1'][bad].min():.3e}, "
                           f"{R['v1'][bad].max():.3e}]; kinds "
                           f"{ {kk: sum(1 for k in bad if (R['A'][k], R[key][k]) == kk) for kk in set((R['A'][k], R[key][k]) for k in bad)} }")
            else:
                out.append(f"    {nm}: 0 disagreements")
    return out


# ================================================================================================
def main():
    print("C4 design emulation (numpy/scipy), NOT SF-41 evidence.  G1 = 1, X0 = 0.1, h = 1/32, grids 64x32x32 "
          "(L1 = 2) and 128x32x32 (L1 = 4).")
    for a in (0.5, 0.4):
        x1s = math.log(TP * a / G1) / TP
        print(f"a = {a}: stagnation point x1s = ln(2 pi a / G1)/(2 pi) = {x1s:.5f} (a > 1/(2 pi) = {1/TP:.5f}); "
              f"inlet backflow lobe on |x2| < arccos(e^(2 pi x1)/(2 pi a))/(2 pi) at x1 = 0: "
              f"{math.acos(1.0 / (TP * a)) / TP:.4f}")
    print()
    print("=== (1) fine scan lattice (x2 = (m+1/2)/Ms - 1/2, Ms = 512): every disagreement with A and its margin")
    SC = {}
    for a in (0.5, 0.4):
        for n1, L1 in ((64, 2.0), (128, 4.0)):
            SC[(a, n1)] = scan(a, L1, n1, 512, "scan")
    print()
    print("=== (2) candidate injection lattices x2 = (m+1/2)/M, m = 0..M-1 (the M x M injection lattice of the unit")
    print("    section has these M distinct x2 values, each repeated M times in x3): counts and disagreements")
    CAND = {}
    for a in (0.5, 0.4):
        for M in (32, 48, 64, 128):
            for n1, L1 in ((64, 2.0), (128, 4.0)):
                x2s = (np.arange(M) + 0.5) / M - 0.5
                CAND[(a, M, n1)] = compare(a, L1, n1, x2s, "cand")
    for a in (0.5, 0.4):
        for M in (32, 48, 64, 128):
            for n1, L1 in ((64, 2.0), (128, 4.0)):
                R = CAND[(a, M, n1)]
                for delta, v1min in ((0.01, 0.05), (0.02, 0.05), (0.03, 0.10)):
                    inc = (R["mg"] >= delta) & (R["v1"] >= v1min)
                    nb = {nm: sum(1 for k in np.nonzero(inc)[0] if R[key][k] != R["A"][k])
                          for nm, key in methods(R)}
                    cA = counts(R["A"], inc)
                    print(f"  a={a} M={M:3d} {n1:3d}x32x32 delta={delta:.2f} v1min={v1min:.2f}: included {int(inc.sum()):3d} "
                          f"of {M} x2 values = {int(inc.sum()) * M} 3-D seeds (A {cA}); disagreements {nb}")
    print()
    print("=== (3) recommended setting, full report")
    REC_M, REC_DELTA, REC_V1 = 64, 0.02, 0.05
    for a in (0.5, 0.4):
        for n1, L1 in ((64, 2.0), (128, 4.0)):
            R = CAND[(a, REC_M, n1)]
            inc = (R["mg"] >= REC_DELTA) & (R["v1"] >= REC_V1)
            texA = R["tA"][inc & (np.array(R["A"], dtype=object) == OUT)]
            T = math.ceil(1.25 * texA.max() * 2.0) / 2.0
            report(a, L1, n1, REC_M, REC_DELTA, REC_V1, R, T)
            print(f"    max exit tau over included outlet seeds: A {texA.max():.4f}  B_nak "
                  f"{np.nanmax(R['tN'][inc & (np.array(R['A'], dtype=object) == OUT)]):.4f}  C "
                  f"{np.nanmax(R['tC'][inc & (np.array(R['A'], dtype=object) == OUT)]):.4f}")
    print()
    print("=== (4) dense scans: where do the representations start to disagree with the analytic classification?")
    for a in (0.5, 0.4):
        for n1, L1 in ((64, 2.0), (128, 4.0)):
            for line in dense_scans(a, L1, n1):
                print(line)
    print()
    print("=== (5) diagnostic: the PERIODIC (SF-28) closure in x1 for the same f samples (B_per), a = 0.5, M = 64,")
    print("    same inclusion rule; capped at 5 L1 arclength or 20000 accepted steps")
    for n1, L1 in ((64, 2.0), (128, 4.0)):
        x2s = (np.arange(64) + 0.5) / 64 - 0.5
        R = compare(0.5, L1, n1, x2s, "per", with_per=True)
        inc = (R["mg"] >= REC_DELTA) & (R["v1"] >= REC_V1)
        print(f"  grid {n1}x32x32: included {int(inc.sum())}: A {counts(R['A'], inc)}  B_per {counts(R['Bp'], inc)}  "
              f"disagreements {sum(1 for k in np.nonzero(inc)[0] if R['Bp'][k] != R['A'][k])}")


if __name__ == "__main__":
    main()
