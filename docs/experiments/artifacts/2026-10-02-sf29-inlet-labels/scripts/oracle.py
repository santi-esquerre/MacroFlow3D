#!/usr/bin/env python3
"""SF-29 N1 entry point: Darcy reference, inlet labels, oracle labels and their checks.

Usage (run from this directory):
  python3 oracle.py --selftest
  python3 oracle.py field:eps:N [field:eps:N ...]          oracle labels + checks for each case
  python3 oracle.py --convergence field:eps [...]           oracle self-consistency over N = 16/32/48
  python3 oracle.py --nphi field:eps [...]                  reference resolution-independence ladder
  python3 oracle.py --returnmap field:eps [...]             closure-note return map at N_phi (positive controls)
  python3 oracle.py --midplane field:eps [...] --grids ...  3-plane mid-slab FD self-consistency (asymptotic-regime probe)
Options:
  --ref-nphi K      override the reference resolution N_phi (default: cases.NPHI table)
  --grids 16,32,48  grids of --convergence
  --no-cache        do not read/write raw/cache

Every metric line uses the shared CASE format of metrics.case_line (cand=oracle).
"""
import sys
import time

import numpy as np

import reference as R
import inlet as I
import tracing as T
import cases as C
import metrics as M


# ------------------------------------------------------------------------------------------------
def _hdr(msg):
    print("=" * 8 + " " + msg, flush=True)


def report_case(field, eps, N, nphi=None, use_cache=True):
    t0 = time.time()
    try:
        case = C.load_case(field, eps, N, nphi=nphi, use_cache=use_cache, verbose=False)
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return None
    meta = case["meta"]; ref = case["ref"]; inl = case["inlet"]
    psi1, psi2 = case["psi_or"]; vD = case["vD"]
    m = M.fd_metrics(psi1, psi2, vD, psi_or=None)
    m["e_psi"] = 0.0      # oracle against itself (trivial)
    rt = np.asarray(meta["roundtrip"]); nfev = np.asarray(meta["nfev"]); nfev_rt = np.asarray(meta["nfev_rt"])
    print("REF field=%s eps=%g nphi=%d shape=%s pcg_res=%.1e its=%d max(d1phi)=%+.4f (v1>0 OK) a=(%+.8f,%+.2e,%+.2e) "
          "modes=%d nf=%d Q0-1=%+.1e cont_res=%.1e inlet_min_v1=%.4f t_ref=%.1fs"
          % (field, eps, ref.nphi, ref.shape, ref.pcg[0], ref.pcg[1], ref.max_d1phi, ref.a[0], ref.a[1], ref.a[2],
             ref.field.nmodes, inl.nf, inl.Q0 - 1.0, ref.continuity_residual(), inl.vmin, ref.t_solve), flush=True)
    print("ORACLE field=%s eps=%g N=%d planes=%d roundtrip_max=%.1e nfev_sum=%d nfev_rt_sum=%d nfev_max=%d "
          "t_oracle=%.1fs t_faces=%.1fs cache_hit=%s"
          % (field, eps, N, N, rt.max(), nfev.sum(), nfev_rt.sum(), nfev.max(), meta["t_oracle"], meta["t_faces"],
             meta["cache_hit"]), flush=True)
    # mean flux per vertex plane (exact for trig interpolants on a periodic plane only to spectral accuracy)
    flux = np.array([vD[0][j].mean() for j in range(N + 1)])
    fdiv = C.face_divergence(case["vD_faces"], N)
    print("CHECK field=%s eps=%g N=%d vertex-plane mean(v1)-1 (N-point rule): max=%.1e | face-flux div_h: rms/v_rms=%.1e max/v_rms=%.1e "
          "| f1 mean-1: max=%.1e" % (field, eps, N, np.abs(flux - 1).max(), M.rms(fdiv) / m["v_rms"],
                                     np.abs(fdiv).max() / m["v_rms"],
                                     np.abs(case["vD_faces"]["f1"].mean(axis=(1, 2)) - 1).max()), flush=True)
    if R.is_constant_head(field):
        x, Y, Z = C.vertex_coords(N)
        vin = ref.velocity(0.0, Y, Z); vout = ref.velocity(1.0, Y, Z)
        print("CHECK field=%s eps=%g constant-head faces: max|v_perp| inlet=%.1e outlet=%.1e (v_rms=%.3f) "
              "transverse_flux=(%.1e,%.1e)" % (field, eps, np.abs(vin[:, 1:]).max(), np.abs(vout[:, 1:]).max(),
                                                m["v_rms"], ref.transverse_flux[0], ref.transverse_flux[1]),
              flush=True)
    else:
        rm = T.return_map_nonuniform(ref)
        print("CHECK field=%s eps=%g return map (closure-note points, 16): nonuniform_rms=%.3e max|d|=%.3e "
              "mean=(%+.2e,%+.2e) roundtrip=%.1e" % (field, eps, rm["nonuniform_rms"], rm["dmax"], rm["mean"][0],
                                                     rm["mean"][1], rm["roundtrip"]), flush=True)
    d_out = max(np.abs(psi1[N] - psi1[0]).max(), np.abs(psi2[N] - psi2[0]).max())
    a2, a3 = M.affine(N)
    d_aff = max(np.abs(psi1 - a2).max(), np.abs(psi2 - a3).max())
    print("CHECK field=%s eps=%g N=%d max|psi_or(outlet)-psi0(inlet)|=%.1e max|psi_or-affine|=%.3e min|c|/v_rms=%.3f "
          "|vD| p0.1/1/5/50 = %.4f/%.4f/%.4f/%.4f min|vD|=%.4f"
          % (field, eps, N, d_out, d_aff, m["min_c"] / m["v_rms"], m["vD_p"][0], m["vD_p"][1], m["vD_p"][2],
             m["vD_p"][3], m["vD_min"]), flush=True)
    M.print_case(field, eps, N, "oracle", m, t=time.time() - t0)
    return case, m


# ------------------------------------------------------------------------------------------------
def convergence(field, eps, grids, nphi=None, use_cache=True):
    rows = []
    for N in grids:
        out = report_case(field, eps, N, nphi=nphi, use_cache=use_cache)
        if out is None:
            return
        rows.append(out[1])
    for key in ("e_v", "e_i1", "e_i2", "e_div"):
        e = [r[key] for r in rows]
        p = M.orders(e, grids)
        print("ORDER field=%s eps=%g %s: %s | orders %s" % (field, eps, key, " ".join("%.3e" % x for x in e),
                                                          " ".join("%.2f" % x for x in p)), flush=True)
    print("ORDER field=%s eps=%g min|c|: %s" % (field, eps, " ".join("%.4f" % r["min_c"] for r in rows)),
          flush=True)


# ------------------------------------------------------------------------------------------------
def _labels_at(ref, inl, Y, Z):
    p0 = inl.labels(Y, Z)
    fy, fz, info = T.trace_to_inlet(ref, 1.0, Y, Z, roundtrip=False)
    p1 = inl.labels(fy, fz)
    return p0, p1, info["nfev"]


def _reldiff(pa, pb, Y, Z):
    """max_i RMS(psi_i^a - psi_i^b) / max_i RMS(psi_i^b - affine_i).

    The denominator is the larger periodic part of the pair (for control2d psi2^0 = x3 exactly and
    its own periodic part vanishes).
    """
    aff = (Y, Z)
    return max(M.rms(pa[i] - pb[i]) for i in range(2)) / max(M.rms(pb[i] - aff[i]) for i in range(2))


def nphi_ladder(field, eps, ntest=16, ladder=C.NPHI_LADDER, tol=C.NPHI_TOL):
    _, Y, Z = C.vertex_coords(ntest)
    refs = {}

    def get(n):
        if n not in refs:
            ref = R.DarcyReference(field, eps, n)
            ref.check_positive_v1()
            inl = I.InletLabels(ref, C.nf_for(n))
            t0 = time.time()
            p0, p1, nfev = _labels_at(ref, inl, Y, Z)
            refs[n] = (ref, inl, p0, p1, nfev, time.time() - t0, ref.continuity_residual())
        return refs[n]

    chosen = None
    for n in ladder:
        n2 = int(round(1.5 * n))
        try:
            ra = get(n); rb = get(n2)
        except RuntimeError as exc:
            print(str(exc), flush=True)
            return None
        din = _reldiff(ra[2], rb[2], Y, Z)
        dout = _reldiff(ra[3], rb[3], Y, Z)
        ok = din < tol and dout < tol
        print("NPHI field=%s eps=%g nphi=%d vs %d | inlet_rel=%.2e outlet_rel=%.2e | cont_res=%.1e/%.1e Q0-1=%+.1e/%+.1e "
              "| pcg_its=%d/%d max(d1phi)=%+.4f modes=%d t_ref=%.1f/%.1fs t_trace=%.1f/%.1fs nfev=%d | %s"
              % (field, eps, n, n2, din, dout, ra[6], rb[6], ra[1].Q0 - 1, rb[1].Q0 - 1, ra[0].pcg[1], rb[0].pcg[1],
                 rb[0].max_d1phi, ra[0].field.nmodes, ra[0].t_solve, rb[0].t_solve, ra[5], rb[5], ra[4],
                 "PASS" if ok else "fail"), flush=True)
        if ok:
            chosen = n
            break
    print("NPHI_TABLE field=%s eps=%g N_phi=%s (tol %.0e, %d^2 outlet test points)"
          % (field, eps, chosen if chosen else "NOT FOUND <= %d" % ladder[-1], tol, ntest), flush=True)
    return chosen


# ------------------------------------------------------------------------------------------------
def midplane(field, eps, grids, x1m=0.5, nphi=None):
    """Asymptotic-regime probe: oracle labels on the three planes x1m - h, x1m, x1m + h only.

    Centered second-order FD at the middle plane (no boundary stencils), c = grad_h psi1 x grad_h psi2
    vs v_D there: e_v_mid, e_i_mid and observed orders, for grids beyond the 16/32/48 matrix at the
    cost of 3 planes per grid.  N must make x1m a vertex (x1m N integer).
    """
    try:
        ref, inl = C.build_reference(field, eps, nphi, verbose=False)
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return
    ev = []; ei = []
    for N in grids:
        t0 = time.time()
        h = 1.0 / N
        _, Y, Z = C.vertex_coords(N)
        P = []
        nf = 0; rt = 0.0
        for x1 in (x1m - h, x1m, x1m + h):
            fy, fz, info = T.trace_to_inlet(ref, x1, Y, Z, roundtrip=True)
            p1, p2 = inl.labels(fy, fz)
            P.append((p1.reshape(N, N) - Y.reshape(N, N), p2.reshape(N, N) - Z.reshape(N, N)))
            nf += info["nfev"] + info["nfev_rt"]; rt = max(rt, info["roundtrip"])
        g = []
        for i in range(2):
            u = [P[k][i] for k in range(3)]
            d1 = (u[2] - u[0]) / (2 * h)
            d2 = (np.roll(u[1], -1, 0) - np.roll(u[1], 1, 0)) / (2 * h)
            d3 = (np.roll(u[1], -1, 1) - np.roll(u[1], 1, 1)) / (2 * h)
            gi = [d1, d2, d3]; gi[1 + i] = gi[1 + i] + 1.0
            g.append(gi)
        c = M.cross(g[0], g[1])
        v = ref.velocity(x1m, Y, Z)
        vD = [v[:, k].reshape(N, N) for k in range(3)]
        vr = M.rms_vec(vD)
        e_v = M.rms_vec([c[k] - vD[k] for k in range(3)]) / vr
        e_i = max(M.rms(vD[0] * gi[0] + vD[1] * gi[1] + vD[2] * gi[2]) / (vr * M.rms_vec(gi)) for gi in g)
        gmax = max(float(np.sqrt(gi[0]**2 + gi[1]**2 + gi[2]**2).max()) for gi in g)
        cn = np.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
        ev.append(e_v); ei.append(e_i)
        print("MIDPLANE field=%s eps=%g nphi=%d x1=%.3f N=%d e_v_mid=%.3e e_i_mid=%.3e max|grad_h psi|=%.2f min|c|=%.4f "
              "min|vD|=%.4f roundtrip=%.1e nfev=%d t=%.1fs"
              % (field, eps, ref.nphi, x1m, N, e_v, e_i, gmax, cn.min(),
                 float(np.sqrt(vD[0]**2 + vD[1]**2 + vD[2]**2).min()), rt, nf, time.time() - t0), flush=True)
    print("MIDPLANE_ORDER field=%s eps=%g grids=%s e_v_mid orders %s | e_i_mid orders %s"
          % (field, eps, ",".join(str(n) for n in grids), " ".join("%.2f" % x for x in M.orders(ev, grids)),
             " ".join("%.2f" % x for x in M.orders(ei, grids))), flush=True)


# ------------------------------------------------------------------------------------------------
CLOSURE_NOTE = {   # non-uniform RMS return-map displacement recorded in raw/closure.txt of the 2026-10-02 probes
    ("lester_brk", 1.0): 8.691e-03, ("gauss", 0.25): 4.655e-03, ("gauss", 0.5): 1.989e-02, ("gauss", 1.0): 8.559e-02,
    ("control2d", 0.5): 7.123e-13, ("lester2021", 1.0): 8.278e-15,
}


def returnmap(field, eps, nphi=None):
    """Closure-note return map (16 points, seed 3) of the reference at N_phi, vs the recorded value."""
    try:
        ref, inl = C.build_reference(field, eps, nphi, verbose=False)
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return
    rm = T.return_map_nonuniform(ref)
    note = CLOSURE_NOTE.get((field, float(eps)))
    print("RETURNMAP field=%s eps=%g nphi=%d nonuniform_rms=%.3e max|d|=%.3e mean=(%+.2e,%+.2e) roundtrip=%.1e nfev=%d "
          "| closure note: %s" % (field, eps, ref.nphi, rm["nonuniform_rms"], rm["dmax"], rm["mean"][0], rm["mean"][1],
                                   rm["roundtrip"], rm["nfev"], ("%.3e" % note) if note is not None else "n/a"),
          flush=True)


# ------------------------------------------------------------------------------------------------
def selftest():
    import closure_probe as cp
    ok = True

    def check(name, val, thr):
        nonlocal ok
        good = bool(val <= thr)
        ok = ok and good
        print("SELFTEST %-58s %.2e <= %.0e  %s" % (name, val, thr, "PASS" if good else "FAIL"), flush=True)

    # 1. darcy_spectral_box bit-identical to the closure-probe original on the cubic path
    X, Y, Z = cp.grid(16)
    for fname, eps in (("gauss", 0.5), ("generic3d", 0.25)):
        f = eps * cp.FIELDS[fname](X, Y, Z)
        same = True
        for j in range(3):
            a = cp.darcy_spectral(f, j); b = R.darcy_spectral_box(f, j)
            same = same and np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2] and np.array_equal(a[3], b[3])
        check("darcy_spectral_box == closure_probe (bitwise) %s:%g" % (fname, eps), 0.0 if same else 1.0, 0.0)
    # 2. separable trig evaluation == closure_probe.TrigField
    ref = R.DarcyReference("gauss", 0.5, 16)
    tf = cp.TrigField(ref.u, -ref.a)
    rng = np.random.default_rng(11)
    pts = rng.random((64, 3)); pts[:, 0] = 0.3711
    check("SepTrigField vs closure_probe.TrigField (max abs)", np.abs(tf.grad(pts) - ref.grad_phi(0.3711, pts[:, 1], pts[:, 2])).max(), 1e-13)
    # 3. analytic grad ln k (complex step) vs central differences
    for fname in ("gauss", "gauss_ch", "lester2021"):
        lk = R.LogConductivity(fname, 0.5)
        p = rng.random((32, 3)); hh = 2e-4
        g = lk.grad_lnk(p[:, 0], p[:, 1], p[:, 2])
        err = 0.0
        for i in range(3):
            def cd(step):
                e = np.zeros(3); e[i] = step
                return (lk.lnk(*(p + e).T) - lk.lnk(*(p - e).T)) / (2 * step)
            fd = (4.0 * cd(hh / 2) - cd(hh)) / 3.0          # Richardson, O(h^4)
            err = max(err, np.abs(fd - g[i]).max())
        check("grad ln k complex step vs Richardson central FD %s" % fname, err, 1e-8)
    # 4. k = 1 positive control: affine labels at roundoff
    case = C.load_case("uniform", 0.0, 8, nphi=8, use_cache=False, write_cache=False, verbose=False)
    a2, a3 = M.affine(8)
    check("k=1: max|psi_or - (x2,x3)| (all vertices)", max(np.abs(case["psi_or"][0] - a2).max(),
                                                         np.abs(case["psi_or"][1] - a3).max()), 1e-14)
    m = M.fd_metrics(case["psi_or"][0], case["psi_or"][1], case["vD"])
    check("k=1: e_v of FD-reconstructed oracle flow", m["e_v"], 1e-14)
    # 5. inlet labels: face-Jacobian identity, affine + periodic structure
    # (the 'V vs true v1' and '|Q0 - 1|' rows are spectral-resolution diagnostics: they pass at an
    #  adequate N_phi, cf. --nphi; the identities are exact for the representation)
    for fname, eps, nphi in (("gauss", 0.5, 32), ("generic3d", 0.5, 24), ("gauss_ch", 0.25, 48)):
        ref = R.DarcyReference(fname, eps, nphi); inl = I.InletLabels(ref, C.nf_for(nphi))
        y = rng.random(200) * 3 - 1; z = rng.random(200) * 3 - 1
        J = inl.jacobian(y, z); V = inl.V_eval(y, z)
        check("%s:%g face Jacobian - V (representation), rel" % (fname, eps), np.abs(J - V).max() / np.abs(V).max(), 1e-13)
        vtrue = ref.velocity(0.0, np.mod(y, 1.0), np.mod(z, 1.0))[:, 0]
        check("%s:%g V vs true v1 at random face points, rel, nphi=%d" % (fname, eps, nphi), np.abs(V - vtrue).max() / np.abs(vtrue).max(), 1e-9)
        # independent: FFT-differentiate sampled labels on a 2*nf grid, Jacobian vs true v1
        Mf = 2 * inl.nf
        s = np.arange(Mf) / float(Mf)
        Yg, Zg = np.meshgrid(s, s, indexing="ij")
        p1, p2 = inl.labels(Yg.ravel(), Zg.ravel())
        u1 = p1.reshape(Mf, Mf) - Yg; u2 = p2.reshape(Mf, Mf) - inl.Q0 * Zg
        kk = 2 * np.pi * np.fft.fftfreq(Mf, d=1.0 / Mf)
        U1 = np.fft.fft2(u1); U2 = np.fft.fft2(u2)
        d2u1 = np.fft.ifft2(1j * kk[:, None] * U1).real; d3u1 = np.fft.ifft2(1j * kk[None, :] * U1).real
        d2u2 = np.fft.ifft2(1j * kk[:, None] * U2).real; d3u2 = np.fft.ifft2(1j * kk[None, :] * U2).real
        Jf = (1 + d2u1) * (inl.Q0 + d3u2) - d3u1 * d2u2
        vt = ref.velocity(0.0, Yg.ravel(), Zg.ravel())[:, 0].reshape(Mf, Mf)
        check("%s:%g FFT-differentiated labels Jacobian vs true v1, rel" % (fname, eps), np.abs(Jf - vt).max() / np.abs(vt).max(), 1e-9)
        check("%s:%g max|d2 psi2^0| (psi2 independent of x2)" % (fname, eps), np.abs(d2u2).max(), 1e-12)
        j1 = np.abs(inl.psi1(y + 1.0, z) - inl.psi1(y, z) - 1.0).max()
        j1b = np.abs(inl.psi1(y, z + 1.0) - inl.psi1(y, z)).max()
        j2 = np.abs(inl.psi2(z + 1.0) - inl.psi2(z) - inl.Q0).max()
        check("%s:%g psi1 jump 1 in x2, 0 in x3; psi2 jump Q0 in x3" % (fname, eps), max(j1, j1b, j2), 1e-13)
        check("%s:%g |Q0 - 1| (face flux) at nphi=%d" % (fname, eps, nphi), abs(inl.Q0 - 1.0), 1e-9)
    # 6. round trip of the tracer
    ref = R.DarcyReference("gauss", 1.0, 16)
    yy = rng.random(64); zz = rng.random(64)
    fy, fz, info = T.trace_to_inlet(ref, 1.0, yy, zz)
    check("tracer round trip gauss:1 x1 1->0->1 (64 pts)", info["roundtrip"], 1e-11)
    # 7. constant-head faces of the mirror construction
    for nphi in (16, 24):
        ref = R.DarcyReference("gauss_ch", 0.25, nphi)
        s = np.arange(32) / 32.0
        Yg, Zg = np.meshgrid(s, s, indexing="ij")
        vin = ref.velocity(0.0, Yg.ravel(), Zg.ravel()); vout = ref.velocity(1.0, Yg.ravel(), Zg.ravel())
        vmax = max(np.abs(vin[:, 1:]).max(), np.abs(vout[:, 1:]).max())
        check("gauss_ch:0.25 nphi=%d max|v_perp| on x1=0 and x1=1" % nphi, vmax, 1e-14)
        check("gauss_ch:0.25 nphi=%d transverse mean flux" % nphi, np.abs(ref.transverse_flux).max(), 1e-14)
        # odd fluctuation potential: u(x1) = -u(-x1) on the grid
        u = ref.u
        refl = np.roll(u[::-1], 1, axis=0)
        check("gauss_ch:0.25 nphi=%d oddness of u about x1=0 (rel)" % nphi, np.abs(u + refl).max() / np.abs(u).max(), 1e-12)
    # 8. face fluxes discretely conservative to quadrature accuracy; CASE line parses
    ref = R.DarcyReference("gauss", 0.5, 32)
    for nn, thr in ((8, 1e-2), (16, 1e-4)):
        fl = C.face_fluxes(ref, nn)
        dv = C.face_divergence(fl, nn)
        check("gauss:0.5 nphi=32 face-flux div_h (N=%d, GL%d) max / mean f1" % (nn, C.NG), np.abs(dv).max() / fl["f1"].mean(), thr)
    line = M.case_line("gauss", 0.5, 16, "oracle", {"e_v": 1.0, "e_psi": 0.0, "e_i1": 2.0, "e_i2": 3.0, "e_div": 4.0,
                                                    "min_c": 5.0, "p0.1": 6, "p1": 7.0, "p5": 8.0, "p50": 9.0},
                       r_F=None, its=3, t=1.0)
    pp = M.parse_case_line(line)
    check("CASE line round trip", 0.0 if (pp["e_v"] == 1.0 and pp["e_i"] == (2.0, 3.0) and pp["N"] == 16) else 1.0, 0.0)
    # C-i4: CASE line with the per-label suffix round trips, and a pre-C-i4 line (no suffix) still parses
    line = M.case_line("gauss", 0.5, 16, "i1", {"e_v": 1.0, "e_psi": 0.5, "e_psi1": 0.5, "e_psi2": 0.25}, its=3, t=1.0)
    pp = M.parse_case_line(line)
    old = ("CASE field=control2d eps=0.25 N=16 cand=i1 | r_F=1.296e-14 its=6 | e_v=2.047e-03 e_psi=4.668e-01 "
           "e_i=(1.585e-03,1.311e-03) e_div=1.204e-03 min_c=9.008e-01 p0.1=9.024e-01 p1=9.082e-01 p5=9.230e-01 "
           "p50=1.008e+00 | t=6.3")
    po = M.parse_case_line(old)
    check("CASE line with e_psi1/e_psi2 suffix round trip; old line parses",
          0.0 if (pp["e_psi1"] == 0.5 and pp["e_psi2"] == 0.25 and pp["t"] == 1.0 and po["e_psi"] == 0.4668
                  and po["t"] == 6.3 and "e_psi1" not in po) else 1.0, 0.0)
    # C-i4 metric fix: control2d psi2 = Q0 x3 has a periodic part at roundoff; per-label e_psi must be sane for
    # oracle-vs-perturbed-oracle (the pre-C-i4 1e-12 threshold divided roundoff-level differences by ~1e-13)
    case = C.load_case("control2d", 0.25, 16, verbose=False)
    po1, po2 = case["psi_or"]
    a2, a3 = M.affine(16)
    den = (M.rms(po1 - a2), M.rms(po2 - a3))
    print("SELFTEST   control2d:0.25 N=16 oracle periodic parts: den1=%.3e den2=%.3e ratio den2/den1=%.1e"
          % (den[0], den[1], den[1] / den[0]), flush=True)
    rng2 = np.random.default_rng(4)
    n1 = rng2.standard_normal(po1.shape); n2 = rng2.standard_normal(po2.shape)
    delta = 1e-3 * max(den)
    mp = M.fd_metrics(po1 + delta * n1, po2 + delta * n2, case["vD"], case["psi_or"])
    exp1 = M.rms(delta * n1) / den[0]; exp2 = M.rms(delta * n2) / max(den)
    print("SELFTEST   control2d:0.25 perturbed oracle (delta = 1e-3 max den): e_psi1=%.3e (expected %.3e) e_psi2=%.3e "
          "(expected %.3e, normalized by den1) a_psi1=%.3e a_psi2=%.3e | the pre-C-i4 normalization would give "
          "e_psi2=%.3e" % (mp["e_psi1"], exp1, mp["e_psi2"], exp2, mp["a_psi1"], mp["a_psi2"],
                            M.rms(delta * n2) / den[1]), flush=True)
    check("control2d:0.25 per-label e_psi of a 1e-3 perturbation, rel. dev.",
          max(abs(mp["e_psi1"] / exp1 - 1), abs(mp["e_psi2"] / exp2 - 1)), 1e-10)
    mo = M.fd_metrics(po1, po2, case["vD"], case["psi_or"])
    check("control2d:0.25 oracle vs itself: e_psi1, e_psi2", max(mo["e_psi1"], mo["e_psi2"]), 0.0)
    # C-i4: 4th-order reconstruction of the oracle labels, observed order of e_v on 16/24/32 (gauss:0.25)
    grids4 = (16, 24, 32)
    ev4 = []; ev2 = []
    for nn in grids4:
        cs = C.load_case("gauss", 0.25, nn, verbose=False)
        p = cs["psi_or"]
        ev4.append(M.fd_metrics(p[0], p[1], cs["vD"], order=4)["e_v"])
        ev2.append(M.fd_metrics(p[0], p[1], cs["vD"], order=2)["e_v"])
    o4 = M.orders(ev4, list(grids4)); o2 = M.orders(ev2, list(grids4))
    print("SELFTEST   gauss:0.25 oracle e_v over N=16/24/32: order-4 FD %s (orders %s) | order-2 FD %s (orders %s)"
          % (" ".join("%.3e" % x for x in ev4), " ".join("%.2f" % x for x in o4), " ".join("%.3e" % x for x in ev2),
             " ".join("%.2f" % x for x in o2)), flush=True)
    check("gauss:0.25 oracle e_v order-4 FD: observed order (24->32) >= 3.5", max(0.0, 3.5 - o4[-1]), 0.0)
    print("SELFTEST %s" % ("ALL PASS" if ok else "FAILURES"), flush=True)
    return ok


# ------------------------------------------------------------------------------------------------
def main(argv):
    use_cache = "--no-cache" not in argv
    nphi = None
    grids = (16, 32, 48)
    args = []
    mode = "cases"
    it = iter(argv)
    for a in it:
        if a == "--selftest":
            mode = "selftest"
        elif a == "--convergence":
            mode = "convergence"
        elif a == "--nphi":
            mode = "nphi"
        elif a == "--returnmap":
            mode = "returnmap"
        elif a == "--midplane":
            mode = "midplane"
        elif a == "--no-cache":
            pass
        elif a == "--ref-nphi":
            nphi = int(next(it))
        elif a == "--grids":
            grids = tuple(int(x) for x in next(it).split(","))
        else:
            args.append(a)
    print("# oracle.py %s | python %s numpy %s" % (" ".join(argv), sys.version.split()[0], np.__version__), flush=True)
    t0 = time.time()
    if mode == "selftest":
        ok = selftest()
        print("# total %.1fs" % (time.time() - t0))
        return 0 if ok else 1
    for spec in args:
        field, eps, N = C.parse_spec(spec)
        if mode == "cases":
            if N is None:
                raise SystemExit("case spec needs N: field:eps:N")
            report_case(field, eps, N, nphi=nphi, use_cache=use_cache)
        elif mode == "convergence":
            convergence(field, eps, grids, nphi=nphi, use_cache=use_cache)
        elif mode == "nphi":
            nphi_ladder(field, eps)
        elif mode == "returnmap":
            returnmap(field, eps, nphi=nphi)
        elif mode == "midplane":
            midplane(field, eps, grids, nphi=nphi)
    print("# total %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
