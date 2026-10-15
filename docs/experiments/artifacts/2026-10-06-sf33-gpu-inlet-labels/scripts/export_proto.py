#!/usr/bin/env python3
"""SF-33 N4: export the SF-29 prototype inputs (and saved prototype solutions) as C-order float64 `.npy` files
for the GPU inlet-label solver (claim (a): identical inputs -> identical metrics).

The SF-29 artifact (docs/experiments/artifacts/2026-10-02-sf29-inlet-labels/) is imported READ-ONLY by sys.path
insertion of its scripts/ directory, exactly as its reference.py imports the 2026-10-02 closure probes.  Nothing is
written there except the oracle caches that `cases.load_case` writes by design into its gitignored raw/cache/
(N != 16 only: the N = 16 cache files are committed there and un-ignored by `!*_N16_*`, so a missing N = 16 cache is
computed but NOT written, which keeps the SF-29 tree clean in git).

Usage (run from this directory; numpy/scipy only; numpy 1.26 compatible):
  python3 export_proto.py field:eps:N [...] --out DIR        case dirs DIR/<field>_<eps:%g>_<N>/ (see README.md)
  python3 export_proto.py --list-amplitudes EPS [...]        reachable continuation amplitudes for a target EPS
  python3 export_proto.py --solutions FILE.npz [...] --out DIR      saved prototype solutions -> DIR/<name>/
  python3 export_proto.py --crosscheck field:eps:N [...] --out DIR  SF-19 cross-check inputs (step 8)
  python3 export_proto.py --selftest                         amplitude-rule checks + 16^3 gauss:0.25 export/reload
Options: --bisect K (continuation bisections, default 4 = candidate_i default)  --no-stages (main files only)
"""
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import scipy

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
SF29_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "2026-10-02-sf29-inlet-labels"))
SF29_SCRIPTS = os.path.join(SF29_DIR, "scripts")
if SF29_SCRIPTS not in sys.path:
    sys.path.insert(0, SF29_SCRIPTS)
import cases as C          # noqa: E402  (read-only reuse of the SF-29 artifact)
import candidate_i as CI   # noqa: E402
import metrics as M        # noqa: E402

EXPORT_VERSION = 1
BISECT_DEFAULT = 4          # candidate_i.py --bisect default
AMP_TOL = 1e-12             # the amplitude comparisons of candidate_i.amplitude_path / solve_case
STAGE_FILES = ("lnk", "grad_lnk_1", "grad_lnk_2", "grad_lnk_3", "psi0_1", "psi0_2", "vperp_2", "vperp_3", "v_rms")


def _rel(path):
    return os.path.relpath(os.path.abspath(path), _REPO)


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


# ------------------------------------------------------------------------------------------------------------------
# reachable continuation amplitudes (the rule of candidate_i.solve_case)
# ------------------------------------------------------------------------------------------------------------------
AMPLITUDE_RULE = (
    "candidate_i.solve_case: todo = amplitude_path(eps) = [a in EPS_LADDER=(0.25, 0.5, 1.0) with a < eps - 1e-12] + "
    "[eps]; e_conv = 0.  Attempt todo[0]: on acceptance e_conv = todo.pop(0); on failure, if fewer than `bisect` "
    "(default 4) bisections were used in total, insert 0.5*(e_conv + failed amplitude) at the front; otherwise give "
    "up and, if the failed amplitude is not eps, make one final attempt at eps.  The reachable set is every "
    "amplitude attempted along any accept/fail outcome sequence (exhaustive recursion over (todo, e_conv, "
    "bisections used)); all amplitudes are dyadic, hence exact in float64 and in '%g'.")


def amplitude_path(eps):
    return [float(a) for a in CI.amplitude_path(float(eps), True)]


def reachable_amplitudes(eps, bisect=BISECT_DEFAULT):
    """Every amplitude the continuation of candidate_i.solve_case can attempt for target `eps` (sorted)."""
    eps = float(eps)
    out = set()
    seen = set()

    def walk(todo, e_conv, nbis):
        key = (todo, e_conv, nbis)
        if key in seen or not todo:
            return
        seen.add(key)
        e = todo[0]
        out.add(e)
        walk(todo[1:], e, nbis)                                    # stage accepted
        if nbis < bisect:                                          # stage failed: bisect from e_conv
            walk((0.5 * (e_conv + e),) + todo, e_conv, nbis + 1)
        elif abs(e - eps) > AMP_TOL:                               # failed, gave up: final attempt at eps
            out.add(eps)

    walk(tuple(amplitude_path(eps)), 0.0, 0)
    return sorted(out)


def simulate_path(eps, outcomes, bisect=BISECT_DEFAULT):
    """Replay the rule with a given accept (True) / fail (False) outcome per attempted stage -> PATH tokens."""
    todo = amplitude_path(eps)
    e_conv = 0.0; nbis = 0; taken = []; it = iter(outcomes)
    while todo:
        e = todo[0]
        ok = next(it)
        taken.append("%g%s" % (e, "" if ok else "(fail)"))
        if ok:
            e_conv = e; todo.pop(0)
        elif nbis < bisect:
            nbis += 1; todo.insert(0, 0.5 * (e_conv + e))
        else:
            if abs(e - eps) > AMP_TOL:
                taken.append("%g(final)" % eps)
            break
    return taken


def parse_summary_paths(path=None):
    """(field, eps, cand, N, path string) of every PATH line of the SF-29 raw/sweep2/summary.md."""
    path = path or os.path.join(SF29_DIR, "raw", "sweep2", "summary.md")
    out = []; field = eps = cand = None
    rs = re.compile(r"^## (\S+), eps = (\S+)\s*$")
    rc = re.compile(r"^### candidate `([^`]+)`")
    rp = re.compile(r"^- PATH N=(\d+): `([^`]*)`")
    with open(path) as fh:
        for line in fh:
            m = rs.match(line)
            if m:
                field, eps, cand = m.group(1), float(m.group(2)), None
                continue
            m = rc.match(line)
            if m:
                cand = m.group(1)
                continue
            if line.startswith("### ") or line.startswith("## "):
                cand = None
                continue
            m = rp.match(line)
            if m and cand is not None:
                out.append((field, eps, cand, int(m.group(1)), m.group(2)))
    return out


def path_tokens(s):
    """'0.25(fail)->0.125->...' -> [(amp, kind)] with kind in {'ok', 'fail', 'final'}."""
    toks = []
    for t in s.split("->"):
        m = re.match(r"^([0-9.eE+-]+)(?:\((fail|final)\))?$", t.strip())
        if not m:
            raise ValueError("unparseable PATH token %r in %r" % (t, s))
        toks.append((float(m.group(1)), m.group(2) or "ok"))
    return toks


def amp_name(a):
    """Directory name of an amplitude: '%g' (C printf-compatible); refuses names that do not round-trip."""
    s = "%g" % a
    if float(s) != float(a):
        raise ValueError("amplitude %r does not round-trip through '%%g' (%s)" % (a, s))
    return s


# ------------------------------------------------------------------------------------------------------------------
# case export
# ------------------------------------------------------------------------------------------------------------------
def _save(d, name, a):
    # np.asarray(order="C") keeps 0-d arrays 0-d (np.ascontiguousarray would promote them to shape (1,))
    np.save(os.path.join(d, name + ".npy"), np.asarray(a, dtype=np.float64, order="C"))


def _save_stage(d, lnk, grad_lnk, psi0, vperp, v_rms):
    os.makedirs(d, exist_ok=True)
    _save(d, "lnk", lnk)
    for i in range(3):
        _save(d, "grad_lnk_%d" % (i + 1), grad_lnk[i])
    _save(d, "psi0_1", psi0[0]); _save(d, "psi0_2", psi0[1])
    _save(d, "vperp_2", vperp[0]); _save(d, "vperp_3", vperp[1])
    _save(d, "v_rms", np.float64(v_rms))


def cache_write_allowed(path):
    """True iff the SF-29 raw/cache/.gitignore ignores the file (`*.npz`, but `!*_N16_*`)."""
    b = os.path.basename(path)
    return b.endswith(".npz") and "_N16_" not in b


def case_dir_name(field, eps, N):
    return "%s_%s_%d" % (field, amp_name(eps), N)


def export_case(field, eps, N, outdir, bisect=BISECT_DEFAULT, stages=True, log=print):
    t0 = time.time()
    nphi = C.nphi_for(field, eps)
    cpath = C.cache_path(field, eps, N, nphi)
    wc = cache_write_allowed(cpath)
    case = C.load_case(field, eps, N, verbose=False, write_cache=wc)
    meta = case["meta"]
    t_case = time.time() - t0
    d = os.path.join(outdir, case_dir_name(field, eps, N))
    os.makedirs(d, exist_ok=True)
    v_rms = M.rms_vec(case["vD"])          # = candidate_i.Ctx.v_rms
    _save(d, "lnk", case["lnk"])
    for i in range(3):
        _save(d, "grad_lnk_%d" % (i + 1), case["grad_lnk"][i])
        _save(d, "vD_%d" % (i + 1), case["vD"][i])
    _save(d, "psi0_1", case["psi0"][0]); _save(d, "psi0_2", case["psi0"][1])
    _save(d, "vperp_2", case["vperp_in"][0]); _save(d, "vperp_3", case["vperp_in"][1])
    _save(d, "psi_or_1", case["psi_or"][0]); _save(d, "psi_or_2", case["psi_or"][1])
    _save(d, "v_rms", np.float64(v_rms))
    # full-precision reference of the oracle ceiling line (cand=oracle_fd4) for the C++ cross-validation
    mo = M.fd_metrics(case["psi_or"][0], case["psi_or"][1], case["vD"], case["psi_or"], order=4)
    with open(os.path.join(d, "ref_metrics.json"), "w") as fh:
        json.dump({"oracle_fd4": CI._jsonable(mo),
                   "definition": "metrics.fd_metrics(psi_or[0], psi_or[1], vD, psi_or, order=4) of the SF-29 "
                                 "artifact; floats are Python repr (exact float64 round trip)",
                   "case_line": M.case_line(field, eps, N, "oracle_fd4", mo)}, fh, indent=1, sort_keys=True)
    amps = reachable_amplitudes(eps, bisect) if stages else [float(eps)]
    stage_info = {}
    for a in amps:
        an = amp_name(a)
        sd = os.path.join(d, "stage", an)
        ts = time.time()
        if abs(a - eps) <= AMP_TOL:        # candidate_i.solve_case uses the main case at the target amplitude
            os.makedirs(sd, exist_ok=True)
            for f in STAGE_FILES:
                shutil.copyfile(os.path.join(d, f + ".npy"), os.path.join(sd, f + ".npy"))
            stage_info[an] = {"source": "main case (cases.load_case)", "nphi": int(nphi), "v_rms": v_rms}
        else:                              # candidate_i.solve_case: light_case(field, e, N, nphi=_nphi_light(field, e))
            na = CI._nphi_light(field, a)
            lc = CI.light_case(field, a, N, nphi=na)
            vr = M.rms_vec(lc["vD"])
            _save_stage(sd, lc["lnk"], lc["grad_lnk"], lc["psi0"], lc["vperp_in"], vr)
            stage_info[an] = {"source": "candidate_i.light_case", "nphi": int(na), "v_rms": vr}
        stage_info[an]["t"] = time.time() - ts
        log("  STAGE_EXPORT %s amp=%s nphi=%d t=%.1fs" % (case_dir_name(field, eps, N), an, stage_info[an]["nphi"],
                                                          stage_info[an]["t"]), flush=True)
    rt = np.asarray(meta.get("roundtrip", np.zeros(N + 1)), dtype=float)
    info = {
        "export_version": EXPORT_VERSION, "field": field, "eps": float(eps), "N": int(N), "nphi": int(meta["nphi"]),
        "nf": int(meta["nf"]), "Q0": float(meta["Q0"]), "inlet_vmin": float(meta["inlet_vmin"]),
        "L": [float(x) for x in meta["L"]], "a": [float(x) for x in meta["a"]], "pcg_res": float(meta["pcg_res"]),
        "pcg_its": int(meta["pcg_its"]), "max_d1phi": float(meta["max_d1phi"]),
        "cache_hit": bool(meta["cache_hit"]), "cache_file": _rel(meta["cache"]),
        "cache_written": bool((not meta["cache_hit"]) and wc), "cache_version": int(meta["cache_version"]),
        "roundtrip_per_plane": [float(x) for x in rt], "roundtrip_max": float(rt.max()) if rt.size else None,
        "v_rms": v_rms, "bisect": int(bisect), "amplitude_rule": AMPLITUDE_RULE,
        "stage_amplitudes": [amp_name(a) for a in amps], "stages": stage_info,
        "t_case": t_case, "t_export": time.time() - t0,
        "export_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": versions(), "source_artifact": _rel(SF29_DIR), "exporter": _rel(__file__),
        "layout": "float64 C-order .npy; full arrays (N+1, N, N) indexed [j, m2, m3] (m3 fastest), plane arrays "
                  "(N, N) indexed [m2, m3]; psi0 / psi_or are FULL labels (affine part included); v_rms 0-d",
    }
    with open(os.path.join(d, "case.json"), "w") as fh:
        json.dump(info, fh, indent=1, sort_keys=True)
    log("EXPORT %s nphi=%d cache_hit=%s cache_written=%s stages=%d t_case=%.1fs t_export=%.1fs -> %s"
        % (case_dir_name(field, eps, N), info["nphi"], info["cache_hit"], info["cache_written"], len(amps),
           t_case, info["t_export"], _rel(d)), flush=True)
    return d, case


def reload_check(d, N, stages=True):
    """np.load of every file of a case dir: shapes, dtype, C order, finite, psi_or_i[0] == psi0_i, stage dirs."""
    full = (N + 1, N, N); plane = (N, N)
    shapes = {"lnk": full, "grad_lnk_1": full, "grad_lnk_2": full, "grad_lnk_3": full, "vD_1": full,
              "vD_2": full, "vD_3": full, "psi0_1": plane, "psi0_2": plane, "vperp_2": plane, "vperp_3": plane,
              "psi_or_1": full, "psi_or_2": full, "v_rms": ()}
    a = {}
    for k, s in shapes.items():
        x = np.load(os.path.join(d, k + ".npy"))
        assert x.shape == s, (k, x.shape, s)
        assert x.dtype == np.float64 and x.flags["C_CONTIGUOUS"], k
        assert np.all(np.isfinite(x)), k
        a[k] = x
    for i in (1, 2):
        assert np.array_equal(a["psi_or_%d" % i][0], a["psi0_%d" % i]), "psi_or_%d[0] != psi0_%d" % (i, i)
    with open(os.path.join(d, "case.json")) as fh:
        info = json.load(fh)
    if stages:
        for s in info["stage_amplitudes"]:
            for k in STAGE_FILES:
                x = np.load(os.path.join(d, "stage", s, k + ".npy"))
                assert x.shape == shapes[k] and x.dtype == np.float64, (s, k, x.shape)
                if abs(float(s) - info["eps"]) <= AMP_TOL:
                    assert np.array_equal(x, a[k]), ("stage at eps differs from the main files", s, k)
    return a, info


# ------------------------------------------------------------------------------------------------------------------
# saved prototype solutions
# ------------------------------------------------------------------------------------------------------------------
def export_solution(npz, outdir, log=print):
    d0 = np.load(npz, allow_pickle=False)
    name = os.path.splitext(os.path.basename(npz))[0]
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    N = int(d0["N"])
    for k in ("u1", "u2"):
        u = d0[k]
        assert u.shape == (N + 1, N, N), (k, u.shape)
        _save(d, k, u)
    info = {"export_version": EXPORT_VERSION, "source_file": _rel(npz), "field": str(d0["field"]),
            "eps": float(d0["eps"]), "N": N, "variant": str(d0["variant"]), "order": int(d0["order"]),
            "cand": str(d0["cand"]), "status": str(d0["status"]), "its": int(d0["its"]), "r_F": float(d0["r_F"]),
            "r_out": float(d0["r_out"]), "t": float(d0["t"]), "path": str(d0["path"]),
            "hist": np.asarray(d0["hist"], dtype=float).tolist(),
            "metrics_json": json.loads(str(d0["metrics_json"])), "opts_json": json.loads(str(d0["opts_json"])),
            "saved_with": {"numpy": str(d0["numpy"]), "python": str(d0["python"])}, "versions": versions(),
            "layout": "u1.npy, u2.npy: periodic parts u_i (psi1 = x2 + u1, psi2 = x3 + u2) on planes 0..N, "
                      "(N+1, N, N) float64 C-order [j, m2, m3]; plane 0 is the inlet data u0"}
    with open(os.path.join(d, "solution.json"), "w") as fh:
        json.dump(info, fh, indent=1, sort_keys=True)
    log("SOLUTION %s status=%s its=%d r_F=%.3e r_out=%.3e path=%s -> %s"
        % (name, info["status"], info["its"], info["r_F"], info["r_out"], info["path"], _rel(d)), flush=True)
    return d


# ------------------------------------------------------------------------------------------------------------------
# SF-19 cross-check inputs (step 8)
# ------------------------------------------------------------------------------------------------------------------
def export_crosscheck(field, eps, N, outdir, log=print):
    if field.endswith("_ch") or field == "uniform":
        raise SystemExit("--crosscheck needs a triply periodic SF-29 field (SF-19 is periodic); got %s" % field)
    t0 = time.time()
    ref, inl = C.build_reference(field, eps, verbose=False)
    h = 1.0 / N
    c = (np.arange(N) + 0.5) * h
    d = os.path.join(outdir, "crosscheck_%s" % case_dir_name(field, eps, N))
    os.makedirs(d, exist_ok=True)
    # Y = ln k at the cell centres in the PROJECT cell layout idx = i + N*(j + N*k): array indexed [k, j, i]
    Zc, Yc, Xc = np.meshgrid(c, c, c, indexing="ij")
    _save(d, "Y_cells", ref.lk.lnk(Xc, Yc, Zc))
    # spectral point v1 at the inlet-face cell centres ((m2+1/2)h, (m3+1/2)h), indexed [m2, m3]
    Y2, Z2 = np.meshgrid(c, c, indexing="ij")
    _save(d, "v1_face_ref", ref.velocity(0.0, Y2.ravel(), Z2.ravel())[:, 0].reshape(N, N))
    # face-averaged v1 over [m2 h, (m2+1) h] x [m3 h, (m3+1) h] (GL NG x NG, as cases.face_fluxes f1[0])
    xg, wg = np.polynomial.legendre.leggauss(C.NG)
    xg = 0.5 * (xg + 1.0); wg = 0.5 * wg
    yy = np.arange(N)[:, None] * h + h * xg[None, :]
    Yq = np.broadcast_to(yy[:, :, None, None], (N, C.NG, N, C.NG))
    Zq = np.broadcast_to(yy[None, None, :, :], (N, C.NG, N, C.NG))
    v1q = ref.velocity(0.0, Yq.ravel(), Zq.ravel())[:, 0].reshape(N, C.NG, N, C.NG)
    v1avg = np.einsum("aibj,ij->ab", v1q, wg[:, None] * wg[None, :])
    _save(d, "v1_faceavg_ref", v1avg)
    # (v2, v3) at the inlet vertices (0, m2 h, m3 h), indexed [m2, m3]
    _, Yv, Zv = C.vertex_coords(N)
    vv = ref.velocity(0.0, Yv, Zv)
    _save(d, "vperp_vertex_ref_2", vv[:, 1].reshape(N, N))
    _save(d, "vperp_vertex_ref_3", vv[:, 2].reshape(N, N))
    info = {"export_version": EXPORT_VERSION, "field": field, "eps": float(eps), "N": int(N), "nphi": int(ref.nphi),
            "a": [float(x) for x in ref.a], "pcg_res": float(ref.pcg[0]), "pcg_its": int(ref.pcg[1]),
            "max_d1phi": float(ref.max_d1phi), "Q0": float(inl.Q0), "mean_v1_faceavg": float(v1avg.mean()),
            "layout": {"Y_cells": "(N, N, N) indexed [k, j, i] = ln k at the cell centre ((i+1/2)h, (j+1/2)h, "
                                  "(k+1/2)h); flat C-order index i + N*(j + N*k) = Grid3D::idx (x FASTEST)",
                       "v1_face_ref": "(N, N) [m2, m3]: point v1 at (0, (m2+1/2)h, (m3+1/2)h)",
                       "v1_faceavg_ref": "(N, N) [m2, m3]: Gauss-Legendre 3x3 average of v1 over the x1 = 0 face "
                                         "cell [m2 h, (m2+1) h] x [m3 h, (m3+1) h]",
                       "vperp_vertex_ref_2/3": "(N, N) [m2, m3]: (v2, v3) at (0, m2 h, m3 h) (slab layout, m3 "
                                               "fastest)"},
            "t": time.time() - t0, "versions": versions(), "source_artifact": _rel(SF29_DIR),
            "export_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(d, "crosscheck.json"), "w") as fh:
        json.dump(info, fh, indent=1, sort_keys=True)
    log("CROSSCHECK %s nphi=%d mean(v1_faceavg)=%.15f t=%.1fs -> %s"
        % (case_dir_name(field, eps, N), ref.nphi, info["mean_v1_faceavg"], info["t"], _rel(d)), flush=True)
    return d


# ------------------------------------------------------------------------------------------------------------------
# selftest
# ------------------------------------------------------------------------------------------------------------------
def selftest():
    t0 = time.time()
    nfail = [0]

    def chk(cond, name):
        print("[%s] %s" % ("PASS" if cond else "FAIL", name), flush=True)
        if not cond:
            nfail[0] += 1

    # 1. the amplitude rule vs every PATH line of the SF-29 corrective sweep (all candidates, i1o4 required)
    paths = parse_summary_paths()
    n_i1o4 = len([p for p in paths if p[2] == "i1o4"])
    chk(n_i1o4 > 0, "summary.md: %d PATH lines, %d of them i1o4" % (len(paths), n_i1o4))
    sets = {}
    for field, eps, cand, N, s in paths:
        if eps not in sets:
            sets[eps] = reachable_amplitudes(eps)
        toks = path_tokens(s)
        missing = sorted(set(a for a, _ in toks if not any(abs(a - b) <= AMP_TOL for b in sets[eps])))
        replay = "->".join(simulate_path(eps, [kind == "ok" for _, kind in toks if kind != "final"]))
        chk(not missing and replay == s,
            "PATH %s:%g N=%d %s: amplitudes in the reachable set%s; rule replay %s"
            % (field, eps, N, cand, "" if not missing else " (MISSING %s)" % missing,
               "identical" if replay == s else "DIFFERS (%s)" % replay))
    for eps in (0.25, 0.5, 1.0):
        s = sets.get(eps) or reachable_amplitudes(eps)
        sets[eps] = s
        ok = True
        try:
            [amp_name(a) for a in s]
        except ValueError:
            ok = False
        chk(ok and abs(s[-1] - eps) <= AMP_TOL, "eps=%g: %d reachable amplitudes, all exact in %%g, max = eps"
            % (eps, len(s)))
    must = (0.125, 0.375, 0.625, 0.75, 0.8125, 0.875, 0.9375)
    chk(all(a in sets[1.0] for a in must), "eps=1 set contains %s (the amplitudes of the SF-29 PATH lines)"
        % ", ".join("%g" % a for a in must))
    # 2. 16^3 gauss:0.25 export from the committed cache + reload
    tmp = tempfile.mkdtemp(prefix="sf33_export_selftest_")
    try:
        d, case = export_case("gauss", 0.25, 16, tmp)
        a = None
        try:
            a, info = reload_check(d, 16)
            chk(True, "reload: every file present, shapes/dtype/C order/finite, psi_or_i[0] == psi0_i, stage dirs "
                      "complete and stage/<eps> == main files")
        except AssertionError as e:
            chk(False, "reload check: %s" % (e,))
        if a is not None:
            chk(info["cache_hit"], "gauss:0.25:16 loaded from the committed SF-29 cache")
            chk(info["stage_amplitudes"] == [amp_name(x) for x in reachable_amplitudes(0.25)],
                "stage dirs = reachable set of eps=0.25 (%s)" % ", ".join(info["stage_amplitudes"]))
            lc = CI.light_case("gauss", 0.25, 16, nphi=CI._nphi_light("gauss", 0.25))
            dl = max([float(np.max(np.abs(lc["lnk"] - a["lnk"])))]
                     + [float(np.max(np.abs(lc["grad_lnk"][i] - a["grad_lnk_%d" % (i + 1)]))) for i in range(3)]
                     + [float(np.max(np.abs(lc["psi0"][i] - a["psi0_%d" % (i + 1)]))) for i in range(2)]
                     + [float(np.max(np.abs(lc["vperp_in"][i] - a["vperp_%d" % (i + 2)]))) for i in range(2)]
                     + [abs(M.rms_vec(lc["vD"]) - float(a["v_rms"]))])
            chk(dl <= 1e-13, "light_case(gauss, 0.25, 16) vs the main-case export at eps: max |diff| = %.3e" % dl)
            chk(float(a["v_rms"]) == M.rms_vec(case["vD"]), "v_rms.npy == metrics.rms_vec(vD) = %.17g"
                % float(a["v_rms"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST %s (%d failures, %.1fs)" % ("PASS" if nfail[0] == 0 else "FAIL", nfail[0], time.time() - t0),
          flush=True)
    return nfail[0] == 0


# ------------------------------------------------------------------------------------------------------------------
def main(argv):
    out = None; bisect = BISECT_DEFAULT; stages = True; mode = "export"; args = []; i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            out = argv[i + 1]; i += 2; continue
        if a == "--bisect":
            bisect = int(argv[i + 1]); i += 2; continue
        if a == "--no-stages":
            stages = False; i += 1; continue
        if a in ("--list-amplitudes", "--solutions", "--crosscheck", "--selftest"):
            mode = a[2:]; i += 1; continue
        if a in ("-h", "--help"):
            print(__doc__); return 0
        args.append(a); i += 1
    if mode == "selftest":
        return 0 if selftest() else 1
    if mode == "list-amplitudes":
        print("RULE " + AMPLITUDE_RULE)
        for e in args:
            s = reachable_amplitudes(float(e), bisect)
            print("AMPLITUDES eps=%g bisect=%d n=%d: %s" % (float(e), bisect, len(s), " ".join(amp_name(x)
                                                                                              for x in s)))
        return 0
    if out is None or not args:
        print(__doc__); return 2
    os.makedirs(out, exist_ok=True)
    for a in args:
        if mode == "solutions":
            export_solution(a, out)
            continue
        field, eps, N = C.parse_spec(a)
        if N is None:
            raise SystemExit("case spec must be field:eps:N, got %r" % a)
        if mode == "crosscheck":
            export_crosscheck(field, eps, N, out)
        else:
            d, _ = export_case(field, eps, N, out, bisect=bisect, stages=stages)
            reload_check(d, N, stages)
            print("RELOAD %s ok" % _rel(d), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
