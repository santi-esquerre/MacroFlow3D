#!/usr/bin/env python3
"""SF-37 digest of V3a (confinement against the verified SF-35 gridded labels) and V3b (Pollock comparator), with D5 and D6.

  digest_v3.py ROOT OUTDIR [--design full|discovered]

Reading rules RR6 / RR7 of the SF-37 activation row, implemented literally (no threshold is tuned here):

V3a (RR6, O1 anchor, eps = 1, seeds 3101-3103)
  * criterion C of SF-35 per seed, by `orders_digest.py --eps 1 labels/raw/eps10_s{S}/N{64,128,256}.json --expected-ns 64,128,256`
    invoked IN PLACE as a subprocess (its "Amplitude verdict (C)" is parsed); a construction exit 17 or 19 -> not_evaluable.
  * where C is `accept` the labels of the seed are a verified reference.  On grid plane j = x1 N (x1 in {0.25, 0.5, 0.75, 1}) the
    fluctuations U1 = psi1 - x2_m, U2 = psi2 - x3_m ((N, N), no duplicate) are interpolated at each contributing particle's landed
    (x2 mod 1, x3 mod 1) by scipy.ndimage.map_coordinates(U, [(x2 mod 1) N, (x3 mod 1) N], order = 3, mode = 'grid-wrap',
    prefilter = True); psi_grid = (x2_u + U1, x3_u + U2) with the UNWRAPPED landing; error = Euclidean norm of psi_grid - psi_p
    (psi_p = the D-1 label at the seed, labels.npy of the v3a run).  Classified quantity: the UNWEIGHTED RMS (and max) over the
    contributing particles with finite psi_p; `concordant` iff the RMS decreases on 64 -> 128 and 128 -> 256 at that plane, else
    `not concordant`.  A seed whose C is not `accept` -> `reference not verified` (numbers still recorded); inlet backflow
    (labels NaN / exit 17) -> `reference not available`.  Control x1 = 0 (seed plane j = 0) is recorded.
  * D5: interpolation proxy per plane (RMS over the 2N lattice vertices of |U_N interpolated at the 2N vertices - U_2N|, N in
    {64, 128}); V3a RMS as a multiple of it; fraction of particles above 10 x proxy; constructor exit code and max round trip;
    stage identity between the trace run and the v3a forward run.

V3b (RR7, independent representation, eps in {1, 1.5, 2}, N in {64, 128, 256, 512})
  * D_P(N) = sigma_T^2(Pollock, N) - sigma_T^2(O6, N) per plane from the v3b run (pollock_* files vs the O6 files of the SAME run
    directory); SE_D by the paired block bootstrap of sf37lib (Pollock weights for Pollock, O6 weights for O6, each over its own
    contributors).  `concordant` iff |D_P| decreases on every refinement 64 -> 128 -> 256 -> 512 AND |D_P(512)| <=
    max(0.05 sigma_T^2(O6, 512), 2 SE_D(512)); else `discordant`; missing data -> not_evaluable.
  * D6: weight part sigma^2(P-traj, P-w) - sigma^2(P-traj, O6-w) and trajectory part sigma^2(P-traj, O6-w) - sigma^2(O6) (each
    with its paired SE_D); signed D_P per N; noise-limited flag per refinement (|D_P(N) - D_P(2N)| < 2 x the larger SE_D); slope of
    D_P against x1 and its scaling with N; D_P on the common contributors only; Pollock non-active count and weight per pass;
    the time-snapshot analogues (var x1, x2, x3 at each t).  Consistency (recorded): the O6 files of each v3b run are bitwise
    identical to those of the v2 run with the same (eps, seed, N); Pollock moments reproduce pollock_moments_*.csv to <= 1e-12.

Outputs (OUTDIR): v3_summary.md, v3a_*.csv, v3b_*.csv, criterionC_s{S}.md / _planes.csv, input_manifest_v3.csv.  python3 + numpy +
scipy (scipy.ndimage only).  Missing data never pass.
"""
import argparse
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf37lib as L  # noqa: E402
import o3 as O3  # noqa: E402  (label export layout and exit-code reader only)

NAN = float("nan")
V3A_EPS = (10,)
V3A_NS = (64, 128, 256)
V3A_PLANES = (0.25, 0.5, 0.75, 1.0)
V3B_NS = (64, 128, 256, 512)
V3B_REL = 0.05
V3B_NSE = 2.0
PROXY_FACTOR = 10.0
ORDERS_REL = os.path.join("docs", "experiments", "artifacts", "2026-10-07-sf35-label-transport", "scripts", "orders_digest.py")

POLLOCK_ALIAS = {"weights.npy": "pollock_weights.npy", "status_plane.npy": "pollock_status_plane.npy",
                 "status_time.npy": "pollock_status_time.npy", "planes_x23.npy": "pollock_planes_x23.npy",
                 "planes_tau.npy": "pollock_planes_t.npy", "times_x.npy": "pollock_times_x.npy"}
POLLOCK_FILES = tuple(POLLOCK_ALIAS.values()) + ("pollock_moments_planes.csv", "pollock_moments_time.csv")
O6_CONSISTENCY_FILES = ("weights.npy", "status_plane.npy", "status_time.npy", "planes_x23.npy", "planes_tau.npy", "planes_I.npy",
                        "times_x.npy", "times_I.npy", "moments_planes.csv", "moments_time.csv")
_RE_V3 = re.compile(r"^v3([ab])_e(\d+)_s(\d+)_N(\d+)$")


def v3a_name(e, s, n):
    return "v3a_e%d_s%d_N%d" % (e, s, n)


def v3b_name(e, s, n):
    return "v3b_e%d_s%d_N%d" % (e, s, n)


def find_orders_digest():
    d = HERE
    for _ in range(8):
        p = os.path.join(d, ORDERS_REL)
        if os.path.isfile(p):
            return p
        d = os.path.dirname(d)
    return None


# ------------------------------------------------------------------------------------------------------------
# V3a primitives
# ------------------------------------------------------------------------------------------------------------
def fluctuations(psi1_plane, psi2_plane, N):
    """U1 = psi1 - x2_m, U2 = psi2 - x3_m on a grid plane ((N, N) indexed [m2, m3], vertex x = m / N, no duplicate)."""
    m = np.arange(N, dtype=np.float64) / N
    return np.asarray(psi1_plane, dtype=np.float64) - m[:, None], np.asarray(psi2_plane, dtype=np.float64) - m[None, :]


def interp_U(U, N, x2, x3):
    """Periodic bicubic B-spline (scipy.ndimage, mode grid-wrap, prefilter) of U at the (x2 mod 1, x3 mod 1) of the points."""
    from scipy import ndimage
    c = [np.mod(np.asarray(x2, dtype=np.float64), 1.0) * N, np.mod(np.asarray(x3, dtype=np.float64), 1.0) * N]
    return ndimage.map_coordinates(U, c, order=3, mode="grid-wrap", prefilter=True)


def label_error(psi1_plane, psi2_plane, N, x2u, x3u, psi_p):
    """RR6 error at the unwrapped landings: psi_grid = (x2_u + U1, x3_u + U2); -> (e1, e2, |e|) with e = psi_grid - psi_p."""
    U1, U2 = fluctuations(psi1_plane, psi2_plane, N)
    x2u = np.asarray(x2u, dtype=np.float64)
    x3u = np.asarray(x3u, dtype=np.float64)
    g1 = x2u + interp_U(U1, N, x2u, x3u)
    g2 = x3u + interp_U(U2, N, x2u, x3u)
    e1 = g1 - psi_p[:, 0]
    e2 = g2 - psi_p[:, 1]
    return e1, e2, np.sqrt(e1 * e1 + e2 * e2)


def interp_proxy(psiN, psi2N, N, j, j2):
    """D5 proxy: RMS over the 2N vertices of the plane of |U_N interpolated at the 2N vertices - U_2N| (both labels,
    Euclidean norm per vertex).  psiN/psi2N = (psi1, psi2) volumes; j, j2 = the plane indices on the two lattices."""
    from scipy import ndimage
    U1, U2 = fluctuations(psiN[0][j], psiN[1][j], N)
    V1, V2 = fluctuations(psi2N[0][j2], psi2N[1][j2], 2 * N)
    c = np.arange(2 * N, dtype=np.float64) * (N / (2.0 * N))
    gi, gj = np.meshgrid(c, c, indexing="ij")
    coords = [gi.ravel(), gj.ravel()]
    i1 = ndimage.map_coordinates(U1, coords, order=3, mode="grid-wrap", prefilter=True).reshape(2 * N, 2 * N)
    i2 = ndimage.map_coordinates(U2, coords, order=3, mode="grid-wrap", prefilter=True).reshape(2 * N, 2 * N)
    return float(np.sqrt(np.mean((i1 - V1) ** 2 + (i2 - V2) ** 2)))


def plane_error_stats(e1, e2, err, w, proxy=NAN):
    """Unweighted RMS / max (classified), flux-weighted RMS (recorded), per-label RMS, D5 ratios."""
    n = int(err.size)
    out = {"n": n, "rms": NAN, "max": NAN, "rms_w": NAN, "rms_l1": NAN, "rms_l2": NAN, "ratio_proxy": NAN, "frac_gt_proxy": NAN}
    if n == 0:
        return out
    out["rms"] = float(np.sqrt(np.mean(err ** 2)))
    out["max"] = float(np.max(err))
    out["rms_l1"] = float(np.sqrt(np.mean(e1 ** 2)))
    out["rms_l2"] = float(np.sqrt(np.mean(e2 ** 2)))
    W = L.seq_sum(w)
    if W > 0.0:
        out["rms_w"] = math.sqrt(L.seq_sum(w * err * err) / W)
    if math.isfinite(proxy) and proxy > 0.0:
        out["ratio_proxy"] = out["rms"] / proxy
        out["frac_gt_proxy"] = float(np.mean(err > PROXY_FACTOR * proxy))
    return out


def classify_v3a(rms_by_n, ns):
    """concordant iff the unweighted RMS decreases strictly on every refinement of ns (64 -> 128 -> 256); else not concordant;
    a missing / non-finite value -> not_evaluable."""
    seq = [rms_by_n.get(n, NAN) for n in ns]
    if len(ns) < 2 or any(not math.isfinite(v) for v in seq):
        return "not_evaluable"
    return "concordant" if all(seq[i + 1] < seq[i] for i in range(len(seq) - 1)) else "not concordant"


def reference_label(avail, c_verdict):
    """RR6 reading: not available (inlet backflow / no labels) > not verified (C is not accept) > verified."""
    if not avail:
        return "reference not available"
    if c_verdict != "accept":
        return "reference not verified"
    return "reference verified"


def final_v3a(reference, concord):
    if reference == "reference verified":
        return concord
    return reference


# ------------------------------------------------------------------------------------------------------------
# criterion C and stage identity
# ------------------------------------------------------------------------------------------------------------
_RE_C = re.compile(r"\*\*Amplitude verdict \(C\): (\S+?)\*\*")


def parse_c_verdict(text):
    m = _RE_C.search(text)
    return m.group(1) if m else None


def run_criterion_c(script, eps_label, files, ns, md_path, csv_path):
    """Invokes orders_digest.py in place; -> (verdict, rc, note).  Any failure to run / parse -> not_evaluable."""
    if script is None:
        return "not_evaluable", None, "orders_digest.py not found"
    if not files:
        return "not_evaluable", None, "no driver summary JSON"
    cmd = [sys.executable, script, "--eps", eps_label] + list(files) + ["--expected-ns", ",".join(str(n) for n in ns), "--out", md_path,
                                                                        "--planes-csv", csv_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    v = parse_c_verdict(r.stdout) if r.returncode == 0 else None
    if v is None:
        return "not_evaluable", r.returncode, "orders_digest.py rc=%s, no 'Amplitude verdict (C)' line" % r.returncode
    return v, r.returncode, ""


def parse_stage_identity(text):
    """-> dict(G=tuple of strings or None, inlet_vmin=str or None, v_rms=str or None) at print precision."""
    out = {"G": None, "inlet_vmin": None, "v_rms": None}
    if text is None:
        return out
    m = re.search(r"\bG=\(([^)]*)\)", text)
    if m:
        out["G"] = tuple(t.strip() for t in m.group(1).split(","))
    m = re.search(r"\binlet_vmin=(\S+)", text) or re.search(r"inlet: vmin=(\S+)", text)
    if m:
        out["inlet_vmin"] = m.group(1)
    m = re.search(r"\bv_rms=([-+0-9.eE]+)", text)
    if m:
        out["v_rms"] = m.group(1)
    return out


def compare_identity(a, b, key):
    va, vb = a.get(key), b.get(key)
    if va is None and vb is None:
        return "absent in both logs"
    if va is None:
        return "absent in the trace log"
    if vb is None:
        return "absent in the forward log"
    return "identical" if va == vb else "DIFFERENT (%s vs %s)" % (va, vb)


# ------------------------------------------------------------------------------------------------------------
# V3b
# ------------------------------------------------------------------------------------------------------------
class PollockRun(L.RunData):
    """The Pollock records of a v3b run directory behind the RunData interface (names mapped onto the O6 names).
    weights = 'pollock' (pollock_weights.npy) or 'o6' (weights.npy: Pollock trajectories with the O6 flux weights)."""

    def __init__(self, path, manifest=None, name=None, weights="pollock"):
        super().__init__(path, manifest, name)
        if weights not in ("pollock", "o6"):
            raise ValueError("weights must be 'pollock' or 'o6'")
        self.weights_kind = weights

    def fpath(self, fname):
        if fname in POLLOCK_ALIAS and (fname != "weights.npy" or self.weights_kind == "pollock"):
            return os.path.join(self.path, POLLOCK_ALIAS[fname])
        return os.path.join(self.path, fname)

    @property
    def complete(self):
        return all(os.path.isfile(os.path.join(self.path, f)) for f in POLLOCK_FILES + ("track.json", "weights.npy"))


def v3b_classify(ns, D, ref_final, se_final):
    """RR7.  D = {N: D_P}; ref_final = sigma_T^2(O6, N_max); se_final = SE_D(N_max).
    -> (class, reason, |D| list, threshold)."""
    vals = [D.get(n, NAN) for n in ns]
    if len(ns) < 2 or not all(math.isfinite(v) for v in vals) or not (math.isfinite(ref_final) and math.isfinite(se_final)):
        return "not_evaluable", "missing or non-finite D_P / sigma_T^2(O6) / SE_D", [abs(v) for v in vals], NAN
    a = [abs(v) for v in vals]
    dec = all(a[i + 1] < a[i] for i in range(len(a) - 1))
    thr = max(V3B_REL * ref_final, V3B_NSE * se_final)
    ok = a[-1] <= thr
    if dec and ok:
        return "concordant", "|D_P| decreases on every refinement and |D_P(%d)| = %.4g <= %.4g" % (ns[-1], a[-1], thr), a, thr
    why = []
    if not dec:
        why.append("|D_P| does not decrease on every refinement")
    if not ok:
        why.append("|D_P(%d)| = %.4g > max(0.05 sigma_T^2, 2 SE_D) = %.4g" % (ns[-1], a[-1], thr))
    return "discordant", "; ".join(why), a, thr


def noise_limited(d_a, se_a, d_b, se_b):
    """|D(N) - D(2N)| < 2 x the larger SE_D."""
    if not all(math.isfinite(x) for x in (d_a, se_a, d_b, se_b)):
        return None
    return abs(d_a - d_b) < V3B_NSE * max(se_a, se_b)


def lsq_slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return NAN
    A = np.stack([np.ones(int(ok.sum())), x[ok]], axis=1)
    return float(np.linalg.lstsq(A, y[ok], rcond=None)[0][1])


def files_identical(a, b):
    """-> True / False / None (a file missing)."""
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return None
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    return L.sha256_file(a) == L.sha256_file(b)


def pollock_moments_check(prun, tol=1e-12):
    """Library moments of the Pollock records against pollock_moments_planes.csv / pollock_moments_time.csv.
    -> (ok, worst relative difference over the compared columns, non_ok matches or None, per-column dict)."""
    prun._note("pollock_moments_planes.csv")
    prun._note("pollock_moments_time.csv")
    core = L._plane_core(prun)
    pc = L.read_moments_planes_csv(prun.fpath("pollock_moments_planes.csv"))
    cols = {}
    mine = {"x1": prun.planes, "contributing": core["n"], "mean_x2": core["mean_x2"], "var_x2": core["var_x2"],
            "mean_x3": core["mean_x3"], "var_x3": core["var_x3"], "mean_t": core["mean_tau"], "var_t": core["var_tau"]}
    for k, v in mine.items():
        cols["planes." + k] = L.max_rel_diff(v, pc[k]) if k in pc else float("inf")
    tm = L.time_moments(prun)
    tc = L.read_moments_time_csv(prun.fpath("pollock_moments_time.csv"))
    for mn, th in (("t", "t"), ("mean_x1", "mean_x"), ("mean_x2", "mean_y"), ("mean_x3", "mean_z"), ("var_x1", "var_x"),
                   ("var_x2", "var_y"), ("var_x3", "var_z"), ("active", "active")):
        cols["time." + mn] = L.max_rel_diff(tm[mn], tc[th]) if th in tc else float("inf")
    worst = max(cols.values()) if cols else 0.0
    non_ok = None
    if "non_ok" in pc:
        non_ok = bool(np.array_equal(np.asarray(pc["non_ok"]), np.full(prun.P, int((prun.mem("status_plane.npy") != 0).sum()))))
    return worst <= tol, worst, non_ok, cols


def status_summary(prun, which):
    """Pollock non-active particles of a pass: -> dict(n_non_active, w_non_active, by_code {code: n})."""
    st = np.asarray(prun.mem("status_plane.npy" if which == "plane" else "status_time.npy"))
    w = prun.w
    bad = st != 0
    codes = {}
    for c in np.unique(st[bad]):
        codes[int(c)] = int((st == c).sum())
    W = L.seq_sum(w)
    return {"n_non_active": int(bad.sum()), "w_non_active": (float(w[bad].sum()) / W) if W > 0 else NAN, "by_code": codes}


# ------------------------------------------------------------------------------------------------------------
# design (full / discovered)
# ------------------------------------------------------------------------------------------------------------
def discover_v3(root):
    """-> ({(e, s): [N]} for v3a, {(e, s): [N]} for v3b) from the run directory names."""
    out = {"a": {}, "b": {}}
    for part, key in (("v3a", "a"), ("v3b", "b")):
        d = os.path.join(root, part)
        if os.path.isdir(d):
            for nm in sorted(os.listdir(d)):
                m = _RE_V3.match(nm)
                if m and os.path.isdir(os.path.join(d, nm)) and m.group(1) == key:
                    out[key].setdefault((int(m.group(2)), int(m.group(3))), []).append(int(m.group(4)))
    for k in out:
        for kk in out[k]:
            out[k][kk] = sorted(out[k][kk])
    return out["a"], out["b"]


def v3_design(kind, root):
    if kind == "full":
        a = {(e, s): list(V3A_NS) for e in V3A_EPS for s in L.SEEDS}
        b = {(e, s): list(V3B_NS) for e in L.EPS_CODES for s in L.SEEDS}
        return a, b
    return discover_v3(root)


# ------------------------------------------------------------------------------------------------------------
# V3a driver
# ------------------------------------------------------------------------------------------------------------
def read_json_file(path, manifest):
    import json
    if not os.path.isfile(path):
        return None
    manifest.add(path)
    try:
        with open(path) as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def read_text(path, manifest):
    if not os.path.isfile(path):
        return None
    manifest.add(path)
    with open(path, errors="replace") as f:
        return f.read()


def do_v3a(root, outdir, design_a, manifest, exits, W, script):
    cells, ctrl, verd, ident, d5ctor, csum = [], [], [], [], [], []
    ns_all = sorted({n for v in design_a.values() for n in v})
    for (e, s), ns in sorted(design_a.items()):
        ns = sorted(ns)
        # ---- criterion C
        raw_files = []
        for n in ns:
            p = os.path.join(root, "labels", "raw", "eps%d_s%d" % (e, s), "N%d.json" % n)
            if os.path.isfile(p):
                raw_files.append(p)
                manifest.add(p)
        md_c = os.path.join(outdir, "criterionC_s%d.md" % s)
        csv_c = os.path.join(outdir, "criterionC_s%d_planes.csv" % s)
        script_v, rc, note = run_criterion_c(script, "%g" % L.eps_of(e), raw_files, ns, md_c, csv_c)
        c_v, c_note = script_v, note
        bad_exit = [(n, exits.get("eps%d_s%d/N%d" % (e, s, n))) for n in ns if exits.get("eps%d_s%d/N%d" % (e, s, n)) in O3.BAD_EXITS]
        if bad_exit:
            c_v = "not_evaluable"
            c_note = "construction exit %s (RR6: exit 17 or 19 -> not evaluable); script verdict %s" % (
                ", ".join("N=%d rc=%d" % x for x in bad_exit), script_v)
        csum.append([e, s, c_v, script_v, c_note])
        # ---- availability (inlet backflow / no labels)
        avail = True
        avail_why = []
        for n in ns:
            rc_n = exits.get("eps%d_s%d/N%d" % (e, s, n))
            js = read_json_file(os.path.join(root, "labels", "raw", "eps%d_s%d" % (e, s), "N%d.json" % n), manifest)
            st = js.get("status") if isinstance(js, dict) else None
            if rc_n in (17, 18) or st in ("inlet_backflow", "darcy_failed"):
                avail = False
                avail_why.append("N=%d exit/status %s" % (n, rc_n if rc_n in (17, 18) else st))
            g = js.get("growth", {}) if isinstance(js, dict) else {}
            d5ctor.append([e, s, n, js.get("exit_code") if isinstance(js, dict) else None, rc_n, st,
                           g.get("max_roundtrip", NAN) if isinstance(g, dict) else NAN])
        # ---- numbers
        labels = {}
        for n in ns:
            labels[n], _ = O3.load_labels(root, e, s, n, manifest)
        rms_by = {}  # (x1) -> {N: rms}
        runs_ok = {}
        for n in ns:
            rdir = os.path.join(root, "v3a", v3a_name(e, s, n))
            r = L.RunData(rdir, manifest) if os.path.isdir(rdir) else None
            need = ("planes_x23.npy", "weights.npy", "labels.npy", "track.json")
            runs_ok[n] = r if (r is not None and all(r.exists(f) for f in need)) else None
            # identity trace vs forward
            tr = parse_stage_identity(read_text(os.path.join(root, "labels", "logs", "eps%d_s%d" % (e, s), "N%d.log" % n), manifest))
            fw = parse_stage_identity(read_text(os.path.join(root, "v3a", v3a_name(e, s, n) + ".log"), manifest))
            ident.append([e, s, n, compare_identity(tr, fw, "G"), compare_identity(tr, fw, "inlet_vmin"),
                          compare_identity(tr, fw, "v_rms"), str(tr["G"]), str(fw["G"]), tr["inlet_vmin"], fw["inlet_vmin"],
                          tr["v_rms"], fw["v_rms"]])
        for n in ns:
            r = runs_ok[n]
            lab = labels[n]
            if r is None or lab is None:
                for x1 in V3A_PLANES:
                    cells.append([e, s, n, x1, "not_evaluable", 0] + [NAN] * 8)
                    rms_by.setdefault(x1, {})[n] = NAN
                continue
            labp = r.read_array("labels.npy").astype(np.float64)
            if labp.shape != (r.Np, 2):
                for x1 in V3A_PLANES:
                    cells.append([e, s, n, x1, "not_evaluable", 0] + [NAN] * 8)
                    rms_by.setdefault(x1, {})[n] = NAN
                continue
            if not np.isfinite(labp).any():
                avail = False
                avail_why.append("N=%d: labels.npy all NaN (inlet backflow)" % n)
            fin = np.isfinite(labp).all(axis=1)
            X = r.mem("planes_x23.npy")
            pm = r.plane_mask()
            w = r.w
            nxt = labels.get(2 * n)
            for x1 in V3A_PLANES:
                idx = [i for i, v in enumerate(r.planes) if abs(v - x1) < 1e-12]
                j = x1 * n
                if not idx or abs(j - round(j)) > 1e-9:
                    cells.append([e, s, n, x1, "not_evaluable", 0] + [NAN] * 8)
                    rms_by.setdefault(x1, {})[n] = NAN
                    continue
                m, j = idx[0], int(round(j))
                sel = pm[:, m] & fin
                proxy = NAN
                if nxt is not None:
                    proxy = interp_proxy(lab, nxt, n, j, int(round(x1 * 2 * n)))
                e1, e2, err = label_error(lab[0][j], lab[1][j], n, X[sel, m, 0], X[sel, m, 1], labp[sel])
                st = plane_error_stats(e1, e2, err, w[sel], proxy)
                cells.append([e, s, n, x1, "ok" if st["n"] else "not_evaluable", st["n"], st["rms"], st["max"], st["rms_w"],
                              st["rms_l1"], st["rms_l2"], proxy, st["ratio_proxy"], st["frac_gt_proxy"]])
                rms_by.setdefault(x1, {})[n] = st["rms"] if st["n"] else NAN
            # control x1 = 0 (seed plane)
            if r.exists("seeds.npy"):
                sd = r.read_array("seeds.npy").astype(np.float64)
                f0 = fin & np.isfinite(sd[:, 1]) & np.isfinite(sd[:, 2])
                e1, e2, err = label_error(lab[0][0], lab[1][0], n, sd[f0, 1], sd[f0, 2], labp[f0])
                st = plane_error_stats(e1, e2, err, w[f0])
                ctrl.append([e, s, n, st["n"], st["rms"], st["max"], st["rms_w"], st["rms_l1"], st["rms_l2"]])
            else:
                ctrl.append([e, s, n, 0, NAN, NAN, NAN, NAN, NAN])
            r.release()
        ref = reference_label(avail, c_v)
        for x1 in V3A_PLANES:
            rm = rms_by.get(x1, {})
            conc = classify_v3a(rm, ns)
            verd.append([e, s, x1, c_v, ref, conc, final_v3a(ref, conc)] + [rm.get(n, NAN) for n in ns_all] +
                        ["; ".join(avail_why)])
    return cells, ctrl, verd, ident, d5ctor, csum


# ------------------------------------------------------------------------------------------------------------
# V3b driver
# ------------------------------------------------------------------------------------------------------------
def do_v3b(root, design_b, camp, manifest, counts, W):
    cells, verd, d6, slopes, pstat, snaps, cons = [], [], [], [], [], [], []
    ns_all = sorted({n for v in design_b.values() for n in v})
    for (e, s), ns in sorted(design_b.items()):
        ns = sorted(ns)
        data = {}
        planes = None
        for n in ns:
            name = v3b_name(e, s, n)
            o6 = camp.run("v3b", name)
            pdir = os.path.join(camp.part_dir("v3b"), name)
            prun = PollockRun(pdir, manifest, name + ":pollock") if os.path.isdir(pdir) else None
            if o6 is None or prun is None or not prun.complete:
                cons.append([e, s, n, "missing run or files", NAN, "", "", "n/a"])
                data[n] = None
                continue
            phyb = PollockRun(pdir, manifest, name + ":pollock-traj+o6w", "o6")
            planes = o6.planes if planes is None else planes
            try:
                tot = L.paired_stats(prun, o6, counts)
                wpart = L.paired_stats(prun, phyb, counts)  # sigma^2(P-traj, P-w) - sigma^2(P-traj, O6-w)
                tpart = L.paired_stats(phyb, o6, counts)    # sigma^2(P-traj, O6-w) - sigma^2(O6)
                com_p = prun.plane_mask() & o6.plane_mask()
                com_t = prun.time_mask() & o6.time_mask()
                com = L.paired_stats(prun, o6, counts, 4, com_p, com_t)
            except ValueError as ex:
                cons.append([e, s, n, "paired comparison failed: %s" % ex, NAN, "", "", "n/a"])
                data[n] = None
                continue
            data[n] = {"tot": tot, "w": wpart, "t": tpart, "com": com, "o6": o6, "p": prun}
            # consistency
            v2dir = os.path.join(camp.part_dir("v2"), L.v2_name(e, s, n))
            ident = []
            for fn in O6_CONSISTENCY_FILES:
                ident.append((fn, files_identical(os.path.join(o6.path, fn), os.path.join(v2dir, fn))))
            for fn, _ in ident:
                for pth in (os.path.join(o6.path, fn), os.path.join(v2dir, fn)):
                    if os.path.isfile(pth):
                        manifest.add(pth)
            if not os.path.isdir(v2dir):
                v2s = "v2 run absent"
            elif any(v is None for _, v in ident):
                v2s = "files missing: " + ",".join(fn for fn, v in ident if v is None)
            elif all(v for _, v in ident):
                v2s = "bitwise identical"
            else:
                v2s = "DIFFERENT: " + ",".join(fn for fn, v in ident if v is False)
            ok, worst, non_ok, _cols = pollock_moments_check(prun)
            cons.append([e, s, n, v2s, worst, "yes" if ok else "NO", "" if non_ok is None else ("yes" if non_ok else "no"),
                         "ok" if (ok and v2s == "bitwise identical") else "see columns"])
            for which in ("plane", "time"):
                ss = status_summary(prun, which)
                pstat.append([e, s, n, which, ss["n_non_active"], ss["w_non_active"], str(ss["by_code"])])
        # ---- cells per plane
        P = len(planes) if planes is not None else len(L.PLANES)
        planes_v = list(planes) if planes is not None else list(L.PLANES)
        Dn, SEn, REF = {}, {}, {}
        for n in ns:
            d = data.get(n)
            for m in range(P):
                if d is None:
                    cells.append([e, s, n, planes_v[m], "not_evaluable"] + [NAN] * 14)
                    continue
                t, w_, tp, c = d["tot"]["sigma_T2"], d["w"]["sigma_T2"], d["t"]["sigma_T2"], d["com"]["sigma_T2"]
                Dn.setdefault(m, {})[n] = float(t["D"][m])
                SEn.setdefault(m, {})[n] = float(t["se"][m])
                REF.setdefault(m, {})[n] = float(t["b"][m])
                cells.append([e, s, n, planes_v[m], "ok", float(t["b"][m]), float(t["a"][m]), float(t["D"][m]), float(t["se"][m]),
                              L.zscore(float(t["D"][m]), float(t["se"][m])), float(w_["D"][m]), float(w_["se"][m]),
                              float(tp["D"][m]), float(tp["se"][m]), float(c["D"][m]), float(c["se"][m]),
                              float(d["tot"]["var_tau"]["D"][m]), float(d["tot"]["var_tau"]["se"][m]),
                              float(w_["D"][m] + tp["D"][m] - t["D"][m])])
        for m in range(P):
            cls, why, a, thr = v3b_classify(ns, Dn.get(m, {}), REF.get(m, {}).get(ns[-1], NAN), SEn.get(m, {}).get(ns[-1], NAN))
            verd.append([e, s, planes_v[m], cls, why, thr] + [Dn.get(m, {}).get(n, NAN) for n in ns_all])
            for i in range(len(ns) - 1):
                a_, b_ = ns[i], ns[i + 1]
                nl = noise_limited(Dn.get(m, {}).get(a_, NAN), SEn.get(m, {}).get(a_, NAN), Dn.get(m, {}).get(b_, NAN),
                                   SEn.get(m, {}).get(b_, NAN))
                d6.append([e, s, planes_v[m], a_, b_, "" if nl is None else ("noise-limited" if nl else "resolved")])
        # ---- slopes of D_P against x1
        sl = {}
        for n in ns:
            dd = [Dn.get(m, {}).get(n, NAN) for m in range(P)]
            sl[n] = lsq_slope(planes_v, dd)
        for i, n in enumerate(ns):
            nxt = ns[i + 1] if i + 1 < len(ns) else None
            ordr = NAN
            if nxt is not None and math.isfinite(sl[n]) and math.isfinite(sl[nxt]) and sl[n] * sl[nxt] > 0:
                ordr = L.order_pair(abs(sl[n]), abs(sl[nxt]))
            slopes.append([e, s, n, sl[n], nxt if nxt is not None else "", sl[nxt] if nxt is not None else NAN, ordr])
        # ---- time snapshots
        for n in ns:
            d = data.get(n)
            if d is None:
                continue
            tt = d["p"].times
            for k in range(len(tt)):
                row = [e, s, n, float(tt[k])]
                for q in ("var_x1", "var_x2", "var_x3"):
                    x = d["tot"][q]
                    row += [float(x["b"][k]), float(x["a"][k]), float(x["D"][k]), float(x["se"][k])]
                snaps.append(row)
        for n in ns:
            if data.get(n) is not None:
                data[n]["o6"].release()
                data[n]["p"].release()
    return cells, verd, d6, slopes, pstat, snaps, cons


# ------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="SF-37 V3a / V3b digest")
    ap.add_argument("root")
    ap.add_argument("outdir")
    ap.add_argument("--design", choices=("full", "discovered"), default="full")
    args = ap.parse_args(argv)
    argv_text = "digest_v3.py %s %s --design %s" % (args.root, args.outdir, args.design)
    manifest = L.Manifest(args.root)
    camp = L.Campaign(args.root, manifest)
    idx16 = L.load_bootstrap(manifest=manifest)
    counts = L.counts_from_idx(idx16, L.BOOT_NBLOCKS)
    exits = O3.read_exit_codes(args.root, manifest)
    design_a, design_b = v3_design(args.design, args.root)
    W = L.OutWriter(args.outdir, manifest, "digest_v3", argv_text)
    script = find_orders_digest()

    a_cells, a_ctrl, a_verd, a_ident, a_ctor, a_c = do_v3a(args.root, args.outdir, design_a, manifest, exits, W, script)
    b_cells, b_verd, b_d6, b_slopes, b_pstat, b_snaps, b_cons = do_v3b(args.root, design_b, camp, manifest, counts, W)

    W.csv("v3a_cells.csv", ["eps_code", "seed", "N", "x1", "status", "n_classified", "rms_unweighted", "max", "rms_fluxweighted",
                            "rms_label1", "rms_label2", "interp_proxy_D5", "rms_over_proxy", "frac_err_gt_10x_proxy"],
          a_cells)
    W.csv("v3a_control_x0.csv", ["eps_code", "seed", "N", "n", "rms_unweighted", "max", "rms_fluxweighted", "rms_label1",
                                 "rms_label2"], a_ctrl)
    ns_a = sorted({n for v in design_a.values() for n in v})
    W.csv("v3a_verdicts.csv", ["eps_code", "seed", "x1", "criterion_C", "reference", "concordance", "classification"] +
          ["rms_N%d" % n for n in ns_a] + ["reference_not_available_reason"], a_verd)
    W.csv("v3a_criterionC.csv", ["eps_code", "seed", "C_verdict", "script_verdict", "note"], a_c)
    W.csv("v3a_stage_identity.csv", ["eps_code", "seed", "N", "G", "inlet_vmin", "v_rms", "G_trace", "G_forward",
                                     "vmin_trace", "vmin_forward", "vrms_trace", "vrms_forward"], a_ident)
    W.csv("v3a_constructor.csv", ["eps_code", "seed", "N", "summary_exit_code", "tsv_exit_code", "summary_status", "max_roundtrip"],
          a_ctor)
    W.csv("v3b_cells.csv", ["eps_code", "seed", "N", "x1", "status", "sigma_T2_O6", "sigma_T2_pollock", "D_P", "SE_D", "z",
                            "D_weight_part", "SE_weight_part", "D_trajectory_part", "SE_trajectory_part", "D_P_common",
                            "SE_D_common", "D_var_tau", "SE_var_tau", "decomposition_residual"], b_cells)
    ns_b = sorted({n for v in design_b.values() for n in v})
    W.csv("v3b_verdicts.csv", ["eps_code", "seed", "x1", "classification", "reason", "threshold_at_Nmax"] +
          ["D_P_N%d" % n for n in ns_b], b_verd)
    W.csv("v3b_noise_limited.csv", ["eps_code", "seed", "x1", "N", "2N", "flag"], b_d6)
    W.csv("v3b_slopes.csv", ["eps_code", "seed", "N", "slope_D_P_vs_x1", "2N", "slope_2N", "order_abs_slope"], b_slopes)
    W.csv("v3b_pollock_status.csv", ["eps_code", "seed", "N", "pass", "n_non_active", "weight_fraction_non_active", "by_code"],
          b_pstat)
    W.csv("v3b_time_snapshots.csv", ["eps_code", "seed", "N", "t", "var_x1_O6", "var_x1_pollock", "D_var_x1", "SE_var_x1",
                                     "var_x2_O6", "var_x2_pollock", "D_var_x2", "SE_var_x2", "var_x3_O6", "var_x3_pollock",
                                     "D_var_x3", "SE_var_x3"], b_snaps)
    W.csv("v3b_consistency.csv", ["eps_code", "seed", "N", "O6_files_vs_v2", "pollock_moments_worst_rel", "pollock_moments_le_1e-12",
                                  "pollock_non_ok_column_matches", "status"], b_cons)

    # ---- markdown
    md = W.md_header("SF-37 digest V3a / V3b (D5, D6)",
                     ["design: `%s`; V3a runs %d (eps, seed) combinations, V3b %d" % (args.design, len(design_a), len(design_b)),
                      "criterion C by `%s` invoked in place" % (os.path.relpath(script, args.root) if script and args.root else script)])
    md += ["", "## V3a - confinement against the verified gridded labels (eps = 1)", "",
           "Rule (RR6): per new seed SF-35 criterion C on the labels at %s decides whether they are a verified reference. Where verified, "
           "the UNWEIGHTED RMS of |psi_grid(x_p) - psi_p| at the grid planes is `concordant` iff it decreases on every refinement. "
           "`reference not verified` (C is not accept) and `reference not available` (inlet backflow) still record the numbers." %
           (list(ns_a),), "",
           L.md_table(["eps_code", "seed", "C verdict", "script verdict", "note"], a_c), "",
           "### Classification per seed and plane", "",
           L.md_table(["eps_code", "seed", "x1", "C", "reference", "RMS decreases", "classification"] + ["RMS N=%d" % n for n in ns_a],
                      [r[:-1] for r in a_verd]), ""]
    nconc = sum(1 for r in a_verd if r[6] == "concordant")
    md += ["Classification counts: %d concordant of %d (seed, plane) cells; the other cells carry their reference state or "
           "`not concordant` / `not_evaluable`." % (nconc, len(a_verd)), ""]
    md += ["### D5", "", "Interpolation proxy (RMS over the 2N lattice of |U_N interpolated - U_2N|), V3a RMS as a multiple of it, the fraction "
           "of particles above 10 x proxy, flux-weighted RMS and per-label RMS: `v3a_cells.csv`; constructor exit code and max round trip: "
           "`v3a_constructor.csv`; stage identity trace vs forward (G, inlet_vmin, v_rms at print precision; absent quantities are "
           "stated): `v3a_stage_identity.csv`; control x1 = 0: `v3a_control_x0.csv`.", ""]
    md += [L.md_table(["eps_code", "seed", "N", "G", "inlet_vmin", "v_rms"], [r[:6] for r in a_ident]), ""]
    md += ["## V3b - Pollock (RT0 on the SF-19 faces) against the Lagrangian O6", "",
           "Rule (RR7): `concordant` iff |D_P| decreases on every refinement %s and |D_P(N_max)| <= max(0.05 sigma_T^2(O6, N_max), "
           "2 SE_D); else `discordant`; missing data `not_evaluable`. Not blocking; a `discordant` at any amplitude is reported to the owner." %
           (" -> ".join(str(n) for n in ns_b),), "",
           L.md_table(["eps_code", "seed", "x1", "classification", "reason", "threshold"] + ["D_P N=%d" % n for n in ns_b],
                      [r[:3] + [r[3], r[4], r[5]] + r[6:] for r in b_verd]), ""]
    nd = sum(1 for r in b_verd if r[3] == "discordant")
    ne = sum(1 for r in b_verd if r[3] == "not_evaluable")
    md += ["Cells: %d discordant, %d not_evaluable, %d concordant of %d." % (nd, ne, sum(1 for r in b_verd if r[3] == "concordant"), len(b_verd)),
           "", "### D6", "", "Decomposition (weight part, trajectory part, each with its paired SE_D), D_P on the common contributors, signed D_P per "
           "N: `v3b_cells.csv`; noise-limited flag per refinement: `v3b_noise_limited.csv`; slope of D_P against x1 and its scaling with N: "
           "`v3b_slopes.csv`; Pollock non-active count and weight per pass: `v3b_pollock_status.csv`; time-snapshot analogues "
           "(var x1, x2, x3 at each t): `v3b_time_snapshots.csv`.", "",
           "### Consistency (recorded)", "",
           L.md_table(["eps_code", "seed", "N", "O6 files vs v2 run", "Pollock moments worst rel. diff", "<= 1e-12", "non_ok column", "status"], b_cons), ""]
    W.md("v3_summary.md", "\n".join(md))
    nman, dig = W.write_manifest("input_manifest_v3.csv")
    with open(os.path.join(args.outdir, "v3_summary.md"), "a") as f:
        f.write("\n## Input manifest\n\n%d input files (path, sha256, bytes in `input_manifest_v3.csv`); sha256 of the manifest listing: `%s`.\n"
                % (nman, dig))
    for c in a_c:
        print("V3a eps=%g seed=%d: C=%s" % (L.eps_of(c[0]), c[1], c[2]))
    print("V3b: %d discordant, %d not_evaluable of %d (eps, seed, plane) cells" % (nd, ne, len(b_verd)))
    print("digest_v3: wrote %d files to %s" % (len(W.written), args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
