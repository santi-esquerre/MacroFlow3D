# SF-37 J2b sf37-audit: orchestrator independent recomputation (scripts/audit_recompute.py, embedded verbatim below).
set -o pipefail
J=~/sf37_audit_job; A=~/sf37_audit
rm -rf "$J" "$A"; mkdir -p "$J" "$A"
cat > "$J/audit_recompute.py" <<'PYEOF'
#!/usr/bin/env python3
"""SF-37 orchestrator FINAL_AUDIT: independent recomputation from the raw campaign outputs.

Written by the orchestrator from the spec text and the activation reading rules (RR1-RR9), WITHOUT importing or
reading the N5 digest code (sf37lib.py, digest_*.py, o3.py).  Not committed; its output is audit evidence that is
compared cell by cell with the committed digests.

usage: audit_recompute.py OUT_ROOT BOOT_NPY AUDIT_DIR [--o3] [--v3a]

Recomputes: moments consistency (vs moments_planes.csv), SE_D (RR3 block bootstrap with the committed matrix),
V1 (all cells), V2 (all cells), production lambda/h (literal + monotone), V3b (all cells), V4 (all cells),
optionally O3a/O3b (all six (eps, seed)) and a V3a spot check (all eps=1 seeds, 4 planes).
"""
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT, BOOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DO_O3 = "--o3" in sys.argv
DO_V3A = "--v3a" in sys.argv
os.makedirs(OUT, exist_ok=True)

M = 256
NP = M * M
PLANES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
EPS = [("10", 1.0), ("15", 1.5), ("20", 2.0)]
SEEDS = [3101, 3102, 3103]
BOOT_SHA = "16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39"

IDX = np.load(BOOT)
sha = hashlib.sha256(np.ascontiguousarray(IDX, dtype="<i8").tobytes()).hexdigest()  # sha of the little-endian int64 data (C-DIGEST)
assert sha == BOOT_SHA, "bootstrap matrix sha256 mismatch %s" % sha
assert IDX.shape == (2000, 16)
assert np.array_equal(IDX, np.random.default_rng(20261017).integers(0, 16, size=(2000, 16)))
CNT = np.zeros((2000, 16))
for b in range(2000):
    for k in IDX[b]:
        CNT[b, k] += 1.0  # multiplicity of block k in resample b

p = np.arange(NP)
BLOCK = (p // M // 64) * 4 + ((p % M) // 64)  # RR3: p = m2 M + m3, block = (m2 // 64) * 4 + m3 // 64

LOG = []


def log(s):
    print(s, flush=True)
    LOG.append(s)


def run_dir(part, name):
    return os.path.join(ROOT, part, name)


def v2name(e, s, n):
    return "v2_e%s_s%d_N%d" % (e, s, n)


def v1name(e, s, n, tol, h):
    return "v1_e%s_s%d_N%d_tol%s_h%d" % (e, s, n, tol, h)


_cache = {}


def load(d, pollock=False):
    key = (d, pollock)
    if key in _cache:
        return _cache[key]
    pre = "pollock_" if pollock else ""
    w = np.load(os.path.join(d, pre + "weights.npy"))
    x23 = np.load(os.path.join(d, pre + "planes_x23.npy"))
    tau = None if pollock else np.load(os.path.join(d, "planes_tau.npy"))
    if pollock:
        tau = np.load(os.path.join(d, "pollock_planes_t.npy"))
    assert w.shape == (NP,) and x23.shape == (NP, len(PLANES), 2), (d, w.shape, x23.shape)
    r = dict(w=w, x23=x23, tau=tau)
    if len(_cache) > 40:
        _cache.clear()
    _cache[key] = r
    return r


def block_sums(w, v, sel):
    """per-block (W, S1, S2) of v shifted by the full-sample weighted mean, over sel."""
    ws = np.where(sel, w, 0.0)
    vv = np.where(sel, v, 0.0)
    W = np.bincount(BLOCK, weights=ws, minlength=16)
    c0 = (ws * vv).sum() / W.sum()
    dv = np.where(sel, v - c0, 0.0)
    S1 = np.bincount(BLOCK, weights=ws * dv, minlength=16)
    S2 = np.bincount(BLOCK, weights=ws * dv * dv, minlength=16)
    return W, S1, S2


def var_from(W, S1, S2):
    m = S1 / W
    return S2 / W - m * m


def q_stats(r, m, q):
    """full-sample value and the 2000 resampled values of statistic q at plane index m (RR1/RR2)."""
    x2 = r["x23"][:, m, 0]
    sel = np.isfinite(x2)
    if q == "sT2":
        out_full, out_bs = 0.0, np.zeros(2000)
        for k in (0, 1):
            W, S1, S2 = block_sums(r["w"], r["x23"][:, m, k], sel)
            out_full += 0.5 * var_from(W.sum(), S1.sum(), S2.sum())
            out_bs += 0.5 * var_from(CNT @ W, CNT @ S1, CNT @ S2)
        return out_full, out_bs, int(sel.sum())
    W, S1, S2 = block_sums(r["w"], r["tau"][:, m], sel)
    return var_from(W.sum(), S1.sum(), S2.sum()), var_from(CNT @ W, CNT @ S1, CNT @ S2), int(sel.sum())


def paired(ra, rb, m, q):
    qa, ba, _ = q_stats(ra, m, q)
    qb, bb, _ = q_stats(rb, m, q)
    return qa, qb, float(np.std(ba - bb, ddof=1))


def order(a, b, c):
    d1, d2 = a - b, b - c
    if d1 == 0.0 or d2 == 0.0 or (d1 > 0) != (d2 > 0):
        return float("nan")
    return math.log2(abs(d1) / abs(d2))


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)


def read_csv_moments(d, fname="moments_planes.csv"):
    with open(os.path.join(d, fname)) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------------------- seed lattice check
s0 = np.load(os.path.join(run_dir("v2", v2name("10", 3101, 64)), "seeds.npy"))
m2s = np.floor(s0[:, 1] * M).astype(int)
m3s = np.floor(s0[:, 2] * M).astype(int)
log("LATTICE p//M==floor(x2 M): %s ; p%%M==floor(x3 M): %s" % (np.array_equal(m2s, p // M), np.array_equal(m3s, p % M)))

# ---------------------------------------------------------------------------------------------- moments consistency
worst = 0.0
nchk = 0
for part, name in [("v2", v2name(e, s, n)) for e, _ in EPS for s in SEEDS for n in (32, 40, 64, 80, 128, 256, 512)] + \
        [("v1", v1name(e, s, n, t, 8)) for e, _ in EPS for s in SEEDS for n in (256, 512) for t in ("1e-8", "1e-10")]:
    d = run_dir(part, name)
    rows = read_csv_moments(d)
    r = load(d)
    for m in range(len(PLANES)):
        st, _, cnt = q_stats(r, m, "sT2")
        vt, _, _ = q_stats(r, m, "tau")
        cs = 0.5 * (float(rows[m]["var_x2"]) + float(rows[m]["var_x3"]))
        ct = float(rows[m]["var_tau"])
        for a, b in ((st, cs), (vt, ct)):
            if np.isfinite(b):
                worst = max(worst, abs(a - b) / max(abs(b), 1e-300))
        nchk += 1
log("MOMENTS max rel diff (sigma_T^2, var tau) recomputed vs moments_planes.csv over %d (run, plane): %.3e" % (nchk, worst))

# ---------------------------------------------------------------------------------------------- V1
rows = []
v1_fail = []
for e, ev in EPS:
    for s in SEEDS:
        for n in (256, 512):
            ra = load(run_dir("v1", v1name(e, s, n, "1e-8", 8)))
            rb = load(run_dir("v1", v1name(e, s, n, "1e-10", 8)))
            for m, x1 in enumerate(PLANES):
                qa, qb, se = paired(ra, rb, m, "sT2")
                D = qa - qb
                det = 1e-3 * abs(qb)
                thr = max(det, 2 * se)
                met = abs(D) <= thr
                branch = "det" if abs(D) <= det else ("stat" if abs(D) <= 2 * se else "fail")
                identical = bool(np.array_equal(ra["x23"][:, m], rb["x23"][:, m], equal_nan=True))
                rows.append([e, s, n, x1, repr(qa), repr(qb), repr(D), repr(se), met, branch, identical])
                if not met:
                    v1_fail.append((e, s, n, x1))
write_csv("v1_cells.csv", ["eps", "seed", "N", "x1", "Q_1e-8", "Q_1e-10", "D", "SE_D", "met", "branch",
                           "x23_bitwise_identical"], rows)
log("V1 cells %d, failing %d: %s" % (len(rows), len(v1_fail), v1_fail[:20]))
for e, ev in EPS:
    sub = [r for r in rows if r[0] == e]
    log("V1 eps %s: met %d/%d ; branches det %d stat %d fail %d ; bitwise-identical planes %d" % (
        e, sum(r[8] for r in sub), len(sub), sum(r[9] == "det" for r in sub), sum(r[9] == "stat" for r in sub),
        sum(r[9] == "fail" for r in sub), sum(r[10] for r in sub)))

# ---------------------------------------------------------------------------------------------- V2 + lambda/h
rows = []
v2_fail = []
LH = {32: 8, 40: 10, 64: 16, 80: 20, 128: 32, 256: 64}
lh_pass = {(e, n): True for e, _ in EPS for n in LH}
lh_rows = []
for e, ev in EPS:
    for s in SEEDS:
        R = {n: load(run_dir("v2", v2name(e, s, n))) for n in (32, 40, 64, 80, 128, 256, 512)}
        for m, x1 in enumerate(PLANES):
            for q in ("sT2", "tau"):
                q128 = q_stats(R[128], m, q)[0]
                q256, q512, se = paired(R[256], R[512], m, q)
                p_ = order(q128, q256, q512)
                thr = max(0.02 * abs(q512), 2 * se)
                met = (np.isfinite(p_) and p_ >= 1.5) or abs(q256 - q512) <= thr
                rows.append([e, s, x1, q, repr(q128), repr(q256), repr(q512), repr(p_), repr(se), met])
                if not met:
                    v2_fail.append((e, s, x1, q))
                for n in LH:
                    qn, q5, se_n = paired(R[n], R[512], m, q)
                    ok = abs(qn - q5) <= max(0.05 * abs(q5), 2 * se_n)
                    lh_rows.append([e, s, n, LH[n], x1, q, repr(qn), repr(q5), repr(se_n), ok])
                    if not ok:
                        lh_pass[(e, n)] = False
write_csv("v2_cells.csv", ["eps", "seed", "x1", "Q", "Q128", "Q256", "Q512", "order", "SE_D_256_512", "met"], rows)
write_csv("lambda_h_cells.csv", ["eps", "seed", "N", "lambda_h", "x1", "Q", "QN", "Q512", "SE_D", "ok"], lh_rows)
log("V2 cells %d, failing %d: %s" % (len(rows), len(v2_fail), v2_fail[:30]))
for e, ev in EPS:
    passing = [LH[n] for n in sorted(LH) if lh_pass[(e, n)]]
    literal = passing[0] if passing else None
    mono = None
    for n in sorted(LH):
        if all(lh_pass[(e, k)] for k in LH if k >= n):
            mono = LH[n]
            break
    log("LAMBDA_H eps %s: passing %s ; literal %s ; monotone %s" % (e, passing, literal, mono))

# ---------------------------------------------------------------------------------------------- V4 (CSV rms_ev)
rows = []
v4_fail = []
for e, ev in EPS:
    for s in SEEDS:
        ev_ = {n: read_csv_moments(run_dir("v2", v2name(e, s, n))) for n in (64, 128, 256, 512)}
        for m, x1 in enumerate(PLANES):
            if x1 > 4:
                continue
            vals = [float(ev_[n][m]["rms_ev"]) for n in (64, 128, 256, 512)]
            dec = all(vals[i + 1] < vals[i] for i in range(3))
            o = math.log2(vals[2] / vals[3]) if vals[2] > 0 and vals[3] > 0 else float("nan")
            met = dec and np.isfinite(o) and o >= 1.8
            rows.append([e, s, x1] + [repr(v) for v in vals] + [repr(o), dec, met])
            if not met:
                v4_fail.append((e, s, x1, round(o, 3)))
write_csv("v4_cells.csv", ["eps", "seed", "x1", "rms_ev_64", "rms_ev_128", "rms_ev_256", "rms_ev_512", "order_256_512",
                           "decreasing", "met"], rows)
log("V4 cells %d, failing %d: %s" % (len(rows), len(v4_fail), v4_fail))
log("V4 order 256->512 range: %.3f .. %.3f" % (min(float(r[7]) for r in rows), max(float(r[7]) for r in rows)))

# ---------------------------------------------------------------------------------------------- V3b
rows = []
for e, ev in EPS:
    for s in SEEDS:
        D = {}
        SE = {}
        S6 = {}
        WP = {}  # D6 weight part: Pollock weights on the O6 positions minus O6 (trajectory part = D - WP)
        for n in (64, 128, 256, 512):
            d = run_dir("v3b", "v3b_e%s_s%d_N%d" % (e, s, n))
            rp, ro = load(d, pollock=True), load(d)
            rh = dict(w=rp["w"], x23=ro["x23"], tau=ro["tau"])
            for m in range(len(PLANES)):
                qp, qo, se = paired(rp, ro, m, "sT2")
                D[(n, m)], SE[(n, m)], S6[(n, m)] = qp - qo, se, qo
                WP[(n, m)] = q_stats(rh, m, "sT2")[0] - qo
        for m, x1 in enumerate(PLANES):
            a = [abs(D[(n, m)]) for n in (64, 128, 256, 512)]
            dec = all(a[i + 1] < a[i] for i in range(3))
            small = a[3] <= max(0.05 * S6[(512, m)], 2 * SE[(512, m)])
            rows.append([e, s, x1] + [repr(float(D[(n, m)])) for n in (64, 128, 256, 512)] + [repr(SE[(512, m)]), dec, small,
                                                                                       "concordant" if dec and small else "discordant"]
                        + [repr(float(WP[(n, m)])) for n in (64, 128, 256, 512)]
                        + [repr(float(D[(n, m)] - WP[(n, m)])) for n in (64, 128, 256, 512)])
write_csv("v3b_cells.csv", ["eps", "seed", "x1", "DP64", "DP128", "DP256", "DP512", "SE_D_512", "decreasing", "small",
                            "class", "WP64", "WP128", "WP256", "WP512", "TP64", "TP128", "TP256", "TP512"], rows)
for e, ev in EPS:
    sub = [r for r in rows if r[0] == e]
    log("V3b eps %s: concordant %d / %d" % (e, sum(r[10] == "concordant" for r in sub), len(sub)))

# ---------------------------------------------------------------------------------------------- V3a spot check
if DO_V3A:
    from scipy.ndimage import map_coordinates
    rows = []
    for s in SEEDS:
        for n in (64, 128, 256):
            ex = os.path.join(ROOT, "labels", "exports", "eps10_s%d" % s, "N%d" % n)
            if not os.path.exists(os.path.join(ex, "psi1.npy")):
                continue
            p1 = np.load(os.path.join(ex, "psi1.npy"), mmap_mode="r")
            p2 = np.load(os.path.join(ex, "psi2.npy"), mmap_mode="r")
            d = run_dir("v3a", "v3a_e10_s%d_N%d" % (s, n))
            x23 = np.load(os.path.join(d, "planes_x23.npy"))
            lab = np.load(os.path.join(d, "labels.npy"))
            g = np.arange(n) / n
            for m, x1 in enumerate((0.25, 0.5, 0.75, 1.0)):
                j = int(round(x1 * n))
                U1 = np.asarray(p1[j]) - g[:, None]
                U2 = np.asarray(p2[j]) - g[None, :]
                xs = x23[:, m, :]
                ok = np.isfinite(xs[:, 0]) & np.isfinite(lab[:, 0]) & np.isfinite(lab[:, 1])
                c = [np.mod(xs[ok, 0], 1.0) * n, np.mod(xs[ok, 1], 1.0) * n]
                i1 = map_coordinates(U1, c, order=3, mode="grid-wrap", prefilter=True)
                i2 = map_coordinates(U2, c, order=3, mode="grid-wrap", prefilter=True)
                err = np.hypot(xs[ok, 0] + i1 - lab[ok, 0], xs[ok, 1] + i2 - lab[ok, 1])
                rows.append([s, n, x1, int(ok.sum()), repr(float(np.sqrt(np.mean(err ** 2)))), repr(float(err.max()))])
    write_csv("v3a_spot.csv", ["seed", "N", "x1", "count", "rms", "max"], rows)
    log("V3A spot rows %d (see v3a_spot.csv)" % len(rows))

# ---------------------------------------------------------------------------------------------- O3
def c_plane(psi1, psi2, n, j):
    """c = grad psi1 x grad psi2 at vertex plane j (1 <= j <= n-1), 2nd-order central, periodic in x2, x3."""
    h = 1.0 / n
    g = np.arange(n) / n

    def U(psi, which, jj):
        a = np.asarray(psi[jj], dtype=np.float64)
        return a - (g[:, None] if which == 1 else g[None, :])

    def grad(psi, which):
        u0 = U(psi, which, j)
        g1 = (U(psi, which, j + 1) - U(psi, which, j - 1)) / (2 * h)
        g2 = (np.roll(u0, -1, axis=0) - np.roll(u0, 1, axis=0)) / (2 * h)
        g3 = (np.roll(u0, -1, axis=1) - np.roll(u0, 1, axis=1)) / (2 * h)
        if which == 1:
            g2 = g2 + 1.0
        else:
            g3 = g3 + 1.0
        return g1, g2, g3

    a1, a2, a3 = grad(psi1, 1)
    b1, b2, b3 = grad(psi2, 2)
    return np.stack([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])


if DO_O3:
    q = 1e-3
    rows = []
    for e, ev in (("15", 1.5), ("20", 2.0)):
        for s in SEEDS:
            ex = lambda n: os.path.join(ROOT, "labels", "exports", "eps%s_s%d" % (e, s), "N%d" % n)
            res = {}
            for n in (128, 256, 512):
                nc = n // 2
                P1f = np.load(os.path.join(ex(n), "psi1.npy"), mmap_mode="r")
                P2f = np.load(os.path.join(ex(n), "psi2.npy"), mmap_mode="r")
                P1c = np.load(os.path.join(ex(nc), "psi1.npy"), mmap_mode="r")
                P2c = np.load(os.path.join(ex(nc), "psi2.npy"), mmap_mode="r")
                vd = run_dir("v2", v2name(e, s, n)) + "/vd"
                V = [np.load(os.path.join(vd, "vD%d.npy" % k), mmap_mode="r") for k in (1, 2, 3)]
                dd, cf2, vdiff2, vd2 = [], [], [], []
                for jc in range(1, nc):
                    jf = 2 * jc
                    cf = c_plane(P1f, P2f, n, jf)[:, ::2, ::2]
                    cc = c_plane(P1c, P2c, nc, jc)
                    vv = np.stack([np.asarray(V[k][jf, ::2, ::2]) for k in range(3)])
                    dd.append(np.sqrt(((cf - cc) ** 2).sum(axis=0)).ravel())
                    cf2.append((cf ** 2).sum(axis=0).ravel())
                    vdiff2.append(((cf - vv) ** 2).sum(axis=0).ravel())
                    vd2.append((vv ** 2).sum(axis=0).ravel())
                dd = np.concatenate(dd)
                cf2 = np.concatenate(cf2)
                vdiff2 = np.concatenate(vdiff2)
                vd2 = np.concatenate(vd2)
                ncom = dd.size
                d = dd / math.sqrt(cf2.mean())
                k = math.ceil(q * ncom)
                order_idx = np.lexsort((np.arange(ncom), -d))  # largest d first, ties by vertex index
                marked = np.zeros(ncom, bool)
                marked[order_idx[:k]] = True
                vrms = math.sqrt(vd2.mean())
                evq = math.sqrt(vdiff2[~marked].mean()) / vrms
                e2 = np.sort(d * d)[::-1]
                cs = np.cumsum(e2)
                nhalf = int(np.searchsorted(cs, 0.5 * cs[-1]) + 1)
                res[n] = dict(evq=evq, fhalf=nhalf / ncom, f01=float((d > 0.1).mean()), ncom=ncom, vrms=vrms)
                rows.append([e, s, n, ncom, k, repr(evq), repr(nhalf / ncom), repr(float((d > 0.1).mean())), repr(vrms)])
                log("O3 eps %s seed %d N %d: e_v^q %.6e f_half %.6e f(d>0.1) %.6e v_rms %.6e" % (
                    e, s, n, evq, nhalf / ncom, float((d > 0.1).mean()), vrms))
            ev3 = [res[n]["evq"] for n in (128, 256, 512)]
            dec = ev3[1] < ev3[0] and ev3[2] < ev3[1]
            o = math.log2(ev3[1] / ev3[2])
            rel = abs(ev3[2] - ev3[1]) / abs(ev3[1])
            o3a = "accept" if dec and o >= 1.8 else ("reject" if rel < 0.10 and ev3[2] > 1e-12 else "unresolved")
            fh = [res[n]["fhalf"] for n in (128, 256, 512)]
            o3b = "shrinking" if fh[1] < fh[0] and fh[2] < fh[1] else "not shrinking"
            log("O3 VERDICT eps %s seed %d: O3a %s (order %.3f, rel %.3f) O3b %s" % (e, s, o3a, o, rel, o3b))
    write_csv("o3_cells.csv", ["eps", "seed", "N", "n_common", "n_marked", "e_v_q", "f_half", "frac_d_gt_0.1", "v_rms"], rows)

with open(os.path.join(OUT, "audit_summary.txt"), "w") as f:
    f.write("\n".join(LOG) + "\n")
print("AUDIT_DONE")
PYEOF
echo "AUDIT_SCRIPT_SHA256=$(sha256sum "$J/audit_recompute.py" | cut -d' ' -f1)"
cd ~/MacroFlow3D-SF-37 || exit 2
python3 -c 'import sys, numpy, scipy; print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__)'
/usr/bin/time -v -o "$A/audit.time" python3 "$J/audit_recompute.py" ~/sf37_campaign \
    docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification/analysis/bootstrap_blocks.npy "$A" --o3 --v3a
echo "AUDIT_EXIT=$?"
ls -l "$A"
echo AUDIT_JOB_END
