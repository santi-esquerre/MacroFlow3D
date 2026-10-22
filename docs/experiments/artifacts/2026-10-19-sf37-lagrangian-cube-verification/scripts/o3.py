#!/usr/bin/env python3
"""SF-37 O3: robust self-convergence metric of the gridded labels (blind application, RR9, D7).

  o3.py ROOT OUTDIR [--design full|discovered]

FROZEN BEFORE ANY eps 1.5 / 2 EXPORT OF SEEDS 3101-3103 IS READ (RR9).  Pre-registered definitions, implemented literally:

  * labels  : ROOT/labels/exports/eps{E}_s{S}/N{N}/{psi1,psi2}.npy, float64 (N+1, N, N), idx = m3 + N (m2 + N j), x = m / N,
              NO periodic duplicate in x2, x3; psi1 = x2 + U1, psi2 = x3 + U2 (E = 15, 20 for eps 1.5, 2).
  * c       : c = grad psi1 x grad psi2 exactly as `label_diagnostics.py` section 3 (differentiate U = psi - x, add the affine
              1, second-order central differences, periodic in x2, x3, interior planes 1..N-1).
  * pair    : (N/2, N), N in {128, 256, 512}.  Common vertices = the N/2 lattice on the interior coarse planes j = 1..N/2-1 with
              all (N/2)^2 transverse vertices.  d(x) = |c_N(x) - c_{N/2}(x)| / RMS|c_N| (RMS over the common vertices).
              M_q(N) = the ceil(q n_common) vertices with the largest d, q = 1e-3, ties broken by the flattened common-vertex index.
  * O3a     : e_v^(q)(N) = RMS over the UNMARKED common vertices of |c_N - vD_N| / v_rms(N), vD_N from
              ROOT/v2/v2_e{E}_s{S}_N{N}/vd/{vD1,vD2,vD3}.npy, v_rms(N) = RMS|vD_N| over ALL common vertices.  Verdict per
              (eps, seed): accept (decreases on 128 -> 256 -> 512 and log2(e(256)/e(512)) >= 1.8), reject (|e(512) - e(256)| /
              e(256) < 0.10 while e(512) > 1e-12), else unresolved; missing grid / vD, construction exit 17 or 19, non-finite
              value -> not_evaluable (never unresolved).
  * O3b     : f_half(N) = fraction of common vertices carrying half of sum d^2 (= label_diagnostics `concentration` frac) and
              the fraction with d > 0.1.  shrinking iff f_half decreases on every refinement, else not shrinking.
  * O3c     : x1 in {0.25, 0.5}: weighted fraction (O6 weights of the v2 run, renormalized over the contributors) of landings
              (mod 1) within one coarse cell of a marked vertex of that plane (periodic, |dx| <= 2/N in x2 and x3), plus the
              pooled fraction; x1 = 1 is the outlet coarse plane (no marks by construction): structurally empty.
  * D7      : N_half and its growth per doubling, d_max and its location (cube units), RMS d over the unmarked vertices and
              its pair order.  Reading note: f_half shrinking means concentration of the defect, not convergence.

Memory: the labels and vD are memory-mapped; c is computed by chunks of coarse planes (a chunk needs the fine planes
2a-1 .. 2b+1 only).  Resident arrays at the N = 512 pair: four (N/2-1) x (N/2)^2 float64 arrays (err2, v2, dv2, vv2; 4 x 134 MB)
+ the ranking (index array 134 MB + sort copies ~ 3 x 134 MB) + one chunk (~0.3 GB): estimated peak ~ 1.5 GB anonymous memory
(plus the mapped label pages, reclaimable); the measured ru_maxrss is written to the summary.  Target <= ~20 GB.

Blinding: only the O3 outputs are printed or written (no per-plane label metrics).  python3 + numpy only (no scipy).
"""
import argparse
import fractions
import json
import math
import os
import re
import resource
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf37lib as L  # noqa: E402

NAN = float("nan")
Q_MARK = fractions.Fraction(1, 1000)  # q = 1e-3 (exact)
D_LARGE = 0.1
O3_EPS = (15, 20)
O3_CHAIN = (128, 256, 512)
O3_SEEDS = L.SEEDS
O3A_ORDER = 1.8
O3A_REJECT_REL = 0.10
O3A_FLOOR = 1e-12
O3C_PLANES = (0.25, 0.5)
CHUNK_BYTES = 48 << 20  # budget of one label slab per component
BAD_EXITS = (17, 19)
OK_EXITS = (0, 21)
READING_NOTE = ("f_half shrinking means concentration of the defect, not convergence "
                "(N_half growth ~8 uniform, ~4 sheet, ~2 curve per doubling)")


# ------------------------------------------------------------------------------------------------------------
# c = grad psi1 x grad psi2 (label_diagnostics.py section 3, chunked)
# ------------------------------------------------------------------------------------------------------------
def _periodic_part(slab, n, which):
    x = np.arange(n, dtype=np.float64) / n
    a = np.asarray(slab)
    return a - (x[None, :, None] if which == 1 else x[None, None, :])


def cross_planes(psi1, psi2, n, jfirst, cnt, step=1):
    """c at the interior planes jfirst, jfirst + step, ..., jfirst + step (cnt - 1) of an n-lattice (1 <= plane <= n - 1).
    Arithmetic identical, element by element, to label_diagnostics.cross (which computes every interior plane): returns the
    three components, each (cnt, n, n)."""
    h = 1.0 / n
    lo = jfirst - 1
    hi = jfirst + step * (cnt - 1) + 1
    if lo < 0 or hi > n:
        raise ValueError("planes %d..%d outside the interior of an n = %d lattice" % (jfirst, jfirst + step * (cnt - 1), n))
    grads = []
    for which, psi in ((1, psi1), (2, psi2)):
        U = _periodic_part(psi[lo:hi + 1], n, which)
        q = U[1:1 + step * cnt:step]
        plus = U[2:2 + step * cnt:step]
        minus = U[0:step * cnt:step]
        g1 = (plus - minus) / (2 * h)
        g2 = (np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1)) / (2 * h)
        g3 = (np.roll(q, -1, axis=2) - np.roll(q, 1, axis=2)) / (2 * h)
        if which == 1:
            g2 = g2 + 1.0
        else:
            g3 = g3 + 1.0
        grads.append((g1, g2, g3))
    (a1, a2, a3), (b1, b2, b3) = grads
    return (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)


def ceil_q(n, q=Q_MARK):
    """ceil(q n) in exact arithmetic."""
    return int(-((-n * q.numerator) // q.denominator))


def chunk_coarse_planes(N):
    per_plane = N * N * 8
    return max(1, ((CHUNK_BYTES // per_plane) - 1) // 2)


def pair_arrays(p1c, p2c, p1f, p2f, N, vD=None, chunk=None):
    """Per common vertex (coarse interior planes 1..nc-1, flattened C order of (nc-1, nc, nc)):
    err2 = |c_{N/2} - c_N|^2, v2 = |c_N|^2 and, if vD = (vD1, vD2, vD3) at the N lattice is given,
    dv2 = |c_N - vD_N|^2, vv2 = |vD_N|^2.  Arrays are returned in shape (nc-1, nc, nc)."""
    nc = N // 2
    if N % 2 or nc < 2:
        raise ValueError("N = %d is not a valid fine lattice" % N)
    if p1f.shape != (N + 1, N, N) or p1c.shape != (nc + 1, nc, nc):
        raise ValueError("label shapes %s / %s do not match the pair (%d, %d)" % (p1c.shape, p1f.shape, nc, N))
    shape = (nc - 1, nc, nc)
    err2 = np.empty(shape)
    v2 = np.empty(shape)
    dv2 = np.empty(shape) if vD is not None else None
    vv2 = np.empty(shape) if vD is not None else None
    cnt_max = chunk or chunk_coarse_planes(N)
    a = 1
    while a <= nc - 1:
        cnt = min(cnt_max, nc - a)
        cc = cross_planes(p1c, p2c, nc, a, cnt, 1)
        cf = [c[:, ::2, ::2] for c in cross_planes(p1f, p2f, N, 2 * a, cnt, 2)]
        err2[a - 1:a - 1 + cnt] = sum((x - y) ** 2 for x, y in zip(cc, cf))
        v2[a - 1:a - 1 + cnt] = sum(y ** 2 for y in cf)
        if vD is not None:
            sl = (slice(2 * a, 2 * (a + cnt - 1) + 1, 2), slice(None, None, 2), slice(None, None, 2))
            vc = [np.asarray(v[sl]) for v in vD]
            dv2[a - 1:a - 1 + cnt] = sum((x - y) ** 2 for x, y in zip(cf, vc))
            vv2[a - 1:a - 1 + cnt] = sum(y ** 2 for y in vc)
        a += cnt
    return err2, v2, dv2, vv2


# ------------------------------------------------------------------------------------------------------------
# metrics of one pair
# ------------------------------------------------------------------------------------------------------------
def f_half_of(d):
    """label_diagnostics.concentration: (N_half, fraction) of the vertices carrying half of sum d^2 (d 1-D)."""
    tot = float(np.sum(d * d))
    srt = np.sort(d)[::-1]
    cs = np.cumsum(srt * srt)
    k = int(np.searchsorted(cs, 0.5 * tot) + 1)
    return k, k / d.size


def mark_top_q(d, q=Q_MARK):
    """Flat indices (ascending rank) of the ceil(q n) largest d, ties broken by ascending flat index."""
    nm = ceil_q(d.size, q)
    order = np.argsort(-d, kind="stable")
    return order[:nm]


def pair_metrics(err2, v2, dv2=None, vv2=None, q=Q_MARK):
    """O3 metrics of one pair from the per-common-vertex squared arrays.  -> dict (with 'marked', the flat marked indices)."""
    shape = err2.shape
    if not (np.isfinite(err2).all() and np.isfinite(v2).all()):
        raise FloatingPointError("non-finite value in the c comparison")
    vrms = float(np.sqrt(v2.mean()))
    d = np.sqrt(err2).ravel() / vrms
    n = d.size
    nhalf, frac = f_half_of(d)
    marked = mark_top_q(d, q)
    mk = np.zeros(n, dtype=bool)
    mk[marked] = True
    j, m2, m3 = np.unravel_index(int(np.argmax(err2)), shape)
    nc = shape[1]
    out = {"n_common": int(n), "n_marked": int(marked.size), "rms_c_N": vrms, "f_half": frac, "N_half": nhalf,
           "frac_d_gt_0.1": float(np.mean(d > D_LARGE)), "d_max": float(d.max()),
           "d_max_loc": ((j + 1) / nc, m2 / nc, m3 / nc), "rms_d_all": float(np.sqrt(np.mean(d * d))),
           "rms_d_unmarked": float(np.sqrt(np.mean(d[~mk] ** 2))) if (~mk).any() else NAN, "marked": marked,
           "e_v_q": NAN, "e_v_all": NAN, "v_rms_common_vD": NAN}
    if dv2 is not None:
        if not (np.isfinite(dv2).all() and np.isfinite(vv2).all()):
            raise FloatingPointError("non-finite value in the vD comparison")
        vr = float(np.sqrt(vv2.mean()))
        out["v_rms_common_vD"] = vr
        dflat = dv2.ravel()
        out["e_v_q"] = float(np.sqrt(np.mean(dflat[~mk]))) / vr if (~mk).any() else NAN
        out["e_v_all"] = float(np.sqrt(np.mean(dflat))) / vr
    return out


# ------------------------------------------------------------------------------------------------------------
# verdict rules
# ------------------------------------------------------------------------------------------------------------
def verdict_o3a(chain, values):
    """chain: ascending N list; values: {N: e_v^(q)}.  -> (verdict, reason, order_last, relchange_last)."""
    seq = [values.get(n, NAN) for n in chain]
    if len(chain) < 2:
        return "not_evaluable", "fewer than two grids in the chain", NAN, NAN
    miss = [n for n in chain if n not in values or not isinstance(values[n], (float, int, np.floating)) or not math.isfinite(values[n])]
    if miss:
        return "not_evaluable", "missing or non-finite e_v^(q) at N=" + ",".join(str(m) for m in miss), NAN, NAN
    dec = all(seq[i + 1] < seq[i] for i in range(len(seq) - 1))
    order = L.order_pair(seq[-2], seq[-1])
    rel = abs(seq[-1] - seq[-2]) / abs(seq[-2]) if seq[-2] != 0.0 else float("inf")
    if dec and math.isfinite(order) and order >= O3A_ORDER:
        return "accept", "decreases on every refinement, order %.3f >= %.1f on %d -> %d" % (order, O3A_ORDER, chain[-2], chain[-1]), order, rel
    if rel < O3A_REJECT_REL and seq[-1] > O3A_FLOOR:
        return "reject", "relative change %.4f < 0.10 between %d and %d while e_v^(q)(%d) = %.3e > 1e-12" % (
            rel, chain[-2], chain[-1], chain[-1], seq[-1]), order, rel
    return "unresolved", "decreases=%s, order %.3f, relative change %.4f" % (dec, order, rel), order, rel


def verdict_o3b(chain, fhalf):
    """shrinking iff f_half decreases strictly on every refinement; else not shrinking; missing -> not_evaluable."""
    seq = [fhalf.get(n, NAN) for n in chain]
    if len(chain) < 2:
        return "not_evaluable", "fewer than two grids in the chain"
    bad = [n for n, v in zip(chain, seq) if not math.isfinite(v)]
    if bad:
        return "not_evaluable", "missing or non-finite f_half at N=" + ",".join(str(b) for b in bad)
    dec = all(seq[i + 1] < seq[i] for i in range(len(seq) - 1))
    return ("shrinking" if dec else "not shrinking"), "f_half = " + ", ".join("%.6g" % v for v in seq)


# ------------------------------------------------------------------------------------------------------------
# O3c geometry
# ------------------------------------------------------------------------------------------------------------
def near_marked(x2, x3, mark_plane, N):
    """True for landings (x2, x3) (cube units, any real, taken mod 1) whose periodic distance to some marked vertex of the
    coarse plane (boolean (nc, nc), vertex (m2, m3) at (m2, m3) 2/N) is <= 2/N in BOTH coordinates."""
    nc = N // 2
    cell = 2.0 / N
    u2 = np.floor(np.mod(x2, 1.0) * nc).astype(np.int64)
    u3 = np.floor(np.mod(x3, 1.0) * nc).astype(np.int64)
    hit = np.zeros(x2.shape, dtype=bool)
    for k2 in (-1, 0, 1, 2):
        m2 = (u2 + k2) % nc
        d2 = x2 - m2 * cell
        d2 = d2 - np.round(d2)
        ok2 = np.abs(d2) <= cell
        for k3 in (-1, 0, 1, 2):
            m3 = (u3 + k3) % nc
            d3 = x3 - m3 * cell
            d3 = d3 - np.round(d3)
            ok3 = np.abs(d3) <= cell
            hit |= mark_plane[m2, m3] & ok2 & ok3
    return hit


def marks_on_plane(marked, N, jc):
    """Boolean (nc, nc) of the marked vertices on coarse plane jc (1 <= jc <= nc - 1)."""
    nc = N // 2
    jj, m2, m3 = np.unravel_index(marked, (nc - 1, nc, nc))
    sel = jj == (jc - 1)
    B = np.zeros((nc, nc), dtype=bool)
    B[m2[sel], m3[sel]] = True
    return B


def o3c_fractions(run, marked, N, planes=O3C_PLANES):
    """-> list of dict rows per plane x1 in `planes` (+ 'pooled' + the structurally empty outlet).  `run`: RunData of the v2 run."""
    rows = []
    nc = N // 2
    w = run.w
    pm = run.plane_mask()
    X = run.mem("planes_x23.npy")
    xs = run.planes
    sels, hits = {}, {}
    for x1 in planes:
        idx = [i for i, v in enumerate(xs) if abs(v - x1) < 1e-12]
        jcf = x1 * nc
        if not idx or abs(jcf - round(jcf)) > 1e-9 or not (1 <= round(jcf) <= nc - 1):
            rows.append({"x1": x1, "status": "not_evaluable", "frac": NAN, "n": 0, "n_hit": 0,
                         "reason": "plane absent from the run or not on the coarse lattice"})
            continue
        m = idx[0]
        jc = int(round(jcf))
        sel = pm[:, m]
        B = marks_on_plane(marked, N, jc)
        hit = near_marked(np.asarray(X[sel, m, 0], dtype=np.float64), np.asarray(X[sel, m, 1], dtype=np.float64), B, N)
        ws = w[sel]
        W = L.seq_sum(ws)
        frac = L.seq_sum(ws[hit]) / W if W > 0.0 else NAN
        rows.append({"x1": x1, "status": "ok", "frac": frac, "n": int(sel.sum()), "n_hit": int(hit.sum()),
                     "n_marked_plane": int(B.sum()), "reason": ""})
        sels[x1] = (m, sel, jc)
        full_hit = np.zeros(run.Np, dtype=bool)
        full_hit[np.flatnonzero(sel)] = hit
        hits[x1] = full_hit
    if len(hits) == len(planes):
        both = np.ones(run.Np, dtype=bool)
        anyhit = np.zeros(run.Np, dtype=bool)
        for x1 in planes:
            both &= sels[x1][1]
            anyhit |= hits[x1]
        ws = w[both]
        W = L.seq_sum(ws)
        rows.append({"x1": "pooled", "status": "ok", "frac": (L.seq_sum(ws[anyhit[both]]) / W) if W > 0.0 else NAN,
                     "n": int(both.sum()), "n_hit": int(anyhit[both].sum()), "reason": ""})
    else:
        rows.append({"x1": "pooled", "status": "not_evaluable", "frac": NAN, "n": 0, "n_hit": 0,
                     "reason": "a plane is not evaluable"})
    rows.append({"x1": 1.0, "status": "structurally_empty", "frac": NAN, "n": 0, "n_hit": 0,
                 "reason": "x1 = 1 is the outlet coarse plane: no marked vertices by construction (interior planes only)"})
    return rows


# ------------------------------------------------------------------------------------------------------------
# campaign handling
# ------------------------------------------------------------------------------------------------------------
def read_exit_codes(root, manifest):
    """case -> rc from ROOT/labels/exit_codes_labels-o3.tsv and exit_codes_labels-eps10.tsv (header: case rc wall_s end_utc)."""
    codes = {}
    for fn in ("exit_codes_labels-o3.tsv", "exit_codes_labels-eps10.tsv"):
        p = os.path.join(root, "labels", fn)
        if not os.path.isfile(p):
            continue
        manifest.add(p)
        with open(p) as f:
            for ln in f:
                t = ln.strip().split()
                if len(t) >= 2 and t[0] != "case":
                    try:
                        codes[t[0]] = int(t[1])
                    except ValueError:
                        pass
    return codes


def export_dir(root, e, s, n):
    return os.path.join(root, "labels", "exports", "eps%d_s%d" % (e, s), "N%d" % n)


def vd_dir(root, e, s, n):
    return os.path.join(root, "v2", L.v2_name(e, s, n), "vd")


def load_labels(root, e, s, n, manifest):
    """-> (psi1, psi2) memory-mapped or (None, reason)."""
    d = export_dir(root, e, s, n)
    arrs = []
    for nm in ("psi1.npy", "psi2.npy"):
        p = os.path.join(d, nm)
        if not os.path.isfile(p):
            return None, "label export missing: %s" % os.path.relpath(p, root)
        manifest.add(p)
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        if a.shape != (n + 1, n, n) or a.dtype != np.float64:
            return None, "label export %s has shape %s dtype %s, expected (%d,%d,%d) float64" % (
                os.path.relpath(p, root), a.shape, a.dtype, n + 1, n, n)
        arrs.append(a)
    return tuple(arrs), ""


def load_vd(root, e, s, n, manifest):
    """-> ((vD1, vD2, vD3), v_rms_json) or (None, reason)."""
    d = vd_dir(root, e, s, n)
    arrs = []
    for nm in ("vD1.npy", "vD2.npy", "vD3.npy"):
        p = os.path.join(d, nm)
        if not os.path.isfile(p):
            return None, "vD export missing: %s" % os.path.relpath(p, root)
        manifest.add(p)
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        if a.shape != (n + 1, n, n) or a.dtype != np.float64:
            return None, "vD export %s has shape %s dtype %s" % (os.path.relpath(p, root), a.shape, a.dtype)
        arrs.append(a)
    vj = NAN
    pj = os.path.join(d, "vD.json")
    if os.path.isfile(pj):
        manifest.add(pj)
        try:
            with open(pj) as f:
                vj = float(json.load(f).get("v_rms", NAN))
        except (ValueError, TypeError, OSError):
            vj = NAN
    return (tuple(arrs), vj), ""


class O3Design:
    def __init__(self, kind, eps_codes, seeds, chain, grids_of):
        self.kind = kind
        self.eps_codes = tuple(eps_codes)
        self.seeds = tuple(seeds)
        self.chain = tuple(chain)
        self.grids_of = grids_of  # (e, s) -> sorted N with an export directory (for 'discovered')

    @staticmethod
    def full():
        return O3Design("full", O3_EPS, O3_SEEDS, O3_CHAIN, None)

    @staticmethod
    def discovered(root):
        base = os.path.join(root, "labels", "exports")
        eps, seeds, found = set(), set(), {}
        if os.path.isdir(base):
            for nm in sorted(os.listdir(base)):
                m = re.match(r"^eps(\d+)_s(\d+)$", nm)
                if not m or int(m.group(1)) not in O3_EPS:
                    continue
                e, s = int(m.group(1)), int(m.group(2))
                ns = []
                for sub in sorted(os.listdir(os.path.join(base, nm))):
                    mm = re.match(r"^N(\d+)$", sub)
                    if mm and os.path.isfile(os.path.join(base, nm, sub, "psi1.npy")):
                        ns.append(int(mm.group(1)))
                eps.add(e)
                seeds.add(s)
                found[(e, s)] = sorted(ns)
        allp = set()
        for ns in found.values():
            allp |= {n for n in ns if n % 2 == 0 and n // 2 in ns}
        chain = tuple(sorted(allp)[-3:])
        return O3Design("discovered", sorted(eps), sorted(seeds), chain, found)


def evaluate_cell(root, e, s, design, manifest, exits):
    """All pairs of one (eps, seed).  -> dict(per_n={N: metrics or None}, reasons={N: str}, marks={N: idx})."""
    per, reasons = {}, {}
    chain = design.chain
    needed = sorted({n for N in chain for n in (N // 2, N)})
    bad = []
    for n in needed:
        rc = exits.get("eps%d_s%d/N%d" % (e, s, n))
        if rc in BAD_EXITS:
            bad.append("N=%d construction exit %d" % (n, rc))
        elif rc is not None and rc not in OK_EXITS:
            bad.append("N=%d unexpected construction exit %d" % (n, rc))
    for N in chain:
        nc = N // 2
        why = [b for b in bad if b.startswith("N=%d " % N) or b.startswith("N=%d " % nc)]
        if why:
            per[N], reasons[N] = None, "; ".join(why)
            continue
        fine, rf = load_labels(root, e, s, N, manifest)
        coarse, rc_ = load_labels(root, e, s, nc, manifest)
        vd, rv = load_vd(root, e, s, N, manifest)
        miss = [r for x, r in ((fine, rf), (coarse, rc_), (vd, rv)) if x is None]
        if miss:
            per[N], reasons[N] = None, "; ".join(miss)
            continue
        (vD, vj) = vd
        try:
            err2, v2, dv2, vv2 = pair_arrays(coarse[0], coarse[1], fine[0], fine[1], N, vD)
            mt = pair_metrics(err2, v2, dv2, vv2)
        except FloatingPointError as ex:
            per[N], reasons[N] = None, "non-finite value: %s" % ex
            continue
        except ValueError as ex:
            per[N], reasons[N] = None, str(ex)
            continue
        del err2, v2, dv2, vv2
        mt["v_rms_json"] = vj
        per[N], reasons[N] = mt, ""
    return {"per_n": per, "reasons": reasons}


def cell_verdicts(design, cell):
    chain = design.chain
    ev = {n: (m["e_v_q"] if m else NAN) for n, m in cell["per_n"].items()}
    fh = {n: (m["f_half"] if m else NAN) for n, m in cell["per_n"].items()}
    va, ra, order, rel = verdict_o3a(chain, ev)
    vb, rb = verdict_o3b(chain, fh)
    if va == "not_evaluable":
        missing = "; ".join((r if r.startswith("N=") else "N=%d: %s" % (n, r)) for n, r in cell["reasons"].items() if r)
        if missing:
            ra = ra + " [" + missing + "]"
    if vb == "not_evaluable":
        missing = "; ".join((r if r.startswith("N=") else "N=%d: %s" % (n, r)) for n, r in cell["reasons"].items() if r)
        if missing:
            rb = rb + " [" + missing + "]"
    return {"O3a": va, "O3a_reason": ra, "order_last": order, "relchange_last": rel, "O3b": vb, "O3b_reason": rb}


def main(argv=None):
    ap = argparse.ArgumentParser(description="SF-37 O3 (blind) - robust self-convergence metric of the gridded labels")
    ap.add_argument("root")
    ap.add_argument("outdir")
    ap.add_argument("--design", choices=("full", "discovered"), default="full")
    args = ap.parse_args(argv)
    argv_text = "o3.py %s %s --design %s" % (args.root, args.outdir, args.design)
    manifest = L.Manifest(args.root)
    exits = read_exit_codes(args.root, manifest)
    design = O3Design.full() if args.design == "full" else O3Design.discovered(args.root)
    W = L.OutWriter(args.outdir, manifest, "o3", argv_text)
    chain = design.chain

    cell_rows, ver_rows, c_rows, d7_rows = [], [], [], []
    md_cells = []
    for e in design.eps_codes:
        for s in design.seeds:
            cell = evaluate_cell(args.root, e, s, design, manifest, exits)
            ver = cell_verdicts(design, cell)
            for N in chain:
                m = cell["per_n"].get(N)
                r = cell["reasons"].get(N, "")
                if m is None:
                    cell_rows.append([e, L.sigma2_of(e), s, N, "not_evaluable", r] + [NAN] * 17)
                    continue
                cell_rows.append([e, L.sigma2_of(e), s, N, "ok", "", m["n_common"], m["n_marked"], m["rms_c_N"],
                                  m["v_rms_common_vD"], m["v_rms_json"],
                                  (m["v_rms_json"] / m["v_rms_common_vD"]) if m["v_rms_common_vD"] else NAN, m["e_v_q"],
                                  m["e_v_all"], m["f_half"], m["N_half"], m["frac_d_gt_0.1"], m["d_max"], m["d_max_loc"][0],
                                  m["d_max_loc"][1], m["d_max_loc"][2], m["rms_d_unmarked"], m["rms_d_all"]])
            # D7 growth factors and pair order of RMS d over the unmarked vertices
            nh = {n: (m["N_half"] if m else NAN) for n, m in cell["per_n"].items()}
            rd = {n: (m["rms_d_unmarked"] if m else NAN) for n, m in cell["per_n"].items()}
            for i in range(len(chain) - 1):
                a, b = chain[i], chain[i + 1]
                growth = (nh[b] / nh[a]) if (math.isfinite(nh.get(a, NAN)) and math.isfinite(nh.get(b, NAN)) and nh[a] > 0) else NAN
                d7_rows.append([e, s, a, b, nh.get(a, NAN), nh.get(b, NAN), growth, rd.get(a, NAN), rd.get(b, NAN),
                                L.order_pair(rd.get(a, NAN), rd.get(b, NAN))])
            ev = [cell["per_n"][n]["e_v_q"] if cell["per_n"].get(n) else NAN for n in chain]
            fh = [cell["per_n"][n]["f_half"] if cell["per_n"].get(n) else NAN for n in chain]
            ver_rows.append([e, L.sigma2_of(e), s, ver["O3a"], ver["O3a_reason"], ver["order_last"], ver["relchange_last"]] + ev
                            + [ver["O3b"], ver["O3b_reason"]] + fh)
            print("O3 eps=%g seed=%d: O3a=%s O3b=%s" % (L.eps_of(e), s, ver["O3a"], ver["O3b"]))
            # O3c
            for N in chain:
                m = cell["per_n"].get(N)
                run_dir = os.path.join(args.root, "v2", L.v2_name(e, s, N))
                if m is None:
                    for x1 in list(O3C_PLANES) + ["pooled"]:
                        c_rows.append([e, s, N, x1, "not_evaluable", NAN, 0, 0, cell["reasons"].get(N, "pair not evaluable")])
                    c_rows.append([e, s, N, 1.0, "structurally_empty", NAN, 0, 0, "outlet coarse plane: no marks by construction"])
                    continue
                run = L.RunData(run_dir, manifest) if os.path.isdir(run_dir) else None
                need = ("weights.npy", "planes_x23.npy", "track.json")
                if run is None or not all(run.exists(f) for f in need):
                    for x1 in list(O3C_PLANES) + ["pooled"]:
                        c_rows.append([e, s, N, x1, "not_evaluable", NAN, 0, 0, "v2 run missing or incomplete"])
                    c_rows.append([e, s, N, 1.0, "structurally_empty", NAN, 0, 0, "outlet coarse plane: no marks by construction"])
                    continue
                for r in o3c_fractions(run, m["marked"], N):
                    c_rows.append([e, s, N, r["x1"], r["status"], r["frac"], r["n"], r["n_hit"], r["reason"]])
                run.release()
            for N in chain:
                if cell["per_n"].get(N):
                    cell["per_n"][N]["marked"] = None
            md_cells.append((e, s, ver))
            del cell

    hdr_cells = ["eps_code", "sigma2", "seed", "N", "status", "reason", "n_common", "n_marked", "rms_c_N", "v_rms_common_vD",
                 "v_rms_vD_json", "ratio_json_over_common", "e_v_q", "e_v_all", "f_half", "N_half", "frac_d_gt_0.1", "d_max",
                 "d_max_x1", "d_max_x2", "d_max_x3", "rms_d_unmarked", "rms_d_all"]
    W.csv("o3_cells.csv", hdr_cells, cell_rows)
    hdr_ver = ["eps_code", "sigma2", "seed", "O3a", "O3a_reason", "order_last_pair", "relchange_last_pair"] + \
              ["e_v_q_N%d" % n for n in chain] + ["O3b", "O3b_reason"] + ["f_half_N%d" % n for n in chain]
    W.csv("o3_verdicts.csv", hdr_ver, ver_rows)
    W.csv("o3c.csv", ["eps_code", "seed", "N", "x1", "status", "weighted_fraction", "n_contributors", "n_near_marked", "note"], c_rows)
    W.csv("o3_d7.csv", ["eps_code", "seed", "N_coarse_pair", "N_fine_pair", "N_half_coarse_pair", "N_half_fine_pair",
                        "N_half_growth", "rms_d_unmarked_coarse_pair", "rms_d_unmarked_fine_pair", "pair_order_rms_d"], d7_rows)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)  # Linux: kB
    md = W.md_header("SF-37 O3 (blind): robust self-convergence metric of the gridded labels",
                     ["design: `%s` (eps codes %s, seeds %s, chain of fine grids %s; pairs (N/2, N))" %
                      (design.kind, list(design.eps_codes), list(design.seeds), list(chain)),
                      "q = 1e-3 (exact), d > 0.1 threshold, O3a order >= 1.8, reject if relative change < 0.10 and e > 1e-12",
                      "peak resident set size of this run (ru_maxrss): %.2f GB (estimate in the module docstring: ~1.5 GB at the 512 pair)" % peak_gb])
    md += ["", "## Verdicts (O3a, O3b) per amplitude and seed", "",
           L.md_table(["sigma2", "seed", "O3a", "order last pair", "rel change last pair"] + ["e_v^(q) N=%d" % n for n in chain] +
                      ["O3b"] + ["f_half N=%d" % n for n in chain],
                      [[r[1], r[2], r[3], r[5], r[6]] + r[7:7 + len(chain)] + [r[7 + len(chain)]] + r[9 + len(chain):] for r in ver_rows]),
           "", "Reasons:", ""]
    for r in ver_rows:
        md.append("- sigma2 %g seed %d: O3a %s (%s); O3b %s (%s)" % (r[1], r[2], r[3], r[4], r[7 + len(chain)], r[8 + len(chain)]))
    md += ["", "## O3c - fraction of the O6 weight landing within one coarse cell of a marked vertex", "",
           L.md_table(["eps_code", "seed", "N", "x1", "status", "weighted fraction", "contributors", "near marked", "note"], c_rows),
           "", "x1 = 1 is the outlet coarse plane: structurally empty (no marks by construction).",
           "", "## D7 - concentration of the defect (recorded, no thresholds)", "",
           "Reading note: " + READING_NOTE + ".", "",
           L.md_table(["eps_code", "seed", "pair N", "pair 2N", "N_half", "N_half(2N)", "growth", "RMS d unmarked", "RMS d unmarked (2N)",
                       "pair order"], d7_rows), "",
           "Per-cell values (n_common, n_marked, RMS|c_N|, v_rms of vD over the common vertices and vD.json, e_v^(q) and the unmasked "
           "e_v, f_half, N_half, fraction d > 0.1, d_max and its location in cube units, RMS d): `o3_cells.csv`."]
    W.md("o3_summary.md", "\n".join(md))
    nman, dig = W.write_manifest("input_manifest_o3.csv")
    with open(os.path.join(args.outdir, "o3_summary.md"), "a") as f:
        f.write("\n## Input manifest\n\n%d input files (path, sha256, bytes in `input_manifest_o3.csv`); sha256 of the manifest listing: `%s`.\n"
                % (nman, dig))
    print("o3: wrote %d files to %s" % (len(W.written), args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
