#!/usr/bin/env python3
"""SF-41 threshold F, Lagrangian e_v ladder on medium (b) (pre-registration RR8): verdict from run_sf41.sh part `evladder`.

  f_verdict.py DIR [--eps 1,2] [--seeds 6101,6102,6103] [--grids 32,64,128,256] [--x0 2.5] [--lambda 0.5] [--nplanes 14]
               [--bootstrap PATH] [--dry]
  f_verdict.py --selftest

DIR holds, for every (eps, seed, ny) a run `F_e<eps>_s<seed>_g<ny>` (output tree DIR/<run>/ with realizations/r_0000/, files
<run>.log, exit_codes.tsv).  Defaults: the RR8 design (eps 1, 2; seeds 6101-6103; ny 32, 64, 128, 256 = lambda/h 16..128 at
ell = 0.5; planes X0 + m lambda, m = 1..14, X0 = 2.5), i.e. 24 runs.

F per (eps, seed, plane): e_v = the tracker's flux-weighted RMS e_v of the plane (`rms_ev` of moments_planes.csv, 15 digits) at the four
rungs; MET iff all four are finite, e_1 > e_2 > e_3 > e_4 (strictly decreasing on every refinement) and the last-pair order
log2(e_3 / e_4) >= 1.8.  A missing run, non-zero exit, missing or non-finite value, or a plane list different from the pre-registered
one is NOT MET.  Per amplitude: MET iff every cell is MET.  Output `F_CELL`, `F_VERDICT eps=.. cells_met=k/n MET|NOT_MET`,
`F_VERDICT ALL MET|NOT_MET`.

Recorded only (RR8; never blocking): D_P = sigma_T^2(Pollock) - sigma_T^2(consumer), sigma_T^2 = (var x2 + var x3)/2, each with its
own flux weights over its own contributors (finite plane records), recomputed from the per-particle files and checked against the
program's moments CSVs (<= 1e-9 relative, else the cell is not evaluable); SE_D by the paired block bootstrap over 4 blocks of the
M x M seed lattice (block of particle p = m2 M + m3: (m2 // (M/2)) * 2 + (m3 // (M/2)); 50 x 50 for M = 100) with the COMMITTED index
matrix analysis/bootstrap_blocks_f.npy = default_rng(20261025).integers(0, 4, size=(2000, 4)) (sha256 verified on load), each resample
recomputing both variances over the resampled multiset (weights x multiplicity), SE_D = sample standard deviation (ddof = 1) of the
2000 paired differences.  V3b classification per (eps, seed, plane): `concordant` iff |D_P| decreases on every refinement and
|D_P(finest)| <= max(0.05 sigma_T^2(consumer, finest), 2 SE_D(finest)); else `discordant`; missing data `not_evaluable`.
Exit 0 iff F ALL MET.  `--dry`: relaxes nothing numerically; it only allows other parameter sets (see the options).
"""
import csv
import hashlib
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf41lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BOOT_PATH = os.path.normpath(os.path.join(HERE, "..", "analysis", "bootstrap_blocks_f.npy"))
BOOT_SHA256 = "bf45a8cb390d691b7c83a8db72964919d498dd719f98d1e652c8327179e8d0fd"
ORDER_GATE = 1.8
NAN = float("nan")


def load_bootstrap(path=BOOT_PATH, sha=BOOT_SHA256):
    m = np.load(path, allow_pickle=False)
    if m.shape != (2000, 4):
        raise ValueError("bootstrap matrix shape %s != (2000, 4)" % (m.shape,))
    got = hashlib.sha256(m.astype("<i8").tobytes()).hexdigest()
    if got != sha:
        raise ValueError("bootstrap matrix sha256 %s != pre-registered %s" % (got, sha))
    return m.astype(np.int64)


def counts_from_idx(idx, nb=4):
    idx = np.asarray(idx)
    return (idx[:, :, None] == np.arange(nb)[None, None, :]).sum(axis=1).astype(np.float64)


def read_planes_csv(path, col):
    """-> (x1 array, column array) of a moments CSV (header line, comma separated), or None."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    try:
        return np.array([float(r["x1"]) for r in rows]), np.array([float(r[col]) for r in rows]), rows
    except (KeyError, ValueError):
        return None


def cell_orders(e):
    """e: sequence of rung values (coarse -> fine) -> (met, decreasing, last_order, orders)"""
    e = [float(x) for x in e]
    if any((not math.isfinite(x)) or x <= 0.0 for x in e):
        return False, False, NAN, []
    dec = all(e[i + 1] < e[i] for i in range(len(e) - 1))
    orders = [math.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
    last = orders[-1]
    return (dec and last >= ORDER_GATE), dec, last, orders


# ----------------------------------------------------------------------------------------------- D_P
def block_ids(Np):
    M = int(round(math.sqrt(Np)))
    if M * M != Np or M % 2 != 0:
        raise ValueError("Np = %d is not the square of an even number" % Np)
    side = M // 2
    p = np.arange(Np)
    return (p // M // side) * 2 + (p % M) // side


def _blocks(w, x2, x3, mask, ids, shift2, shift3):
    wm = np.where(mask, w, 0.0)
    y2 = np.where(mask, x2 - shift2, 0.0)
    y3 = np.where(mask, x3 - shift3, 0.0)
    W = np.bincount(ids, weights=wm, minlength=4)
    A2 = np.bincount(ids, weights=wm * y2, minlength=4)
    B2 = np.bincount(ids, weights=wm * y2 * y2, minlength=4)
    A3 = np.bincount(ids, weights=wm * y3, minlength=4)
    B3 = np.bincount(ids, weights=wm * y3 * y3, minlength=4)
    return W, A2, B2, A3, B3


def _sigma_T2(W, A2, B2, A3, B3):
    with np.errstate(divide="ignore", invalid="ignore"):
        v2 = B2 / W - (A2 / W) ** 2
        v3 = B3 / W - (A3 / W) ** 2
        return np.where(W > 0, 0.5 * (v2 + v3), np.nan)


def sigma_T2_plane(w, x23, mask):
    """Observed flux-weighted sigma_T^2 over the contributors (sequential two-pass, the program's formula)."""
    ww = w[mask]
    W = ww.sum()
    if not (W > 0):
        return NAN, NAN, NAN
    out = []
    for k in (0, 1):
        x = x23[mask, k]
        m = (ww * x).sum() / W
        out.append((ww * (x - m) ** 2).sum() / W)
    return 0.5 * (out[0] + out[1]), out[0], out[1]


def method_arrays(rdir, pollock):
    """-> (w, x23 (Np,P,2)) of the consumer (tracker) or of the Pollock comparator in a realization directory."""
    if pollock:
        pd = os.path.join(rdir, "pollock")
        return np.load(os.path.join(pd, "pollock_weights.npy")), np.load(os.path.join(pd, "pollock_planes_x23.npy"))
    return np.load(os.path.join(rdir, "weights.npy")), np.load(os.path.join(rdir, "planes_x23.npy"))


def reach_mask(x23):
    return np.isfinite(x23[:, :, 0]) & np.isfinite(x23[:, :, 1])


def dp_for_run(rdir, counts):
    """-> dict with arrays over planes: sc (consumer sigma_T^2), sp (Pollock), D, SE, csv_err, or {'error': text}."""
    try:
        wc, xc = method_arrays(rdir, False)
        wp, xp = method_arrays(rdir, True)
    except (OSError, ValueError) as e:
        return {"error": "unreadable:%s" % e}
    if xc.shape != xp.shape or wc.shape != wp.shape:
        return {"error": "shape_mismatch"}
    Np, P = xc.shape[0], xc.shape[1]
    ids = block_ids(Np)
    mc, mp = reach_mask(xc), reach_mask(xp)
    csvc = read_planes_csv(os.path.join(rdir, "moments_planes.csv"), "var_x2")
    csvp = read_planes_csv(os.path.join(rdir, "pollock", "pollock_moments_planes.csv"), "var_x2")
    sc = np.full(P, NAN)
    sp = np.full(P, NAN)
    D = np.full(P, NAN)
    SE = np.full(P, NAN)
    csv_err = 0.0
    for m in range(P):
        sc[m], v2c, v3c = sigma_T2_plane(wc, xc[:, m, :], mc[:, m])
        sp[m], v2p, v3p = sigma_T2_plane(wp, xp[:, m, :], mp[:, m])
        for tab, v2, v3 in ((csvc, v2c, v3c), (csvp, v2p, v3p)):
            if tab is None:
                csv_err = NAN
                continue
            rows = tab[2]
            ref2, ref3 = float(rows[m]["var_x2"]), float(rows[m]["var_x3"])
            for a, b in ((v2, ref2), (v3, ref3)):
                if math.isfinite(a) and math.isfinite(b):
                    csv_err = max(csv_err, abs(a - b) / max(abs(b), 1e-300)) if math.isfinite(csv_err) else csv_err
                elif math.isfinite(a) != math.isfinite(b):
                    csv_err = NAN
        D[m] = sp[m] - sc[m]
        # paired block bootstrap: shifts = the full-sample weighted means of each method
        res = []
        for (w, x, mk) in ((wc, xc[:, m, :], mc[:, m]), (wp, xp[:, m, :], mp[:, m])):
            ww = np.where(mk, w, 0.0)
            W = ww.sum()
            if not (W > 0):
                res.append(None)
                continue
            s2 = (ww * np.where(mk, x[:, 0], 0.0)).sum() / W
            s3 = (ww * np.where(mk, x[:, 1], 0.0)).sum() / W
            Wb, A2, B2, A3, B3 = _blocks(w, x[:, 0], x[:, 1], mk, ids, s2, s3)
            Ws = counts @ Wb
            As2, Bs2, As3, Bs3 = counts @ A2, counts @ B2, counts @ A3, counts @ B3
            res.append(_sigma_T2(Ws, As2, Bs2, As3, Bs3))
        if res[0] is not None and res[1] is not None:
            SE[m] = np.std(res[1] - res[0], ddof=1)
    return {"sc": sc, "sp": sp, "D": D, "SE": SE, "csv_err": csv_err, "P": P}


def v3b_class(D, sc_fine, SE_fine):
    """D: |D_P| per rung (coarse -> fine) -> concordant | discordant | not_evaluable"""
    vals = list(D) + [sc_fine, SE_fine]
    if any(not math.isfinite(x) for x in vals):
        return "not_evaluable"
    a = [abs(x) for x in D]
    dec = all(a[i + 1] < a[i] for i in range(len(a) - 1))
    small = a[-1] <= max(0.05 * sc_fine, 2.0 * SE_fine)
    return "concordant" if (dec and small) else "discordant"


# ----------------------------------------------------------------------------------------------- report
def run_name(eps, seed, ny):
    return "F_e%s_s%s_g%s" % (eps, seed, ny)


def report(d, eps_l=("1", "2"), seeds=("6101", "6102", "6103"), grids=(32, 64, 128, 256), x0=2.5, lam=0.5, nplanes=14,
           boot=BOOT_PATH, dry=False):
    out = []
    counts = counts_from_idx(load_bootstrap(boot)) if os.path.exists(boot) else None
    if counts is None:
        out.append("RECORDED F_BOOTSTRAP missing %s (D_P standard errors not evaluable)" % boot)
    want = [x0 + (m + 1) * lam for m in range(nplanes)]
    allmet = True
    for eps in eps_l:
        met = tot = runs_ok = 0
        conc = disc = nev = 0
        for seed in seeds:
            e_by_rung, planes_ok, dps = [], True, []
            rung_ok = True
            for ny in grids:
                run = run_name(eps, seed, ny)
                rdir = os.path.join(d, run, "realizations", "r_0000")
                rc = L.rc_of(d, run)
                tab = read_planes_csv(os.path.join(rdir, "moments_planes.csv"), "rms_ev")
                if rc != 0 or tab is None:
                    out.append("F_RUN run=%s exit=%s moments=%s NOT_MET" % (run, rc, "present" if tab is not None else "missing"))
                    e_by_rung.append(None)
                    rung_ok = False
                    continue
                runs_ok += 1
                x1, ev, _ = tab
                if len(x1) != nplanes or any(abs(a - b) > 1e-12 for a, b in zip(x1, want)):
                    out.append("F_RUN run=%s planes=%s != pre-registered %s NOT_MET" % (run, list(x1)[:3], want[:3]))
                    e_by_rung.append(None)
                    rung_ok = False
                    continue
                e_by_rung.append(ev)
                if counts is not None:
                    dps.append(dp_for_run(rdir, counts))
                else:
                    dps.append({"error": "no_bootstrap"})
            for m in range(nplanes):
                tot += 1
                if not rung_ok or any(e is None for e in e_by_rung):
                    out.append("F_CELL eps=%s seed=%s plane=%d x1=%.10g NOT_MET (missing run or planes)" % (eps, seed, m, want[m]))
                    continue
                ev = [float(e[m]) for e in e_by_rung]
                ok, dec, last, orders = cell_orders(ev)
                met += int(ok)
                out.append("F_CELL eps=%s seed=%s plane=%d x1=%.10g e=%s orders=%s decreasing=%d last_order=%.4f %s" % (
                    eps, seed, m, want[m], ",".join("%.6e" % x for x in ev), ",".join("%.4f" % o for o in orders), int(dec),
                    last, "MET" if ok else "NOT_MET"))
            # recorded D_P / V3b
            if rung_ok and len(dps) == len(grids) and all("error" not in x for x in dps):
                for m in range(nplanes):
                    D = [x["D"][m] for x in dps]
                    cls = v3b_class(D, dps[-1]["sc"][m], dps[-1]["SE"][m])
                    csv_bad = any((not math.isfinite(x["csv_err"])) or x["csv_err"] > 1e-9 for x in dps)
                    if csv_bad:
                        cls = "not_evaluable"
                    conc += cls == "concordant"
                    disc += cls == "discordant"
                    nev += cls == "not_evaluable"
                    out.append("RECORDED F_DP eps=%s seed=%s plane=%d x1=%.10g rungs=%s D_P=%s SE_D=%s sigma_T2_consumer=%s "
                               "sigma_T2_pollock=%s csv_max_rel=%s V3b=%s" % (
                                   eps, seed, m, want[m], ",".join(str(g) for g in grids),
                                   ",".join("%.6e" % x for x in D), ",".join("%.6e" % x["SE"][m] for x in dps),
                                   ",".join("%.6e" % x["sc"][m] for x in dps), ",".join("%.6e" % x["sp"][m] for x in dps),
                                   ",".join("%.2e" % x["csv_err"] for x in dps), cls))
            else:
                nev += nplanes
                out.append("RECORDED F_DP eps=%s seed=%s not_evaluable (%s)" % (eps, seed, ",".join(
                    x["error"] for x in dps if "error" in x) or "missing runs"))
        ok_eps = met == tot and tot == len(seeds) * nplanes and runs_ok == len(seeds) * len(grids)
        allmet = allmet and ok_eps
        out.append("F_VERDICT eps=%s cells_met=%d/%d runs_ok=%d/%d %s" % (eps, met, tot, runs_ok, len(seeds) * len(grids),
                                                                           "MET" if ok_eps else "NOT_MET"))
        out.append("RECORDED F_V3B_SUMMARY eps=%s concordant=%d discordant=%d not_evaluable=%d (of %d cells; non-blocking)" % (
            eps, conc, disc, nev, len(seeds) * nplanes))
    out.append("F_VERDICT ALL %s" % ("MET" if allmet else "NOT_MET"))
    return out, allmet


# ---------------------------------------------------------------------------------------------------------------
def _write_npy_run(d, run, ev, rng, Np=100, bias=0.0, P=4, x0=2.5, lam=0.5, rc=0, make_pollock=True):
    """Synthetic realization: planes x0 + m lam, consumer / Pollock per-particle arrays and the moments CSVs."""
    M = int(round(math.sqrt(Np)))
    rdir = os.path.join(d, run, "realizations", "r_0000")
    os.makedirs(os.path.join(rdir, "pollock"), exist_ok=True)
    w = rng.uniform(0.5, 1.5, Np)
    w /= w.sum()
    x23 = rng.normal(0, 0.1, (Np, P, 2))
    x23[:5, P - 1, :] = np.nan  # five particles never reach the last plane
    xp = x23 + rng.normal(0, 0.01 * (1 + bias), x23.shape)
    np.save(os.path.join(rdir, "weights.npy"), w)
    np.save(os.path.join(rdir, "planes_x23.npy"), x23)
    np.save(os.path.join(rdir, "pollock", "pollock_weights.npy"), w)
    np.save(os.path.join(rdir, "pollock", "pollock_planes_x23.npy"), xp)

    def write_csv(path, X, extra=None):
        with open(path, "w") as f:
            f.write("x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_tau,var_tau,rms_ev,max_ev\n")
            for m in range(P):
                mk = np.isfinite(X[:, m, 0])
                W = w[mk].sum()
                v = []
                for k in (0, 1):
                    mu = (w[mk] * X[mk, m, k]).sum() / W
                    v.append((w[mk] * (X[mk, m, k] - mu) ** 2).sum() / W)
                f.write("%.15e,%d,0,0,%.15e,0,%.15e,0,0,%.15e,0\n" % (x0 + (m + 1) * lam, int(mk.sum()), v[0], v[1],
                                                                       ev[m] if extra is None else ev[m]))
    write_csv(os.path.join(rdir, "moments_planes.csv"), x23)
    with open(os.path.join(rdir, "pollock", "pollock_moments_planes.csv"), "w") as f:
        f.write("x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_t,var_t\n")
        for m in range(P):
            mk = np.isfinite(xp[:, m, 0])
            W = w[mk].sum()
            v = []
            for k in (0, 1):
                mu = (w[mk] * xp[mk, m, k]).sum() / W
                v.append((w[mk] * (xp[mk, m, k] - mu) ** 2).sum() / W)
            f.write("%.15e,%d,0,0,%.15e,0,%.15e,0,0\n" % (x0 + (m + 1) * lam, int(mk.sum()), v[0], v[1]))
    with open(os.path.join(d, "exit_codes.tsv"), "a") as f:
        f.write("%s\t%d\n" % (run, rc))


def selftest():
    import tempfile
    fails = []

    def expect(n, c):
        if not c:
            fails.append(n)
            print("SELFTEST FAIL %s" % n)

    expect("orders_met", cell_orders([1.0, 0.25, 0.0625, 0.015625])[0])
    expect("orders_last_pair_slow", not cell_orders([1.0, 0.25, 0.0625, 0.05])[0])
    expect("orders_not_decreasing", not cell_orders([1.0, 0.25, 0.30, 0.01])[0])
    expect("orders_equal_not_strict", not cell_orders([1.0, 0.25, 0.25, 0.0625])[0])
    expect("orders_nan", not cell_orders([1.0, NAN, 0.1, 0.01])[0])
    expect("orders_gate_boundary", cell_orders([1.0, 0.5, 0.25, 0.25 / 2 ** 1.81])[0] and not cell_orders([1.0, 0.5, 0.25, 0.25 / 2 ** 1.79])[0])
    expect("v3b_concordant", v3b_class([0.1, 0.05, 0.02, 0.01], 1.0, 0.001) == "concordant")
    expect("v3b_not_decreasing", v3b_class([0.1, 0.05, 0.06, 0.01], 1.0, 0.001) == "discordant")
    expect("v3b_final_too_big", v3b_class([0.1, 0.05, 0.02, 0.01], 0.1, 0.001) == "discordant")
    expect("v3b_noise_allows", v3b_class([0.1, 0.05, 0.02, 0.01], 0.1, 0.01) == "concordant")
    expect("v3b_nan", v3b_class([0.1, NAN, 0.02, 0.01], 1.0, 0.1) == "not_evaluable")
    ids = block_ids(10000)
    expect("blocks_50x50", ids[0] == 0 and ids[49] == 0 and ids[50] == 1 and ids[5000] == 2 and ids[9999] == 3
           and np.bincount(ids).tolist() == [2500] * 4)
    m = load_bootstrap()
    expect("bootstrap_loaded", m.shape == (2000, 4) and m.min() >= 0 and m.max() <= 3)
    rng0 = np.random.default_rng(20261025)
    expect("bootstrap_reproducible", np.array_equal(m, rng0.integers(0, 4, size=(2000, 4))))
    try:
        load_bootstrap(sha="0" * 64)
        expect("bootstrap_sha_checked", False)
    except ValueError:
        pass
    # full synthetic design: 1 amplitude, 2 seeds, 4 rungs, 4 planes
    grids = (32, 64, 128, 256)
    P = 4

    def build(d, evs=None, rc_bad=None):
        rng = np.random.default_rng(7)
        for seed in ("6101", "6102"):
            for k, ny in enumerate(grids):
                ev = [1e-2 * 0.25 ** k * (1 + 0.1 * mm) for mm in range(P)] if evs is None else evs(seed, k)
                run = run_name("1", seed, ny)
                _write_npy_run(d, run, ev, rng, bias=1.0 / (1 + k), rc=(1 if rc_bad == run else 0))

    kw = dict(eps_l=("1",), seeds=("6101", "6102"), grids=grids, nplanes=P)
    with tempfile.TemporaryDirectory() as d:
        build(d)
        out, ok = report(d, **kw)
        expect("report_met", ok and "F_VERDICT eps=1 cells_met=8/8 runs_ok=8/8 MET" in out and out[-1] == "F_VERDICT ALL MET")
        expect("report_dp_recorded", any(x.startswith("RECORDED F_DP eps=1 seed=6101 plane=0") and "csv_max_rel=" in x for x in out)
               and any(x.startswith("RECORDED F_V3B_SUMMARY eps=1") for x in out))
        csv_ok = [float(x.split("csv_max_rel=")[1].split()[0].split(",")[0]) for x in out if "csv_max_rel=" in x]
        expect("dp_consistent_with_program_csv", csv_ok and max(csv_ok) < 1e-12)
        # D_P concordance: the Pollock difference shrinks with the rungs by construction (bias 1/(1+k))
        expect("v3b_has_classes", any("V3b=concordant" in x or "V3b=discordant" in x for x in out))
    with tempfile.TemporaryDirectory() as d:  # one plane's last pair too slow
        build(d, evs=lambda s, k: [1e-2 * 0.25 ** k] * 3 + [(1e-2 * 0.25 ** k if k < 3 else 1e-2 * 0.25 ** 2 * 0.9)])
        out, ok = report(d, **kw)
        expect("report_notmet_order", not ok and any("F_VERDICT eps=1 cells_met=6/8" in x for x in out))
    with tempfile.TemporaryDirectory() as d:
        build(d, rc_bad=run_name("1", "6102", 128))
        out, ok = report(d, **kw)
        expect("report_missing_run", not ok)
    with tempfile.TemporaryDirectory() as d:  # a plane that is not X0 + m lambda
        build(d)
        out, ok = report(d, x0=2.6, **kw)
        expect("report_wrong_planes", not ok)
    with tempfile.TemporaryDirectory() as d:  # absent runs
        out, ok = report(d, **kw)
        expect("report_absent", not ok)
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK f_verdict")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    if not a or a[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    opt = {"--eps": "1,2", "--seeds": "6101,6102,6103", "--grids": "32,64,128,256", "--x0": "2.5", "--lambda": "0.5",
           "--nplanes": "14", "--bootstrap": BOOT_PATH}
    i = 1
    while i < len(a):
        if a[i] in opt and i + 1 < len(a):
            opt[a[i]] = a[i + 1]
            i += 2
        elif a[i] == "--dry":
            i += 1
        else:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
    o, ok = report(a[0], tuple(opt["--eps"].split(",")), tuple(opt["--seeds"].split(",")),
                   tuple(int(x) for x in opt["--grids"].split(",")), float(opt["--x0"]), float(opt["--lambda"]),
                   int(opt["--nplanes"]), opt["--bootstrap"])
    print("\n".join(o))
    sys.exit(0 if ok else 1)
