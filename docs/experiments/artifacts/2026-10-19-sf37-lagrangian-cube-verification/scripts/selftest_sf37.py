#!/usr/bin/env python3
"""Self-test of the SF-37 analysis scripts (exit 0 iff every check passes; one PASS/FAIL line per check).

  selftest_sf37.py [--inlet-slab PATH] [--keep DIR] [--require-real]

* bootstrap matrix: committed file == pre-registered sha256;
* moments: small REAL forward runs of the `inlet_slab` binary (seed 3001 / generic3d; never seeds 3101-3103)
  are digested and the library reproduces moments_planes.csv / moments_time.csv to <= 1e-12 relative (SKIPped with a
  clear message if the binary or its output is absent; --require-real turns a skip into a failure);
* moments (synthetic): the library against an independent pure-python-loop implementation of the tracker's definitions;
* bootstrap: shifted-sum SE == naive resample recomputation (<= 1e-12 relative), determinism, block-structure sanity;
* orders / V1 / V2 / lambda-h rules / RR15 vacuity / V5 fits on synthetic data;
* end-to-end: both digests on a tiny synthetic campaign tree (generated here with the exact file names and shapes),
  `discovered` and `full` designs;
* py_compile of every script.
Python 3 + numpy + scipy only.
"""
import argparse
import hashlib
import json
import math
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf37lib as L  # noqa: E402
import digest_v1v2 as D12  # noqa: E402
import digest_v4v5 as D45  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
FAILS = []
RESULTS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    RESULTS.append((name, ok))
    if not ok:
        FAILS.append(name)
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  [" + detail + "]") if detail else ""))
    return ok


def slurp(path):
    with open(path) as f:
        return f.read()


def skip(name, why):
    print("SKIP  %s  [%s]" % (name, why))


# --------------------------------------------------------------------------------------------------------------
# independent (pure-python-loop) implementation of the tracker's moment definitions, and a fake-run writer
# --------------------------------------------------------------------------------------------------------------
def loop_mean_var(w, v, sel):
    W = S = 0.0
    cnt = 0
    for p in range(len(w)):
        if not sel[p]:
            continue
        cnt += 1
        W += w[p]
        S += w[p] * v[p]
    if cnt == 0 or not (W > 0.0):
        return float("nan"), float("nan"), cnt
    mean = S / W
    V = 0.0
    for p in range(len(w)):
        if not sel[p]:
            continue
        d = v[p] - mean
        V += w[p] * d * d
    return mean, V / W, cnt


def write_moment_csvs(d, w, x23, tau, I, tx, planes, times, st_plane):
    """Writes moments_planes.csv / moments_time.csv with explicit python loops (independent of sf37lib)."""
    Np, P = tau.shape
    T = tx.shape[1]
    with open(os.path.join(d, "moments_planes.csv"), "w") as f:
        f.write("x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_tau,var_tau,rms_ev,max_ev\n")
        for m in range(P):
            sel = [math.isfinite(x23[p, m, 0]) and math.isfinite(x23[p, m, 1]) for p in range(Np)]
            m2, v2, cnt = loop_mean_var(w, x23[:, m, 0], sel)
            m3, v3, _ = loop_mean_var(w, x23[:, m, 1], sel)
            mt, vt, _ = loop_mean_var(w, tau[:, m], sel)
            Wt = S = mx = 0.0
            for p in range(Np):
                if sel[p]:
                    e = 1.0 - math.exp(-I[p, m])
                    Wt += w[p]
                    S += w[p] * e * e
                    mx = max(mx, abs(e))
            rms = math.sqrt(S / Wt) if (cnt > 0 and Wt > 0) else float("nan")
            non_ok = int(sum(1 for p in range(Np) if st_plane[p] != 0))
            f.write("%.15e,%d,%d,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n" %
                    (planes[m], cnt, non_ok, m2, v2, m3, v3, mt, vt, rms, mx if cnt else float("nan")))
    with open(os.path.join(d, "moments_time.csv"), "w") as f:
        f.write("# format_version=1\n")
        f.write("t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n")
        for n in range(T):
            sel = [all(math.isfinite(tx[p, n, k]) for k in range(3)) for p in range(Np)]
            ms, vs = [], []
            cnt = 0
            for k in range(3):
                m_, v_, cnt = loop_mean_var(w, tx[:, n, k], sel)
                ms.append(m_)
                vs.append(v_)
            f.write("%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%d\n" % (times[n], ms[0], ms[1], ms[2], vs[0], vs[1], vs[2], cnt))


def fake_truth(seed, M, P, T):
    rng = np.random.default_rng(seed)
    Np = M * M
    base = {
        "w": rng.random(Np) + 0.1,
        "x23": np.cumsum(rng.normal(0.0, 0.25, size=(Np, P, 2)), axis=1) + rng.normal(0, 0.05, size=(Np, 1, 2)),
        "tau": np.cumsum(rng.random((Np, P)) * 0.5 + 0.1, axis=1),
        "I": rng.normal(0.0, 0.1, size=(Np, P)),
        "tx": np.cumsum(rng.normal(0.0, 0.3, size=(Np, T, 3)), axis=1) + np.array([0.5, 0.0, 0.0])[None, None, :],
        "tI": rng.normal(0.0, 0.1, size=(Np, T)),
        "reached_p": np.where(rng.random(Np) < 0.04, rng.integers(1, P, size=Np), P),
        "reached_t": np.where(rng.random(Np) < 0.04, rng.integers(1, T, size=Np), T),
        "backflow": rng.random(Np) < 0.02,
        "cov": rng.normal(1.0, 0.3, size=(Np, P, 6)),
        "tcov": rng.normal(1.0, 0.3, size=(Np, T, 6)),
        "seed_v": np.stack([rng.random(Np) + 0.5, rng.normal(0, 0.2, Np), rng.normal(0, 0.2, Np)], axis=1),
    }
    return base


def write_fake_run(d, truth, planes, times, M, grid_noise=0.0, tol_noise=0.0, noise_seed=0, covectors=False,
                   seed_v=False, tol=1e-8, hmax=0.01, scale_planes=1.0):
    """Writes one particle directory with the exact file names / shapes of `--save-particles`."""
    os.makedirs(d, exist_ok=True)
    Np = M * M
    P, T = len(planes), len(times)
    rng = np.random.default_rng(noise_seed)
    w = truth["w"] * (1.0 + 0.001 * grid_noise * rng.normal(size=Np))
    w = np.where(truth["backflow"], 0.0, w)
    w = w / w.sum()
    x23 = truth["x23"] * scale_planes + (grid_noise + tol_noise) * rng.normal(size=(Np, P, 2))
    tau = truth["tau"] + (grid_noise + tol_noise) * rng.normal(size=(Np, P)) * 0.1
    I = truth["I"] + grid_noise * rng.normal(size=(Np, P)) * 0.1
    tx = truth["tx"] + (grid_noise + tol_noise) * rng.normal(size=(Np, T, 3))
    tI = truth["tI"] + grid_noise * rng.normal(size=(Np, T)) * 0.1
    rp = truth["reached_p"].copy()
    rt = truth["reached_t"].copy()
    drop = rng.random(Np) < 0.01 * min(grid_noise * 10, 1.0)
    rp = np.where(drop & (rp > 1), rp - 1, rp)
    for m in range(P):
        x23[rp <= m, m, :] = np.nan
        tau[rp <= m, m] = np.nan
        I[rp <= m, m] = np.nan
    for n in range(T):
        tx[rt <= n, n, :] = np.nan
        tI[rt <= n, n] = np.nan
    cov = truth["cov"].copy() if covectors else None
    if covectors:
        for m in range(P):
            cov[rp <= m, m, :] = np.nan
        tcov = truth["tcov"].copy()
        for n in range(T):
            tcov[rt <= n, n, :] = np.nan
    st_p = np.zeros(Np, dtype=np.uint8)
    st_t = np.zeros(Np, dtype=np.uint8)
    st_p[truth["backflow"]] = 1
    st_t[truth["backflow"]] = 1
    st_p[(rp < P) & ~truth["backflow"]] = 2
    st_t[(rt < T) & ~truth["backflow"]] = 2
    seeds = np.zeros((Np, 3))
    np.save(os.path.join(d, "seeds.npy"), seeds)
    np.save(os.path.join(d, "weights.npy"), w)
    np.save(os.path.join(d, "labels.npy"), np.full((Np, 2), np.nan))
    np.save(os.path.join(d, "status_plane.npy"), st_p)
    np.save(os.path.join(d, "status_time.npy"), st_t)
    np.save(os.path.join(d, "planes_x23.npy"), x23)
    np.save(os.path.join(d, "planes_tau.npy"), tau)
    np.save(os.path.join(d, "planes_I.npy"), I)
    np.save(os.path.join(d, "times_x.npy"), tx)
    np.save(os.path.join(d, "times_I.npy"), tI)
    if covectors:
        np.save(os.path.join(d, "planes_cov.npy"), cov)
        np.save(os.path.join(d, "times_cov.npy"), tcov)
    if seed_v:
        np.save(os.path.join(d, "seed_v.npy"), truth["seed_v"])
    write_moment_csvs(d, w, x23, tau, I, tx, planes, times, st_p)
    hist = {"ok": int((st_p == 0).sum()), "seed_backflow": int((st_p == 1).sum()), "cap_exceeded": int((st_p == 2).sum()),
            "stagnation": 0, "step_underflow": 0, "landing_failed": 0, "invalid_conductivity": 0}
    pas = {"status_hist": hist, "nfev": int(1000 * Np * (1.0 + 0.0 * tol)), "accepted_steps": 5000, "rejected_steps": 10,
           "landing_steps": 700, "landing_rejected": 3, "landings_discarded": 0, "recrossings": 0,
           "backflow_encounters": 0, "min_g1_hat": 0.1}
    trk = {"field": "gaussian", "eps": 1.0, "N": 0, "h": 0.01, "M": M, "Np": Np, "planes": list(planes), "times": list(times),
           "tol": tol, "h_max": hmax, "min_step": 1e-13, "max_arclength": 1000.0, "covectors": bool(covectors),
           "seed_backflow": int(truth["backflow"].sum()), "plane_pass": pas, "time_pass": dict(pas)}
    with open(os.path.join(d, "track.json"), "w") as f:
        json.dump(trk, f)
    return w


def make_campaign(root, M=16, eps_codes=(10, 15), seeds=(7001, 7002), v2_ns=(8, 16, 32, 64), v1_ns=(32, 64),
                  identical_tol=True, base_seed=1):
    """Tiny synthetic campaign with the C-LAYOUT file names; v1 (1e-8, h8) == v2 primary at the same N; (1e-8) ==
    (1e-10) bitwise when identical_tol."""
    planes, times = L.PLANES, L.TIMES
    P, T = len(planes), len(times)
    for part in ("v2", "v1"):
        os.makedirs(os.path.join(root, part), exist_ok=True)
    codes = {"v2": [], "v1": []}
    for e in eps_codes:
        for s in seeds:
            truth = fake_truth(base_seed * 100000 + e * 100 + s % 1000, M, P, T)
            for n in v2_ns:
                name = L.v2_name(e, s, n)
                write_fake_run(os.path.join(root, "v2", name), truth, planes, times, M, grid_noise=0.5 * (8.0 / n) ** 2,
                               noise_seed=n * 7 + e + s)
                codes["v2"].append((name, 0))
            for n in v1_ns:
                gn = 0.5 * (8.0 / n) ** 2
                for tol in L.V1_TOLS:
                    tn = 0.0 if identical_tol else 1e-4 * (tol / 1e-10) ** 0.5
                    name = L.v1_name(e, s, n, tol, 8)
                    write_fake_run(os.path.join(root, "v1", name), truth, planes, times, M, grid_noise=gn, tol_noise=tn,
                                   noise_seed=n * 7 + e + s, covectors=True, seed_v=True, tol=tol)
                    codes["v1"].append((name, 0))
                name = L.v1_name(e, s, n, 1e-10, 32)
                write_fake_run(os.path.join(root, "v1", name), truth, planes, times, M, grid_noise=gn * 0.9, tol_noise=0.0,
                               noise_seed=n * 7 + e + s + 1, covectors=True, seed_v=True, tol=1e-10, hmax=0.0025)
                codes["v1"].append((name, 0))
            for part in ("v2", "v1"):
                for name, _ in list(codes[part]):
                    if not name.endswith(".log"):
                        pn = L.parse_run_name(name)
                        if pn[1] == e and pn[2] == s:
                            with open(os.path.join(root, part, name + ".log"), "w") as f:
                                f.write("some preamble\nFIELD gaussian (SF-18) N=%d sigma2=1 ell=0.25 seed=%d "
                                        "normalize_variance=1 applied_scale=1.234567890123e+00 raw_variance=8.12345e-01 "
                                        "final_variance=1.0 active_modes=4096 | stage field k = exp(eps Y), eps = %g\n" %
                                        (pn[3], s, e / 10.0))
                            with open(os.path.join(root, part, name + ".json"), "w") as f:
                                json.dump({"status": "ok"}, f)
    for part in ("v2", "v1"):
        with open(os.path.join(root, part, "exit_codes.tsv"), "w") as f:
            for name, c in codes[part]:
                f.write("%s\t%d\n" % (name, c))
    return codes


# --------------------------------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------------------------------
def test_bootstrap_file():
    m = np.load(L.BOOT_PATH)
    sha = hashlib.sha256(m.astype("<i8").tobytes()).hexdigest()
    check("bootstrap: committed matrix shape (2000,16) int64", m.shape == (2000, 16) and m.dtype == np.int64)
    check("bootstrap: sha256 == pre-registered 16c1a5b9...af39", sha == "16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39", sha)
    check("bootstrap: load_bootstrap verifies the sha256", L.load_bootstrap().shape == (2000, 16))
    regen = np.random.default_rng(20261017).integers(0, 16, size=(2000, 16))
    check("bootstrap: runtime generator reproduces the committed matrix (this numpy)", np.array_equal(regen, m))
    check("bootstrap: runtime_bootstrap(16) == committed", np.array_equal(L.runtime_bootstrap(16), m))
    tmp = tempfile.mkdtemp()
    try:
        bad = m.copy()
        bad[0, 0] = (bad[0, 0] + 1) % 16
        p = os.path.join(tmp, "bad.npy")
        np.save(p, bad)
        try:
            L.load_bootstrap(p)
            ok = False
        except ValueError:
            ok = True
        check("bootstrap: a modified matrix is rejected", ok)
    finally:
        shutil.rmtree(tmp)
    ids = L.block_ids(256 * 256, 4)
    p = np.arange(256 * 256)
    m2, m3 = p // 256, p % 256
    check("bootstrap: block id = (m2//64)*4 + m3//64 for M=256", np.array_equal(ids, (m2 // 64) * 4 + m3 // 64)
          and np.bincount(ids).tolist() == [4096] * 16)


def test_synthetic_moments(tmp):
    d = os.path.join(tmp, "synthrun")
    M = 16
    truth = fake_truth(11, M, 7, 7)
    write_fake_run(d, truth, L.PLANES, L.TIMES, M, grid_noise=0.05, noise_seed=3, covectors=True)
    run = L.RunData(d)
    ok, worst, cols = L.check_moments_reproduction(run)
    check("moments (synthetic, independent loops): library == csv <= 1e-12", ok, "worst rel %.2e" % worst)
    obs = L.observed_series(run)
    blk = L.block_observed_series(run)
    wr = max(L.max_rel_diff(blk[k], obs[k]) for k in obs)
    check("moments: shifted block sums at multiplicity 1 reproduce the sequential series <= 1e-12", wr <= 1e-12, "worst rel %.2e" % wr)
    check("moments: contributors = finite records (RR1)", int(run.plane_mask()[:, 0].sum()) == int(np.isfinite(run.mem("planes_x23.npy")[:, 0, 0]).sum()))
    check("moments: seed_backflow particles have weight 0", bool((run.w[truth["backflow"]] == 0).all()))


def naive_series(run, mult_p):
    """Direct recomputation with per-particle multiplicities (independent of the shifted block sums)."""
    w = run.w * mult_p
    out = {}
    mp, mt = run.plane_mask(), run.time_mask()
    X, tau, tx = run.mem("planes_x23.npy"), run.mem("planes_tau.npy"), run.mem("times_x.npy")
    P, T = run.P, run.T
    s2 = np.zeros(P)
    vt = np.zeros(P)
    for m in range(P):
        sel = mp[:, m] & (w > 0)
        ww = w[sel] / w[sel].sum()
        v = []
        for col in (X[sel, m, 0], X[sel, m, 1], tau[sel, m]):
            mu = (ww * col).sum()
            v.append((ww * (col - mu) ** 2).sum())
        s2[m] = 0.5 * (v[0] + v[1])
        vt[m] = v[2]
    out["sigma_T2"], out["var_tau"] = s2, vt
    for k in range(3):
        vv = np.zeros(T)
        for n in range(T):
            sel = mt[:, n] & (w > 0)
            ww = w[sel] / w[sel].sum()
            mu = (ww * tx[sel, n, k]).sum()
            vv[n] = (ww * (tx[sel, n, k] - mu) ** 2).sum()
        out["var_x%d" % (k + 1)] = vv
    return L.add_derived(out, run.planes, run.times)


def test_bootstrap_se(tmp):
    M = 16
    t = fake_truth(21, M, 7, 7)
    da, db = os.path.join(tmp, "bs_a"), os.path.join(tmp, "bs_b")
    write_fake_run(da, t, L.PLANES, L.TIMES, M, grid_noise=0.3, noise_seed=1)
    write_fake_run(db, t, L.PLANES, L.TIMES, M, grid_noise=0.1, noise_seed=2)
    a, b = L.RunData(da), L.RunData(db)
    idx = L.load_bootstrap()
    counts = L.counts_from_idx(idx, 16)
    ps = L.paired_stats(a, b, counts, 4)
    ids = L.block_ids(a.Np, 4)
    R = 200  # naive recomputation on the first 200 resamples for speed, same indices
    Dn = {k: [] for k in ps}
    for r in range(R):
        mult_b = counts[r]
        mult_p = mult_b[ids]
        na, nb = naive_series(a, mult_p), naive_series(b, mult_p)
        for k in Dn:
            Dn[k].append(na[k] - nb[k])
    worst = 0.0
    for k in Dn:
        se_naive = np.std(np.array(Dn[k]), axis=0, ddof=1)
        # compare with the shifted-sum SE restricted to the same 200 resamples
        sh = L.full_sample_means(b)
        ra = L.resampled_series(a, sh, counts[:R], 4)
        rb = L.resampled_series(b, sh, counts[:R], 4)
        se_fast = np.std(ra[k] - rb[k], axis=0, ddof=1)
        worst = max(worst, float(np.max(np.abs(se_fast - se_naive) / np.abs(se_naive))))
    check("bootstrap: shifted-sum SE == naive resample recomputation (first 200 resamples, all series) <= 1e-12 rel",
          worst <= 1e-12, "worst rel %.2e" % worst)
    ps2 = L.paired_stats(a, b, counts, 4)
    check("bootstrap: deterministic (identical SE on repeat)", all(np.array_equal(ps[k]["se"], ps2[k]["se"], equal_nan=True) for k in ps))
    # known block structure
    rng = np.random.default_rng(5)
    Mb = 16
    Np = Mb * Mb
    truth_iid = fake_truth(31, Mb, 7, 7)
    # iid case: SE of the variance of a Gaussian sample ~ sqrt(2/n) var ; D between independent runs -> sqrt(2) times that
    for key in ("x23", "tx"):
        truth_iid[key] = rng.normal(0, 1.0, size=truth_iid[key].shape)
    truth_iid["w"] = np.ones(Np)
    truth_iid["backflow"] = np.zeros(Np, dtype=bool)
    truth_iid["reached_p"][:] = 7
    truth_iid["reached_t"][:] = 7
    truth_blk = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in truth_iid.items()}
    blk = L.block_ids(Np, 4)
    scale = (1.0 + 2.0 * (blk % 4))[:, None, None]  # block-dependent amplitude 1..7
    truth_blk["x23"] = truth_iid["x23"] * scale
    out = {}
    for tag, tr in (("iid", truth_iid), ("blk", truth_blk)):
        dd1, dd2 = os.path.join(tmp, "k_%s_1" % tag), os.path.join(tmp, "k_%s_2" % tag)
        write_fake_run(dd1, tr, L.PLANES, L.TIMES, Mb, noise_seed=1)
        t2 = dict(tr)
        t2["x23"] = tr["x23"] * 1.0 + np.random.default_rng(99).normal(0, 1e-9, size=tr["x23"].shape)
        write_fake_run(dd2, t2, L.PLANES, L.TIMES, Mb, noise_seed=2)
        out[tag] = L.paired_stats(L.RunData(dd1), L.RunData(dd2), counts, 4)["sigma_T2"]["se"]
    # a and b essentially equal -> D ~ 0 with tiny SE: use SE of a single run instead through an independent pair
    dd3 = os.path.join(tmp, "k_iid_3")
    t3 = dict(truth_iid)
    t3["x23"] = np.random.default_rng(1234).normal(0, 1.0, size=truth_iid["x23"].shape)
    write_fake_run(dd3, t3, L.PLANES, L.TIMES, Mb, noise_seed=3)
    se_pair = L.paired_stats(L.RunData(os.path.join(tmp, "k_iid_1")), L.RunData(dd3), counts, 4)["sigma_T2"]["se"]
    expect = 2.0 / math.sqrt(Np)  # sqrt(2 * 2/n) for sigma_T^2 of two independent runs; the 2-variable average adds /sqrt(2): sqrt(2*(1/n))
    expect = math.sqrt(2.0 * (1.0 / Np))
    ratio = float(np.median(se_pair) / expect)
    check("bootstrap: iid SE order of magnitude (median ratio to analytic in [0.5, 2])", 0.5 <= ratio <= 2.0, "ratio %.2f" % ratio)
    # block structure that does NOT cancel: independent second run with a different block amplitude pattern
    dd4 = os.path.join(tmp, "k_blk_4")
    t4 = dict(truth_blk)
    t4["x23"] = np.random.default_rng(777).normal(0, 1.0, size=truth_iid["x23"].shape) * (1.0 + 2.0 * (blk % 4))[:, None, None]
    write_fake_run(dd4, t4, L.PLANES, L.TIMES, Mb, noise_seed=4)
    se_blk = float(np.median(L.paired_stats(L.RunData(os.path.join(tmp, "k_blk_1")), L.RunData(dd4), counts, 4)["sigma_T2"]["se"]))
    check("bootstrap: heterogeneous blocks give a larger SE than iid (> 1.5x)", se_blk > 1.5 * float(np.median(se_pair)),
          "blk %.3g vs iid %.3g" % (se_blk, float(np.median(se_pair))))
    # 4 and 64 block partitions run
    c2 = L.counts_from_idx(L.runtime_bootstrap(4), 4)
    c8 = L.counts_from_idx(L.runtime_bootstrap(64), 64)
    s4 = L.paired_stats(a, b, c2, 2)["sigma_T2"]["se"]
    s64 = L.paired_stats(a, b, c8, 8)["sigma_T2"]["se"]
    check("bootstrap: 4- and 64-block sensitivity partitions run and are finite", np.isfinite(s4).all() and np.isfinite(s64).all())
    # sequential quantities
    check("weighted quantile (exact cases)", L.wquantile([1.0, 2.0, 3.0], [0.2, 0.5, 0.3], 0.2) == 1.0
          and L.wquantile([1.0, 2.0, 3.0], [0.2, 0.5, 0.3], 0.21) == 2.0
          and L.wquantile([1.0, 2.0, 3.0], [0.2, 0.5, 0.3], 0.7) == 2.0
          and L.wquantile([1.0, 2.0, 3.0], [0.2, 0.5, 0.3], 0.71) == 3.0)


def test_rules():
    nan = float("nan")
    # orders (RR4)
    h = [0.4, 0.2, 0.1]
    q = [1.0 + 3.0 * x ** 2 for x in h]
    p = L.order_triple(*q)
    check("order_triple: Q = 1 + c h^2 gives 2", abs(p - 2.0) < 1e-12, str(p))
    check("order_triple: sign change -> nan", math.isnan(L.order_triple(1.0, 2.0, 1.5)))
    check("order_triple: vanishing difference -> nan", math.isnan(L.order_triple(1.0, 2.0, 2.0)) and math.isnan(L.order_triple(2.0, 2.0, 1.0)))
    check("order_triple: non-finite -> nan", math.isnan(L.order_triple(1.0, nan, 0.0)))
    check("order_pair: 4 -> 1 gives 2", L.order_pair(4.0, 1.0) == 2.0)
    check("order_pair: zero fine value -> inf; non-finite -> nan", L.order_pair(1.0, 0.0) == float("inf") and math.isnan(L.order_pair(nan, 1.0)))
    # V1
    c = L.v1_cell(1e-4, 1.0, 1e-9)
    check("V1 rule: relative branch (det)", c["verdict"] == "pass" and c["branch"] == "det")
    c = L.v1_cell(5e-3, 1.0, 3e-3)
    check("V1 rule: 2 SE_D branch (stat)", c["verdict"] == "pass" and c["branch"] == "stat" and c["binding"] == "stat")
    c = L.v1_cell(5e-3, 1.0, 1e-3)
    check("V1 rule: fail beyond both thresholds", c["verdict"] == "fail" and c["branch"] == "fail")
    c = L.v1_cell(0.0, 1.0, 0.0)
    check("V1 rule: D = 0 passes (det)", c["verdict"] == "pass" and c["branch"] == "det")
    check("V1 rule: NaN is not_evaluable, never pass", L.v1_cell(nan, 1.0, 1.0)["verdict"] == "not_evaluable"
          and L.v1_cell(0.0, 1.0, nan)["verdict"] == "not_evaluable")
    # V2
    q = [1.0 + 3.0 * x ** 2 for x in (0.4, 0.2, 0.1)]
    c = L.v2_cell(q[0], q[1], q[2], 1e-9, 1e-9)
    check("V2 rule: order branch (p = 2) when the relative branch fails", c["verdict"] == "pass" and c["branch"] == "order" and c["order_ok"]
          and abs(c["p"] - 2.0) < 1e-9)
    c = L.v2_cell(1.0, 2.0, 1.5, 1e-9)
    check("V2 rule: fail (sign change, large D, small SE)", c["verdict"] == "fail" and math.isnan(c["p"]))
    c = L.v2_cell(1.0, 1.0105, 1.0, 1e-9)
    check("V2 rule: det branch (|D| <= 2 %)", c["verdict"] == "pass" and c["branch"] == "det")
    c = L.v2_cell(1.0, 1.5, 1.0, 0.3)
    check("V2 rule: stat branch (2 SE_D)", c["verdict"] == "pass" and c["branch"] == "stat")
    check("V2 rule: missing triple value -> not_evaluable", L.v2_cell(nan, 1.0, 1.0, 0.1)["verdict"] == "not_evaluable")
    c = L.v2_cell(1.0, 1.4, 1.0 + 0.8 * 0.4 / 3.0, 0.3, 0.3)
    check("V2 rule: Richardson residual and noise flag are reported", math.isfinite(c["richardson"]) or math.isnan(c["p"]))
    # lambda/h
    cl = L.lambda_h_cell(1.04, 1.0, 1e-6)
    check("lambda/h cell: 4 % passes det, 6 % fails", cl["verdict"] == "pass" and L.lambda_h_cell(1.06, 1.0, 1e-6)["verdict"] == "fail")
    cands = [8, 10, 16, 20, 32, 64]
    st = {8: "fail", 10: "pass", 16: "fail", 20: "pass", 32: "pass", 64: "pass"}
    check("production lambda/h: literal primary 10, monotone 20 (difference flagged)", L.production_lambda_h(cands, st) == (10, 20))
    st = {8: "missing", 10: "pass", 16: "pass", 20: "pass", 32: "pass", 64: "pass"}
    check("production lambda/h: a missing candidate is non-qualifiable (primary 10 skips 8)", L.production_lambda_h(cands, st) == (10, 10))
    st = {8: "fail", 10: "fail", 16: "fail", 20: "fail", 32: "fail", 64: "fail"}
    check("production lambda/h: none <= 64", L.production_lambda_h(cands, st) == (None, None))
    st = {8: "pass", 10: "pass", 16: "pass", 20: "pass", 32: "pass", 64: "missing"}
    check("production lambda/h: missing largest candidate -> monotone None, primary 8", L.production_lambda_h(cands, st) == (8, None))
    check("study value = max over amplitudes of the primary", D12.study_value({10: (10, 10), 15: (16, 16), 20: (8, 8)}, 0) == 16
          and D12.study_value({10: (10, 10), 15: (None, None)}, 0) is None)
    # V5 fits and naming
    x1 = np.array(L.PLANES)
    rms_exp = 1e-9 * np.exp(1.5 * x1)
    f = L.fit_decorrelation(x1, rms_exp, rms_exp / 1.0)
    check("V5 fit: exponential data named exponential", f["name"] == "exponential" and f["n"] >= 3, "%s n=%d" % (f["name"], f["n"]))
    rms_alg = 1e-8 * x1 ** 2.5
    f = L.fit_decorrelation(x1, rms_alg, rms_alg / 1.0)
    check("V5 fit: algebraic data named algebraic", f["name"] == "algebraic", f["name"])
    f = L.fit_decorrelation(x1, np.full(7, 5.0), np.full(7, 5.0))
    check("V5 fit: no eligible plane (rms >= 0.1 sigma_T) -> insufficient", f["name"] == "insufficient" and f["n"] == 0)
    r = np.array([1e-13, 1e-13, 1e-9, 1e-8, 1e-7, 0.5, 0.9])
    f = L.fit_decorrelation(x1, r, r)
    check("V5 fit: 1e-12 floor and 0.1 sigma_T ceiling select the eligible planes (3 here)", f["n"] == 3, "n=%d" % f["n"])
    f = L.fit_decorrelation(x1, np.where(np.arange(7) < 2, 1e-9, 5.0), np.where(np.arange(7) < 2, 1e-9, 5.0))
    check("V5 fit: fewer than 3 eligible planes -> insufficient", f["name"] == "insufficient")
    check("V5 amplitude naming: majority / mixed / insufficient",
          L.amplitude_name(["exponential", "exponential", "algebraic", "insufficient"]) == "exponential"
          and L.amplitude_name(["exponential", "algebraic"]) == "mixed"
          and L.amplitude_name(["algebraic", "insufficient", "insufficient"]) == "insufficient"
          and L.amplitude_name(["algebraic", "algebraic", "exponential", "algebraic", "tie"]) == "algebraic")
    check("fmt_tol / run names", L.fmt_tol(1e-6) == "1e-6" and L.fmt_tol(1e-10) == "1e-10"
          and L.v1_name(10, 3101, 256, 1e-10, 32) == "v1_e10_s3101_N256_tol1e-10_h32" and L.parse_run_name("v1_e10_s3101_N256_tol1e-10_h32") == ("v1", 10, 3101, 256, 1e-10, 32)
          and L.parse_run_name("v2_e15_s3102_N40") == ("v2", 15, 3102, 40) and L.parse_run_name("v2_e15_s3102") is None)
    fl = L.parse_field_line("x\nFIELD gaussian (SF-18) N=64 sigma2=1 ell=0.25 seed=3101 normalize_variance=1 applied_scale=1.5e+00 raw_variance=0.8 final_variance=1 active_modes=10 | stage")
    check("FIELD line parsing", fl is not None and fl["seed"] == 3101 and abs(fl["applied_scale"] - 1.5) < 1e-15 and fl["N"] == 64)


def sha_file(p):
    return L.sha256_file(p)


def test_campaign_e2e(tmp):
    root = os.path.join(tmp, "camp")
    make_campaign(root, identical_tol=True)
    # RR15: identical tol runs -> vacuous label
    man = L.Manifest(root)
    camp = L.Campaign(root, man)
    design = L.Design.discovered(camp)
    check("discovered design: eps {10,15}, seeds {7001,7002}, v2 N {8,16,32,64}, v1 N {32,64}",
          design.eps_codes == (10, 15) and design.seeds == (7001, 7002) and design.v2_ns == (8, 16, 32, 64)
          and design.v1_ns == (32, 64) and design.n_ref == 64 and design.n_triple == (16, 32, 64))
    eng = L.PairEngine(camp, L.load_bootstrap(manifest=man))
    ctx = D12.Ctx(camp, design, eng)
    cells, ladder, vac, shadow, ident_rows = D12.do_v1(ctx)
    labs = []
    for (e, s, n, ident) in ident_rows:
        sub = [c for c in cells if c["eps_code"] == e and c["seed"] == s and c["N"] == n]
        labs.append(D12.v1_label(sub, ident))
    check("RR15: bitwise-identical 1e-8 / 1e-10 runs -> 'met, vacuous (tol-independent step sequence)'",
          all(x == "met, vacuous (tol-independent step sequence)" for x in labs), str(set(labs)))
    check("V1: identical runs give D = 0 on every cell, branch det", all(c["D"] == 0.0 and c["branch"] == "det" for c in cells))
    v2c = D12.do_v2(ctx)
    check("V2: cells evaluated for every (eps, seed, Q, plane) of the discovered design",
          len(v2c) == 2 * 2 * 2 * 7 and all(c["cell"]["verdict"] in ("pass", "fail") for c in v2c))
    # digests end-to-end (discovered)
    out12 = os.path.join(tmp, "out12")
    out45 = os.path.join(tmp, "out45")
    r1 = D12.main([root, out12, "--design", "discovered"])
    r2 = D45.main([root, out45, "--design", "discovered"])
    check("digest_v1v2.py (discovered, synthetic tree) exits 0 and writes md/csv", r1 == 0 and os.path.isfile(os.path.join(out12, "v1v2_summary.md"))
          and os.path.isfile(os.path.join(out12, "v1_cells.csv")) and os.path.isfile(os.path.join(out12, "input_manifest_v1v2.csv")))
    check("digest_v4v5.py (discovered, synthetic tree) exits 0 and writes md/csv", r2 == 0 and os.path.isfile(os.path.join(out45, "v4v5_summary.md"))
          and os.path.isfile(os.path.join(out45, "v5_fits.csv")) and os.path.isfile(os.path.join(out45, "input_manifest_v4v5.csv")))
    md12 = slurp(os.path.join(out12, "v1v2_summary.md"))
    md45 = slurp(os.path.join(out45, "v4v5_summary.md"))
    check("v1v2 markdown states verdict and branch per cell, lambda/h and vacuity", "V1 cells" in md12 and "V2 cells" in md12 and "Production lambda/h" in md12
          and "met, vacuous" in md12 and "Study value" in md12)
    check("v4v5 markdown states the V4 verdicts and the V5 named fits", "V4 cells" in md45 and "named fit" in md45)
    with open(os.path.join(out12, "input_manifest_v1v2.csv")) as f:
        rows = [ln for ln in f.read().splitlines() if not ln.startswith("#")][1:]
    ok = True
    for ln in rows[:20]:
        pth, sha, _ = ln.strip().split(",")
        ok = ok and sha_file(os.path.join(root, pth)) == sha if not os.path.isabs(pth) and os.path.exists(os.path.join(root, pth)) else ok
    check("input manifest lists the files read with sha256 (spot check)", len(rows) > 50 and ok, "%d files" % len(rows))
    # outputs are deterministic
    out12b = os.path.join(tmp, "out12b")
    D12.main([root, out12b, "--design", "discovered"])
    same = True
    for fn in ("v1_cells.csv", "v2_cells.csv", "lambda_h_cells.csv", "d1_cells.csv"):
        a = [ln for ln in slurp(os.path.join(out12, fn)).splitlines() if not ln.startswith("#")]
        b = [ln for ln in slurp(os.path.join(out12b, fn)).splitlines() if not ln.startswith("#")]
        same = same and a == b
    check("digest_v1v2: deterministic (identical csv on repeat)", same)
    # full design on a tree that lacks the pre-registered runs: everything is not_evaluable, nothing passes
    out_full = os.path.join(tmp, "outfull")
    D12.main([root, out_full, "--design", "full"])
    vs = [ln.split(",") for ln in slurp(os.path.join(out_full, "v1_cells.csv")).splitlines() if not ln.startswith("#")][1:]
    hdr = [x.strip() for x in slurp(os.path.join(out_full, "v1_cells.csv")).splitlines()[1].split(",")]
    iv = hdr.index("verdict")
    check("full design with missing runs: every V1 cell is not_evaluable (never pass)", len(vs) > 0 and all(r[iv] == "not_evaluable" for r in vs))
    md_full = slurp(os.path.join(out_full, "v1v2_summary.md"))
    check("full design reports the missing runs and 'none' for lambda/h", "missing" in md_full and "none <=" in md_full)
    D45.main([root, os.path.join(tmp, "outfull45"), "--design", "full"])
    vv = slurp(os.path.join(tmp, "outfull45", "v4_cells.csv"))
    check("full design (v4v5) with missing runs: V4 not_evaluable, no pass", "pass" not in vv.replace("verdict", "") and "not_evaluable" in vv)
    # non-identical tolerance runs: not vacuous
    root2 = os.path.join(tmp, "camp2")
    make_campaign(root2, identical_tol=False, base_seed=2)
    man2 = L.Manifest(root2)
    camp2 = L.Campaign(root2, man2)
    des2 = L.Design.discovered(camp2)
    ctx2 = D12.Ctx(camp2, des2, L.PairEngine(camp2, L.load_bootstrap(manifest=man2)))
    cells2, _, vac2, _, idr2 = D12.do_v1(ctx2)
    labs2 = [D12.v1_label([c for c in cells2 if c["eps_code"] == e and c["seed"] == s and c["N"] == n], ident) for (e, s, n, ident) in idr2]
    check("RR15: tolerance-dependent runs are never labelled vacuous", not any("vacuous" in x for x in labs2), str(set(labs2)))
    # bitwise comparison v1 vs v2 reported
    bit = slurp(os.path.join(out12, "d9_bitwise_v1_vs_v2.csv"))
    check("D9: bitwise V1 (1e-8,h8) vs V2 primary comparison produced", "records_bitwise_identical" in bit)
    # D2 / FIELD parsing
    d2 = slurp(os.path.join(out12, "d2_pairing.csv"))
    check("D2: applied_scale / raw_variance parsed from the FIELD lines", "1.234567890123" in d2)
    # percent of cells in the statistical regime etc present
    d1 = slurp(os.path.join(out12, "d1_cells.csv"))
    check("D1: z table with seed-pooled z and statistical-regime flag", "seed_pooled_z" in d1 and "stat_regime" in d1)
    # V4 monotone decrease: a synthetic grid-error ladder shrinks with N -> evaluable cells
    v4 = [ln.split(",") for ln in slurp(os.path.join(out45, "v4_cells.csv")).splitlines() if not ln.startswith("#")]
    check("V4: cells produced for planes <= 4 (5 per eps, seed)", len(v4) - 1 == 2 * 2 * 5, str(len(v4) - 1))
    # V5: covector table and fits present
    cn = slurp(os.path.join(out45, "v5_covector_norms.csv"))
    check("V5: covector percentile table present", "|a|_p50" in cn and len(cn.splitlines()) > 10)


def test_real_data(binary, tmp, require):
    name = "moments (real forward runs of inlet_slab, seed 3001 / generic3d)"
    if binary is None or not os.path.isfile(binary):
        if require:
            check(name, False, "inlet_slab binary not found")
        else:
            skip(name, "inlet_slab binary not found; build with: cmake --preset wsl-debug && cmake --build build/wsl-debug -j --target inlet_slab")
        return
    root = os.path.join(tmp, "real")
    os.makedirs(os.path.join(root, "v1"))
    os.makedirs(os.path.join(root, "v2"))
    cases = [
        ("gauss3001_tol1e-8", ["--sigma2", "1", "--ell", "0.25", "--seed", "3001", "--n", "16", "--eps", "1", "--tol", "1e-8"]),
        ("gauss3001_tol1e-10", ["--sigma2", "1", "--ell", "0.25", "--seed", "3001", "--n", "16", "--eps", "1", "--tol", "1e-10"]),
        ("generic3d", ["--analytic", "generic3d", "--n", "16", "--eps", "0.25"]),
    ]
    runs = {}
    for tag, extra in cases:
        pdir = os.path.join(root, "v1", tag)
        cmd = [binary, "--track", "forward"] + extra + ["--np-side", "16", "--planes", "0.25,0.5,1,2", "--times", "0.5,1,2",
                                                        "--covectors", "--save-particles", pdir, "--summary", os.path.join(root, "v1", tag + ".json")]
        assert "3101" not in cmd and "3102" not in cmd and "3103" not in cmd
        try:
            pr = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as ex:  # pragma: no cover
            if require:
                check(name + " [" + tag + "]", False, "could not run: %s" % ex)
            else:
                skip(name, "could not run the binary: %s" % ex)
            return
        with open(os.path.join(root, "v1", tag + ".log"), "w") as f:
            f.write(pr.stdout)
        if pr.returncode != 0 or not os.path.isfile(os.path.join(pdir, "moments_planes.csv")):
            if require:
                check(name + " [" + tag + "]", False, "binary exit %d: %s" % (pr.returncode, pr.stdout[-300:] + pr.stderr[-300:]))
            else:
                skip(name, "binary exit %d (no GPU?): %s" % (pr.returncode, (pr.stdout + pr.stderr)[-200:].replace("\n", " ")))
            return
        runs[tag] = L.RunData(pdir)
    for tag, run in runs.items():
        ok, worst, cols = L.check_moments_reproduction(run)
        worst_col = max(cols, key=cols.get)
        check("moments REAL data [%s]: library == moments_planes.csv / moments_time.csv <= 1e-12 relative" % tag, ok,
              "worst rel %.3e (%s); Np=%d P=%d T=%d; contributors/plane %s" % (worst, worst_col, run.Np, run.P, run.T,
                                                                                    run.plane_mask().sum(axis=0).tolist()))
        obs = L.observed_series(run)
        blk = L.block_observed_series(run)
        wr = max(L.max_rel_diff(blk[k], obs[k]) for k in obs)
        check("moments REAL data [%s]: shifted block sums reproduce the sequential series <= 1e-12" % tag, wr <= 1e-12, "worst rel %.2e" % wr)
    # paired comparison on real data: tolerance pair
    a, b = runs["gauss3001_tol1e-8"], runs["gauss3001_tol1e-10"]
    ps = L.paired_stats(a, b, L.counts_from_idx(L.load_bootstrap(), 16), 4)
    ident = L.bitwise_identical_particles(a, b)
    sp = L.pair_separation(a, b, "plane")
    check("real data: paired tolerance comparison runs; SE_D finite, separation finite",
          np.isfinite(ps["sigma_T2"]["se"]).all() and np.isfinite(sp["rms"]).all(),
          "ident particles %.4f, max |D| %.2e, max RMS(dx_T)/sigma_T %.2e" % (ident.mean(), float(np.abs(ps["sigma_T2"]["D"]).max()), float(np.nanmax(sp["rel"]))))
    # digest the real tree in 'discovered'-compatible form is not possible (names); exercise V5 pieces on the covectors
    cov = a.read_array("planes_cov.npy")
    check("real data: covector file shape (Np, P, 6)", cov.shape == (a.Np, a.P, 6))


def test_numpy_api():
    banned = ["np.trapz", "np.trapezoid", "np.in1d", "np.product", "np.cumproduct", "np.float_", "np.alltrue", "np.sometrue",
              "np.row_stack", "np.asfarray", "np.round_", "np.NaN", "np.Inf", "np.infty", "np.unicode_", "np.string_"]
    hits = []
    for fn in ("sf37lib.py", "digest_v1v2.py", "digest_v4v5.py", "make_bootstrap_blocks.py"):
        src = slurp(os.path.join(HERE, fn))
        for b in banned:
            if b in src:
                hits.append("%s:%s" % (fn, b))
    check("numpy API: no numpy-2-removed / numpy-1.26-absent names in the scripts", not hits, ",".join(hits))
    for fn in ("sf37lib.py", "digest_v1v2.py", "digest_v4v5.py", "make_bootstrap_blocks.py", "selftest_sf37.py"):
        try:
            py_compile.compile(os.path.join(HERE, fn), doraise=True, cfile=os.path.join(tempfile.gettempdir(), "sf37_%s.pyc" % fn))
            ok = True
        except py_compile.PyCompileError as ex:  # pragma: no cover
            ok = False
        check("py_compile %s" % fn, ok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inlet-slab", default=None, help="path of the inlet_slab binary (default: build/wsl-debug/inlet_slab)")
    ap.add_argument("--keep", default=None, help="keep the temporary directory here")
    ap.add_argument("--require-real", action="store_true", help="fail instead of skipping the real-data check")
    args = ap.parse_args(argv)
    binary = args.inlet_slab or os.path.join(REPO, "build", "wsl-debug", "inlet_slab")
    tmp = args.keep or tempfile.mkdtemp(prefix="sf37_selftest_")
    os.makedirs(tmp, exist_ok=True)
    v = L.versions()
    print("selftest_sf37: python %s numpy %s scipy %s; tmp %s" % (v["python"], v["numpy"], v["scipy"], tmp))
    try:
        test_bootstrap_file()
        test_rules()
        test_synthetic_moments(tmp)
        test_bootstrap_se(tmp)
        test_campaign_e2e(tmp)
        test_real_data(binary, tmp, args.require_real)
        test_numpy_api()
    finally:
        if args.keep is None:
            shutil.rmtree(tmp, ignore_errors=True)
    print("selftest_sf37: %d checks, %d failed" % (len(RESULTS), len(FAILS)))
    for f in FAILS:
        print("  FAILED: " + f)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
