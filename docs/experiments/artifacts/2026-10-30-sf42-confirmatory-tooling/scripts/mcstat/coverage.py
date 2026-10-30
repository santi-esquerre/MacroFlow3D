"""Coverage simulator of the SF-42 confirmatory tooling (spec item 8, acceptance F; decision G5.3).

It qualifies, per procedure, distribution and sample size, the one-sided miss rate at every tail level SF-43 will use:
misses, replicates, rate, the 95 % one-sided Clopper-Pearson upper limit (UCL), the ratio to the nominal tail, and

    PASS  iff  UCL <= 1.25 x tail           (G5.3: judged at the 95 % Monte Carlo upper limit)

with at least ``ceil(175 / tail)`` replicates per cell (so a procedure exactly at nominal passes with UCL/tail about 1.13).
The procedures, tails, arm structure and bootstrap sizes are in ``covproc`` (docstring), the distributions (normal and the stress
family at the upper bootstrap bounds of SF-39's skewness and kurtosis) in ``covdist``.  BCa cells are reported, never judged.

Design (restart-safe, reproducible, memory-bounded, parallel)
-------------------------------------------------------------
* A *plan* (``plan.json``) lists the cells; a cell is (procedure, scenario, distribution, n) with its tails, replicate count ``R``,
  batch size ``m`` and chunk size.  The plan is written once and reused on restart (a different plan in the same directory is an
  error).
* A *batch* is ``m`` replicates run in one vectorized call; its random stream is
  ``default_rng(SeedSequence([seed, sha256(cell id)[:8], batch index]))``, so the numbers depend on neither the number of workers, nor
  the chunking, nor the order or the interruptions of the run.  A *chunk* is ``chunk`` consecutive replicates (a multiple of ``m``,
  sized to about a minute of work): the unit of scheduling and restart.  Each finished chunk is written atomically to
  ``chunks/<cell>/<index>.json``; a rerun skips the existing chunks.  Memory per worker is one batch (``m x B`` floats per bootstrap
  array, ``B`` a few 10^4, ``m`` a few tens at most).
* Workers: ``multiprocessing`` with the ``spawn`` start method (identical on every platform, and a fork with live BLAS threads warns
  on recent Pythons); set ``OMP_NUM_THREADS=1`` (``run_coverage.sh`` does).
* Shared bootstrap draws.  A bootstrap-heavy procedure draws one resampling-index matrix per *batch* of ``m`` replicates
  (``share_draws=True`` of ``intervals`` / ``budget``) instead of one per replicate (the draw is most of the cost).  Every
  replicate's limit is still valid, but replicates of a batch are dependent through the shared draws, which inflates the variance of
  a miss count above the binomial one.  The sum and sum of squares of the per-batch counts are stored; the summary estimates the design effect
  ``D = Var(batch count) / (m p (1 - p))`` per tail (at least 1) and judges ``UCL`` on the effective sample (``k / D`` misses out of
  ``R / D``).  Measured (one-off runs of the N6b report, unpaired and paired contrasts, ``n`` 20-30, ``B`` 300-500, ``m`` 100-300,
  3000 batches): ``D = 1.00 +- 0.03`` against the upper bound ``1 + (m - 1) / B`` = 1.2-2.0 that a fully common bootstrap quantile
  would give: different replicates look at the shared resamples through different data directions.  The production cells use
  ``m`` = 10 and ``B`` of 10^4, where even the bound is 1.001.  Procedures without bootstrap and the exact-independence cells
  (``m`` = 1 draw per replicate, the bootstrap-t cells) have ``D = 1``.
* Default full study (``--profile full``; 573 cells, 84.5 million replicates, 17 thousand chunks, every cell with ``R tail >= 175``
  for its smallest tail; each distribution is ``normal``, ``stress`` (skewness 1.936, excess kurtosis 4.711: the 95 % bootstrap upper
  bounds of the sigma^2 = 4 series, which dominate those of 0.25, 1 and 2.25 in both moments) and its reflection ``stress-``):

  ==============================  ====================================================================  ===========  =================
  procedure (label)               what is simulated                                                     n            R per cell
  ==============================  ====================================================================  ===========  =================
  sampling:student / :boott       lower and upper limit of a mean, 8 tails 3.57e-3 .. 0.05              20-1650 (8)  49 000
  tost:gate5                      Gate 5 TOST at true ratio 1.10 and 1/1.10, rho 0, 0.5, 0.9, cv 0.3    20-170 (5)   25 000
  bca:mean                        BCa 95 % / 90 % two-sided interval, REPORTED ONLY                     20-100 (4)   7 000
  contrast:paired/unpaired/       UCB of |Delta| at Delta = 5 standard errors (E_int, E_W, 2 x 2 E_WxH)  20-200 (5)   196 000
  interaction                     at the 13 component tails 8.9e-4 .. 8.3e-3
  ladder:E_h                      E_h bound of the production bias: true order 2, 1.5, 1, non-monotone;  20-200 (5)   196 000
                                  production level middle/finest; bias 12, 25, 50 level-mean errors
  pipeline:K3 / :K4               RL(a) > 0 at the 7 Holm levels, K = 3 (N 100, 300, 1000) and K = 4     N 30-1000    24 500
                                  (N 30, 100, 300); biases 0 and +2+2+2 standard errors, order 1 and 2
  ==============================  ====================================================================  ===========  =================

  Cost, measured as CPU seconds per replicate of every (procedure, n, distribution) group on the development workstation
  (``python3 -m mcstat.coverage bench``, one core, numpy 2.5 / scipy 1.18): 114 core-hours, of which the ladder cells 99, the
  bootstrap-t cells 5.8, the pipeline 4.6, the contrast bounds 4.2; projected 2.4 wall-hours on 80 workers at 90 % parallel
  efficiency and a host core 1.5 times slower (3.2 h at 2 times).  Peak memory per worker 0.4 GiB (the K = 3, N = 1000 pipeline).
  Not covered (reduce or extend with ``--ladder-ns``, ``--dists``, ``--only``): ladder sequences of 4 or more levels, ladder ratios
  other than 2, width x resolution interactions other than the simulated 2 x 2, per-amplitude stress members (use
  ``--dists stress_s1,...``), bias magnitudes other than 0 and 2 standard errors.
* Commands (``python3 -m mcstat.coverage ...``): ``stress`` (recompute ``stress_moments.json`` from the SF-39 tables), ``plan``,
  ``bench`` (measure the cost per replicate of every cell and project the host time), ``run``, ``summarize``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import re
import sys
import time

import numpy as np
from scipy import stats as _st

from . import covdist, covproc

VERSION = 1
TARGET = 175  # replicates x tail (G5.3)
CRITERION = 1.25  # allowed ratio UCL / tail
CONF = 0.95  # one-sided Clopper-Pearson level of the UCL
DEFAULT_SEED = 20261030


# ----------------------------------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------------------------------
def cp_upper(k, R, conf=CONF):
    """One-sided Clopper-Pearson upper limit of a binomial proportion: ``Beta(conf; k + 1, R - k)`` (real ``k``, ``R`` allowed:
    the effective sample of a clustered estimate).  ``k = 0`` gives ``1 - (1 - conf)^(1 / R)``; ``k >= R`` gives 1."""
    if R <= 0 or k >= R:
        return 1.0
    return float(_st.beta.ppf(conf, k + 1.0, R - k))


def design_effect_from_moments(total, sumsq, nbatch, m):
    """Per-tail variance inflation of a miss count from replicates that share bootstrap draws in batches of ``m``.

    ``total`` / ``sumsq``: sum and sum of squares over the ``nbatch`` batches of the per-batch miss counts.  ``D = Var_b(count) /
    (m p (1 - p))``, ``p`` the pooled rate; 1 where ``p`` is 0 or 1, with fewer than 2 batches, or below 1 (an under-dispersed
    estimate is not used to shrink the interval).
    """
    total, sumsq = np.asarray(total, dtype=float), np.asarray(sumsq, dtype=float)
    if nbatch < 2 or m < 2:
        return np.ones(total.shape)
    p = total / (nbatch * m)
    var = (sumsq - total * total / nbatch) / (nbatch - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = var / (m * p * (1.0 - p))
    return np.maximum(np.where((p > 0.0) & (p < 1.0) & np.isfinite(d), d, 1.0), 1.0)


def design_effect(batch_counts, m):
    """``design_effect_from_moments`` of an array (nbatch, T) of per-batch miss counts."""
    c = np.asarray(batch_counts, dtype=float)
    return design_effect_from_moments(c.sum(axis=0), (c * c).sum(axis=0), c.shape[0], m)


# ----------------------------------------------------------------------------------------------------------------------
# Plan
# ----------------------------------------------------------------------------------------------------------------------
def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.+=-]", "_", s)


def make_cell(proc, label, scen, dist, n, params, target=TARGET, m=1, chunk=None, share=False, report_only=False, min_reps=0):
    """One cell of the plan: ``R = ceil(target / smallest tail)`` replicates (at least ``min_reps``), as a multiple of ``m``."""
    params = dict(params, n=n, share=bool(share))
    tails = covproc.PROCS[proc].tails(params)
    R = max(int(math.ceil(target / float(tails.min()) * (1.0 - 1e-12))), int(min_reps), 1)
    R = int(math.ceil(R / m) * m)
    chunk = R if chunk is None else int(min(R, math.ceil(chunk / m) * m))
    return {"id": _sanitize("%s__%s__%s__n%d" % (label, scen or "base", dist, n)), "proc": proc, "label": label, "scen": scen or "",
            "dist": dist, "n": int(n), "params": params, "tails": [float(t) for t in tails], "R": R, "m": int(m), "chunk": chunk,
            "report_only": bool(report_only)}


def plan_hash(cells, seed):
    return hashlib.sha256(json.dumps({"seed": seed, "cells": cells}, sort_keys=True).encode()).hexdigest()


def env_info():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": __import__("scipy").__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count()}


def write_json(path, obj, compact=False):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as f:
        if compact:
            json.dump(obj, f, separators=(",", ":"))
        else:
            json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _chunk_path(out, cell_id, j):
    return os.path.join(out, "chunks", cell_id, "%05d.json" % j)


def nchunks(cell):
    return int(math.ceil(cell["R"] / cell["chunk"]))


def init_plan(out, cells, seed, profile):
    """Write ``plan.json`` into ``out`` (or verify that the existing one is identical) and return it."""
    ids = [c["id"] for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cell ids in the plan")
    os.makedirs(out, exist_ok=True)
    plan = {"version": VERSION, "profile": profile, "seed": seed, "target_per_tail": TARGET, "criterion": CRITERION, "conf": CONF,
            "plan_hash": plan_hash(cells, seed), "cells": cells}
    path = os.path.join(out, "plan.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old.get("plan_hash") != plan["plan_hash"]:
            raise ValueError("%s holds a different plan (hash %s, requested %s): use a new output directory" %
                             (path, old.get("plan_hash"), plan["plan_hash"]))
        return old
    plan["env"] = env_info()
    write_json(path, plan)
    return plan


def load_plan(out):
    with open(os.path.join(out, "plan.json")) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------------------------------------------------
def batch_rng(seed, cell_id, b):
    """The random stream of batch ``b`` of a cell (see the module docstring)."""
    h = int.from_bytes(hashlib.sha256(cell_id.encode()).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([seed, h, b]))


def run_cell_batches(cell, seed, first, nbatch, dist=None):
    """Run batches ``first .. first + nbatch - 1`` of ``cell['m']`` replicates: ``(rows, extra)`` with ``rows[name]`` of shape
    (nbatch, T)."""
    proc = covproc.PROCS[cell["proc"]]
    dist = covdist.get_dist(cell["dist"]) if dist is None else dist
    rows, extra = {}, {}
    for b in range(first, first + nbatch):
        r, e = proc.run(cell["params"], dist, batch_rng(seed, cell["id"], b), cell["m"])
        for k, v in r.items():
            rows.setdefault(k, []).append(v)
        for k, v in e.items():
            extra.setdefault(k, []).append(v)
    return ({k: np.array(v, dtype=np.int64) for k, v in rows.items()}, {k: np.array(v, dtype=np.int64) for k, v in extra.items()})


def run_chunk(cell, j, seed):
    """Replicates ``[j chunk, (j + 1) chunk)`` of a cell: the record that is written to disk."""
    size = min(cell["chunk"], cell["R"] - j * cell["chunk"])
    t0 = time.time()
    rows, extra = run_cell_batches(cell, seed, j * cell["chunk"] // cell["m"], size // cell["m"])
    rec = {"cell": cell["id"], "chunk": j, "R": size, "seconds": time.time() - t0,
           "extra": {k: v.sum(axis=0).tolist() for k, v in extra.items()}}
    rec["counts"] = {k: v.sum(axis=0).tolist() for k, v in rows.items()}
    if cell["params"].get("share") and cell["m"] > 1:  # sufficient statistics of the design effect: sum of squares of batch counts
        rec["sumsq"] = {k: (v * v).sum(axis=0).tolist() for k, v in rows.items()}
        rec["nbatch"] = size // cell["m"]
    return rec


_W = {}


def _init_worker(out):
    plan = load_plan(out)
    _W["cells"] = {c["id"]: c for c in plan["cells"]}
    _W["seed"] = plan["seed"]
    _W["out"] = out


def _work(task):
    cid, j = task
    rec = run_chunk(_W["cells"][cid], j, _W["seed"])
    write_json(_chunk_path(_W["out"], cid, j), rec, compact=True)
    return cid, j, rec["seconds"]


def pending_tasks(out, plan):
    tasks = []
    for c in plan["cells"]:
        for j in range(nchunks(c)):
            if not os.path.exists(_chunk_path(out, c["id"], j)):
                tasks.append((c["id"], j))
    return tasks


def run(out, workers=1, max_chunks=None, log=None):
    """Run every missing chunk of ``out/plan.json`` (at most ``max_chunks`` of them: a test aid that simulates an interruption)."""
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    plan = load_plan(out)
    tasks = pending_tasks(out, plan)
    total = sum(nchunks(c) for c in plan["cells"])
    if max_chunks is not None:
        tasks = tasks[:max_chunks]
    log("coverage: %d of %d chunks pending, %d workers" % (len(tasks), total, workers))
    for c in plan["cells"]:
        os.makedirs(os.path.join(out, "chunks", c["id"]), exist_ok=True)
    t0, done, busy, last = time.time(), 0, 0.0, 0.0
    if workers <= 1:
        _init_worker(out)
        results = map(_work, tasks)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(out,))
        results = pool.imap_unordered(_work, tasks, chunksize=1)
    try:
        for cid, j, sec in results:
            done += 1
            busy += sec
            if time.time() - last >= 60.0 or done == len(tasks):
                last = time.time()
                el = last - t0
                log("coverage: %d/%d chunks, %.0f s elapsed, ETA %.0f s (%s)" % (done, len(tasks), el, el / done * (len(tasks) - done), cid))
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    log("coverage: finished %d chunks in %.0f s (%.0f core-seconds)" % (done, time.time() - t0, busy))
    return done


# ----------------------------------------------------------------------------------------------------------------------
# Aggregation and summaries
# ----------------------------------------------------------------------------------------------------------------------
def aggregate_cell(out, cell):
    """Summed chunk records of a cell, or ``None`` if a chunk is missing."""
    rows, sumsq, nbatch, extra, R, sec = {}, {}, 0, {}, 0, 0.0
    for j in range(nchunks(cell)):
        path = _chunk_path(out, cell["id"], j)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            rec = json.load(f)
        R += rec["R"]
        sec += rec["seconds"]
        nbatch += rec.get("nbatch", 0)
        for k, v in rec["counts"].items():
            rows[k] = rows.get(k, 0) + np.array(v, dtype=np.int64)
        for k, v in rec.get("sumsq", {}).items():
            sumsq[k] = sumsq.get(k, 0) + np.array(v, dtype=np.int64)
        for k, v in rec["extra"].items():
            extra[k] = extra.get(k, 0) + np.array(v, dtype=np.int64)
    if R != cell["R"]:
        raise ValueError("cell %s: chunks hold %d replicates, the plan says %d" % (cell["id"], R, cell["R"]))
    return {"R": R, "seconds": sec, "rows": rows, "sumsq": sumsq, "nbatch": nbatch, "extra": extra}


def evaluate_cell(cell, agg):
    """Per row and tail: misses, rate, naive and design-effect-adjusted UCL, ratio to the tail, PASS (``None``: report-only cell,
    or fewer than 175 expected misses at nominal, which G5.3 does not accept as a verdict)."""
    out = {}
    tails = np.array(cell["tails"])
    for row, k in agg["rows"].items():
        D = (design_effect_from_moments(k, agg["sumsq"][row], agg["nbatch"], cell["m"]) if row in agg["sumsq"] else np.ones(len(tails)))
        entries = []
        for i, tau in enumerate(tails):
            R = agg["R"]
            ucl_naive = cp_upper(int(k[i]), R)
            ucl = cp_upper(k[i] / D[i], R / D[i])
            entries.append({"tail": float(tau), "misses": int(k[i]), "R": R, "rate": float(k[i]) / R, "ucl_naive": ucl_naive,
                            "design_effect": float(D[i]), "ucl": ucl, "ucl_over_tail": ucl / float(tau),
                            "rate_over_tail": float(k[i]) / R / float(tau), "powered": bool(R * tau >= TARGET - 1e-9),
                            "pass": None if cell["report_only"] or R * tau < TARGET - 1e-9 else bool(ucl <= CRITERION * tau)})
        out[row] = entries
    return out


def summarize(out, write=True):
    plan = load_plan(out)
    cells, incomplete = [], []
    for c in plan["cells"]:
        agg = aggregate_cell(out, c)
        if agg is None:
            incomplete.append(c["id"])
            continue
        cells.append({**{k: c[k] for k in ("id", "proc", "label", "scen", "dist", "n", "m", "R", "tails", "report_only", "params")},
                      "seconds": agg["seconds"], "extra": {k: v.tolist() for k, v in agg["extra"].items()},
                      "rows": evaluate_cell(c, agg)})
    verdicts = make_verdicts(cells)
    summ = {"version": VERSION, "plan_hash": plan["plan_hash"], "profile": plan.get("profile"), "seed": plan["seed"],
            "env": plan.get("env"), "criterion": "UCL(95%% one-sided Clopper-Pearson) <= %.2f x tail, >= %d replicates x tail" % (CRITERION, TARGET),
            "cells_total": len(plan["cells"]), "cells_complete": len(cells), "incomplete": incomplete, "cells": cells, "verdicts": verdicts,
            "core_seconds": sum(c["seconds"] for c in cells)}
    if write:
        write_json(os.path.join(out, "summary.json"), summ)
        with open(os.path.join(out, "summary.md"), "w") as f:
            f.write(render_markdown(summ))
    return summ


def make_verdicts(cells):
    """Per (procedure, row, scenario, n): tails that pass in EVERY simulated distribution, and the failures.  ``qualified`` is
    ``None`` when nothing was judged (report-only cell, or too few replicates for the verdict)."""
    groups = {}
    for c in cells:
        for row, entries in c["rows"].items():
            g = groups.setdefault((c["label"], row, c["scen"], c["n"]), {"dists": [], "per_tail": {}, "report_only": c["report_only"]})
            g["dists"].append(c["dist"])
            for e in entries:
                g["per_tail"].setdefault(e["tail"], []).append((c["dist"], e))
    out = []
    for (label, row, scen, n), g in sorted(groups.items()):
        fails, passed = [], []
        for tau, lst in sorted(g["per_tail"].items()):
            bad = [(d, e) for d, e in lst if e["pass"] is False]
            if bad:
                fails += [{"tail": tau, "dist": d, "ucl_over_tail": e["ucl_over_tail"], "rate_over_tail": e["rate_over_tail"]} for d, e in bad]
            elif all(e["pass"] for _, e in lst):
                passed.append(tau)
        out.append({"label": label, "row": row, "scen": scen, "n": n, "dists": sorted(g["dists"]), "report_only": g["report_only"],
                    "tails_pass": passed, "tails_fail": fails, "tails_total": len(g["per_tail"]),
                    "qualified": None if not (passed or fails) else not fails and len(passed) == len(g["per_tail"])})
    return out


def _fmt_tail(t):
    return "%.3g" % t


def render_markdown(s):
    L = ["# Coverage simulation summary", "",
         "Criterion (decision G5.3): one-sided miss rate with 95%% one-sided Clopper-Pearson upper limit (UCL) at most %.2f x the nominal "
         "tail, at least %d replicates x tail per cell. Cells complete: %d of %d. Core-seconds: %.0f." %
         (CRITERION, TARGET, s["cells_complete"], s["cells_total"], s["core_seconds"]), "",
         "Seed %s, profile %s, plan hash `%s`; environment: %s." % (s["seed"], s["profile"], s["plan_hash"][:16], json.dumps(s["env"])), ""]
    if s["incomplete"]:
        L += ["**Incomplete cells (%d):** %s" % (len(s["incomplete"]), ", ".join(s["incomplete"][:20]) + (" ..." if len(s["incomplete"]) > 20 else "")), ""]
    ver = s["verdicts"]
    L += ["## Tail levels", "",
          "- sampling limits (Student, bootstrap-t; lower and upper): " + ", ".join(_fmt_tail(t) for t in covproc.sampling_tails()),
          "- budget bounds (ContrastBound, LadderBound; K = 3, 4): " + ", ".join(_fmt_tail(t) for t in covproc.component_tails()),
          "- Gate 5 TOST and whole robust-sign pipeline (Holm levels, m = 7): " + ", ".join(_fmt_tail(t) for t in covproc.holm_levels()), "",
          "## Qualification matrix", "",
          "Entry: `ALL` if every tail passes in every simulated distribution, else `passing/total tails`; `rep` = reported, never judged (BCa); `n/a` = fewer than 175 expected misses at nominal, not judged.",
          ""]
    keys = sorted({(v["label"], v["row"], v["scen"]) for v in ver})
    ns = sorted({v["n"] for v in ver})
    L += ["| procedure | row | scenario | " + " | ".join("n=%d" % n for n in ns) + " |", "|" + "---|" * (3 + len(ns))]
    idx = {(v["label"], v["row"], v["scen"], v["n"]): v for v in ver}
    for k in keys:
        cellsv = []
        for n in ns:
            v = idx.get(k + (n,))
            if v is None:
                cellsv.append("")
            elif v["report_only"]:
                cellsv.append("rep")
            elif v["qualified"] is None:
                cellsv.append("n/a")
            else:
                cellsv.append("ALL" if v["qualified"] else "%d/%d" % (len(v["tails_pass"]), v["tails_total"]))
        L.append("| %s | %s | %s | %s |" % (k[0], k[1], k[2] or "-", " | ".join(cellsv)))
    L += ["", "## Verdict list", "", "Qualified = every tail passes in every simulated distribution of that cell group.", ""]
    for k in keys:
        vs = sorted((idx[k + (n,)] for n in ns if k + (n,) in idx), key=lambda v: v["n"])
        if vs[0]["report_only"]:
            continue
        ok = [v["n"] for v in vs if v["qualified"]]
        bad = ["%d (%d of %d tails)" % (v["n"], len(v["tails_pass"]), v["tails_total"]) for v in vs if v["qualified"] is False]
        na = [v["n"] for v in vs if v["qualified"] is None]
        L.append("- **%s / %s%s**: qualified at n = %s; not qualified at n = %s%s" % (
            k[0], k[1], (" / " + k[2]) if k[2] else "", ", ".join(map(str, ok)) or "none", ", ".join(bad) or "none",
            ("; not judged at n = " + ", ".join(map(str, na))) if na else ""))
    fails = [(v, f) for v in ver for f in v["tails_fail"]]
    L += ["", "## Failures (%d tail x distribution entries)" % len(fails), ""]
    if fails:
        L += ["`marginal`: the point estimate of the miss rate is within 1.25 x tail and only the 95% limit exceeds it (Monte Carlo "
              "noise of an exactly nominal procedure fails a cell this way with probability about 1e-3; rerun with more replicates).", "",
              "| procedure | row | scenario | n | dist | tail | rate/tail | UCL/tail | |", "|---|---|---|---|---|---|---|---|---|"]
        for v, f in fails[:400]:
            L.append("| %s | %s | %s | %d | %s | %s | %.2f | %.2f | %s |" % (
                v["label"], v["row"], v["scen"] or "-", v["n"], f["dist"], _fmt_tail(f["tail"]), f["rate_over_tail"], f["ucl_over_tail"],
                "marginal" if f["rate_over_tail"] <= CRITERION else ""))
        if len(fails) > 400:
            L.append("| ... %d more in summary.json | | | | | | | | |" % (len(fails) - 400))
    else:
        L.append("none")
    L += ["", "## UCL / tail by cell", "", "Entries are UCL/tail (pass iff at most %.2f; `*` marks a failure); the header lists the tails." % CRITERION, ""]
    groups = {}
    for c in s["cells"]:
        for row, entries in c["rows"].items():
            groups.setdefault((c["label"], row, c["scen"], c["dist"]), []).append((c["n"], c, entries))
    for (label, row, scen, dist), lst in sorted(groups.items()):
        tails = [e["tail"] for e in lst[0][2]]
        L += ["### %s / %s%s / %s" % (label, row, (" / " + scen) if scen else "", dist), "",
              "| n | R | maxD | " + " | ".join(_fmt_tail(t) for t in tails) + " |", "|" + "---|" * (3 + len(tails))]
        for n, c, entries in sorted(lst, key=lambda t: t[0]):
            cellsv = ["%.2f%s" % (e["ucl_over_tail"], "*" if e["pass"] is False else "") for e in entries]
            L.append("| %d | %d | %.3f | %s |" % (n, c["R"], max(e["design_effect"] for e in entries), " | ".join(cellsv)))
        L.append("")
    extras = [(c["id"], c["extra"]) for c in s["cells"] if any(np.sum(v) > 0 for k, v in c["extra"].items() if k in ("unresolved", "nonfinite"))]
    L += ["## Unresolved bootstrap tails", "", ("cells with infinite (unresolved) limits: %d (see summary.json `extra`)" % len(extras)) if extras else
          "none: every bootstrap tail was resolved by its B", ""]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------------------------------------------------------
SAMPLING_NS = (20, 30, 50, 100, 200, 500, 1000, 1650)  # n of the sampling procedures (SF-43 N: 30 to 1650)
GATE5_NS = (20, 30, 50, 100, 170)  # paired seeds of Gate 5 (SF-43: about 30 to 170 per arm)
COMPONENT_NS = (20, 30, 50, 100, 200)  # seeds per budget arm (budget_arm_size: N / 2 between 20 and 200)
LADDER_NS = (20, 30, 50, 100, 200)
# production sizes of the pipeline: (K, ladder production index, N): the SF-43 sigma^2 = 0.25 / 1, 2.25 and 4 structures
PIPELINE_CONFIGS = ((3, 1, 100), (3, 1, 300), (3, 1, 1000), (4, 2, 30), (4, 2, 100), (4, 2, 300))


def _est_spr(proc, n, params):
    """Rough seconds per replicate on the development workstation (fits of the ``bench`` output): it only sizes the chunks."""
    B = params.get("B", 0) or params.get("B_c", 0)
    return {"sampling": (3e-6 + 3e-8 * n) if params.get("method") == "student" else 2e-4 + 2.5e-8 * n * B,
            "tost": 1e-5 * (1.0 + n / 50.0), "contrast": 8e-4 + 3.4e-5 * n, "ladder": 5e-3 + 1e-4 * n,
            "pipeline": 8e-3 + 3e-5 * n, "bca": 5e-3 * (1.0 + n / 50.0)}[proc]


def _chunk_for(spr, m, seconds=60.0, cap=50000):
    return int(max(m, min(cap, seconds / max(spr, 1e-7))))


def profile_full(target=TARGET, boot_per_tail=10, m_share=10, dists=("normal", "stress", "stress-"),
                 ladder_ns=LADDER_NS, component_ns=COMPONENT_NS):
    """The default full study.  ``dists``: ``normal`` plus the stress family (``stress`` right-skewed, ``stress-`` its reflection:
    a one-sided limit or a bias bound is anti-conservative for one of the two skew directions)."""
    cells = []
    t_s = float(covproc.sampling_tails().min())
    t_c = float(covproc.component_tails().min())
    t_a = float(covproc.holm_levels().min())

    def add(proc, label, scen, dist, n, params, share=False, m=1, report_only=False):
        c = make_cell(proc, label, scen, dist, n, params, target, m, None, share, report_only)
        spr = _est_spr(proc, n, params)
        c["chunk"] = int(min(c["R"], math.ceil(_chunk_for(spr, m) / m) * m))
        cells.append(c)

    for d in dists:
        for n in SAMPLING_NS:
            add("sampling", "sampling:student", "", d, n, {"method": "student"}, m=1000)
            add("sampling", "sampling:boott", "", d, n, {"method": "boott", "B": covproc.boot_size(t_s, boot_per_tail)}, m=1)
        for rho in (0.0, 0.5, 0.9):
            for n in GATE5_NS:
                add("tost", "tost:gate5", "rho=%g" % rho, d, n, {"rho": rho, "cv": 0.3}, m=1000)
        for n in (20, 30, 50, 100):
            add("bca", "bca:mean", "", d, n, {"B": 1999}, m=100, report_only=True)
    Bc = covproc.boot_size(t_c, boot_per_tail)
    Bl = covproc.boot_size(t_c / 2.0, boot_per_tail)
    for d in dists:
        for kind in ("paired", "unpaired", "interaction"):
            for n in component_ns:
                add("contrast", "contrast:" + kind, "", d, n, {"kind": kind, "kappa": 5.0, "B": Bc}, share=True, m=m_share)
        for ip in (1, 2):
            for scenario, order in (("power", 2.0), ("power", 1.5), ("power", 1.0), ("nonmono", 1.0)):
                for kappa in (12.0, 25.0, 50.0):
                    for n in ladder_ns:
                        scen = "%s,ip=%d,kappa=%g" % ("order=%g" % order if scenario == "power" else "nonmono", ip, kappa)
                        add("ladder", "ladder:E_h", scen, d, n, {"scenario": scenario, "order": order, "ip": ip, "kappa": kappa, "B": Bl},
                            share=True, m=m_share)
        for K, ip, N in PIPELINE_CONFIGS:
            Bs = covproc.boot_size(t_a / 2.0, boot_per_tail)
            for bname, b, order in (("zero", 0.0, 1.0), ("bias,order=1", 2.0, 1.0), ("bias,order=2", 2.0, 2.0)):
                pp = {"N": N, "nb": covproc.budget_arm_size(N), "K": K, "ip": ip, "order": order, "b_h": b, "b_int": b, "b_w": b,
                      "sampling": "student", "B_c": covproc.boot_size(t_a / (2.0 * K), boot_per_tail),
                      "B_l": covproc.boot_size(t_a / (4.0 * K), boot_per_tail), "B_s": Bs}
                add("pipeline", "pipeline:K%d,ip=%d" % (K, ip), bname, d, N, pp, share=True, m=m_share)
                if N <= 100 and bname == "bias,order=1":
                    add("pipeline", "pipeline:K%d,ip=%d" % (K, ip), "boott," + bname, d, N, dict(pp, sampling="boott"), share=True, m=m_share)
    return cells


def profile_smoke(target=2.0):
    """A tiny plan for a quick end-to-end check (tests, installation): a few cells of every procedure, few replicates."""
    cells = []
    add = lambda *a, **k: cells.append(make_cell(*a, target=target, **k))
    add("sampling", "sampling:student", "", "normal", 30, {"method": "student"}, m=100, chunk=300)
    add("sampling", "sampling:boott", "", "stress", 20, {"method": "boott", "B": 1000}, m=1, chunk=100)
    add("tost", "tost:gate5", "rho=0.5", "normal", 30, {"rho": 0.5, "cv": 0.3}, m=100, chunk=300)
    add("contrast", "contrast:paired", "", "stress-", 30, {"kind": "paired", "kappa": 5.0, "B": 2000}, m=10, chunk=100, share=True)
    add("contrast", "contrast:unpaired", "", "normal", 30, {"kind": "unpaired", "kappa": 5.0, "B": 2000}, m=10, chunk=100, share=True)
    add("ladder", "ladder:E_h", "order=1,ip=1,kappa=12", "normal", 30,
        {"scenario": "power", "order": 1.0, "ip": 1, "kappa": 12.0, "B": 2000}, m=10, chunk=100, share=True)
    add("pipeline", "pipeline:K3", "student,bias", "normal", 30,
        {"N": 50, "nb": 25, "K": 3, "ip": 1, "order": 1.0, "b_h": 2.0, "b_int": 2.0, "b_w": 2.0, "sampling": "student",
         "B_c": 2000, "B_l": 4000, "B_s": 1000}, m=10, chunk=100, share=True)
    add("bca", "bca:mean", "", "normal", 30, {"B": 199}, m=10, chunk=30, report_only=True)
    return cells


PROFILES = {"full": profile_full, "smoke": profile_smoke}


# ----------------------------------------------------------------------------------------------------------------------
# Cost benchmark
# ----------------------------------------------------------------------------------------------------------------------
def bench(cells, min_seconds=0.5, per_group=2, seed=1, log=None):
    """CPU seconds per replicate of every cell (process time: independent of other load on the machine): one warm-up batch, then
    batches until ``min_seconds`` of CPU time have elapsed (at least 2).  Cells that differ only by scenario parameters share the
    cost (same procedure, ``n``, distribution, sizes): ``per_group`` of each group are timed and the others get the group mean
    (``per_group=0`` times every cell)."""
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    res, groups = {}, {}
    for c in cells:
        key = (c["label"], c["n"], c["dist"], c["params"].get("B"), c["m"])
        g = groups.setdefault(key, [])
        g.append(c["id"])
        if per_group and len(g) > per_group:
            continue
        dist = covdist.get_dist(c["dist"])
        run_cell_batches(c, seed, 0, 1, dist)
        t0, nb = time.process_time(), 0
        while nb < 2 or time.process_time() - t0 < min_seconds:
            run_cell_batches(c, seed, 1 + nb, 1, dist)
            nb += 1
        res[c["id"]] = (time.process_time() - t0) / (nb * c["m"])
        log("bench %-70s %.3g s/replicate, R=%d -> %.0f core-s" % (c["id"], res[c["id"]], c["R"], res[c["id"]] * c["R"]))
    for ids in groups.values():
        timed = [res[i] for i in ids if i in res]
        for i in ids:
            res.setdefault(i, sum(timed) / len(timed))
    return res


def project(cells, spr, workers=80, host_factor=1.5, efficiency=0.9):
    """Core-seconds per label and the projected wall time on ``workers`` cores (``host_factor``: slower host core, ``efficiency``:
    parallel efficiency)."""
    by = {}
    for c in cells:
        by[c["label"]] = by.get(c["label"], 0.0) + spr[c["id"]] * c["R"]
    core = sum(by.values())
    return {"core_seconds_local": core, "core_hours_local": core / 3600.0, "by_label_core_hours": {k: v / 3600.0 for k, v in sorted(by.items())},
            "host_factor": host_factor, "efficiency": efficiency, "workers": workers,
            "wall_hours_host": core * host_factor / (workers * efficiency) / 3600.0}


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
def _build(args):
    kw = {}
    if args.profile == "full":
        kw = {"target": args.target, "boot_per_tail": args.boot_per_tail, "m_share": args.share_m}
        if args.ladder_ns:
            kw["ladder_ns"] = tuple(int(x) for x in args.ladder_ns.split(","))
        if args.dists:
            kw["dists"] = tuple(args.dists.split(","))
    else:
        kw = {"target": args.target}
    return PROFILES[args.profile](**kw)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m mcstat.coverage", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "bench"):
        sp = sub.add_parser(name)
        sp.add_argument("--profile", choices=sorted(PROFILES), default="full")
        sp.add_argument("--target", type=float, default=None, help="replicates x tail (default 175 for full, 2 for smoke)")
        sp.add_argument("--boot-per-tail", type=float, default=10.0)
        sp.add_argument("--share-m", type=int, default=10, help="replicates per shared bootstrap draw")
        sp.add_argument("--ladder-ns", default=None, help="seeds per ladder arm, comma-separated (default 20,30,50,100,200)")
        sp.add_argument("--dists", default=None, help="comma-separated distributions (full profile)")
        sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
        sp.add_argument("--only", default=None, help="keep only the cells whose id matches this regular expression")
        if name != "bench":
            sp.add_argument("--out", required=True)
        if name == "run":
            sp.add_argument("--workers", type=int, default=os.cpu_count() or 1)
            sp.add_argument("--max-chunks", type=int, default=None, help="stop after this many chunks (testing aid)")
        if name == "bench":
            sp.add_argument("--workers", type=int, default=80, help="host cores for the projection")
            sp.add_argument("--host-factor", type=float, default=1.5)
            sp.add_argument("--per-group", type=int, default=2, help="cells timed per (procedure, n, distribution, B) group; 0 = all")
            sp.add_argument("--json", default=None, help="write the projection here")
    sp = sub.add_parser("summarize")
    sp.add_argument("--out", required=True)
    sp = sub.add_parser("stress")
    sp.add_argument("--sf39", required=True, help="SF-39 artifact root")
    sp.add_argument("--out", default=covdist.STRESS_JSON)
    sp.add_argument("--conf", type=float, default=0.95)
    sp.add_argument("--B", type=int, default=10000)
    args = ap.parse_args(argv)
    if args.cmd == "stress":
        st = covdist.stress_from_sf39(args.sf39, args.conf, args.B)
        write_json(args.out, st)
        print(json.dumps({"envelope": st["envelope"], "amplitudes": {a: {k: v[k] for k in ("skew", "ekurt")} for a, v in st["amplitudes"].items()}}, indent=1))
        return 0
    if args.cmd == "summarize":
        s = summarize(args.out)
        print("summary: %d of %d cells complete; %d verdict lines; written to %s" % (s["cells_complete"], s["cells_total"], len(s["verdicts"]), args.out))
        return 0
    if args.target is None:
        args.target = TARGET if args.profile == "full" else 2.0
    cells = _build(args)
    if args.only:
        cells = [c for c in cells if re.search(args.only, c["id"])]
        if not cells:
            raise SystemExit("--only %r matches no cell" % (args.only,))
    if args.cmd == "plan":
        plan = init_plan(args.out, cells, args.seed, args.profile)
        print("plan: %d cells, %d chunks, hash %s" % (len(plan["cells"]), sum(nchunks(c) for c in plan["cells"]), plan["plan_hash"][:16]))
        return 0
    if args.cmd == "bench":
        spr = bench(cells, per_group=args.per_group, seed=args.seed)
        proj = project(cells, spr, args.workers, args.host_factor)
        print(json.dumps(proj, indent=1))
        if args.json:
            write_json(args.json, {"seconds_per_replicate": spr, "projection": proj})
        return 0
    init_plan(args.out, cells, args.seed, args.profile)
    run(args.out, args.workers, args.max_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
