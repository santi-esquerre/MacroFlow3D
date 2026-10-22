#!/usr/bin/env python3
"""SF-40 N8: derived tables of the experiment note, computed only from the imported raw evidence.

  note_tables.py RAW_DIR OUT_DIR      reads RAW_DIR (= artifact `raw/`), writes into OUT_DIR:
                                        feasibility.csv, feasibility.md, pid_plateaus.md, completeness.md, summary.md
  note_tables.py --selftest           assert-based arithmetic checks on hand-computed examples (no raw files)

Stdlib only; deterministic (re-running on the same raw/ regenerates byte-identical files).
Every number printed comes from a raw file (cited in the output) or is a documented function of such numbers.
1 MiB = 1048576 B.
"""
import itertools
import os
import re
import sys

MIB = 1048576.0
LIMIT_MIB = 30000
RUNS = ("d512", "d3328_i", "d3328_ii")
PHASES = ("F", "S1", "S2", "S3", "T")
SIGMA_LAMBDA = ((1.0, 8), (2.25, 16), (4.0, 32))  # SF-37: lambda/h 8 / 16 / 32 at sigma^2 1 / 2.25 / 4
BDD_LAMBDA = (204.8, 25.6, 25.6)  # audit 2026-10-22 section 9: B&dD domain in lambda


# ------------------------------------------------------------------------------------------ parsing
def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def tokens(line):
    """`k=v` tokens of a log line (v up to the next space)."""
    return dict(re.findall(r"(\w+)=(\S+)", line))


def parse_items(s):
    out = []
    if s in ("-", ""):
        return out
    for tok in s.split(","):
        k, v = tok.rsplit(":", 1)
        out.append((k, int(v)))
    return out


def parse_run_log(path):
    """-> {'geom': {...}, 'phases': {P: {...}}} from MEM_GEOM / MEM_PHASE lines."""
    geom, phases = {}, {}
    for line in read(path).splitlines():
        if line.startswith("MEM_GEOM "):
            t = tokens(line)
            geom = {"pid": int(t["pid"]), "nx": int(t["nx"]), "ny": int(t["ny"]), "nz": int(t["nz"]),
                    "n": int(t["n"]), "levels": int(t["levels"]),
                    "level_cells": [int(x) for x in t["level_cells"].split(",")], "coarsest": t["coarsest"],
                    "face_max": int(t["face_max"]), "bhat": t["bhat"], "bhat_host_bytes": int(t["bhat_host_bytes"])}
        elif line.startswith("MEM_PHASE "):
            t = tokens(line)
            p = t["phase"]
            phases[p] = {"exact": int(t["exact_bytes"]), "cufft_work": int(t["cufft_work"]),
                         "device_used": int(t["device_used"]), "seconds": float(t["seconds"]),
                         "items": parse_items(t.get("items", "-")),
                         "cufft_named": [(k, int(v)) for k, v in t.items() if k.startswith("cufft_") and k != "cufft_work"]}
    return {"geom": geom, "phases": phases}


def parse_apps(path, pid):
    out = []
    for line in read(path).splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3 and p[1] == str(pid):
            out.append((p[0], int(p[2])))
    return out


def rle(series):
    """series of (timestamp, value) -> [(value, count, first_ts)]"""
    return [(k, len(g), g[0][0]) for k, g in ((k, list(g)) for k, g in itertools.groupby(series, key=lambda x: x[1]))]


# ------------------------------------------------------------------------------------------ arithmetic
def closed_form_s2(n, m, level_cells, s_scr):
    """S+ closed form at the S2 peak: 8[7n + 4 sum_{l>=1} n_l] + 32 (m - n) + S_scr (preregistration RR3)."""
    return 8 * (7 * n + 4 * sum(level_cells[1:])) + 32 * (m - n) + s_scr


def mg_levels_box(ny):
    """box_auto_mg_levels for nx a multiple of 2^(levels-1): the transverse extent governs (LongBoxStage.cu)."""
    can = lambda k: k % 2 == 0 and k // 2 >= 4 and (k // 2) % 2 == 0
    lv, a = 1, ny
    while can(a):
        a //= 2
        lv += 1
    return lv


def feasibility(n_meas, s2_exact, ctx_mib, peak_mib, ny, sig_lam, bdd, limit_mib=LIMIT_MIB):
    """Derived capacity at `limit_mib`: peak(n') = ctx + U + b n', b = s2_exact / n_meas (exact B/cell),
    U = peak - ctx - s2_exact / MiB (unmodeled overhead measured at n_meas, held constant)."""
    b = s2_exact / n_meas
    u = peak_mib - ctx_mib - s2_exact / MIB
    cells_max = (limit_mib - ctx_mib - u) * MIB / b
    mult = 2 ** (mg_levels_box(ny) - 1)
    nx_max = int(cells_max // (ny * ny)) // mult * mult
    rows = []
    for sigma2, r in sig_lam:
        bdd_cells = (bdd[0] * r) * (bdd[1] * r) * (bdd[2] * r)
        need = bdd_cells * b / MIB + ctx_mib + u
        rows.append({"sigma2": sigma2, "lam_over_h": r, "box_in_lambda": (n_meas / ny / ny / r, ny / r, ny / r),
                     "nx_max": nx_max, "lx_max_lambda": nx_max / r, "cells_max": cells_max,
                     "bdd_cells": bdd_cells, "bdd_need_mib": need, "bdd_fits": bdd_cells <= cells_max})
    return {"b": b, "u": u, "cells_max": cells_max, "nx_max": nx_max, "rows": rows}


# ------------------------------------------------------------------------------------------ analysis blocks
class Raw:
    def __init__(self, root):
        self.root = root
        self.camp = os.path.join(root, "sf40_campaign")
        mem = os.path.join(self.camp, "mem")
        self.mem = mem
        self.mres = read(os.path.join(mem, "results.txt"))
        self.logs = {r: parse_run_log(os.path.join(mem, r + ".log")) for r in RUNS}
        self.ctx_mib = int(re.search(r"^C_CONTEXT mib=(\d+)", self.mres, re.M).group(1))
        self.probe_dev = int(re.search(r"^PROBE run=probe1 .*device_used_bytes=(\d+)", self.mres, re.M).group(1))
        self.pid = {r: int(read(os.path.join(mem, r + ".pid_tree")).split()[0]) for r in RUNS}
        self.apps = {r: parse_apps(os.path.join(mem, r + ".apps.csv"), self.pid[r]) for r in RUNS}
        for r in RUNS:  # the PID printed by the run itself equals the process-tree PID
            assert self.logs[r]["geom"]["pid"] == self.pid[r], r
        self.peak = {r: max(v for _, v in self.apps[r]) for r in RUNS}


def completeness_md(R):
    out = ["# Completeness instrument (RR3): device_used against exact bytes", "",
           "Source: `sf40_campaign/mem/{d512,d3328_i,d3328_ii}.log` (`MEM_PHASE` lines) and `sf40_campaign/mem/results.txt`",
           "(`PROBE ... device_used_bytes`). The stage-start `device_used` is not printed; the context probe value %d B"
           % R.probe_dev, "is used as `start` (same process creation sequence, nothing else allocated).", "",
           "- `pre` = device_used - start - exact_bytes (as pre-registered).",
           "- `adj` = device_used - start - (exact_bytes - cufft_work) (excludes the cuFFT plan work areas, which no longer",
           "  exist when `LongBoxStage.cu` samples `device_used` right after the SF-18 / SF-28 call returns; S2 and T have",
           "  cufft_work = 0, so pre = adj there).",
           "- `rounded excess` = sum over allocations rounded up to 2 MiB minus exact (`C_INFO ... sum_rounded_2MiB_per_alloc`).", ""]
    for r in RUNS:
        out += ["## %s" % r, "", "| phase | exact B | cufft_work B | device_used B | pre B | adj B | adj MiB | rounded excess B |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for p in PHASES:
            ph = R.logs[r]["phases"][p]
            pre = ph["device_used"] - R.probe_dev - ph["exact"]
            adj = ph["device_used"] - R.probe_dev - (ph["exact"] - ph["cufft_work"])
            m = re.search(r"^C_INFO run=%s phase=%s exact=(\d+) sum_rounded_2MiB_per_alloc=(\d+)" % (r, p), R.mres, re.M)
            rex = int(m.group(2)) - int(m.group(1))
            out.append("| %s | %d | %d | %d | %d | %d | %.2f | %d |" % (p, ph["exact"], ph["cufft_work"], ph["device_used"],
                                                                      pre, adj, adj / MIB, rex))
        out.append("")
    return "\n".join(out) + "\n"


def plateaus_md(R):
    out = ["# PID time series (200 ms) of the D runs against exact bytes + context", "",
           "Source: `sf40_campaign/mem/<run>.apps.csv` filtered to the PID of `<run>.pid_tree` (= `MEM_GEOM pid=`).",
           "Context C = %d MiB (`C_CONTEXT`, `mem/results.txt`). expected(phase) = exact_bytes / 2^20 + C." % R.ctx_mib,
           "The series is split structurally (windows below); per phase the sample nearest to expected inside its window is reported.", ""]
    for r in RUNS:
        ser = R.apps[r]
        runs = rle(ser)
        gmax = max(v for _, v in ser)
        # S2 plateau = longest run at the global maximum
        s2 = max((i for i, x in enumerate(runs) if x[0] == gmax), key=lambda i: runs[i][1])
        pre = sorted({x[0] for x in runs[:s2]}, reverse=True)[:3]
        post = sorted({x[0] for x in runs[s2 + 1:]}, reverse=True)[:3]
        # structural windows of the series: F = before the host gap (first run of >= 3 samples at <= C + 16 MiB after the
        # ramp), S1 = gap .. S2 plateau, S2 = the plateau, S3/T = after the plateau
        ramp = next(i for i, x in enumerate(runs) if x[0] > R.ctx_mib + 1000)
        gap = next(i for i, x in enumerate(runs) if i > ramp and x[1] >= 3 and x[0] <= R.ctx_mib + 16)
        wins = {"F": runs[:gap], "S1": runs[gap + 1:s2], "S2": [runs[s2]], "S3": runs[s2 + 1:], "T": runs[s2 + 1:]}
        out += ["## %s" % r, "", "samples %d, PID peak %d MiB." % (len(ser), gmax), "",
                "| phase | window | exact MiB | expected = exact + C | max sample in window | nearest sample (#, first ts) | delta MiB | within 64 MiB |",
                "|---|---|---:|---:|---:|---|---:|---|"]
        for p in PHASES:
            e = R.logs[r]["phases"][p]["exact"] / MIB
            exp = e + R.ctx_mib
            w = wins[p]
            near = min(w, key=lambda x: (abs(x[0] - exp), x[2]))
            mx = "%d" % max(x[0] for x in w) if p in ("F", "S1", "S2") else "-"
            win = {"F": "before the host gap", "S1": "gap .. S2 plateau", "S2": "plateau", "S3": "after the plateau",
                   "T": "after the plateau"}[p]
            out.append("| %s | %s | %.2f | %.2f | %s | %d (x%d, %s) | %+.2f | %s |" % (
                p, win, e, exp, mx, near[0], near[1], near[2][11:], near[0] - exp, "yes" if abs(near[0] - exp) <= 64 else "no"))
        longr = [x for x in runs if x[1] >= 30]
        t_exp = R.logs[r]["phases"]["T"]["exact"] / MIB + R.ctx_mib
        out += ["", "Highest distinct sampled levels before the S2 plateau: %s MiB; after it: %s MiB."
                % (", ".join(map(str, pre)), ", ".join(map(str, post))),
                "S2 plateau: %d MiB x %d samples (%s .. ), S2 expected %.2f MiB, delta %+.2f MiB."
                % (gmax, runs[s2][1], runs[s2][2][11:], R.logs[r]["phases"]["S2"]["exact"] / MIB + R.ctx_mib,
                   gmax - (R.logs[r]["phases"]["S2"]["exact"] / MIB + R.ctx_mib)),
                "Last run of >= 30 samples (tracker phase): %d MiB x %d samples, T expected %.2f MiB, delta %+.2f MiB."
                % (longr[-1][0], longr[-1][1], t_exp, longr[-1][0] - t_exp), "",
                "Run-length encoding (value MiB x samples @ first timestamp):", "", "```"]
        out.append("  ".join("%d x%d@%s" % (v, c, ts[11:]) for v, c, ts in runs))
        out += ["```", ""]
        # any sample that equals the S2 level minus one 8m buffer
        m8 = 8 * R.logs[r]["geom"]["face_max"] / MIB
        hit = [v for v in sorted({x[0] for x in runs}) if abs((gmax - v) - m8) <= 1.0]
        out.append("One 8m buffer = 8 * face_max / 2^20 = %.2f MiB; sampled levels within 1 MiB of (S2 plateau - 8m): %s."
                   % (m8, ", ".join(map(str, hit)) if hit else "none"))
        out.append("")
    return "\n".join(out) + "\n"


def feasibility_out(R):
    r = "d3328_i"
    g = R.logs[r]["geom"]
    s2 = R.logs[r]["phases"]["S2"]["exact"]
    f = feasibility(g["n"], s2, R.ctx_mib, R.peak[r], g["ny"], SIGMA_LAMBDA, BDD_LAMBDA)
    for q in RUNS[1:]:
        assert R.peak[q] == R.peak[r] and R.logs[q]["phases"]["S2"]["exact"] == s2
    b_meas = (R.peak[r] - R.ctx_mib) * MIB / g["n"]
    csv = ["sigma2,lambda_over_h,box_lx_lambda,box_ly_lz_lambda,cells_box_measured,pid_peak_mib_measured,"
           "cells_max_30000_derived,nx_max_ny384_derived,lx_max_lambda_derived,bdd_cells_derived,bdd_need_mib_derived,bdd_fits_derived"]
    for x in f["rows"]:
        csv.append("%g,%d,%.6g,%.6g,%d,%d,%.6g,%d,%.6g,%.6g,%.2f,%s" % (
            x["sigma2"], x["lam_over_h"], x["box_in_lambda"][0], x["box_in_lambda"][1], g["n"], R.peak[r],
            x["cells_max"], x["nx_max"], x["lx_max_lambda"], x["bdd_cells"], x["bdd_need_mib"],
            "fit" if x["bdd_fits"] else "no_fit"))
    md = ["# Feasibility table (Recorded, no threshold)", "",
          "Inputs (measured, `sf40_campaign/mem/d3328_i.log`, `d3328_i.apps.csv`, `results.txt`): n = %d cells (%dx%dx%d), "
          "S2 exact bytes = %d (%.4f B/cell), context C = %d MiB, PID peak = %d MiB (identical in `d3328_ii`)."
          % (g["n"], g["nx"], g["ny"], g["nz"], s2, f["b"], R.ctx_mib, R.peak[r]), "",
          "- measured B/cell at the peak phase, exact bytes: %.4f (S2 exact / n)" % f["b"],
          "- measured B/cell at the peak, PID basis: %.4f ((PID peak - C) * 2^20 / n; includes the unmodeled overhead)" % b_meas,
          "- measured unmodeled overhead U at 3328x384x384 = PID peak - C - S2 exact / 2^20 = %.2f MiB" % f["u"],
          "- measured at 512^3 (`d512`): PID peak %d MiB, S2 exact %.2f MiB, overhead %.2f MiB"
          % (R.peak["d512"], R.logs["d512"]["phases"]["S2"]["exact"] / MIB,
             R.peak["d512"] - R.ctx_mib - R.logs["d512"]["phases"]["S2"]["exact"] / MIB), "",
          "Derived (extrapolation, not measured): peak(n') = C + U + b n' with b = exact B/cell and U held at its 3328 value;",
          "cells_max = (30000 - C - U) * 2^20 / b = %.6g cells; nx_max (ny = nz = 384, nx a multiple of %d so that the "
          "automatic hierarchy keeps %d levels) = %d."
          % (f["cells_max"], 2 ** (mg_levels_box(g["ny"]) - 1), mg_levels_box(g["ny"]), f["nx_max"]),
          "The linear model ignores the (m - n) capacity term and the scratch growth with nx (both below 0.1 B/cell), and any "
          "change of the PCG iteration count; it is not a prediction for a box that was not run.", "",
          "| sigma^2 | lambda/h | measured box (lambda) | measured cells | measured PID peak MiB | derived cells_max | derived nx_max | "
          "derived Lx (lambda) | B&dD cells (derived) | B&dD need MiB (derived) | B&dD |",
          "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for x in f["rows"]:
        md.append("| %g | %d | %.4g x %.4g x %.4g | %d | %d | %.4g | %d | %.4g | %.4g | %.1f | %s |" % (
            x["sigma2"], x["lam_over_h"], x["box_in_lambda"][0], x["box_in_lambda"][1], x["box_in_lambda"][1],
            g["n"], R.peak[r], x["cells_max"], x["nx_max"], x["lx_max_lambda"], x["bdd_cells"], x["bdd_need_mib"],
            "fits" if x["bdd_fits"] else "does not fit"))
    md += ["", "B&dD domain = %g x %g x %g lambda (audit `docs/experiments/2026-10-22-lagrangian-memory-audit.md` section 9); "
           "cells = product of the extents times (lambda/h)^3 (square transverse section, h = 1/ny)." % BDD_LAMBDA,
           "The (sigma^2, lambda/h) pairs are the SF-37 values; this file makes no statement about them.", ""]
    return "\n".join(csv) + "\n", "\n".join(md) + "\n"


def summary_md(R):
    c = R.camp
    out = ["# SF-40 derived tables (all values from `raw/`)", ""]
    # ---- A
    res = read(os.path.join(c, "bytecmp", "results.txt")).splitlines()
    cnt = {"IDENTICAL": 0, "DIFF": 0, "MISSING": 0, "SKIP": 0}
    sets = {}
    for l in res:
        w = l.split(" ", 1)
        if w[0] in cnt:
            cnt[w[0]] += 1
            it = w[1].split(":")[0]
            kind = ("lean" if it.startswith("lean_") else "l6" if it.startswith("l6_") else "r" if it.startswith("r:") or it == "r"
                    else "binary" if it == "binary" else "other")
            s = sets.setdefault(kind, {"cases": set(), "items": 0, "other": 0})
            s["cases"].add(it)
            s["items"] += 1 if w[0] == "IDENTICAL" else 0
            s["other"] += 0 if w[0] == "IDENTICAL" else 1
    rr = {}
    for l in res:
        m = re.match(r"IDENTICAL r:(\w+):", l)
        if m:
            rr[m.group(1)] = rr.get(m.group(1), 0) + 1
    r_cases = sorted({":".join(l.split(" ", 1)[1].split(":")[:2 if l.split(" ", 1)[1].startswith("r:prod32") else 3])
                      for l in res if l.startswith("IDENTICAL r:")})
    r_items = sum(1 for l in res if l.startswith("IDENTICAL r:"))
    out += ["## A / R: `sf40_campaign/bytecmp/results.txt`", "",
            "Totals: IDENTICAL %d, DIFF %d, MISSING %d, SKIP %d; last line `%s`."
            % (cnt["IDENTICAL"], cnt["DIFF"], cnt["MISSING"], cnt["SKIP"], res[-1]), "",
            "| set | cases | IDENTICAL items | DIFF/MISSING |", "|---|---:|---:|---:|"]
    for k in ("lean", "l6"):
        out.append("| %s | %d | %d | %d |" % (k.upper(), len(sets[k]["cases"]), sets[k]["items"], sets[k]["other"]))
    out.append("| R (default path, SF-35/36/37) | %d runs | %d | 0 |" % (len(r_cases), r_items))
    out += ["", "R items by group: %s." % ", ".join("%s %d" % (k, v) for k, v in sorted(rr.items())),
            "`binary` item: `%s`." % next(l for l in res if l.startswith("IDENTICAL binary")), ""]
    # ---- C
    out += ["## C: exact bytes and S+ closed form", "",
            "Source: `sf40_campaign/mem/{d512,d3328_i,d3328_ii}.log`; `C_VERDICT` lines of `mem/results.txt`.", ""]
    for r in RUNS:
        L = R.logs[r]
        g = L["geom"]
        n = g["n"]
        m = g["face_max"]
        s2 = dict(L["phases"]["S2"]["items"])
        scr = {k: v for k, v in s2.items() if k.startswith("scratch_")}
        s_scr = sum(scr.values())
        cf = closed_form_s2(n, m, g["level_cells"], s_scr)
        assert cf == L["phases"]["S2"]["exact"], (r, cf)
        out += ["### %s: %dx%dx%d" % (r, g["nx"], g["ny"], g["nz"]), "",
                "n = %d, m = face_max = %d (m - n = %d), levels = %d, n_l (l >= 1) = %s, sum_{l>=1} n_l = %d."
                % (n, m, m - n, g["levels"], ",".join(map(str, g["level_cells"][1:])), sum(g["level_cells"][1:])),
                "S_scr items: %s; S_scr = %d." % (", ".join("%s=%d" % kv for kv in scr.items()), s_scr),
                "closed form = 8 [7 %d + 4 %d] + 32 (%d) + %d = %d; reported S2 exact = %d; difference %d."
                % (n, sum(g["level_cells"][1:]), m - n, s_scr, cf, L["phases"]["S2"]["exact"],
                   cf - L["phases"]["S2"]["exact"]),
                "S2 = %.4f B/cell = %.2f MiB (nominal audit level S+ 60.571 B/cell; excess %.4f B/cell: rule-7 capacity "
                "32 (m - n)/n = %.4f, scratch %.5f)." % (cf / n, cf / MIB, cf / n - 60.571428571, 32.0 * (m - n) / n, s_scr / n), "",
                "| phase | exact B | B/cell | MiB | cufft_work B | items (name:bytes) |", "|---|---:|---:|---:|---:|---|"]
        for p in PHASES:
            ph = L["phases"][p]
            its = ph["items"] if p != "S2" else [(k, v) for k, v in ph["items"] if not k.startswith(("mg_l", "scratch_"))]
            txt = ", ".join("%s:%d" % kv for kv in its) + (", mg_l*:... (%d bytes), scratch_*: %d" % (
                sum(v for k, v in ph["items"] if k.startswith("mg_l")), s_scr) if p == "S2" else "")
            out.append("| %s | %d | %.4f | %.2f | %d | %s |" % (p, ph["exact"], ph["exact"] / n, ph["exact"] / MIB, ph["cufft_work"], txt))
        out += ["", "S2 MG items: " + ", ".join("%s:%d" % kv for kv in L["phases"]["S2"]["items"] if kv[0].startswith("mg_l")), ""]
    cu = read(os.path.join(c, "controls", "long_box_cufft3328.log"))
    out += ["cuFFT work areas (`controls/long_box_cufft3328.log`): `%s`" % re.search(r"^CUFFT_SIZE .*$", cu, re.M).group(0), ""]
    # ---- D / E
    out += ["## D / E: `mem/results.txt`, `mem/*.log`, `mem/*.time`", "",
            "| run | exit | PID samples | PID peak MiB | device-wide peak MiB | E_max MiB | C MiB | peak - E_max - C MiB | host max RSS kB | wall (time -v) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in RUNS:
        tm = read(os.path.join(R.mem, r + ".time"))
        rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", tm).group(1))
        ex = re.search(r"Exit status: (\d+)", tm).group(1)
        wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", tm).group(1)
        emax = max(p["exact"] for p in R.logs[r]["phases"].values()) / MIB
        dev = max(int(l.split(",")[-1]) for l in read(os.path.join(R.mem, r + ".dev.csv")).splitlines() if l.strip())
        out.append("| %s | %s | %d | %d | %d | %.3f | %d | %.3f | %d | %s |" % (r, ex, len(R.apps[r]), R.peak[r], dev, emax,
                                                                                R.ctx_mib, R.peak[r] - emax - R.ctx_mib, rss, wall))
    out.append("")
    prss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", read(os.path.join(R.mem, "probe1.time"))).group(1))
    out += ["Host RSS per cell (RSS kB * 1024 / n); the context probe (`mem/probe1.time`) has RSS %d kB:" % prss, "",
            "| run | n | RSS B/cell | (RSS - probe RSS) B/cell |", "|---|---:|---:|---:|"]
    for r, rl in (("d512", "d512"), ("d512_bhatdev", "d512"), ("d3328_i", "d3328_i"), ("d3328_ii", "d3328_ii")):
        rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", read(os.path.join(R.mem, r + ".time"))).group(1))
        n = R.logs[rl]["geom"]["n"]
        out.append("| %s | %d | %.3f | %.3f |" % (r, n, rss * 1024.0 / n, (rss - prss) * 1024.0 / n))
    out.append("")
    sv = {}
    for r in ("d512", "d512_bhatdev"):
        lg = read(os.path.join(R.mem, r + ".log"))
        ph = parse_run_log(os.path.join(R.mem, r + ".log"))["phases"]
        sv[r] = (float(re.search(r"SETUP stage eps=\S+ built in ([\d.]+)s", lg).group(1)), ph["S2"]["seconds"], ph["S2"]["exact"])
    out += ["S vs S+ at 512^3 (`mem/d512.log` = S+ mapped b_hat, `mem/d512_bhatdev.log` = S, device b_hat):", "",
            "| variant | stage built in s | S2 (lean SF-19) s | S2 exact B |", "|---|---:|---:|---:|",
            "| S+ (mapped) | %.3f | %.6f | %d |" % sv["d512"], "| S (device) | %.3f | %.6f | %d |" % sv["d512_bhatdev"], "",
            "S2 time difference S+ - S = %.6f s; stage time difference = %.3f s; derived scaling of the S2 difference by cells to "
            "3328x384x384 (x%.4f) = %.2f s (derived, not measured)."
            % (sv["d512"][1] - sv["d512_bhatdev"][1], sv["d512"][0] - sv["d512_bhatdev"][0],
               R.logs["d3328_i"]["geom"]["n"] / R.logs["d512"]["geom"]["n"],
               (sv["d512"][1] - sv["d512_bhatdev"][1]) * R.logs["d3328_i"]["geom"]["n"] / R.logs["d512"]["geom"]["n"]), ""]
    for r in RUNS:
        lg = read(os.path.join(R.mem, r + ".log")).splitlines()
        lines = [l for l in lg if l.startswith(("SETUP   SF-19", "SETUP stage", "SETUP STAGE", "SETUP   inlet", "BACKFLOW", "TSTATUS", "TIMING", "STATUS"))]
        out += ["### %s log excerpts" % r, "", "```"] + [l[:240] for l in lines] + ["```",
                "S2 (lean SF-19) seconds = %.3f; stage phase seconds: %s." % (
                    R.logs[r]["phases"]["S2"]["seconds"], ", ".join("%s=%.3f" % (p, R.logs[r]["phases"][p]["seconds"]) for p in PHASES)), ""]
        npl = [l for l in lg if l.startswith("TPLANE")]
        nok = sorted({tokens(l)["n_ok"] for l in npl})
        out.append("TPLANE lines: %d, distinct n_ok values: %s." % (len(npl), ",".join(nok)))
        out.append("")
    out += ["Recorded lines of `mem/results.txt`:", "", "```"]
    out += [l[:300] for l in R.mres.splitlines() if l.startswith(("D_", "D3328", "E_VERDICT", "RECORDED", "C_CONTEXT", "PROBE"))]
    out += ["```", ""]
    # ---- F
    out += ["## F: `controls/long_box_full.log`", "", "```"]
    for l in read(os.path.join(c, "controls", "long_box_full.log")).splitlines():
        if re.match(r"(CHECK F[1-4]_|\[RECORD\] F[1-4]_|  F[1-4]_)", l) and "stage_ready" not in l:
            out.append(l[:260])
    out += ["```", ""]
    # ---- L6
    tsv = [l.split("\t") for l in read(os.path.join(c, "l6", "l6_timings.tsv")).splitlines()]
    col = {h: i for i, h in enumerate(tsv[0])}
    row = {t[0]: t for t in tsv[1:]}
    pl, tm = float(row["default"][col["plane_kernel_s"]]), float(row["default"][col["time_kernel_s"]])
    out += ["## L6: `sf40_campaign/l6/l6_timings.tsv`", "", "```", read(os.path.join(c, "l6", "l6_timings.tsv")).rstrip(), "```",
            "Derived per-pass ratios default / fused: plane %.4f, time %.4f (plane %.9f / %.9f s, time %.9f / %.9f s)."
            % (pl / float(row["fused"][col["plane_kernel_s"]]), tm / float(row["fused"][col["time_kernel_s"]]),
               pl, float(row["fused"][col["plane_kernel_s"]]), tm, float(row["fused"][col["time_kernel_s"]])), ""]
    # ---- Par2
    pr = read(os.path.join(c, "par2", "results.txt"))
    pk = {k: int(v) for k, v in re.findall(r"PAR2_PEAK (\w+): PID_PEAK .*?pid_peak_mib=(\d+)", pr)}
    nx, ny, nz = 2048, 256, 256
    brefs = read(os.path.join(R.root, "sf40_base_refs", "results.txt"))
    bpk = {k: int(v) for k, v in re.findall(r"PAR2_PEAK (par2_[ab]): PID_PEAK .*?pid_peak_mib=(\d+)", brefs)}
    out += ["## G: `par2/results.txt`, `sf40_base_refs/results.txt`", "",
            "PID peaks MiB (head job): %s; base references: %s."
            % (", ".join("%s=%d" % kv for kv in sorted(pk.items())), ", ".join("%s=%d" % kv for kv in sorted(bpk.items()))),
            "Config grid (apps/config_pipeline_par2.yaml at H1): %d x %d x %d = %d cells; 24 B/cell x cells = %d B = %.2f MiB."
            % (nx, ny, nz, nx * ny * nz, 24 * nx * ny * nz, 24.0 * nx * ny * nz / MIB), ""]
    base = int(re.search(r"PAR2_PEAK base_par2_a: PID_PEAK .*?pid_peak_mib=(\d+)", pr).group(1))
    head = int(re.search(r"PAR2_PEAK head_par2: PID_PEAK .*?pid_peak_mib=(\d+)", pr).group(1))
    out += ["base - head PID peak = %d - %d = %d MiB (expected 24 B/cell: %.2f MiB; difference %.2f MiB)."
            % (base, head, base - head, 24.0 * nx * ny * nz / MIB, base - head - 24.0 * nx * ny * nz / MIB), ""]
    # ---- ctest
    fr = read(os.path.join(c, "final", "results.txt"))
    out += ["## R / G byte items", "", "SF-27 trio (`final/results.txt`): %s; last verdict line `%s`."
            % (", ".join("%s %d" % (k, v) for k, v in sorted(
                {k: len(re.findall(r"^IDENTICAL trio:%s:" % k, fr, re.M)) for k in
                 ("config_pspta_small", "config_streamfunctions_homogeneous", "config_streamfunctions_continuation")}.items())),
               next(l for l in fr.splitlines() if l.startswith("TRIO_"))),
            "Par2 head vs base (`par2/results.txt`): IDENTICAL file items %d, other lines starting DIFF/MISSING: %d; verdict `%s`."
            % (len(re.findall(r"^IDENTICAL par2:head_vs_base", pr, re.M)), len(re.findall(r"^(DIFF|MISSING)", pr, re.M)),
               next(l for l in pr.splitlines() if l.startswith("PAR2_"))), ""]
    ct = read(os.path.join(c, "final", "ctest.log"))
    out += ["## R: `final/ctest.log`", "", "```"] + [l for l in ct.splitlines() if "tests passed" in l or "Total Test time" in l] + ["```",
            "Tests listed: %d. SF-37 wall (spec): 2901.55 s." % len(re.findall(r"Test +#\d+:", ct)), ""]
    return "\n".join(out) + "\n"


def run(raw, out):
    R = Raw(raw)
    os.makedirs(out, exist_ok=True)
    csv, md = feasibility_out(R)
    files = {"feasibility.csv": csv, "feasibility.md": md, "pid_plateaus.md": plateaus_md(R),
             "completeness.md": completeness_md(R), "summary.md": summary_md(R)}
    for k, v in files.items():
        with open(os.path.join(out, k), "w") as f:
            f.write(v)
    print("wrote %s" % ", ".join(sorted(files)))
    return 0


# ------------------------------------------------------------------------------------------ selftest
def selftest():
    # tokens / items
    assert tokens("MEM_PHASE phase=S2 exact_bytes=10 cufft_work=0") == {"phase": "S2", "exact_bytes": "10", "cufft_work": "0"}
    assert parse_items("a:1,b_c:22") == [("a", 1), ("b_c", 22)] and parse_items("-") == []
    # run-length encoding
    assert rle([("t0", 1), ("t1", 1), ("t2", 5), ("t3", 1)]) == [(1, 2, "t0"), (5, 1, "t2"), (1, 1, "t3")]
    # closed form, hand computed: n = 8, levels n_l = 1 (one coarse level), m = 9, scratch 100
    # 8 * (7*8 + 4*1) + 32 * (9 - 8) + 100 = 8*60 + 32 + 100 = 612
    assert closed_form_s2(8, 9, [8, 1], 100) == 612
    # coarse levels: 3328x384x384 values of the run logs (levels 7): n_1.. sum = 70,077,... checked against the raw in run()
    assert closed_form_s2(1, 1, [1], 0) == 56
    # multigrid levels on the transverse extent: 384 -> 192 -> 96 -> 48 -> 24 -> 12 -> 6 (7 levels), 64 -> 5 levels
    assert mg_levels_box(384) == 7 and mg_levels_box(64) == 5 and mg_levels_box(512) == 8
    # feasibility, hand computed: n = 2^20, S2 = 64 B/cell, C = 100 MiB, peak = 100 + 64 + 4 = 168 MiB, limit 1000 MiB, ny = 64
    #   b = 64, U = 168 - 100 - 64 = 4, cells_max = (1000 - 100 - 4) * 2^20 / 64 = 896 * 16384 = 14680064,
    #   mg levels 5 -> nx multiple of 16: floor(14680064 / 4096) = 3584 = 16 * 224
    f = feasibility(2 ** 20, 64 * 2 ** 20, 100, 168, 64, ((1.0, 8),), (1.0, 1.0, 1.0), limit_mib=1000)
    assert f["b"] == 64.0 and f["u"] == 4.0 and f["cells_max"] == 14680064.0 and f["nx_max"] == 3584
    x = f["rows"][0]
    # bdd cells = (1*8)^3 = 512 -> need 512 * 64 / 2^20 + 104 MiB; fits; lx_max = 3584 / 8 = 448; box 2^20/4096/8 = 32 lambda x 8 x 8
    assert x["bdd_cells"] == 512.0 and x["bdd_fits"] and x["lx_max_lambda"] == 448.0
    assert abs(x["bdd_need_mib"] - (512 * 64 / MIB + 104)) < 1e-12 and x["box_in_lambda"] == (32.0, 8.0, 8.0)
    # no-fit case: bdd of (400, 1, 1) lambda at lambda/h = 8 -> 400*8*8*8 = 204800 cells < cells_max still fits; (1e6, 1, 1) does not
    assert not feasibility(2 ** 20, 64 * 2 ** 20, 100, 168, 64, ((1.0, 8),), (1e6, 1.0, 1.0), limit_mib=1000)["rows"][0]["bdd_fits"]
    # completeness arithmetic: device_used - start - (exact - cufft)
    assert 1000 - 100 - (700 - 150) == 350
    print("SELFTEST OK note_tables")
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    return run(argv[1], argv[2])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
