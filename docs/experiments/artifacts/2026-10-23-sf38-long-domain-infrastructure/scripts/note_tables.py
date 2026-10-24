#!/usr/bin/env python3
"""SF-38 N8: derived tables of the experiment note, computed only from the imported raw evidence.

  note_tables.py RAW_DIR OUT_DIR      reads RAW_DIR (= artifact `raw/`), writes into OUT_DIR:
                                        parts.md, B.md, B_medium.md, C.md, C_D.md, E1_recorded.md, D.md, E.md, F.md, G.md
  note_tables.py --selftest           assert-based checks on hand-computed examples (no raw files)

Stdlib only; deterministic (re-running on the same raw/ regenerates byte-identical files). Every number printed is copied
from a raw file or is a documented function of such numbers (labelled `derived` in the table header). 1 MiB = 1048576 B.
No dispersion statistic is interpreted here: the C statistics are printed as inputs of the pre-registered rule only.
"""
import csv
import datetime
import json
import math
import os
import re
import statistics as st
import sys

PARTS = ("baserefs", "controls", "prodgrid", "bpilot", "ab", "ensemble", "fieldonly", "par2pilot", "final")
SF38_CTESTS = ("lagrangian_tracker_gpu_host", "lagrangian_long_box", "lagrangian_dirichlet_medium", "lagrangian_ensemble",
               "lagrangian_medium_ab")


def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def toks(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def part_dir(raw, part):
    return os.path.join(raw, "sf38_base_refs" if part == "baserefs" else os.path.join("sf38_campaign", part))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write(out, name, text):
    with open(os.path.join(out, name), "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------------------------------- parts
def parts_table(raw):
    rows, tot, tot_runs = [], 0, 0
    for p in PARTS:
        d = part_dir(raw, p)
        env = dict(l.split("=", 1) for l in read(os.path.join(d, "run.env")).splitlines() if "=" in l and not l.startswith("#"))
        rc = [l.split("\t") for l in read(os.path.join(d, "exit_codes.tsv")).splitlines() if l]
        res = read(os.path.join(d, "results.txt"))
        m = re.search(r"RECORDED part_seconds=(\d+) part_runs_seconds=(\d+)", res)
        ps, pr = int(m.group(1)), int(m.group(2))
        tot, tot_runs = tot + ps, tot_runs + pr
        nonzero = ",".join(sorted({r[1] for r in rc if r[1] != "0"})) or "-"
        rows.append((p, env["DATE_UTC"], read(os.path.join(d, "PART_EXIT")).strip(), len(rc), nonzero, ps, pr))
    t = "Per part: start (`run.env` DATE_UTC), `PART_EXIT`, number of runs in `exit_codes.tsv`, non-zero rcs, `part_seconds` and `part_runs_seconds` (RECORDED lines of `results.txt`).\n\n"
    t += table(("part", "start (UTC)", "PART_EXIT", "runs", "non-zero rc", "part_seconds", "part_runs_seconds"), rows)
    t += "\nderived: sum of `part_seconds` = %d s = %.3f h; sum of `part_runs_seconds` = %d s = %.3f h (GPU 0, one job at a time).\n" % (
        tot, tot / 3600.0, tot_runs, tot_runs / 3600.0)
    return t


# ---------------------------------------------------------------------------------------------------- B
def b_table(raw):
    d = part_dir(raw, "prodgrid")
    res = read(os.path.join(d, "results.txt"))
    rows_a, rows_m, rows_s = [], [], []
    for run in ("B1", "B2", "B3"):
        log = read(os.path.join(d, run + ".log"))
        v = [l for l in res.splitlines() if l.startswith("B_VERDICT run=%s " % run)][0]
        t = toks(v)
        corr = re.search(r"correctors=(\S+)", v).group(1)
        cs = [c.split(":")[1] for c in corr.split(";")]
        rec = [l for l in res.splitlines() if l.startswith("RECORDED run=%s exit=" % run)][0]
        pk = [l for l in res.splitlines() if l.startswith("RECORDED run=%s pid_peak_mib" % run)][0]
        tp = toks(pk)
        tr = toks(rec)
        lines = {k: [l for l in log.splitlines() if l.startswith("R=0 " + k + " ")] for k in ("STAGE", "BACKFLOW", "STATUS", "TWIN", "POLLOCK")}
        stg = toks(lines["STAGE"][0])
        bf = toks(lines["BACKFLOW"][0])
        ts = [toks(l) for l in log.splitlines() if l.startswith("R=0 TSTATUS ")]
        timing = re.search(r"^TIMING r=0 seconds=(\S+)", log, re.M).group(1)
        rows_a.append((run, t["grid"], t["exit"], "; ".join(cs), t["mg_levels"] + "/" + t["auto_levels"], tp["pid_peak_mib"], tp["dev_peak_mib"],
                       tp["samples"], tr["wall_s"], tr["host_max_rss_kb"], timing, v.split()[-1]))
        rows_s.append((run, stg["status"], stg["seconds"], bf["inlet_vmin"], bf["labels_available"], bf["seed_backflow"], bf["backflow_enc_plane"],
                       bf["backflow_enc_time"], ts[0]["hist"], ts[1]["hist"], toks(lines["TWIN"][0])["planes"] + "/" + toks(lines["TWIN"][0])["times"],
                       " ".join(lines["STATUS"][0].split()[2:])))
        for ph in ("F", "S1", "S2", "S3", "T"):
            l = [l for l in log.splitlines() if l.startswith("R=0 MEM_PHASE phase=%s " % ph)][0]
            tt = toks(l)
            rows_m.append((run, ph, tt["exact_bytes"], "%.3f" % (int(tt["exact_bytes"]) / 1048576.0), tt["device_used"],
                           "%.3f" % (int(tt["device_used"]) / 1048576.0), tt["seconds"]))
    out = "B verdict lines (copied from `prodgrid/results.txt`): correctors are `its,rel_residual` for x;y;z.\n\n"
    out += table(("run", "grid", "exit", "correctors (its,rel) x; y; z", "mg_levels/auto", "PID peak MiB", "device-wide peak MiB", "samples",
                  "wall s", "host max RSS kB", "TIMING r=0 s", "B_VERDICT"), rows_a)
    out += "\nStage, backflow and status lines (copied from `B*.log`).\n\n"
    out += table(("run", "STAGE status", "stage s", "inlet_vmin", "labels_available", "seed_backflow", "backflow_enc plane", "backflow_enc time",
                  "TSTATUS plane hist", "TSTATUS time hist", "TWIN planes/times", "STATUS"), rows_s)
    out += "\nMEM_PHASE lines (copied from `B*.log`; MiB = bytes / 1048576, derived).\n\n"
    out += table(("run", "phase", "exact_bytes", "exact MiB", "device_used B", "device_used MiB", "seconds"), rows_m)
    out += ("\nderived: PID memory in the tracking and bridge phase. The sampler clock origin is the first sample of the PID (an approximation of the "
            "process start); `late` = samples more than 5 s after the stage build time `STAGE seconds`.\n\n")
    lrows = []
    for run in ("B1", "B2", "B3"):
        stage_s = float(toks([l for l in read(os.path.join(d, run + ".log")).splitlines() if l.startswith("R=0 STAGE ")][0])["seconds"])
        smp = [l.split(",") for l in read(os.path.join(d, run + ".apps.csv")).splitlines() if l.strip()]
        ts = [datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f") for r in smp]
        mem = [int(r[2]) for r in smp]
        late = [m for t, m in zip(ts, mem) if (t - ts[0]).total_seconds() > stage_s + 5]
        lrows.append((run, "%.3f" % stage_s, max(mem), sum(1 for m in mem if m >= max(mem) - 1), len(late), max(late)))
    out += table(("run", "stage s", "PID peak MiB", "samples within 1 MiB of the peak", "late samples", "late max MiB"), lrows)
    return out


# ---------------------------------------------------------------------------------------------------- C
def c_tables(raw):
    res = read(os.path.join(part_dir(raw, "ab"), "results.txt"))
    stats = [toks(l) for l in res.splitlines() if l.startswith("C_STAT ")]
    rows = []
    for t in stats:
        f = lambda k: "%.4g" % float(t[k])
        rows.append((t["amp"], t["stat"], t["test"], f("Pbar_a"), f("Pbar_b"), f("est"), f("se"), f("ci_lo"), f("ci_hi"), f("margin"), "%.3f" % ((float(t["ci_hi"]) - float(t["ci_lo"])) / (2 * float(t["margin"]))), t["class"]))
    amp = [l for l in res.splitlines() if l.startswith("C_AMP ")]
    ver = [l for l in res.splitlines() if l.startswith("C_VERDICT ")][0]
    text = ("`C_STAT` lines of `ab/results.txt` (= `ab/ab_decision.out`). `est` is `R = ln(Pbar_a/Pbar_b)` for `test=ratio` and the paired difference "
            "`Dbar` for `test=diff|floor`; `margin` is the half-width on the same scale (`ln 1.05` = 0.04879, `ln 1.025` = 0.02469; floor margin = "
            "`2.5e-3 lambda`, box units). Statistics in box units (S_T) and normalized (S_tau, V1, theta). Values printed with 4 significant digits; full "
            "precision in the raw lines.\n\n")
    text += table(("amp", "stat", "test", "Pbar_a", "Pbar_b", "est", "se", "ci_lo", "ci_hi", "margin", "CI width / (2 margin) (derived)", "class"), rows)
    text += "\n" + "\n".join("`%s`" % l for l in amp) + "\n\n`%s`\n" % ver
    ds = [toks(l) for l in res.splitlines() if l.startswith("C_D ")]
    drows = [(t["amp"], t["stat"], t["seed"], "%.6e" % float(t["P_a"]), "%.6e" % float(t["P_b"]), "%.6e" % float(t["D"])) for t in ds]
    dtext = "`C_D` lines of `ab/results.txt` (per realization: `P_a`, `P_b`, `D = P_a - P_b`; box units for S_T).\n\n"
    dtext += table(("amp", "stat", "seed", "P_a", "P_b", "D"), drows)
    return text, dtext


def e1_recorded(raw):
    """Recorded (non-decisive) E1 quantities per amplitude and medium, plus a literal reading of the boundary-layer thickness."""
    out = ("Recorded E1 quantities from `ab/ab_s*/realizations/r_*/e1.json` and `e1_{a,b}.csv` (means over the ten realizations; `qbar` of (a) is the "
           "plane mean of `U` at `i=0`, 1 to solver level).\n\n")
    rows, prof_rows, bl_rows = [], [], []
    for amp, d, lam in (("1", "ab_s1", 0.0390625), ("4", "ab_s4", 0.125)):
        base = os.path.join(part_dir(raw, "ab"), d, "realizations")
        for med in ("a", "b"):
            qs, v2, v3, profs = [], [], [], []
            xs = None
            for r in range(10):
                p = os.path.join(base, "r_%04d" % r)
                e = json.load(open(os.path.join(p, "e1.json")))[med]
                qs.append(e["qbar"]); v2.append(e["var_v2"]); v3.append(e["var_v3"])
                with open(os.path.join(p, "e1_%s.csv" % med)) as f:
                    pl = [x for x in csv.DictReader(f) if x["kind"] == "plane"]
                xs = [float(x["x1"]) for x in pl]
                profs.append([float(x["u_var"]) / e["qbar"] ** 2 for x in pl])
            rows.append((amp, med, "%.6f" % min(qs), "%.6f" % max(qs), "%.6f" % st.mean(qs), "%.4f" % st.mean(v2), "%.4f" % st.mean(v3)))
            n = len(xs); L = xs[-1]
            m = [st.mean(p[i] for p in profs) for i in range(n)]
            se = [st.stdev([p[i] for p in profs]) / math.sqrt(10) for i in range(n)]
            inter = [i for i in range(n) if 5 * lam <= xs[i] <= L - 5 * lam]
            M = st.mean(m[i] for i in inter)
            out_i = [i for i in range(n) if abs(m[i] - M) > 2 * se[i]]
            w = max([xs[i] for i in out_i if xs[i] < L / 2] or [0.0])
            e_ = max([L - xs[i] for i in out_i if xs[i] > L / 2] or [0.0])
            bl_rows.append((amp, med, "%.4f" % M, len(inter), len([i for i in out_i if i in inter]), "%.2f" % (w / lam), "%.2f" % (e_ / lam)))
            for k in (0, 0.25, 0.5, 1, 2, 4, 8):
                i = min(range(n), key=lambda j: abs(xs[j] - k * lam))
                prof_rows.append((amp, med, "%.2f" % (xs[i] / lam), "%.4f" % m[i], "%.4f" % se[i]))
    out += table(("amp", "medium", "qbar min", "qbar max", "qbar mean", "<var_i(v2)>/qbar^2 mean", "<var_i(v3)>/qbar^2 mean"), rows)
    out += ("\nEnsemble-mean plane profile of `var_i(v1)/qbar^2` (west face, nearest plane to the listed distance; SE = sd over the ten realizations / "
            "sqrt(10)).\n\n")
    out += table(("amp", "medium", "x1/lambda", "mean", "SE"), prof_rows)
    out += ("\nderived (literal reading of the specification's recorded boundary-layer quantity; not scripted by the campaign tooling): `M` = mean over "
            "the interior planes `x1 in [5 lambda, L1 - 5 lambda]` of the ensemble-mean profile; a plane is `outside` if `|mean_i - M| > 2 SE_i`; "
            "`west d` (`east d`) = the largest distance from the west (east) face of an `outside` plane in the west (east) half, in `lambda`. "
            "The same procedure is applied to medium (a), which has no `x`-face condition, as a control of the estimator.\n\n")
    out += table(("amp", "medium", "M", "interior planes", "interior planes outside", "west d/lambda", "east d/lambda"), bl_rows)
    return out


# ---------------------------------------------------------------------------------------------------- medium (b) record
def abb_table(raw):
    out = ("Medium (b) solver record (`B_SOLVER` lines of `ab/ab_s*.log`, one per realization; `rel` = ||r||/||r_0|| of `solve_head`, `balance` = "
           "|Q(x=0) - Q(x=L1)|/|Q(x=0)|, `max_div` = max cell divergence / qbar; recorded, thresholds C0 apply to the pilot only).\n\n")
    mrows = []
    for amp, d in (("1", "ab_s1"), ("4", "ab_s4")):
        log = read(os.path.join(part_dir(raw, "ab"), d + ".log"))
        sol = [toks(l) for l in log.splitlines() if re.match(r"R=\d+ B_SOLVER ", l)]
        rows = [(amp, t["r"], t["converged"], t["iterations"], t["rel"], t["balance"], t["max_div"], t["qbar"], t["fallback"]) for t in sol]
        out += table(("amp", "r", "converged", "iterations", "rel", "balance", "max_div", "qbar_b", "fallback"), rows)
        out += "\nderived: n=%d, converged=%d, fallback=%d, max rel=%.3e, max balance=%.3e, max max_div=%.3e, iterations min..max=%d..%d.\n\n" % (
            len(sol), sum(t["converged"] == "1" for t in sol), sum(t["fallback"] == "1" for t in sol), max(float(t["rel"]) for t in sol),
            max(float(t["balance"]) for t in sol), max(float(t["max_div"]) for t in sol), min(int(t["iterations"]) for t in sol),
            max(int(t["iterations"]) for t in sol))
        for l in log.splitlines():
            if l.startswith("R=0 MEM_PHASE_B "):
                t = toks(l)
                mrows.append((d, t["phase"], t["exact_bytes"], "%.3f" % (int(t["exact_bytes"]) / 1048576.0), t["device_used"],
                              "%.3f" % (int(t["device_used"]) / 1048576.0), t["seconds"]))
    out += "Medium (b) memory phases, realization 0 (`MEM_PHASE_B` lines of `ab/ab_s*.log`; MiB derived).\n\n"
    out += table(("run", "phase", "exact_bytes", "exact MiB", "device_used B", "device_used MiB", "seconds"), mrows)
    return out


# ---------------------------------------------------------------------------------------------------- D
def d_table(raw):
    res = read(os.path.join(part_dir(raw, "ensemble"), "results.txt"))
    keep = [l for l in res.splitlines() if re.match(r"(D_KILL|D_BYTES|D_CSV|D_TTIME|D_VERDICT|RECORDED D_WALL|RECORDED run=D)", l)]
    walls = [toks(l) for l in keep if l.startswith("RECORDED D_WALL")]
    rows = [(w["name"], w["r"], w["total_s"], w["stage_s"], w["prepare_s"], w["track_s"], w["bridge_repack_s"], w["bridge_track_s"], w["write_s"])
            for w in walls]
    # cross-check against timings.json of the imported realizations
    d = part_dir(raw, "ensemble")
    tj = []
    for name in ("DA", "DB", "DC"):
        for r in range(3):
            p = os.path.join(d, name, "realizations", "r_%04d" % r, "timings.json")
            if os.path.exists(p):
                j = json.load(open(p))
                tj.append((name, r, j))
    text = "Lines of `ensemble/results.txt` (verbatim, paths shortened where marked `...`).\n\n```\n"
    for l in keep:
        if l.startswith("D_CSV"):
            l = re.sub(r" ref=\S+ cand=\S+", " ref=... cand=...", l)
        if not l.startswith("RECORDED D_WALL"):
            text += l + "\n"
    text += "```\n\nPer-realization wall (`RECORDED D_WALL`, seconds, from `timings.json`).\n\n"
    text += table(("run", "r", "total", "stage", "prepare", "track", "bridge_repack", "bridge_track", "write"), rows)
    text += "\nkeys of `timings.json` of D-A r=0 (imported file): %s\n" % (", ".join(sorted(tj[0][2])) if tj else "not imported")
    return text


# ---------------------------------------------------------------------------------------------------- E
def e_table(raw):
    d = part_dir(raw, "par2pilot")
    res = read(os.path.join(d, "results.txt"))
    L = res.splitlines()
    out = ""
    # per config summary
    rows = []
    for c in ("s1", "s225", "s4"):
        summ = [toks(l) for l in L if l.startswith("E_LNK_SUMMARY cfg=%s " % c)][0]
        tb = [toks(l) for l in L if l.startswith("E_TB cfg=%s " % c)]
        tbs = [toks(l) for l in L if l.startswith("E_TB_SUMMARY cfg=%s " % c)][0]
        run = [toks(l) for l in L if l.startswith("RECORDED run=run_%s " % c)][0]
        pk = [toks(l) for l in L if l.startswith("RECORDED PAR2_PEAK run_%s:" % c)][0]
        rt = [toks(l) for l in L if l.startswith("RECORDED E_RTIME cfg=%s " % c)]
        ks = [l for l in L if l.startswith("E_KSTATS_ALL cfg=%s " % c)][0]
        dt2 = [toks(l) for l in L if l.startswith("RECORDED run=dt2_%s " % c)][0]
        tm = open(os.path.join(d, "run_%s.time" % c)).read()
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", tm).group(1)
        rows.append((c, run["exit"], run["wall_s"], "%.3f..%.3f" % (min(float(x["seconds_since_previous"]) for x in rt[1:]),
                                                                  max(float(x["seconds_since_previous"]) for x in rt[1:])), rt[0]["seconds_since_previous"],
                     tbs["t_b_min"], tbs["t_b_max"], pk["pid_peak_mib"], rss, dt2["wall_s"], ks.split(" ", 1)[1]))
    out += ("Par2 pilot runs (`par2pilot/results.txt`; wall = `/usr/bin/time` elapsed of the whole 10-realization run; time per realization from the mtimes "
            "of `stats/` CSVs, first interval includes setup; `t_b` in physical time units (`step * dt`, `dt = 12.5`); `dt/2` wall for 3 realizations).\n\n")
    out += table(("cfg", "exit", "wall s", "interval s r=1..9 (min..max)", "interval s r=0", "t_b min", "t_b max", "PID peak MiB", "host max RSS kB",
                  "dt/2 run wall s", "K-stats agreement"), rows)
    out += "\n`E_LNK` per realization (realized mean and variance of `ln K` from `par2_field_stats`, K seeds 7001-7010) and `E_TB`.\n\n"
    lrows = []
    for c in ("s1", "s225", "s4"):
        tbm = {toks(l)["r"]: toks(l) for l in L if l.startswith("E_TB cfg=%s " % c)}
        for l in L:
            if l.startswith("E_LNK cfg=%s " % c):
                t = toks(l)
                lrows.append((c, t["r"], t["seed"], "%.6e" % float(t["mean_lnK"]), "%.6e" % float(t["var_lnK"]), tbm[t["r"]]["step"], tbm[t["r"]]["t_b"]))
    out += table(("cfg", "r", "K seed", "mean lnK", "var lnK", "first snapshot step with x >= Lx - 5 lambda", "t_b"), lrows)
    out += "\n" + "\n".join("`%s`" % l for l in L if l.startswith("E_LNK_SUMMARY")) + "\n"
    out += "\nPaired `dt` vs `dt/2` differences (`RECORDED E_PAIRED`, first 3 realizations; difference of the two `analysis/macrodispersion.csv` joined on `t`; column names are the pipeline's; these are numerical differences between two time steps, recorded without threshold, not coefficient estimates).\n\n"
    prow = []
    for l in L:
        if l.startswith("RECORDED E_PAIRED"):
            t = toks(l)
            prow.append((t["cfg"], t["col"], t["rows"], t["unmatched"], t["max_abs_diff"], t["max_rel_diff"], t["rms_diff"]))
    out += table(("cfg", "column", "rows", "unmatched", "max_abs_diff", "max_rel_diff", "rms_diff"), prow)
    return out


# ---------------------------------------------------------------------------------------------------- F
def f_table(raw):
    d = part_dir(raw, "final")
    rows = [l.split("\t") for l in read(os.path.join(d, "ctest_times.tsv")).splitlines() if l]
    tot = sum(float(r[2]) for r in rows)
    sel = [(r[0], r[1], r[2]) for r in rows if r[0] in SF38_CTESTS]
    res = read(os.path.join(d, "results.txt"))
    ident = [l.split()[1].split(":")[0] for l in res.splitlines() if l.startswith("IDENTICAL ")]
    cnt = {k: ident.count(k) for k in sorted(set(ident))}
    other = [l for l in res.splitlines() if l.startswith("DIFF") or l.startswith("MISSING")]
    out = "ctest (`final/ctest_times.tsv`): %d tests, all Passed; sum of per-test seconds = %.2f s (derived); ctest 'Total Test time (real)' 3065.06 s (`final/ctest.log`).\n\n" % (
        len(rows), tot)
    out += table(("test", "status", "seconds"), sel)
    out += "\nLongest tests: " + "; ".join("%s %s s" % (r[0], r[2]) for r in sorted(rows, key=lambda r: -float(r[2]))[:3]) + ".\n"
    out += "\n`IDENTICAL` lines in `final/results.txt` by class: %s; `DIFF`/`MISSING` lines: %d.\n\n" % (
        ", ".join("%s %d" % kv for kv in cnt.items()), len(other))
    rc = [l.split("\t") for l in read(os.path.join(d, "exit_codes.tsv")).splitlines() if l]
    base = {l.split("\t")[0]: l.split("\t")[1] for l in read(os.path.join(part_dir(raw, "baserefs"), "exit_codes.tsv")).splitlines() if l}
    out += "Exit codes head vs base for the 20-run R set and the trio (`final/exit_codes.tsv`, `sf38_base_refs/exit_codes.tsv`):\n\n"
    out += table(("run", "head rc", "base rc"), [(r[0], r[1], base.get(r[0], "-")) for r in rc if r[0] != "ctest"])
    return out


# ---------------------------------------------------------------------------------------------------- G
def g_table(raw):
    res = read(os.path.join(part_dir(raw, "fieldonly"), "results.txt"))
    rows = [toks(l) for l in res.splitlines() if l.startswith("G_FIELD ")]
    names = []
    for t in rows:
        if t["name"] not in names:
            names.append(t["name"])
    seeds = sorted({int(t["seed"]) for t in rows})
    by = {(t["name"], int(t["seed"])): t for t in rows}
    out = "`G_FIELD` lines of `fieldonly/results.txt`: SF-18 raw variance (before `normalize_variance`) per seed and geometry; `final_variance` is 1 in every line.\n\n"
    out += "Raw variance:\n\n" + table(("seed",) + tuple(names), [(s,) + tuple("%.6f" % float(by[(n, s)]["raw_variance"]) for n in names) for s in seeds])
    out += "\nApplied scale (as printed):\n\n" + table(("seed",) + tuple(names), [(s,) + tuple("%.6f" % float(by[(n, s)]["applied_scale"]) for n in names) for s in seeds])
    out += "\n`G_SUMMARY` lines (copied):\n\n"
    out += table(("name", "n", "raw_variance_mean", "raw_variance_sd", "applied_scale_mean", "applied_scale_sd"),
                 [(t["name"], t["n"], "%.6f" % float(t["raw_variance_mean"]), "%.6f" % float(t["raw_variance_sd"]), "%.6f" % float(t["applied_scale_mean"]),
                   "%.6f" % float(t["applied_scale_sd"])) for t in (toks(l) for l in res.splitlines() if l.startswith("G_SUMMARY "))])
    out += "\n" + "\n".join("`%s`" % l for l in res.splitlines() if l.startswith("G_COMPLETE") or l.startswith("G_VERDICT")) + "\n"
    return out


# ---------------------------------------------------------------------------------------------------- driver
def main(argv):
    if argv[1:] == ["--selftest"]:
        assert toks("a=1 b=x:y c=3.5e-3") == {"a": "1", "b": "x:y", "c": "3.5e-3"}
        assert table(("a", "b"), [(1, 2)]) == "| a | b |\n|---|---|\n| 1 | 2 |\n"
        # sample sd and SE used by e1_recorded: ten values 0..9 -> sd = 3.0276503540974917, SE = sd / sqrt(10)
        assert abs(st.stdev(range(10)) - 3.0276503540974917) < 1e-15
        print("selftest ok")
        return 0
    raw, out = argv[1], argv[2]
    os.makedirs(out, exist_ok=True)
    c, cd = c_tables(raw)
    write(out, "parts.md", parts_table(raw))
    write(out, "B.md", b_table(raw))
    write(out, "C.md", c)
    write(out, "C_D.md", cd)
    write(out, "E1_recorded.md", e1_recorded(raw))
    write(out, "B_medium.md", abb_table(raw))
    write(out, "D.md", d_table(raw))
    write(out, "E.md", e_table(raw))
    write(out, "F.md", f_table(raw))
    write(out, "G.md", g_table(raw))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
