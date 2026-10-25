#!/usr/bin/env python3
"""SF-41 N8: derived tables of the experiment note, computed only from the imported raw evidence.

  note_tables.py RAW_DIR OUT_DIR      reads RAW_DIR (= artifact `raw/`), writes into OUT_DIR:
                                        parts.md, A.md, B.md, C.md, D.md, E.md, F.md, G.md, H.md
  note_tables.py --selftest           assert-based checks on hand-computed examples (no raw files)

Stdlib only; deterministic (re-running on the same raw/ regenerates byte-identical files). Every number printed is copied
from a raw file or is a documented function of such numbers (labelled `derived`). 1 MiB = 1048576 B.
No dispersion statistic is interpreted here: D_P / sigma_T^2 / theta are printed as recorded inputs (Rule 1, lester-2023-key-claims.md).
Nothing here reads outside RAW_DIR, runs a campaign script, or uses the network. analysis/bootstrap_blocks_f.* is not touched.
"""
import datetime
import math
import os
import re
import statistics as st
import sys

HEAD_PARTS = ("controls", "cross", "prodgrid", "evladder", "abrerun", "final")
MIB = 1048576.0


# ---------------------------------------------------------------------------------------------------- helpers
def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def kv(line):
    return dict(re.findall(r"(?:^|\s)([\w().,]+)=(\S+)", line))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write(out, name, text):
    with open(os.path.join(out, name), "w") as f:
        f.write(text)


def fl(x):
    return float(x)


def e3(x):
    return "%.3e" % x


def mib(b):
    return "%.3f" % (int(b) / MIB)


def order(e_coarse, e_fine):
    return math.log2(e_coarse / e_fine)


def lines_starting(text, prefix):
    return [l for l in text.splitlines() if l.startswith(prefix)]


def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def tsv(path):
    return [l.split("\t") for l in read(path).splitlines() if l.strip()]


def part_dir(raw, part):
    return os.path.join(raw, "sf41_base_refs" if part == "baserefs" else os.path.join("sf41_campaign", part))


def env_of(path):
    return dict(l.split("=", 1) for l in read(path).splitlines() if "=" in l and not l.startswith("#"))


def group_rows(lines, keys=("name",)):
    """Group `A_CHECK`/`C_CHECK` lines by name: count, extreme value, gate, verdicts. Worst = max value except *_order_ge_* (min)."""
    g = {}
    for l in lines:
        t = kv(l)
        g.setdefault(t["name"], []).append(t)
    rows = []
    for name, ts_ in g.items():
        is_min = "_order_ge_" in name
        vals = [(fl(t["value"]), t) for t in ts_]
        v, w = (min if is_min else max)(vals, key=lambda p: p[0])
        gates = sorted({t.get("gate", "") for t in ts_})
        verd = sorted({t.get("verdict", "") for t in ts_})
        where = w.get("box", w.get("grid", "")) + "/" + w.get("side", w.get("path", ""))
        rows.append((name, len(ts_), "min" if is_min else "max", e3(v), where, ",".join(gates), ",".join(verd), margin(v, gates[0], is_min)))
    return rows


def margin(v, gate, is_min):
    """derived: gate / value for upper-bound gates, value / gate for order gates; '-' when either is zero or not numeric."""
    try:
        g = float(gate.lstrip("<>="))
    except ValueError:
        return "-"
    if v == 0 or g == 0 or not (gate.startswith("<=") or gate[:1].isdigit()):
        return "-"
    return "%.2f" % ((v / g) if is_min else (g / v))


# ---------------------------------------------------------------------------------------------------- parts
def job_rows(raw):
    rows = []
    jl = os.path.join(raw, "job_logs")
    for f in sorted(os.listdir(jl)):
        name = f[:-4]
        text = read(os.path.join(jl, f))
        s = re.search(r"^\[(\S+)\] job=\S+ session=\S+ started", text, re.M).group(1)
        m = re.search(r"^\[(\S+)\] job=\S+ finished exit=(\S*)", text, re.M)
        tree = re.search(r"^TREE=(.*)$", text, re.M)
        st_env = env_of(os.path.join(raw, "job_status", name + ".env"))
        fin = m.group(1) if m else "-"
        win = int((ts(fin) - ts(s)).total_seconds()) if m else "-"
        rows.append((name, "no (pre-check)" if name.startswith("sf41-pre-") else "yes", s, fin, win, m.group(2) if m else "-",
                     st_env.get("STATE", "-") + "/" + st_env.get("EXIT_CODE", "-"), tree.group(1)[:70] if tree else "-"))
    return rows


def part_row(raw, p):
    d = part_dir(raw, p)
    env = env_of(os.path.join(d, "run.env"))
    rc = tsv(os.path.join(d, "exit_codes.tsv"))
    res = read(os.path.join(d, "results.txt"))
    m = re.search(r"RECORDED part_seconds=(\d+) part_runs_seconds=(\d+)", res)
    nz = ",".join(sorted({r[1] for r in rc if r[1] != "0"})) or "-"
    return (p, env["DATE_UTC"], read(os.path.join(d, "PART_EXIT")).strip(), len(rc), nz, int(m.group(1)), int(m.group(2)),
            env.get("SF41_COMMIT", "-")[:7], env.get("BINARIES_CHECK", "-"), env)


def parts_md(raw):
    out = "#### Jobs (every file of `raw/job_logs/`, status from `raw/job_status/`; `window` = finished minus started, derived)\n\n"
    out += table(("job", "evidence", "started (UTC)", "finished (UTC)", "window s", "finished exit", "status STATE/EXIT_CODE", "TREE echo"), job_rows(raw))
    rows, tot, tot_runs = [], 0, 0
    for p in ("baserefs", "build") + HEAD_PARTS:
        r = part_row(raw, p)
        rows.append(r[:9])
        tot, tot_runs = tot + r[5], tot_runs + r[6]
    sup = os.path.join(raw, "sf41_campaign_superseded", "build_attempt1_tmpfull")
    sr = read(os.path.join(sup, "results.txt"))
    sm = re.search(r"RECORDED part_seconds=(\d+) part_runs_seconds=(\d+)", sr)
    srow = ("build attempt 1 (superseded, not counted)", env_of(os.path.join(sup, "run.env"))["DATE_UTC"], read(os.path.join(sup, "PART_EXIT")).strip(),
            len(tsv(os.path.join(sup, "exit_codes.tsv"))), ",".join(sorted({r[1] for r in tsv(os.path.join(sup, "exit_codes.tsv")) if r[1] != "0"})) or "-",
            int(sm.group(1)), int(sm.group(2)), "269c658", "-")
    rows.append(srow)
    out += "\n#### Parts (start = `run.env` DATE_UTC; PART_EXIT; runs = rows of `exit_codes.tsv`; part_seconds / part_runs_seconds = `RECORDED` line of `results.txt`)\n\n"
    out += table(("part", "start (UTC)", "PART_EXIT", "runs", "non-zero rc", "part_seconds", "part_runs_seconds", "SF41_COMMIT", "BINARIES_CHECK"), rows)
    bb = read(os.path.join(raw, "job_logs", "sf41-basebuild.log"))
    b0, b1 = re.search(r"^\[(\S+)\] job=sf41-basebuild .* started", bb, re.M).group(1), re.search(r"^\[(\S+)\] job=sf41-basebuild finished", bb, re.M).group(1)
    bwin = int((ts(b1) - ts(b0)).total_seconds())
    out += ("\nderived (GPU-hours = wall hours: one GPU, one job at a time): sum of `part_seconds` over baserefs + build (attempt 2) + the six head parts = %d s = %.3f h; "
            "sum of `part_runs_seconds` = %d s = %.3f h; adding the base build job window (`sf41-basebuild`, %s to %s = %d s) gives %d s = %.3f h; the superseded build attempt 1 "
            "(%d s) is not counted.\n") % (tot, tot / 3600.0, tot_runs, tot_runs / 3600.0, b0, b1, bwin, tot + bwin, (tot + bwin) / 3600.0, srow[5])
    gh = [(r[0], "%.3f" % (r[5] / 3600.0)) for r in rows]
    out += "\n" + table(("part", "GPU-hours (part_seconds / 3600, derived)"), gh)
    # per-run walls
    wr = []
    for p in ("controls", "cross", "prodgrid", "abrerun", "final"):
        for r in tsv(os.path.join(part_dir(raw, p), "walls.tsv")):
            wr.append((p, r[0], r[1], r[2]))
    out += "\n#### Per-run wall seconds (`walls.tsv`; the 24 runs of `evladder` are summarised in F.md)\n\n" + table(("part", "run", "wall s", "end (UTC)"), wr)
    # binaries and clock offset
    ce = env_of(os.path.join(part_dir(raw, "controls"), "run.env"))
    be = env_of(os.path.join(part_dir(raw, "baserefs"), "run.env"))
    brows = [(k[8:], ce[k], be.get(k, "-")) for k in sorted(ce) if k.startswith("BIN_MD5_")]
    bmd5 = {l.split()[1]: l.split()[0] for l in read(os.path.join(part_dir(raw, "build"), "binaries.md5")).splitlines() if l.strip()}
    bmd5_ok = all(bmd5.get(r[0]) == r[1] for r in brows)
    out += "\n#### Binary md5 (head: `controls/run.env`, identical in every head part: %s; base: `sf41_base_refs/run.env`)\n\n" % (
        "yes" if all({k: v for k, v in part_row(raw, p)[9].items() if k.startswith("BIN_MD5_")} == {k: v for k, v in ce.items() if k.startswith("BIN_MD5_")}
                     for p in HEAD_PARTS) else "NO")
    out += table(("binary", "head md5", "base md5"), brows)
    out += "\n`build/binaries.md5` (attempt-2 build part) lists %d binaries; its md5 for the %d binaries above %s the `BIN_MD5_*` of the head `run.env`.\n" % (
        len(bmd5), len(brows), "equal" if bmd5_ok else "DO NOT equal")
    banner = [l for l in read(os.path.join(part_dir(raw, "controls"), "run.env")).splitlines() if l.startswith("# nvidia-smi banner: ") and "2026" in l][0]
    bt = datetime.datetime.strptime(banner.split("banner: ")[1].strip(), "%a %b %d %H:%M:%S %Y")
    dt = (ts(ce["DATE_UTC"]) - bt).total_seconds() / 3600.0
    out += "\nHost: `%s`, GPU `%s`, driver %s, CUDA (nvidia-smi) %s, nvcc %s, %s; `TREE_src_digest=%s` (`TREE_git=%s`).\n" % (
        ce["HOST"], ce["GPU_NAME"], ce["NVIDIA_DRIVER"], ce["CUDA_VERSION_NVIDIA_SMI"], ce["NVCC"], ce["GCC"], ce["TREE_src_digest"], ce["TREE_git"])
    out += "derived clock offset: `run.env` DATE_UTC %s minus the `nvidia-smi` banner time %s = %.2f h.\n" % (ce["DATE_UTC"], bt.strftime("%Y-%m-%d %H:%M:%S"), dt)
    return out


# ---------------------------------------------------------------------------------------------------- A
PRED_LADDER = {"val_max": 3.9024, "val_rms": 4.3659, "grad_max": 2.9042, "grad_rms": 3.3858, "hess_max": 1.9154}
GATE_LADDER = {"val_max": 3.8, "val_rms": 3.8, "grad_max": 2.8, "grad_rms": 2.8, "hess_max": 1.8}
PRED_XONLY = {"val_max": 3.98, "grad_max": 2.98, "hess_max": 1.99}


def ladder_min(lines, kind, side):
    pairs = [kv(l) for l in lines if l.startswith("RECORDED A_LADDER_ORDER") and kv(l)["kind"] == kind and kv(l)["side"] == side]
    return {q: min(fl(p[q]) for p in pairs) for q in PRED_LADDER}, pairs


def a_md(raw):
    d = part_dir(raw, "controls")
    alog = read(os.path.join(d, "A_full.log"))
    res = read(os.path.join(d, "results.txt"))
    checks = lines_starting(alog, "A_CHECK ")
    gated = [l for l in checks if kv(l)["verdict"] in ("PASS", "FAIL")]
    recd = [l for l in checks if kv(l)["verdict"] == "RECORDED"]
    out = "Tier `%s`; checks in `A_full.log`: gated %d (PASS %d, FAIL %d), recorded %d; `A_VERDICT` line of the log: `%s`.\n\n" % (
        re.search(r"A_TIER tier=(\S+)", alog).group(1), len(gated),
        sum(1 for l in gated if kv(l)["verdict"] == "PASS"), sum(1 for l in gated if kv(l)["verdict"] == "FAIL"), len(recd),
        lines_starting(alog, "A_VERDICT")[-1])
    out += "#### Gated checks grouped by name (worst row over boxes and sides; `min` for orders, `max` otherwise)\n\n"
    out += table(("check", "rows", "worst", "value", "worst at (box/side)", "gate", "verdict", "margin (derived)"), group_rows(gated))
    out += "\nmargin = gate / worst value for upper-bound gates, worst value / gate for order gates; `-` for exact-zero gates or zero values.\n"
    host, hp = ladder_min(res.splitlines(), "3d", "host")
    dev, _ = ladder_min(res.splitlines(), "3d", "device")
    out += "\n#### Order ladder `f = exp(x1) sin(3 x1) (1 + 0.05 S)` on 32x16x16, 64x32x32, 128x64x64: minimum over the pairs 16->32, 32->64 (`A_LADDER_ORDER`, host; device equal: %s)\n\n" % (
        "yes" if host == dev else "NO")
    rows = []
    for q in PRED_LADDER:
        rows.append((q, "%.4f" % PRED_LADDER[q], "%.6f" % host[q], "%+.4f" % (host[q] - PRED_LADDER[q]), GATE_LADDER[q], "%.4f" % (host[q] - GATE_LADDER[q])))
    out += table(("quantity", "predicted (PRE, A_ladder emulation)", "measured min", "measured - predicted (derived)", "gate", "margin over gate (derived)"), rows)
    out += "\nPer pair (host):\n\n" + table(("pair", "val_max", "val_rms", "grad_max", "grad_rms", "hess_max"),
                                          [(p["pair"], p["val_max"], p["val_rms"], p["grad_max"], p["grad_rms"], p["hess_max"]) for p in hp])
    xo = [kv(l) for l in res.splitlines() if l.startswith("RECORDED A_LADDER_ORDER") and kv(l)["kind"] == "xonly" and kv(l)["side"] == "host"]
    xo_dev = [kv(l) for l in res.splitlines() if l.startswith("RECORDED A_LADDER_ORDER") and kv(l)["kind"] == "xonly" and kv(l)["side"] == "device"]
    out += "\n#### Recorded: x-only sub-ladder `g = exp(x1) sin(3 x1)`, `L1 = 2`, boxes `nx x 4 x 4` with 16/32/64/128 cells per unit length (host; device equal: %s)\n\n" % (
        "yes" if [{k: v for k, v in p.items() if k != "side"} for p in xo] == [{k: v for k, v in p.items() if k != "side"} for p in xo_dev] else "NO")
    out += table(("pair", "val_max", "val_rms", "grad_max", "grad_rms", "hess_max"), [(p["pair"], p["val_max"], p["val_rms"], p["grad_max"], p["grad_rms"], p["hess_max"]) for p in xo])
    last = xo[-1]
    out += "\nPredicted 64->128 orders (SPEC) vs measured: " + "; ".join("%s %.2f vs %s" % (q, PRED_XONLY[q], last[q]) for q in PRED_XONLY) + ".\n"
    em = [kv(l) for l in res.splitlines() if l.startswith("RECORDED A_EMUL")]
    emh = [e for e in em if e["side"] == "host"]
    emd = [e for e in em if e["side"] == "device"]
    out += "\n#### Recorded: measured over emulated error per level (`A_EMUL`, host; device lines equal the host lines: %s)\n\n" % (
        "yes" if [(e["level"], e["quantity"], e["measured"], e["emulated"]) for e in emh] == [(e["level"], e["quantity"], e["measured"], e["emulated"]) for e in emd] else "NO")
    out += table(("quantity", "level h", "measured", "emulated", "measured / emulated"), [(e["quantity"], e["level"], e["measured"], e["emulated"], e["measured_over_emulated"]) for e in emh])
    ratios = [fl(e["measured_over_emulated"]) for e in emh]
    out += "\nderived: measured/emulated over the 15 host rows: min %.6f, max %.6f.\n" % (min(ratios), max(ratios))
    out += "\n#### Recorded (ungated) checks (`A_CHECK ... verdict=RECORDED`, host and device rows)\n\n"
    out += table(("check", "rows", "worst", "value", "worst at (box/side)"), [r[:5] for r in group_rows(recd) if not r[0].startswith("xonly_")])
    mm = read(os.path.join(d, "A_memcheck.log"))
    out += "\n`A_memcheck.log` (compute-sanitizer memcheck, `--only domain`): `%s`; %s.\n" % (lines_starting(mm, "A_VERDICT")[-1], re.search(r"ERROR SUMMARY: .*", mm).group(0))
    mem = [l for l in alog.splitlines() if l.startswith("A_MEMORY")]
    out += "`A_MEMORY` (32x16x16): `%s`.\n" % mem[0]
    out += "No-allocation record: `%s` (`cudaMemGetInfo` free bytes before / after).\n" % lines_starting(res, "RECORDED A_NOALLOC")[0].replace("RECORDED ", "")
    return out


# ---------------------------------------------------------------------------------------------------- B
def b_md(raw):
    d = part_dir(raw, "cross")
    log = read(os.path.join(d, "B.log"))
    res = read(os.path.join(d, "results.txt"))
    out = "`B_CONFIG`: `%s`.\n\n" % lines_starting(log, "B_CONFIG")[0][len("B_CONFIG "):]
    out += "#### Correctors (`B_SOLVE`) and metrics (`B_METRIC`; points = lattice 1e5 + cell centres of three x1 lines inside `[0.5, L1 - 0.5]`)\n\n"
    out += table(("corrector", "converged", "iterations", "initial", "final", "rel"),
                 [(kv(l)["corrector"], kv(l)["converged"], kv(l)["iterations"], kv(l)["initial"], kv(l)["final"], kv(l)["rel"]) for l in lines_starting(log, "B_SOLVE corrector=")])
    out += "\n" + lines_starting(log, "B_SOLVE G=")[0][len("B_SOLVE "):] + "\n\n"
    out += table(("field", "quantity", "rel difference", "gate", "verdict", "points"),
                 [(kv(l)["field"], kv(l)["quantity"], kv(l)["rel"], kv(l)["gate"], kv(l)["verdict"], kv(l)["points"]) for l in lines_starting(log, "B_METRIC ")])
    out += "\nlog verdict line: `%s`.\n" % lines_starting(log, "B_VERDICT")[-1]
    out += "\n#### Recorded: per-knot decay factor of the end effect (`B_DECAY`, fitted on knots above 1e3 times the interior floor)\n\n"
    dl = [kv(l) for l in lines_starting(log, "B_DECAY ")]
    out += table(("field", "line (j,k)", "face", "quantity", "factor", "knots used", "floor"), [(t["field"], t["line"], t["face"], t["quantity"], t["factor"], t["knots_used"], t["floor"]) for t in dl])
    for f_ in ("h_tilde", "lnK"):
        for q in ("gradient", "hessian"):
            v = [fl(t["factor"]) for t in dl if t["field"] == f_ and t["quantity"] == q]
            out += "\nderived: %s %s factor over %d fits: mean %.6f, min %.6f, max %.6f." % (f_, q, len(v), sum(v) / len(v), min(v), max(v))
    out += "\n\n`2 - sqrt(3)` = %.6f (derived formula; SPEC \"~0.268\").\n" % (2 - math.sqrt(3))
    # profile
    prof = [l.split() for l in lines_starting(log, "B_PROFILE ")]
    rows, byk = [], {}
    for p in prof:
        t = dict(x.split("=") for x in p[1:])
        byk.setdefault((t["field"], t["line"], t["face"]), {})[int(t["knot"])] = (fl(t["value"]), fl(t["gradient"]), fl(t["hessian"]))
    for key in sorted(byk):
        kn = byk[key]
        n = max(kn) + 1
        interior = [kn[k] for k in range(64, n)]
        med = [st.median(v[i] for v in interior) for i in range(3)]
        rows.append(key + (n, e3(kn[0][1]), e3(kn[0][2]), e3(kn[8][1]), e3(kn[8][2]), e3(med[1]), e3(med[2])))
    out += "\n#### Recorded: relative difference profile `B_PROFILE` (%d rows in the log; per knot distance from the face; first knot = nearest cell centre)\n\n" % len(prof)
    out += table(("field", "line", "face", "knots per face", "gradient @knot0", "hessian @knot0", "gradient @knot8", "hessian @knot8", "gradient median knots 64..", "hessian median knots 64.."), rows)
    out += "\n`RECORDED B_PROFILE` line of `results.txt`: `%s`.\n" % lines_starting(res, "RECORDED B_PROFILE")[0].replace("RECORDED ", "")
    return out


# ---------------------------------------------------------------------------------------------------- C
def c_md(raw):
    d = part_dir(raw, "controls")
    clog = read(os.path.join(d, "C_full.log"))
    res = read(os.path.join(d, "results.txt"))
    cc = lines_starting(clog, "C_CHECK ")
    out = "`C_full.log`: %d `C_CHECK` rows, all `verdict=PASS`: %s; log line `%s`.\n\n" % (len(cc), "yes" if all(kv(l)["verdict"] == "PASS" for l in cc) else "NO", lines_starting(clog, "C_VERDICT")[-1])
    for case in ("C1", "C2", "C3", "C4a", "C4b"):
        rows = group_rows([l for l in cc if kv(l)["name"].startswith(case + "_")])
        out += "#### %s (gated rows grouped by name; worst row; grids/paths in the worst-at column)\n\n" % case
        out += table(("check", "rows", "worst", "value", "worst at (grid/path)", "gate", "verdict", "margin (derived)"), rows) + "\n"
    out += "#### C2 `tau` against the independent quadrature (pre-activation emulation: about 5e-11; gate 1e-9)\n\n"
    out += table(("grid", "path", "tau rel error", "gate"), [(kv(l)["grid"], kv(l)["path"], kv(l)["value"], kv(l)["gate"]) for l in cc if kv(l)["name"] == "C2_tau_vs_quadrature_rel"])
    out += "\n#### Recorded `C_RECORD` lines (no threshold)\n\n"
    out += "```\n" + "\n".join(l.replace("RECORDED ", "") for l in res.splitlines() if l.startswith("RECORDED C_RECORD")) + "\n```\n\n"
    out += "#### C3 against the design emulation (max abs about 1.1e-15, max rel about 3.7e-15 on random points)\n\n"
    rr = [kv(l.replace("RECORDED C_RECORD ", "")) for l in res.splitlines() if l.startswith("RECORDED C_RECORD name=C3_random_shift")]
    out += table(("grid", "path", "max abs", "max rel", "bitwise-equal points (of 4096)"),
                 [(t["grid"], t["path"], t["max_abs"], t["max_rel"], t["bitwise_equal_points"]) for t in rr])
    out += "\n#### C4a Pollock non-periodic checks (`pollock_nonperiodic_tests --only C4`, `C4_pollock.log`)\n\n"
    pl = read(os.path.join(d, "C4_pollock.log"))
    out += table(("grid", "case", "mode", "exit_inlet", "exit_outlet", "reached", "expected", "mismatches", "verdict"),
                 [(t["grid"], t["case"], t["mode"], t["exit_inlet"], t["exit_outlet"], t["reached"], t["expected"], t.get("mismatches", "-"), t["verdict"])
                  for t in (kv(l) for l in lines_starting(pl, "C4_CHECK "))])
    out += "\nlog line: `%s`.\n" % ([l for l in pl.splitlines() if l.startswith("C4_VERDICT") or l.startswith("VERDICT")] or ["(none)"])[-1]
    out += "\n#### C5 solver on the C1 / C2 media through the lean stage (`C5.log`)\n\n"
    c5 = read(os.path.join(d, "C5.log"))
    out += table(("case", "size", "converged", "iterations", "initial", "final", "rel", "balance", "max head rel diff", "pass"),
                 [(t["case"], t["size"], t["converged"], t["iterations"], t["initial"], t["final"], t["rel"], t["balance"], t["max_head_rel_diff"], t["pass"])
                  for t in (kv(l) for l in lines_starting(c5, "C5_CHECK "))])
    out += "\nre-run verdict lines are in the note (`c_verdict.py`).\n"
    return out


# ---------------------------------------------------------------------------------------------------- D
def d_md(raw):
    d = part_dir(raw, "controls")
    res = read(os.path.join(d, "results.txt"))
    out = "Compared by `d_verdict.py` from the `D.log` lines; campaign lines copied from `results.txt`:\n\n"
    out += "```\n" + "\n".join(l for l in res.splitlines() if l.startswith("D_VERDICT")) + "\n```\n\n"
    rows = []
    for l in lines_starting(res, "RECORDED D_REF"):
        t = kv(l)
        sz = l.split()[2][len("size="):]
        lean = [x for x in lines_starting(res, "RECORDED D_LEAN") if x.split()[2] == "size=" + sz][0]
        rows.append((sz, "yes" if l.split(" ", 2)[2].split("B_SOLVER", 1)[1] == lean.split(" ", 2)[2].split("B_SOLVER", 1)[1] else "NO",
                     t["iterations"], t["initial"], t["final"], t["rel"], t["qbar"], t["Q_in"], t["Q_out"], t["balance"], t["max_div"]))
    rows.sort(key=lambda r: int(r[0].split("x")[0]))
    out += "#### `B_SOLVER` fields (reference `DirichletMedium`; the lean stage line is string-equal: column 2)\n\n"
    out += table(("size", "lean line equal", "iterations", "initial", "final", "rel", "qbar", "Q_in", "Q_out", "balance", "max_div"), rows)
    out += "\n`%s`\n" % lines_starting(res, "RECORDED D_C0_COMPARE")[0].replace("RECORDED ", "")
    # lean MEM_PHASE: five groups of five phases in order 128, 128 mapped, 512, 512 mapped, 3328
    mp = [kv(l) for l in lines_starting(res, "RECORDED run=D R=- MEM_PHASE")]
    mp = mp[: len(mp) // 2]  # the block is printed twice in results.txt (RECORDED once per sampler read); use the first copy
    groups = [mp[i:i + 5] for i in range(0, len(mp), 5)]
    labels = [("128x32x32", 131072, "device head"), ("128x32x32", 131072, "mapped-host head"), ("512x64x64", 2097152, "device head"),
              ("512x64x64", 2097152, "mapped-host head"), ("3328x256x256", 218103808, "device head")]
    rows = []
    for g, lab in zip(groups, labels):
        s2 = [t for t in g if t["phase"] == "S2"][0]
        mx = max(g, key=lambda t: int(t["exact_bytes"]))
        rows.append((lab[0], lab[2], lab[1], s2["exact_bytes"], mib(s2["exact_bytes"]), "%.3f" % (int(s2["exact_bytes"]) / lab[1]), mx["phase"],
                     ",".join("%s=%s" % (t["phase"], t["exact_bytes"]) for t in g)))
    out += "\n#### Lean-stage `MEM_PHASE` exact bytes in the D process (order of the block: 128 device, 128 mapped, 512 device, 512 mapped, 3328 device; B/cell = S2 / cells, derived)\n\n"
    out += table(("size", "head", "cells", "S2 exact B", "S2 MiB", "S2 B/cell", "max phase", "phases (exact bytes)"), rows)
    dl = read(os.path.join(d, "D.log"))
    cl = lines_starting(dl, "CHECK ")
    out += "\n#### `D.log` program checks: %d `CHECK` lines, PASS %d, other %d; `D_CHECK` lines and the closed-form S2 checks\n\n```\n" % (
        len(cl), sum(1 for l in cl if l.split()[2] == "PASS"), sum(1 for l in cl if l.split()[2] != "PASS"))
    out += "\n".join(l for l in dl.splitlines() if l.startswith("D_CHECK") or "S2_closed_form" in l or l.startswith("DIAG ")) + "\n```\n"
    out += "\n`%s`\n" % lines_starting(res, "RECORDED D_MEM")[0].replace("RECORDED ", "")
    out += "`%s`\n" % [l for l in res.splitlines() if l.startswith("RECORDED run=D exit=")][0].replace("RECORDED ", "")
    out += "`%s`\n" % [l for l in res.splitlines() if l.startswith("RECORDED run=D pid_peak_mib")][0].replace("RECORDED ", "")
    return out


# ---------------------------------------------------------------------------------------------------- E
def e_md(raw):
    d = part_dir(raw, "prodgrid")
    res = read(os.path.join(d, "results.txt"))
    cfg = {"E1": ("2048x256x256", 1, "0.0390625"), "E2": ("3328x384x384", 1.5, "0.041666666666666664"),
           "E3": ("3328x384x384", 2, "0.08333333333333333"), "E4": ("6656x256x256", 2, "0.25")}
    rows, ph_rows, fl_rows, st_rows = [], [], [], []
    for run in ("E1", "E2", "E3", "E4"):
        log = read(os.path.join(d, run + ".log"))
        t = {k: kv(l) for k, l in (("solver", lines_starting(log, "R=0 B_SOLVER")[0]),)}["solver"]
        pk = kv([l for l in res.splitlines() if l.startswith("RECORDED run=%s pid_peak_mib" % run)][0])
        ex = kv([l for l in res.splitlines() if l.startswith("RECORDED run=%s exit=" % run)][0])
        runl = kv([l for l in res.splitlines() if l.startswith("E_RUN run=%s " % run)][0])
        stage = kv(lines_starting(log, "R=0 STAGE")[0])
        mps = {kv(l)["phase"]: kv(l) for l in lines_starting(log, "R=0 MEM_PHASE")}
        ex_b = {p: int(v["exact_bytes"]) for p, v in mps.items()}
        mx = max(ex_b, key=lambda p: ex_b[p])
        nx, ny, nz = (int(x) for x in cfg[run][0].split("x"))
        n = nx * ny * nz
        rows.append((run, cfg[run][0], cfg[run][1], cfg[run][2], runl["solve"], t["converged"], t["iterations"], t["rel"], t["balance"], t["max_div"], t["fallback"],
                     pk["pid_peak_mib"], pk["dev_peak_mib"], pk["samples"], pk["status"], ex["wall_s"], ex["host_max_rss_kb"], stage["seconds"], runl["state"]))
        ph_rows.append((run, mx, ex_b["S2"], mib(ex_b["S2"]), "%.3f" % (ex_b["S2"] / n), ",".join("%s=%s" % (p, mib(ex_b[p])) for p in ("F", "S1", "S2", "V", "H", "S3", "T", "B")),
                        "%.3f" % (int(pk["pid_peak_mib"]) - ex_b["S2"] / MIB), "%.1f" % (30000 - int(pk["pid_peak_mib"])), "yes" if all(ex_b[p] <= ex_b["S2"] for p in ex_b) else "NO"))
        st = kv(lines_starting(log, "R=0 STATUS")[0])
        te = {kv(l)["pass"]: kv(l) for l in lines_starting(log, "R=0 TEXIT")}
        ts_ = {kv(l)["pass"]: kv(l) for l in lines_starting(log, "R=0 TSTATUS")}
        pol = kv(lines_starting(log, "R=0 POLLOCK")[0])
        bf = kv(lines_starting(log, "R=0 BACKFLOW")[0])
        st_rows.append((run, st["tracker_non_ok_plane"], st["tracker_non_ok_time"], st["seed_backflow"], te["plane"]["exit_inlet"], te["plane"]["exit_outlet"],
                        te["time"]["exit_inlet"], te["time"]["exit_outlet"], ts_["time"]["hist"], pol["exit_inlet_plane"], pol["exit_outlet_plane"], pol["exit_inlet_time"],
                        pol["exit_outlet_time"], pol["weighted_not_active"], bf["inlet_vmin"], bf["backflow_enc_plane"], bf["backflow_enc_time"]))
        fs = kv(lines_starting(log, "R=0 FLOWSTATS")[0])
        fl_rows.append((run, fs["qbar"], fs["x_faces_nonpositive"], fs["speed_pct(0,0.1,1,50)"], fs["interior_planes"], fs["mean_u"], fs["mean_v"], fs["mean_w"], fs["theta_2"], fs["theta_3"]))
    out = "Settings echoed by each log (`RESOLVED` line), e.g. E4: `%s`\n\n" % lines_starting(read(os.path.join(d, "E4.log")), "RESOLVED")[0][:200]
    out += "#### Solve, memory sampler, wall (verdict `E_RUN` fields: solve mode, PID peak against 30000 MiB; sampler `pid_peak.py`)\n\n"
    out += table(("run", "grid", "eps", "ell", "solve", "converged", "iterations", "rel", "balance", "max_div", "fallback", "PID peak MiB", "device-wide peak MiB",
                  "samples", "sampler status", "wall s", "host max RSS kB", "stage s", "E_RUN state"), rows)
    out += "\n#### Per-phase exact bytes (`MEM_PHASE`; MiB = bytes / 1048576, derived; B/cell = S2 / cells, derived; context+overhead = PID peak - S2 MiB, derived)\n\n"
    out += table(("run", "max phase", "S2 exact B", "S2 MiB", "S2 B/cell", "phases (MiB): F, S1, S2, V, H, S3, T, B", "PID peak - S2 MiB", "headroom to 30000 MiB", "every phase <= S2"), ph_rows)
    # closed forms vs MEMMODEL
    cres = read(os.path.join(part_dir(raw, "controls"), "results.txt"))
    mm = {}
    for tag in ("a", "b"):
        for l in lines_starting(cres, "RECORDED MEMMODEL_%s MEM_MODEL phase=" % tag):
            t = kv(l)
            mm[(tag, t["phase"])] = int(t["exact_bytes"])
    rows = []
    for run, tag in (("E2", "a"), ("E3", "a"), ("E4", "b")):
        log = read(os.path.join(d, run + ".log"))
        ex_b = {kv(l)["phase"]: int(kv(l)["exact_bytes"]) for l in lines_starting(log, "R=0 MEM_PHASE")}
        rows.append((run, "MEMMODEL_" + tag) + tuple("%s %s" % (p, "equal" if ex_b[p] == mm[(tag, p)] else "differs (%d vs %d)" % (ex_b[p], mm[(tag, p)])) for p in ("F", "S2", "V", "H", "S3")))
    out += "\n#### Closed-form memory model (`controls` MEMMODEL_a / MEMMODEL_b, printed before the runs) against the measured exact bytes (derived comparison)\n\n"
    out += table(("run", "model", "F", "S2", "V", "H", "S3"), rows)
    out += "\n#### Tracking, exits and non-ok (`STATUS`, `TEXIT`, `TSTATUS`, `POLLOCK`, `BACKFLOW` lines of the logs)\n\n"
    out += table(("run", "tracker non_ok plane", "non_ok time", "seed_backflow", "exit_inlet plane", "exit_outlet plane", "exit_inlet time", "exit_outlet time", "TSTATUS time hist",
                  "Pollock exit_inlet plane", "Pollock exit_outlet plane", "Pollock exit_inlet time", "Pollock exit_outlet time", "Pollock weighted_not_active", "inlet_vmin",
                  "backflow_enc plane", "backflow_enc time"), st_rows)
    out += "\n#### `flow_stats.json` values (`FLOWSTATS` line; `theta_2`, `theta_3` = tilt of the interior mean face-flux vector)\n\n"
    out += table(("run", "qbar", "x_faces_nonpositive", "speed/qbar percentiles (0, 0.1, 1, 50 %)", "interior planes", "mean_u", "mean_v", "mean_w", "theta_2", "theta_3"), fl_rows)
    mapped = [f for f in os.listdir(d) if "mapped" in f]
    out += "\nMapped-host re-run: files with `mapped` in the E part directory: %s; `E_RUN` lines carry `(device_head)` only.\n" % (",".join(mapped) if mapped else "none")
    out += "\n```\n" + "\n".join(l for l in res.splitlines() if l.startswith("E_VERDICT") or l.startswith("E_RUN")) + "\n```\n"
    return out


# ---------------------------------------------------------------------------------------------------- F
def v3b(dp, se, sig128):
    """Recomputed SF-37 V3b class (RR8): concordant iff |D_P| decreases on every refinement and |D_P(finest)| <= max(0.05 sigma_T^2(consumer), 2 SE_D)."""
    a = [abs(x) for x in dp]
    dec = all(a[i + 1] < a[i] for i in range(len(a) - 1))
    return "concordant" if dec and a[-1] <= max(0.05 * sig128, 2 * se) else "discordant"


def f_md(raw):
    d = part_dir(raw, "evladder")
    res = read(os.path.join(d, "results.txt"))
    cells = [kv(l) for l in lines_starting(res, "F_CELL ")]
    out = "Rungs: `ny` = 32, 64, 128, 256 (`lambda/h` = 16, 32, 64, 128; `ell` 0.5), planes `x1` = %s (`RECORDED F_PLANES`).\n\n" % lines_starting(res, "RECORDED F_PLANES")[0].split()[-1]
    out += "```\n" + "\n".join(l for l in res.splitlines() if l.startswith("F_VERDICT")) + "\n```\n\n"
    rows = []
    for eps in ("1", "2"):
        cs = [c for c in cells if c["eps"] == eps]
        ords = [[fl(x) for x in c["orders"].split(",")] for c in cs]
        last = [o[-1] for o in ords]
        mn = min(range(len(cs)), key=lambda i: last[i])
        mx = max(range(len(cs)), key=lambda i: last[i])
        rows.append((eps, len(cs), sum(1 for c in cs if c["decreasing"] == "1"), sum(1 for c in cs if c["MET"] == "MET") if "MET" in cs[0] else sum(1 for l in lines_starting(res, "F_CELL eps=%s " % eps) if l.endswith(" MET")),
                     "%.4f" % min(o[0] for o in ords), "%.4f" % max(o[0] for o in ords), "%.4f" % min(o[1] for o in ords), "%.4f" % max(o[1] for o in ords),
                     "%.4f (seed %s plane %s)" % (last[mn], cs[mn]["seed"], cs[mn]["plane"]), "%.4f (seed %s plane %s)" % (last[mx], cs[mx]["seed"], cs[mx]["plane"]),
                     "%.2e" % min(fl(c["e"].split(",")[-1]) for c in cs), "%.2e" % max(fl(c["e"].split(",")[-1]) for c in cs)))
    out += "#### Observed order of the RMS `e_v` ladder per amplitude (`F_CELL`; order of a pair = log2(e_N / e_2N); last pair = 128 -> 256; 42 cells = 3 seeds x 14 planes)\n\n"
    out += table(("eps", "cells", "strictly decreasing", "MET", "min order 32->64", "max order 32->64", "min order 64->128", "max order 64->128", "min last-pair order (cell)",
                  "max last-pair order (cell)", "min e_v at ny=256", "max e_v at ny=256"), rows)
    out += "\nRMS `e_v` of the first and last plane per seed (`F_CELL`; four rungs ny = 32, 64, 128, 256):\n\n"
    rr = []
    for c in cells:
        if c["plane"] in ("0", "13"):
            rr.append((c["eps"], c["seed"], c["plane"], c["x1"], c["e"].replace(",", " / "), c["orders"].replace(",", " / ")))
    out += table(("eps", "seed", "plane", "x1", "e_v at ny=32 / 64 / 128 / 256", "orders"), rr)
    # runs
    rr = []
    for eps in ("1", "2"):
        for seed in ("6101", "6102", "6103"):
            for g in ("32", "64", "128", "256"):
                run = "F_e%s_s%s_g%s" % (eps, seed, g)
                log = read(os.path.join(d, run + ".log"))
                wall = [r for r in tsv(os.path.join(d, "walls.tsv")) if r[0] == run][0][1]
                rc = [r for r in tsv(os.path.join(d, "exit_codes.tsv")) if r[0] == run][0][1]
                so = kv(lines_starting(log, "R=0 B_SOLVER")[0])
                stt = kv(lines_starting(log, "R=0 STATUS")[0])
                rr.append((run, rc, wall, so["iterations"], so["rel"], so["balance"], stt["tracker_non_ok_plane"], stt["tracker_non_ok_time"], stt["exit_inlet_plane"],
                           stt["exit_outlet_plane"], stt["exit_inlet_time"], stt["exit_outlet_time"], stt["seed_backflow"]))
    out += "\n#### The 24 runs (exit code, wall s from `walls.tsv`, `B_SOLVER`, `STATUS` lines)\n\n"
    out += table(("run", "rc", "wall s", "PCG iterations", "rel", "balance", "non_ok plane", "non_ok time", "exit_inlet plane", "exit_outlet plane", "exit_inlet time", "exit_outlet time", "seed_backflow"), rr)
    ws = [int(r[2]) for r in rr]
    out += "\nderived: wall of the 24 runs: min %d s, max %d s, sum %d s.\n" % (min(ws), max(ws), sum(ws))
    # D_P (campaign results.txt lines)
    dps = [kv(l) for l in lines_starting(res, "RECORDED F_DP eps=") if "D_P=" in l]
    out += "\n#### Recorded (no threshold): `D_P = sigma_T^2(Pollock) - sigma_T^2(consumer)` from the campaign `F_DP` lines of `results.txt` (computed on the V100 from the per-particle `.npy` arrays, absent from `raw/`; %d lines)\n\n" % len(dps)
    rows, chk = [], 0
    for eps in ("1", "2"):
        cs = [t for t in dps if t["eps"] == eps]
        lab = {"concordant": 0, "discordant": 0}
        agree = 0
        for t in cs:
            dp = [fl(x) for x in t["D_P"].split(",")]
            se = [fl(x) for x in t["SE_D"].split(",")]
            sc = [fl(x) for x in t["sigma_T2_consumer"].split(",")]
            rec = v3b(dp, se[-1], sc[-1])
            lab[t["V3b"]] += 1
            agree += rec == t["V3b"]
        chk += agree
        a = lambda i: [abs(fl(t["D_P"].split(",")[i])) for t in cs]
        sig = lambda i: [fl(t["sigma_T2_consumer"].split(",")[i]) for t in cs]
        rows.append((eps, len(cs), lab["concordant"], lab["discordant"], "%d / %d" % (agree, len(cs)),
                     " / ".join("%.2e" % st.median(a(i)) for i in range(4)), " / ".join("%.2e" % max(a(i)) for i in range(4)),
                     "%.4e - %.4e" % (min(sig(3)), max(sig(3)))))
    out += table(("eps", "cells", "concordant (campaign)", "discordant (campaign)", "recomputed class agrees", "median |D_P| ny=32/64/128/256", "max |D_P| ny=32/64/128/256",
                  "consumer sigma_T^2 range at ny=256"), rows)
    out += ("\nV3b recomputation (derived, RR8: concordant iff |D_P| decreases on every refinement and |D_P(ny=256)| <= max(0.05 sigma_T^2(consumer, ny=256), 2 SE_D(ny=256))): "
            "agrees with the campaign label on %d of %d cells. Recorded only; no `alpha` content.\n" % (chk, len(dps)))
    out += "\n`F_V3B_SUMMARY` lines of the campaign: `%s`; `%s`.\n" % tuple(l.replace("RECORDED ", "") for l in lines_starting(res, "RECORDED F_V3B_SUMMARY"))
    mx = max((fl(t["csv_max_rel"].split(",")[i]) for t in dps for i in range(4)))
    out += "derived: max `csv_max_rel` (recomputed var_x2/var_x3 against the programs' moments CSVs) over all F_DP lines: %.3e.\n" % mx
    ex = [kv(l) for l in lines_starting(res, "RECORDED F_DP eps=") if "D_P=" not in l]
    out += "Lines in the campaign `results.txt` that are not `D_P` rows: %d.\n" % len(ex)
    return out


# ---------------------------------------------------------------------------------------------------- G
def g_md(raw):
    d = part_dir(raw, "final")
    res = read(os.path.join(d, "results.txt"))
    out = "```\n" + "\n".join(l for l in res.splitlines() if l.startswith("G_")) + "\n```\n\n"
    ident = [l for l in res.splitlines() if l.startswith("IDENTICAL ")]
    pref = lambda l: l.split()[1].split(":")[0]
    trio = [l for l in ident if pref(l) == "trio"]
    rset = [l for l in ident if pref(l) == "r"]
    rbin = [l for l in ident if pref(l) == "binary"]
    gii = [l for l in ident if pref(l) == "g_runner_a"]
    giii = [l for l in ident if pref(l) == "g_ab"]
    other = [l for l in res.splitlines() if re.match(r"(DIFF|MISSING|EXTRA|NOT_)", l)]
    ctl = tsv(os.path.join(d, "ctest_times.tsv"))
    out += table(("item", "count (derived from `results.txt` lines)"),
                 [("`IDENTICAL trio:` items (SF-27 trio)", len(trio)), ("`IDENTICAL r:` items (SF-38 20-run R set)", len(rset)),
                  ("`IDENTICAL binary:md5_differs` item (R set, head != base binary)", len(rbin)), ("R set total (`r:` + `binary:`)", len(rset) + len(rbin)),
                  ("`IDENTICAL g_runner_a:` items (G(ii), the rc line)", len(gii)), ("`IDENTICAL g_ab:` items (G(iii), the rc line)", len(giii)),
                  ("all `IDENTICAL` lines", len(ident)), ("DIFF / MISSING / EXTRA lines", len(other)), ("`ctest_times.tsv` rows", len(ctl)),
                  ("rows `Passed`", sum(1 for r in ctl if r[1] == "Passed")), ("sum of per-test seconds", "%.2f" % sum(fl(r[2]) for r in ctl))])
    out += "\nbinary line: `%s`\n" % rbin[0]
    out += "\n#### Named ctest entries (`ctest_times.tsv`)\n\n"
    out += table(("test", "status", "seconds"), [r for r in ctl if r[0] in ("interpolation_periodic_tricubic", "interpolation_hessian", "interpolation_nonperiodic_x", "lagrangian_cross_representation",
                                                                           "lagrangian_dirichlet_long_box_stage", "lagrangian_pollock_nonperiodic", "lagrangian_dirichlet_controls",
                                                                           "lagrangian_ensemble_medium_b", "lagrangian_ensemble", "lagrangian_medium_ab", "lagrangian_tracker_gpu_host",
                                                                           "inlet_slab_trace_gpu", "streamfunction_operator_tests")])
    top = sorted(ctl, key=lambda r: -fl(r[2]))[:5]
    out += "\nFive longest tests: " + "; ".join("%s %s s" % (r[0], r[2]) for r in top) + ".\n"
    out += "\n#### Base reference runs (`sf41_base_refs`) and head runs of the byte-compare sets\n\n"
    br = read(os.path.join(raw, "sf41_base_refs", "results.txt"))
    out += "`%s`\n" % lines_starting(br, "BASEREFS_RECORDED")[0]
    ct = tsv(os.path.join(raw, "sf41_base_refs", "exit_codes.tsv"))
    hd = tsv(os.path.join(d, "exit_codes.tsv"))
    bnz = {r[0]: r[1] for r in ct if r[1] != "0"}
    hnz = {r[0]: r[1] for r in hd if r[1] != "0"}
    bruns = {r[0] for r in ct}
    out += ("base reference runs in `exit_codes.tsv`: %d; non-zero (run: rc) in the base part: %s; in the head `final` part: %s (%d head runs); "
            "same non-zero set and codes: %s; every head run name except `ctest` also a base run: %s (derived).\n") % (
        len(ct), ", ".join("%s: %s" % kv_ for kv_ in sorted(bnz.items())), ", ".join("%s: %s" % kv_ for kv_ in sorted(hnz.items())), len(hd),
        "yes" if bnz == hnz else "NO", "yes" if all(r[0] in bruns for r in hd if r[0] != "ctest") else "NO")
    return out


# ---------------------------------------------------------------------------------------------------- H
def h_md(raw):
    d = part_dir(raw, "abrerun")
    res = read(os.path.join(d, "results.txt"))
    out = "```\n" + "\n".join(l for l in res.splitlines() if l.startswith("H_VERDICT") or l.startswith("RECORDED H_AB_PRIMARY")) + "\n```\n\n"
    rows = []
    for l in lines_starting(res, "RECORDED H_EXITS_BY_PLANE"):
        t = kv(l)
        r = [x for x in lines_starting(res, "H_REALIZATION amp=%s r=%s " % (t["amp"], t["r"]))][0]
        rt = kv(r)
        inl = [int(x) for x in t["inlet"].split(",")]
        outl = [int(x) for x in t["outlet"].split(",")]
        rows.append((t["amp"], t["r"], t["Np"], len(inl), sum(inl), sum(outl), rt["e2_b"], rt["e2_a_control"], r.split()[-1]))
    out += table(("amp", "r", "Np", "planes", "x=0 exits (all planes)", "outlet exits (all planes)", "e2_b.csv vs SF-38", "e2_a.csv vs SF-38 (control)", "class"), rows)
    out += "\nplanes (x1) of amplitude s1: %s\n\nplanes (x1) of amplitude s4: %s\n" % (
        [kv(l)["x1"] for l in lines_starting(res, "RECORDED H_EXITS_BY_PLANE amp=s1")][0], [kv(l)["x1"] for l in lines_starting(res, "RECORDED H_EXITS_BY_PLANE amp=s4")][0])
    out += "\nderived: realizations %d; x = 0 exits total %d; e2_b identical %d of %d; e2_a control identical %d of %d.\n" % (
        len(rows), sum(r[4] for r in rows), sum(1 for r in rows if r[6] == "IDENTICAL"), len(rows), sum(1 for r in rows if r[7] == "IDENTICAL"), len(rows))
    out += "\nPer-run walls (`walls.tsv`): " + "; ".join("%s %s s" % (r[0], r[1]) for r in tsv(os.path.join(d, "walls.tsv"))) + ".\n"
    return out


# ---------------------------------------------------------------------------------------------------- main
def main(raw, out):
    os.makedirs(out, exist_ok=True)
    for name, fn in (("parts.md", parts_md), ("A.md", a_md), ("B.md", b_md), ("C.md", c_md), ("D.md", d_md), ("E.md", e_md), ("F.md", f_md), ("G.md", g_md), ("H.md", h_md)):
        write(out, name, fn(raw))


def selftest():
    assert kv("a=1 b=x,y c=3e-3") == {"a": "1", "b": "x,y", "c": "3e-3"}
    assert table(("a", "b"), [(1, 2)]) == "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert abs(order(4e-3, 1e-3) - 2.0) < 1e-15 and abs(order(8.0, 1.0) - 3.0) < 1e-15
    assert mib(1048576 * 3) == "3.000" and e3(0.5) == "5.000e-01"
    assert ts("2026-10-25T00:01:00Z") - ts("2026-10-24T23:59:30Z") == datetime.timedelta(seconds=90)
    # V3b: |D_P| 4,3,2,1 decreasing, finest 1 <= max(0.05*30=1.5, 2*0.1) -> concordant; non-monotone or too large -> discordant
    assert v3b([4, -3, 2, -1], 0.1, 30.0) == "concordant"
    assert v3b([4, 5, 2, 1], 0.1, 30.0) == "discordant"
    assert v3b([4, 3, 2, 1], 0.1, 10.0) == "discordant"  # 1 > max(0.5, 0.2)
    assert v3b([4, 3, 2, 1], 0.6, 10.0) == "concordant"  # 1 <= 1.2
    # group_rows: max for ordinary checks, min for *_order_ge_*, grid/box labels
    rows = group_rows(["A_CHECK name=x box=b1 side=host value=1e-15 gate=1e-12 verdict=PASS",
                       "A_CHECK name=x box=b2 side=device value=3e-15 gate=1e-12 verdict=PASS",
                       "A_CHECK name=o_order_ge_v box=b side=host value=3.9 gate=3.8 verdict=PASS",
                       "A_CHECK name=o_order_ge_v box=b side=device value=4.1 gate=3.8 verdict=PASS"])
    assert rows[0] == ("x", 2, "max", "3.000e-15", "b2/device", "1e-12", "PASS", "333.33")
    assert rows[1] == ("o_order_ge_v", 2, "min", "3.900e+00", "b/host", "3.8", "PASS", "1.03")
    assert margin(0.0, "<=1e-12", False) == "-" and margin(2e-12, "==0", False) == "-" and margin(5e-11, "<=1e-09", False) == "20.00"
    # ladder_min over two pairs
    ln = ["RECORDED A_LADDER_ORDER kind=3d side=host pair=16->32 val_max=3.9 val_rms=4.4 grad_max=2.9 grad_rms=3.4 hess_max=1.9",
          "RECORDED A_LADDER_ORDER kind=3d side=host pair=32->64 val_max=3.95 val_rms=4.3 grad_max=2.95 grad_rms=3.5 hess_max=1.85",
          "RECORDED A_LADDER_ORDER kind=xonly side=host pair=16->32 val_max=1.0 val_rms=1.0 grad_max=1.0 grad_rms=1.0 hess_max=1.0"]
    m, _ = ladder_min(ln, "3d", "host")
    assert m == {"val_max": 3.9, "val_rms": 4.3, "grad_max": 2.9, "grad_rms": 3.4, "hess_max": 1.85}
    assert abs((2 - math.sqrt(3)) - 0.2679491924) < 1e-9
    print("note_tables selftest: OK")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
