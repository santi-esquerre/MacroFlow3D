#!/usr/bin/env python3
"""SF-36 campaign digest: parse the logs written by run_sf36.sh into markdown tables.

  digest_sf36.py LOGROOT OUTDIR

LOGROOT holds one directory per part (`controls/`, `bytecmp/`, `mem/`, `cost/`, `stress/`; the layout of
`$SF36_OUT_ROOT`, default `~/sf36_campaign`, or a copy of it inside the artifact `logs/`).  OUTDIR receives
`controls.md`, `bytecmp.md`, `mem.md`, `cost.md`, `stress.md`; a part that is absent is reported as such, never as a
pass.  The digests tabulate; they pre-register nothing (the thresholds are in the increment specification).
Pure python3 standard library.  Exit status 0 unless a log that exists cannot be parsed.
"""
import glob
import json
import os
import re
import sys


def read_lines(path):
    try:
        with open(path, errors="replace") as f:
            return [ln.rstrip("\n") for ln in f]
    except OSError:
        return None


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def exit_codes(part_dir):
    ln = read_lines(os.path.join(part_dir, "exit_codes.tsv"))
    rows = []
    for x in ln or []:
        p = x.split("\t")
        if len(p) == 2:
            rows.append((p[0], p[1]))
    return rows


def write(outdir, name, text):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, name), "w") as f:
        f.write(text.rstrip("\n") + "\n")
    print("wrote %s" % os.path.join(outdir, name))


def time_file(path):
    """(wall string, max RSS kB string, exit status string) from a /usr/bin/time -v file."""
    wall = rss = st = "n/a"
    for ln in read_lines(path) or []:
        s = ln.strip()
        if s.startswith("Elapsed (wall clock)"):
            wall = s.split()[-1]
        elif s.startswith("Maximum resident set size"):
            rss = s.split()[-1]
        elif s.startswith("Exit status"):
            st = s.split()[-1]
    return wall, rss, st


def header_block(part_dir, title):
    env = read_lines(os.path.join(part_dir, "run.env"))
    s = ["# %s" % title, ""]
    if env:
        s.append("Run environment (`run.env`): " + "; ".join(env))
    else:
        s.append("`run.env` absent.")
    return s


# ---------------------------------------------------------------------------------------------------------------
def digest_controls(root, outdir):
    d = os.path.join(root, "controls")
    if not os.path.isdir(d):
        write(outdir, "controls.md", "# SF-36 controls\n\nPart `controls` absent under %s (not run)." % root)
        return
    s = header_block(d, "SF-36 controls (A, B, C, D, E1/E2, I at the campaign grids)")
    s += ["", "## Exit codes", "", md_table(["run", "exit code"], exit_codes(d))]
    for name, title in (("controls_full", "lagrangian_controls_tests --full"),
                        ("controls_diag", "lagrangian_controls_tests --diag"),
                        ("gpu_host_full", "lagrangian_gpu_host_tests --full")):
        lines = read_lines(os.path.join(d, name + ".log"))
        s += ["", "## %s (`%s.log`)" % (title, name), ""]
        if lines is None:
            s.append("log absent (not run).")
            continue
        counts = {"PASS": 0, "FAIL": 0, "RECORD": 0}
        rows, tail = [], []
        for ln in lines:
            m = re.match(r"^\[(PASS|FAIL|RECORD)\]\s*(.*)$", ln)
            if m:
                counts[m.group(1)] += 1
                rows.append((m.group(1), m.group(2)[:260]))
            elif ln.startswith("SUMMARY:") or ln.strip() in ("ALL PASS", "SOME FAILED"):
                tail.append(ln.strip())
        s.append("PASS %d, FAIL %d, RECORD %d." % (counts["PASS"], counts["FAIL"], counts["RECORD"]))
        s.append("")
        s.append("Final lines: " + (" ; ".join("`%s`" % t for t in tail) if tail else "**none (no SUMMARY line)**"))
        s += ["", md_table(["kind", "check / detail"], rows) if rows else "no PASS/FAIL/RECORD lines."]
    write(outdir, "controls.md", "\n".join(s))


# ---------------------------------------------------------------------------------------------------------------
def digest_bytecmp(root, outdir):
    d = os.path.join(root, "bytecmp")
    if not os.path.isdir(d):
        write(outdir, "bytecmp.md", "# SF-36 byte comparison (F)\n\nPart `bytecmp` absent under %s (not run)." % root)
        return
    s = header_block(d, "SF-36 byte comparison against the base-commit references (criterion F)")
    res = read_lines(os.path.join(d, "results.txt")) or []
    rows, bad = [], 0
    for ln in res:
        m = re.match(r"^BYTECMP (\S+) npy=(\S+) lines=(\S+) rc=(\S+)\s*(.*)$", ln)
        if not m:
            continue
        case, npy, lines, rc, note = m.groups()
        verdict = "SKIP"
        if "SKIP" not in (npy, lines):
            base_rc, _, head_rc = rc.partition("/")
            ok = npy in ("IDENTICAL", "n/a") and lines == "IDENTICAL" and (base_rc in ("?", head_rc))
            verdict = "PASS" if ok else "FAIL"
            bad += 0 if ok else 1
        rows.append((case, npy, lines, rc, verdict, note))
    s += ["", "Verdicts are mechanical: `PASS` = psi1/psi2 `cmp`-identical (or n/a), selected lines identical after masking "
          "the timing fields, and equal exit codes (`?` = the base exit code was not recorded for that case).", ""]
    s.append(md_table(["case", "npy", "lines", "rc base/head", "verdict", "note"], rows) if rows else
             "no BYTECMP lines in `results.txt`.")
    s += ["", "Mismatches: %d; skipped: %d." % (bad, sum(1 for r in rows if r[4] == "SKIP")),
          "", "## Exit codes of the head runs", "", md_table(["run", "exit code"], exit_codes(d))]
    for p in sorted(glob.glob(os.path.join(d, "*.linecmp"))):
        s += ["", "### %s" % os.path.basename(p), "", "```"] + (read_lines(p) or []) + ["```"]
    write(outdir, "bytecmp.md", "\n".join(s))


# ---------------------------------------------------------------------------------------------------------------
def digest_mem(root, outdir):
    d = os.path.join(root, "mem")
    if not os.path.isdir(d):
        write(outdir, "mem.md", "# SF-36 device peak (G)\n\nPart `mem` absent under %s (not run)." % root)
        return
    s = header_block(d, "SF-36 device peak (criterion G)")
    res = read_lines(os.path.join(d, "results.txt")) or []
    s += ["", "## Result lines", "", "```"] + res + ["```"]
    for p in sorted(glob.glob(os.path.join(d, "*.mem"))):
        vals = []
        for ln in read_lines(p) or []:
            try:
                vals.append(int(ln.strip()))
            except ValueError:
                pass
        s += ["", "Sampler `%s`: %d samples, max %s MiB, min %s MiB." %
              (os.path.basename(p), len(vals), max(vals) if vals else "n/a", min(vals) if vals else "n/a")]
    for p in sorted(glob.glob(os.path.join(d, "*.log"))):
        for ln in read_lines(p) or []:
            if ln.startswith("MEMORY") or ln.startswith("STATUS"):
                s.append("- `%s`: `%s`" % (os.path.basename(p), ln[:240]))
    s += ["", "## Exit codes", "", md_table(["run", "exit code"], exit_codes(d))]
    write(outdir, "mem.md", "\n".join(s))


# ---------------------------------------------------------------------------------------------------------------
def parse_tstatus(lines, which):
    for ln in lines:
        if ln.startswith("TSTATUS pass=%s " % which):
            m = re.search(r"hist=([0-9,]+)", ln)
            n = re.search(r"nfev=(\d+)", ln)
            b = re.search(r"backflow_enc=(\d+)", ln)
            if m and n:
                return ([int(x) for x in m.group(1).split(",")], int(n.group(1)), int(b.group(1)) if b else 0)
    return None


def num_list(txt):
    out = []
    for x in txt.split(","):
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def digest_cost(root, outdir):
    d = os.path.join(root, "cost")
    if not os.path.isdir(d):
        write(outdir, "cost.md", "# SF-36 cost (J)\n\nPart `cost` absent under %s (not run)." % root)
        return
    s = header_block(d, "SF-36 cost of the forward tracker (criterion J, recorded, no threshold)")
    s += ["", "fevals/particle/length = nfev(plane pass) / (n_ok(plane pass) x X_last); fevals/particle/time = "
          "nfev(time pass) / (n_ok(time pass) x T_last).  `n_ok` is `hist[0]` of the TSTATUS line of that pass; "
          "`X_last`, `T_last` the last entries of the TRACK `planes=` / `times=` lists.  Device peak = max of the "
          "nvidia-smi 200 ms sampler (whole GPU) and `peak_device_bytes` of the driver MEMORY line.", ""]
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "N*_e*_*.log")),
                    key=lambda q: (int(re.search(r"N(\d+)_", q).group(1)), os.path.basename(q))):
        name = os.path.basename(p)[:-4]
        lines = read_lines(p) or []
        np_ = x_last = t_last = None
        for ln in lines:
            if ln.startswith("TRACK "):
                m = re.search(r"\bNp=(\d+)", ln)
                np_ = int(m.group(1)) if m else None
                m = re.search(r"\bplanes=(\S*)", ln)
                pl = num_list(m.group(1)) if m else []
                x_last = pl[-1] if pl else None
                m = re.search(r"\btimes=(\S*)", ln)
                tl = num_list(m.group(1)) if m else []
                t_last = tl[-1] if tl else None
        tp, tt = parse_tstatus(lines, "plane"), parse_tstatus(lines, "time")

        def per(ts, last):
            if not ts or not last or not ts[0] or ts[0][0] <= 0:
                return "n/a"
            return "%.1f" % (ts[1] / (ts[0][0] * last))

        tracker = "n/a"
        mem_drv = "n/a"
        total = "n/a"
        try:
            with open(os.path.join(d, name + ".json")) as f:
                J = json.load(f)
            tt_s = J.get("tracker_timings_s", {})
            tracker = "plane %.2f / time %.2f / dl %.2f / mom %.2f" % (
                tt_s.get("plane_kernel", float("nan")), tt_s.get("time_kernel", float("nan")),
                tt_s.get("download", float("nan")), tt_s.get("moments", float("nan")))
            mem_drv = "%.0f" % (J["memory"]["peak_device_bytes"] / 2.0**20)
            total = "%.1f" % J["timing"]["total"]
        except (OSError, ValueError, KeyError):
            for ln in lines:
                m = re.match(r"^MEMORY peak_device_bytes=(\d+)", ln)
                if m:
                    mem_drv = "%.0f" % (int(m.group(1)) / 2.0**20)
                m = re.match(r"^TIMING .*total=([0-9.]+)s", ln)
                if m:
                    total = m.group(1)
        smp = []
        for ln in read_lines(os.path.join(d, name + ".mem")) or []:
            try:
                smp.append(int(ln.strip()))
            except ValueError:
                pass
        wall, rss, st = time_file(os.path.join(d, name + ".time"))
        status = next((ln.split(" ", 1)[1] for ln in lines if ln.startswith("STATUS ")), "no STATUS line")
        rows.append((name, np_ if np_ is not None else "n/a", status, wall, total, tracker,
                     (tp[1] if tp else "n/a"), per(tp, x_last), (tt[1] if tt else "n/a"), per(tt, t_last),
                     (tp[2] + tt[2]) if tp and tt else "n/a",
                     "%s / %s" % (max(smp) if smp else "n/a", mem_drv), rss))
    s.append(md_table(["run", "Np", "status", "wall [h:mm:ss]", "driver total [s]",
                       "tracker [s]", "nfev plane", "fevals/part/len", "nfev time", "fevals/part/time",
                       "backflow enc", "device peak MiB (smi / drv)", "host max RSS kB"], rows)
             if rows else "no cost logs.")
    s += ["", "## Exit codes", "", md_table(["run", "exit code"], exit_codes(d))]
    res = read_lines(os.path.join(d, "results.txt")) or []
    if res:
        s += ["", "## Result lines", "", "```"] + res + ["```"]
    write(outdir, "cost.md", "\n".join(s))


# ---------------------------------------------------------------------------------------------------------------
def digest_stress(root, outdir):
    d = os.path.join(root, "stress")
    if not os.path.isdir(d):
        write(outdir, "stress.md", "# SF-36 stress (N5 finding)\n\nPart `stress` absent under %s (not run)." % root)
        return
    s = header_block(d, "SF-36 inlet-backflow stress field (N5 finding; informative)")
    rc = exit_codes(d)
    s += ["", "Exit codes (124 = did not terminate within the `timeout`; 22 = `track_fail`; 0 = ok):", "",
          md_table(["run", "exit code"], rc)]
    res = read_lines(os.path.join(d, "results.txt")) or []
    if res:
        s += ["", "```"] + res + ["```"]
    for p in sorted(glob.glob(os.path.join(d, "*.log"))):
        lines = read_lines(p) or []
        s += ["", "### `%s` (%d lines)" % (os.path.basename(p), len(lines)), "", "```"]
        s += [ln[:260] for ln in lines if re.match(r"^(TRACK|BACKFLOW|TSTATUS|STATUS|TPLANE|TTIME)", ln)] or ["(no TRACK/TSTATUS/STATUS line)"]
        s += ["```"]
    write(outdir, "stress.md", "\n".join(s))


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    root, outdir = argv[1], argv[2]
    rc = 0
    for fn in (digest_controls, digest_bytecmp, digest_mem, digest_cost, digest_stress):
        try:
            fn(root, outdir)
        except Exception as e:  # report and continue: a digest crash must not hide the other tables
            print("digest %s FAILED: %r" % (fn.__name__, e), file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
