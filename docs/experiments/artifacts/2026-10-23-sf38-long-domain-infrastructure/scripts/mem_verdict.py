#!/usr/bin/env python3
"""SF-38 shared memory / bookkeeping readers for the B and C0 verdicts (adapted from SF-40 mem_verdict.py).

  mem_verdict.py peak DIR RUN [--limit MIB]     one PID_PEAK line (pid_peak.py format) + LIMIT verdict; exit 0 iff OK
  mem_verdict.py --selftest

Per-run files in DIR (written by run_sf38.sh case_run with the S flag): RUN.log RUN.time RUN.free RUN.apps.csv RUN.dev.csv
RUN.pid_tree and DIR/exit_codes.tsv (`RUN<TAB>rc`).

Differences from SF-40: the ensemble runner / A/B application prefix every per-realization line with `R=<r> `
(`R=0 MEM_GEOM pid=...`, `R=0 MEM_PHASE ...`), so the log readers here accept that prefix (pid_peak.py, which is
byte-identical to SF-40, reads only unprefixed lines).  When the log carries no pid (an application that prints no
MEM_GEOM line) the PID of the process tree found by run_sf38.sh (`RUN.pid_tree`, descendant of /usr/bin/time) is used.
Stdlib only.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pid_peak  # noqa: E402

LIMIT_MIB = 30000
PREFIX = re.compile(r"^R=\d+ ")


def rd(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def rc_of(d, run):
    rc = None
    for line in (rd(os.path.join(d, "exit_codes.tsv")) or "").splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] == run:
            try:
                rc = int(p[1])
            except ValueError:
                pass
    return rc


def time_info(d, run):
    t = rd(os.path.join(d, run + ".time")) or ""
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\S+)", t)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([\d:.]+)", t)
    w = None
    if wall:
        w = 0.0
        for part in wall.group(1).split(":"):
            w = w * 60 + float(part)
    return (rss.group(1) if rss else None), w


def free_line(d, run):
    for ln in (rd(os.path.join(d, run + ".free")) or "").splitlines():
        if ln.startswith("Mem:"):
            return " ".join(ln.split())
    return "-"


def strip_prefix(line):
    return PREFIX.sub("", line, count=1)


def log_info(path):
    """-> (pid or None, max exact_bytes of the MEM_PHASE lines or None); R=<r> prefix tolerant."""
    pid, mx = None, None
    text = rd(path) or ""
    for raw in text.splitlines():
        line = strip_prefix(raw)
        if pid is None:
            m = re.match(r"(?:MEM_GEOM|CONTEXT_PROBE) .*?\bpid=(\d+)", line)
            if m:
                pid = int(m.group(1))
        if line.startswith("MEM_PHASE "):
            m = re.search(r"\bexact_bytes=(\d+)", line)
            if m:
                v = int(m.group(1))
                mx = v if mx is None else max(mx, v)
    return pid, mx


def peak_of(d, run):
    """pid_peak.evaluate result for RUN (the RR2 gates); NOT_EVALUABLE with reasons when files are missing."""
    ap, dv, lg = (os.path.join(d, run + e) for e in (".apps.csv", ".dev.csv", ".log"))
    if not (os.path.exists(ap) and os.path.exists(dv) and os.path.exists(lg)):
        return {"status": "NOT_EVALUABLE", "reasons": ["missing_files"], "pid": None, "samples": 0, "pid_peak": None,
                "dev_peak": None, "max_exact_mib": None, "tree_pid": None}
    pid, mx = log_info(lg)
    tree = (rd(os.path.join(d, run + ".pid_tree")) or "").strip()
    tree = int(tree) if tree.isdigit() else None
    if pid is None:
        pid = tree  # the log prints no pid: the process tree found by the campaign script
    return pid_peak.evaluate(pid_peak.read_apps(ap), pid_peak.read_dev(dv), pid, mx, tree)


def mem_records(d, run):
    """RECORDED lines: wall, RSS, free -b, per-phase MEM lines (all realizations), dev peak / samples."""
    out = []
    rss, wall = time_info(d, run)
    out.append("RECORDED run=%s exit=%s wall_s=%s host_max_rss_kb=%s free_b_before=[%s]"
               % (run, rc_of(d, run), wall, rss, free_line(d, run)))
    for raw in (rd(os.path.join(d, run + ".log")) or "").splitlines():
        line = strip_prefix(raw)
        r = PREFIX.match(raw)
        tag = r.group(0).strip() if r else "R=-"
        if line.startswith("MEM_GEOM "):
            out.append("RECORDED run=%s %s %s" % (run, tag, re.sub(r"\s+level_cells=\S+", "", line)[:300]))
        elif line.startswith("MEM_PHASE "):
            ph = re.search(r"phase=(\S+)", line)
            kv = {k: re.search(r"\b%s=(\d+(?:\.\d+)?)" % k, line) for k in ("exact_bytes", "cufft_work", "device_used", "seconds")}
            out.append("RECORDED run=%s %s MEM_PHASE phase=%s %s" % (
                run, tag, ph.group(1) if ph else "?",
                " ".join("%s=%s" % (k, v.group(1) if v else "-") for k, v in kv.items())))
    pk = peak_of(d, run)
    out.append("RECORDED run=%s pid_peak_mib=%s dev_peak_mib=%s samples=%s max_exact_mib=%s tree_pid=%s status=%s reasons=%s"
               % (run, pk["pid_peak"], pk["dev_peak"], pk["samples"],
                  "-" if pk["max_exact_mib"] is None else "%.3f" % pk["max_exact_mib"], pk["tree_pid"], pk["status"],
                  ",".join(pk["reasons"]) or "-"))
    return out


def peak_line(d, run, limit=LIMIT_MIB):
    pk = peak_of(d, run)
    lines = [pid_peak.fmt(pk)]
    if pk["status"] != "OK":
        v = "NOT_EVALUABLE"
    else:
        v = "PASS" if pk["pid_peak"] <= limit else "FAIL"
    lines.append("LIMIT run=%s pid_peak_mib=%s limit_mib=%d %s" % (run, pk["pid_peak"], limit, v))
    return lines, pk["status"] == "OK"


# ---------------------------------------------------------------------------------------------------------------
def selftest():
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    def write_run(d, name, log, rc, pid, peak, devpeak, nsamp=20, tree=None, rss=4096):
        with open(os.path.join(d, name + ".log"), "w") as f:
            f.write(log)
        with open(os.path.join(d, name + ".time"), "w") as f:
            f.write("\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n\tMaximum resident set size (kbytes): %d\n" % rss)
        with open(os.path.join(d, name + ".free"), "w") as f:
            f.write("              total        used\nMem:    134000000000  1000000000\n")
        with open(os.path.join(d, name + ".apps.csv"), "w") as f:
            for i in range(nsamp):
                f.write("2026/10/23 10:00:%02d.000, %d, %d\n" % (i, pid, peak * (i + 1) // nsamp))
                f.write("2026/10/23 10:00:%02d.100, 7, 50\n" % i)
        with open(os.path.join(d, name + ".dev.csv"), "w") as f:
            for i in range(nsamp):
                f.write("2026/10/23 10:00:%02d.000, %d\n" % (i, devpeak * (i + 1) // nsamp))
        with open(os.path.join(d, name + ".pid_tree"), "w") as f:
            f.write(str(tree if tree is not None else pid) + "\n")
        with open(os.path.join(d, "exit_codes.tsv"), "a") as f:
            f.write("%s\t%d\n" % (name, rc))

    with tempfile.TemporaryDirectory() as d:
        log = ("R=0 MEM_GEOM pid=4242 nx=8 ny=8 nz=8 n=512 levels=2 level_cells=512,64 coarsest=4x4x4\n"
               "R=0 MEM_PHASE phase=S2 exact_bytes=%d cufft_work=0 device_used=%d seconds=1.5 items=a:1\n"
               % (2500 * 2**20, 2600 * 2**20))
        write_run(d, "r1", log, 0, 4242, 3000, 3100)
        pid, mx = log_info(os.path.join(d, "r1.log"))
        expect("prefixed_pid", pid == 4242 and mx == 2500 * 2**20)
        pk = peak_of(d, "r1")
        expect("peak_ok", pk["status"] == "OK" and pk["pid_peak"] == 3000)
        lines, ok = peak_line(d, "r1")
        expect("limit_pass", ok and lines[1].endswith("PASS") and "limit_mib=30000" in lines[1])
        write_run(d, "r2", log, 0, 4242, 30001, 30100)
        expect("limit_fail", peak_line(d, "r2")[0][1].endswith("FAIL"))
        write_run(d, "r3", "no pid here\n", 0, 5151, 1000, 1100, tree=5151)
        expect("tree_pid_fallback", peak_of(d, "r3")["status"] == "OK" and peak_of(d, "r3")["pid"] == 5151)
        write_run(d, "r4", log, 0, 4242, 1000, 1100, nsamp=3)
        expect("few_samples", peak_line(d, "r4")[0][1].endswith("NOT_EVALUABLE"))
        write_run(d, "r5", log, 0, 4242, 3000, 3100, tree=1)
        expect("tree_mismatch", peak_of(d, "r5")["status"] == "NOT_EVALUABLE")
        recs = mem_records(d, "r1")
        txt = "\n".join(recs)
        expect("records", "host_max_rss_kb=4096" in txt and "wall_s=62.5" in txt and "MEM_PHASE phase=S2 exact_bytes=%d" % (2500 * 2**20) in txt
               and "R=0 MEM_GEOM" in txt and "level_cells" not in txt)
        expect("absent", peak_of(d, "nope")["status"] == "NOT_EVALUABLE")
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK mem_verdict")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if len(sys.argv) >= 4 and sys.argv[1] == "peak":
        lim = LIMIT_MIB
        if len(sys.argv) == 6 and sys.argv[4] == "--limit":
            lim = int(sys.argv[5])
        ls, ok = peak_line(sys.argv[2], sys.argv[3], lim)
        print("\n".join(ls))
        sys.exit(0 if ok else 1)
    print(__doc__, file=sys.stderr)
    sys.exit(2)
