#!/usr/bin/env python3
"""SF-40 criterion D instrument (RR5): per-PID device-memory peak from the raw nvidia-smi sampler files.

  pid_peak.py --apps APPS.csv --dev DEV.csv (--pid N | --log LOG) [--tree-pid N] [--min-samples 5]
  pid_peak.py --selftest

APPS.csv : `nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,noheader,nounits -lms 200`
           (all processes, all GPUs; rows `<timestamp>, <pid>, <MiB>`).
DEV.csv  : `nvidia-smi --query-gpu=timestamp,memory.used ... -lms 200` (rows `<timestamp>, <MiB>`; a one-column file
           of MiB values is also accepted).
PID      : from the `MEM_GEOM pid=` / `CONTEXT_PROBE pid=` line of --log (the process that created the CUDA context),
           cross-checked with --tree-pid (the descendant of /usr/bin/time found by pgrep -P); a mismatch is NOT_EVALUABLE.
Gates (RR5): >= 5 samples of the PID; PID peak >= max exact_bytes / 2^20 over the MEM_PHASE lines of the log (when present);
PID peak <= device-wide peak.  Output one line
  PID_PEAK pid= samples= pid_peak_mib= dev_peak_mib= max_exact_mib= tree_pid= status=OK|NOT_EVALUABLE reasons=...
Exit 0 iff OK, 1 NOT_EVALUABLE, 2 usage / unreadable input.  Stdlib only.
"""
import re
import sys


def read_apps(path):
    """-> {pid: [MiB, ...]} ; rows whose pid or memory are not integers are ignored (e.g. `[N/A]`)."""
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            try:
                pid, mem = int(parts[-2]), int(parts[-1])
            except ValueError:
                continue
            out.setdefault(pid, []).append(mem)
    return out


def read_dev(path):
    vals = []
    with open(path, errors="replace") as f:
        for line in f:
            tok = line.strip().split(",")[-1].strip()
            try:
                vals.append(int(tok))
            except ValueError:
                continue
    return vals


def log_info(path):
    """-> (pid or None, max_exact_bytes or None)"""
    pid, mx = None, None
    with open(path, errors="replace") as f:
        for line in f:
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


def evaluate(apps, dev, pid, max_exact_bytes=None, tree_pid=None, min_samples=5):
    reasons = []
    if pid is None:
        return {"status": "NOT_EVALUABLE", "reasons": ["no_pid"], "pid": None, "samples": 0, "pid_peak": None,
                "dev_peak": max(dev) if dev else None, "max_exact_mib": None, "tree_pid": tree_pid}
    if tree_pid is not None and tree_pid != pid:
        reasons.append("pid_mismatch(log=%d,tree=%d)" % (pid, tree_pid))
    series = apps.get(pid, [])
    peak = max(series) if series else None
    dpeak = max(dev) if dev else None
    mx = max_exact_bytes / 2.0**20 if max_exact_bytes is not None else None
    if len(series) < min_samples:
        reasons.append("samples<%d(%d)" % (min_samples, len(series)))
    if peak is not None and mx is not None and peak < mx:
        reasons.append("pid_peak<max_exact(%d<%.3f)" % (peak, mx))
    if not dev:
        reasons.append("no_device_samples")
    elif peak is not None and peak > dpeak:
        reasons.append("pid_peak>device_peak(%d>%d)" % (peak, dpeak))
    return {"status": "NOT_EVALUABLE" if reasons else "OK", "reasons": reasons, "pid": pid, "samples": len(series),
            "pid_peak": peak, "dev_peak": dpeak, "max_exact_mib": mx, "tree_pid": tree_pid}


def fmt(r):
    return "PID_PEAK pid=%s samples=%d pid_peak_mib=%s dev_peak_mib=%s max_exact_mib=%s tree_pid=%s status=%s reasons=%s" % (
        r["pid"], r["samples"], r["pid_peak"], r["dev_peak"],
        "-" if r["max_exact_mib"] is None else "%.3f" % r["max_exact_mib"],
        r["tree_pid"], r["status"], ",".join(r["reasons"]) if r["reasons"] else "-")


def selftest():
    import os
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    with tempfile.TemporaryDirectory() as d:
        ap = os.path.join(d, "apps.csv")
        dv = os.path.join(d, "dev.csv")
        lg = os.path.join(d, "run.log")
        rows = []
        for i in range(12):  # other process 999 on another GPU, ours 4242 growing to 3000 MiB
            rows.append("2026/10/22 10:00:%02d.000, 999, 100" % i)
            rows.append("2026/10/22 10:00:%02d.200, 4242, %d" % (i, 300 + 250 * i))
        rows.append("2026/10/22 10:00:13.000, 4242, [N/A]")
        with open(ap, "w") as f:
            f.write("\n".join(rows) + "\n")
        with open(dv, "w") as f:
            f.write("\n".join("2026/10/22 10:00:%02d.000, %d" % (i, 400 + 250 * i + 100) for i in range(12)) + "\n")
        with open(lg, "w") as f:
            f.write("noise\nMEM_GEOM pid=4242 nx=8 ny=8 nz=8 n=512\nMEM_PHASE phase=S2 exact_bytes=%d cufft_work=0 items=a:1\n"
                    % (2500 * 2**20))
        apps, dev = read_apps(ap), read_dev(dv)
        pid, mx = log_info(lg)
        expect("pid_from_log", pid == 4242 and mx == 2500 * 2**20)
        r = evaluate(apps, dev, pid, mx, tree_pid=4242)
        expect("ok", r["status"] == "OK" and r["pid_peak"] == 3050 and r["samples"] == 12 and r["dev_peak"] == 3250)
        expect("fmt", "status=OK" in fmt(r))
        expect("pid_filter", evaluate(apps, dev, 999, None)["pid_peak"] == 100)
        expect("tree_mismatch", evaluate(apps, dev, pid, mx, tree_pid=1)["status"] == "NOT_EVALUABLE")
        expect("few_samples", evaluate({4242: [1, 2, 3]}, dev, 4242, None)["status"] == "NOT_EVALUABLE")
        expect("peak_below_exact", evaluate(apps, dev, pid, 4000 * 2**20)["status"] == "NOT_EVALUABLE")
        expect("peak_above_device", evaluate(apps, [100] * 12, pid, mx)["status"] == "NOT_EVALUABLE")
        expect("unknown_pid", evaluate(apps, dev, 77, None)["status"] == "NOT_EVALUABLE")
        expect("no_pid", evaluate(apps, dev, None)["status"] == "NOT_EVALUABLE")
        # one-column device file and CONTEXT_PROBE log
        with open(dv, "w") as f:
            f.write("\n".join(str(3300) for _ in range(12)) + "\n")
        expect("dev_one_column", max(read_dev(dv)) == 3300)
        with open(lg, "w") as f:
            f.write("CONTEXT_PROBE pid=4242 device=0 driver=12040 runtime=12020 device_used=123456\n")
        expect("probe_pid", log_info(lg) == (4242, None))
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK pid_peak")
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
    opts = {"--apps": None, "--dev": None, "--pid": None, "--log": None, "--tree-pid": None, "--min-samples": "5"}
    i = 1
    while i < len(argv):
        if argv[i] in opts and i + 1 < len(argv):
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            print("pid_peak.py: bad argument %r" % argv[i], file=sys.stderr)
            return 2
    if not opts["--apps"] or not opts["--dev"] or (opts["--pid"] is None and not opts["--log"]):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        apps, dev = read_apps(opts["--apps"]), read_dev(opts["--dev"])
        pid, mx = None, None
        if opts["--log"]:
            pid, mx = log_info(opts["--log"])
        if opts["--pid"] is not None:
            pid = int(opts["--pid"])
    except OSError as e:
        print("UNREADABLE %s" % e)
        return 2
    tree = int(opts["--tree-pid"]) if opts["--tree-pid"] not in (None, "", "none") else None
    r = evaluate(apps, dev, pid, mx, tree, int(opts["--min-samples"]))
    print(fmt(r))
    return 0 if r["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
