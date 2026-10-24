#!/usr/bin/env python3
"""SF-38 threshold B (RR2, RR3): production-grid verdicts from the outputs of run_sf38.sh part `prodgrid`.

  b_verdict.py DIR [--runs B1,B2,B3]
  b_verdict.py --selftest

DIR holds, for each run R (default B1 B2 B3): R.log R.time R.free R.apps.csv R.dev.csv R.pid_tree and exit_codes.tsv.
Output:
  B_VERDICT run= grid= exit= correctors=<x|y|z:status,its,rel;...> sf19_converged= mg_levels= stage_mg_levels= auto_levels=
            pid_peak_mib= limit_mib=30000 mem=OK|NOT_EVALUABLE(reasons) PASS|FAIL|NOT_EVALUABLE
  B_ALL PASS|FAIL|NOT_EVALUABLE            (all runs listed; FAIL dominates NOT_EVALUABLE)
  RECORDED ...                              (wall, host RSS, free -b, MEM lines, STATUS / TSTATUS / POLLOCK / TWIN lines)
PASS iff: exit 0; the three SF-19 corrector lines `converged` with rel_residual <= 1e-10 and `SF-19 ... converged=1`;
`mg_levels` printed (SETUP SF-19 line and STAGE line) equals box_auto_mg_levels(nx, ny, nz); PID peak <= 30000 MiB with the
RR2 sampler gates (>= 5 samples, PID peak <= device-wide peak, PID consistent with the process tree).  Sampler gates failing
with everything else met -> NOT_EVALUABLE (re-run with identical settings, disclosed).  Stdlib only.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mem_verdict as mv  # noqa: E402

REL_MAX = 1e-10


def box_auto_mg_levels(nx, ny, nz):
    """Python copy of lagrangian_labels::box_auto_mg_levels (LongBoxStage.cu)."""
    def can(n):
        return n % 2 == 0 and n // 2 >= 4 and (n // 2) % 2 == 0
    lv, a, b, c = 1, nx, ny, nz
    while can(a) and can(b) and can(c):
        a //= 2
        b //= 2
        c //= 2
        lv += 1
    return lv


def parse_run(text):
    """-> dict from the R=0 lines of a runner log."""
    t = "\n".join(mv.strip_prefix(l) for l in (text or "").splitlines() if l.startswith("R=0 "))
    cor = re.findall(r"SETUP {3}SF-19 corrector ([xyz]): status=(\S+) its=(\d+) rel_residual=(\S+)", t)
    m = re.search(r"SETUP {3}SF-19: mg_levels=(\d+) converged=(\d)", t)
    st = re.search(r"^STAGE status=(\S+) grid=(\d+)x(\d+)x(\d+) mg_levels=(\d+) ", t, re.M)
    return {"cor": cor, "mg": int(m.group(1)) if m else None, "conv": int(m.group(2)) if m else None,
            "stage_grid": tuple(int(st.group(i)) for i in (2, 3, 4)) if st else None,
            "stage_mg": int(st.group(5)) if st else None, "stage_status": st.group(1) if st else None}


def verdict(d, run):
    text = mv.rd(os.path.join(d, run + ".log"))
    rc = mv.rc_of(d, run)
    p = parse_run(text)
    pk = mv.peak_of(d, run)
    auto = box_auto_mg_levels(*p["stage_grid"]) if p["stage_grid"] else None
    cor_ok = len(p["cor"]) == 3 and all(s == "converged" and float(r) <= REL_MAX for _, s, _, r in p["cor"]) and p["conv"] == 1
    levels_ok = auto is not None and p["mg"] == auto and p["stage_mg"] == auto
    desc = ";".join("%s:%s,%s,%s" % (c, s, i, r) for c, s, i, r in p["cor"]) or "-"
    mem = "OK" if pk["status"] == "OK" else "NOT_EVALUABLE(%s)" % ",".join(pk["reasons"])
    within = pk["pid_peak"] is not None and pk["pid_peak"] <= mv.LIMIT_MIB
    ok_core = rc == 0 and cor_ok and levels_ok
    if not ok_core or (pk["status"] == "OK" and not within):
        v = "FAIL"
    elif pk["status"] != "OK":
        v = "NOT_EVALUABLE"
    else:
        v = "PASS"
    grid = "x".join(str(x) for x in p["stage_grid"]) if p["stage_grid"] else "-"
    line = ("B_VERDICT run=%s grid=%s exit=%s correctors=%s sf19_converged=%s mg_levels=%s stage_mg_levels=%s auto_levels=%s "
            "pid_peak_mib=%s limit_mib=%d mem=%s %s"
            % (run, grid, rc, desc, "-" if p["conv"] is None else p["conv"], p["mg"], p["stage_mg"], auto, pk["pid_peak"],
               mv.LIMIT_MIB, mem, v))
    return line, v, text


def records(d, run, text):
    out = list(mv.mem_records(d, run))
    keep = ("STATUS", "TSTATUS", "POLLOCK", "TWIN", "BACKFLOW", "TIMING", "STAGE", "ANALYSIS", "RUN_DONE")
    for raw in (text or "").splitlines():
        line = mv.strip_prefix(raw)
        tok = line.split(None, 1)[0] if line.strip() else ""
        if tok in keep:
            out.append("RECORDED run=%s %s" % (run, raw[:320]))
    out.append("RECORDED run=%s outputs: RUN_DONE=%d TWIN_line=%d POLLOCK_line=%d" % (
        run, int("RUN_DONE" in (text or "")), int(" TWIN " in (text or "")), int(" POLLOCK " in (text or ""))))
    return out


def report(d, runs):
    out, vs = [], []
    for r in runs:
        line, v, text = verdict(d, r)
        out.append(line)
        vs.append(v)
        out.extend(records(d, r, text))
    allv = "FAIL" if "FAIL" in vs else ("NOT_EVALUABLE" if "NOT_EVALUABLE" in vs else "PASS")
    out.append("B_ALL %s (%s)" % (allv, ",".join("%s=%s" % (r, v) for r, v in zip(runs, vs))))
    return out, allv


# ---------------------------------------------------------------------------------------------------------------
def selftest():
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    expect("auto_3328", box_auto_mg_levels(3328, 384, 384) == 7)
    expect("auto_2048", box_auto_mg_levels(2048, 256, 256) == 7)
    expect("auto_3328x256", box_auto_mg_levels(3328, 256, 256) == 7)
    expect("auto_cube", box_auto_mg_levels(64, 64, 64) == 5)
    expect("auto_64x16x16", box_auto_mg_levels(64, 16, 16) == 3)

    def log(grid="3328x384x384", mg=7, status="converged", conv=1, rel="3.0e-11"):
        s = ""
        for c in "xyz":
            s += "R=0 SETUP   SF-19 corrector %s: status=%s its=20 rel_residual=%s final_residual=1e-12\n" % (c, status, rel)
        return ("R=0 MEM_GEOM pid=4242 nx=3328 ny=384 nz=384 n=1 levels=7 level_cells=1 coarsest=52x6x6\n"
                "R=0 MEM_PHASE phase=S2 exact_bytes=%d cufft_work=0 device_used=1 seconds=1.5 items=a:1\n"
                "R=0 SETUP   SF-19: mg_levels=%d converged=%d G=(1,1,1)\n" % (2500 * 2**20, mg, conv) + s +
                "R=0 STAGE status=ok grid=%s mg_levels=%d G=1,1,1 darcy_converged=1 flow_ready=1 seconds=1.0\n" % (grid, mg) +
                "R=0 TWIN Np=10000 weights=1/Np\nR=0 POLLOCK Np=10000\nR=0 STATUS ok tracker_non_ok_plane=0\n"
                "TIMING r=0 seconds=3.0\nRUN_DONE nr=1\n")

    def write_run(d, name, text, rc, peak, devpeak, nsamp=20):
        for ext, body in ((".log", text), (".time", "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n\tMaximum resident set size (kbytes): 4096\n"),
                          (".free", "Mem: 1 2\n"), (".pid_tree", "4242\n")):
            with open(os.path.join(d, name + ext), "w") as f:
                f.write(body)
        with open(os.path.join(d, name + ".apps.csv"), "w") as f:
            for i in range(nsamp):
                f.write("2026/10/23 10:00:%02d.000, 4242, %d\n" % (i, peak * (i + 1) // nsamp))
        with open(os.path.join(d, name + ".dev.csv"), "w") as f:
            for i in range(nsamp):
                f.write("2026/10/23 10:00:%02d.000, %d\n" % (i, devpeak * (i + 1) // nsamp))
        with open(os.path.join(d, "exit_codes.tsv"), "a") as f:
            f.write("%s\t%d\n" % (name, rc))

    with tempfile.TemporaryDirectory() as d:
        write_run(d, "B3", log(), 0, 28885, 29000)
        out, allv = report(d, ["B3"])
        expect("pass", out[0].endswith(" PASS") and allv == "PASS" and "pid_peak_mib=28885" in out[0])
        expect("records", any("RECORDED run=B3 R=0 STATUS ok" in l for l in out) and any("outputs: RUN_DONE=1 TWIN_line=1 POLLOCK_line=1" in l for l in out))
        write_run(d, "B3", log(), 0, 30001, 30100)
        expect("over_limit", verdict(d, "B3")[1] == "FAIL")
        write_run(d, "B3", log(rel="2.0e-10"), 0, 28885, 29000)
        expect("rel_over", verdict(d, "B3")[1] == "FAIL")
        write_run(d, "B3", log(status="max_iterations", conv=0), 0, 28885, 29000)
        expect("not_converged", verdict(d, "B3")[1] == "FAIL")
        write_run(d, "B3", log(mg=6), 0, 28885, 29000)
        expect("levels_mismatch", verdict(d, "B3")[1] == "FAIL")
        write_run(d, "B3", log(), 1, 28885, 29000)
        expect("exit_nonzero", verdict(d, "B3")[1] == "FAIL")
        write_run(d, "B3", log(), 0, 28885, 29000, nsamp=3)
        expect("few_samples", verdict(d, "B3")[1] == "NOT_EVALUABLE")
        write_run(d, "B3", log(), 0, 28885, 28000)  # PID peak above the device-wide peak
        expect("pid_above_device", verdict(d, "B3")[1] == "NOT_EVALUABLE")
        write_run(d, "B1", log(grid="2048x256x256", mg=7), 0, 27000, 27100)
        expect("b1_pass", verdict(d, "B1")[1] == "PASS")
        expect("absent", verdict(d, "Bx")[1] == "FAIL")
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK b_verdict")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    args = sys.argv[1:]
    runs = ["B1", "B2", "B3"]
    if len(args) == 3 and args[1] == "--runs":
        runs = args[2].split(",")
        args = args[:1]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    o, v = report(args[0], runs)
    print("\n".join(o))
    sys.exit(0 if v == "PASS" else 1)
