#!/usr/bin/env python3
"""SF-41 threshold D (pre-registration RR6): lean stage bitwise against SF-38 `DirichletMedium`, from the log of
`dirichlet_long_box_stage_tests --only D` (run by run_sf41.sh part `controls` with the RR2 memory sampler).

  d_verdict.py DIR [--run D] [--c0-log PATH] [--dry]
  d_verdict.py --selftest

D is met iff
  * the program exited 0 and printed `VERDICT mode=D ... checks_fail=0`;
  * for every size 128x32x32, 512x64x64, 3328x256x256 a `D_CHECK size=` line has head_identical=1 and bsolver_identical=1, and the
    `D_REF` / `D_LEAN` B_SOLVER lines (compared HERE, string-equal after removing the trailing ` size=`) are identical;
  * the mapped-host head variant is identical at the two small sizes (mapped_identical=1) and not run (na) at 3328x256x256;
Memory (RR2 sampler, whole process) is RECORDED, not gated: `RECORDED D_MEM` carries the PID peak, the device-wide peak, the sample count
and the sampler status; threshold D is the bitwise comparison only (the 30000 MiB gate belongs to E), so D fails only on an RR6 comparison
or a non-zero exit.
Recorded: every B_SOLVER line, the comparison of the 3328x256x256 line with SF-38's committed C0 line (`--c0-log`, default the
committed bpilot.log; an environment control, no gate), the MEM_PHASE lines and wall/RSS.  `--dry` (local, not evidence) uses
the fast-tier sizes found in the log (every size must have head/B_SOLVER identical; mapped identical where run), no D_MEM record.
Output: D_VERDICT ... PASS|FAIL.  Exit 0 iff PASS.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf41lib as L  # noqa: E402
import mem_verdict as mv  # noqa: E402

SIZES = ("128x32x32", "512x64x64", "3328x256x256")
SOLVER_FIELDS = ("converged", "iterations", "initial", "final", "rel", "qbar", "Q_in", "Q_out", "balance", "max_div", "fallback")
HERE = os.path.dirname(os.path.abspath(__file__))
C0_LOG = os.path.normpath(os.path.join(HERE, "..", "..", "2026-10-23-sf38-long-domain-infrastructure", "raw", "sf38_campaign",
                                       "bpilot", "bpilot.log"))


def solver_lines(text, tag):
    """-> {size: B_SOLVER text without the trailing ` size=..`} for lines `<tag> B_SOLVER ... size=S`."""
    out = {}
    for ln in (text or "").splitlines():
        if ln.startswith(tag + " B_SOLVER "):
            m = re.match(r"^" + tag + r" (B_SOLVER .*) size=(\S+)\s*$", ln)
            if m:
                out[m.group(2)] = m.group(1)
    return out


def d_checks(text):
    out = {}
    for ln in (text or "").splitlines():
        if ln.startswith("D_CHECK "):
            d = L.kv(ln)
            out[d.get("size")] = d
    return out


def evaluate(text, rc, dry=False):
    ref, lean, chk = solver_lines(text, "D_REF"), solver_lines(text, "D_LEAN"), d_checks(text)
    why, rows = [], []
    sizes = sorted(chk, key=lambda s: [int(x) for x in s.split("x")]) if dry else list(SIZES)
    if dry and not sizes:
        why.append("no_D_CHECK")
    for s in sizes:
        c = chk.get(s)
        if c is None:
            why.append("missing_D_CHECK:%s" % s)
            continue
        same = s in ref and s in lean and ref[s] == lean[s]
        h, b, m = c.get("head_identical") == "1", c.get("bsolver_identical") == "1", c.get("mapped_identical")
        big = s == "3328x256x256"
        m_ok = (m == "na") if big else (m == "1")
        if dry:
            m_ok = m in ("1", "na")
        rows.append("%s:head=%d,bsolver=%d,lines_equal=%d,mapped=%s" % (s, h, b, same, m))
        if not (h and b and same and m_ok):
            why.append("not_identical:%s" % s)
    v = [L.kv(x) for x in (text or "").splitlines() if x.startswith("VERDICT ")]
    if not v or (v[-1].get("mode") != "D" and not dry) or v[-1].get("checks_fail") != "0":
        why.append("program_VERDICT_not_clean")
    if rc != 0:
        why.append("exit=%s" % rc)
    return not why, why, rows, ref, lean


def c0_compare(lean_line, c0_text):
    """Field-by-field comparison of the lean 3328x256x256 line with SF-38's C0 line; -> (identical_fields, differing, available)."""
    c0 = [x for x in (c0_text or "").splitlines() if L.strip_prefix(x).startswith("B_SOLVER ")]
    if not c0 or not lean_line:
        return [], [], False
    a, b = L.kv(lean_line), L.kv(L.strip_prefix(c0[-1]))
    same = [k for k in SOLVER_FIELDS if a.get(k) == b.get(k)]
    diff = ["%s(lean=%s,c0=%s)" % (k, a.get(k), b.get(k)) for k in SOLVER_FIELDS if a.get(k) != b.get(k)]
    return same, diff, True


def report(d, run="D", c0_log=C0_LOG, dry=False):
    text = L.read(os.path.join(d, run + ".log"))
    rc = L.rc_of(d, run)
    if text is None:
        return ["D_VERDICT run=%s exit=%s FAIL (no log)" % (run, rc)], False
    ok, why, rows, ref, lean = evaluate(text, rc, dry)
    out = []
    # Memory is RECORDED, never gated: threshold D is the bitwise comparison only (the 30000 MiB gate belongs to E).
    mem_txt = "SKIPPED_DRY"
    if not dry:
        pk = mv.peak_of(d, run)
        mem_txt = "pid_peak_mib=%s dev_peak_mib=%s samples=%s sampler=%s" % (pk["pid_peak"], pk["dev_peak"], pk["samples"], pk["status"])
        out.append("RECORDED D_MEM run=%s %s reasons=%s (recorded, not gated)" % (run, mem_txt, ",".join(pk["reasons"]) or "-"))
    verdict = "PASS" if ok else "FAIL"
    out.insert(0, "D_VERDICT run=%s exit=%s sizes=[%s] %s%s" % (
        run, rc, "; ".join(rows), verdict, "" if ok else " [%s]" % ",".join(why[:8])))
    for s in sorted(ref):
        out.append("RECORDED D_REF size=%s %s" % (s, ref[s]))
        out.append("RECORDED D_LEAN size=%s %s" % (s, lean.get(s, "-")))
    if "3328x256x256" in lean:
        same, diff, avail = c0_compare(lean["3328x256x256"], L.read(c0_log))
        if avail:
            out.append("RECORDED D_C0_COMPARE size=3328x256x256 identical_fields=%d/%d differing=[%s] (environment control vs SF-38 C0 line)"
                       % (len(same), len(SOLVER_FIELDS), ",".join(diff) or "-"))
        else:
            out.append("RECORDED D_C0_COMPARE unavailable (%s)" % c0_log)
    out.extend(x for x in (mv.mem_records(d, run) if not dry else []))
    return out, verdict == "PASS"


# ---------------------------------------------------------------------------------------------------------------
def _line(n):
    return "B_SOLVER r=0 converged=1 iterations=%d initial=6.724494e+06 final=6.526221e-04 rel=9.705149e-11 qbar=1.0 Q_in=1.0 Q_out=1.0 balance=2.4e-10 max_div=1.0e-07 fallback=0" % n


def _synth(ident=True, mapped_big="na", extra_lean_diff=False, fail_prog=False):
    L_ = []
    for s in SIZES:
        L_.append("D_REF %s size=%s" % (_line(50), s))
        L_.append("D_LEAN %s size=%s" % (_line(51 if (extra_lean_diff and s == SIZES[1]) else 50), s))
        big = s == SIZES[2]
        L_.append("D_CHECK size=%s head_identical=%d bsolver_identical=%d mapped_identical=%s"
                  % (s, int(ident), int(ident), mapped_big if big else "1"))
    L_.append("VERDICT mode=D checks_pass=12 checks_fail=%d" % (1 if fail_prog else 0))
    return "\n".join(L_) + "\n"


def selftest():
    import tempfile
    fails = []

    def expect(n, c):
        if not c:
            fails.append(n)
            print("SELFTEST FAIL %s" % n)

    expect("pass", evaluate(_synth(), 0)[0])
    expect("head_differs", not evaluate(_synth(ident=False), 0)[0])
    expect("lines_differ_even_if_flag_says_equal", not evaluate(_synth(extra_lean_diff=True), 0)[0])
    expect("mapped_run_at_big_size_is_not_expected", not evaluate(_synth(mapped_big="1"), 0)[0])
    expect("program_fail", not evaluate(_synth(fail_prog=True), 0)[0])
    expect("exit", not evaluate(_synth(), 1)[0])
    expect("missing_size", not evaluate(_synth().replace("D_CHECK size=512x64x64", "X size=512x64x64"), 0)[0])
    expect("dry", evaluate("D_REF %s size=64x16x16\nD_LEAN %s size=64x16x16\nD_CHECK size=64x16x16 head_identical=1 bsolver_identical=1 "
                           "mapped_identical=1\nVERDICT mode=D checks_pass=3 checks_fail=0\n" % (_line(5), _line(5)), 0, dry=True)[0])
    s, dd, av = c0_compare(_line(50), "R=0 " + _line(50).replace("r=0 ", "r=0 ") + "\n")
    expect("c0_same", av and not dd and len(s) == len(SOLVER_FIELDS))
    s, dd, av = c0_compare(_line(51), "R=0 " + _line(50) + "\n")
    expect("c0_diff", av and any(x.startswith("iterations") for x in dd))
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "D.log"), "w") as f:
            f.write(_synth() + "MEM_PHASE phase=S2 exact_bytes=%d cufft_work=0 device_used=1 seconds=1.0 items=a:1\n" % (2500 * 2**20))
        for ext, body in ((".time", "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:00.0\n\tMaximum resident set size (kbytes): 5\n"),
                          (".free", "Mem: 1 2\n"), (".pid_tree", "4242\n")):
            with open(os.path.join(d, "D" + ext), "w") as f:
                f.write(body)
        with open(os.path.join(d, "D.apps.csv"), "w") as f:
            f.write("".join("2026/10/25 10:00:%02d.000, 4242, %d\n" % (i, 100 * (i + 1) + 2500) for i in range(12)))
        with open(os.path.join(d, "D.dev.csv"), "w") as f:
            f.write("".join("2026/10/25 10:00:%02d.000, %d\n" % (i, 100 * (i + 1) + 2600) for i in range(12)))
        with open(os.path.join(d, "exit_codes.tsv"), "w") as f:
            f.write("D\t0\n")
        c0 = os.path.join(d, "c0.log")
        with open(c0, "w") as f:
            f.write("R=0 " + _line(50) + "\n")
        out, ok = report(d, c0_log=c0)
        expect("report_pass", ok and out[0].endswith("PASS") and any("D_C0_COMPARE size=3328x256x256 identical_fields=11/11" in x for x in out))
        with open(os.path.join(d, "D.apps.csv"), "w") as f:  # PID peak above 30000 MiB: recorded, D still PASS
            f.write("".join("2026/10/25 10:00:%02d.000, 4242, %d\n" % (i, 31000) for i in range(12)))
        with open(os.path.join(d, "D.dev.csv"), "w") as f:
            f.write("".join("2026/10/25 10:00:%02d.000, %d\n" % (i, 31100) for i in range(12)))
        out, ok = report(d, c0_log=c0)
        expect("mem_over_limit_not_gated", ok and out[0].endswith("PASS") and any("D_MEM" in x and "pid_peak_mib=31000" in x and "dev_peak_mib=31100" in x for x in out))
        with open(os.path.join(d, "D.apps.csv"), "w") as f:  # sampler unusable (too few samples): recorded, D still PASS
            f.write("2026/10/25 10:00:00.000, 4242, 100\n")
        out, ok = report(d, c0_log=c0)
        expect("mem_not_evaluable_not_gated", ok and any("D_MEM" in x and "sampler=NOT_EVALUABLE" in x for x in out))
        with open(os.path.join(d, "exit_codes.tsv"), "w") as f:  # a non-zero exit still fails D
            f.write("D\t1\n")
        out, ok = report(d, c0_log=c0)
        expect("exit_nonzero_fails", not ok and out[0].startswith("D_VERDICT run=D exit=1") and out[0].endswith("FAIL [exit=1]"))
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK d_verdict")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    if not a or a[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    run, c0, dry = "D", C0_LOG, False
    i = 1
    while i < len(a):
        if a[i] == "--run" and i + 1 < len(a):
            run = a[i + 1]
            i += 2
        elif a[i] == "--c0-log" and i + 1 < len(a):
            c0 = a[i + 1]
            i += 2
        elif a[i] == "--dry":
            dry = True
            i += 1
        else:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
    o, ok = report(a[0], run, c0, dry)
    print("\n".join(o))
    sys.exit(0 if ok else 1)
