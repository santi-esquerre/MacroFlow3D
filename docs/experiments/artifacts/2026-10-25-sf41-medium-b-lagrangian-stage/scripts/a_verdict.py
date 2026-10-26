#!/usr/bin/env python3
"""SF-41 threshold A (pre-registration RR3): verdict from the logs of run_sf41.sh part `controls`.

  a_verdict.py DIR [--full RUN] [--memcheck RUN] [--dry]
  a_verdict.py --selftest

DIR holds RUN.log for `interpolation_nonperiodic_x_tests --full` (default RUN `A_full`) and for the same program under
`compute-sanitizer --tool memcheck ... --full --only domain` (default RUN `A_memcheck`), and exit_codes.tsv.

Every gated `A_CHECK name= box= side= value= gate= verdict=` line is RE-EVALUATED here (names containing `_order_ge_`
pass iff value >= gate, every other gated check iff value <= gate; a non-finite value fails) and compared with the
program's own verdict field; a disagreement is a failure.  A is met iff
  * the program exited 0 and printed `A_VERDICT all_pass=1`,
  * no gated check fails and every REQUIRED (name, box, side) of the RR3 full tier is present (missing = failure),
  * the memcheck run exited 0 and prints `ERROR SUMMARY: 0 errors`.
`--dry` (local dry run, not evidence) drops the REQUIRED list and the memcheck run and only needs a non-empty
passing set.  Output: A_VERDICT line, RECORDED lines for the ungated items (RECORDED A_CHECKs, A_LADDER_ORDER,
A_EMUL, A_MEMORY, A_NOALLOC).  Exit 0 iff PASS.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf41lib as L  # noqa: E402

BOTH = ("host", "device")
Q5 = ("val_max", "val_rms", "grad_max", "grad_rms", "hess_max")
# RR3 full tier: (name, boxes or None = any, sides)
REQUIRED = (
    [("memory_report_formula_mismatch", ("32x16x16",), ("device",))]
    + [("node_interp_rel", ("32x16x16", "64x16x16"), BOTH)]
    + [("pou_value_abs", ("32x16x16", "64x32x32", "128x64x64"), BOTH)]
    + [("repro_%s_%s_rel" % (k, q), ("64x16x16", "128x16x16"), BOTH) for k in ("cubic", "quad") for q in ("value", "gradient")]
    + [("ladder_min_order_ge_" + q, None, BOTH) for q in Q5]
    + [("cpugpu_full_value_rel", ("32x16x16", "64x32x32"), ("both",))]
    + [("cpugpu_evalonly_%s_rel" % q, ("32x16x16", "64x32x32"), ("both",)) for q in ("value", "gradient", "hessian")]
    + [("hessian_path_bitwise_mismatches", ("64x32x32",), BOTH)]
    + [("noalloc_eval_free_bytes_delta", ("64x32x32",), ("device",))]
    + [("prefilter_repeat_workspace_reallocations", ("64x32x32",), ("device",))]
    + [("shift_%s_maxabs" % q, ("32x16x16", "64x32x32"), BOTH) for q in ("value", "gradient", "hessian")]
    + [("domain_margin_nonfinite_or_outside", ("32x16x16", "64x32x32"), BOTH)]
    + [("domain_margin_vs_independent_%s_rel" % q, ("32x16x16", "64x32x32"), BOTH) for q in ("value", "gradient")]
    + [("domain_outside_reported_without_read", ("32x16x16", "64x32x32"), BOTH)]
    + [("domain_half_open_violations", ("32x16x16", "64x32x32"), ("host",))]
)


def parse_checks(text):
    out = []
    for ln in (text or "").splitlines():
        if not ln.startswith("A_CHECK "):
            continue
        d = L.kv(ln)
        if {"name", "box", "side", "value", "gate", "verdict"} <= set(d):
            out.append(d)
    return out


def reevaluate(d):
    """-> 'PASS' | 'FAIL' | 'RECORDED' recomputed from value and gate."""
    g = L.num(d["gate"])
    v = L.num(d["value"])
    if g is None or math.isnan(g):
        return "RECORDED"
    if v is None or not math.isfinite(v):
        return "FAIL"
    ok = v >= g if "_order_ge_" in d["name"] else v <= g
    return "PASS" if ok else "FAIL"


def evaluate(text, rc, mtext, mrc, dry=False):
    checks = parse_checks(text)
    fails, mism, npass, nrec = [], [], 0, 0
    seen = set()
    for d in checks:
        r = reevaluate(d)
        if r == "RECORDED":
            nrec += 1
            continue
        seen.add((d["name"], d["box"], d["side"]))
        if r != d["verdict"]:
            mism.append("%s@%s/%s" % (d["name"], d["box"], d["side"]))
        if r == "PASS":
            npass += 1
        else:
            fails.append("%s@%s/%s=%s" % (d["name"], d["box"], d["side"], d["value"]))
    missing = []
    if not dry:
        for name, boxes, sides in REQUIRED:
            for side in sides:
                if boxes is None:
                    if not any(s[0] == name and s[2] == side for s in seen):
                        missing.append("%s/%s" % (name, side))
                else:
                    for b in boxes:
                        if (name, b, side) not in seen:
                            missing.append("%s@%s/%s" % (name, b, side))
    pv = L.lines_of(text, "A_VERDICT")
    prog = L.kv(pv[-1]) if pv else {}
    prog_ok = prog.get("all_pass") == "1" and prog.get("checks_fail") == "0"
    m = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+error", mtext or "")
    mem_errors = int(m[-1]) if m else None
    mem_ok = dry or (mrc == 0 and mem_errors == 0)
    ok = rc == 0 and prog_ok and not fails and not mism and not missing and npass > 0 and mem_ok
    info = {"npass": npass, "nfail": len(fails), "nrec": nrec, "missing": missing, "fails": fails, "mism": mism,
            "prog_ok": prog_ok, "mem_errors": mem_errors, "mrc": mrc, "rc": rc}
    return ok, info


def report(d, full="A_full", mem="A_memcheck", dry=False):
    text = L.read(os.path.join(d, full + ".log"))
    mtext = L.read(os.path.join(d, mem + ".log"))
    rc, mrc = L.rc_of(d, full), L.rc_of(d, mem)
    out = []
    if text is None:
        return ["A_VERDICT run=%s exit=- FAIL (no log)" % full], False
    ok, i = evaluate(text, rc, mtext, mrc, dry)
    out.append("A_VERDICT run=%s exit=%s program_all_pass=%d gated_pass=%d gated_fail=%d inconsistent=%d missing=%d "
               "memcheck_exit=%s memcheck_errors=%s %s%s"
               % (full, rc, int(i["prog_ok"]), i["npass"], i["nfail"], len(i["mism"]), len(i["missing"]), mrc,
                  i["mem_errors"], "PASS" if ok else "FAIL",
                  "" if ok else " [fails=%s inconsistent=%s missing=%s]" % (
                      ",".join(i["fails"][:8]) or "-", ",".join(i["mism"][:8]) or "-", ",".join(i["missing"][:8]) or "-")))
    for ln in text.splitlines():
        if ln.startswith(("A_LADDER_ORDER", "A_EMUL", "A_MEMORY", "A_NOALLOC", "A_TIER")):
            out.append("RECORDED " + ln[:300])
        elif ln.startswith("A_CHECK ") and ln.rstrip().endswith("verdict=RECORDED"):
            out.append("RECORDED " + ln[:300])
    return out, ok


# ---------------------------------------------------------------------------------------------------------------
def _synth(fail=None, drop=None, order_value=None):
    lines = ["A_TIER tier=full only=all"]
    for name, boxes, sides in REQUIRED:
        for b in (boxes or ("32x16x16..64x32x32..128x64x64",)):
            for s in sides:
                if drop == name:
                    continue
                ge = "_order_ge_" in name
                gate = 3.8 if ge else 1e-12
                val = (order_value if (order_value is not None and ge) else (4.2 if ge else 1e-15))
                if fail == name:
                    val = 0.1 if ge else 1.0
                ok = (val >= gate) if ge else (val <= gate)
                lines.append("A_CHECK name=%s box=%s side=%s value=%.6e gate=%.3e verdict=%s"
                             % (name, b, s, val, gate, "PASS" if ok else "FAIL"))
    lines.append("A_CHECK name=pou_gradient_abs_ps box=32x16x16 side=host value=1.0e-13 gate=nan verdict=RECORDED")
    lines.append("A_LADDER_ORDER kind=3d side=host pair=16->32 val_max=3.9")
    nf = sum(1 for l in lines if l.startswith("A_CHECK") and "verdict=FAIL" in l)
    ng = sum(1 for l in lines if l.startswith("A_CHECK") and "verdict=PASS" in l)
    lines.append("A_VERDICT all_pass=%d checks_pass=%d checks_fail=%d" % (int(nf == 0), ng, nf))
    return "\n".join(lines) + "\n"


def selftest():
    import tempfile
    fails = []

    def expect(n, c):
        if not c:
            fails.append(n)
            print("SELFTEST FAIL %s" % n)

    mem_ok = "========= ERROR SUMMARY: 0 errors\n"
    ok, i = evaluate(_synth(), 0, mem_ok, 0)
    expect("pass", ok and i["nfail"] == 0 and not i["missing"])
    ok, i = evaluate(_synth(fail="node_interp_rel"), 1, mem_ok, 0)
    expect("fail_value", not ok and i["nfail"] > 0)
    ok, i = evaluate(_synth(drop="shift_value_maxabs"), 0, mem_ok, 0)
    expect("missing_required", not ok and any("shift_value_maxabs" in m for m in i["missing"]))
    ok, i = evaluate(_synth(fail="ladder_min_order_ge_hess_max"), 1, mem_ok, 0)
    expect("order_gate_sense", not ok)
    ok, i = evaluate(_synth(order_value=3.7), 1, mem_ok, 0)
    expect("order_below_gate", not ok)
    ok, i = evaluate(_synth(), 0, "========= ERROR SUMMARY: 2 errors\n", 0)
    expect("memcheck_errors", not ok and i["mem_errors"] == 2)
    ok, i = evaluate(_synth(), 0, None, None)
    expect("memcheck_absent", not ok)
    ok, i = evaluate(_synth(), 2, mem_ok, 0)
    expect("exit_nonzero", not ok)
    # a program that lies (verdict field contradicts value/gate) is inconsistent
    bad = _synth().replace("name=node_interp_rel box=32x16x16 side=host value=1.000000e-15",
                           "name=node_interp_rel box=32x16x16 side=host value=5.000000e-10")
    ok, i = evaluate(bad, 0, mem_ok, 0)
    expect("inconsistent_verdict_field", not ok and i["mism"])
    ok, i = evaluate("garbage\n", 0, mem_ok, 0)
    expect("empty", not ok)
    ok, i = evaluate(_synth(drop="shift_value_maxabs"), 0, None, None, dry=True)
    expect("dry_relaxed", ok)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "A_full.log"), "w") as f:
            f.write(_synth())
        with open(os.path.join(d, "A_memcheck.log"), "w") as f:
            f.write(mem_ok)
        with open(os.path.join(d, "exit_codes.tsv"), "w") as f:
            f.write("A_full\t0\nA_memcheck\t0\n")
        out, ok = report(d)
        expect("report", ok and out[0].startswith("A_VERDICT run=A_full exit=0") and out[0].endswith("PASS")
               and any(l.startswith("RECORDED A_LADDER_ORDER") for l in out))
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK a_verdict")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    if not a or a[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    full, mem, dry = "A_full", "A_memcheck", False
    i = 1
    while i < len(a):
        if a[i] == "--full" and i + 1 < len(a):
            full = a[i + 1]
            i += 2
        elif a[i] == "--memcheck" and i + 1 < len(a):
            mem = a[i + 1]
            i += 2
        elif a[i] == "--dry":
            dry = True
            i += 1
        else:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
    o, ok = report(a[0], full, mem, dry)
    print("\n".join(o))
    sys.exit(0 if ok else 1)
