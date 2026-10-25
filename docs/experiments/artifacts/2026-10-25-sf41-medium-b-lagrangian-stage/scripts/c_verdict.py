#!/usr/bin/env python3
"""SF-41 threshold C, controls C1-C5 (pre-registration RR5): verdicts from the logs of run_sf41.sh part `controls`.

  c_verdict.py DIR [--controls RUN] [--pollock RUN] [--c5 RUN] [--dry]
  c_verdict.py --selftest

DIR holds RUN.log of
  C_full    `dirichlet_controls_tests --full`                 (C_CHECK / C_RECORD / C_VERDICT; C1, C2, C3, C4a, C4b tracker + Pollock)
  C4_pollock `pollock_nonperiodic_tests --only C4`            (C4_CHECK / C4_VERDICT; the Pollock non-periodic core, V100 sizes)
  C5        `dirichlet_long_box_stage_tests --only C5`        (C5_CHECK / VERDICT mode=C5)
and exit_codes.tsv.

Every gated `C_CHECK name= grid= path= value= gate= verdict=` is RE-EVALUATED here (gate `<=<x>`: finite value <= x; `==0`: value
== 0) and compared with the program's verdict; each case also requires its REQUIRED (name, grid, path) set of the RR5 full tier
(missing = failure).  Case rules:
  C1  64x16x16, 128x16x16 (k = 1)             C2  64x32x32, 128x32x32 (layered)       C3  64x16x16, 128x16x16 (written field)
  C4  C4a 64x16x16, 128x16x16 and C4b 64x32x32, 128x32x32: tracker (fused, unfused) in `C_full` AND the Pollock non-periodic mode
      (path pollock in `C_full`, and every `C4_CHECK` of C4_pollock: grid/case sets, verdict=PASS, exit counts equal `expected=`)
  C5  C1 on 64x16x16, 128x16x16 and C2 on 64x32x32, 128x32x32: converged=1, rel <= 1e-10, balance <= 1e-8 recomputed from the
      fields; the C1 medium additionally has initial = 0, rel = 0, 0 iterations (the PCG returns at once); the maximum
      relative head difference to the exact head is RECORDED.
Output: C1_VERDICT .. C5_VERDICT lines, `C_VERDICT PASS|FAIL`, RECORDED lines (C_RECORD, C5 head difference).  Exit 0 iff PASS.
`--dry` (local dry run, not evidence): the REQUIRED lists are dropped; a case needs only >= 1 gated check and no failure.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf41lib as L  # noqa: E402

TP = ("fused", "unfused")
C1G, C2G, C3G = ("64x16x16", "128x16x16"), ("64x32x32", "128x32x32"), ("64x16x16", "128x16x16")
C4AG, C4BG = ("64x16x16", "128x16x16"), ("64x32x32", "128x32x32")

REQ = {"C1": [], "C2": [], "C3": [], "C4": []}
for _g in C1G:
    for _n in ("C1_particles_not_ok", "C1_transverse_displacement", "C1_tau_vs_dx1_over_G1_rel",
               "C1_x1_at_times_vs_X0_plus_G1_t_rel", "C1_abs_I", "C1_covectors_constant", "C1_exit_statuses"):
        REQ["C1"] += [(_n, _g, p) for p in TP]
    REQ["C1"] += [(n, _g, "host") for n in ("C1_f_samples_zero_exactly", "C1_coefficients_zero_exactly", "C1_v_equals_G1_e1")]
    REQ["C1"] += [("C1_fused_equals_unfused_bitwise", _g, "fused")]
for _g in C2G:
    for _n in ("C2_particles_not_ok", "C2_transverse_positions_constant", "C2_covectors_constant", "C2_tau_vs_quadrature_rel"):
        REQ["C2"] += [(_n, _g, p) for p in TP]
    REQ["C2"] += [("C2_fused_equals_unfused_bitwise", _g, "fused")]
for _g in C3G:  # C3 case names (st.name) are program-defined: matched by pattern in REQ_PATTERN
    REQ["C3"] += [("C3_domain_report_outside_false_NaN_no_read", _g, "host+fused")]
REQ_PATTERN = {"C3": [(r"^C3_.+_shift_plus_1_violations_1e-14$", p) for p in ("host", "device", "fused")]
               + [(r"^C3_.+_fused_equals_unfused_eval_bitwise$", "fused")]}
for _g in C4AG:
    for _n in ("C4a_particles_not_exit_outlet", "C4a_first_plane_time_not_reached_by_all",
               "C4a_reached_records_nonfinite_or_after_exit_not_NaN", "C4a_exit_counts_inexact"):
        REQ["C4"] += [(_n, _g, p) for p in TP + ("pollock",)]
    REQ["C4"] += [("C4a_fused_equals_unfused_bitwise", _g, "fused"), ("C4a_track_json_and_exit_files", _g, "unfused"),
                  ("C4a_stage_outside_domain_exercised_rejected_steps", _g, "fused+unfused")]
    REQ["C4"] += [("C4a_stage_outside_domain_hmax_2h_statuses", _g, p) for p in TP]
for _g in C4BG:
    for _n in ("C4b_plane_pass_misclassified_seeds", "C4b_time_pass_misclassified_seeds", "C4b_plane_counts_inexact",
               "C4b_time_counts_inexact"):
        REQ["C4"] += [(_n, _g, p) for p in TP + ("pollock",)]
    REQ["C4"] += [("C4b_fused_equals_unfused_bitwise", _g, "fused")]
C4_POLLOCK_SET = [(g, "C4a", m) for g in C4AG for m in ("plane", "time")] + [(g, "C4b", m) for g in C4BG for m in ("plane", "time")]


def parse_checks(text):
    out = []
    for ln in (text or "").splitlines():
        if ln.startswith("C_CHECK "):
            d = L.kv(ln)
            if {"name", "grid", "path", "value", "gate", "verdict"} <= set(d):
                out.append(d)
    return out


def reeval(d):
    v = L.num(d["value"])
    g = d["gate"]
    if v is None or not math.isfinite(v):
        return "FAIL"
    if g == "==0":
        return "PASS" if v == 0 else "FAIL"
    m = re.match(r"^<=([-+0-9.eE]+)$", g)
    if m:
        return "PASS" if v <= float(m.group(1)) else "FAIL"
    m = re.match(r"^>([-+0-9.eE]+)$", g)  # e.g. `>0`: the rejected-step count of the stage-outside-the-domain case
    if m:
        return "PASS" if v > float(m.group(1)) else "FAIL"
    return "FAIL"  # unknown gate grammar is never a pass


def case_verdict(case, checks, dry):
    """-> (ok, n_pass, why)"""
    mine = [d for d in checks if d["name"].startswith(case + "_") or (case == "C4" and d["name"].startswith(("C4a_", "C4b_")))]
    why, npass = [], 0
    seen = set()
    for d in mine:
        r = reeval(d)
        seen.add((d["name"], d["grid"], d["path"]))
        if r != d["verdict"]:
            why.append("inconsistent:%s@%s/%s" % (d["name"], d["grid"], d["path"]))
        if r == "PASS":
            npass += 1
        else:
            why.append("fail:%s@%s/%s=%s" % (d["name"], d["grid"], d["path"], d["value"]))
    if npass == 0:
        why.append("no_gated_checks")
    if not dry:
        for n, g, p in REQ[case]:
            if (n, g, p) not in seen:
                why.append("missing:%s@%s/%s" % (n, g, p))
        for pat, p in REQ_PATTERN.get(case, []):
            for g in C3G:
                if not any(re.match(pat, n) and gg == g and pp == p for (n, gg, pp) in seen):
                    why.append("missing:%s@%s/%s" % (pat, g, p))
    return not why, npass, why


def c4_pollock(text, rc, dry):
    """Pollock non-periodic program: C4_CHECK lines, all PASS, exit counts equal the expected ones."""
    why, seen = [], set()
    for ln in (text or "").splitlines():
        if not ln.startswith("C4_CHECK "):
            continue
        d = L.kv(ln)
        seen.add((d.get("grid"), d.get("case"), d.get("mode")))
        if d.get("verdict") != "PASS":
            why.append("program_fail:%s/%s/%s" % (d.get("grid"), d.get("case"), d.get("mode")))
        exp = dict(kv.split(":") for kv in d.get("expected", "").split(",") if ":" in kv)
        got = {"inlet": d.get("exit_inlet"), "outlet": d.get("exit_outlet"), "reached": d.get("reached")}
        for k, v in exp.items():
            if got.get(k) != v:
                why.append("count_mismatch:%s/%s/%s %s got=%s expected=%s" % (d.get("grid"), d.get("case"), d.get("mode"), k, got.get(k), v))
        if d.get("case") == "C4b" and d.get("mismatches") != "0":
            why.append("misclassified:%s/%s" % (d.get("grid"), d.get("mode")))
    if not dry:
        for t in C4_POLLOCK_SET:
            if t not in seen:
                why.append("missing:%s/%s/%s" % t)
    elif not seen:
        why.append("no_C4_CHECK_lines")
    if dry:  # the default tier prints `ALL PASS`, not the --only C4 verdict line
        if "ALL PASS" not in (text or ""):
            why.append("program_not_ALL_PASS")
    elif not (text or "").count("C4_VERDICT method=pollock PASS"):
        why.append("program_C4_VERDICT_not_PASS")
    if rc != 0:
        why.append("exit=%s" % rc)
    return not why, why


def c5(text, rc, dry):
    why, seen = [], set()
    rows = []
    for ln in (text or "").splitlines():
        if ln.startswith("C5_CHECK "):
            d = L.kv(ln)
            rows.append(d)
            key = (d.get("case"), d.get("size"))
            seen.add(key)
            conv = d.get("converged") == "1"
            rel, bal = L.num(d.get("rel")), L.num(d.get("balance"))
            ini, fin, its = L.num(d.get("initial")), L.num(d.get("final")), d.get("iterations")
            if not conv:
                why.append("not_converged:%s/%s" % key)
            if rel is None or not math.isfinite(rel) or rel > 1e-10:
                why.append("rel:%s/%s=%s" % (key + (d.get("rel"),)))
            if bal is None or not math.isfinite(bal) or bal > 1e-8:
                why.append("balance:%s/%s=%s" % (key + (d.get("balance"),)))
            if d.get("case") == "C1" and not (ini == 0 and rel == 0 and its == "0"):
                why.append("C1_not_instant:%s (initial=%s rel=%s iterations=%s)" % (d.get("size"), d.get("initial"), d.get("rel"), its))
            if d.get("pass") != "1":
                why.append("program_pass_flag:%s/%s" % key)
    if not dry:
        for k in (("C1", "64x16x16"), ("C1", "128x16x16"), ("C2", "64x32x32"), ("C2", "128x32x32")):
            if k not in seen:
                why.append("missing:%s/%s" % k)
    elif not rows:
        why.append("no_C5_CHECK_lines")
    v = [L.kv(x) for x in (text or "").splitlines() if x.startswith("VERDICT ")]
    if not v or v[-1].get("checks_fail") != "0":
        why.append("program_VERDICT_not_clean")
    if rc != 0:
        why.append("exit=%s" % rc)
    return not why, why, rows


def report(d, controls="C_full", pollock="C4_pollock", c5run="C5", dry=False):
    out, allok = [], True
    ct = L.read(os.path.join(d, controls + ".log"))
    crc = L.rc_of(d, controls)
    checks = parse_checks(ct)
    pv = L.lines_of(ct, "C_VERDICT")
    prog_ok = bool(pv) and L.kv(pv[-1]).get("all_pass") == "1" and "C_ERROR" not in (ct or "") and crc == 0
    for case in ("C1", "C2", "C3", "C4"):
        ok, npass, why = case_verdict(case, checks, dry)
        if case == "C4":
            pt = L.read(os.path.join(d, pollock + ".log"))
            pok, pwhy = c4_pollock(pt, L.rc_of(d, pollock), dry)
            ok, why = ok and pok, why + pwhy
        ok = ok and prog_ok
        if not prog_ok:
            why = why + ["controls_program_not_clean(exit=%s)" % crc]
        out.append("%s_VERDICT exit=%s gated_pass=%d %s%s" % (case, crc, npass, "PASS" if ok else "FAIL",
                                                           "" if ok else " [%s]" % ",".join(why[:10])))
        allok = allok and ok
    t5 = L.read(os.path.join(d, c5run + ".log"))
    ok5, why5, rows = c5(t5, L.rc_of(d, c5run), dry)
    out.append("C5_VERDICT exit=%s rows=%d %s%s" % (L.rc_of(d, c5run), len(rows), "PASS" if ok5 else "FAIL",
                                                   "" if ok5 else " [%s]" % ",".join(why5[:10])))
    allok = allok and ok5
    out.append("C_VERDICT %s" % ("PASS" if allok else "FAIL"))
    for ln in (ct or "").splitlines():
        if ln.startswith("C_RECORD "):
            out.append("RECORDED " + ln[:300])
    for r in rows:
        out.append("RECORDED C5 case=%s size=%s iterations=%s initial=%s final=%s rel=%s balance=%s max_head_rel_diff=%s"
                   % (r.get("case"), r.get("size"), r.get("iterations"), r.get("initial"), r.get("final"), r.get("rel"),
                      r.get("balance"), r.get("max_head_rel_diff")))
    return out, allok


# ---------------------------------------------------------------------------------------------------------------
def _synth_controls(drop=None, failname=None):
    L_ = []
    for n, g, p in REQ["C1"] + REQ["C2"] + REQ["C3"] + REQ["C4"]:
        if drop == n:
            continue
        zero = n.endswith(("_exactly", "G1_e1", "_not_ok", "_statuses", "_bitwise", "_no_read")) or "C4" in n
        val, gate = (0.0, "==0") if zero else (1e-15, "<=1e-12")
        if n.endswith("exercised_rejected_steps"):
            val, gate = 3072.0, ">0"
        if failname == n:
            val = 1.0
        out = "PASS" if (val == 0 if gate == "==0" else (val > 0 if gate == ">0" else val <= 1e-12)) else "FAIL"
        L_.append("C_CHECK name=%s grid=%s path=%s value=%.6e gate=%s verdict=%s" % (n, g, p, val, gate, out))
    for g in C3G:
        for p in ("host", "device", "fused"):
            L_.append("C_CHECK name=C3_potential_shift_plus_1_violations_1e-14 grid=%s path=%s value=0.0e+00 gate===0 verdict=PASS" % (g, p))
        L_.append("C_CHECK name=C3_potential_fused_equals_unfused_eval_bitwise grid=%s path=fused value=0 gate===0 verdict=PASS" % g)
    L_.append("C_RECORD name=C2_I_vs_quadrature grid=64x32x32 path=fused max_rel=1e-3")
    fails = sum(1 for x in L_ if "verdict=FAIL" in x)
    L_.append("C_VERDICT all_pass=%d" % int(fails == 0))
    return "\n".join(L_) + "\n"


def _synth_pollock(bad=False):
    L_ = []
    for g in C4AG:
        for m in ("plane", "time"):
            L_.append("C4_CHECK grid=%s case=C4a method=pollock mode=%s exit_inlet=0 exit_outlet=%d reached=%d expected=inlet:0,outlet:%d,reached:%d verdict=PASS"
                      % (g, m, 256, 0 if not bad else 1, 256, 0))
    for g in C4BG:
        L_.append("C4_CHECK grid=%s case=C4b method=pollock mode=plane exit_inlet=768 exit_outlet=0 reached=1792 expected=inlet:768,outlet:0,reached:1792 mismatches=0 verdict=PASS" % g)
        L_.append("C4_CHECK grid=%s case=C4b method=pollock mode=time exit_inlet=768 exit_outlet=1792 reached=0 expected=inlet:768,outlet:1792,reached:0 mismatches=0 verdict=PASS" % g)
    L_.append("C4_VERDICT method=pollock PASS (20 checks, 0 failed)")
    return "\n".join(L_) + "\n"


def _synth_c5(bad=False):
    L_ = []
    for c, s in (("C1", "64x16x16"), ("C1", "128x16x16"), ("C2", "64x32x32"), ("C2", "128x32x32")):
        if c == "C1":
            L_.append("C5_CHECK case=C1 size=%s converged=1 iterations=0 initial=0.000000e+00 final=0.000000e+00 rel=0.000000e+00 balance=0.0e+00 qbar=1 max_head_rel_diff=0.0e+00 pass=1" % s)
        else:
            L_.append("C5_CHECK case=C2 size=%s converged=1 iterations=20 initial=1.0e+03 final=1.0e-08 rel=%s balance=1.0e-12 qbar=1 max_head_rel_diff=3.0e-11 pass=1"
                      % (s, "2.0e-10" if bad else "1.0e-11"))
    L_.append("VERDICT mode=C5 checks_pass=6 checks_fail=0")
    L_.append("ALL_PASS")
    return "\n".join(L_) + "\n"


def selftest():
    import tempfile
    fails = []

    def expect(n, c):
        if not c:
            fails.append(n)
            print("SELFTEST FAIL %s" % n)

    def run(ct, pt, c5t, crc=0, prc=0, c5rc=0, dry=False):
        with tempfile.TemporaryDirectory() as d:
            for nm, t in (("C_full", ct), ("C4_pollock", pt), ("C5", c5t)):
                if t is not None:
                    with open(os.path.join(d, nm + ".log"), "w") as f:
                        f.write(t)
            with open(os.path.join(d, "exit_codes.tsv"), "w") as f:
                f.write("C_full\t%d\nC4_pollock\t%d\nC5\t%d\n" % (crc, prc, c5rc))
            return report(d, dry=dry)

    out, ok = run(_synth_controls(), _synth_pollock(), _synth_c5())
    expect("pass", ok and "C_VERDICT PASS" in out)
    expect("pass_lines", all(any(x.startswith("C%d_VERDICT" % k) and x.endswith("PASS") for x in out) for k in range(1, 6)))
    out, ok = run(_synth_controls(failname="C2_tau_vs_quadrature_rel"), _synth_pollock(), _synth_c5(), crc=1)
    expect("c2_fail", not ok and any(x.startswith("C2_VERDICT") and "FAIL" in x for x in out))
    out, ok = run(_synth_controls(drop="C1_abs_I"), _synth_pollock(), _synth_c5())
    expect("c1_missing", not ok and any(x.startswith("C1_VERDICT") and "missing:C1_abs_I" in x for x in out))
    out, ok = run(_synth_controls(), _synth_pollock(bad=True), _synth_c5())
    expect("pollock_count", not ok and any(x.startswith("C4_VERDICT") and "count_mismatch" in x for x in out))
    out, ok = run(_synth_controls(), _synth_pollock(), _synth_c5(bad=True))
    expect("c5_rel", not ok and any(x.startswith("C5_VERDICT") and "rel:" in x for x in out))
    out, ok = run(_synth_controls(), None, _synth_c5())
    expect("pollock_log_absent", not ok)
    out, ok = run(_synth_controls(), _synth_pollock(), None)
    expect("c5_log_absent", not ok)
    out, ok = run(_synth_controls(drop="C1_abs_I"), _synth_pollock() + "ALL PASS\n", _synth_c5(), dry=True)
    expect("dry_relaxed", ok)
    # a gate the program passed but whose value breaks it is caught
    bad = _synth_controls().replace("C1_transverse_displacement grid=64x16x16 path=fused value=1.000000e-15",
                                    "C1_transverse_displacement grid=64x16x16 path=fused value=3.000000e-12")
    out, ok = run(bad, _synth_pollock(), _synth_c5())
    expect("reevaluation_catches_lie", not ok and any("inconsistent:C1_transverse_displacement" in x for x in out))
    expect("gate_gt0", reeval({"value": "3.0e+03", "gate": ">0"}) == "PASS" and reeval({"value": "0", "gate": ">0"}) == "FAIL"
           and reeval({"value": "nan", "gate": ">0"}) == "FAIL" and reeval({"value": "1", "gate": "weird"}) == "FAIL")
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK c_verdict")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    if not a or a[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    opt = {"--controls": "C_full", "--pollock": "C4_pollock", "--c5": "C5"}
    dry = False
    i = 1
    while i < len(a):
        if a[i] in opt and i + 1 < len(a):
            opt[a[i]] = a[i + 1]
            i += 2
        elif a[i] == "--dry":
            dry = True
            i += 1
        else:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
    o, ok = report(a[0], opt["--controls"], opt["--pollock"], opt["--c5"], dry)
    print("\n".join(o))
    sys.exit(0 if ok else 1)
