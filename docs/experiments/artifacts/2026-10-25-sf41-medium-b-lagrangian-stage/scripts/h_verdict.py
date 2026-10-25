#!/usr/bin/env python3
"""SF-41 threshold H, SF-38 A/B re-run with the exit-aware Pollock (pre-registration RR10): reading of run_sf41.sh part `abrerun`.

  h_verdict.py DIR [--ref-root PATH] [--nr 10] [--lambda1 0.0390625] [--lambda4 0.125]
  h_verdict.py --selftest

DIR holds the two head runs ab_s1 (sigma^2 = 1, 2048x256x256) and ab_s4 (sigma^2 = 4, 3328x256x256), their logs and exit_codes.tsv.
--ref-root = the committed SF-38 campaign tree (default: <repo>/docs/experiments/artifacts/2026-10-23-sf38-long-domain-infrastructure/
raw/sf38_campaign/ab) with ab_s{1,4}/realizations/r_000k/{e2_a,e2_b}.csv, ab_s*/ab_primary.csv and ab_decision.out.

Per realization (`H_REALIZATION`): the exit counts of realizations/r_000k/exits_b.csv (per plane and the `# exit_inlet_total` /
`# exit_outlet_total` lines: particles of medium (b) that left through x = 0 / x = L1 before the plane), and the byte comparison of
e2_b.csv (the exit-aware Pollock) and e2_a.csv (periodic Pollock, unchanged code: the ENVIRONMENT control) with the committed SF-38 file of the
same realization.  Failure policy (specification, RR10), per realization:
  OK_ZERO_EXITS_IDENTICAL   no exit, e2_b and e2_a identical
  EXITS_RECORDED            exits > 0, e2_a identical (e2_b may differ: recorded; the SF-38 decision is not re-opened)
  DEFECT_EXIT_AWARE_POLLOCK no exit, e2_b different while the control e2_a is identical           -> BLOCKING
  ENVIRONMENT_INVESTIGATE   e2_a different                                                         -> investigate and record first
  MISSING                   a needed file is absent
If any realization has exits (inlet or outlet) the SF-38 `ab_decision.py` (unchanged copy) is run on the head ab_primary.csv files and its
C_STAT / C_AMP / C_VERDICT lines are recorded as `RECORDED H_RECOMPUTED ...` beside the originals `RECORDED H_ORIGINAL ...` from the
committed ab_decision.out.  Output: `H_VERDICT amp=..` per amplitude and `H_VERDICT ALL class=<worst class> exits_inlet=.. exits_outlet=..
OK|NOT_OK`.  Exit 0 iff no realization is DEFECT / ENVIRONMENT / MISSING and both runs exited 0 (exits themselves are only RECORDED).
"""
import filecmp
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf41lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "2026-10-23-sf38-long-domain-infrastructure", "raw", "sf38_campaign", "ab"))
AMPS = (("s1", "ab_s1", 0.0390625), ("s4", "ab_s4", 0.125))
WORST = ["OK_ZERO_EXITS_IDENTICAL", "EXITS_RECORDED", "DEFECT_EXIT_AWARE_POLLOCK", "ENVIRONMENT_INVESTIGATE", "MISSING"]


def read_exits(path):
    """-> dict(rows=[(x1, inlet, outlet)], np, inlet_total, outlet_total) or None."""
    t = L.read(path)
    if t is None:
        return None
    rows, tot = [], {}
    for ln in t.splitlines()[1:]:
        if ln.startswith("#"):
            if "=" in ln:
                k, v = ln[1:].strip().split("=", 1)
                tot[k.strip()] = v.strip()
        elif ln.strip():
            p = ln.split(",")
            rows.append((p[0], int(p[1]), int(p[2])))
    try:
        return {"rows": rows, "np": int(tot["Np"]), "inlet": int(tot["exit_inlet_total"]), "outlet": int(tot["exit_outlet_total"])}
    except (KeyError, ValueError):
        return None


def same(a, b):
    if not os.path.exists(a) or not os.path.exists(b):
        return "MISSING"
    return "IDENTICAL" if filecmp.cmp(a, b, shallow=False) else "DIFF"


def classify(exits_total, e2b, e2a):
    if "MISSING" in (e2b, e2a):
        return "MISSING"
    if e2a == "DIFF":
        return "ENVIRONMENT_INVESTIGATE"
    if exits_total == 0:
        return "OK_ZERO_EXITS_IDENTICAL" if e2b == "IDENTICAL" else "DEFECT_EXIT_AWARE_POLLOCK"
    return "EXITS_RECORDED"


def report(d, ref_root=REF_ROOT, nr=10, lam1=0.0390625, lam4=0.125):
    out, classes = [], []
    tot_in = tot_out = 0
    runs_ok = True
    any_exit = False
    for amp, run, _ in AMPS:
        rc = L.rc_of(d, run)
        runs_ok = runs_ok and rc == 0
        ident_b = ident_a = 0
        amp_in = amp_out = 0
        amp_cls = []
        for r in range(nr):
            rn = "r_%04d" % r
            hd = os.path.join(d, run, "realizations", rn)
            rd = os.path.join(ref_root, run, "realizations", rn)
            ex = read_exits(os.path.join(hd, "exits_b.csv"))
            e2b, e2a = same(os.path.join(hd, "e2_b.csv"), os.path.join(rd, "e2_b.csv")), same(os.path.join(hd, "e2_a.csv"), os.path.join(rd, "e2_a.csv"))
            if ex is None:
                cls = "MISSING"
                itot = otot = -1
                npl = 0
            else:
                itot, otot = ex["inlet"], ex["outlet"]
                npl = sum(1 for _, i, o in ex["rows"] if i or o)
                cls = classify(itot + otot, e2b, e2a)
                amp_in += itot
                amp_out += otot
                any_exit = any_exit or (itot + otot) > 0
                out.append("RECORDED H_EXITS_BY_PLANE amp=%s r=%d Np=%d x1=%s inlet=%s outlet=%s" % (
                    amp, r, ex["np"], ",".join(x for x, _, _ in ex["rows"]), ",".join(str(i) for _, i, _ in ex["rows"]),
                    ",".join(str(o) for _, _, o in ex["rows"])))
            ident_b += e2b == "IDENTICAL"
            ident_a += e2a == "IDENTICAL"
            amp_cls.append(cls)
            out.append("H_REALIZATION amp=%s r=%d exit_inlet_total=%d exit_outlet_total=%d planes_with_exits=%d e2_b=%s e2_a_control=%s %s" % (
                amp, r, itot, otot, npl, e2b, e2a, cls))
        worst = max(amp_cls, key=WORST.index) if amp_cls else "MISSING"
        classes.append(worst)
        tot_in += amp_in
        tot_out += amp_out
        out.append("H_VERDICT amp=%s run_exit=%s realizations=%d e2_b_identical=%d/%d e2_a_control_identical=%d/%d exit_inlet_total=%d "
                   "exit_outlet_total=%d class=%s" % (amp, rc, nr, ident_b, nr, ident_a, nr, amp_in, amp_out, worst))
        pa = same(os.path.join(d, run, "ab_primary.csv"), os.path.join(ref_root, run, "ab_primary.csv"))
        out.append("RECORDED H_AB_PRIMARY amp=%s ab_primary.csv_vs_SF38=%s" % (amp, pa))
    if any_exit:
        prim1, prim4 = os.path.join(d, "ab_s1", "ab_primary.csv"), os.path.join(d, "ab_s4", "ab_primary.csv")
        orig = L.read(os.path.join(ref_root, "ab_decision.out")) or ""
        for ln in orig.splitlines():
            if ln.startswith(("C_STAT", "C_AMP", "C_VERDICT")):
                out.append("RECORDED H_ORIGINAL " + ln)
        try:
            p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "ab_decision.py"), prim1, prim4, "--lambda1", repr(lam1), "--lambda4", repr(lam4)],
                               capture_output=True, text=True, timeout=300)
            for ln in p.stdout.splitlines():
                if ln.startswith(("C_STAT", "C_AMP", "C_VERDICT")):
                    out.append("RECORDED H_RECOMPUTED " + ln)
            out.append("RECORDED H_RECOMPUTED_EXIT ab_decision.py rc=%d" % p.returncode)
        except (OSError, subprocess.SubprocessError) as e:
            out.append("RECORDED H_RECOMPUTED_EXIT ab_decision.py failed (%s)" % e)
    worst_all = max(classes, key=WORST.index) if classes else "MISSING"
    ok = runs_ok and worst_all in ("OK_ZERO_EXITS_IDENTICAL", "EXITS_RECORDED")
    out.append("H_VERDICT ALL class=%s exits_inlet=%d exits_outlet=%d runs_exit_ok=%d %s" % (worst_all, tot_in, tot_out, int(runs_ok),
                                                                                         "OK" if ok else "NOT_OK"))
    return out, ok


# ---------------------------------------------------------------------------------------------------------------
def selftest():
    import tempfile
    fails = []

    def expect(n, c):
        if not c:
            fails.append(n)
            print("SELFTEST FAIL %s" % n)

    expect("classify_ok", classify(0, "IDENTICAL", "IDENTICAL") == "OK_ZERO_EXITS_IDENTICAL")
    expect("classify_defect", classify(0, "DIFF", "IDENTICAL") == "DEFECT_EXIT_AWARE_POLLOCK")
    expect("classify_env", classify(0, "IDENTICAL", "DIFF") == "ENVIRONMENT_INVESTIGATE" and classify(3, "DIFF", "DIFF") == "ENVIRONMENT_INVESTIGATE")
    expect("classify_exits", classify(2, "DIFF", "IDENTICAL") == "EXITS_RECORDED")
    expect("classify_missing", classify(0, "MISSING", "IDENTICAL") == "MISSING")

    def mk(root, amp_run, r, e2b, e2a, exits):
        rd = os.path.join(root, amp_run, "realizations", "r_%04d" % r)
        os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "e2_b.csv"), "w") as f:
            f.write(e2b)
        with open(os.path.join(rd, "e2_a.csv"), "w") as f:
            f.write(e2a)
        if exits is not None:
            with open(os.path.join(rd, "exits_b.csv"), "w") as f:
                f.write("x1,exit_inlet,exit_outlet\n0.5,%d,0\n1,%d,0\n# Np=100\n# exit_inlet_total=%d\n# exit_outlet_total=0\n" % (exits, exits, exits))

    with tempfile.TemporaryDirectory() as d:
        ref, head = os.path.join(d, "ref"), os.path.join(d, "head")
        for run in ("ab_s1", "ab_s4"):
            for r in range(2):
                mk(ref, run, r, "b%d\n" % r, "a%d\n" % r, None)
                mk(head, run, r, "b%d\n" % r, "a%d\n" % r, 0)
            with open(os.path.join(ref, run, "ab_primary.csv"), "w") as f:
                f.write("x\n")
            with open(os.path.join(head, run, "ab_primary.csv"), "w") as f:
                f.write("x\n")
        with open(os.path.join(head, "exit_codes.tsv"), "w") as f:
            f.write("ab_s1\t0\nab_s4\t0\n")
        out, ok = report(head, ref, nr=2)
        expect("all_identical", ok and out[-1].startswith("H_VERDICT ALL class=OK_ZERO_EXITS_IDENTICAL") and out[-1].endswith("OK"))
        mk(head, "ab_s1", 1, "DIFFERENT\n", "a1\n", 0)  # no exits, e2_b differs, control identical: defect
        out, ok = report(head, ref, nr=2)
        expect("defect", not ok and any("DEFECT_EXIT_AWARE_POLLOCK" in x for x in out))
        mk(head, "ab_s1", 1, "DIFFERENT\n", "a1\n", 3)  # exits present: recorded, not a failure
        out, ok = report(head, ref, nr=2)
        expect("exits_recorded", ok and out[-1].startswith("H_VERDICT ALL class=EXITS_RECORDED exits_inlet=3"))
        expect("exits_trigger_recompute_entry", any(x.startswith("RECORDED H_RECOMPUTED_EXIT") for x in out))
        mk(head, "ab_s4", 0, "b0\n", "CHANGED\n", 0)  # control differs: environment
        out, ok = report(head, ref, nr=2)
        expect("environment", not ok and any("ENVIRONMENT_INVESTIGATE" in x for x in out))
        with open(os.path.join(head, "exit_codes.tsv"), "w") as f:
            f.write("ab_s1\t1\nab_s4\t0\n")
        mk(head, "ab_s4", 0, "b0\n", "a0\n", 0)
        out, ok = report(head, ref, nr=2)
        expect("run_exit_nonzero", not ok)
        os.remove(os.path.join(head, "ab_s4", "realizations", "r_0001", "exits_b.csv"))
        out, ok = report(head, ref, nr=2)
        expect("missing_exits_file", not ok and any("MISSING" in x for x in out))
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK h_verdict")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    if not a or a[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    opt = {"--ref-root": REF_ROOT, "--nr": "10", "--lambda1": "0.0390625", "--lambda4": "0.125"}
    i = 1
    while i < len(a):
        if a[i] in opt and i + 1 < len(a):
            opt[a[i]] = a[i + 1]
            i += 2
        else:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
    o, ok = report(a[0], opt["--ref-root"], int(opt["--nr"]), float(opt["--lambda1"]), float(opt["--lambda4"]))
    print("\n".join(o))
    sys.exit(0 if ok else 1)
