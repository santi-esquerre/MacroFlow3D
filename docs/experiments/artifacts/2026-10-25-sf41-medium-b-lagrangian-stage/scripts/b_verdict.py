#!/usr/bin/env python3
"""SF-41 threshold B (pre-registration RR4): verdict from the log of `sf41_cross_representation_tests --only B`.

  b_verdict.py DIR [--run B] [--dry]
  b_verdict.py --selftest

DIR holds RUN.log (default `B`) and exit_codes.tsv.  B is met iff
  * the program exited 0 and printed `B_VERDICT all_pass=1`;
  * `B_CONFIG tier=RR4 grid=1664x128x128 ell=0.25 lambda_over_h=32 eps=2 seed=6001` (the RR4 configuration);
  * the three SF-19 correctors converged (`B_SOLVE corrector=k converged=1`);
  * for field=h_tilde the three `B_METRIC` quantities (value, gradient, hessian) are present and each
    rel <= 1e-11 (re-evaluated here, not read from the program's verdict field), finite.
Recorded (no gate): the lnK `B_METRIC` lines, every `B_DECAY` line and the per-(field, quantity) mean over the faces/lines of
the fitted per-knot factor (expected ~0.268, derivation), and the `B_PROFILE` row count.  `--dry` accepts the fast-tier
configuration (any grid, gate not applied by the program: rel only recorded) and requires only finiteness and B_VERDICT.
Output: `B_VERDICT ... PASS|FAIL` and RECORDED lines.  Exit 0 iff PASS.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf41lib as L  # noqa: E402

GATE = 1e-11
QUANT = ("value", "gradient", "hessian")


def evaluate(text, rc, dry=False):
    why = []
    cfg = [L.kv(x) for x in (text or "").splitlines() if x.startswith("B_CONFIG ")]
    if not cfg:
        why.append("no_B_CONFIG")
    else:
        c = cfg[0]
        if not dry and not (c.get("tier") == "RR4" and c.get("grid") == "1664x128x128" and L.num(c.get("ell")) == 0.25
                            and L.num(c.get("lambda_over_h")) == 32 and L.num(c.get("eps")) == 2 and c.get("seed") == "6001"):
            why.append("config_not_RR4(%s)" % " ".join("%s=%s" % kv for kv in sorted(c.items())))
    solves = [L.kv(x) for x in (text or "").splitlines() if x.startswith("B_SOLVE corrector=")]
    if sorted(s.get("corrector") for s in solves) != ["1", "2", "3"] or any(s.get("converged") != "1" for s in solves):
        why.append("sf19_correctors_not_converged")
    metrics = {}
    for x in (text or "").splitlines():
        if x.startswith("B_METRIC "):
            d = L.kv(x)
            metrics[(d.get("field"), d.get("quantity"))] = d
    rels = {}
    for q in QUANT:
        d = metrics.get(("h_tilde", q))
        v = L.num(d.get("rel")) if d else None
        rels[q] = v
        if v is None or not math.isfinite(v):
            why.append("h_tilde_%s_missing_or_nonfinite" % q)
        elif not dry and not v <= GATE:
            why.append("h_tilde_%s=%.3e>1e-11" % (q, v))
    pv = L.lines_of(text, "B_VERDICT")
    if not pv or L.kv(pv[-1]).get("all_pass") != "1":
        why.append("program_B_VERDICT_not_all_pass")
    if rc != 0:
        why.append("exit=%s" % rc)
    return not why, why, rels, metrics


def decay_summary(text):
    acc = {}
    for x in (text or "").splitlines():
        if x.startswith("B_DECAY "):
            d = L.kv(x)
            f = L.num(d.get("factor"))
            if f is not None and math.isfinite(f):
                acc.setdefault((d.get("field"), d.get("quantity")), []).append(f)
    return acc


def report(d, run="B", dry=False):
    text = L.read(os.path.join(d, run + ".log"))
    rc = L.rc_of(d, run)
    if text is None:
        return ["B_VERDICT run=%s exit=%s FAIL (no log)" % (run, rc)], False
    ok, why, rels, metrics = evaluate(text, rc, dry)
    out = ["B_VERDICT run=%s exit=%s h_tilde_value=%s h_tilde_gradient=%s h_tilde_hessian=%s gate=1e-11 %s%s" % (
        run, rc, *("-" if rels[q] is None else "%.6e" % rels[q] for q in QUANT), "PASS" if ok else "FAIL",
        "" if ok else " [%s]" % ",".join(why))]
    for (f, q), m in sorted(metrics.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if f != "h_tilde":
            out.append("RECORDED B_METRIC field=%s quantity=%s rel=%s (recorded)" % (f, q, m.get("rel")))
    for (f, q), fs in sorted(decay_summary(text).items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        out.append("RECORDED B_DECAY_SUMMARY field=%s quantity=%s n=%d mean_factor=%.6f min=%.6f max=%.6f (expected ~0.268)"
                   % (f, q, len(fs), sum(fs) / len(fs), min(fs), max(fs)))
    for x in text.splitlines():
        if x.startswith(("B_DECAY ", "B_SOLVE ", "B_CONFIG ")):
            out.append("RECORDED " + x[:300])
    out.append("RECORDED B_PROFILE rows=%d (raw in the log)" % sum(1 for x in text.splitlines() if x.startswith("B_PROFILE ")))
    return out, ok


def _synth(rel=(1e-13, 1e-12, 3e-12), conv=1, cfg=None, allpass=1):
    c = cfg or "B_CONFIG tier=RR4 grid=1664x128x128 ell=0.25 lambda_over_h=32 eps=2 seed=6001 L1=13"
    L_ = [c]
    for k in (1, 2, 3):
        L_.append("B_SOLVE corrector=%d converged=%d status=0 iterations=20 initial=1.0e+00 final=1.0e-11 rel=1.0e-11" % (k, conv))
    for f, scale in (("h_tilde", 1.0), ("lnK", 5.0)):
        for q, r in zip(QUANT, rel):
            L_.append("B_METRIC field=%s quantity=%s rel=%.6e gate=1e-11 verdict=X points=100100" % (f, q, r * scale))
    L_.append("B_DECAY field=h_tilde line=0,0 face=west factor=0.268100 knots_used=40 quantity=gradient floor=1e-14")
    L_.append("B_DECAY field=h_tilde line=0,0 face=east factor=0.267900 knots_used=40 quantity=gradient floor=1e-14")
    L_.append("B_PROFILE field=h_tilde line=0,0 face=west knot=0 value=1e-3 gradient=1e-3 hessian=1e-3")
    L_.append("B_VERDICT all_pass=%d" % allpass)
    return "\n".join(L_) + "\n"


def selftest():
    import tempfile
    fails = []

    def expect(n, c):
        if not c:
            fails.append(n)
            print("SELFTEST FAIL %s" % n)

    expect("pass", evaluate(_synth(), 0)[0])
    expect("hessian_over", not evaluate(_synth(rel=(1e-13, 1e-12, 2e-11)), 1)[0])
    ok, why, _, _ = evaluate(_synth(rel=(1e-13, 1e-12, 2e-11)), 0)
    expect("hessian_over_even_if_program_lies", not ok and any("hessian" in w for w in why))
    expect("not_converged", not evaluate(_synth(conv=0), 0)[0])
    expect("wrong_config", not evaluate(_synth(cfg="B_CONFIG tier=fast grid=128x16x16 ell=0.25 lambda_over_h=4 eps=2 seed=6001"), 0)[0])
    expect("program_not_all_pass", not evaluate(_synth(allpass=0), 0)[0])
    expect("exit_nonzero", not evaluate(_synth(), 1)[0])
    expect("nan", not evaluate(_synth(rel=(float("nan"), 0, 0)), 0)[0])
    expect("empty", not evaluate("", 0)[0])
    expect("dry_accepts_fast_tier", evaluate(_synth(cfg="B_CONFIG tier=fast grid=128x16x16 ell=0.25 lambda_over_h=4 eps=2 seed=6001",
                                                     rel=(1e-6, 1e-6, 1e-6)), 0, dry=True)[0])
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "B.log"), "w") as f:
            f.write(_synth())
        with open(os.path.join(d, "exit_codes.tsv"), "w") as f:
            f.write("B\t0\n")
        out, ok = report(d)
        expect("report", ok and out[0].startswith("B_VERDICT run=B exit=0") and any("B_DECAY_SUMMARY" in x and "0.268000" in x for x in out))
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK b_verdict")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    if not a or a[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    run, dry = "B", False
    i = 1
    while i < len(a):
        if a[i] == "--run" and i + 1 < len(a):
            run = a[i + 1]
            i += 2
        elif a[i] == "--dry":
            dry = True
            i += 1
        else:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
    o, ok = report(a[0], run, dry)
    print("\n".join(o))
    sys.exit(0 if ok else 1)
