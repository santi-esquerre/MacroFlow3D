#!/usr/bin/env python3
"""SF-38 threshold C0 (RR4, decision D8): medium (b) solver pilot verdict from the outputs of run_sf38.sh part `bpilot`.

  c0_verdict.py DIR RUN
  c0_verdict.py --selftest

RR4: only the solver record and the memory are read.  This script opens RUN.log (and the sampler files); it reads ONLY lines
whose first token (after an optional `R=<r> ` prefix) is B_SOLVER, plus the MEM_/PID sampler data.  It never opens
ab_primary.csv, the E1/E2 outputs or any other output file, and it prints no E1/E2/primary value.

Assumed grammar of the A/B application's solver record (N4 contract; aliases accepted, first present wins):
  B_SOLVER converged=<0|1|true|false>  (or status=converged|...)  iters=<n>  rel_residual=<x>  balance=<x>  max_div=<x>
           [Q0=<x> QL=<x>]
  converged : converged | conv | status (value `converged`)           rel     : rel_residual | rel | relres | rel_res
  iters     : iters | iterations | its                                balance : balance | flux_balance | q_balance
  max_div   : max_div | max_cell_div | maxdiv | max_cell_divergence   Q0, QL : Q0|q0|Q_0 , QL|ql|Q_L (balance recomputed
  as |Q0 - QL| / |Q0| when no balance key is present)
Verdict (threshold C0 and its fallback):
  PASS            converged and rel <= 1e-10, balance <= 1e-8, PID peak <= 30000 MiB, exit 0
  PASS_FALLBACK   rel in (1e-10, 1e-8] (or not converged but rel <= 1e-8), balance <= 1e-8, PID peak <= 30000 MiB, exit 0
                  (the A/B then uses the fallback tolerance 1e-8, recorded)
  FAIL            otherwise (rel > 1e-8, balance > 1e-8, peak above the limit, exit != 0)
  NOT_EVALUABLE   exit 0 and the checks that could be evaluated pass but the record is incomplete or the RR2 sampler gates fail
Output: C0_VERDICT run= exit= solver_converged= iters= rel= balance= max_div= pid_peak_mib= limit_mib=30000 mem= mode= <verdict>
Exit 0 iff PASS or PASS_FALLBACK.  Stdlib only.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mem_verdict as mv  # noqa: E402

REL_PRIMARY = 1e-10
REL_FALLBACK = 1e-8
BAL_MAX = 1e-8

ALIASES = {
    "conv": ["converged", "conv"],
    "status": ["status"],
    "rel": ["rel_residual", "rel", "relres", "rel_res"],
    "iters": ["iters", "iterations", "its"],
    "balance": ["balance", "flux_balance", "q_balance"],
    "max_div": ["max_div", "max_cell_div", "maxdiv", "max_cell_divergence"],
    "Q0": ["Q0", "q0", "Q_0"],
    "QL": ["QL", "ql", "Q_L"],
}


def read_solver_record(text):
    """key=value tokens of every B_SOLVER line (later lines override earlier keys)."""
    kv = {}
    n = 0
    for raw in (text or "").splitlines():
        line = mv.strip_prefix(raw)
        if line.split(None, 1)[:1] == ["B_SOLVER"]:
            n += 1
            for k, v in re.findall(r"(\w+)=(\S+)", line):
                kv[k] = v
    return kv, n


def pick(kv, key):
    for a in ALIASES[key]:
        if a in kv:
            return kv[a]
    return None


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def truthy(v):
    return v is not None and v.strip().lower() in ("1", "true", "yes", "converged")


def evaluate(text, rc, pk):
    kv, n = read_solver_record(text)
    reasons = []
    if n == 0:
        reasons.append("no_B_SOLVER_line")
    conv_raw = pick(kv, "conv")
    if conv_raw is None and pick(kv, "status") is not None:
        conv_raw = "1" if pick(kv, "status") == "converged" else "0"
    conv = None if conv_raw is None else truthy(conv_raw)
    rel = fnum(pick(kv, "rel"))
    it = pick(kv, "iters")
    bal = fnum(pick(kv, "balance"))
    q0, ql = fnum(pick(kv, "Q0")), fnum(pick(kv, "QL"))
    if bal is None and q0 is not None and ql is not None and q0 != 0:
        bal = abs(q0 - ql) / abs(q0)
    mdiv = fnum(pick(kv, "max_div"))
    if n and conv is None:
        reasons.append("no_converged_key")
    if n and rel is None:
        reasons.append("no_rel_residual_key")
    if n and bal is None:
        reasons.append("no_balance_key")
    mem_ok = pk["status"] == "OK"
    within = pk["pid_peak"] is not None and pk["pid_peak"] <= mv.LIMIT_MIB
    fails = []
    if rc != 0:
        fails.append("exit")
    if rel is not None and rel > REL_FALLBACK:
        fails.append("rel>1e-8")
    if bal is not None and bal > BAL_MAX:
        fails.append("balance>1e-8")
    if mem_ok and not within:
        fails.append("pid_peak>30000")
    if rel is not None and rel != rel:
        fails.append("rel_nan")
    mode = "-"
    if rel is not None and rel == rel:
        mode = "primary" if (conv and rel <= REL_PRIMARY) else ("fallback" if rel <= REL_FALLBACK else "failed")
    if fails:
        v = "FAIL"
    elif reasons or not mem_ok:
        v = "NOT_EVALUABLE"
    else:
        v = "PASS" if mode == "primary" else "PASS_FALLBACK"
    why = list(fails) + reasons + ([] if mem_ok else ["mem(%s)" % ",".join(pk["reasons"])])
    return v, {"conv": conv, "iters": it, "rel": rel, "bal": bal, "mdiv": mdiv, "mode": mode, "why": why}


def report(d, run):
    text = mv.rd(os.path.join(d, run + ".log"))
    rc = mv.rc_of(d, run)
    pk = mv.peak_of(d, run)
    v, i = evaluate(text, rc, pk)

    def f(x):
        return "-" if x is None else "%.6e" % x
    line = ("C0_VERDICT run=%s exit=%s solver_converged=%s iters=%s rel=%s balance=%s max_div=%s pid_peak_mib=%s limit_mib=%d "
            "mem=%s mode=%s %s%s"
            % (run, rc, "-" if i["conv"] is None else int(i["conv"]), i["iters"], f(i["rel"]), f(i["bal"]), f(i["mdiv"]),
               pk["pid_peak"], mv.LIMIT_MIB, "OK" if pk["status"] == "OK" else "NOT_EVALUABLE", i["mode"], v,
               (" [%s]" % ",".join(i["why"])) if i["why"] else ""))
    out = [line]
    out.extend(mv.mem_records(d, run))  # wall, RSS, MEM lines, PID peak: memory only
    return out, v


# ---------------------------------------------------------------------------------------------------------------
def selftest():
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    ok_pk = {"status": "OK", "reasons": [], "pid_peak": 29000, "pid": 1}
    ne_pk = {"status": "NOT_EVALUABLE", "reasons": ["samples<5(3)"], "pid_peak": 29000, "pid": 1}

    def rec(**kw):
        d = dict(converged=1, iters=45, rel_residual="3.2e-11", balance="2.0e-10", max_div="4.0e-9")
        d.update(kw)
        return "R=0 B_SOLVER " + " ".join("%s=%s" % (k, v) for k, v in d.items() if v is not None) + "\n"

    v, i = evaluate(rec(), 0, ok_pk)
    expect("pass", v == "PASS" and i["mode"] == "primary" and i["mdiv"] == 4e-9)
    v, i = evaluate(rec(converged=0, rel_residual="5e-9"), 0, ok_pk)
    expect("fallback_stall", v == "PASS_FALLBACK" and i["mode"] == "fallback")
    v, i = evaluate(rec(converged=1, rel_residual="2e-10"), 0, ok_pk)
    expect("fallback_between", v == "PASS_FALLBACK")
    v, _ = evaluate(rec(converged=0, rel_residual="2e-8"), 0, ok_pk)
    expect("fail_rel", v == "FAIL")
    v, _ = evaluate(rec(balance="2e-8"), 0, ok_pk)
    expect("fail_balance", v == "FAIL")
    v, _ = evaluate(rec(), 1, ok_pk)
    expect("fail_exit", v == "FAIL")
    v, _ = evaluate(rec(), 0, dict(ok_pk, pid_peak=30001))
    expect("fail_peak", v == "FAIL")
    v, _ = evaluate(rec(), 0, ne_pk)
    expect("not_evaluable_sampler", v == "NOT_EVALUABLE")
    v, i = evaluate("no record here\n", 0, ok_pk)
    expect("not_evaluable_no_record", v == "NOT_EVALUABLE" and "no_B_SOLVER_line" in i["why"])
    v, _ = evaluate(rec(rel_residual=None), 0, ok_pk)
    expect("not_evaluable_missing_key", v == "NOT_EVALUABLE")
    # aliases, status key, balance from Q0/QL
    v, i = evaluate("B_SOLVER status=converged its=40 rel=1e-11 Q0=2.0 QL=2.0000000001 max_cell_div=1e-9\n", 0, ok_pk)
    expect("aliases", v == "PASS" and i["iters"] == "40" and abs(i["bal"] - 5e-11) < 1e-15 and i["mdiv"] == 1e-9)
    # the record is read from B_SOLVER lines only: a line that merely contains the word does not count
    v, _ = evaluate("note: B_SOLVER converged=1 rel_residual=1e-12 balance=0\n", 0, ok_pk)
    expect("only_first_token", v == "NOT_EVALUABLE")
    # RR4: no E1/E2/primary value is read or printed: run through report() with an ab_primary.csv present
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "bp.log"), "w") as f:
            f.write("R=0 MEM_GEOM pid=4242 nx=1\nR=0 PRIMARY S_T=9.87654321e-3 V1=1.2345e-1\n" + rec())
        for ext, body in ((".time", "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:10.0\n\tMaximum resident set size (kbytes): 5\n"),
                          (".free", "Mem: 1 2\n"), (".pid_tree", "4242\n")):
            with open(os.path.join(d, "bp" + ext), "w") as f:
                f.write(body)
        with open(os.path.join(d, "bp.apps.csv"), "w") as f:
            f.write("".join("2026/10/23 10:00:%02d.000, 4242, %d\n" % (i, 100 * (i + 1)) for i in range(10)))
        with open(os.path.join(d, "bp.dev.csv"), "w") as f:
            f.write("".join("2026/10/23 10:00:%02d.000, %d\n" % (i, 100 * (i + 1) + 50) for i in range(10)))
        with open(os.path.join(d, "exit_codes.tsv"), "w") as f:
            f.write("bp\t0\n")
        out, v = report(d, "bp")
        txt = "\n".join(out)
        expect("report_pass", v == "PASS" and out[0].startswith("C0_VERDICT run=bp exit=0 solver_converged=1"))
        expect("rr4_no_primary_printed", "9.87654321e-3" not in txt and "S_T" not in txt and "PRIMARY" not in txt)
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK c0_verdict")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    o, v = report(sys.argv[1], sys.argv[2])
    print("\n".join(o))
    sys.exit(0 if v in ("PASS", "PASS_FALLBACK") else 1)
