#!/usr/bin/env python3
"""SF-41 threshold E, production feasibility (pre-registration RR7, RR2): verdict from the outputs of run_sf41.sh part `prodgrid`.

  e_verdict.py DIR [--ids E1,E2,E3,E4] [--dry]
  e_verdict.py --selftest

DIR holds, for each run R (E1..E4, and R_mapped when the mapped-host fallback was run): R.log R.time R.free R.apps.csv R.dev.csv
R.pid_tree, the runner output tree R/ and exit_codes.tsv.  Per run (`E_RUN`), all of:
  * solve (RR7): the stage's `B_SOLVER` line has converged=1 and rel <= 1e-10  -> `primary`; or rel <= 1e-8 with the balance met
    (the C0 fallback, recorded)  -> `fallback`; otherwise the solve is not met (stop, owner);
  * flux balance |Q_in - Q_out| / |Q_in| <= 1e-8 recomputed from the printed Q_in, Q_out (and the stage's own balance field);
  * runner exit 0, the `R=0 STAGE` line grid equal to the pre-registered grid of the run;
  * memory (RR2): PID peak <= 30000 MiB with the sampler gates (>= 5 samples, PID <= device-wide peak, PID consistent with the
    process tree); a failing sampler gate -> NOT_EVALUABLE;
  * non-solve phases (specification E): from the `MEM_PHASE` lines, EXACT bytes of every phase other than S2 (F, S1, V, H, S3, T and
    the bridge B) <= the exact bytes of S2 and <= 30000 MiB; a violation is a placement defect (BLOCKING).
Per grid (`E_VERDICT id=`): PASS (device head run met); PASS_MAPPED_HEAD (the device run's PID peak exceeded 30000 MiB and the
`R_mapped` re-run met everything: the pre-authorized fallback E1, both runs reported); FAIL otherwise, with the reason
(`solve`, `placement`, `peak_above_30000_after_fallback` = R3 for the owner, `fallback_not_run`, `exit`).  A device run whose
PID peak is above 30000 MiB requires the mapped run (RR7); a mapped run is never a substitute for a device run that met
everything.  Last line `E_VERDICT ALL PASS|FAIL|NOT_EVALUABLE`.
Recorded (no gate): wall, host RSS, free -b, per-phase bytes, STATUS / TEXIT / TSTATUS / FLOWSTATS / RESOLVED_TIMES lines,
exit and non-ok counts, status.json totals, flow_stats.json.  Exit 0 iff ALL is PASS.  `--dry`: grids are not checked, the
memory gates are not applied (local dry run, not evidence).
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf41lib as L  # noqa: E402
import mem_verdict as mv  # noqa: E402

GRID = {"E1": "2048x256x256", "E2": "3328x384x384", "E3": "3328x384x384", "E4": "6656x256x256"}
LIMIT_BYTES = mv.LIMIT_MIB * 2**20


def phases(text):
    """-> {phase: exact_bytes} from the `R=0 MEM_PHASE` lines (last wins)."""
    out = {}
    for ln in L.lines_of(text, "MEM_PHASE"):
        m = re.search(r"\bphase=(\S+)", ln)
        e = re.search(r"\bexact_bytes=(\d+)", ln)
        if m and e:
            out[m.group(1)] = int(e.group(1))
    return out


def solve_class(sv):
    """-> (class, rel, balance_recomputed) with class in primary | fallback | not_met | no_record."""
    if not sv:
        return "no_record", None, None
    rel, bal_f = L.num(sv.get("rel")), L.num(sv.get("balance"))
    qi, qo = L.num(sv.get("Q_in")), L.num(sv.get("Q_out"))
    bal = abs(qi - qo) / abs(qi) if (qi not in (None, 0) and qo is not None) else None
    if rel is None or not math.isfinite(rel) or bal is None or not math.isfinite(bal) or bal_f is None:
        return "not_met", rel, bal
    bal_ok = bal <= 1e-8 and bal_f <= 1e-8
    if sv.get("converged") == "1" and rel <= 1e-10 and bal_ok:
        return "primary", rel, bal
    if rel <= 1e-8 and bal_ok:
        return "fallback", rel, bal
    return "not_met", rel, bal


def run_state(d, run, grid, dry=False):
    """-> dict with the per-run evaluation."""
    text = L.read(os.path.join(d, run + ".log"))
    rc = L.rc_of(d, run)
    r = {"run": run, "rc": rc, "text": text, "why": []}
    if text is None:
        r.update(state="FAIL", cls="no_log", rel=None, bal=None, phases={}, peak=None, mem="NOT_EVALUABLE", pk=None)
        r["why"].append("no_log")
        return r
    sl = L.lines_of(text, "B_SOLVER")
    sv = L.kv(sl[-1]) if sl else None
    cls, rel, bal = solve_class(sv)
    ph = phases(text)
    st = [L.kv(x) for x in L.lines_of(text, "STAGE")]
    sgrid = None
    for x in L.lines_of(text, "STAGE"):
        m = re.search(r"\bgrid=(\S+)", x)
        if m:
            sgrid = m.group(1)
    pk = mv.peak_of(d, run)
    r.update(cls=cls, rel=rel, bal=bal, phases=ph, pk=pk, peak=pk["pid_peak"], sv=sv, sgrid=sgrid)
    r["mem"] = "OK" if pk["status"] == "OK" else "NOT_EVALUABLE"
    if rc != 0:
        r["why"].append("exit=%s" % rc)
    if cls not in ("primary", "fallback"):
        r["why"].append("solve_%s" % cls)
    if not dry and sgrid != grid:
        r["why"].append("grid=%s!=%s" % (sgrid, grid))
    s2 = ph.get("S2")
    placement = []
    if s2 is None:
        placement.append("no_S2_phase")
    else:
        for p, b in sorted(ph.items()):
            if p != "S2" and (b > s2 or b > LIMIT_BYTES):
                placement.append("%s=%d>%s" % (p, b, "S2=%d" % s2 if b > s2 else "30000MiB"))
    r["placement"] = placement
    over = (not dry) and pk["status"] == "OK" and pk["pid_peak"] is not None and pk["pid_peak"] > mv.LIMIT_MIB
    r["peak_over"] = over
    if placement:
        r["state"] = "FAIL_PLACEMENT"
    elif cls not in ("primary", "fallback") or rc != 0 or any(w.startswith("grid=") for w in r["why"]):
        r["state"] = "FAIL"
    elif dry:
        r["state"] = "PASS"
    elif pk["status"] != "OK":
        r["state"] = "NOT_EVALUABLE"
    elif over:
        r["state"] = "PEAK_OVER"
    else:
        r["state"] = "PASS"
    return r


def fmt_run(r):
    f = lambda x: "-" if x is None else "%.6e" % x
    return ("E_RUN run=%s exit=%s solve=%s rel=%s balance=%s pid_peak_mib=%s limit_mib=%d mem=%s phases_le_S2=%d state=%s%s"
            % (r["run"], r["rc"], r["cls"], f(r["rel"]), f(r["bal"]), r["peak"], mv.LIMIT_MIB, r["mem"],
               int(not r.get("placement")), r["state"],
               (" [%s]" % ",".join(r["why"] + r.get("placement", []))) if (r["why"] or r.get("placement")) else ""))


def grid_verdict(dev, mapped):
    """-> (verdict, reason)"""
    s = dev["state"]
    if s == "PASS":
        return "PASS", "device_head"
    if s == "FAIL_PLACEMENT":
        return "FAIL", "placement_defect"
    if s == "FAIL":
        return "FAIL", ",".join(dev["why"]) or "run"
    if s == "NOT_EVALUABLE":
        return "NOT_EVALUABLE", "sampler:" + ",".join(dev["pk"]["reasons"] if dev["pk"] else ["?"])
    # PEAK_OVER: the pre-authorized fallback
    if mapped is None:
        return "FAIL", "fallback_not_run"
    if mapped["state"] == "PASS":
        return "PASS_MAPPED_HEAD", "device_peak_%sMiB>30000;mapped_run_met" % dev["peak"]
    if mapped["state"] == "PEAK_OVER":
        return "FAIL", "peak_above_30000_after_fallback(R3,owner)"
    if mapped["state"] == "NOT_EVALUABLE":
        return "NOT_EVALUABLE", "mapped_sampler"
    return "FAIL", "mapped_" + ("placement_defect" if mapped["state"] == "FAIL_PLACEMENT" else ",".join(mapped["why"]) or "run")


def records(d, r):
    out = list(mv.mem_records(d, r["run"]))
    ph = r.get("phases") or {}
    out.append("RECORDED run=%s phase_exact_bytes %s" % (r["run"], " ".join("%s=%d" % kv for kv in sorted(ph.items()))))
    for tok in ("STAGE", "STATUS", "TEXIT", "TSTATUS", "FLOWSTATS", "RESOLVED_TIMES", "TIMING", "BACKFLOW", "POLLOCK", "TWIN"):
        for ln in L.lines_of(r["text"], tok):
            out.append("RECORDED run=%s %s" % (r["run"], ln[:360]))
    rd = os.path.join(r.get("dir", ""), r["run"], "realizations", "r_0000")
    for fn in ("solver_b.json", "status.json", "flow_stats.json"):
        t = L.read(os.path.join(rd, fn))
        if t is not None:
            try:
                j = json.loads(t)
                out.append("RECORDED run=%s %s %s" % (r["run"], fn, json.dumps(j, sort_keys=True)[:600]))
            except ValueError:
                out.append("RECORDED run=%s %s unreadable" % (r["run"], fn))
    return out


def report(d, ids=("E1", "E2", "E3", "E4"), dry=False):
    out, vs = [], []
    for i in ids:
        grid = GRID.get(i)
        dev = run_state(d, i, grid, dry)
        dev["dir"] = d
        out.append(fmt_run(dev))
        mp = None
        if os.path.exists(os.path.join(d, i + "_mapped.log")) or L.rc_of(d, i + "_mapped") is not None:
            mp = run_state(d, i + "_mapped", grid, dry)
            mp["dir"] = d
            out.append(fmt_run(mp))
        v, why = grid_verdict(dev, mp)
        vs.append(v)
        out.append("E_VERDICT id=%s grid=%s %s (%s)" % (i, grid, v, why))
        out.extend(records(d, dev))
        if mp is not None:
            out.extend(records(d, mp))
    allv = "FAIL" if "FAIL" in vs else ("NOT_EVALUABLE" if "NOT_EVALUABLE" in vs else ("PASS" if vs else "FAIL"))
    out.append("E_VERDICT ALL %s (%s)" % (allv, ",".join("%s=%s" % kv for kv in zip(ids, vs))))
    return out, allv == "PASS"


# ---------------------------------------------------------------------------------------------------------------
def _write_run(d, run, grid, peak, rc=0, rel="3.0e-11", conv=1, qout="1.000000001", phases_extra=None, nsamp=20, s2=20000 * 2**20,
               qin="1.0", bal="1.0e-9"):
    ph = {"F": 9000 * 2**20, "S1": 8000 * 2**20, "S2": s2, "V": 12000 * 2**20, "H": 3000 * 2**20, "S3": 15000 * 2**20, "T": 14000 * 2**20,
          "B": 16000 * 2**20}
    ph.update(phases_extra or {})
    text = "".join("R=0 MEM_PHASE phase=%s exact_bytes=%d cufft_work=0 device_used=1 seconds=1.0 items=a:1\n" % kv for kv in ph.items())
    text += ("R=0 B_SOLVER r=0 converged=%d iterations=50 initial=6.7e+06 final=6.5e-04 rel=%s qbar=1.0 Q_in=%s Q_out=%s balance=%s "
             "max_div=1.0e-07 fallback=0\n" % (conv, rel, qin, qout, bal))
    text += "R=0 STAGE medium=b status=converged grid=%s mg_levels=7 G=1,0,0 solver_converged=1 solver_usable=1 flow_ready=1 seconds=9.0\n" % grid
    text += "R=0 TEXIT pass=plane exit_inlet=0 exit_outlet=0 (statuses 7 / 8, outside the TSTATUS histogram)\nR=0 STATUS ok tracker_non_ok_plane=0\n"
    for ext, body in ((".log", text), (".time", "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n\tMaximum resident set size (kbytes): 4096\n"),
                      (".free", "Mem: 1 2\n"), (".pid_tree", "4242\n")):
        with open(os.path.join(d, run + ext), "w") as f:
            f.write(body)
    with open(os.path.join(d, run + ".apps.csv"), "w") as f:
        for i in range(nsamp):
            f.write("2026/10/25 10:00:%02d.000, 4242, %d\n" % (i, peak * (i + 1) // nsamp))
    with open(os.path.join(d, run + ".dev.csv"), "w") as f:
        for i in range(nsamp):
            f.write("2026/10/25 10:00:%02d.000, %d\n" % (i, (peak + 100) * (i + 1) // nsamp))
    with open(os.path.join(d, "exit_codes.tsv"), "a") as f:
        f.write("%s\t%d\n" % (run, rc))


def selftest():
    import tempfile
    fails = []

    def expect(n, c):
        if not c:
            fails.append(n)
            print("SELFTEST FAIL %s" % n)

    G = GRID["E1"]

    def one(**kw):
        with tempfile.TemporaryDirectory() as d:
            _write_run(d, "E1", G, kw.pop("peak", 28000), **kw)
            out, ok = report(d, ("E1",))
            return out, ok

    out, ok = one()
    expect("pass", ok and any(l.startswith("E_VERDICT id=E1") and " PASS " in l for l in out))
    out, ok = one(rel="5e-9", conv=0)
    expect("fallback_c0_is_met", ok and any("solve=fallback" in l for l in out))
    out, ok = one(rel="2e-8", conv=0)
    expect("solve_not_met", not ok and any("solve=not_met" in l for l in out))
    out, ok = one(qout="1.00000002")
    expect("balance_over", not ok)
    out, ok = one(phases_extra={"S3": 21000 * 2**20})
    expect("phase_over_S2_is_placement_defect", not ok and any("state=FAIL_PLACEMENT" in l for l in out))
    out, ok = one(phases_extra={"B": 31000 * 2**20}, s2=32000 * 2**20, peak=32500)
    expect("phase_over_30000", not ok and any("FAIL_PLACEMENT" in l for l in out))
    out, ok = one(peak=30500)
    expect("peak_over_without_fallback_run", not ok and any("fallback_not_run" in l for l in out))
    out, ok = one(rc=1)
    expect("exit", not ok)
    out, ok = one(nsamp=3)
    expect("few_samples", not ok and any(l.startswith("E_VERDICT ALL NOT_EVALUABLE") for l in out))
    # fallback: device over, mapped met
    with tempfile.TemporaryDirectory() as d:
        _write_run(d, "E1", G, 31000)
        _write_run(d, "E1_mapped", G, 24000, s2=15000 * 2**20, phases_extra={"F": 9000 * 2**20, "S3": 14000 * 2**20, "V": 12000 * 2**20,
                                                                           "T": 14000 * 2**20, "B": 14500 * 2**20, "H": 3000 * 2**20})
        out, ok = report(d, ("E1",))
        expect("mapped_fallback_pass", ok and any("PASS_MAPPED_HEAD" in l for l in out))
    with tempfile.TemporaryDirectory() as d:  # device over, mapped also over: R3
        _write_run(d, "E1", G, 31000)
        _write_run(d, "E1_mapped", G, 30500)
        out, ok = report(d, ("E1",))
        expect("mapped_also_over", not ok and any("peak_above_30000_after_fallback" in l for l in out))
    with tempfile.TemporaryDirectory() as d:  # a mapped run never rescues a device run that met everything, nor a solve failure
        _write_run(d, "E1", G, 28000, rel="2e-8", conv=0)
        _write_run(d, "E1_mapped", G, 20000)
        out, ok = report(d, ("E1",))
        expect("solve_failure_not_rescued", not ok)
    with tempfile.TemporaryDirectory() as d:  # wrong grid in the STAGE line
        _write_run(d, "E1", "2048x256x128", 28000)
        out, ok = report(d, ("E1",))
        expect("wrong_grid", not ok)
    with tempfile.TemporaryDirectory() as d:
        out, ok = report(d, ("E1",))
        expect("absent", not ok)
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK e_verdict")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        sys.exit(selftest())
    if not a or a[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    ids, dry = ["E1", "E2", "E3", "E4"], False
    i = 1
    while i < len(a):
        if a[i] == "--ids" and i + 1 < len(a):
            ids = a[i + 1].split(",")
            i += 2
        elif a[i] == "--dry":
            dry = True
            i += 1
        else:
            print(__doc__, file=sys.stderr)
            sys.exit(2)
    o, ok = report(a[0], tuple(ids), dry)
    print("\n".join(o))
    sys.exit(0 if ok else 1)
