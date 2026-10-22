#!/usr/bin/env python3
"""SF-40 criteria C, D, E and the RR9 "Recorded" items from the outputs of run_sf40.sh part `mem` (RR3, RR5).

  mem_verdict.py MEM_DIR [--dry]
  mem_verdict.py --selftest

MEM_DIR holds, for each run R in probe1 probe2 probe3 probe_warm d512 d512_bhatdev d3328_i d3328_ii:
  R.log R.time R.free R.apps.csv R.dev.csv R.pid_tree   and   exit_codes.tsv (`R<TAB>rc`).
Output (all lines are also written to results.txt by the campaign script):
  PROBE run= pid= pid_peak_mib= device_used_bytes= driver= runtime= status=       C_CONTEXT mib= (minimum over the 3 probes)
  C_VERDICT / C_ITEM / C_LARGEST / C_RUN ...     (closed_form.py; S+ runs d512, d3328_i, d3328_ii)
  CS_*                                           (same checks for the level-S run d512_bhatdev: recorded, not criterion C)
  D_VERDICT run=d512 pid_peak_mib= E_max_mib= C_mib= limit_mib= PASS|FAIL|NOT_EVALUABLE   (+ D_ALT)
  D3328_VERDICT run= exit= pid_peak_mib= limit_mib=30000 PASS|FAIL|NOT_EVALUABLE
  E_VERDICT run= correctors=<status,its,rel_res x3> converged= levels= coarsest= PASS|FAIL
  RECORDED ...                                   (host RSS, free -b, phase seconds, versions, max cells, S vs S+ time)
--dry waives the E expectation levels=7 / coarsest=52x6x6 (tiny dry-run grids); dry outputs are NOT evidence.
Stdlib only.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_form  # noqa: E402
import pid_peak  # noqa: E402

LIMIT_3328_MIB = 30000
S_PLUS_RUNS = ["d512", "d3328_i", "d3328_ii"]


def rd(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def rc_of(mem, run):
    rc = None
    t = rd(os.path.join(mem, "exit_codes.tsv"))
    for line in (t or "").splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] == run:
            rc = int(p[1])
    return rc


def time_info(mem, run):
    t = rd(os.path.join(mem, run + ".time")) or ""
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", t)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([\d:.]+)", t)
    w = None
    if wall:
        w = 0.0
        for part in wall.group(1).split(":"):
            w = w * 60 + float(part)
    return (int(rss.group(1)) if rss else None), w


def peak_of(mem, run, log_text):
    ap, dv = os.path.join(mem, run + ".apps.csv"), os.path.join(mem, run + ".dev.csv")
    if log_text is None or not os.path.exists(ap) or not os.path.exists(dv):
        return {"status": "NOT_EVALUABLE", "reasons": ["missing_files"], "pid": None, "samples": 0, "pid_peak": None,
                "dev_peak": None, "max_exact_mib": None, "tree_pid": None}
    pid, mx = pid_peak.log_info(os.path.join(mem, run + ".log"))
    tree = (rd(os.path.join(mem, run + ".pid_tree")) or "").strip()
    tree = int(tree) if tree.isdigit() else None
    return pid_peak.evaluate(pid_peak.read_apps(ap), pid_peak.read_dev(dv), pid, mx, tree)


def probe_info(text):
    m = re.search(r"CONTEXT_PROBE .*", text or "")
    kv = closed_form.parse_kv(m.group(0)) if m else {}
    return kv


def corrector_info(text):
    cor = re.findall(r"SETUP {3}SF-19 corrector ([xyz]): status=(\S+) its=(\d+) rel_residual=(\S+)", text or "")
    m = re.search(r"SETUP {3}SF-19: mg_levels=(\d+) converged=(\d)", text or "")
    return cor, (int(m.group(1)) if m else None), (int(m.group(2)) if m else None)


def e_verdict(run, text, rc, dry):
    cor, mgl, conv = corrector_info(text)
    geom, _ = closed_form.parse(text or "")
    levels = geom["levels"] if geom else None
    coarsest = geom["coarsest"] if geom else None
    desc = " ".join("%s,%s,%s" % (s, i, r) for _, s, i, r in cor) if cor else "-"
    ok = len(cor) == 3 and all(s == "converged" and float(r) <= 1e-10 for _, s, _, r in cor) and conv == 1
    if not dry:
        ok = ok and levels == 7 and coarsest == "52x6x6"
    status = "PASS" if ok else "FAIL"
    if not ok and rc not in (None, 0) and (len(cor) < 3 or conv is None):
        status = "NOT_EVALUABLE"
    return ("E_VERDICT run=%s correctors=%s converged=%s levels=%s coarsest=%s %s%s"
            % (run, desc, "-" if conv is None else conv, levels, coarsest, status,
               " (dry: levels/coarsest expectation waived)" if dry else "")), status


def run_report(mem, dry):
    out = []
    logs = {r: rd(os.path.join(mem, r + ".log")) for r in
            ["probe1", "probe2", "probe3", "probe_warm", "d512", "d512_bhatdev", "d3328_i", "d3328_ii"]}
    rcs = {r: rc_of(mem, r) for r in logs}
    # ---- context probes
    ctx_peaks, ctx_dev = [], []
    for r in ["probe1", "probe2", "probe3"]:
        kv = probe_info(logs[r])
        pk = peak_of(mem, r, logs[r])
        out.append("PROBE run=%s exit=%s pid=%s pid_peak_mib=%s device_used_bytes=%s driver=%s runtime=%s status=%s %s"
                   % (r, rcs[r], kv.get("pid", "-"), pk["pid_peak"], kv.get("device_used", "-"), kv.get("driver", "-"),
                      kv.get("runtime", "-"), pk["status"], ",".join(pk["reasons"]) or "-"))
        if pk["status"] == "OK":
            ctx_peaks.append(pk["pid_peak"])
        if "device_used" in kv:
            ctx_dev.append(int(kv["device_used"]))
    C = min(ctx_peaks) if ctx_peaks else None
    out.append("C_CONTEXT mib=%s (minimum PID peak over %d evaluable of 3 probes)" % (C if C is not None else "NOT_EVALUABLE",
                                                                                      len(ctx_peaks)))
    if logs["probe_warm"] is None or rcs["probe_warm"] not in (0,):
        out.append("RECORDED warm probe unavailable (exit=%s)" % rcs["probe_warm"])
    else:
        kv = probe_info(logs["probe_warm"])
        pk = peak_of(mem, "probe_warm", logs["probe_warm"])
        out.append("RECORDED warm_probe pid_peak_mib=%s device_used_bytes=%s status=%s"
                   % (pk["pid_peak"], kv.get("device_used", "-"), pk["status"]))
    probe_dev = min(ctx_dev) if ctx_dev else None
    # ---- C
    for r in S_PLUS_RUNS + ["d512_bhatdev"]:
        if logs[r] is None:
            out.append("C_RUN run=%s NOT_EVALUABLE (run absent)" % r if r in S_PLUS_RUNS else
                       "CS_RUN run=%s NOT_EVALUABLE (run absent)" % r)
            continue
        lines, _ = closed_form.evaluate(logs[r], r, complete=(rcs[r] == 0), probe_device_used=probe_dev)
        if r == "d512_bhatdev":
            lines = ["CS_" + l[2:] if l.startswith("C_") else l for l in lines]
        out.extend(lines)
    # ---- D(512)
    if logs["d512"] is not None:
        pk = peak_of(mem, "d512", logs["d512"])
        _, mx = pid_peak.log_info(os.path.join(mem, "d512.log"))
        emax = mx / 2.0**20 if mx is not None else None
        if pk["status"] != "OK" or C is None or emax is None:
            out.append("D_VERDICT run=d512 pid_peak_mib=%s E_max_mib=%s C_mib=%s limit_mib=- exit=%s NOT_EVALUABLE %s"
                       % (pk["pid_peak"], "-" if emax is None else "%.3f" % emax, C, rcs["d512"], ",".join(pk["reasons"])))
        else:
            lim = emax * 1.03 + C
            alt = (emax + C) * 1.03
            st = "PASS" if (rcs["d512"] == 0 and pk["pid_peak"] <= lim) else "FAIL"
            out.append("D_VERDICT run=d512 pid_peak_mib=%d E_max_mib=%.3f C_mib=%d limit_mib=%.3f exit=%s %s"
                       % (pk["pid_peak"], emax, C, lim, rcs["d512"], st))
            out.append("D_ALT run=d512 limit_alt_mib=%.3f ((E_max + C) * 1.03; reported, not the verdict) would_be=%s"
                       % (alt, "PASS" if (rcs["d512"] == 0 and pk["pid_peak"] <= alt) else "FAIL"))
    else:
        out.append("D_VERDICT run=d512 pid_peak_mib=- E_max_mib=- C_mib=%s limit_mib=- exit=- NOT_EVALUABLE (run absent)" % C)
    # ---- D(3328) and E
    for r in ["d3328_i", "d3328_ii"]:
        if logs[r] is None:
            out.append("D3328_VERDICT run=%s exit=- pid_peak_mib=- limit_mib=%d NOT_EVALUABLE (run absent)" % (r, LIMIT_3328_MIB))
            out.append("E_VERDICT run=%s correctors=- converged=- levels=- coarsest=- NOT_EVALUABLE (run absent)" % r)
            continue
        pk = peak_of(mem, r, logs[r])
        rc = rcs[r]
        cls = "ok" if rc == 0 else ("memory" if re.search(r"out of memory|cudaErrorMemoryAllocation|bad_alloc", logs[r], re.I)
                                    else "other")
        if pk["status"] != "OK":
            st = "NOT_EVALUABLE"
        elif rc != 0 or pk["pid_peak"] > LIMIT_3328_MIB:
            st = "FAIL"
        else:
            st = "PASS"
        out.append("D3328_VERDICT run=%s exit=%s pid_peak_mib=%s limit_mib=%d exit_class=%s %s%s"
                   % (r, rc, pk["pid_peak"], LIMIT_3328_MIB, cls, st,
                      " (%s)" % ",".join(pk["reasons"]) if pk["reasons"] else ""))
        out.append(e_verdict(r, logs[r], rc, dry)[0])
    if logs["d512"] is not None:
        cor, mgl, conv = corrector_info(logs["d512"])
        out.append("RECORDED E_d512 correctors=%s converged=%s mg_levels=%s"
                   % (" ".join("%s,%s,%s,%s" % c for c in cor) or "-", conv, mgl))
    # ---- RECORDED
    maxn, maxrun = None, None
    for r in ["d512", "d512_bhatdev", "d3328_i", "d3328_ii"]:
        if logs[r] is None:
            continue
        rss, wall = time_info(mem, r)
        geom, phases = closed_form.parse(logs[r])
        free = (rd(os.path.join(mem, r + ".free")) or "").splitlines()
        mem_line = next((l for l in free if l.startswith("Mem:")), "-")
        out.append("RECORDED run=%s exit=%s host_max_rss_kb=%s wall_s=%s free_b_before=[%s]"
                   % (r, rcs[r], rss, wall, " ".join(mem_line.split())))
        out.append("RECORDED run=%s phase_seconds=%s exact_bytes=%s" % (
            r, ",".join("%s:%s" % (p, phases[p][-1]["seconds"]) for p in closed_form.PHASES if p in phases),
            ",".join("%s:%d" % (p, phases[p][-1]["exact"]) for p in closed_form.PHASES if p in phases)))
        if geom and rcs[r] == 0 and (maxn is None or geom["n"] > maxn):
            maxn, maxrun = geom["n"], r
    out.append("RECORDED max_cells_exit0 n=%s run=%s" % (maxn, maxrun))
    s = closed_form.parse(logs["d512"] or "")[1]
    sd = closed_form.parse(logs["d512_bhatdev"] or "")[1]
    if "S2" in s and "S2" in sd:
        out.append("RECORDED S_vs_S+ run=512 S2_seconds_S+(mapped)=%s S2_seconds_S(device)=%s wall_s_S+=%s wall_s_S=%s"
                   % (s["S2"][-1]["seconds"], sd["S2"][-1]["seconds"], time_info(mem, "d512")[1],
                      time_info(mem, "d512_bhatdev")[1]))
    else:
        out.append("RECORDED S_vs_S+ not available (S2 phase line missing in d512 or d512_bhatdev)")
    kv = probe_info(logs["probe1"])
    out.append("RECORDED versions driver=%s runtime=%s" % (kv.get("driver", "-"), kv.get("runtime", "-")))
    return out


# ---------------------------------------------------------------------------------------------------------------
def selftest():
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    def setup_lines(status="converged", conv=1, rel="3.0e-11"):
        return ("SETUP   SF-19: mg_levels=7 converged=%d G=(1,1,1)\n" % conv
                + "".join("SETUP   SF-19 corrector %s: status=%s its=%d rel_residual=%s final_residual=1e-12\n"
                          % (c, status, 20 + i, rel) for i, c in enumerate("xyz")))

    def write_run(d, name, log, rc, pid, peak, devpeak, nsamp=20, tree=None):
        with open(os.path.join(d, name + ".log"), "w") as f:
            f.write(log)
        with open(os.path.join(d, name + ".time"), "w") as f:
            f.write("\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n\tMaximum resident set size (kbytes): 4096\n")
        with open(os.path.join(d, name + ".free"), "w") as f:
            f.write("              total        used\nMem:    134000000000  1000000000\n")
        with open(os.path.join(d, name + ".apps.csv"), "w") as f:
            for i in range(nsamp):
                f.write("2026/10/22 10:00:%02d.000, %d, %d\n" % (i, pid, peak * (i + 1) // nsamp))
                f.write("2026/10/22 10:00:%02d.100, 7, 50\n" % i)
        with open(os.path.join(d, name + ".dev.csv"), "w") as f:
            for i in range(nsamp):
                f.write("2026/10/22 10:00:%02d.000, %d\n" % (i, devpeak * (i + 1) // nsamp))
        with open(os.path.join(d, name + ".pid_tree"), "w") as f:
            f.write(str(tree if tree is not None else pid) + "\n")
        with open(os.path.join(d, "exit_codes.tsv"), "a") as f:
            f.write("%s\t%d\n" % (name, rc))

    with tempfile.TemporaryDirectory() as d:
        for i, pk in enumerate([310, 300, 305], 1):
            write_run(d, "probe%d" % i, "CONTEXT_PROBE pid=10%d device=0 driver=12040 runtime=12020 device_used=%d\n"
                      % (i, pk * 2**20), 0, 100 + i, pk, pk + 10)
        small, _ = closed_form.synth(32, 32, 32, 3)
        dev, _ = closed_form.synth(32, 32, 32, 3, bhat="device")
        write_run(d, "d512", small + setup_lines(), 0, 4242, 301, 320)
        write_run(d, "d512_bhatdev", dev + setup_lines(), 0, 4242, 303, 320)
        big, g = closed_form.synth(3328, 384, 384, 7)
        emax_mib = max(int(x.split("exact_bytes=")[1].split()[0]) for x in big.splitlines() if x.startswith("MEM_PHASE")) / 2.0**20
        write_run(d, "d3328_i", big + setup_lines(), 0, 4242, int(emax_mib) + 400, int(emax_mib) + 450)
        # ii: OOM in S3 -> truncated log, exit 1, memory class
        trunc = "\n".join(l for l in big.splitlines() if not l.startswith(("MEM_PHASE phase=S3", "MEM_PHASE phase=T")))
        write_run(d, "d3328_ii", trunc + "\nCUDA error: out of memory (phase=S3)\n" + setup_lines(), 1, 4242,
                  int(emax_mib) + 400, int(emax_mib) + 450)
        out = run_report(d, dry=False)
        txt = "\n".join(out)
        expect("context", "C_CONTEXT mib=300" in txt)
        expect("c_pass_d512", "C_RUN run=d512 PASS" in txt)
        expect("c_pass_3328_i", "C_RUN run=d3328_i PASS" in txt)
        expect("c_nev_3328_ii", "C_RUN run=d3328_ii NOT_EVALUABLE" in txt)
        expect("cs_prefix", "CS_RUN run=d512_bhatdev PASS" in txt)
        expect("d512_pass", re.search(r"D_VERDICT run=d512 .* PASS", txt) is not None and "D_ALT" in txt)
        expect("d3328_i_pass", re.search(r"D3328_VERDICT run=d3328_i exit=0 .* exit_class=ok PASS", txt) is not None)
        expect("d3328_ii_fail_memory", re.search(r"D3328_VERDICT run=d3328_ii exit=1 .* exit_class=memory FAIL", txt) is not None)
        # E: levels=7 but coarsest 52x6x6 holds for the big geometry only
        expect("e_pass_3328_i", re.search(r"E_VERDICT run=d3328_i .*52x6x6 PASS", txt) is not None)
        expect("warm_unavailable", "warm probe unavailable" in txt)
        expect("recorded_max_cells", "max_cells_exit0 n=490733568 run=d3328_i" in txt)
        expect("recorded_S_vs_S+", "S_vs_S+ run=512 S2_seconds_S+(mapped)=1.5" in txt)
        # limits: peak above 30000 -> FAIL ; D512 above limit -> FAIL
        write_run(d, "d3328_i", big + setup_lines(), 0, 4242, 30001, 30100)
        txt2 = "\n".join(run_report(d, dry=False))
        expect("d3328_over", re.search(r"D3328_VERDICT run=d3328_i exit=0 pid_peak_mib=30001 .* FAIL", txt2) is not None)
        write_run(d, "d512", small + setup_lines(), 0, 4242, 400, 420)
        txt3 = "\n".join(run_report(d, dry=False))
        expect("d512_over", re.search(r"D_VERDICT run=d512 .* FAIL", txt3) is not None)
        # few samples -> NOT_EVALUABLE
        write_run(d, "d512", small + setup_lines(), 0, 4242, 301, 320, nsamp=3)
        txt4 = "\n".join(run_report(d, dry=False))
        expect("d512_samples", re.search(r"D_VERDICT run=d512 .* NOT_EVALUABLE", txt4) is not None)
        # E failing: unconverged correctors
        write_run(d, "d3328_i", big + setup_lines(status="max_iterations", conv=0), 0, 4242, int(emax_mib) + 400,
                  int(emax_mib) + 450)
        txt5 = "\n".join(run_report(d, dry=False))
        expect("e_fail", re.search(r"E_VERDICT run=d3328_i .* FAIL", txt5) is not None)
        # E: the small geometry (levels=3) fails the literal expectation and passes with --dry
        sm = small + setup_lines()
        expect("e_literal_small", e_verdict("x", sm, 0, False)[1] == "FAIL")
        expect("e_dry_small", e_verdict("x", sm, 0, True)[1] == "PASS")
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK mem_verdict")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if len(sys.argv) < 2 or any(a not in ("--dry",) for a in sys.argv[2:]):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    print("\n".join(run_report(sys.argv[1], "--dry" in sys.argv[2:])))
