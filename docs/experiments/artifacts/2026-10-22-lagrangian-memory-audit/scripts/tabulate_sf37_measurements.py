#!/usr/bin/env python3
"""Tabulate, from the versioned SF-37 artifacts, the measurements used by the memory audit.

Stdlib only, deterministic, read-only. Reads (relative to ../../2026-10-19-sf37-lagrangian-cube-verification):
  logs/jobs/sf37-v2.log, logs/jobs/sf37-science.log   MEM_PEAK lines (nvidia-smi 200 ms, MiB)
  raw/v2/v2_e*_s*_N512.{json,time,log}                  timings, tracker timings, host RSS, corrector iterations
Usage: python3 tabulate_sf37_measurements.py
"""
import collections
import glob
import json
import re
from pathlib import Path

A = Path(__file__).resolve().parents[2] / "2026-10-19-sf37-lagrangian-cube-verification"
N512 = 512 ** 3


def main():
    print("== SF-37 MEM_PEAK values of the 512^3 runs (nvidia-smi memory.used, 200 ms, device-wide, MiB) ==")
    total = collections.Counter()
    for f in ("logs/jobs/sf37-v2.log", "logs/jobs/sf37-science.log"):
        c = collections.Counter()
        for m in re.finditer(r"MEM_PEAK run=\S*_N512\S* mib=(\d+)", (A / f).read_text()):
            c[int(m.group(1))] += 1
        total.update(c)
        print(f"  {f}: " + ", ".join(f"{v} x {k}" for k, v in sorted(c.items())) + f"  (runs: {sum(c.values())})")
    print("  both logs: " + ", ".join(f"{v} x {k}" for k, v in sorted(total.items())) + f"  (runs: {sum(total.values())})")
    base = min(total)
    for k, v in sorted(total.items()):
        print(f"    {k} MiB: {v:2d} runs; {k * 2**20 / N512:7.2f} B/cell; minus {base}: {k - base:5d} MiB")

    print("\n== SF-37 V2 512^3 runs (raw/v2/v2_e{10,15,20}_s{3101,3102,3103}_N512), 65536 particles ==")
    print("  run                 field[s] stage[s] track[s] total[s] plane[s] time[s]  nfev_plane   evals/step  RSS[kB]   peak_dev(PhaseLog)")
    its = collections.Counter()
    rss = []
    for jf in sorted(glob.glob(str(A / "raw/v2/v2_e*_N512.json"))):
        j = json.loads(Path(jf).read_text())
        name = Path(jf).stem
        t, tt, pp = j["timing"], j["tracker_timings_s"], j["plane_pass"]
        steps = pp["accepted_steps"] + pp["rejected_steps"] + pp["landing_steps"]
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", Path(jf[:-5] + ".time").read_text())
        rss.append(int(m.group(1)))
        for mm in re.finditer(r"its=(\d+)", Path(jf[:-5] + ".log").read_text()):
            its[int(mm.group(1))] += 1
        print(f"  {name:<19s} {t['field']:8.1f} {t['stage']:8.1f} {t['track']:8.1f} {t['total']:8.1f} "
              f"{tt['plane_kernel']:8.1f} {tt['time_kernel']:8.1f} {pp['nfev']:13d} {pp['nfev'] / steps:10.4f} "
              f"{int(m.group(1)):9d} {j['memory']['peak_device_bytes']}")
        assert pp["rejected_steps"] == 0
    print(f"  PCG iterations per SF-19 corrector over all 512^3 V2 logs: {dict(its)}")
    print(f"  host RSS: min {min(rss)} kB, max {max(rss)} kB = {max(rss) * 1024 / N512:.1f} B/cell")
    v3 = sorted(glob.glob(str(A / "raw/v3b/v3b_e*_N512.time")))
    vals = [int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", Path(f).read_text()).group(1)) for f in v3]
    print(f"  v3b 512^3 runs (keep_face_fluxes, Pollock): {len(vals)} runs, host RSS min {min(vals)} max {max(vals)} kB"
          f" = {max(vals) * 1024 / N512:.1f} B/cell")


if __name__ == "__main__":
    main()
