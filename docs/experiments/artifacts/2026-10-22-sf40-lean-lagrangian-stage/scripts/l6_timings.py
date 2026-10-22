#!/usr/bin/env python3
"""SF-40 RR9 helper (run_sf40.sh part l6): tracker kernel timings of the four L6 variants.

  l6_timings.py PART_DIR          PART_DIR holds default/ fused/ single/ both/ (each a --save-particles directory with
                                  track.json), <variant>.log, exit_codes.tsv (`l6_<variant><TAB>rc`), walls.tsv.
  l6_timings.py --selftest
Prints the TSV `l6_timings.tsv` (header + one row per variant) followed by `RECORDED L6_SPEEDUP` lines.  Kernel seconds
come from track.json `timings_s` (`plane_kernel`, `time_kernel`, or `single_kernel` with --single-launch, where
plane/time are null); `kernel_total_s` = plane + time, or single_kernel; nfev from the TSTATUS lines of the log.
Speedup = kernel_total_s(default) / kernel_total_s(variant).  Not a test: the numbers are recorded, no threshold.
"""
import json
import os
import re
import sys

VARIANTS = ["default", "fused", "single", "both"]
HEAD = ["variant", "rc", "wall_s", "plane_kernel_s", "time_kernel_s", "single_kernel_s", "download_s", "moments_s",
        "kernel_total_s", "nfev_plane", "nfev_time", "nfev_total", "speedup_vs_default"]


def tsv_last(path, key):
    val = ""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p and p[0] == key and len(p) > 1:
                    val = p[1]
    return val


def row(part, variant):
    d = os.path.join(part, variant)
    r = {"variant": variant, "rc": tsv_last(os.path.join(part, "exit_codes.tsv"), "l6_" + variant),
         "wall_s": tsv_last(os.path.join(part, "walls.tsv"), "l6_" + variant)}
    t = {}
    try:
        with open(os.path.join(d, "track.json")) as f:
            t = json.load(f).get("timings_s", {}) or {}
    except (OSError, ValueError):
        pass
    for k in ("plane_kernel", "time_kernel", "single_kernel", "download", "moments"):
        v = t.get(k)
        r[k + "_s"] = "" if v is None else repr(float(v))
    nf = {}
    try:
        with open(os.path.join(part, variant + ".log"), errors="replace") as f:
            for line in f:
                m = re.match(r"TSTATUS pass=(\w+) .*\bnfev=(\d+)", line)
                if m:
                    nf[m.group(1)] = int(m.group(2))
    except OSError:
        pass
    r["nfev_plane"] = str(nf["plane"]) if "plane" in nf else ""
    r["nfev_time"] = str(nf["time"]) if "time" in nf else ""
    r["nfev_total"] = str(sum(nf.values())) if nf else ""
    tot = None
    if t.get("single_kernel") is not None:
        tot = float(t["single_kernel"])
    elif t.get("plane_kernel") is not None and t.get("time_kernel") is not None:
        tot = float(t["plane_kernel"]) + float(t["time_kernel"])
    r["kernel_total_s"] = "" if tot is None else repr(tot)
    r["_tot"] = tot
    return r


def report(part):
    rows = [row(part, v) for v in VARIANTS]
    base = rows[0]["_tot"]
    for r in rows:
        r["speedup_vs_default"] = "" if (base is None or r["_tot"] in (None, 0.0)) else "%.4f" % (base / r["_tot"])
    lines = ["\t".join(HEAD)] + ["\t".join(r[h] for h in HEAD) for r in rows]
    for r in rows[1:]:
        lines.append("RECORDED L6_SPEEDUP variant=%s kernel_total_s=%s default_kernel_total_s=%s speedup=%s launches=%d"
                     % (r["variant"], r["kernel_total_s"] or "n/a", rows[0]["kernel_total_s"] or "n/a",
                        r["speedup_vs_default"] or "n/a", 1 if r["variant"] in ("single", "both") else 2))
    return lines


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for v, tj in [("default", {"plane_kernel": 4.0, "time_kernel": 2.0, "download": 0.1, "moments": 0.2}),
                      ("fused", {"plane_kernel": 3.0, "time_kernel": 1.5, "download": 0.1, "moments": 0.2}),
                      ("single", {"plane_kernel": None, "time_kernel": None, "single_kernel": 5.0, "download": 0.1,
                                  "moments": 0.2}),
                      ("both", {"plane_kernel": None, "time_kernel": None, "single_kernel": 3.0, "download": 0.1,
                                "moments": 0.2})]:
            os.makedirs(os.path.join(d, v))
            with open(os.path.join(d, v, "track.json"), "w") as f:
                json.dump({"timings_s": tj}, f)
            with open(os.path.join(d, v + ".log"), "w") as f:
                f.write("TSTATUS pass=plane hist=1,0,0,0,0,0,0 landings_discarded=0 backflow_enc=0 nfev=1000\n"
                        "TSTATUS pass=time hist=1,0,0,0,0,0,0 landings_discarded=0 backflow_enc=0 nfev=500\n")
            with open(os.path.join(d, "exit_codes.tsv"), "a") as f:
                f.write("l6_%s\t0\n" % v)
            with open(os.path.join(d, "walls.tsv"), "a") as f:
                f.write("l6_%s\t10\t2026-10-22T00:00:00Z\n" % v)
        out = report(d)
        ok = (len(out) == 8 and out[0].startswith("variant\trc") and "speedup=2.0000" in out[-1]
              and "speedup=1.2000" in out[6] and out[1].split("\t")[9:12] == ["1000", "500", "1500"]
              and out[3].split("\t")[3] == "")
        if not ok:
            print("SELFTEST FAILED\n" + "\n".join(out))
            return 1
    print("SELFTEST OK l6_timings")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    print("\n".join(report(sys.argv[1])))
