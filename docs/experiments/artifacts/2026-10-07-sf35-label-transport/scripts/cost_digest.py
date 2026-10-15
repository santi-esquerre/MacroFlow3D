#!/usr/bin/env python3
"""SF-35 H: wall time, host max RSS and peak device memory at 256^3 per amplitude.

  cost_digest.py raw/matrix_eps*/N256.json logs/matrix_eps*/N256.time [--out analysis/cost_256.md]

Pass the JSON files and the /usr/bin/time -v files; they are paired by the matrix_eps<label> path component.
A JSON without a matching time file still reports the driver numbers (RSS then `nan`).  Deterministic.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import fnum, get, load_json, md_table, num, parse_time_file, write_out  # noqa: E402


def label(path):
    m = re.search(r"matrix_eps(\w+)", path)
    return m.group(1) if m else path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="N256.json and N256.time files of the amplitudes")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    js = {label(f): f for f in a.files if f.endswith(".json")}
    tf = {label(f): f for f in a.files if f.endswith(".time")}
    rows = []
    for lab in sorted(js):
        J = load_json(js[lab])
        T = parse_time_file(tf[lab]) if lab in tf else {"wall_s": float("nan"), "max_rss_kB": float("nan")}
        pd = num(get(J, "memory", "peak_device_bytes"))
        rows.append(("eps%s" % lab, J.get("status"), fnum(get(J, "timing", "total"), "%.1f"), fnum(T["wall_s"], "%.1f"),
                     fnum(T["max_rss_kB"] / 1024.0 / 1024.0, "%.2f"), fnum(pd / 2.0 ** 30, "%.2f"),
                     fnum(num(get(J, "timing", "trace")), "%.1f")))
    L = ["# SF-35 H: cost at 256^3", ""]
    L.append(md_table(["amplitude", "status", "driver total (s)", "wall /usr/bin/time (s)", "host max RSS (GiB)",
                       "peak device (GiB, device-wide poll)", "trace phase (s)"], rows))
    L.append("")
    write_out("\n".join(L), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
