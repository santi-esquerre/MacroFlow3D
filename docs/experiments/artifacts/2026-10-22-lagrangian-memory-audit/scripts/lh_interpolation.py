#!/usr/bin/env python3
"""Log-log interpolation of the SF-37 post-hoc grid errors to the legacy MECOM resolution.

Stdlib only, deterministic. Input: the versioned output of the post-hoc script lh10.py
(../../2026-10-19-sf37-lagrangian-cube-verification/analysis/posthoc/lh10.txt), which holds, per amplitude,
the worst-of-3-seeds grid error against N = 512 at lambda/h = 10 (N = 40) and 20 (N = 80). Between those two
points the error is interpolated as a power law in lambda/h (straight line in log-log) to
lambda/h = 56.4 / 5 = 11.28 (legacy MECOM: ell = 56.4, h = 5; posthoc note II.3). The numbers are
discretization errors of the O6 grid ladder with respect to N = 512, NOT values of any dispersivity.
Usage: python3 lh_interpolation.py
"""
import math
import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[2] / "2026-10-19-sf37-lagrangian-cube-verification"
       / "analysis" / "posthoc" / "lh10.txt")
LH_LEGACY = 56.4 / 5.0


def main():
    sec = None
    data = {}
    for ln in SRC.read_text().splitlines():
        m = re.match(r"sigma\^2 ([\d.]+) \(worst", ln)
        if m:
            sec = m.group(1)
            data[sec] = {}
            continue
        m = re.match(r"\s+lam/h (\d+): sigT2\(16\) ([\d.]+)% .* aT-rate abs err ([\d.e+-]+) lambda \| aL-rate ([\d.]+)%", ln)
        if m and sec:
            data[sec][int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    print(f"source: {SRC.name}; interpolation point lambda/h = 56.4/5 = {LH_LEGACY:.2f}")
    print("sigma^2   quantity                       err(10)     err(20)   slope(log-log)   err(11.28)")
    for s2, d in data.items():
        for i, (q, unit) in enumerate((("sigma_T^2(x1=16) error [%]", ""), ("transverse-rate error [lambda]", ""),
                                       ("longitudinal-rate error [%]", ""))):
            e10, e20 = d[10][i], d[20][i]
            p = math.log(e20 / e10) / math.log(2.0)
            e = e10 * (LH_LEGACY / 10.0) ** p
            fmt = "{:10.3e}" if i == 1 else "{:10.2f}"
            print(f"{s2:>7s}   {q:<30s} {fmt.format(e10)}  {fmt.format(e20)}   {p:13.3f}   {fmt.format(e)}")


if __name__ == "__main__":
    main()
