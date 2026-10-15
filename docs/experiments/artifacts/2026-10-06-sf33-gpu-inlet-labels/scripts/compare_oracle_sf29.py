#!/usr/bin/env python3
"""SF-33 N8': production oracle (GPU driver, SF-19 Darcy on the SF-28 spline) vs the SF-29 spectral oracle.

Usage:
  compare_oracle_sf29.py --case N GPU_ORACLE_DIR SF29_CASE_DIR [--case ...] [--out md]

GPU_ORACLE_DIR: `inlet_slab --production --cells <crosscheck_gauss_0.25_N>/Y_cells.npy --eps 1 --save-oracle DIR`
(psi_or_{1,2}.npy = primary run h/8, tol 1e-8; ladder runs with suffixes). SF29_CASE_DIR: the export_proto.py case
directory gauss_0.25_N (psi_or_{1,2}.npy, DOP853 oracle on the spectral Darcy reference). Both are FULL labels,
(N+1, N, N), slab layout [j, m2, m3] (vertex (j/N, m2/N, m3/N)).

Per label i: d = psi_or_i^GPU - psi_or_i^SF29 on planes 0..N; normalized by RMS(psi_or_i^SF29 - affine_i)
(affine_1 = x2 = m2/N, affine_2 = x3 = m3/N); reported RMS(d), max|d|, mean(d), RMS(d - mean d), plane 0 (inlet labels
only: D-1 from the SF-19 face flux vs the spectral reference) and planes 1..N separately. Observed orders between
consecutive grids: log(e_a / e_b) / log(N_b / N_a). numpy only.
"""
import argparse
import math
import os
import sys

import numpy as np


def stats(gpu, ref, N, label):
    j = np.arange(N + 1)[:, None, None]
    m2 = np.arange(N)[None, :, None]
    m3 = np.arange(N)[None, None, :]
    aff = (m2 / N + 0 * j + 0 * m3) if label == 1 else (m3 / N + 0 * j + 0 * m2)
    scale = math.sqrt(np.mean((ref - aff) ** 2))
    d = gpu - ref
    rms = lambda a: math.sqrt(float(np.mean(a ** 2)))
    out = {
        "scale": scale,
        "rms": rms(d) / scale,
        "max": float(np.max(np.abs(d))) / scale,
        "mean": float(np.mean(d)) / scale,
        "rms_demeaned": rms(d - np.mean(d)) / scale,
        "rms_inlet": rms(d[0]) / scale,
        "max_inlet": float(np.max(np.abs(d[0]))) / scale,
        "rms_interior": rms(d[1:]) / scale,
        "max_interior": float(np.max(np.abs(d[1:]))) / scale,
        "max_plane": int(np.argmax(np.max(np.abs(d), axis=(1, 2)))),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", nargs=3, action="append", metavar=("N", "GPU_DIR", "SF29_DIR"), required=True)
    ap.add_argument("--suffix", default="", help="GPU oracle file suffix (ladder run), e.g. _h16_tol1e-10")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for n_s, gdir, rdir in a.case:
        N = int(n_s)
        r = {"N": N}
        for i in (1, 2):
            g = np.load(os.path.join(gdir, f"psi_or_{i}{a.suffix}.npy"))
            f = np.load(os.path.join(rdir, f"psi_or_{i}.npy"))
            if g.shape != (N + 1, N, N) or f.shape != (N + 1, N, N):
                sys.exit(f"shape mismatch at N={N}: gpu {g.shape} sf29 {f.shape}")
            r[i] = stats(g, f, N, i)
        rows.append(r)
    rows.sort(key=lambda r: r["N"])
    keys = ["rms", "max", "mean", "rms_demeaned", "rms_inlet", "max_inlet", "rms_interior", "max_interior"]
    L = []
    L.append(f"GPU oracle file suffix: '{a.suffix or '(primary h/8, tol 1e-8)'}'")
    L.append("")
    L.append("| N | label | scale RMS(psi_or^SF29 - affine) | " + " | ".join(keys) + " | plane of max |")
    L.append("|---|---|---|" + "---|" * len(keys) + "---|")
    for r in rows:
        for i in (1, 2):
            s = r[i]
            L.append(f"| {r['N']} | psi{i} | {s['scale']:.6e} | " + " | ".join(f"{s[k]:.3e}" for k in keys)
                     + f" | {s['max_plane']} |")
    L.append("")
    L.append("Observed orders log(e_a/e_b)/log(N_b/N_a) between consecutive grids:")
    L.append("")
    L.append("| pair | label | " + " | ".join(keys) + " |")
    L.append("|---|---|" + "---|" * len(keys))
    for ra, rb in zip(rows, rows[1:]):
        for i in (1, 2):
            o = []
            for k in keys:
                ea, eb = abs(ra[i][k]), abs(rb[i][k])
                o.append(f"{math.log(ea / eb) / math.log(rb['N'] / ra['N']):.2f}" if ea > 0 and eb > 0 else "n/a")
            L.append(f"| {ra['N']}->{rb['N']} | psi{i} | " + " | ".join(o) + " |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
