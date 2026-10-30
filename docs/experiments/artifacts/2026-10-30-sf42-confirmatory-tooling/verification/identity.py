#!/usr/bin/env python3
"""SF-42 R1 helper of identity.sh (stdlib + numpy).

  identity.py config <case> <out.yaml> <par2_out_dir>   write the Par2 config of a field-identity case
  identity.py window <out.yaml> <par2_out_dir>          write the Par2 config of the injection-window case
  identity.py fields <consumer_dir> <par2_out_dir>      byte comparison of Y.npy / K.npy and var Y = 1
  identity.py injection <par2_out_dir>                  every injected particle inside the window

Cases (consumer flags and Par2 keys are the same numbers): name nx ny nz ell eps seed0. dx = '%.17g' % (1/ny), the consumer's
string; ell is written '%.17g'. Seeds are in the SF-42 reserved range 420201-420299.
"""
import filecmp
import math
import sys

import numpy as np

CASES = {
    "p2_e1": (64, 16, 16, 0.25, 1.0, 420211),   # power-of-two ny, eps 1
    "p2_e2": (64, 16, 16, 0.25, 2.0, 420212),   # power-of-two ny, eps 2
    "f3_e1": (96, 24, 24, 0.25, 1.0, 420213),   # ny = 24 = 8 * 3, eps 1
    "f3_e2": (96, 24, 24, 0.25, 2.0, 420214),   # ny = 24 = 8 * 3, eps 2
}
WINDOW = (0.1, 0.9, 0.2, 0.8)
WIN_GRID = (128, 32, 32, 0.25)  # nx, ny, nz, ell
WIN_NP = 2000


def g(x):
    return "%.17g" % x


def par2_yaml(nx, ny, nz, ell, out, sf18=None, window=None, np_=100, steps=2, dump=True, planes=None):
    dx = 1.0 / ny
    L1 = nx * dx
    y = ["run_mode: ensemble", "grid: {nx: %d, ny: %d, nz: %d, dx: %s}" % (nx, ny, nz, g(dx)),
         "stochastic:", "  sigma2: 1.0", "  corr_length: %s" % g(ell), "  n_modes: 1000", "  covariance_type: 1",
         "  seed: 42", "  K_mean: 1.0"]
    if sf18:
        y += ["  field_source: sf18_medium_b",
              "  sf18_medium_b: {sigma2: %s, eps: %s, seed0: %d}" % (g(1.0), g(sf18[0]), sf18[1])]
    y += ["flow:", "  solver: mg_cg", "  mg_levels: 3", "  mg_pre_smooth: 4", "  mg_post_smooth: 4", "  mg_coarse_iters: 50",
          "  mg_max_cycles: 20", "  cg_max_iter: 2000", "  cg_rtol: 1.0e-10", "  cg_check_every: 10", "  rtol: 1.0e-6",
          "  pin: {mode: off}", "  bc:", "    west:   { type: dirichlet, value: %s }" % g(L1),
          "    east:   { type: dirichlet, value: 0.0 }", "    south:  { type: periodic }", "    north:  { type: periodic }",
          "    bottom: { type: periodic }", "    top:    { type: periodic }",
          "transport:", "  n_particles: %d" % np_, "  dt: 0.05", "  n_steps: %d" % steps, "  seed: 777", "  output_every: 1000",
          "  inject_x: %s" % g(5 * ell)]
    if window:
        y.append("  inject_window: [%s]" % ", ".join(g(w) for w in window))
    y += ["analysis:", "  macrodispersion: {enabled: true, NR: 1, lambda: %s, vmean_norm: 1.0, sample_every: 1, var_estimator: biased}" % g(ell)]
    if planes:
        y += ["  plane_crossings:", "    enabled: true", "    dump_injection: true", "    planes: [%s]" % ", ".join(g(p) for p in planes)]
    y += ["output:", "  output_dir: %s" % out, "  save_K: false"]
    if dump:
        y += ["  dump_field:", "    enabled: true", "    realizations: [0]"]
    return "\n".join(y) + "\n"


def cmd_config(case, path, out):
    nx, ny, nz, ell, eps, seed0 = CASES[case]
    open(path, "w").write(par2_yaml(nx, ny, nz, ell, out, sf18=(eps, seed0)))


def cmd_window(path, out):
    nx, ny, nz, ell = WIN_GRID
    planes = [1.5, 2.0, 2.5]  # inside (5 ell, L1) = (1.25, 4): crossing statistics are not the object here
    open(path, "w").write(par2_yaml(nx, ny, nz, ell, out, sf18=(1.0, 420221), window=WINDOW, np_=WIN_NP, steps=2, dump=False, planes=planes))


def cmd_fields(cdir, pdir):
    ok = True
    for name in ("Y", "K"):
        a, b = "%s/r_0000/%s.npy" % (cdir, name), "%s/fields/r_0000/%s.npy" % (pdir, name)
        same = filecmp.cmp(a, b, shallow=False)
        A, B = np.load(a), np.load(b)
        same_data = A.shape == B.shape and A.dtype == B.dtype and A.tobytes() == B.tobytes()
        print("%s: files byte-identical %s; arrays (shape %s, dtype %s) byte-identical %s" % (name, same, A.shape, A.dtype, same_data))
        ok = ok and same_data
        if name == "Y":
            Yf = B.astype(np.float64).ravel()
            var = float(np.mean((Yf - Yf.mean()) ** 2))
            rel = abs(var - 1.0)
            print("var Y (mean-subtracted, biased) = %.17g, |var-1| = %.3e (limit 1e-12)" % (var, rel))
            ok = ok and rel <= 1e-12
    print("FIELD_IDENTITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def read_csv(path):
    hdr, rows = None, []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        if hdr is None:
            hdr = ln.strip().split(",")
        else:
            rows.append([float(v) for v in ln.strip().split(",")])
    return hdr, np.array(rows)


def cmd_injection(pdir):
    nx, ny, nz, ell = WIN_GRID
    Ly, Lz = ny * (1.0 / ny), nz * (1.0 / ny)
    hdr, r = read_csv("%s/crossings/realization_0000_injection.csv" % pdir)
    c1, c2, c3 = hdr.index("x1"), hdr.index("x2"), hdr.index("x3")
    w = WINDOW
    tol = 4e-16
    in2 = (r[:, c2] >= w[0] * Ly - tol) & (r[:, c2] <= w[1] * Ly + tol)
    in3 = (r[:, c3] >= w[2] * Lz - tol) & (r[:, c3] <= w[3] * Lz + tol)
    on_x = r[:, c1] == 5 * ell
    print("injected rows %d (expected %d); x1 == 5 ell: %d; inside x2 window: %d; inside x3 window: %d" % (len(r), WIN_NP, on_x.sum(), in2.sum(), in3.sum()))
    print("x2 range [%.6f, %.6f] window [%.6f, %.6f]; x3 range [%.6f, %.6f] window [%.6f, %.6f]" % (
        r[:, c2].min(), r[:, c2].max(), w[0] * Ly, w[1] * Ly, r[:, c3].min(), r[:, c3].max(), w[2] * Lz, w[3] * Lz))
    ok = len(r) == WIN_NP and on_x.all() and in2.all() and in3.all()
    print("INJECTION_WINDOW %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "config":
        cmd_config(*sys.argv[2:5])
    elif c == "window":
        cmd_window(*sys.argv[2:4])
    elif c == "fields":
        sys.exit(cmd_fields(*sys.argv[2:4]))
    elif c == "injection":
        sys.exit(cmd_injection(sys.argv[2]))
    elif c == "cases":
        for k, v in CASES.items():
            print(k, *v)
    else:
        sys.exit("unknown command")
