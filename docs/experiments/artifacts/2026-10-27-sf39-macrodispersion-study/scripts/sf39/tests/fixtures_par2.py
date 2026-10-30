"""Fake Par2 output trees and config helpers for test_par2 (synthetic known answers; no real run data)."""
import os

import numpy as np

# Small synthetic config: Lx = 64 * 5 = 320, lambda = 20 (threshold Lx - 5 lambda = 220), dt = 12.5, inject_x = 100.
SYN_CFG = """# synthetic
grid:
  nx: 64
  ny: 16
  nz: 16
  dx: 5.0
stochastic:
  sigma2: 1.0
  corr_length: 20.0
  seed: 9001
transport:
  dt: 12.5
  seed: 123456789
  inject_x: 100.0
analysis:
  macrodispersion:
    NR: 20
    sample_every: 10
  snapshots:
    every: 10
output:
  output_dir: raw/par2_s1
"""
LX, LAM, DT, INJ = 320.0, 20.0, 12.5, 100.0


def find_apps_cfg(name):
    """Real SF-38 config text, found by walking up from this file."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        p = os.path.join(d, "apps", name)
        if os.path.isfile(p):
            with open(p) as f:
                return f.read()
        if os.path.dirname(d) == d:
            raise FileNotFoundError(name)
        d = os.path.dirname(d)


def write_series(run_dir, r, t, mean_x=0.0, var=(0.0, 0.0, 0.0), sub="stats"):
    """Par2-format series (`# format_version` line, header, %.15e values); mean_x / var entries are scalars or callables of t."""
    t = np.asarray(t, float)

    def col(f):
        return np.asarray(f(t) if callable(f) else np.zeros_like(t) + f, float)

    cols = [t, col(mean_x), col(0.0), col(0.0)] + [col(f) for f in var]
    os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    with open(os.path.join(run_dir, sub, "realization_%04d_timeseries.csv" % r), "w") as f:
        f.write("# format_version=2\nt,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n")
        for i in range(len(t)):
            f.write(",".join("%.15e" % c[i] for c in cols) + ",1\n")


def write_snapshots(run_dir, r, by_step):
    """snapshots/r_%04d/step_%08d.csv with header `id,x coord,y coord,z coord`; by_step: {step: [x, ...]}."""
    d = os.path.join(run_dir, "snapshots", "r_%04d" % r)
    os.makedirs(d, exist_ok=True)
    for step, xs in by_step.items():
        with open(os.path.join(d, "step_%08d.csv" % step), "w") as f:
            f.write("id,x coord,y coord,z coord\n")
            for k, x in enumerate(xs):
                f.write("%d,%.15f,1.0,2.0\n" % (k, x))


def make_tree(root, n_pilot, n_ext, tb_steps, u_of_g, a_of_g=None):
    """Pilot (root/raw/par2_s1, g = r) and extension (root/raw/par2_s1_ext, g = 20 + r) trees.
    Realization g: snapshots at steps 10, 20, ...; clean (max x 200) up to and including step tb_steps[g] (a multiple of 10), then one
    dirty at x = 230 >= 220 and a later clean one (the scan must stop at the first dirty one): n_snapshots = tb_steps[g]/10 + 2.
    Series: t = 125 k (k = 1..40), mean_x = INJ + u t, var_k = a_k t + 7 (a_of_g(g) -> (ax, ay, az)).
    Returns (pilot_dir, ext_dir or None)."""
    dirs = [os.path.join(root, "raw", "par2_s1"), os.path.join(root, "raw", "par2_s1_ext") if n_ext else None]
    t = 125.0 * np.arange(1, 41)
    for base, d, n in ((0, dirs[0], n_pilot), (20, dirs[1], n_ext)):
        for r in range(n):
            g = base + r
            u = u_of_g(g)
            a = a_of_g(g) if a_of_g else (1.0, 0.5, 0.25)
            write_series(d, r, t, mean_x=lambda tt, u=u: INJ + u * tt, var=tuple((lambda tt, ak=ak: ak * tt + 7.0) for ak in a))
            s = tb_steps[g]
            snaps = {st: [50.0, 200.0] for st in range(10, s + 1, 10)}
            snaps[s + 10] = [50.0, 230.0]
            snaps[s + 20] = [50.0, 100.0]
            write_snapshots(d, r, snaps)
    return dirs[0], dirs[1]
