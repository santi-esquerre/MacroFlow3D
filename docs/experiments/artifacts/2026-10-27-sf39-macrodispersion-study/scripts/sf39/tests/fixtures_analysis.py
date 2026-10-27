"""Fake COMPLETE SF-39 campaign for the analysis tests (small counts; seconds to a minute).

Built on `fixtures_campaign` (pilot of every arm, Par2 pilots) and `fixtures_runner`: the pilot is built and checkpointed first (the frozen
`tau_b_star.json` is the real checkpoint output), then every arm is extended to the budget counts of BUDGET, directory B, the s4 width arms
and the Par2 extensions are added, and `A/configs/budget.json` is written.  Global realization index g: directory A r = g, directory B
r = g - N_A (seed 5001 + g), narrow / twin arms r = g, width arms seed 7101/7201 + r (generating index WIDTH_G0 + r).  The content of
global realization g is `fx.gen(g)` scaled per arm role (`fc.arm_factors`), so every window / plane value has a closed form
(`fc.expected_alpha_TAM`, `fc.expected_alpha_TP`).  s1 is built with identity role factors: every rung and the twin equal the production
content (converged ladders by construction); s225 and s4 keep the fixture's role factors (not converged by construction).
"""
import json
import os
from unittest import mock

import numpy as np

from sf39 import campaign as cp
from sf39 import checkpoint as ck
from sf39.tests import fixtures_campaign as fc
from sf39.tests import fixtures_par2 as fp
from sf39.tests import fixtures_runner as fx

N_PILOT = 6
BUDGET = {"s1": {"N_narrow": 8, "N_twin": 7, "N_hmax": 7, "N_R": 10, "N_A": 7, "N_run": 10, "par2_NR": 10},
          "s225": {"N_narrow": 7, "N_twin": 6, "N_hmax": 6, "N_R": 8, "N_A": 6, "N_run": 8, "par2_NR": 8},
          "s4": {"N_narrow": 8, "N_twin": 7, "N_hmax": 7, "N_R": 10, "N_A": 7, "N_run": 10, "par2_NR": 10}}
REF_FLOOR = 9        # tests patch power.REF_FLOOR / WIDTH_ARMS to this (the real values are 30) to keep the fixture small
WIDTH_G0 = 100       # generating index of width arm realization r
IDENT = {k: (1.0, 1.0) for k in fc.ROLE_F}


def par2_a(g):
    """Par2 variance slopes (x, y, z) of global realization g (distinct per g so the per-realization alphas vary)."""
    return (1.0 + 0.05 * g, 0.5 + 0.01 * g, 0.25 + 0.02 * (g % 3))


def par2_u(g, amp):
    return fc.PAR2_U[g] * (1.0 + 0.1 * fc.AMP_F[amp])


def par2_var(g):
    """Realized var_lnK of global Par2 realization g (matches fc.build_par2's pilot lines for g < n_pilot)."""
    return 1.0 + 0.01 * g


def par2_tb_step(g):
    return fc.PAR2_TB_STEPS[g]


def build_single(out_root, arm, amp, nr, g0, seed0, role_arm=None):
    """Single-pass arm with nr DONE realizations whose index r holds global realization g0 + r (seed seed0 + r)."""
    out = cp.arm_dir(out_root, arm)
    tf, qf = fc.arm_factors(role_arm or arm, amp)
    for r in range(nr):
        g0r = fx.gen(g0 + r)
        g = dict(g0r, c2=g0r["c2"] * tf, c3=g0r["c3"] * tf)
        R = "%s/realizations/r_%04d" % (out, r)
        tx = fx.write_pass(R, g, fx.PLANES)
        fx._wjson(R + "/field.json", {"kind": "gaussian", "sigma2": 1.0, "ell": fx.ELL, "seed": seed0 + r, "applied_scale": 0.9,
                                       "raw_variance": 1.2, "final_variance": 1.0})
        fx._wjson(R + "/flow_stats.json", {"qbar": g["qbar"], "h": 1.0 / 256, "L1": fx.L1, "lambda": fx.ELL, "x_faces_nonpositive": 0,
                                           "x_nonpositive_fraction": 0.0,
                                           "speed_over_qbar_percentiles": {"p0": 0.2, "p0.1": 0.25, "p1": 0.3, "p50": 0.9},
                                           "interior": {"x1_lo": fx.X0, "x1_hi": fx.L1 - fx.X0, "mean_u": g["qbar"], "mean_v": 0.0,
                                                        "mean_w": 0.0, "theta_2": 0.0, "theta_3": 0.0}})
        fx.write_pollock(R + "/pollock", g0r, fx.PLANES)
        fx.write_series("%s/stats_twin/realization_%04d_timeseries.csv" % (out, r), tx)
        fc._rewrite_flow(R, qf)
        open(R + "/DONE", "w").close()


def build_par2_ext(out_root, amp, n_pilot, n_ext):
    """Par2 extension directory (global index g = n_pilot + r) and its FIELDSTAT file; the pilot series are rewritten with par2_a."""
    c = cp.PAR2_ID[amp]
    t = 125.0 * np.arange(1, 41)
    raw0 = os.path.join(out_root, "par2", "raw", "par2_s" + c)
    raw1 = raw0 + "_ext"
    for raw, base, n in ((raw0, 0, n_pilot), (raw1, n_pilot, n_ext)):
        for r in range(n):
            g = base + r
            u = par2_u(g, amp)
            fp.write_series(raw, r, t, mean_x=lambda tt, u=u: fp.INJ + u * tt,
                            var=tuple((lambda tt, ak=ak: ak * tt + 7.0) for ak in par2_a(g)))
            if base:
                s = par2_tb_step(g)
                snaps = {st: [50.0, 200.0] for st in range(10, s + 1, 10)}
                snaps[s + 10] = [50.0, 230.0]
                snaps[s + 20] = [50.0, 100.0]
                fp.write_snapshots(raw, r, snaps)
    with open(os.path.join(out_root, "par2", "field_stats_par2_s%s_ext.txt" % c), "w") as f:
        for r in range(n_ext):
            g = n_pilot + r
            f.write("FIELDSTAT r=%d seed=%d n=1024 mean_lnK=1.0e-03 var_lnK=%.6e Kmin=0.1 Kmax=9.0 Kmean=1.2\n" % (r, 9001 + g, par2_var(g)))


def build_complete(out_root, artifact, real_scripts, n_pilot=N_PILOT, bud=BUDGET):
    """Pilot + checkpoint + extension to the budget + budget.json.  Returns the checkpoint exit code (0)."""
    fc.build_campaign(out_root, artifact, n_pilot, real_scripts)
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        rc = ck.run(out_root, artifact, n_pilot=n_pilot)
    assert rc == 0, rc
    for amp in cp.AMPS:
        b, a = bud[amp], cp.arm_ids(amp)
        ctx = mock.patch.dict(fc.ROLE_F, IDENT) if amp == "s1" else mock.patch.dict(fc.ROLE_F, {})
        with ctx:
            fc.build_arm(out_root, a["A"], amp, b["N_A"])
            for arm in a["narrow"]:
                n = max(b["N_narrow"], REF_FLOOR) if arm == a["narrow"][0] and amp == "s4" else b["N_narrow"]
                fc.build_arm(out_root, arm, amp, n)
            fc.build_arm(out_root, a["tw"], amp, max(b["N_twin"], REF_FLOOR) if amp == "s4" else b["N_twin"])
            if b["N_run"] > b["N_A"]:
                build_single(out_root, a["B"], amp, b["N_run"] - b["N_A"], b["N_A"], cp.PILOT_SEED0 + b["N_A"], role_arm=a["A"])
            if amp == "s4":
                build_single(out_root, "s4_w24", amp, REF_FLOOR, WIDTH_G0, 7101, role_arm=a["tw"])
                build_single(out_root, "s4_w12", amp, REF_FLOOR, WIDTH_G0 + 50, 7201, role_arm=a["tw"])
        build_par2_ext(out_root, amp, n_pilot, b["par2_NR"] - n_pilot)
    write_budget(artifact, bud)
    return rc


def write_budget(artifact, bud):
    os.makedirs(os.path.join(artifact, "configs"), exist_ok=True)
    with open(os.path.join(artifact, "configs", "budget.json"), "w") as f:
        json.dump(bud, f)
