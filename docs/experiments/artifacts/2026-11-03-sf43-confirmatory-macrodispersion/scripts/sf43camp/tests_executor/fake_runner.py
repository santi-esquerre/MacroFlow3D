"""Fake `lagrangian_ensemble` / `macroflow3d_pipeline` for the executor tests.

    python3 fake_runner.py consumer <lagrangian_ensemble args>      (cwd irrelevant; --out given)
    python3 fake_runner.py par2 config.yaml                         (cwd = segment directory, output_dir relative)

They parse the real command line / YAML, apply the usage rules of the real binaries that matter here (exit 2 usage, planes strictly
increasing, > x0, < L1; Par2: the validator rules of ``miniyaml.validate_par2``) and write minimal outputs with the SF-42 synthetic
generators (``sf42red.tests.synth``), plus the log lines the executor parses.  Failure injection by environment variables:

  FAKE_FAIL4_AT=r      consumer: STAGE_FAILED (exit 4) at realization index r
  FAKE_FALLBACK_AT=r   consumer: solver verdict 'fallback' (accepted) at r
  FAKE_ERR_AT=r        consumer/Par2: print FAKE_ERR_TEXT on stderr and exit FAKE_ERR_EXIT (default 1) before realization r
  FAKE_NOCONV_SEED=s / FAKE_FALLBACK_SEED=s   the same for the realization whose field seed is s (Par2 and consumer, independent of the
                       position inside the invocation)
  FAKE_PAR2_INCOMPLETE_SEED=s   Par2: the realization whose field seed is s has 99 of its 100 particles crossing every plane (completeness 0.99)
  FAKE_NOCONV_AT=r     Par2: realization r prints Converged=NO beyond the fallback (final/initial 1e-5; exit stays 0)
  FAKE_FALLBACK_AT=r   Par2: realization r prints Converged=NO with final/initial 1e-12 (verdict 'fallback', usable)
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "2026-10-30-sf42-confirmatory-tooling", "scripts"))
sys.path.insert(0, HERE)

from sf42red.tests import synth  # noqa: E402

import miniyaml  # noqa: E402


def env_int(name):
    v = os.environ.get(name)
    return None if v is None else int(v)


def seed_hit(name, seed):
    return os.environ.get(name) is not None and int(os.environ[name]) == seed


def inject_error(r):
    if env_int("FAKE_ERR_AT") == r:
        sys.stderr.write(os.environ.get("FAKE_ERR_TEXT", "ERROR: something") + "\n")
        sys.exit(int(os.environ.get("FAKE_ERR_EXIT", "1")))


def b_solver(r, converged, iterations, fallback=0, rel="1.000000e-12", balance="1.000000e-10", max_div="2.000000e-08"):
    """The B_SOLVER line of DirichletLongBoxStage (format_b_solver)."""
    return ("B_SOLVER r=%d converged=%d iterations=%d initial=1.000000e+07 final=1.000000e-04 rel=%s qbar=1.000000000000e+00 "
            "Q_in=1.000000000000e+00 Q_out=1.000000000001e+00 balance=%s max_div=%s fallback=%d" % (r, converged, iterations, rel, balance, max_div, fallback))


def consumer(argv):
    val, flags, i = {}, set(), 0
    while i < len(argv):
        if argv[i] in ("--fused-splines", "--single-launch"):
            flags.add(argv[i])
            i += 1
        else:
            val[argv[i]], i = argv[i + 1], i + 2
    for k in ("--grid", "--ell", "--seed0", "--out", "--eps", "--np-side", "--planes"):
        if k not in val:
            sys.stderr.write("usage error: %s required\n" % k)
            return 2
    nx, ny, nz = (int(v) for v in val["--grid"].split(","))
    ell, M, s0, nr = float(val["--ell"]), int(val["--np-side"]), int(val["--seed0"]), int(val.get("--nr", "1"))
    x0, L1 = 5.0 * ell, nx / ny
    planes = [float(v) for v in val["--planes"].split(",")]
    if any(b <= a for a, b in zip(planes, planes[1:])) or planes[0] <= x0 or planes[-1] >= L1 or M % 2:
        sys.stderr.write("usage error: planes/np-side\n")
        return 2
    hm = [int(v) for v in val.get("--hmax-div", "8").split(",")]
    out = val["--out"]
    d_planes = [(p - x0) / ell for p in planes]
    os.makedirs(os.path.join(out, "realizations"), exist_ok=True)
    for r in range(nr):
        seed = s0 + r
        if os.path.exists(os.path.join(out, "realizations", "r_%04d" % r, "DONE")):
            print("R=%d SKIP done (DONE present)" % r)
            continue
        inject_error(r)
        print("R=%d FIELD gaussian (SF-18) N=128 grid=%dx%dx%d sigma2=1 ell=%s seed=%d normalize_variance=1" % (r, nx, ny, nz, val["--ell"], seed))
        print("MEM_PHASE phase=S1 exact_bytes=%d cufft_work=1024 device_used=%d seconds=0.1 items=a:1" % (3 * 2 ** 30 + r, 4 * 2 ** 30 + r))
        if env_int("FAKE_FAIL4_AT") == r or seed_hit("FAKE_NOCONV_SEED", seed):
            print("R=%d %s" % (r, b_solver(r, 0, 4000, 0, rel="3.390000e-07", balance="2.640000e-07")))
            print("R=%d STAGE medium=b status=not_converged_beyond_fallback grid=%dx%dx%d solver_converged=0 solver_usable=0" % (r, nx, ny, nz))
            print("R=%d STAGE_FAILED status=solver_failed eps=%s verdict=not_converged_beyond_fallback: the medium (b) solve is not usable"
                  % (r, val["--eps"]))
            return 4
        verdict = "fallback" if env_int("FAKE_FALLBACK_AT") == r or seed_hit("FAKE_FALLBACK_SEED", seed) else "converged"
        # the tracker totals of `track.json` (plane_pass / seed_backflow; `reduce --tracker-counts`): deterministic from the seed and the pass
        proots = [synth.write_pass(out, M=M, ell=ell, x0=x0, d_planes=d_planes, r=r, hmax=d if len(hm) > 1 else None, seed=seed,
                                   a_T=0.03 * (1.0 + 0.1 * (seed % 7)),
                                   tracker={"recrossings": seed % 5 + d // 8, "backflow_encounters": seed % 3, "seed_backflow": seed % 2}) for d in hm]
        rdir = os.path.join(out, "realizations", "r_%04d" % r)
        if val.get("--pollock-bridge", "off") == "on":  # R/pollock/ in the format of EnsembleRunner.cu: once per realization, whatever the passes
            tw = synth.write_pass(os.path.join(out, "_pollock_cloud"), M=M, ell=ell, x0=x0, d_planes=d_planes, seed=seed, U0=2.5,
                                  a_T=0.041 * (1.0 + 0.13 * (seed % 5)))  # a different cloud: the Pollock contrast is not zero
            synth.write_pollock(rdir, proots[0], planes_x23=np.load(os.path.join(tw, "planes_x23.npy")),
                                planes_t=np.load(os.path.join(tw, "planes_tau.npy")))
            shutil.rmtree(os.path.join(out, "_pollock_cloud"))
        with open(os.path.join(rdir, "solver_b.json"), "w") as f:
            json.dump({"r": r, "seed": seed, "verdict": verdict, "usable": True, "converged": verdict == "converged", "iterations": 30 + r,
                       "rel": 1e-12, "balance": 1e-10, "max_div": 2.0e-08, "fallback": verdict == "fallback"}, f)
        fb = verdict == "fallback"
        print("R=%d %s" % (r, b_solver(r, 0 if fb else 1, 30 + r, int(fb), rel="5.000000e-09" if fb else "1.000000e-12")))
        print("R=%d STAGE medium=b status=%s grid=%dx%dx%d solver_converged=%d solver_usable=1" % (r, verdict, nx, ny, nz, 0 if fb else 1))
        print("R=%d STATUS ok" % r)
        print("TIMING r=%d seconds=%.3f" % (r, 10.0 + r))
    print("RUN_DONE nr=%d" % nr)
    return 0


def par2(argv):
    sys.stdout.reconfigure(encoding="utf-8")  # the real pipeline prints a UTF-8 arrow in the residual line
    with open(argv[0]) as f:
        cfg = miniyaml.load(f.read())
    errs = miniyaml.validate_par2(cfg)
    if errs:
        sys.stderr.write("Config validation failed: %s\n" % errs)
        return 1
    NR = cfg["analysis"]["macrodispersion"]["NR"]
    planes = cfg["analysis"]["plane_crossings"]["planes"]
    ell, x0 = cfg["analysis"]["macrodispersion"]["lambda"], cfg["transport"]["inject_x"]
    cross = os.path.join(cfg["output"]["output_dir"], "crossings")
    os.makedirs(cross, exist_ok=True)
    for r in range(NR):
        print("  Realization %d / %d" % (r + 1, NR))
        inject_error(r)
        sf18 = cfg["stochastic"].get("field_source") == "sf18_medium_b"
        fseed = (cfg["stochastic"]["sf18_medium_b"]["seed0"] if sf18 else cfg["stochastic"]["seed"]) + r
        if sf18:
            print("  [4] SF-18 medium (b) field (seed=%d, sigma2=1.0000e+00, corr_length=4.1667e-02, eps=5.0000e-01); K=exp(eps*Y)" % fseed)
        else:
            print("  [4] K generation (seed=%d)" % fseed)
        fb = env_int("FAKE_FALLBACK_AT") == r or seed_hit("FAKE_FALLBACK_SEED", fseed)
        ok = env_int("FAKE_NOCONV_AT") != r and not seed_hit("FAKE_NOCONV_SEED", fseed) and not fb
        final = "1.00e-12" if ok or fb else "1.00e-02"
        print("       Converged=%s  iters=%d  res=1.00e+03 \u2192 %s" % ("YES" if ok else "NO", 40 + r, final))
        if not ok:
            sys.stderr.write("       WARNING: Head solve did NOT converge (r=%d)!\n" % r)
        rows = []
        for x1 in planes:
            off = x1 - x0
            v = 2 * 0.03 * ell * off + 1.0 / 12
            rows.append([x1, 100, 99 if seed_hit("FAKE_PAR2_INCOMPLETE_SEED", fseed) else 100, off / 2.0, 0.02 * off, 0.5, 0.5, v, v, 0.0, 0.0, 0.0, v - 1.0 / 12, v - 1.0 / 12, 0.0])
        synth.write_par2(os.path.join(cross, "realization_%04d_planes.csv" % r), rows)
        if cfg["analysis"].get("snapshots", {}).get("enabled"):
            sd = os.path.join(cfg["output"]["output_dir"], "snapshots", "r_%04d" % r)
            os.makedirs(sd, exist_ok=True)
            with open(os.path.join(sd, "step_00000100.csv"), "w") as f:
                f.write("x coord,y coord,z coord\n1.0,0.5,0.5\n")
        print("       Transport complete (r=%d)." % r)
    print("Ensemble finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit({"consumer": consumer, "par2": par2}[sys.argv[1]](sys.argv[2:]))
