"""Fake `lagrangian_ensemble` / `macroflow3d_pipeline` for the executor tests.

    python3 fake_runner.py consumer <lagrangian_ensemble args>      (cwd irrelevant; --out given)
    python3 fake_runner.py par2 config.yaml                         (cwd = segment directory, output_dir relative)

They parse the real command line / YAML, apply the usage rules of the real binaries that matter here (exit 2 usage, planes strictly
increasing, > x0, < L1; Par2: the validator rules of ``miniyaml.validate_par2``) and write minimal outputs with the SF-42 synthetic
generators (``sf42red.tests.synth``), plus the log lines the executor parses.  Failure injection by environment variables:

  FAKE_FAIL4_AT=r      consumer: STAGE_FAILED (exit 4) at realization index r
  FAKE_FALLBACK_AT=r   consumer: solver verdict 'fallback' (accepted) at r
  FAKE_ERR_AT=r        consumer/Par2: print FAKE_ERR_TEXT on stderr and exit FAKE_ERR_EXIT (default 1) before realization r
  FAKE_NOCONV_AT=r     Par2: realization r prints Converged=NO (exit stays 0)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "2026-10-30-sf42-confirmatory-tooling", "scripts"))
sys.path.insert(0, HERE)

from sf42red.tests import synth  # noqa: E402

import miniyaml  # noqa: E402


def env_int(name):
    v = os.environ.get(name)
    return None if v is None else int(v)


def inject_error(r):
    if env_int("FAKE_ERR_AT") == r:
        sys.stderr.write(os.environ.get("FAKE_ERR_TEXT", "ERROR: something") + "\n")
        sys.exit(int(os.environ.get("FAKE_ERR_EXIT", "1")))


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
        if env_int("FAKE_FAIL4_AT") == r:
            print("R=%d B_SOLVER r=%d converged=0 iterations=4000 fallback=1" % (r, r))
            print("R=%d STAGE medium=b status=not_converged_beyond_fallback grid=%dx%dx%d" % (r, nx, ny, nz))
            print("R=%d STAGE_FAILED status=solver_failed eps=%s verdict=not_converged_beyond_fallback: the medium (b) solve is not usable"
                  % (r, val["--eps"]))
            return 4
        verdict = "fallback" if env_int("FAKE_FALLBACK_AT") == r else "converged"
        for d in hm:
            synth.write_pass(out, M=M, ell=ell, x0=x0, d_planes=d_planes, r=r, hmax=d if len(hm) > 1 else None, seed=seed,
                             a_T=0.03 * (1.0 + 0.1 * (seed % 7)))
        rdir = os.path.join(out, "realizations", "r_%04d" % r)
        with open(os.path.join(rdir, "solver_b.json"), "w") as f:
            json.dump({"r": r, "seed": seed, "verdict": verdict, "usable": True, "converged": verdict == "converged"}, f)
        print("R=%d B_SOLVER r=%d converged=1 iterations=30 fallback=%d" % (r, r, verdict == "fallback"))
        print("R=%d STAGE medium=b status=%s grid=%dx%dx%d" % (r, verdict, nx, ny, nz))
        print("R=%d STATUS ok" % r)
        print("TIMING r=%d seconds=%.3f" % (r, 10.0 + r))
    print("RUN_DONE nr=%d" % nr)
    return 0


def par2(argv):
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
        ok = env_int("FAKE_NOCONV_AT") != r
        print("       Converged=%s  iters=%d  res=1.00e+03 -> 1.00e-12" % ("YES" if ok else "NO", 40 + r))
        if not ok:
            sys.stderr.write("       WARNING: Head solve did NOT converge (r=%d)!\n" % r)
        rows = []
        for x1 in planes:
            off = x1 - x0
            v = 2 * 0.03 * ell * off + 1.0 / 12
            rows.append([x1, 100, 100, off / 2.0, 0.02 * off, 0.5, 0.5, v, v, 0.0, 0.0, 0.0, v - 1.0 / 12, v - 1.0 / 12, 0.0])
        synth.write_par2(os.path.join(cross, "realization_%04d_planes.csv" % r), rows)
        print("       Transport complete (r=%d)." % r)
    print("Ensemble finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit({"consumer": consumer, "par2": par2}[sys.argv[1]](sys.argv[2:]))
