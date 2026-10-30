"""Pre-submission GPU-h projection of the pilot jobs with the SF-39-only cost model (written BEFORE each job is submitted).

    python3 -m pilots.presubmit <repo root> [--out json]

Per run: predicted GPU-s per realization x realizations; consumer runs with the ``lagrangian`` fit (real M^2, one pass), Par2 runs with
SF-39's ``par2`` fit (cells only; SF-39 used ``cg_rtol`` 1e-6, the pilots 1e-10, so this under-predicts the Par2 flow stage:
flagged ``approx``).  ``in_range`` False = outside the SF-39 calibration cells.
"""
import argparse
import json
import sys

from mcstat import costmodel as cm

from . import cost, design


def arm_of(r):
    s2 = cost.SIG2[r.amp]
    if r.code == "consumer":
        return cm.lagrangian_arm(r.nx, r.ny, r.ny, s2, float(r.M) ** 2, (8,), cost.PLANES, r.id)
    return cm.par2_arm(r.nx, r.ny, r.ny, s2, 1e4, "par2", r.id)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    model, _ = cm.fit_sf39(a.repo)
    res = {}
    for job in ("J1", "J2", "J3"):
        rows, tot = [], 0.0
        for r in design.JOBS[job]():
            arm = arm_of(r)
            t = model.predict(arm)
            rows.append({"id": r.id, "code": r.code, "n": r.nr, "M": r.M, "gpu_s_per_real": t, "gpu_h": r.nr * t / 3600.0,
                         "in_range": model.in_range(arm), "approx": r.code == "par2"})
            tot += r.nr * t / 3600.0
        res[job] = {"rows": rows, "gpu_h": tot}
        print("%s projected %.3f GPU-h (%d runs)" % (job, tot, len(rows)))
        for x in rows:
            print("   %-22s n=%d  %.1f s/real  %.3f GPU-h  %s%s" % (x["id"], x["n"], x["gpu_s_per_real"], x["gpu_h"],
                                                                  "" if x["in_range"] else "OUT_OF_RANGE ", "approx" if x["approx"] else ""))
    res["J1_J2_J3_gpu_h"] = sum(res[j]["gpu_h"] for j in ("J1", "J2", "J3"))
    print("J1+J2+J3 projected %.3f GPU-h (budget rule: cut J2/J3 if above 8)" % res["J1_J2_J3_gpu_h"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
