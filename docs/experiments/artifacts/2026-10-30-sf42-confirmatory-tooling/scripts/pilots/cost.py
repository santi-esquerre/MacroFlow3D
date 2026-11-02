"""Cost-model calibration with the R4 pilot walls and the SF-43 planning projection (decision G10 budget, about 150 GPU-h).

The pilot points are added to the SF-39 fit (``mcstat.costmodel.CostModel.with_points``): consumer runs under ``lagrangian`` with their
real ``M^2`` and pass work (one pass, h/8), Par2 under its own code tag ``par2_rtol1e-10`` (``cg_rtol`` 1e-10 and plane crossings; fitted
from its own points only).  A point's wall is the process wall of the run divided by its realizations (start-up included, as SF-39's
job walls).  The projection table is a PLANNING table (not a commitment): ``plan(...)`` below.
"""
import json
import os

from mcstat import costmodel as cm

PAR2_TAG = "par2_rtol1e-10"                  # Par2 at cg_rtol 1e-10 with SF-39's horizon rule (factor 1.2)
PAR2_H36_TAG = "par2_rtol1e-10_h3.6"         # same with horizon factor 3.6 (J3B): the cost of the horizon SF-43 plane estimators need
PLANES = 70
AMPS = ("0.25", "1", "2.25", "4")
SIG2 = {"0.25": 0.25, "1": 1.0, "2.25": 2.25, "4": 4.0}


def realization_seconds(r):
    """Sum over the run's realizations of the code's own per-realization time (no process start-up, no post-processing): consumer =
    ``timings.json`` ``seconds.total``; Par2 = stage times [4] -> 'Transport complete' (field + solve + velocity + transport).
    SF-39's walls spread one start-up over >= 20 realizations; for pilot runs of 1-8 realizations this basis is the comparable one."""
    if r["code"] == "consumer":
        return sum(x["seconds"]["total"] for x in r["realizations"])
    return sum(x["flow_stage_s"] + x["transport_s"] for x in r["par2_stage_times"])


def points_from_runs(root, jobs=("J1", "J2", "J3", "J3B"), basis="realization"):
    """Calibration points from ``<root>/<job>/<id>/RUN.json`` (exit 0 only); ``basis``: 'realization' (default, see
    ``realization_seconds``) or 'wall' (process wall, start-up included)."""
    pts = []
    for job in jobs:
        jd = os.path.join(root, job)
        if not os.path.isdir(jd):
            continue
        for name in sorted(os.listdir(jd)):
            p = os.path.join(jd, name, "RUN.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                r = json.load(f)
            if r["exit"] != 0:
                continue
            s2 = SIG2[r["amp"]]
            label = "%s/%s" % (job, r["id"])      # run ids repeat across jobs (J1 and J3 share grids): the label must be unique
            if r["code"] == "consumer":
                arm = cm.lagrangian_arm(r["nx"], r["ny"], r["nz"], s2, float(r["M"]) ** 2, (8,), PLANES, label)
            else:
                tag = PAR2_H36_TAG if r.get("par2_horizon_factor", 1.2) > 2.0 else PAR2_TAG
                arm = cm.par2_arm(r["nx"], r["ny"], r["nz"], s2, 1e4, tag, label)
            w = realization_seconds(r) if basis == "realization" and ("realizations" in r or "par2_stage_times" in r) else r["wall_s"]
            pts.append(cm.Point(arm, r["nr"], w, "%s %s %s" % (job, r["id"], basis)))
    return pts


def plan(M, g5_n, v7=480, scale=None, par2_tag=PAR2_H36_TAG):
    """The SF-43 planning table: list of dicts (family, arm label, arm, n).

    ``M``: {amp: lattice size}; ``g5_n``: {amp: Gate 5 realizations (both codes)}; ``scale``: optional function (family, amp, n) -> n
    (variant B); ``par2_tag``: cost code of the planned Par2 rows (default: the 3.6 horizon SF-43 plane estimators need)."""
    rows = []

    def lag(fam, amp, nx, ny, n, hmax=(8,)):
        nn = scale(fam, amp, n) if scale else n
        rows.append({"family": fam, "amp": amp, "n": nn, "arm": cm.lagrangian_arm(nx, ny, ny, SIG2[amp], float(M[amp]) ** 2, hmax, PLANES,
                                                                                  "%s %s %dx%d" % (fam, amp, nx, ny))})

    def par2(fam, amp, nx, ny, n):
        rows.append({"family": fam, "amp": amp, "n": n, "arm": cm.par2_arm(nx, ny, ny, SIG2[amp], 1e4, par2_tag,
                                                                           "%s %s %dx%d" % (fam, amp, nx, ny))})

    # production (h/8)
    lag("production", "0.25", 1040, 240, 1100)
    lag("production", "1", 1040, 240, 1650)
    lag("production", "2.25", 1664, 384, 110)
    lag("production", "4", 3328, 384, 30)
    # E_h non-production rungs
    for amp in ("0.25", "1"):
        for nx, ny in ((520, 120), (2080, 480)):
            lag("E_h", amp, nx, ny, 300)
    for nx, ny in ((832, 192), (2080, 480)):
        lag("E_h", "2.25", nx, ny, 110)
    for nx, ny in ((832, 96), (1664, 192)):
        lag("E_h", "4", nx, ny, 30)
    # E_W
    lag("E_W", "0.25", 1040, 480, 1100)
    lag("E_W", "1", 1040, 480, 1650)
    for nx, ny in ((832, 192), (832, 384), (1040, 240), (1040, 480)):
        lag("E_W", "2.25", nx, ny, 110)
    for nx, ny in ((832, 96), (832, 192), (1664, 192), (1664, 384)):
        lag("E_W", "4", nx, ny, 30)
    # h_max passes 8 / 16 / 32 (counted as full three-pass realizations: conservative, they are a subset of production in SF-39)
    for amp, nx, ny, n in (("0.25", 1040, 240, 50), ("1", 1040, 240, 50), ("2.25", 1664, 384, 50), ("4", 3328, 384, 30)):
        rows.append({"family": "hmax", "amp": amp, "n": n, "arm": cm.lagrangian_arm(nx, ny, ny, SIG2[amp], float(M[amp]) ** 2, (8, 16, 32), PLANES,
                                                                                  "hmax %s %dx%d" % (amp, nx, ny))})
    # Gate 5, both codes
    for amp, nx, ny in (("0.25", 1040, 240), ("1", 1040, 240), ("2.25", 1248, 288), ("4", 1664, 192)):
        lag("gate5_consumer", amp, nx, ny, g5_n[amp])
        par2("gate5_par2", amp, nx, ny, g5_n[amp])
    # G7 factorial (Par2)
    for amp in ("0.25", "1"):
        par2("g7_par2", amp, 1040, 240, v7)
    # distance arms (208 lambda)
    lag("distance", "1", 2080, 240, 100)
    lag("distance", "2.25", 3328, 384, 100)
    return rows


def project(model, rows, reserve=0.05):
    """GPU-h per row, per family, total (with the reserve fraction), out-of-range arms, GPU-h per 100 realizations of each arm."""
    res = model.project([(r["arm"], r["n"]) for r in rows])
    out_rows, fam = [], {}
    for r, p in zip(rows, res["rows"]):
        out_rows.append({"family": r["family"], "amp": r["amp"], "label": r["arm"].label, "code": r["arm"].code, "n": r["n"],
                         "gpu_s_per_real": p["gpu_s_per_real"], "gpu_h": p["gpu_h"], "gpu_h_per_100": 100.0 * p["gpu_s_per_real"] / 3600.0,
                         "in_range": model.in_range(r["arm"])})
        fam[r["family"]] = fam.get(r["family"], 0.0) + p["gpu_h"]
    base = res["gpu_h"]
    return {"rows": out_rows, "by_family": fam, "subtotal_gpu_h": base, "reserve_gpu_h": reserve * base,
            "total_gpu_h": base * (1.0 + reserve), "out_of_range": res["out_of_range"]}
