"""Assembly of the SF-43 plan, pre-registration, seed registry and design table from ``design.py`` (stdlib only, deterministic).

Only ``engine_map`` knows the key names of the campaign engine for the fields added by the engine node; everything else here is the
schema of ``sf43camp/plan.py`` / ``sf43ana/config.py`` at the base.
"""
import hashlib
import json
import math
import os
from collections import OrderedDict, namedtuple
from decimal import Decimal
from fractions import Fraction

from . import design as D
from . import engine_map as EM

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)                                        # <A>/scripts
ART = os.path.dirname(SCRIPTS)                                         # <A>
REPO = os.path.normpath(os.path.join(ART, "..", "..", "..", ".."))
ART_REL = os.path.relpath(ART, REPO).replace(os.sep, "/")
SF42_REL = "docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts"
PLACEHOLDER_MD5 = "0" * 32


# ---------------------------------------------------------------------------------------------------------------------------
# geometry and per-arm derived values
# ---------------------------------------------------------------------------------------------------------------------------
def window_string(W, S):
    """Centred source window (W - S)/(2W) .. (W + S)/(2W) as exact short decimals on both axes (as ``seedlattice.plan``)."""
    W, S = Decimal(str(W)), Decimal(str(S))
    out = []
    for num in (W - S, W + S):
        s = format(num / (2 * W), "f")
        s = s.rstrip("0").rstrip(".") if "." in s else s
        if Fraction(Decimal(s)) != Fraction(num) / Fraction(2 * W):
            raise ValueError("window fraction is not an exact short decimal: (%s)/(2 %s)" % (num, W))
        out.append(s)
    return "%s,%s,%s,%s" % (out[0], out[1], out[0], out[1])


def grid(r):
    return [r.L * r.lh, r.W * r.lh, r.W * r.lh]


def ell(r):
    return r.lh / (r.W * r.lh)                                          # = lh / ny, the float of the SF-42 pilots


def planes_d(r):
    return list(range(D.D_LO, (D.D_FAR if "dist" in r.flags else D.D_HI) + 1))


def amp_info(r):
    return D.AMPS[r.amp]


def est_seconds(r):
    return round(D.SEC[r.sec], 1)


def peak_mib(r):
    nx, ny, _ = grid(r)
    k = (r.run, nx, ny)
    return D.PEAK_MIB.get(k) or D.PEAK_EST[k]


def min_free_mib(r):
    if r.mem:
        return D.MEM_FREE[r.mem]
    return int(math.ceil((1.1 * peak_mib(r) + 500.0) / 500.0)) * 500   # measured own-PID peak + 10 % + 500 MiB, rounded up to 500


def is_heavy(r):
    return r.mem is not None


def raw_gb(r):
    if r.run == "p":
        return 0.065 if "snap" in r.flags else 0.0001                  # 100 snapshots of 10000 particles (about 60 MB) / crossings only
    M = amp_info(r)[3]
    return round(D.PLANES_RAW_GB_M128 * (M / 128.0) ** 2 * (len(planes_d(r)) / 70.0), 3)


def unit_size(r):
    if r.run == "p":
        return 20 if r.n % 20 == 0 else 10                              # Par2: 10-20 per unit, dividing n
    return 10 if D.SEC[r.sec] <= 60.0 else 5                           # production arms: 5 or 10, both divide 30


def n_steps_par2(r):
    nx, ny, _ = grid(r)
    dx = 1.0 / ny
    dt = 0.0244140625 * dx
    return int(math.ceil(3.6 * (nx * dx - 10.0 * ell(r)) / dt))


def consumer_dict(r):
    sigma2, label, eps, M, S = amp_info(r)
    return OrderedDict([("grid", grid(r)), ("eps", eps), ("ell", ell(r)), ("window", window_string(r.W, S)), ("np_side", M),
                        ("planes_d", planes_d(r)), ("x0_ell", 5), ("hmax_div", [8]), ("twin", True), ("pollock_bridge", True),
                        ("tol", "1e-8"), ("times", "auto"), ("sigma2", "1"), ("fused_splines", True), ("single_launch", True)])


def par2_dict(r, cell=None):
    sigma2, label, eps, M, S = amp_info(r)
    d = OrderedDict([("grid", grid(r)), ("eps", eps), ("ell", ell(r)), ("window", window_string(r.W, S)), ("planes_d", planes_d(r)),
                     ("x0_ell", 5), ("n_particles", 10000), ("cg_max_iter", 2000), ("cg_rtol", "1.0e-10"), ("hz", 3.6),
                     ("dt_coeff", 0.0244140625), ("dt_half", False), ("generator", "sf18"), ("inject", "window"),
                     ("transport_seed", D.PAR2_TRANSPORT_SEED)])
    if "snap" in r.flags:
        d["snapshots_every"] = 1000 * int(math.ceil(n_steps_par2(r) / 100.0 / 1000.0))   # about 100 snapshots; a multiple of sample_every
    if cell:
        g, i, dd, rr = cell
        d["generator"] = "sf18" if g > 0 else "random_spectral"
        if g < 0:
            d["n_modes"] = D.G7_RS_MODES
            d["sigma2"] = sigma2
        d["inject"] = "window" if i > 0 else "full"
        d["dt_half"] = dd > 0
        d["cg_rtol"] = "1.0e-10" if rr > 0 else "1.0e-6"
    return d


def reduce_dict(r):
    sigma2, label, eps, M, S = amp_info(r)
    if r.run == "p":
        d = OrderedDict([("code", "par2"), ("d_windows", [[D.D_LO, D.D_HI]]), ("blind", {"weights": "unif", "basis": "disp", "set": "perplane"})])
        if "snap" in r.flags:
            d["U_time"] = round(math.exp(sigma2 / 6.0), 6)               # first order K_eff = exp(sigma^2 / 6); the consumer's qbar agrees to 0.7 %
            d["par2_exit_x"] = r.L * r.lh / (r.W * r.lh)                 # = L1, the consumer's outlet face (first exit)
        return d
    wins = D.WINDOWS if "primary" in r.flags else D.WINDOWS_DIST if "dist" in r.flags else [[D.D_LO, D.D_HI]]
    return OrderedDict([("code", "consumer"), ("d_windows", wins),
                        ("blind", {"weights": "flux", "basis": "disp", "set": "perplane"})])


# ---------------------------------------------------------------------------------------------------------------------------
# entries: the arms of one universe (the production plan or the smoke plan) with their group, seed block and sizes
# ---------------------------------------------------------------------------------------------------------------------------
Entry = namedtuple("Entry", "id row n nmax grp prio start lane heavy hm_n")


def g7_cells():
    out = []
    for g in (1, -1):
        for i in (-1, 1):
            for d in (-1, 1):
                for r in (-1, 1):
                    if g * i * d * r == 1:
                        out.append((g, i, d, r))
    return out


def g7_id(amp, cell):
    g, i, d, r = cell
    return "g7_%s_%s%s%s%s" % (amp, "S" if g > 0 else "R", "W" if i > 0 else "F", "H" if d > 0 else "N", "T" if r > 0 else "L")


def g7_rows(amps=D.G7_AMPS, group=lambda amp, c: "g7_%s_%s" % (amp, "S" if c[0] > 0 else "R")):
    rows, cells = [], OrderedDict()
    for amp in amps:
        for c in g7_cells():
            rid = g7_id(amp, c)
            rows.append(D.row(rid, amp, "g7_par2", "p", 24, 10, 104, D.G7_SEEDS, group(amp, c), 3, ("g7_cell_mean", D.AMPS[amp][1]), flags=("g7",)))
            cells[rid] = c
    return rows, cells


G7_ROWS, G7_CELLS = g7_rows()
ALL_ROWS = list(D.ROWS) + G7_ROWS


def production_entries():
    return [Entry(r.id, r, r.n, r.nmax, r.grp, r.prio, D.SEED_START[r.grp], None, is_heavy(r), D.HMAX_FIRST["n"] if "hm3" in r.flags else None)
            for r in ALL_ROWS]


def smoke_entries():
    """One smoke arm per distinct (amplitude, grid, runner) of the production plan, the eight G7 cells at 0.25 (all code paths of the
    factorial) and the determinism pairs (the cheapest consumer grid and a Par2 grid, each twice, lanes A and B)."""
    S, out, seen = D.SMOKE, [], set()
    for r in D.ROWS:
        key = (r.amp, tuple(grid(r)), r.run)
        if key in seen:
            continue
        seen.add(key)
        n = S["n"] if ("primary" in r.flags or est_seconds(r) <= S["slow_s"]) else S["n_slow"]
        nmax = n
        if r.id == S["reest_arm"]:
            n, nmax = S["reest_n_conf"], S["reest_n_max"]
        g = "smoke_" + r.amp
        out.append(Entry("sm_" + r.id, r, n, nmax, g, 1, D.SMOKE_START[g], None, is_heavy(r), S["hmax_first_n"] if "hm3" in r.flags else None))
    rows, _ = g7_rows(("s025",), lambda amp, c: "smoke_g7_" + ("S" if c[0] > 0 else "R"))
    for r in rows:
        out.append(Entry("sm_" + r.id, r, S["n"], S["n"], r.grp, 1, D.SMOKE_START[r.grp], None, False, None))
    amp, nx, ny, lh, W = S["det_consumer"]
    rc = D.row("det_c", amp, "determinism", "c", W, lh, 104, S["n"], "smoke_detc", 0, ("416x48", "4"), flags=("det",))
    amp, nx, ny, lh, W = S["det_par2"]
    rp = D.row("det_p", amp, "determinism", "p", W, lh, 104, S["n"], "smoke_detp", 0, ("par2_1040x240", "0.25"), flags=("det",))
    for r in (rc, rp):
        for lane in ("A", "B"):   # not heavy: the engine's lane pin alone puts the member on its lane (a pinned light arm may run on lane B)
            out.append(Entry("sm_%s_%s" % (r.id, lane), r, S["n"], S["n"], r.grp, 0, D.SMOKE_START[r.grp], lane, False, None))
    return out


def group_blocks(entries, frac_of):
    """OrderedDict group -> {start, count, reserves {start, count}, arms}: count = largest n_max of the group, reserves follow the block."""
    out = OrderedDict()
    for e in entries:
        b = out.setdefault(e.grp, {"start": e.start, "count": 0, "arms": []})
        if b["start"] != e.start:
            raise ValueError("group %s: arms with different seed starts" % e.grp)
        b["count"] = max(b["count"], e.nmax)
        b["arms"].append(e.id)
    for g, b in out.items():
        b["reserves"] = {"start": b["start"] + b["count"], "count": max(3, int(math.ceil(frac_of(g) * b["count"])))}
    return out


def reserve_frac(grp):
    return D.RESERVE_FRAC["s4"] if grp.endswith("s4") else D.RESERVE_FRAC["default"]


def retained_seeds(r, start, n):
    if r.run == "p":
        return []
    idx = set(range(min(D.RETAIN_FIRST, n)))
    idx |= {min(n - 1, max(0, int(round(p * n)) - 1)) for p in D.REVIEW_POSITIONS}
    return [start + k for k in sorted(idx)]


# ---------------------------------------------------------------------------------------------------------------------------
# blinded re-estimation (G5.2 / H9): half-width targets -> the planning SD that makes N_req = n_conf
# ---------------------------------------------------------------------------------------------------------------------------
def blinded_section(ents, hw_scale=1.0, fraction=None):
    """H9 / G5.2: per arm the planning half-width (spec item 6, two-sided 90 %, alpha_TP and alpha_TAP) the engine turns into N_req at the
    blinded SD's 80 % upper bound.  ``hw_scale`` < 1 (smoke) makes N_req exceed n_conf so that the re-estimation path runs."""
    arms = [e for e in ents if "reest" in e.row.flags]
    hw = OrderedDict((e.id, OrderedDict((k, float("%.6g" % (hw_scale * D.TARGETS[e.row.amp]))) for k in ("alpha_TP", "alpha_TAP"))) for e in arms)
    return OrderedDict([("fraction", D.BLINDED["fraction"] if fraction is None else fraction), ("max_factor", D.BLINDED["max_factor"]),
                        ("arms", [e.id for e in arms]), ("hw_target", hw), ("level", D.BLINDED["level"]),
                        ("note", "hw_target = the spec's 90 % half-width target (item 6) of alpha_TP / alpha_TAP; N_req = ceil((t_0.95 SD_UB80 / hw)^2); "
                                 "inert unless the SD is several times the planning value (H9)")])


# ---------------------------------------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------------------------------------
def plan_arm(e, blocks):
    r, b = e.row, blocks[e.grp]
    a = OrderedDict([("id", e.id), ("family", r.fam), ("amplitude", amp_info(r)[0]), ("runner", "consumer" if r.run == "c" else "par2"),
                     ("priority", e.prio), ("heavy", e.heavy), ("min_free_mib", min_free_mib(r)), ("group", e.grp),
                     ("seeds", OrderedDict([("start", b["start"]), ("count", e.nmax)])), ("n_conf", e.n), ("n_max", e.nmax),
                     ("unit_size", unit_size(r)), ("est_seconds", est_seconds(r)), ("raw_gb", raw_gb(r))])
    if r.run == "c":
        a["consumer"] = consumer_dict(r)
        if e.hm_n:                                   # the first n positions run --hmax-div 8,16,32 (engine key: engine_map.consumer_fields)
            a["consumer"].update(EM.consumer_fields(e))
    else:
        a["par2"] = par2_dict(r, G7_CELLS.get(r.id))
    a["reduce"] = reduce_dict(r)
    a["retained"] = retained_seeds(r, b["start"], e.n)
    a.update(EM.arm_fields(e))
    if e.hm_n:                                       # planning extras (ignored by the engine)
        a["hmax_extra"] = OrderedDict([("est_seconds_extra", round(D.H3_EXTRA[r.amp], 1)), ("raw_gb_factor", D.H3_RAW_FACTOR)])
    return a


ANALYSE_SH = ART_REL + "/scripts/analyse_sf43.sh"
COV_SUMMARY_REL = ART_REL + "/coverage/summary.json"
COV_QUAL_REL = ART_REL + "/coverage/qualification.json"
ZERO_SHA256 = "0" * 64


def analysis_section(hashes, smoke=False):
    """Engine ``analysis``: the wrapper (``analyse_sf43.sh``) and the files it was frozen with; the smoke overlay passes ``--smoke``."""
    cmd = ["bash", ANALYSE_SH, "{campaign_dir}", "{build_dir}"] + (["--smoke"] if smoke else [])
    return OrderedDict([("command", cmd), ("sha256_list", hashes)])


def recompute_section(ents, campaign_dir_key="{campaign_dir}"):
    """``recompute43`` command: every primary arm must have its plane arrays (H21); single-pass roots of the three-pass arms carry hmax_div 8."""
    prim = [e.id for e in ents if "primary" in e.row.flags]
    cmd = ["python3", "-W", "error", "-m", "recompute43", "run", "--campaign", "{campaign_dir}", "--reduced", "{campaign_dir}/reduced", "--out",
           "{campaign_dir}.recompute.txt"]
    for p in prim:
        cmd += ["--primary-arm", p]
    return {"command": cmd + ["--single-pass-hmax", str(D.HMAX_FIRST["divs"][0])]}


def build_plan(binaries, campaign_dir, hashes, smoke_hashes):
    ents = production_entries()
    blocks = group_blocks(ents, reserve_frac)
    plan = OrderedDict()
    plan["campaign"] = D.CAMPAIGN
    plan["schema_version"] = D.SCHEMA_VERSION
    plan["cutoff_utc"] = D.CUTOFF_UTC
    plan["grace_minutes"] = D.GRACE_MINUTES
    plan["stage1_max_hours"] = D.STAGE1_MAX_HOURS
    plan["budget_gpu_h"] = D.BUDGET_GPU_H
    plan["disk_min_free_gb"] = D.DISK_MIN_FREE_GB
    plan["campaign_dir"] = campaign_dir
    plan.update(EM.top_level())
    plan["lane_b_min_free_mib"] = D.MEM_FREE["half"]
    plan["binaries"] = OrderedDict(sorted(binaries.items()))
    if all(v == PLACEHOLDER_MD5 for v in binaries.values()):
        plan["binaries_note"] = "PLACEHOLDER md5 (all zeros): the md5 gate refuses to start until the generator is re-run with the real binaries.json"
    if ZERO_SHA256 in hashes.values():
        plan["coverage_note"] = ("SMOKE-ONLY plan: the sha256 of the coverage summary / qualification are all-zero placeholders, so the analysis gate "
                                 "refuses a production run; regenerate with --coverage-summary / --coverage-qualification (the smoke job's outputs, "
                                 "committed under coverage/)")
    plan["analysis"] = analysis_section(hashes)
    plan["recompute"] = recompute_section(ents)
    plan["reconcile"] = {"command": ["python3", "-W", "error", "-m", "reconcile43", "run", "--campaign", "{campaign_dir}", "--out",
                                     "{campaign_dir}.reconcile.txt"]}
    plan["blinded_reestimation"] = blinded_section(ents)
    plan["failure"] = dict(D.FAILURE)
    plan["groups"] = OrderedDict((g, {"reserves": b["reserves"]}) for g, b in blocks.items())
    plan["arms"] = [plan_arm(e, blocks) for e in ents]
    plan["smoke"] = build_smoke(campaign_dir, smoke_hashes)
    return plan, blocks


def build_smoke(campaign_dir, smoke_hashes):
    ents = smoke_entries()
    blocks = group_blocks(ents, lambda g: 0.15)
    arms = [plan_arm(e, blocks) for e in ents]
    for a, e in zip(arms, ents):
        a["unit_size"] = D.SMOKE["unit"] if e.n % D.SMOKE["unit"] == 0 else 1
        a["retained"] = [blocks[e.grp]["start"]] if e.row.run == "c" else []
    total_h = sum(e.n * est_seconds(e.row) for e in ents) / 3600.0 + sum(
        (e.hm_n or 0) * D.H3_EXTRA.get(e.row.amp, 0.0) for e in ents) / 3600.0
    red = [e for e in ents if e.row.id == D.SMOKE["reest_arm"]]
    sm = OrderedDict()
    sm["campaign_dir"] = campaign_dir.rstrip("/") + D.SMOKE["campaign_suffix"]
    sm["cutoff_minutes"] = D.SMOKE["cutoff_minutes"]
    sm["budget_gpu_h"] = max(D.SMOKE["budget_min"], int(math.ceil(D.SMOKE["budget_factor"] * total_h)))
    sm["stage1_max_hours"] = D.SMOKE["stage1_max_hours"]
    sm["groups"] = OrderedDict((g, {"reserves": b["reserves"]}) for g, b in blocks.items())
    sm["blinded_reestimation"] = blinded_section(red, hw_scale=D.SMOKE["reest_hw_scale"], fraction=D.SMOKE["reest_fraction"])
    sm["analysis"] = analysis_section(smoke_hashes, smoke=True)          # --smoke: prereg_smoke.json, outputs to files, only check-finite on stdout
    sm["recompute"] = recompute_section(ents)
    sm["arms"] = arms
    sm.update(EM.smoke_fields([("det_consumer", "sm_det_c_A", "sm_det_c_B"), ("det_par2", "sm_det_p_A", "sm_det_p_B")], campaign_dir,
                              D.SMOKE["background_wait_minutes"], ART_REL))
    sm["_smoke_work_gpu_h_estimate"] = round(total_h, 2)
    return sm


# ---------------------------------------------------------------------------------------------------------------------------
# pre-registration (one builder for the production universe and the smoke universe)
# ---------------------------------------------------------------------------------------------------------------------------
def prereg_arm(aid, amp, grp, start, n_seeds, reserves, n_conf, floor, source=None, hmax=-1, code="consumer", role="", weights=None, n_max=None,
               reest=False):
    """One analysis entry.  ``seeds`` = the first ``n_max`` seeds of the group's block (what the engine may run), ``n_conf`` = the PLANNED size:
    the loader cuts at ``n_conf``, or at the engine's ``n_new`` (ledger ``reest``) for a ``reest`` entry."""
    n_max = n_conf if n_max is None else n_max
    a = OrderedDict([("id", aid), ("amplitude", amp), ("group", grp), ("seeds", list(range(start, start + n_max))),
                     ("reserves", list(range(reserves["start"], reserves["start"] + reserves["count"]))), ("n_conf", n_conf), ("n_max", n_max),
                     ("floor", floor)])
    if source:
        a["source"] = source
    a["hmax_div"] = hmax
    a["code"] = code
    a["role"] = role
    if weights:
        a["selection"] = {"weights": weights}
    if reest:
        a["reest"] = True
    return a


FLOOR = {"production": 100, "E_h": 30, "E_W": 30, "gate5_consumer": 30, "distance": 30, "g7_par2": 30, "positive_control": 20}
G5C = OrderedDict([("s025", "s025_prod"), ("s1", "s1_prod"), ("s225", "s225_c12"), ("s4", "s4_l16")])    # engine arm that is the consumer twin
B_PROD = OrderedDict([("B_sampling", 33599), ("B_budget", 4000), ("B_ucl", 4000), ("B_bca", 5000), ("seed", 20261103)])  # = sf43ana.config.DEFAULTS
B_SMOKE = OrderedDict([("B_sampling", 3999), ("B_budget", 400), ("B_ucl", 400), ("B_bca", 200), ("seed", 20261103)])
POSITIVE_CONTROL_EXPECTATION = (
    "R = (mean over the window planes [25, 94] of the measured FLUX-weighted displacement-basis second moment X22(d)) / (the same mean of the "
    "first-order box prediction at sigma^2 = 0.04, W = 24 lambda, L = 104 lambda), with a 90 % seed-bootstrap interval.  EXPECTED: the interval "
    "inside [0.8, 1.25] (positive_control_ok); inconclusive if it straddles a bound; failed if entirely outside.  Rationale: the prediction is "
    "first order in sigma^2 and the O(sigma^2) nonlinear corrections are about 4 % at 0.04; the discretization (lambda/h = 10) and the Dirichlet "
    "inflow/outflow faces of medium (b), which the prediction neglects, are expected to move the window mean by a few per cent at most.  A value "
    "outside the bounds indicts the estimand pipeline or the runner, not the physics.  Descriptive; never a gate of the family.")


def build_prereg(smoke=False, coverage_summary_rel=COV_SUMMARY_REL, qualification_rel=COV_QUAL_REL):
    ents = smoke_entries() if smoke else production_entries()
    pre_ = "sm_" if smoke else ""
    blocks = group_blocks(ents, (lambda g: 0.15) if smoke else reserve_frac)
    by = {e.id: e for e in ents}
    has = lambda rid: pre_ + rid in by  # noqa: E731
    nm = lambda rid: pre_ + rid  # noqa: E731
    lo = 2 if smoke else None
    arms = []
    for e in ents:
        r = e.row
        if r.fam == "determinism":
            continue
        b = blocks[e.grp]
        n_conf, n_max = (max(2, e.n), max(2, e.nmax)) if smoke else (e.n, e.nmax)
        fl = lo or (r.n if r.fam == "gate5_par2" else FLOOR[r.fam])
        arms.append(prereg_arm(e.id, r.amp, e.grp, b["start"], n_max, b["reserves"], n_conf, fl, code="par2" if r.run == "p" else "consumer",
                               hmax=8 if r.fam == "production" else -1, role=r.fam, weights="unif" if r.run == "p" else None, n_max=n_max,
                               reest="reest" in r.flags))
    for amp in G5C:
        if not has(amp + "_prod"):
            continue
        e = by[nm(amp + "_prod")]
        b = blocks[e.grp]
        n = 2 if smoke else D.HMAX_FIRST["n"]
        for tag, div in (("h16", 16), ("h8", 8)):
            arms.append(prereg_arm(nm("%s_%s" % (amp, tag)), amp, e.grp, b["start"], n, b["reserves"], n, lo or 20, source=e.id, hmax=div,
                                   role="h_max_pass"))
        # Pollock contrast of acceptance B: the production analysis set (flux weights, hmax_div 8, grows with the re-estimation)
        n_conf, n_max = (max(2, e.n), max(2, e.nmax)) if smoke else (e.n, e.nmax)
        arms.append(prereg_arm(nm(amp + "_prodpol"), amp, e.grp, b["start"], n_max, b["reserves"], n_conf, lo or FLOOR["production"], source=e.id,
                               hmax=8, code="pollock", role="pollock_production", n_max=n_max, reest=True))
        # basis contrast (H2): the same production set read with uniform weights (control: displacement = position by transverse stationarity)
        arms.append(prereg_arm(nm(amp + "_produnif"), amp, e.grp, b["start"], n_max, b["reserves"], n_conf, lo or FLOOR["production"], source=e.id,
                               hmax=8, role="basis_control_uniform", weights="unif", n_max=n_max, reest=True))
    for amp, src in G5C.items():
        e = by[nm(src)]
        b = blocks[e.grp]
        n = max(2, by[nm(amp + "_g5p")].n) if smoke else by[amp + "_g5p"].n
        hm = 8 if e.row.fam == "production" else -1
        arms.append(prereg_arm(nm(amp + "_g5c"), amp, e.grp, b["start"], n, b["reserves"], n, lo or n, source=e.id, role="gate5_consumer_twin",
                               hmax=hm, weights="unif"))
        arms.append(prereg_arm(nm(amp + "_g5pol"), amp, e.grp, b["start"], n, b["reserves"], n, lo or n, source=e.id, role="gate5_pollock_twin",
                               hmax=hm, code="pollock", weights="unif"))
    amps = OrderedDict()
    for amp, (sigma2, label, eps, M, S) in D.AMPS.items():
        if amp == "s004":
            amps[amp] = OrderedDict([("sigma2", sigma2), ("control", True)])
            continue
        c = OrderedDict([("sigma2", sigma2), ("primary", True), ("production", nm(amp + "_prod"))])
        if amp in ("s025", "s1"):
            c["components"] = ["E_h", "E_W", "E_int"]
            c["ladder"] = OrderedDict([("arms", [nm(amp + "_eh5"), nm(amp + "_prod"), nm(amp + "_eh20")]), ("resolutions", [5, 10, 20]),
                                       ("production_index", 1)])
            c["width"] = OrderedDict([("wide_fine", nm(amp + "_w48")), ("prod_fine", nm(amp + "_prod"))])
        elif amp == "s225":
            c["components"] = ["E_h", "E_W", "E_WxH", "E_int"]
            c["ladder"] = OrderedDict([("arms", [nm("s225_eh4"), nm("s225_eh8"), nm("s225_prod")]), ("resolutions", [4, 8, 16]),
                                       ("production_index", 2),
                                       ("gates", [OrderedDict([("type", "extra_level"), ("triple", [0, 1, 2]), ("arm", nm("s225_c12")), ("resolution", 12)])])])
            c["width"] = OrderedDict([("wide_fine", nm("s225_w48_12")), ("prod_fine", nm("s225_c12")), ("wide_coarse", nm("s225_w48_8")),
                                      ("prod_coarse", nm("s225_eh8"))])
        else:
            c["components"] = ["E_h", "E_W", "E_WxH", "E_int"]
            c["ladder"] = OrderedDict([("arms", [nm("s4_l4"), nm("s4_l8"), nm("s4_l16"), nm("s4_prod")]), ("resolutions", [4, 8, 16, 32]),
                                       ("production_index", 3), ("triples", [[0, 1, 2], [1, 2, 3]]),
                                       ("gates", [OrderedDict([("type", "orders_agree"), ("triples", [[0, 1, 2], [1, 2, 3]])])])])
            c["width"] = OrderedDict([("wide_fine", nm("s4_w24_16")), ("prod_fine", nm("s4_l16")), ("wide_coarse", nm("s4_w24_8")),
                                      ("prod_coarse", nm("s4_l8"))])
        c["hmax"] = OrderedDict([("h16", nm(amp + "_h16")), ("h8", nm(amp + "_h8"))])
        if has(amp + "_dist"):
            c["distance_arm"] = nm(amp + "_dist")
        amps[amp] = c
    pre = _prereg_common(amps, arms, smoke)
    if smoke:
        pre["name"] = D.CAMPAIGN + "-smoke"
        pre["note"] = ("smoke: arms of the smoke plan (sm_ prefix); every floor 2 so that every analysis code path runs; arms with one realization "
                       "exercise the below_floor paths; B reduced; Gate 5 reads the smoke job's own coverage output (analyse_sf43.sh --smoke)")
        pre["floors"] = OrderedDict((k, 2) for k in ("production", "ladder", "width", "hmax", "gate5"))
    if not smoke:        # the smoke run reads its own qualification output (analyse_sf43.sh --smoke passes --qualification when it exists)
        pre["qualification"] = qualification_rel
    pre["gate5"] = _gate5_section(nm, by, smoke, coverage_summary_rel)
    pre["g7"] = _g7_section(nm, by)
    pre["pollock"] = _pollock_section(nm)
    pre["secondaries"] = _secondaries(nm, by)
    # H17: recrossing / backflow distributions per consumer arm (the arms that are not duplicates of an engine arm's rows)
    pre["secondaries"]["recrossings"] = OrderedDict((a["id"], {"arm": a["id"]}) for a in arms
                                                    if a["code"] == "consumer" and a["role"] not in ("gate5_consumer_twin", "basis_control_uniform"))
    pre["gate4_power"] = _gate4_power_section(smoke)
    pre["basis"] = _basis_section(nm)
    pre["failures"] = _failures()
    pre["positive_control"] = _positive_control(nm("pc_s004"), smoke)
    return pre


def _prereg_common(amps, arms, smoke=False):
    pre = OrderedDict()
    pre["name"] = D.CAMPAIGN
    pre["variant"] = D.VARIANT
    # semantic window labels; the executor's table stem d%g-%g is derived from the bounds by sf43ana.loader.window_stem (one rule)
    pre["windows"] = OrderedDict([("main", [25, 94]), ("h1", [25, 59.5]), ("h2", [59.5, 94]), ("far", [94, 194])])
    pre["primary_window"] = "main"
    pre["selection"] = {"code": "consumer", "family": "plane", "weights": "flux", "set": "perplane"}
    pre["bases"] = {"primary": "disp", "secondary": "pos"}
    pre["ledger"] = "ledger.jsonl"
    pre["ok_verdicts"] = ["converged", "fallback"]
    pre["fallback_verdicts"] = ["fallback"]
    pre["levels"] = {"fwer": 0.05, "interval": 0.90, "ucl": 0.05}
    pre["null_per_sigma2"] = D.NULL_PER_SIGMA2
    pre["numerical_fraction"] = 0.5
    pre["eh"] = {"p_range": [0.5, 2.1], "p_cap": 2.0, "safety": 1.25, "fallback_factor": 3.0}
    pre["bootstrap"] = OrderedDict(B_SMOKE if smoke else B_PROD)
    pre["sampling_method"] = "bootstrap-t"
    pre["family"] = list(D.PRIMARY_FAMILY)
    pre["report_levels"] = [0.05, 0.05 / 8]
    pre["targets"] = OrderedDict((a, {"alpha_TP": h, "alpha_TAP": h}) for a, h in D.TARGETS.items())
    pre["first_order"] = OrderedDict((a, {"alpha_TP": D.NULL_PER_SIGMA2 * D.AMPS[a][0], "alpha_TAP": D.NULL_PER_SIGMA2 * D.AMPS[a][0]}) for a in D.TARGETS)
    pre["amplitudes"] = amps
    pre["arms"] = arms
    return pre


def _gate5_section(nm, by, smoke, coverage_summary_rel):
    amps = OrderedDict()
    for amp in G5C:
        a = OrderedDict([("consumer", nm(amp + "_g5c")), ("par2", nm(amp + "_g5p")), ("floor", 2 if smoke else by[nm(amp + "_g5p")].n)])
        if amp == "s4":
            a["j3b"] = True                                   # sigma^2 = 4: the J3B-bound family is part of the qualification (H5)
        amps[amp] = a
    g = OrderedDict([("family_m", 8), ("fwer", 0.05), ("margin_ratio", 1.10), ("level", 0.90), ("B", 399 if smoke else 33599), ("seed", 20261103),
                     ("key", "alpha_LP"), ("ref", "t")])
    if not smoke:                  # the smoke run reads its own coverage output: analyse_sf43.sh --smoke passes --coverage-summary
        g["coverage_summary"] = coverage_summary_rel
    g["floor"] = 2 if smoke else 200
    g["completeness_min"] = 0.999          # H12: Par2 crossing completeness n_last / n_traced below this is flagged
    g["amplitudes"] = amps
    return g


def _gate4_power_section(smoke):
    """Achieved Gate 4 power (acceptance C): descriptive Monte Carlo at the realized sizes under the planning alternatives (sf43ana/gate4_power.py)."""
    return OrderedDict([("R", 20 if smoke else 1000), ("B_budget", 399 if smoke else 4000), ("seed", 20261103), ("chunk", 100),
                        ("scenarios", ["no_bias", "observed"]),
                        ("alternatives", OrderedDict((a, OrderedDict([("alpha_TP", tp), ("alpha_TAP", tap)])) for a, (tp, tap) in D.GATE4_ALTERNATIVES.items())),
                        ("source", D.GATE4_ALTERNATIVES_SOURCE)])


def _basis_section(nm):
    """H2 basis contrasts: flux weights (the production set) and the uniform-weight control, per amplitude."""
    return OrderedDict([("level", 0.90), ("B", 2000), ("seed", 20261103),
                        ("amplitudes", OrderedDict((amp, OrderedDict([("flux", nm(amp + "_prod")), ("unif", nm(amp + "_produnif"))])) for amp in G5C))])


def _g7_section(nm, by):
    twins = OrderedDict((amp, OrderedDict([("consumer", nm(amp + "_g5c")), ("pollock", nm(amp + "_g5pol")), ("par2", nm(amp + "_g5p"))]))
                        for amp in G5C)
    fac = OrderedDict()
    for amp in D.G7_AMPS:
        cs = [OrderedDict([("g", c[0]), ("i", c[1]), ("d", c[2]), ("r", c[3]), ("arm", nm(rid))]) for rid, c in G7_CELLS.items()
              if rid.startswith("g7_%s_" % amp) and nm(rid) in by]
        if cs:
            fac[amp] = cs
    return OrderedDict([("level", 0.90), ("log_bootstrap", 2000), ("seed", 20261103), ("responses", ["alpha_LP", "alpha_TP"]), ("twins", twins),
                        ("factorial", fac)])


def _pollock_section(nm):
    """Acceptance B (production set, flux, both bases) and the G5.5 / item 8 decomposition on the Gate 5 pairs; reported, outside the family."""
    return OrderedDict([
        ("level", 0.90), ("B", 2000), ("seed", 20261103), ("responses", ["alpha_LP", "alpha_TP:disp", "alpha_TP", "alpha_TAP:disp", "alpha_TAP"]),
        ("estimands", ["alpha_TP", "alpha_TAP", "alpha_LP", "U_r"]),
        ("contrast", OrderedDict((amp, OrderedDict([("consumer", nm(amp + "_prod")), ("pollock", nm(amp + "_prodpol"))])) for amp in G5C)),
        ("decomposition", OrderedDict((amp, OrderedDict([("consumer", nm(amp + "_g5c")), ("pollock", nm(amp + "_g5pol")), ("par2", nm(amp + "_g5p"))]))
                                      for amp in G5C))])


def _secondaries(nm, by):
    tw, bdd, par2t, expi = OrderedDict(), [], OrderedDict(), OrderedDict()
    for rid in [r.id for r in D.ROWS if r.fam in ("production", "distance", "gate5_par2")]:
        if nm(rid) not in by:
            continue
        tw[nm(rid)] = {"amplitude": by[nm(rid)].row.amp}
        if by[nm(rid)].row.fam != "production":
            bdd.append(OrderedDict([("name", "bdd_" + nm(rid)), ("arm", nm(rid)), ("amplitude", by[nm(rid)].row.amp), ("t_key", "curve_t"),
                                    ("alpha_key", "curve_alpha")]))
    for amp in G5C:
        par2t[amp] = {"par2": nm(amp + "_g5p"), "consumer": nm(amp + "_g5c")}
        expi["expI_" + amp] = {"arm": nm(amp + "_prod")}
    return OrderedDict([("level", 0.90), ("seed", 20261103), ("B", 1000), ("t_scale", 1.0), ("time_window", tw), ("expI", expi),
                        ("par2_transverse", par2t), ("bdd_fit", bdd),
                        ("note", "Par2 time-window parameters fixed a priori (reduce.U_time = exp(sigma^2 / 6), reduce.par2_exit_x = the outlet plane L / W in "
                                 "box units, regular snapshots every plan par2.snapshots_every steps, about 100 per realization): sf43plan/design.py")])


def _failures():
    return OrderedDict([("max_rate", 0.01), ("failure_classes", ["unusable_solve"]), ("fallback_verdicts", ["fallback"]),
                        ("estimands", ["alpha_TP", "alpha_TAP", "alpha_LP"]), ("q_low", 0.05), ("q_high", 0.95),
                        ("expected", OrderedDict([("unusable_solve_rate", OrderedDict(D.EXPECTED_FAILURE_RATE)),
                                                  ("engine_halt_rate", D.FAILURE["halt_rate"]),
                                                  ("note", "pre-registered expectation, compared with the observed rates in the report; the engine halts an arm only above "
                                                           "halt_rate (a systematic-failure detector); particle-stagnation fractions are reported, never failures")]))])


def _positive_control(aid, smoke):
    return OrderedDict([("arm", aid), ("sigma2", 0.04), ("width_lambda", 24), ("length_lambda", 104), ("window", [25, 94]),
                        ("weights", ["flux", "unif"]), ("level", 0.90), ("B", 200 if smoke else 2000), ("seed", 20261103), ("kmax", 12.0),
                        ("R_bounds", [0.8, 1.25]), ("R_weights", "flux"), ("expectation", POSITIVE_CONTROL_EXPECTATION)])


# ---------------------------------------------------------------------------------------------------------------------------
# seeds registry and design table
# ---------------------------------------------------------------------------------------------------------------------------
PURPOSE = {"g_pc": "positive control sigma^2 = 0.04", "g_d_s1": "distance arm sigma^2 1 (L = 208 lambda)",
           "g_d_s225": "distance arm sigma^2 2.25 (L = 208 lambda)", "g_d_s4": "distance extension sigma^2 4 (L = 208 lambda, 3328x192x192)"}


def _purpose(g, amp):
    if g in PURPOSE:
        return PURPOSE[g]
    if g.startswith("g7_"):
        return "G7 cells of sigma^2 %s, generator %s (whole plot: own seeds)" % (D.AMPS[amp][1], "SF-18" if g.endswith("_S") else "Par2 random-spectral")
    if g.startswith("smoke_"):
        return "smoke test"
    return "CRN group of sigma^2 %s: production, ladders, widths, Gate 5 twin and Par2, h_max passes" % D.AMPS[amp][1]


def seeds_registry():
    reg = OrderedDict()
    reg["forbidden_field_seeds"] = [OrderedDict([("first", a), ("last", b), ("why", w)]) for a, b, w in D.SEED_BASE_FORBIDDEN]
    reg["forbidden_transport_seeds"] = [OrderedDict([("first", a), ("last", b), ("why", w)]) for a, b, w in D.TRANSPORT_FORBIDDEN]
    reg["par2_transport_rule"] = "transport.seed of a unit = %d + 1000 * seed0; realization r of a unit gets +1000 r, so field seed s uses %d + 1000 s" % (
        D.PAR2_TRANSPORT_SEED, D.PAR2_TRANSPORT_SEED)
    for key, ents, fr in (("blocks", production_entries(), reserve_frac), ("smoke_blocks", smoke_entries(), lambda g: 0.15)):
        blocks = group_blocks(ents, fr)
        amp_of = {e.grp: e.row.amp for e in ents}
        reg[key] = [OrderedDict([("group", g), ("purpose", _purpose(g, amp_of[g])), ("start", b["start"]), ("count", b["count"]),
                                 ("end", b["start"] + b["count"] - 1), ("reserves", b["reserves"]),
                                 ("reserve_fraction", round(b["reserves"]["count"] / b["count"], 4)), ("arms", b["arms"])]) for g, b in blocks.items()]
    return reg


def check_seeds(reg):
    """Raise ValueError unless every block and reserve range is disjoint from the others and from the forbidden ranges (field and Par2 transport
    seeds, the latter B + 1000 s for every field seed s)."""
    spans = []
    for kind in ("blocks", "smoke_blocks"):
        for b in reg[kind]:
            spans.append((b["start"], b["end"], b["group"] + " block"))
            spans.append((b["reserves"]["start"], b["reserves"]["start"] + b["reserves"]["count"] - 1, b["group"] + " reserves"))
    spans.sort()
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        if b0 <= a1:
            raise ValueError("seed ranges overlap: %s [%d, %d] and %s [%d, %d]" % (an, a0, a1, bn, b0, b1))
    for a0, a1, an in spans:
        for f in reg["forbidden_field_seeds"]:
            if a0 <= f["last"] and a1 >= f["first"]:
                raise ValueError("%s [%d, %d] overlaps forbidden %s" % (an, a0, a1, f["why"]))
        t0, t1 = D.PAR2_TRANSPORT_SEED + 1000 * a0, D.PAR2_TRANSPORT_SEED + 1000 * a1
        for f in reg["forbidden_transport_seeds"]:
            if t0 <= f["last"] and t1 >= f["first"]:
                raise ValueError("%s: Par2 transport seeds [%d, %d] overlap forbidden %s" % (an, t0, t1, f["why"]))
        for b0, b1, bn in spans:
            if bn != an and not (t1 < b0 or t0 > b1):
                raise ValueError("%s: Par2 transport seeds overlap the field seeds of %s" % (an, bn))
    return True


def total_gpu_h(ents):
    return sum(e.n * D.SEC[e.row.sec] for e in ents) / 3600.0, sum((e.hm_n or 0) * D.H3_EXTRA.get(e.row.amp, 0.0) for e in ents) / 3600.0


def design_table_md(plan, reg, placeholder, smoke_only=False):
    ents = production_entries()
    out = ["# SF-43 design table, variant A (generated by `python3 -m sf43plan generate`; do not edit)", ""]
    if smoke_only:
        out += ["**SMOKE-ONLY plan: the coverage sha256 are placeholders; regenerate with the smoke job's coverage files before the production run.**", ""]
    if placeholder:
        out += ["**The binaries md5 are PLACEHOLDERS (all zeros): the campaign engine's md5 gate refuses to start until the plan is regenerated.**", ""]
    out += ["## Arms", "", "| arm | sigma^2 | family | runner | grid | W | lambda/h | L | M | n_conf | n_max | group | prio | heavy | min_free MiB | unit | est s | GPU-h (n_conf) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    by_fam = OrderedDict()
    for e in ents:
        r, g = e.row, grid(e.row)
        h = e.n * D.SEC[r.sec] / 3600.0
        by_fam[r.fam] = by_fam.get(r.fam, 0.0) + h
        out.append("| %s | %s | %s | %s | %dx%dx%d | %d | %d | %d | %s | %d | %d | %s | %d | %s | %d | %d | %.1f | %.3f |" % (
            e.id, amp_info(r)[1], r.fam, "consumer" if r.run == "c" else "par2", g[0], g[1], g[2], r.W, r.lh, r.L,
            amp_info(r)[3] if r.run == "c" else "-", e.n, e.nmax, e.grp, e.prio, "yes" if e.heavy else "no", min_free_mib(r), unit_size(r),
            D.SEC[r.sec], h))
    arms_h, extra_h = total_gpu_h(ents)
    out += ["", "Planning GPU-h at n_conf: arms %.2f + three-pass extra on the first 30 production seeds %.2f = **%.2f** (`design_cost.py` variant A: 59.6)." % (
        arms_h, extra_h, arms_h + extra_h), "", "By family (GPU-h): " + ", ".join("%s %.2f" % (k, v) for k, v in by_fam.items()), "",
        "## Seed blocks", "", "| group | start | count | end | reserves | reserve % | purpose |", "|---|---|---|---|---|---|---|"]
    for b in reg["blocks"]:
        out.append("| %s | %d | %d | %d | %d-%d | %.1f | %s |" % (b["group"], b["start"], b["count"], b["end"], b["reserves"]["start"],
                                                                 b["reserves"]["start"] + b["reserves"]["count"] - 1, 100 * b["reserve_fraction"], b["purpose"]))
    out += ["", "Smoke blocks: " + ", ".join("%s %d-%d" % (b["group"], b["start"], b["end"]) for b in reg["smoke_blocks"]), "",
            "Forbidden field seeds: " + "; ".join("%d-%d" % (f["first"], f["last"]) for f in reg["forbidden_field_seeds"]), "",
            "Smoke: %d arms, %.2f GPU-h of work estimated, budget %s GPU-h, cutoff %s min after the start." % (
                len(plan["smoke"]["arms"]), plan["smoke"]["_smoke_work_gpu_h_estimate"], plan["smoke"]["budget_gpu_h"], plan["smoke"]["cutoff_minutes"]), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------------
# analysis file hashes
# ---------------------------------------------------------------------------------------------------------------------------
def analysis_files():
    """Repo-relative paths of the analysis files present now: non-test .py/.sh/.json of the analysis packages and of the SF-42 libraries
    they import, plus the wrapper ``analyse_sf43.sh``."""
    pk = [(ART_REL + "/scripts", ("sf43ana", "sf43cov", "recompute43", "reconcile43")), (SF42_REL, ("mcstat", "sf42red", "seedlattice"))]
    out = []
    for base, names in pk:
        for n in names:
            for dp, dns, fns in os.walk(os.path.join(REPO, base, n)):
                dns[:] = sorted(d for d in dns if d not in ("__pycache__", "tests"))
                out += [os.path.relpath(os.path.join(dp, f), REPO).replace(os.sep, "/") for f in sorted(fns) if f.endswith((".py", ".sh", ".json"))]
    if not os.path.isfile(os.path.join(REPO, ANALYSE_SH)):
        raise FileNotFoundError(ANALYSE_SH)
    return sorted(out + [ANALYSE_SH])


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def analysis_hashes(prereg_text, prereg_rel, coverage_files=None):
    """{repo-relative path: sha256}: the analysis files, the pre-registration (its text, as generated) and, for a production plan, the coverage
    summary and qualification (``coverage_files`` = (summary path, qualification path); None: not included, see ``with_coverage_placeholders``)."""
    h = OrderedDict()
    for p in analysis_files():
        with open(os.path.join(REPO, p), "rb") as f:
            h[p] = sha256_bytes(f.read())
    h[prereg_rel] = sha256_bytes(prereg_text.encode())
    if coverage_files is not None:
        for rel, path in zip((COV_SUMMARY_REL, COV_QUAL_REL), coverage_files):
            with open(path, "rb") as f:
                h[rel] = sha256_bytes(f.read())
    return OrderedDict(sorted(h.items()))


def with_coverage_placeholders(h):
    return OrderedDict(sorted(dict(h, **{COV_SUMMARY_REL: ZERO_SHA256, COV_QUAL_REL: ZERO_SHA256}).items()))


def dumps(o):
    return json.dumps(o, indent=1) + "\n"
