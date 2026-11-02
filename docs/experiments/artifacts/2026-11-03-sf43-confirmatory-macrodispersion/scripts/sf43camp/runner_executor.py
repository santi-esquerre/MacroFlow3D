"""SF-43 campaign engine: the RUNNER EXECUTOR (consumer `lagrangian_ensemble` and Par2 `macroflow3d_pipeline`).

Duck-typed object the engine core calls; it builds the command of one unit (a consecutive seed range of one arm, one *segment
directory*), classifies the outcome, lists complete realizations (resume), reduces with the SF-42 reducer, prunes the per-particle
arrays and gives the blinded standard deviations.  Nothing here launches a process, picks a GPU (the core sets CUDA_VISIBLE_DEVICES per
lane) or reads a mean: ``blinded_sd`` returns standard deviations and counts only (``mcstat.blinded``).

Layout of a segment directory ``D`` (the unit's cwd): ``D/segment.json`` (seed0, nr, n_planes, dt, runner), ``D/run.log`` (stdout+stderr),
consumer ``D/out`` (``--out``; ``out/realizations/r_%04d[/hmax_<d>]``), Par2 ``D/config.yaml`` and ``D/out`` (``output.output_dir``;
``out/crossings/realization_%04d_planes.csv``, ``out/stats/``, ``out/snapshots/``).  Realization index r <-> seed s = seed0 + r in
both codes (consumer ``--seed0 --nr``; Par2 ``sf18_medium_b.seed0`` or ``stochastic.seed`` = seed0 and ``NR``).  A stopped unit is never
resumed in place (seed0 is part of the consumer's DONE signature): its remaining seeds go to a NEW segment directory.

ARM DICT (from the plan JSON).  Common: ``id``, ``family``, ``amplitude``, ``runner`` (``consumer``|``par2``), ``raw_gb`` (per seed),
``reduce`` and the runner sub-dict.  Lengths are in box units (the box is 1 wide; ``ell`` = 1/W with W the transverse width in lambda).

``consumer`` (required unless a default is given): ``grid`` [nx,ny,nz]; ``eps`` (``--eps`` with ``--sigma2 1``, so sigma^2 = eps^2);
``ell``; ``window`` "w2a,w2b,w3a,w3b" (string, passed as is); ``np_side`` M; ``planes_d`` travel distances d in lambda (planes at
x1 = x0 + d ell with x0 = ``x0_ell`` ell, ``x0_ell`` default 5, which is the consumer's own default x0 for medium b; the float expression
is the SF-42 pilots' ``5.0*ell + d*ell``); ``hmax_div`` list (default [8]; [8,16,32] = three passes of one stage build); ``twin`` and
``pollock_bridge`` (default true -> ``on``); ``tol`` (default "1e-8"); ``times`` (default "auto"); ``sigma2`` (default "1");
``fused_splines``/``single_launch`` (default true); ``hmax_div_first`` = {"n": N, "divs": [8, 16, 32]} (optional): a unit whose seeds
are all among the first N positions of the arm's seed range (``seeds.start`` .. +N) runs with ``--hmax-div 8,16,32``, every other
unit (later positions, reserve and extension seeds) with ``hmax_div``; the plan validates that ``unit_size`` divides N so that no
unit straddles N.  The reduced rows of a single-pass segment are labelled with the production pass ``hmax_div`` (the first of
``hmax_div``) instead of -1, so the arm has one label for its production pass in both layouts.  Strings are passed as they are,
numbers with ``repr``.

``par2`` (SF-42 keys; config-only G7 factors): ``grid``; ``eps``; ``ell``; ``window``; ``planes_d``; ``x0_ell`` (default 5);
``n_particles`` (default 10000); ``cg_max_iter`` (default 2000, spec item 3); ``cg_rtol`` (default "1.0e-10"; factor R: "1.0e-6");
``hz`` horizon factor (default 3.6, SF-42 J3B); ``dt_coeff`` (default 0.0244140625; ``dt = dt_coeff * dx``, dx = 1/ny); ``dt_half``
(factor D: dt/2 and EXACTLY 2 x n_steps, so the time grid nests; default false); ``generator`` ``sf18`` (default: the consumer's SF-18
medium (b) field, same seeds) | ``random_spectral`` (factor G: Par2's own generator, field seed = ``stochastic.seed + r`` with seed =
seed0, ``stochastic.sigma2`` = ``sigma2`` or eps^2, ``n_modes`` REQUIRED, SF-39 used 10000); ``inject`` ``window`` (default) | ``full``
(factor I: ``inject_window`` omitted); ``n_modes`` (default 1000 and unused with ``sf18``); ``transport_seed`` base B: the unit's
``transport.seed`` is B + 1000 seed0 so that the pipeline's per-realization +1000 r gives every field seed s the stream B + 1000 s
whatever the segmentation (default 420800001); ``snapshots_every`` (default 0 = no ``analysis.snapshots``; >0 enables regular
snapshots, needed only by the time-window secondary); ``mg_levels`` (default: the rule of ``pilots.design.mg_levels``).

``reduce``: ``code`` (``consumer``|``par2``); ``d_windows`` list of [d_lo, d_hi] (default [[25, 94]]; one table per window);
``weights``/``bases`` (informational: the SF-42 reducer always writes every variant); optional ``blind`` = {weights, basis, set,
window, hmax_div} picks the rows ``blinded_sd`` reads (defaults: consumer flux / Par2 unif, basis ``disp`` (H2), set ``perplane``,
window [25, 94], consumer production pass ``hmax_div`` 8 / Par2 -1); time-window secondary (optional): ``U_time`` and
``par2_exit_x`` (Par2 needs both and ``snapshots_every`` > 0).

POLLOCK ROWS (the Pollock contrast, spec items 5 and 8).  A consumer arm with ``pollock_bridge`` on (default) is reduced with
``sf42red reduce --pollock``: besides the ``code=consumer`` rows the tables hold ``code=pollock`` plane rows of ``<realization>/pollock/``
(same columns and estimators; ``blinded_sd`` reads the arm's runner code only, so these never enter the re-estimation).  ``prune``
then deletes the Pollock arrays of non-retained realizations (their rows are in the table) and never one whose rows are missing.

``primary`` (arm flag, default false): the arm feeds the primary family; ``prune`` then keeps, for EVERY realization, the plane arrays
(``seeds``, ``weights``, ``planes_x23``, ``planes_tau``: exactly what the displacement-basis plane estimators read, so ``recompute43``
can rebuild them) and deletes only the others (time-pass arrays, Pollock and ``planes_I``); a non-primary arm keeps only its
retained realizations (all arrays).  Par2 ``prune`` deletes the regular snapshots after the reduction (those of retained realizations
stay) and keeps the crossing CSVs.

SEED MAP.  ``command`` writes ``D/seedmap.json`` = {seed0, nr, seeds: {r: seed0 + r}} (both codes).  The Par2 rows of the reduced
tables carry ``seed = seed0 + r`` (``sf42red reduce --par2-seed0``), so the analysis pairs Gate 5 by seed.  The field seed Par2 prints
(``SF-18 medium (b) field (seed=`` or ``K generation (seed=``) is cross-checked against the map in the solver records.

PAR2 USABILITY (H5, H12).  The pipeline exits 0 after ``Converged=NO``; it prints ``Converged=YES|NO  iters=N  res=<initial> -> <final>``
(absolute norms; no balance).  Verdict per seed: ``converged`` (YES); ``fallback`` (NO with final/initial <= 1e-8: usable, as the
consumer's fallback; the balance is not printed, so the rule uses the residual only); ``not_converged_beyond_fallback`` (else: unusable,
the seed is replaced group-wide); ``non_finite``.
"""
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys

_SF42 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "2026-10-30-sf42-confirmatory-tooling",
                                      "scripts"))
if _SF42 not in sys.path:
    sys.path.insert(0, _SF42)

from mcstat import blinded as _blinded  # noqa: E402
from sf42red import prune as _prune  # noqa: E402
from sf42red import reduce as _red  # noqa: E402

LOG = "run.log"
PAR2_FORMAT = "# format_version=1 sf42_plane_crossings"
# Exit code and text signatures (see the report of the executability audit for the source lines).
_OOM = [re.compile(p, re.I) for p in (
    r"CUDA error at \S+ - out of memory", r"cuFFT error at \S+ - code 2\b", r"cuBLAS error at \S+ - code [13]\b", r"out of memory",
    r"Failed to allocate (RNG states|corner velocity buffers|drift correction buffers|status array|wrap counter arrays)",
    r"std::bad_alloc")]
_DISK = re.compile(r"write failed|cannot write", re.I)
_CONVERGED = re.compile(r"Converged=(YES|NO)\b(?:\s+iters=(\d+)\s+res=(\S+)\s+\S+\s+(\S+))?")
_FIELD_SEED = re.compile(r"\[4\] (?:SF-18 .*?|K generation )\(seed=(\d+)")
USABLE = ("converged", "fallback")
KEEP_ARRAYS = ("seeds", "weights", "planes_x23", "planes_tau")  # what the displacement-basis plane estimators read
FALLBACK_TOL = 1e-8
BLIND_EST = ("alpha_TP", "alpha_TAP", "alpha_LP")


def _num(v):
    """Strings pass as they are (the plan fixes the text), numbers with repr (shortest round-trip of the same double)."""
    return v if isinstance(v, str) else repr(v) if isinstance(v, float) else str(v)


def _need(d, keys, what):
    miss = [k for k in keys if k not in d]
    if miss:
        raise KeyError("%s: missing key(s) %s" % (what, miss))


def _seed0(seeds):
    s = [int(x) for x in seeds]
    if not s or s != list(range(s[0], s[0] + len(s))):
        raise ValueError("seeds must be a non-empty consecutive range, got %r" % (seeds,))
    return s[0], len(s)


def hmax_divs(arm, seeds):
    """The ``--hmax-div`` list of a unit: the arm's ``hmax_div_first`` divs if every seed is among its first N positions, else ``hmax_div``
    (default [8]).  A unit that straddles N is an error (the plan makes it impossible)."""
    c = arm["consumer"]
    first = c.get("hmax_div_first")
    if not first:
        return [int(d) for d in c.get("hmax_div", [8])]
    pos = [int(s) - int(arm["seeds"]["start"]) for s in seeds]
    inn = [0 <= q < int(first["n"]) for q in pos]
    if any(inn) and not all(inn):
        raise ValueError("arm %s: unit %s straddles position %s of hmax_div_first" % (arm["id"], list(seeds), first["n"]))
    return [int(d) for d in (first["divs"] if all(inn) else c.get("hmax_div", [8]))]


def peak_mib(log_text):
    """Largest device memory the consumer reports (``MEM_PHASE`` ``device_used`` / ``exact_bytes``, bytes -> MiB), or None."""
    v = [int(m) for m in re.findall(r"^MEM_PHASE\S* .*?\bexact_bytes=(\d+)", log_text, re.M)]
    v += [int(m) for m in re.findall(r"\bdevice_used=(\d+)", log_text)]
    return max(v) / 1048576.0 if v else None


def planes_of(ell, planes_d, x0_ell=5.0):
    """x1 = x0 + d ell with x0 = x0_ell ell (SF-42 pilots' expression; the same list in both codes)."""
    x0 = x0_ell * ell
    return [x0 + d * ell for d in planes_d]


def mg_levels(nx, ny, nz):
    """Rule of ``pilots.design.mg_levels`` (coarsest extent >= 8)."""
    def can(n):
        return n % 2 == 0 and n // 2 >= 8 and (n // 2) % 2 == 0
    lv = 1
    while can(nx) and can(ny) and can(nz):
        nx, ny, nz, lv = nx // 2, ny // 2, nz // 2, lv + 1
    return lv


def par2_time(nx, ny, ell, hz, dt_coeff, dt_half=False):
    dx = 1.0 / ny
    dt = dt_coeff * dx
    n = int(math.ceil(hz * (nx * dx - 10.0 * ell) / dt))
    return (dt / 2.0, 2 * n) if dt_half else (dt, n)


def par2_yaml(arm, seeds):
    p = arm["par2"]
    _need(p, ("grid", "eps", "ell", "window", "planes_d"), "arm %s par2" % arm.get("id"))
    s0, nr = _seed0(seeds)
    nx, ny, nz = (int(v) for v in p["grid"])
    ell, eps = float(p["ell"]), float(p["eps"])
    gen = p.get("generator", "sf18")
    if gen not in ("sf18", "random_spectral"):
        raise ValueError("par2.generator must be sf18|random_spectral, got %r" % (gen,))
    inject = p.get("inject", "window")
    if inject not in ("window", "full"):
        raise ValueError("par2.inject must be window|full, got %r" % (inject,))
    if gen == "random_spectral" and "n_modes" not in p:
        raise KeyError("arm %s par2: generator random_spectral needs n_modes" % arm.get("id"))
    w = [float(v) for v in str(p["window"]).split(",")]
    if len(w) != 4:
        raise ValueError("par2.window needs 4 entries, got %r" % (p["window"],))
    dt, n_steps = par2_time(nx, ny, ell, float(p.get("hz", 3.6)), float(p.get("dt_coeff", 0.0244140625)), bool(p.get("dt_half", False)))
    x0 = p.get("x0_ell", 5.0) * ell
    planes = ", ".join(repr(x) for x in planes_of(ell, p["planes_d"], p.get("x0_ell", 5.0)))
    if gen == "sf18":
        field = "  field_source: sf18_medium_b\n  sf18_medium_b: {sigma2: 1, eps: %s, seed0: %d}\n" % (repr(eps), s0)
        sig2, seed = "1.0", 42
    else:
        sig2, seed, field = _num(p.get("sigma2", eps * eps)), s0, ""
    snap = ("  snapshots: {enabled: true, every: %d}\n" % int(p["snapshots_every"])) if int(p.get("snapshots_every", 0)) > 0 else ""
    win = "  inject_window: [%s]\n" % ", ".join(repr(v) for v in w) if inject == "window" else ""
    mgl = p.get("mg_levels", mg_levels(nx, ny, nz))
    return """# SF-43 arm {id} seeds {s0}..{s1} (generated by sf43camp.runner_executor.par2_yaml)
run_mode: ensemble
grid: {{nx: {nx}, ny: {ny}, nz: {nz}, dx: {dx}}}
stochastic:
  sigma2: {sig2}
  corr_length: {ell}
  n_modes: {nmodes}
  covariance_type: 1
  seed: {seed}
  K_mean: 1.0
{field}flow:
  solver: mg_cg
  mg_levels: {mgl}
  mg_pre_smooth: 4
  mg_post_smooth: 4
  mg_coarse_iters: 50
  mg_max_cycles: 20
  cg_max_iter: {cgmax}
  cg_rtol: {rtol}
  cg_check_every: 10
  rtol: 1.0e-6
  pin: {{mode: off}}
  bc:
    west:   {{ type: dirichlet, value: {head} }}
    east:   {{ type: dirichlet, value: 0.0 }}
    south:  {{ type: periodic }}
    north:  {{ type: periodic }}
    bottom: {{ type: periodic }}
    top:    {{ type: periodic }}
transport:
  n_particles: {npart}
  dt: {dt}
  n_steps: {nst}
  porosity: 1.0
  diffusion: 0.0
  alpha_l: 0.0
  alpha_t: 0.0
  seed: {tseed}
  output_every: {nst}
  snapshot_every: 0
  inject_x: {x0}
{win}analysis:
  macrodispersion: {{enabled: true, NR: {nr}, lambda: {ell}, vmean_norm: 1.0, sample_every: 1000, var_estimator: biased}}
{snap}  plane_crossings:
    enabled: true
    planes: [{planes}]
diagnostics:
  velocity_field: false
output:
  output_dir: out
  save_K: false
  save_head: false
  save_velocity: false
  save_particles: false
""".format(id=arm["id"], s0=s0, s1=s0 + nr - 1, nx=nx, ny=ny, nz=nz, dx=repr(1.0 / ny), sig2=sig2, ell=repr(ell),
           nmodes=int(p.get("n_modes", 1000)), seed=seed, field=field, mgl=mgl, cgmax=int(p.get("cg_max_iter", 2000)),
           rtol=_num(p.get("cg_rtol", "1.0e-10")), head=repr(nx * (1.0 / ny)), npart=int(p.get("n_particles", 10000)), dt=repr(dt),
           nst=n_steps, tseed=int(p.get("transport_seed", 420800001)) + 1000 * s0, x0=repr(x0), win=win, nr=nr, snap=snap,
           planes=planes)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def par2_solves(log_text):
    """[{conv, iters, res0, res1, rel, verdict, usable, field_seed}] in realization order, from a Par2 log (see the module docstring)."""
    seeds = [int(m) for m in _FIELD_SEED.findall(log_text)]
    out = []
    for i, m in enumerate(_CONVERGED.finditer(log_text)):
        yes = m.group(1) == "YES"
        try:
            res0, res1 = float(m.group(3)), float(m.group(4))
        except (TypeError, ValueError):
            res0 = res1 = float("nan")
        rel = (res1 / res0 if res0 > 0.0 else 0.0) if math.isfinite(res0) and math.isfinite(res1) else float("nan")
        verdict = ("converged" if yes else "non_finite" if not math.isfinite(rel) and m.group(3) is not None else
                   "fallback" if rel <= FALLBACK_TOL else "not_converged_beyond_fallback")
        out.append({"conv": yes, "iters": None if m.group(2) is None else int(m.group(2)), "res0": res0, "res1": res1, "rel": rel,
                    "verdict": verdict, "usable": verdict in USABLE, "field_seed": seeds[i] if i < len(seeds) else None})
    return out


class RunnerExecutor:
    def __init__(self, plan=None, build_dir=None):
        """The engine builds the executor as ``Class(plan, build_dir)``; the methods take everything they need as arguments."""
        self.plan, self.build_dir = plan, build_dir

    # ---- command ---------------------------------------------------------------------------------------------------------------
    def command(self, arm, seeds, segment_dir, build_dir):
        s0, nr = _seed0(seeds)
        meta = {"arm": arm["id"], "runner": arm["runner"], "seed0": s0, "nr": nr}
        if arm["runner"] == "consumer":
            c = arm["consumer"]
            _need(c, ("grid", "eps", "ell", "window", "np_side", "planes_d"), "arm %s consumer" % arm["id"])
            ell = float(c["ell"])
            planes = planes_of(ell, c["planes_d"], c.get("x0_ell", 5.0))
            onoff = lambda v: "on" if v else "off"  # noqa: E731
            argv = [os.path.join(build_dir, "lagrangian_ensemble"), "--medium", "b", "--grid", ",".join(str(int(v)) for v in c["grid"]),
                    "--sigma2", _num(c.get("sigma2", "1")), "--eps", _num(c["eps"]), "--ell", repr(ell), "--window", str(c["window"]),
                    "--np-side", str(int(c["np_side"])), "--planes", ",".join(repr(x) for x in planes),
                    "--times", _num(c.get("times", "auto")), "--tol", _num(c.get("tol", "1e-8")),
                    "--hmax-div", ",".join(str(d) for d in hmax_divs(arm, seeds))]
            argv += (["--fused-splines"] if c.get("fused_splines", True) else []) + (["--single-launch"] if c.get("single_launch", True) else [])
            argv += ["--twin", onoff(c.get("twin", True)), "--pollock-bridge", onoff(c.get("pollock_bridge", True)),
                     "--seed0", str(s0), "--nr", str(nr), "--out", os.path.join(segment_dir, "out")]
            files = {}
            meta["n_planes"] = len(planes)
            meta["hmax_div"] = hmax_divs(arm, seeds)
        elif arm["runner"] == "par2":
            p = arm["par2"]
            argv = ["stdbuf", "-oL", os.path.join(build_dir, "macroflow3d_pipeline"), "config.yaml"]
            files = {os.path.join(segment_dir, "config.yaml"): par2_yaml(arm, seeds)}
            meta["n_planes"] = len(p["planes_d"])
            meta["dt"] = par2_time(int(p["grid"][0]), int(p["grid"][1]), float(p["ell"]), float(p.get("hz", 3.6)),
                                   float(p.get("dt_coeff", 0.0244140625)), bool(p.get("dt_half", False)))[0]
        else:
            raise ValueError("unknown runner %r" % (arm["runner"],))
        files[os.path.join(segment_dir, "segment.json")] = json.dumps(meta, sort_keys=True) + "\n"
        files[os.path.join(segment_dir, "seedmap.json")] = json.dumps(
            {"runner": arm["runner"], "seed0": s0, "nr": nr, "seeds": {str(r): s0 + r for r in range(nr)}}, sort_keys=True) + "\n"
        return {"argv": argv, "cwd": segment_dir, "env": {}, "files": files, "log_path": os.path.join(segment_dir, LOG)}

    # ---- completed realizations ------------------------------------------------------------------------------------------------
    def _consumer_dirs(self, segment_dir):
        """{seed: realization directory} of the DONE realizations."""
        out = {}
        for d in sorted(glob.glob(os.path.join(segment_dir, "out", "realizations", "r_[0-9][0-9][0-9][0-9]"))):
            fj = _read_json(os.path.join(d, "field.json"))
            if os.path.isfile(os.path.join(d, "DONE")) and fj and "seed" in fj:
                out[int(fj["seed"])] = d
        return out

    def _par2_state(self, segment_dir):
        """(meta, {r: solve record of ``par2_solves``}, {r: crossing CSV path}) of the complete realizations of a Par2 segment."""
        meta = _read_json(os.path.join(segment_dir, "segment.json"))
        if not meta:
            return None, {}, {}
        conv = dict(enumerate(par2_solves(_read_text(os.path.join(segment_dir, LOG)))))
        csvs = {}
        for r in range(meta["nr"]):
            path = os.path.join(segment_dir, "out", "crossings", "realization_%04d_planes.csv" % r)
            lines = [ln for ln in _read_text(path).splitlines() if ln.strip() and not ln.startswith("#")]
            if lines and _read_text(path).startswith(PAR2_FORMAT) and len(lines) - 1 == meta["n_planes"]:
                csvs[r] = path
        return meta, conv, csvs

    def completed(self, arm, segment_dir):
        """Seeds with complete outputs.  Par2: a realization whose head solve is unusable (verdict neither converged nor fallback) is
        NOT complete (unusable_solve)."""
        if arm["runner"] == "consumer":
            return sorted(self._consumer_dirs(segment_dir))
        meta, conv, csvs = self._par2_state(segment_dir)
        return [] if meta is None else [meta["seed0"] + r for r in sorted(csvs) if r in conv and conv[r]["usable"]]

    # ---- classification --------------------------------------------------------------------------------------------------------
    def classify(self, arm, seeds, segment_dir, exit_code, log_text):
        """-> {cls, completed_seeds, failed_seed, failed_seeds, verdicts {seed: verdict}, seconds, solver {seed: record}, [note]}.
        ``solver``: per seed with a solve record: verdict, iterations, rel, balance (consumer), max_div (consumer, if in solver_b.json),
        field_seed (Par2).  ``failed_seeds``: every seed of the unit with an unusable solve (Par2 runs on after one)."""
        s0, nr = _seed0(seeds)
        done = [s for s in self.completed(arm, segment_dir) if s0 <= s < s0 + nr]
        verdicts, seconds, solver = {}, {}, {}
        failed, bad, note = None, [], ""
        if arm["runner"] == "consumer":
            for r, v in re.findall(r"^R=(\d+) STAGE medium=b status=(\w+)", log_text, re.M):
                verdicts[s0 + int(r)] = v
            for r, t in re.findall(r"^TIMING r=(\d+) seconds=([0-9.eE+-]+)", log_text, re.M):
                seconds[s0 + int(r)] = float(t)
            for s, d in self._consumer_dirs(segment_dir).items():
                sv = _read_json(os.path.join(d, "solver_b.json"))
                if sv and "verdict" in sv:
                    verdicts[s] = sv["verdict"]
                if sv:
                    solver[s] = {"verdict": sv.get("verdict"), "iterations": sv.get("iterations"), "rel": sv.get("rel"),
                                 "balance": sv.get("balance"), "max_div": sv.get("max_div")}
            sf = re.findall(r"^R=(\d+) STAGE_FAILED .*?verdict=(\w+)", log_text, re.M)
            if sf:
                failed = s0 + int(sf[-1][0])
                verdicts[failed] = sf[-1][1]
                bad = [failed]
        else:
            meta, conv, csvs = self._par2_state(segment_dir)
            for r, c in conv.items():
                verdicts[s0 + r] = c["verdict"]
                solver[s0 + r] = {"verdict": c["verdict"], "iterations": c["iters"], "rel": c["rel"], "balance": None, "max_div": None,
                                  "field_seed": c["field_seed"]}
                if c["field_seed"] is not None and c["field_seed"] != s0 + r:
                    note += "field seed %s printed for seed %d; " % (c["field_seed"], s0 + r)
            prev = os.path.getmtime(os.path.join(segment_dir, "segment.json")) if meta else None
            for r in sorted(csvs):  # wall time between successive crossing CSVs (r = 0: from the unit's launch files); approximate
                t = os.path.getmtime(csvs[r])
                seconds[s0 + r] = max(0.0, t - prev)
                prev = t
            bad = [s0 + r for r, c in sorted(conv.items()) if not c["usable"]]
            if bad:
                failed = bad[0]
                note += "unusable Par2 solves for seeds %s" % bad
        if exit_code in (137, -9):
            cls = "oom"
        elif exit_code == 4 and arm["runner"] == "consumer":
            cls = "unusable_solve"
        elif exit_code in (2, 3, 7):
            cls = "config_error"
        elif exit_code != 0 and any(p.search(log_text) for p in _OOM):
            cls = "oom"
        elif exit_code != 0 and _DISK.search(log_text):
            cls = "disk_full"
        elif exit_code != 0:
            cls = "other_error"
        elif failed is not None and arm["runner"] == "par2":
            cls = "unusable_solve"
        elif sorted(done) != list(range(s0, s0 + nr)):
            cls, note = "other_error", "exit 0 but complete seeds are %s" % done
        else:
            cls = "ok"
        if arm["runner"] == "par2" and cls != "unusable_solve":
            failed, bad = None, []  # an unusable seed is a failure of the unit only if nothing else stopped it
        missing = [s for s in range(s0, s0 + nr) if s not in done]  # the runners go in seed order: the first missing seed is the failing one
        failed = None if cls == "ok" else failed if failed is not None else missing[0] if missing else None
        out = {"cls": cls, "completed_seeds": done, "failed_seed": failed, "peak_mib": peak_mib(log_text), "failed_seeds": bad if cls == "unusable_solve" else [],
               "verdicts": verdicts, "seconds": seconds, "solver": solver}
        if note:
            out["note"] = note.strip("; ")
        return out

    # ---- reduction, pruning, blinded SDs ---------------------------------------------------------------------------------------
    def reduce(self, arm, segment_dir, out_dir):
        rd = arm.get("reduce", {})
        done = self.completed(arm, segment_dir)
        if not done:
            raise _red.ReduceError("segment %s has no complete realization" % segment_dir)
        os.makedirs(out_dir, exist_ok=True)
        run = arm[arm["runner"]]
        ell = float(run["ell"])
        x0 = run.get("x0_ell", 5.0) * ell
        kw, consumer, par2, series = {}, [], [], []
        if arm["runner"] == "consumer":
            consumer = [os.path.join(segment_dir, "out")]
            if run.get("pollock_bridge", True):  # the arm's runs carry the bridge (`--pollock-bridge on`): reduce its arrays too (code=pollock rows)
                kw["pollock"] = True
            if run.get("hmax_div_first"):  # one label for the production pass whether the segment ran one or several passes
                kw["single_pass_hmax"] = int(run.get("hmax_div", [8])[0])
        else:
            meta, _, csvs = self._par2_state(segment_dir)
            par2 = [csvs[s - meta["seed0"]] for s in done]
            kw = {"par2_seed0": meta["seed0"]}  # the Par2 rows carry seed = seed0 + r (Gate 5 pairs by seed)
            if int(run.get("snapshots_every", 0)) > 0 and "U_time" in rd and "par2_exit_x" in rd:
                series = [os.path.join(segment_dir, "out", "stats", "realization_%04d_timeseries.csv" % (s - meta["seed0"])) for s in done]
                kw.update(par2_exit_x=rd["par2_exit_x"], par2_dt=meta["dt"])
        res = {"tables": [], "planes": [], "manifests": [], "manifest_sha256": {}, "seeds": done}
        for lo, hi in rd.get("d_windows", [[25, 94]]):
            out = os.path.join(out_dir, "%s__%s__d%g-%g.csv" % (arm["id"], os.path.basename(os.path.normpath(segment_dir)), lo, hi))
            _red.reduce_all(consumer, par2, series, ell, x0, float(lo), float(hi), rd.get("U_time"), out, segment_dir, **kw)
            man = out + ".manifest.json"
            res["tables"].append(out)
            res["planes"].append(out[:-4] + ".planes.csv")
            res["manifests"].append(man)
            res["manifest_sha256"][os.path.basename(man)] = _red.sha256_file(man)
        return res

    def prune(self, arm, segment_dir, reduced, retained_seeds):
        """Free the raw data of a reduced segment.  ``retained_seeds`` is an ARM-level list (entries outside this segment are ignored;
        the arm-level minimum of 5 is the engine's).

        Consumer: delete the per-particle .npy of the realizations not in ``retained_seeds``; for a ``primary`` arm the plane arrays
        (``KEEP_ARRAYS``) of EVERY realization stay.  Verification and table coverage are the SF-42 pruner's (the manifest of the FIRST
        table is checked: all tables of a segment record the same inputs).  Par2: delete the regular snapshots (those of retained
        realizations stay) after verifying the manifest; the crossing CSVs always stay."""
        if arm["runner"] != "consumer":
            return self._prune_par2(segment_dir, reduced, retained_seeds)
        keep = set(int(s) for s in retained_seeds)
        dirs = self._consumer_dirs(segment_dir)
        kept = sorted(s for s in dirs if s in keep)
        retained = os.path.join(segment_dir, "retained.txt")
        with open(retained, "w") as f:
            f.write("# arm-level retained seeds present in this segment\n")
            for s in kept:
                f.write(os.path.relpath(dirs[s], segment_dir) + "\n")
        log = os.path.join(segment_dir, "prune_log.csv")
        keep_arrays = KEEP_ARRAYS if arm.get("primary") else ()
        _prune.prune([os.path.join(segment_dir, "out")], retained, reduced["tables"][0], segment_dir, True, log, out=lambda s: None,
                     min_retained=0, keep_arrays=keep_arrays)
        with open(log, newline="") as f:
            rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
        gone = [r for r in rows if r["action"] == "delete"]
        return {"deleted_files": len(gone), "deleted_bytes": sum(int(r["size"]) for r in gone), "retained": kept, "log": log,
                "kept_arrays": list(keep_arrays), "log_sha256": _red.sha256_file(log)}

    def _prune_par2(self, segment_dir, reduced, retained_seeds):
        _prune._guard(segment_dir)
        _prune.verify(reduced["tables"][0], segment_dir)  # the reduction is intact: its inputs are what was reduced
        meta = _read_json(os.path.join(segment_dir, "segment.json"))
        keep = {"r_%04d" % (int(s) - meta["seed0"]) for s in retained_seeds if 0 <= int(s) - meta["seed0"] < meta["nr"]}
        snaps, n, size = os.path.join(segment_dir, "out", "snapshots"), 0, 0
        for dp, dn, fn in os.walk(snaps, followlinks=False):
            if os.path.basename(dp) in keep or os.path.basename(os.path.dirname(dp)) in keep:
                continue
            for name in fn:
                path = os.path.join(dp, name)
                if os.path.islink(path):
                    raise _prune.PruneRefused("symlink in the snapshots: %s" % path)
                size += os.path.getsize(path)
                os.remove(path)
                n += 1
        return {"deleted_files": n, "deleted_bytes": size, "retained": sorted(meta["seed0"] + int(r[2:]) for r in keep),
                "log": None, "kept_arrays": [], "note": "par2: regular snapshots deleted, crossing CSVs kept"}

    def blinded_sd(self, arm, reduced):
        """Blinded SDs for the engine's re-estimation: ``({estimator: sd}, n)``.  ``reduced`` is the list of reduce-result dicts of
        the arm's reduced segments (``info["tables"]``; plain table paths are accepted too).  Rows: ``code`` = the arm's runner, plane
        family, window [25, 94], the production pass (consumer ``hmax_div`` 8, or -1 for a single-pass root; Par2 -1), basis ``disp`` (H2), flux (consumer) / uniform
        (Par2) weights, set ``perplane`` (``reduce.blind`` overrides).  Estimators ``alpha_TP``, ``alpha_TAP``, ``alpha_LP``: the sample
        SD (ddof 1) of the finite values, one per seed, for estimators with at least 4 values; ``n`` = the smallest such count.  No
        location statistic is read into the result (mcstat.blinded)."""
        cons = arm["runner"] == "consumer"
        blind = arm.get("reduce", {}).get("blind", {})
        sel = {"weights": "flux" if cons else "unif", "basis": "disp", "set": "perplane"}
        sel.update({k: blind[k] for k in ("weights", "basis", "set") if k in blind})
        lo, hi = (float(v) for v in blind.get("window", (25.0, 94.0)))
        hms = {int(blind["hmax_div"])} if "hmax_div" in blind else {8, -1} if cons else {-1}  # -1: a single-pass root
        vals = {e: {} for e in BLIND_EST}
        for info in reduced:
            for t in (info["tables"] if isinstance(info, dict) else [info]):
                with open(t, newline="") as f:
                    for row in csv.DictReader(f):
                        if (row["code"] != arm["runner"] or row["family"] != "plane" or any(row[k] != v for k, v in sel.items())
                                or float(row["d_lo"]) != lo or float(row["d_hi"]) != hi or int(float(row["hmax_div"])) not in hms):
                            continue
                        key = int(float(row["seed"])) if row.get("seed", "") not in ("", "nan") else (t, row["r"])
                        for e in BLIND_EST:
                            v = float(row[e]) if row[e] != "" else float("nan")
                            if math.isfinite(v):
                                vals[e][key] = v
        usable = {e: list(v.values()) for e, v in vals.items() if len(v) >= 4}
        sd = _blinded.blinded_sds(usable)
        return sd, (min(len(v) for v in usable.values()) if usable else max((len(v) for v in vals.values()), default=0))

    def expected_raw_gb(self, arm, n_seeds):
        return arm["raw_gb"] * n_seeds
