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
``fused_splines``/``single_launch`` (default true).  Strings are passed as they are, numbers with ``repr``.

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
``weights``/``bases`` (informational: the SF-42 reducer always writes every variant; they only pick the rows ``blinded_sd`` reads,
defaults consumer flux/pos, Par2 unif/pos, set ``perplane``, via optional ``blind`` = {weights, basis, set}); time-window secondary
(optional): ``U_time`` and ``par2_exit_x`` (Par2 needs both and ``snapshots_every`` > 0).
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
_CONVERGED = re.compile(r"Converged=(YES|NO)\b")


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


class RunnerExecutor:
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
                    "--hmax-div", ",".join(str(int(d)) for d in c.get("hmax_div", [8]))]
            argv += (["--fused-splines"] if c.get("fused_splines", True) else []) + (["--single-launch"] if c.get("single_launch", True) else [])
            argv += ["--twin", onoff(c.get("twin", True)), "--pollock-bridge", onoff(c.get("pollock_bridge", True)),
                     "--seed0", str(s0), "--nr", str(nr), "--out", os.path.join(segment_dir, "out")]
            files = {}
            meta["n_planes"] = len(planes)
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
        """(meta, {r: Converged flag}, {r: crossing CSV path}) of the complete realizations of a Par2 segment."""
        meta = _read_json(os.path.join(segment_dir, "segment.json"))
        if not meta:
            return None, {}, {}
        conv = {i: m == "YES" for i, m in enumerate(_CONVERGED.findall(_read_text(os.path.join(segment_dir, LOG))))}
        csvs = {}
        for r in range(meta["nr"]):
            path = os.path.join(segment_dir, "out", "crossings", "realization_%04d_planes.csv" % r)
            lines = [ln for ln in _read_text(path).splitlines() if ln.strip() and not ln.startswith("#")]
            if lines and _read_text(path).startswith(PAR2_FORMAT) and len(lines) - 1 == meta["n_planes"]:
                csvs[r] = path
        return meta, conv, csvs

    def completed(self, arm, segment_dir):
        """Seeds with complete outputs.  Par2: a realization whose head solve did not converge is NOT complete (unusable_solve)."""
        if arm["runner"] == "consumer":
            return sorted(self._consumer_dirs(segment_dir))
        meta, conv, csvs = self._par2_state(segment_dir)
        return [] if meta is None else [meta["seed0"] + r for r in sorted(csvs) if conv.get(r) is True]

    # ---- classification --------------------------------------------------------------------------------------------------------
    def classify(self, arm, seeds, segment_dir, exit_code, log_text):
        s0, nr = _seed0(seeds)
        done = [s for s in self.completed(arm, segment_dir) if s0 <= s < s0 + nr]
        verdicts, seconds = {}, {}
        failed, note = None, ""
        if arm["runner"] == "consumer":
            for r, v in re.findall(r"^R=(\d+) STAGE medium=b status=(\w+)", log_text, re.M):
                verdicts[s0 + int(r)] = v
            for r, t in re.findall(r"^TIMING r=(\d+) seconds=([0-9.eE+-]+)", log_text, re.M):
                seconds[s0 + int(r)] = float(t)
            for s, d in self._consumer_dirs(segment_dir).items():
                sv = _read_json(os.path.join(d, "solver_b.json"))
                if sv and "verdict" in sv:
                    verdicts[s] = sv["verdict"]
            sf = re.findall(r"^R=(\d+) STAGE_FAILED .*?verdict=(\w+)", log_text, re.M)
            if sf:
                failed = s0 + int(sf[-1][0])
                verdicts[failed] = sf[-1][1]
        else:
            meta, conv, csvs = self._par2_state(segment_dir)
            for r, ok in conv.items():
                verdicts[s0 + r] = "converged" if ok else "not_converged"
            prev = os.path.getmtime(os.path.join(segment_dir, "segment.json")) if meta else None
            for r in sorted(csvs):  # wall time between successive crossing CSVs (r = 0: from the unit's launch files); approximate
                t = os.path.getmtime(csvs[r])
                seconds[s0 + r] = max(0.0, t - prev)
                prev = t
            bad = [s0 + r for r, ok in sorted(conv.items()) if not ok]
            if bad:
                failed = bad[0]
                note = "Converged=NO for seeds %s" % bad
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
            failed = None  # a Converged=NO seed is a failure of the unit only if nothing else stopped it
        missing = [s for s in range(s0, s0 + nr) if s not in done]  # the runners go in seed order: the first missing seed is the failing one
        failed = None if cls == "ok" else failed if failed is not None else missing[0] if missing else None
        out = {"cls": cls, "completed_seeds": done, "failed_seed": failed, "verdicts": verdicts, "seconds": seconds}
        if note:
            out["note"] = note
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
        else:
            meta, _, csvs = self._par2_state(segment_dir)
            par2 = [csvs[s - meta["seed0"]] for s in done]
            if int(run.get("snapshots_every", 0)) > 0 and "U_time" in rd and "par2_exit_x" in rd:
                series = [os.path.join(segment_dir, "out", "stats", "realization_%04d_timeseries.csv" % (s - meta["seed0"])) for s in done]
                kw = {"par2_exit_x": rd["par2_exit_x"], "par2_dt": meta["dt"]}
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
        """Delete the per-particle .npy of the realizations not in ``retained_seeds`` (an ARM-level list; entries outside this
        segment are ignored, the arm-level minimum is the caller's).  Verification and the table coverage are the SF-42 pruner's; the
        manifest of the FIRST table is checked (all tables of a segment record the same inputs).  Par2 has no per-particle arrays."""
        if arm["runner"] != "consumer":
            return {"deleted_files": 0, "deleted_bytes": 0, "retained": [], "log": None, "note": "par2 has no per-particle arrays"}
        keep = set(int(s) for s in retained_seeds)
        dirs = self._consumer_dirs(segment_dir)
        kept = sorted(s for s in dirs if s in keep)
        retained = os.path.join(segment_dir, "retained.txt")
        with open(retained, "w") as f:
            f.write("# arm-level retained seeds present in this segment\n")
            for s in kept:
                f.write(os.path.relpath(dirs[s], segment_dir) + "\n")
        log = os.path.join(segment_dir, "prune_log.csv")
        _prune.prune([os.path.join(segment_dir, "out")], retained, reduced["tables"][0], segment_dir, True, log, out=lambda s: None,
                     min_retained=0)
        with open(log, newline="") as f:
            rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
        gone = [r for r in rows if r["action"] == "delete"]
        return {"deleted_files": len(gone), "deleted_bytes": sum(int(r["size"]) for r in gone), "retained": kept, "log": log,
                "log_sha256": _red.sha256_file(log)}

    def blinded_sd(self, arm, reduced_tables):
        """Standard deviations (ddof 1) of the per-realization plane estimators over all given tables, per (d window, hmax pass,
        estimator), plus the numbers of values.  No location statistic is read into the result."""
        sel = {"weights": "unif" if arm["runner"] == "par2" else "flux", "basis": "pos", "set": "perplane"}
        sel.update(arm.get("reduce", {}).get("blind", {}))
        groups = {}
        for t in reduced_tables:
            with open(t, newline="") as f:
                for row in csv.DictReader(f):
                    if row["code"] != arm["runner"] or row["family"] != "plane" or any(row[k] != v for k, v in sel.items()):
                        continue
                    for est in _red.PLANE_EST:
                        v = float(row[est]) if row[est] != "" else float("nan")
                        if math.isfinite(v):
                            groups.setdefault("d%g-%g|hmax%s|%s" % (float(row["d_lo"]), float(row["d_hi"]), row["hmax_div"], est), []).append(v)
        usable = {k: v for k, v in groups.items() if len(v) >= 4}
        res = _blinded.blinded_sds(usable)
        return {"sd": res, "n_obs": {k: len(v) for k, v in groups.items()}}

    def expected_raw_gb(self, arm, n_seeds):
        return arm["raw_gb"] * n_seeds
