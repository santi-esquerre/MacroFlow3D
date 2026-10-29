"""SF-39 checkpoint part (contract C-GLUE, preregistration sections 4 C, 6, 7 RR4): freeze, sealed pilot estimates, `nr_rule`, blinded report.

    python3 -m sf39.checkpoint --out-root $OUT --artifact A
    python3 -m sf39.checkpoint --out-root $OUT --artifact A --amps s025 --subdir s025      (the separate s025 case, see below)

Steps (all under `$OUT/checkpoint/`):
  1. completeness: every pilot runner arm of the amplitudes of the case (5 per amplitude) has realizations r = 0..19 DONE (+ `timings.json`, + the three passes in directory A),
     the Par2 pilot has 20 series, snapshots and FIELDSTAT lines per amplitude, `pilot/walls.tsv` has the ids needed; else exit 2
     listing what is missing (no partial checkpoint).
  2. freeze (write-once): `tau_b_star.json` = per amplitude t_b,r (h/8 pass), t_b,min, U_prod, ell, tau_b* = U_prod t_b,min/ell and the
     Gate 5 frozen times (`campaign.gate5_frozen`).  An existing file with different content -> exit 2, never overwritten.
  3. sealed (`sealed/`): `estimates.csv` (per arm, pass and realization: alpha_TAM, alpha_TP, alpha_LAM, flux weights, W from the
     frozen tau_b*, U by the rules of `campaign`), `nr_input.json` (schema of `scripts/nr_rule.py`), `nr_out.json`, `nr_stdout.txt`
     (nr_rule run as a subprocess).
  4. `costs.json` ({arm: GPU-s per realization}, the key format of `sf39.power`), `bytes.json` ({arm: bytes per realization}).
  5. projections with the nr_rule N values.
  6. `checkpoint_report.txt` / `.json`, also printed: tau_b* per amplitude, N values, GPU-s, projected GPU-h/wall/bytes.  BLINDING (RR4):
     no plume statistic (no mean, spread, margin, verdict, alpha, qbar, U_c) is written to the report.

Cost bookkeeping (stated in the report):
  * GPU-s per realization of a pilot arm = average of `timings.json` seconds.total over its pilot realizations (measured).
  * directory B (single pass) per realization = directory-A total minus prepare+track of the hmax_16 and hmax_32 passes (estimate);
    `s4_w24` = estimated s4 directory B x (cells of s4_w24)/(cells of s4_B) (estimate); `s4_w12` = measured `s4_tw` (same geometry).
  * Par2 per realization = interval between consecutive series files of the pilot (file times, realizations 1..n-1; fallback: the
    walls.tsv wall of `par2_s{c}` / n) + the wall of `fs_par2_s{c}` / n.
  * wall/GPU-busy ratio = sum of the pilot walls of the runner arms in `pilot/walls.tsv` / sum of their timings.json seconds.total
    (Par2 has no separate busy figure); projected wall = projected GPU-hours x ratio.
Bytes per realization = size of the arm's `realizations/r_*` directories / count (Par2: size of `raw/par2_s{c}` / n); directory B
bytes = A minus the hmax_16 and hmax_32 pass directories (estimate); `s4_w24` bytes = the `s4_B` estimate (same particles/planes,
shorter box: an upper bound); `s4_w12` = measured `s4_tw`.

The s025 case (addendum 2026-10-29, A5): without options the checkpoint is exactly the one of the three original amplitudes
(`BASE_AMPS`, never a four-amplitude freeze).  `--amps s025 --subdir s025` runs the same steps for s025 alone and writes EVERYTHING under
`checkpoint/s025/`: the write-once `tau_b_star_s025.json` (same schema and content rules), `sealed/`, `costs.json`, `bytes.json`,
`checkpoint_report.txt/.json`.  The files of the original case are never touched; the s4-only width cost/byte lines apply only when
s4 is in the case.  The cases refuse to mix (`Refused`, exit 2): no `--subdir` needs the three original amplitudes, a `--subdir`
needs exactly one amplitude, named like it, outside them.

Exit codes: 0 done, 2 refused (incomplete pilot, or write-once conflict), 1 failure (window/data error, nr_rule failure).
The pilot count is 20; `run(..., n_pilot=)` exists for tests only (the CLI has no option for it).
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

from . import campaign as cp
from . import io as sio
from . import par2, power
from . import reduce as red

N_PILOT = 20
BASE_AMPS = ("s1", "s225", "s4")  # the amplitudes of checkpoint/tau_b_star.json (write-once); s025 is a separate case (checkpoint/s025/)
NR_LINE = re.compile(r"^s\d+ [A-Za-z0-9_]+=(>500|\d+)$")


class Refused(Exception):
    """The checkpoint refuses to proceed (exit 2)."""


# ------------------------------------------------------------------------------------------------
# bookkeeping files
# ------------------------------------------------------------------------------------------------
def read_walls(out_root):
    """{id: wall_s} of `$OUT/pilot/walls.tsv` (`id<TAB>wall_s<TAB>end_utc`; the last record of an id wins); {} if absent."""
    path = os.path.join(out_root, "pilot", "walls.tsv")
    out = {}
    try:
        with open(path) as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0]:
                    out[p[0]] = float(p[1])
    except FileNotFoundError:
        pass
    return out


def dir_bytes(path):
    """Total size in bytes of the regular files under a directory."""
    tot = 0
    for root, _, files in os.walk(path):
        for fn in files:
            tot += os.path.getsize(os.path.join(root, fn))
    return tot


# ------------------------------------------------------------------------------------------------
# 1. completeness
# ------------------------------------------------------------------------------------------------
def missing_items(out_root, artifact, n_pilot, amps=BASE_AMPS):
    """List of strings describing everything the checkpoint of `amps` needs and cannot find (empty list = complete)."""
    miss = []
    walls = read_walls(out_root)
    for arm in cp.all_pilot_arms(amps):
        recs = {x.r: x for x in cp.done_records(out_root, arm)}
        absent = [r for r in range(n_pilot) if r not in recs]
        if absent:
            miss.append("%s: realizations not DONE: %s" % (arm, ",".join(str(r) for r in absent)))
        for r in range(n_pilot):
            if r not in recs:
                continue
            x = recs[r]
            if not os.path.exists(os.path.join(x.rdir, "timings.json")):
                miss.append("%s: r_%04d has no timings.json" % (arm, r))
            if cp.is_multipass(arm):
                for d in cp.HMAX_DIVS:
                    if not os.path.isdir(os.path.join(x.rdir, "hmax_%d" % d)):
                        miss.append("%s: r_%04d has no hmax_%d pass" % (arm, r, d))
        if cp.pilot_walls_id(arm, n_pilot) not in walls:
            miss.append("pilot/walls.tsv: no record for %s" % cp.pilot_walls_id(arm, n_pilot))
    for amp in amps:
        raw, cfg, fsf = cp.par2_paths(out_root, artifact, amp)
        c = cp.PAR2_ID[amp]
        if not os.path.isfile(cfg):
            miss.append("Par2 config %s" % cfg)
        bad_s = [r for r in range(n_pilot) if not os.path.isfile(os.path.join(raw, "stats", "realization_%04d_timeseries.csv" % r))]
        if bad_s:
            miss.append("par2_s%s: series missing for r = %s" % (c, ",".join(str(r) for r in bad_s)))
        bad_p = []
        for r in range(n_pilot):
            d = os.path.join(raw, "snapshots", "r_%04d" % r)
            if not (os.path.isdir(d) and any(n.startswith("step_") for n in os.listdir(d))):
                bad_p.append(r)
        if bad_p:
            miss.append("par2_s%s: snapshots missing for r = %s" % (c, ",".join(str(r) for r in bad_p)))
        try:
            with open(fsf) as f:
                n_fs = len(par2.parse_fieldstat(f.read()))
            if n_fs != n_pilot:
                miss.append("%s: %d FIELDSTAT lines, expected %d" % (fsf, n_fs, n_pilot))
        except FileNotFoundError:
            miss.append("FIELDSTAT file %s" % fsf)
        for i in ("par2_s" + c, "fs_par2_s" + c):
            if i not in walls:
                miss.append("pilot/walls.tsv: no record for %s" % i)
    return miss


# ------------------------------------------------------------------------------------------------
# 2. freeze
# ------------------------------------------------------------------------------------------------
def freeze(out_root, artifact, n_pilot, amps=BASE_AMPS):
    """The content of the frozen file (`tau_b_star.json`, or `tau_b_star_<case>.json`) for `amps` (pure computation from the pilot files)."""
    entries = {}
    for amp in amps:
        recs = cp.pilot_records(out_root, cp.arm_ids(amp)["A"], n_pilot)
        tb = cp.breakthrough_times(recs)
        tbm = float(min(tb))
        u_prod = cp.u_standalone(recs)
        ell = cp.realization(recs[0]).ell
        entry = {"seeds": [x.seed for x in recs], "t_b_r": [float(t) for t in tb], "t_b_min": tbm, "U_prod": u_prod,
                 "ell": float(ell), "tau_b_star": u_prod * tbm / ell}
        entry["gate5"] = cp.gate5_frozen(out_root, artifact, amp, recs, tbm, ell, n_pilot)
        entries[amp] = entry
    return {"schema": "sf39-tau_b_star-v1", "n_pilot": n_pilot, "amplitudes": entries}


def dumps(obj):
    """Deterministic JSON text."""
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def write_once(path, text):
    """Write `text` to `path` unless it exists: identical content passes, different content raises `Refused` (never overwrites)."""
    if os.path.exists(path):
        with open(path) as f:
            old = f.read()
        if old != text:
            raise Refused("%s exists with different content (write-once freeze: never overwritten)" % path)
        return "identical"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return "written"


# ------------------------------------------------------------------------------------------------
# 3. sealed pilot estimates and nr_rule input
# ------------------------------------------------------------------------------------------------
def sealed_estimates(out_root, frz, n_pilot, amps=BASE_AMPS):
    """(rows, nr_input): per amplitude and arm the estimates of every pilot realization (production h/8 and h/16 passes, three narrow
    rungs, twin; flux weights; U = mean qbar of the arm's own pilot seeds, the pilot being the pair set) and the nr_rule input."""
    rows, nr_in = [], {}
    for amp in amps:
        tau = frz["amplitudes"][amp]["tau_b_star"]
        a = cp.arm_ids(amp)
        recs = {arm: cp.pilot_records(out_root, arm, n_pilot) for arm in cp.pilot_arms(amp)}
        est = {}
        for arm, rr in recs.items():
            U = cp.u_paired(recs[a["A"]], rr)[1] if arm != a["A"] else cp.u_standalone(rr)
            divs = (8, 16) if cp.is_multipass(arm) else (cp.PROD_DIV,)
            for d in divs:
                # every pass of a production realization uses the production-side U (the A arm's own pilot mean)
                est[(arm, d)] = cp.reduce_arm(rr, d, tau, U)
                for x in rr:
                    rows.append(dict(arm=arm, hmax_div=d, r=x.r, seed=x.seed, qbar=x.qbar, U=U, tau_b_star=tau,
                                     **est[(arm, d)][x.seed]))
        n05, n10, n20 = a["narrow"]
        nr_amp = {}
        for key in power.ESTIMATORS:
            def diff(p, q):
                seeds, u, v = cp.paired(est[p], est[q], key)
                if len(seeds) != n_pilot:
                    raise sio.ReduceError("%s vs %s: only %d common seeds" % (p, q, len(seeds)))
                return [float(t) for t in (u - v)]
            prod = [est[(a["A"], 8)][s][key] for s in sorted(est[(a["A"], 8)])]
            nr_amp[key] = {
                "D_f": diff((n20, 8), (n10, 8)),
                "D_tw": diff((a["A"], 8), (a["tw"], 8)),
                "d16_8": diff((a["A"], 16), (a["A"], 8)),
                "prod": [float(v) for v in prod],
                "mean_2x": float(np.mean([v[key] for v in est[(n20, 8)].values()])),
                "mean_h16": float(np.mean([v[key] for v in est[(a["A"], 16)].values()])),
            }
        nr_in[amp] = nr_amp
    return rows, nr_in


def run_nr_rule(artifact, nr_input_path, nr_out_path):
    """Run `A/scripts/nr_rule.py` as a subprocess; returns its stdout lines (checked to be N lines only: blinding)."""
    script = os.path.join(artifact, "scripts", "nr_rule.py")
    env = dict(os.environ)
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the directory that holds `sf39`
    env["PYTHONPATH"] = pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run([sys.executable, script, nr_input_path, "--out", nr_out_path], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError("nr_rule.py failed (rc %d): %s" % (p.returncode, p.stderr.strip()[-400:]))
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    bad = [ln for ln in lines if not NR_LINE.match(ln)]
    if bad:
        raise RuntimeError("nr_rule.py printed something other than N lines: %r" % bad[:3])
    return lines


# ------------------------------------------------------------------------------------------------
# 4. costs and bytes
# ------------------------------------------------------------------------------------------------
def _timings(x):
    return sio.read_json(os.path.join(x.rdir, "timings.json"))


def _par2_cost(out_root, amp, n_pilot, walls):
    """(GPU-s per realization, source text) of the Par2 pilot + par2_field_stats (see the module docstring)."""
    c = cp.PAR2_ID[amp]
    raw = os.path.join(out_root, "par2", "raw", "par2_s" + c)
    iv = [dt for _, _, dt in par2.wall_times(raw) if dt is not None]
    if iv and float(np.mean(iv)) > 0.0:
        per, src = float(np.mean(iv)), "series file times"
    else:
        per, src = walls["par2_s" + c] / n_pilot, "walls.tsv par2_s%s / %d" % (c, n_pilot)
    return per + walls["fs_par2_s" + c] / n_pilot, "%s + fs wall / %d" % (src, n_pilot)


def costs_and_bytes(out_root, n_pilot, amps=BASE_AMPS):
    """(costs, kinds, nbytes, byte_kinds, ratio_parts): {arm: GPU-s}, {arm: 'measured'|'estimate'}, {arm: bytes}, kinds, and
    (sum of runner walls, sum of runner timings totals) for the wall/GPU-busy ratio."""
    walls = read_walls(out_root)
    costs, kinds, nbytes, bkinds = {}, {}, {}, {}
    wall_sum = busy_sum = 0.0
    for amp in amps:
        a = cp.arm_ids(amp)
        for arm in cp.pilot_arms(amp):
            recs = cp.pilot_records(out_root, arm, n_pilot)
            tot = [_timings(x)["seconds"]["total"] for x in recs]
            costs[arm], kinds[arm] = float(np.mean(tot)), "measured"
            wall_sum += walls[cp.pilot_walls_id(arm, n_pilot)]
            busy_sum += float(np.sum(tot))
            nbytes[arm] = int(round(sum(dir_bytes(x.rdir) for x in recs) / len(recs)))
            bkinds[arm] = "measured"
            if arm == a["A"]:
                excess_t, excess_b = [], []
                for x in recs:
                    t = _timings(x)
                    ex = sum(p["prepare"] + p["track"] for p in t["passes"] if p["hmax_div"] in (16, 32))
                    excess_t.append(t["seconds"]["total"] - ex)
                    excess_b.append(dir_bytes(x.rdir) - sum(dir_bytes(os.path.join(x.rdir, "hmax_%d" % d)) for d in (16, 32)))
                costs[a["B"]], kinds[a["B"]] = float(np.mean(excess_t)), "estimate"
                nbytes[a["B"]], bkinds[a["B"]] = int(round(float(np.mean(excess_b)))), "estimate"
        c2, _ = _par2_cost(out_root, amp, n_pilot, walls)
        costs[amp + "_par2"], kinds[amp + "_par2"] = c2, "measured"
        raw = os.path.join(out_root, "par2", "raw", "par2_s" + cp.PAR2_ID[amp])
        nbytes[amp + "_par2"], bkinds[amp + "_par2"] = int(round(dir_bytes(raw) / n_pilot)), "measured"
    if "s4" in amps:  # the width arms exist at s4 only
        ratio = cp.cell_count("s4_w24") / cp.cell_count("s4_B")
        costs["s4_w24"], kinds["s4_w24"] = costs["s4_B"] * ratio, "estimate"
        costs["s4_w12"], kinds["s4_w12"] = costs["s4_tw"], "measured"
        nbytes["s4_w24"], bkinds["s4_w24"] = nbytes["s4_B"], "estimate"
        nbytes["s4_w12"], bkinds["s4_w12"] = nbytes["s4_tw"], "measured"
    return costs, kinds, nbytes, bkinds, (wall_sum, busy_sum)


# ------------------------------------------------------------------------------------------------
# 5-6. projections and report
# ------------------------------------------------------------------------------------------------
def done_counts(n_pilot, amps=BASE_AMPS):
    """Realizations already run: n_pilot for every directory A, narrow, twin and Par2 arm; 0 (absent) for B and the width arms."""
    d = {}
    for amp in amps:
        for arm in cp.pilot_arms(amp) + (amp + "_par2",):
            d[arm] = n_pilot
    return d


def projections(nr_out, costs, nbytes, ratio_parts, n_pilot, amps=BASE_AMPS):
    """dict(projected_gpu_h, pilot_wall_over_gpu_busy, projected_wall_h, projected_bytes) from the nr_rule N values."""
    N = {amp: nr_out[amp] for amp in amps}
    gpu_h = power.projected_gpu_h(N, costs, done_counts(n_pilot, amps))
    ratio = ratio_parts[0] / ratio_parts[1]
    nb = 0
    for amp in amps:
        for arm, cnt in power.arm_counts(N[amp], amp).items():
            nb += cnt * nbytes[arm]
    return {"projected_gpu_h": gpu_h, "pilot_wall_over_gpu_busy": ratio, "projected_wall_h": gpu_h * ratio, "projected_bytes": int(nb)}


def build_report(frz, nr_lines, nr_out, costs, kinds, nbytes, bkinds, proj, n_pilot, amps=BASE_AMPS, frz_name="tau_b_star.json"):
    """(text, json dict) of the blinded report; `frz_name` is the frozen file of the case (named in the report)."""
    tau = {amp: frz["amplitudes"][amp]["tau_b_star"] for amp in amps}
    n_vals = {amp: {k: nr_out[amp][k] for k in ("N_narrow", "N_twin", "N_hmax", "N_R", "N_A", "N_run", "par2_NR")} for amp in amps}
    capped = {amp: list(nr_out[amp]["capped"]) for amp in amps}
    js = {"n_pilot": n_pilot, "tau_b_star": tau, "N": n_vals, "N_capped_at_500": capped,
          "gpu_s_per_realization": {k: {"value": costs[k], "kind": kinds[k]} for k in sorted(costs)},
          "bytes_per_realization": {k: {"value": nbytes[k], "kind": bkinds[k]} for k in sorted(nbytes)}}
    js.update(proj)
    t = ["SF-39 CHECKPOINT REPORT (blinded, RR4; pilot n=%d per arm%s)" % (n_pilot, "" if tuple(amps) == BASE_AMPS else "; amplitudes: " + ",".join(amps)),
         "",
         "FROZEN tau_b* (F2 pilot record; the other frozen times are in %s)" % frz_name]
    t += ["TAU_B_STAR %s=%.12g" % (amp, tau[amp]) for amp in amps]
    t += ["", "N values printed by nr_rule (>500: cap reached without meeting the target, 500 used)"] + ["NR %s" % ln for ln in nr_lines]
    t += ["", "GPU-seconds per realization (kind: measured | estimate)"]
    t += ["GPU_S %s %.6g %s" % (k, costs[k], kinds[k]) for k in sorted(costs)]
    t += ["", "PROJECTED_GPU_H %.6g (realizations still to run, N values above, GPU-s per realization above)" % proj["projected_gpu_h"],
          "PILOT_WALL_OVER_GPU_BUSY %.6g (sum of pilot runner walls / sum of pilot timings.json seconds.total)" % proj["pilot_wall_over_gpu_busy"],
          "PROJECTED_WALL_H %.6g (one shared GPU: projected GPU-h x that ratio)" % proj["projected_wall_h"],
          "", "BYTES per realization (kind: measured | estimate)"]
    t += ["BYTES %s %d %s" % (k, nbytes[k], bkinds[k]) for k in sorted(nbytes)]
    t += ["PROJECTED_BYTES %d (final counts x bytes per realization, all arms)" % proj["projected_bytes"], ""]
    return "\n".join(t), js


# ------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------
def check_case(amps, subdir):
    """Refuse a mix of the cases (`Refused`): no subdir = exactly `BASE_AMPS`; a subdir = exactly one amplitude, named like the subdir,
    outside `BASE_AMPS`."""
    if subdir and not re.fullmatch(r"[A-Za-z0-9_]+", subdir):
        raise Refused("--subdir %r is not a plain directory name" % subdir)
    bad = [a for a in amps if a not in cp.AMPS]
    if bad:
        raise Refused("unknown amplitude(s) %s (known: %s)" % (bad, ",".join(cp.AMPS)))
    if not subdir and tuple(amps) != BASE_AMPS:
        raise Refused("without --subdir the checkpoint is the original case %s; got --amps %s" % (",".join(BASE_AMPS), ",".join(amps)))
    if subdir and tuple(amps) != (subdir,):
        raise Refused("--subdir %s needs exactly --amps %s; got --amps %s" % (subdir, subdir, ",".join(amps)))
    if subdir and subdir in BASE_AMPS:
        raise Refused("--subdir %s is an amplitude of the original case %s" % (subdir, ",".join(BASE_AMPS)))


def run(out_root, artifact, n_pilot=N_PILOT, out=None, amps=BASE_AMPS, subdir=""):
    """Run the checkpoint; returns the exit code.  `n_pilot` is a test hook (the CLI always uses 20).  `amps` / `subdir`: the original
    case (defaults; files directly under `checkpoint/`) or a separate case (e.g. s025 in `checkpoint/s025/`, frozen file
    `tau_b_star_s025.json`), see the module docstring."""
    out = out or sys.stdout
    amps = tuple(amps)
    try:
        check_case(amps, subdir)
    except Refused as e:
        print("CHECKPOINT_REFUSED: %s" % e, file=sys.stderr)
        return 2
    miss = missing_items(out_root, artifact, n_pilot, amps)
    if miss:
        print("CHECKPOINT_INCOMPLETE: %d item(s) missing; no partial checkpoint is written" % len(miss), file=sys.stderr)
        for m in miss:
            print("  MISSING " + m, file=sys.stderr)
        return 2
    ck = os.path.join(out_root, "checkpoint", subdir)
    frz_name = "tau_b_star_%s.json" % subdir if subdir else "tau_b_star.json"
    os.makedirs(ck, exist_ok=True)
    try:
        frz = freeze(out_root, artifact, n_pilot, amps)
        state = write_once(os.path.join(ck, frz_name), dumps(frz))
    except Refused as e:
        print("CHECKPOINT_REFUSED: %s" % e, file=sys.stderr)
        return 2
    print("FREEZE %s %s" % (frz_name, state), file=sys.stderr)
    sealed = os.path.join(ck, "sealed")
    os.makedirs(sealed, exist_ok=True)
    rows, nr_in = sealed_estimates(out_root, frz, n_pilot, amps)
    red.write_csv(os.path.join(sealed, "estimates.csv"), rows)
    nr_in_path, nr_out_path = os.path.join(sealed, "nr_input.json"), os.path.join(sealed, "nr_out.json")
    with open(nr_in_path, "w") as f:
        f.write(dumps(nr_in))
    nr_lines = run_nr_rule(artifact, nr_in_path, nr_out_path)
    with open(os.path.join(sealed, "nr_stdout.txt"), "w") as f:
        f.write("\n".join(nr_lines) + "\n")
    with open(nr_out_path) as f:
        nr_out = json.load(f)
    costs, kinds, nbytes, bkinds, ratio_parts = costs_and_bytes(out_root, n_pilot, amps)
    for name, obj in (("costs.json", costs), ("bytes.json", nbytes)):
        with open(os.path.join(ck, name), "w") as f:
            f.write(dumps(obj))
    proj = projections(nr_out, costs, nbytes, ratio_parts, n_pilot, amps)
    text, js = build_report(frz, nr_lines, nr_out, costs, kinds, nbytes, bkinds, proj, n_pilot, amps, frz_name)
    with open(os.path.join(ck, "checkpoint_report.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(ck, "checkpoint_report.json"), "w") as f:
        f.write(dumps(js))
    out.write(text)
    print("CHECKPOINT_DONE", file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m sf39.checkpoint", description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", required=True, help="$SF39_OUT_ROOT")
    ap.add_argument("--artifact", required=True, help="the artifact root A (configs/, scripts/nr_rule.py)")
    ap.add_argument("--amps", default=",".join(BASE_AMPS), help="comma-separated amplitudes of the case (default: the original three)")
    ap.add_argument("--subdir", default="", help="separate case: write everything under checkpoint/<subdir>/ (e.g. --amps s025 --subdir s025)")
    a = ap.parse_args(argv)
    try:
        return run(a.out_root, a.artifact, amps=tuple(x for x in a.amps.split(",") if x), subdir=a.subdir)
    except (sio.ReduceError, ValueError, RuntimeError, OSError) as e:
        print("CHECKPOINT_FAILED %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
