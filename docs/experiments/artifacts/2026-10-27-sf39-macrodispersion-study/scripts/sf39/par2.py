"""C-PAR2 (preregistration sections 1, 4, 8 R8-R15): Par2 RWPT baseline reducer for SF-39 Gate 5.

Readers of the Par2 pipeline outputs (series `stats/realization_%04d_timeseries.csv`, snapshots
`snapshots/r_%04d/step_%08d.csv`), the Par2 breakthrough time t_b (LAST snapshot with no particle at x >= Lx - 5 lambda),
the centroid-slope velocity `U_c`, the common-window per-realization `alpha`, the FIELDSTAT parser, per-realization wall
times and the generators of the pilot / extension configs.  The series functions are format-generic (series, t, anchor) and
serve the Lagrangian twin series (`stats_twin/realization_%04d_timeseries.csv`, same CSV format) as well.

Pooled index: g = r in the pilot directory, 20 + r in the extension directory (R15).  numpy + stdlib only.

CLI (`python3 -m sf39.par2`): tb | uc | alpha | fieldstat | walls | mkpilot | mkext  (see `--help` of each).
"""
import argparse
import glob
import io
import os
import re
import sys

import numpy as np

N_PILOT = 20
FIELD_SEED_PILOT = 9001
FIELD_SEED_EXT = 9021
TRANSPORT_SEED_PILOT = 123456789
TRANSPORT_SEED_EXT = TRANSPORT_SEED_PILOT + 1000 * N_PILOT  # 123476789
INJ_STRIDE = 1000  # injection seed = transport.seed + 1000 r (pipeline)
AMPLITUDES = ("1", "225", "4")
WINDOW_FRAC = 0.75
SERIES_COLS = ("t", "mean_x", "mean_y", "mean_z", "var_x", "var_y", "var_z")


# ------------------------------------------------------------------ config (minimal YAML subset: nested `key:` blocks, scalar values)
_KV = re.compile(r"^(\s*)([A-Za-z_]\w*):[ \t]*(.*?)\s*$")


def _scan(text):
    """Yield (line_index, dotted_path, raw_value) for every scalar `key: value` line (comments stripped, section lines skipped)."""
    stack = []  # (indent, key)
    for i, line in enumerate(text.split("\n")):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KV.match(re.sub(r"\s+#.*$", "", line))
        if not m:
            continue
        indent, key, val = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not val:
            stack.append((indent, key))
        else:
            yield i, ".".join([k for _, k in stack] + [key]), val


def parse_cfg(text):
    """dotted path -> raw scalar string; a repeated path is an error."""
    out = {}
    for _, path, val in _scan(text):
        if path in out:
            raise ValueError("duplicate config key %s" % path)
        out[path] = val
    return out


def cfg_values(text):
    """The keys the reducer needs: nx, dx, Lx, lam (corr_length), dt, inject_x, sigma2, seeds, NR, output_dir, snapshots_every."""
    c = parse_cfg(text)

    def g(path, cast):
        if path not in c:
            raise ValueError("config key not found: %s" % path)
        return cast(c[path])

    v = {"nx": g("grid.nx", int), "dx": g("grid.dx", float), "lam": g("stochastic.corr_length", float),
         "dt": g("transport.dt", float), "inject_x": g("transport.inject_x", float), "sigma2": g("stochastic.sigma2", float),
         "field_seed": g("stochastic.seed", int), "transport_seed": g("transport.seed", int),
         "nr": g("analysis.macrodispersion.NR", int), "output_dir": c.get("output.output_dir"),
         "snapshots_every": g("analysis.snapshots.every", int), "sample_every": g("analysis.macrodispersion.sample_every", int)}
    v["Lx"] = v["nx"] * v["dx"]
    return v


def set_cfg_key(text, path, value):
    """Replace the scalar value of `path` (exactly one occurrence), keeping indentation and trailing comment."""
    hits = [(i, p) for i, p, _ in _scan(text) if p == path]
    if len(hits) != 1:
        raise ValueError("%s: expected exactly one occurrence, found %d" % (path, len(hits)))
    lines = text.split("\n")
    i = hits[0][0]
    m = re.match(r"^(\s*[A-Za-z_]\w*:[ \t]*)(\S+)(.*)$", lines[i])
    if not m:
        raise ValueError("%s: cannot rewrite line %r" % (path, lines[i]))
    lines[i] = m.group(1) + str(value) + m.group(3)
    return "\n".join(lines)


def _pilot_header(text, c):
    lines = text.split("\n")
    if not (len(lines) > 3 and lines[1].startswith("# MacroFlow3D") and "SF-38 (s%s)" % c in lines[1]):
        raise ValueError("unexpected SF-38 config header (expected a title line naming SF-38 (s%s))" % c)
    lines[1] = "# MacroFlow3D — Par2 RWPT baseline, SF-39 pilot (s%s)" % c
    note = [
        "# SF-39 (docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/preregistration.md, sec. 1): copia de",
        "# apps/config_pipeline_par2_sf38_s%s.yaml generada por `python3 -m sf39.par2 mkpilot`; solo cambian" % c,
        "# stochastic.seed (7001 -> %d), analysis.macrodispersion.NR (10 -> %d, piloto) y output.output_dir (-> raw/par2_s%s)." % (
            FIELD_SEED_PILOT, N_PILOT, c),
        "# El texto SF-38 que sigue (semillas 7001-7010, NR = 10, piloto N_R = 10) es heredado y queda superado por esas claves.",
        "#",
    ]
    return "\n".join(lines[:3] + note + lines[3:])


def make_pilot_config(sf38_cfg_text, c):
    """SF-39 pilot config: stochastic.seed 9001, NR 20, output_dir raw/par2_s{c}, updated header; everything else byte-identical."""
    c = str(c)
    if c not in AMPLITUDES:
        raise ValueError("c must be one of %s" % (AMPLITUDES,))
    t = _pilot_header(sf38_cfg_text, c)
    t = set_cfg_key(t, "stochastic.seed", FIELD_SEED_PILOT)
    t = set_cfg_key(t, "analysis.macrodispersion.NR", N_PILOT)
    return set_cfg_key(t, "output.output_dir", "raw/par2_s%s" % c)


def make_ext_config(pilot_cfg_text, n_r):
    """Extension config for global N_R = n_r: stochastic.seed 9021, transport.seed 123476789, NR n_r - 20, output_dir raw/par2_s{c}_ext;
    a derivation header is prepended; everything else byte-identical to the pilot copy."""
    if n_r <= N_PILOT:
        raise ValueError("extension needs n_r > %d (got %d)" % (N_PILOT, n_r))
    out = parse_cfg(pilot_cfg_text).get("output.output_dir", "")
    m = re.fullmatch(r"raw/par2_s(1|225|4)", out)
    if not m:
        raise ValueError("pilot output_dir %r is not raw/par2_s{1,225,4}" % out)
    c = m.group(1)
    t = set_cfg_key(pilot_cfg_text, "stochastic.seed", FIELD_SEED_EXT)
    t = set_cfg_key(t, "transport.seed", TRANSPORT_SEED_EXT)
    t = set_cfg_key(t, "analysis.macrodispersion.NR", n_r - N_PILOT)
    t = set_cfg_key(t, "output.output_dir", "raw/par2_s%s_ext" % c)
    head = ("# SF-39 Par2 EXTENSION (s%s), generated by `python3 -m sf39.par2 mkext` from the pilot copy: stochastic.seed %d, transport.seed %d "
            "(= %d + %d x %d), NR = %d - %d = %d, output_dir raw/par2_s%s_ext; global index g = %d + r (R15). Nothing else differs.\n"
            % (c, FIELD_SEED_EXT, TRANSPORT_SEED_EXT, TRANSPORT_SEED_PILOT, INJ_STRIDE, N_PILOT, n_r, N_PILOT, n_r - N_PILOT, c, N_PILOT))
    return head + t


def check_ext_consistency(pilot_cfg_text, ext_cfg_text):
    """Problems (list of str) if the extension is not the uninterrupted continuation of the pilot (R15); empty list = consistent."""
    p, e = cfg_values(pilot_cfg_text), cfg_values(ext_cfg_text)
    bad = []
    if e["field_seed"] != p["field_seed"] + p["nr"]:
        bad.append("field seed %d != pilot %d + NR %d" % (e["field_seed"], p["field_seed"], p["nr"]))
    if e["transport_seed"] != p["transport_seed"] + INJ_STRIDE * p["nr"]:
        bad.append("transport seed %d != pilot %d + %d x NR %d" % (e["transport_seed"], p["transport_seed"], INJ_STRIDE, p["nr"]))
    for k in ("nx", "dx", "lam", "dt", "inject_x", "sigma2", "snapshots_every", "sample_every"):
        if e[k] != p[k]:
            bad.append("%s differs: pilot %r ext %r" % (k, p[k], e[k]))
    return bad


# ------------------------------------------------------------------ series
def read_series(path):
    """Par2 / twin per-realization series CSV -> dict column -> float array (comment lines skipped, header required)."""
    with open(path) as f:
        rows = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if len(rows) < 2:
        raise ValueError("%s: no data rows" % path)
    head = [h.strip() for h in rows[0].split(",")]
    miss = [c for c in SERIES_COLS if c not in head]
    if miss:
        raise ValueError("%s: missing columns %s (header %s)" % (path, miss, head))
    data = np.loadtxt(io.StringIO("\n".join(rows[1:])), delimiter=",", ndmin=2)
    if data.shape[1] != len(head):
        raise ValueError("%s: %d columns vs header %d" % (path, data.shape[1], len(head)))
    s = {h: data[:, k] for k, h in enumerate(head)}
    if np.any(np.diff(s["t"]) <= 0):
        raise ValueError("%s: t not strictly increasing" % path)
    return s


def interp(series, col, t):
    """Linear interpolation of a series column at t; error (never clip) outside [t_first, t_last]."""
    ts = series["t"]
    tol = 1e-12 * max(1.0, abs(ts[-1]))
    if t < ts[0] - tol or t > ts[-1] + tol:
        raise ValueError("t=%.15g outside the series [%.15g, %.15g]" % (t, ts[0], ts[-1]))
    return float(np.interp(t, ts, series[col]))


def uc_mean(series_list, t_b, anchor):
    """U_c = [mean_x(t_b) - anchor] / t_b, mean_x interpolated linearly in each series, averaged over the series."""
    if not series_list:
        raise ValueError("no series")
    u = [(interp(s, "mean_x", t_b) - anchor) / t_b for s in series_list]
    return float(np.mean(u)), u


def tau_b_star(u_c, t_b_min, lam):
    return u_c * t_b_min / lam


def window(tau_c, lam, u_c):
    """W_c = [0.75, 1] x tau_c lam / U_c (arm's own time and length units)."""
    t_hi = tau_c * lam / u_c
    return WINDOW_FRAC * t_hi, t_hi


def alpha_window(series, lam, u_c, t_lo, t_hi):
    """Per-realization alpha_k = [V_k(t_hi) - V_k(t_lo)] / (2 lam U_c (t_hi - t_lo)), k = x, y, z (units of lam)."""
    if not t_hi > t_lo:
        raise ValueError("empty window")
    d = 2.0 * lam * u_c * (t_hi - t_lo)
    return {k: (interp(series, "var_" + k, t_hi) - interp(series, "var_" + k, t_lo)) / d for k in "xyz"}


# ------------------------------------------------------------------ pooled realizations
def _indices(d, sub="stats"):
    rs = sorted(int(re.search(r"realization_(\d+)_timeseries\.csv$", p).group(1))
                for p in glob.glob(os.path.join(d, sub, "realization_*_timeseries.csv")))
    if rs != list(range(len(rs))):
        raise ValueError("%s/%s: realization indices not contiguous from 0: %s" % (d, sub, rs))
    return rs


def pooled(pilot_dir, ext_dir=None, field_seed0=FIELD_SEED_PILOT, transport_seed0=TRANSPORT_SEED_PILOT, n_pilot=N_PILOT):
    """List of dict(g, r, dir, field_seed, injection_seed) over the pilot directory (g = r) and the extension (g = n_pilot + r).
    Seeds follow one uninterrupted launch: 9001 + g and 123456789 + 1000 g (R15)."""
    out = []
    for base, d in ((0, pilot_dir), (n_pilot, ext_dir)):
        if d is None:
            continue
        for r in _indices(d):
            g = base + r
            out.append({"g": g, "r": r, "dir": d, "field_seed": field_seed0 + g, "injection_seed": transport_seed0 + INJ_STRIDE * g})
    gs = [e["g"] for e in out]
    if gs != list(range(len(gs))):
        raise ValueError("pooled indices not contiguous (pilot incomplete before extension?): %s" % gs)
    return out


def series_of(entry, sub="stats"):
    return read_series(os.path.join(entry["dir"], sub, "realization_%04d_timeseries.csv" % entry["r"]))


# ------------------------------------------------------------------ t_b
def read_snapshot_x(path):
    """x coordinate column of a snapshot CSV (header `id,x coord,y coord,z coord`)."""
    with open(path) as f:
        head = [h.strip().lower() for h in f.readline().split(",")]
        if "x coord" not in head:
            raise ValueError("%s: no `x coord` column (header %s)" % (path, head))
        k = head.index("x coord")
        xs = [float(ln.split(",")[k]) for ln in f if ln.strip()]
    return np.array(xs)


def snapshot_steps(entry):
    d = os.path.join(entry["dir"], "snapshots", "r_%04d" % entry["r"])
    files = glob.glob(os.path.join(d, "step_*.csv"))
    if not files:
        raise ValueError("no snapshots in %s" % d)
    return sorted((int(re.search(r"step_(\d+)\.csv$", p).group(1)), p) for p in files)


def t_b_last_clean(entry, lx, lam, dt):
    """(step, t_b, n_snapshots, reached): last snapshot (step order, stop at the first one with a particle at x >= Lx - 5 lam) with
    none such; t_b = step dt.  `reached` False if no snapshot has a particle beyond the threshold (t_b is then the last snapshot).
    step None if even the first snapshot has one (t_b undefined)."""
    thr = lx - 5.0 * lam
    steps = snapshot_steps(entry)
    last, reached = None, False
    for step, p in steps:
        x = read_snapshot_x(p)
        if x.size and x.max() >= thr:
            reached = True
            break
        last = step
    return last, (None if last is None else last * dt), len(steps), reached


def t_b_table(entries, lx, lam, dt):
    """Per pooled realization: dict(g, step, t_b, n_snapshots, reached)."""
    out = []
    for e in entries:
        step, tb, n, reached = t_b_last_clean(e, lx, lam, dt)
        out.append({"g": e["g"], "step": step, "t_b": tb, "n_snapshots": n, "reached": reached})
    return out


def t_b_min(table, n_pilot=N_PILOT):
    """Minimum t_b over the first n_pilot realizations (error if one has no clean snapshot or fewer than n_pilot exist)."""
    first = [r for r in table if r["g"] < n_pilot]
    if len(first) < n_pilot:
        raise ValueError("only %d of the %d pilot realizations present" % (len(first), n_pilot))
    if any(r["t_b"] is None for r in first):
        raise ValueError("a pilot realization has no snapshot without a particle beyond Lx - 5 lambda (t_b undefined)")
    return min(r["t_b"] for r in first)


def late_count(table, t_hi, n_pilot=N_PILOT):
    """(late_beyond, n_beyond, late_pilot): realizations whose own last clean snapshot time precedes t_hi (end of W_c); reported, never dropped.
    A realization with no clean snapshot counts as late."""
    def late(r):
        return r["t_b"] is None or r["t_b"] < t_hi
    beyond = [r for r in table if r["g"] >= n_pilot]
    return sum(late(r) for r in beyond), len(beyond), sum(late(r) for r in table if r["g"] < n_pilot)


# ------------------------------------------------------------------ FIELDSTAT, wall times
_FS = re.compile(r"^FIELDSTAT r=(\d+) seed=(\d+) n=(\d+) mean_lnK=(\S+) var_lnK=(\S+) Kmin=(\S+) Kmax=(\S+) Kmean=(\S+)")


def parse_fieldstat(text):
    """seed -> dict(r, seed, n, mean_lnK, var_lnK, Kmin, Kmax, Kmean) from `par2_field_stats` stdout; a repeated seed is an error."""
    out = {}
    for ln in text.splitlines():
        m = _FS.match(ln)
        if not m:
            continue
        r, seed, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if seed in out:
            raise ValueError("FIELDSTAT seed %d repeated" % seed)
        vals = [float(x) for x in m.groups()[3:]]
        out[seed] = dict(zip(("r", "seed", "n", "mean_lnK", "var_lnK", "Kmin", "Kmax", "Kmean"), (r, seed, n, *vals)))
    return out


def wall_times(run_dir, start_epoch=None):
    """[(r, mtime, seconds_since_previous)] from mtimes of consecutive <run_dir>/stats/realization_%04d_timeseries.csv (written at the
    end of each realization); the first interval is measured from start_epoch (None if not given; it includes setup)."""
    out, prev = [], start_epoch
    for r in _indices(run_dir):
        t = os.stat(os.path.join(run_dir, "stats", "realization_%04d_timeseries.csv" % r)).st_mtime
        out.append((r, t, None if prev is None else t - prev))
        prev = t
    return out


# ------------------------------------------------------------------ CLI
def _read(path):
    with open(path) as f:
        return f.read()


def _entries(a):
    ds = a.dirs
    return pooled(ds[0], ds[1] if len(ds) > 1 else None)


def _common(a):
    v = cfg_values(_read(a.cfg))
    es = _entries(a)
    tab = t_b_table(es, v["Lx"], v["lam"], v["dt"])
    return v, es, tab


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m sf39.par2", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tb", "uc", "alpha"):
        p = sub.add_parser(name)
        p.add_argument("--cfg", required=True, help="pilot (or extension) pipeline config: nx, dx, corr_length, dt, inject_x")
        p.add_argument("dirs", nargs="+", help="pilot output dir [, extension output dir]")
        if name != "tb":
            p.add_argument("--first", type=int, default=None, help="use only the first N pooled realizations (default all; pilot freeze: 20)")
        if name == "alpha":
            p.add_argument("--tau-c", type=float, required=True)
            p.add_argument("--uc", type=float, default=None, help="final U_c (default: over the selected realizations)")
    p = sub.add_parser("fieldstat")
    p.add_argument("logs", nargs="+")
    p = sub.add_parser("walls")
    p.add_argument("dir")
    p.add_argument("--start-epoch", type=float, default=None)
    p = sub.add_parser("mkpilot")
    p.add_argument("sf38_cfg")
    p.add_argument("c", choices=AMPLITUDES)
    p = sub.add_parser("mkext")
    p.add_argument("pilot_cfg")
    p.add_argument("n_r", type=int)
    a = ap.parse_args(argv)

    if a.cmd == "mkpilot":
        sys.stdout.write(make_pilot_config(_read(a.sf38_cfg), a.c))
    elif a.cmd == "mkext":
        sys.stdout.write(make_ext_config(_read(a.pilot_cfg), a.n_r))
    elif a.cmd == "fieldstat":
        fs = {}
        for lg in a.logs:
            for s, d in parse_fieldstat(_read(lg)).items():
                if s in fs:
                    raise ValueError("seed %d in more than one log" % s)
                fs[s] = d
        for s in sorted(fs):
            print("FIELDSTAT seed=%d r=%d n=%d var_lnK=%.12e mean_lnK=%.12e" % (s, fs[s]["r"], fs[s]["n"], fs[s]["var_lnK"], fs[s]["mean_lnK"]))
    elif a.cmd == "walls":
        for r, t, dt in wall_times(a.dir, a.start_epoch):
            print("WALL r=%d mtime=%.3f seconds_since_previous=%s" % (r, t, "none" if dt is None else "%.3f" % dt))
    elif a.cmd == "tb":
        v, es, tab = _common(a)
        for r in tab:
            print("TB g=%d step=%s t_b=%s n_snapshots=%d reached=%d" % (
                r["g"], r["step"], "none" if r["t_b"] is None else "%.10g" % r["t_b"], r["n_snapshots"], int(r["reached"])))
        print("TB_MIN t_b_min=%.10g n_pilot=%d Lx=%.10g threshold_x=%.10g" % (t_b_min(tab), N_PILOT, v["Lx"], v["Lx"] - 5 * v["lam"]))
    else:
        v, es, tab = _common(a)
        tbm = t_b_min(tab)
        sel = es[:a.first] if a.first else es
        uc, _ = uc_mean([series_of(e) for e in sel], tbm, v["inject_x"])
        tau = tau_b_star(uc, tbm, v["lam"])
        if a.cmd == "uc":
            print("UC U_c=%.12e t_b_min=%.10g tau_b_star=%.12e n=%d anchor=%.10g lambda=%.10g" % (uc, tbm, tau, len(sel), v["inject_x"], v["lam"]))
        else:
            u = a.uc if a.uc is not None else uc
            t_lo, t_hi = window(a.tau_c, v["lam"], u)
            al = [alpha_window(series_of(e), v["lam"], u, t_lo, t_hi) for e in sel]
            for e, x in zip(sel, al):
                print("ALPHA g=%d alpha_x=%.12e alpha_y=%.12e alpha_z=%.12e" % (e["g"], x["x"], x["y"], x["z"]))
            m = {k: float(np.mean([x[k] for x in al])) for k in "xyz"}
            lb, nb, lp = late_count([r for r in tab if r["g"] < len(sel)], t_hi)
            print("ALPHA_MEAN n=%d U_c=%.12e t_lo=%.10g t_hi=%.10g alpha_x=%.12e alpha_y=%.12e alpha_z=%.12e alpha_T=%.12e" % (
                len(al), u, t_lo, t_hi, m["x"], m["y"], m["z"], 0.5 * (m["y"] + m["z"])))
            print("LATE beyond_pilot=%d/%d pilot=%d (own last clean snapshot time < t_hi=%.10g; reported, not dropped)" % (lb, nb, lp, t_hi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
