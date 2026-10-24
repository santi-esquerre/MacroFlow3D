#!/usr/bin/env python3
"""SF-38 threshold E (RR8, D12, D17) helpers for run_sf38.sh part `par2pilot`.  Everything is RECORDED, nothing interpreted.

  par2_pilot.py make-copy --src CFG --dst CFG2 --kind dt2|kstats|dry --outdir NAME
  par2_pilot.py outputs   --cfg-name N --out-dir DIR --nr N
  par2_pilot.py tb        --cfg CFG --cfg-name N --out-dir DIR --nr N
  par2_pilot.py fieldstat --cfg-name N --log FIELDSTAT.log
  par2_pilot.py kstats    --cfg-name N --fieldstat-log LOG --pipeline-log LOG
  par2_pilot.py rtimes    --cfg-name N --stats-dir DIR --start-epoch S --nr N
  par2_pilot.py paired    --cfg-name N --a CSV --b CSV --out FILE [--time-rtol 1e-9]
  par2_pilot.py --selftest

make-copy : rewrites a pipeline config (text edit, each key must occur exactly once; otherwise error).
    dt2    : dt 12.5 -> 6.25 (halved); n_steps, snapshots.every, sample_every, output_every doubled; macrodispersion NR -> 3;
             output_dir -> ./output/NAME.  (RR8: dt = 6.25, n_steps doubled, snapshot interval doubled, NR = 3, own output
             directory.  sample_every and output_every are doubled as well: the macrodispersion time series is sampled every
             `sample_every` steps, so without it the dt/2 series would be sampled at twice the time density and the paired
             differences of analysis/macrodispersion.csv would need re-sampling; reported as a reading of RR8.)
    dry    : tiny plumbing copy for SF38_DRYRUN=1 (64x16x16, dx 5, lambda 20, 1000 particles, 2400 steps); never evidence.
    kstats : n_steps -> 1, sample_every -> 1, output_every -> 1 (NR stays 10: the diagnostics build prints one `K stats` line
             per realization); output_dir -> ./output/NAME.
outputs   : E_OUTPUTS cfg= stats=<found>/<nr> macrodispersion=0|1 ensemble=0|1 snapshot_files= realizations_with_snapshots= ok=0|1
tb        : E_TB cfg= r= step= t_b= n_snapshots=   (t_b = step * dt of the first snapshot `snapshots/r_%04d/step_%08d.csv`
            with some particle at x >= Lx - 5 lambda, Lx = nx dx, lambda = stochastic.corr_length; `none` when no snapshot
            reaches it) and E_TB_SUMMARY.
fieldstat : E_LNK cfg= r= seed= mean_lnK= var_lnK= Kmin= Kmax= Kmean=  from the FIELDSTAT lines of par2_field_stats + E_LNK_SUMMARY
kstats    : E_KSTATS cfg= r= agree=0|1 tool=<min,max,mean at %.4e> pipeline=<...>  (string equality of the `%.4e` prints) and
            E_KSTATS_ALL cfg= agree=<n>/<m>.
rtimes    : RECORDED E_RTIME cfg= r= finished_at_s= seconds_since_previous=  (mtime of stats/realization_%04d_timeseries.csv, written
            at the end of each realization; the first interval includes setup).
paired    : RECORDED E_PAIRED cfg= col= rows= unmatched= max_abs_diff= max_rel_diff= rms_diff=  between two analysis/macrodispersion.csv
            files joined on t (first = dt run, restricted to its first 3 realizations; second = dt/2 run) and the difference series
            written to --out.
Stdlib only.
"""
import csv
import glob
import math
import os
import re
import sys


# ------------------------------------------------------------------ config rewriting
def _edit(lines, key_re, fn, what):
    hits = [i for i, ln in enumerate(lines) if not ln.lstrip().startswith("#") and re.match(key_re, ln)]
    if len(hits) != 1:
        raise ValueError("%s: expected exactly one line matching %s, found %d" % (what, key_re, len(hits)))
    i = hits[0]
    m = re.match(r"^(\s*[\w]+:\s*)([^\s#]+)(.*)$", lines[i])
    if not m:
        raise ValueError("%s: cannot parse line %r" % (what, lines[i]))
    lines[i] = m.group(1) + fn(m.group(2)) + m.group(3)


def make_copy(text, kind, outdir):
    lines = text.split("\n")
    if kind == "dt2":
        _edit(lines, r"^\s+dt:\s", lambda v: "%.15g" % (float(v) / 2.0), "dt")
        _edit(lines, r"^\s+n_steps:\s", lambda v: str(2 * int(v)), "n_steps")
        _edit(lines, r"^\s+every:\s", lambda v: str(2 * int(v)), "snapshots.every")
        _edit(lines, r"^\s+sample_every:\s", lambda v: str(2 * int(v)), "sample_every")
        _edit(lines, r"^\s+output_every:\s", lambda v: str(2 * int(v)), "output_every")
        _edit(lines, r"^\s+NR:\s", lambda v: "3", "NR")
    elif kind == "kstats":
        _edit(lines, r"^\s+n_steps:\s", lambda v: "1", "n_steps")
        _edit(lines, r"^\s+sample_every:\s", lambda v: "1", "sample_every")
        _edit(lines, r"^\s+output_every:\s", lambda v: "1", "output_every")
    elif kind == "dry":
        # tiny plumbing copy (SF38_DRYRUN=1, never evidence): 64x16x16, dx 5 (Lx 320), lambda 20, 1000 particles, 2400 steps
        for k, v in (("nx", "64"), ("ny", "16"), ("nz", "16"), ("mg_levels", "3"), ("corr_length", "20.0"), ("lambda", "20.0"),
                     ("n_steps", "2400"), ("n_particles", "1000"), ("inject_x", "100.0"), ("every", "24"), ("output_every", "80")):
            _edit(lines, r"^\s+%s:\s" % k, lambda _v, v=v: v, k)
        wl = [i for i, ln in enumerate(lines) if re.match(r"^\s+west:\s", ln) and not ln.lstrip().startswith("#")]
        if len(wl) != 1:
            raise ValueError("west: expected exactly one line")
        lines[wl[0]] = re.sub(r"value:\s*[-+0-9.eE]+", "value: 3.125", lines[wl[0]])
    else:
        raise ValueError("kind must be dt2, kstats or dry")
    _edit(lines, r"^\s+output_dir:\s", lambda v: "./output/" + outdir, "output_dir")
    return "\n".join(lines)


def cfg_values(text):
    def g(pat, cast=float):
        m = re.search(pat, text, re.M)
        if not m:
            raise ValueError("config key not found: %s" % pat)
        return cast(m.group(1))
    return {"nx": g(r"^\s+nx:\s*(\d+)", int), "dx": g(r"^\s+dx:\s*([-+0-9.eE]+)"), "lam": g(r"^\s+corr_length:\s*([-+0-9.eE]+)"),
            "dt": g(r"^\s+dt:\s*([-+0-9.eE]+)")}


# ------------------------------------------------------------------ outputs / t_b
def outputs(name, out_dir, nr):
    stats = len(glob.glob(os.path.join(out_dir, "stats", "realization_*_timeseries.csv")))
    mac = os.path.isfile(os.path.join(out_dir, "analysis", "macrodispersion.csv"))
    ens = os.path.isfile(os.path.join(out_dir, "ensemble", "ensemble_timeseries.csv"))
    snaps = glob.glob(os.path.join(out_dir, "snapshots", "r_*", "step_*.csv"))
    rs = len({os.path.basename(os.path.dirname(p)) for p in snaps})
    ok = stats == nr and mac and ens and rs == nr
    return "E_OUTPUTS cfg=%s stats=%d/%d macrodispersion=%d ensemble=%d snapshot_files=%d realizations_with_snapshots=%d/%d ok=%d" % (
        name, stats, nr, int(mac), int(ens), len(snaps), rs, nr, int(ok)), ok


def max_x(path):
    with open(path, errors="replace") as f:
        header = f.readline().strip().split(",")
        idx = None
        for k, h in enumerate(header):
            if h.strip().lower() in ("x coord", "x"):
                idx = k
                break
        if idx is None:
            raise ValueError("no x column in %s (header %s)" % (path, header))
        best = -math.inf
        for ln in f:
            if not ln.strip():
                continue
            v = float(ln.split(",")[idx])
            if v > best:
                best = v
    return best


def tb_all(name, cfg_text, out_dir, nr):
    v = cfg_values(cfg_text)
    lx, thr = v["nx"] * v["dx"], v["nx"] * v["dx"] - 5.0 * v["lam"]
    lines, tbs = [], []
    for r in range(nr):
        files = sorted(glob.glob(os.path.join(out_dir, "snapshots", "r_%04d" % r, "step_*.csv")))
        steps = sorted((int(re.search(r"step_(\d+)\.csv$", p).group(1)), p) for p in files)
        found = None
        for step, p in steps:
            if max_x(p) >= thr:
                found = step
                break
        tb = None if found is None else found * v["dt"]
        tbs.append(tb)
        lines.append("E_TB cfg=%s r=%d step=%s t_b=%s n_snapshots=%d Lx=%.10g threshold_x=%.10g dt=%.10g"
                     % (name, r, "none" if found is None else found, "none" if tb is None else "%.10g" % tb, len(steps), lx, thr, v["dt"]))
    ok = [t for t in tbs if t is not None]
    lines.append("E_TB_SUMMARY cfg=%s found=%d/%d t_b_min=%s t_b_max=%s" % (
        name, len(ok), nr, "%.10g" % min(ok) if ok else "none", "%.10g" % max(ok) if ok else "none"))
    return lines


# ------------------------------------------------------------------ field statistics
FS = re.compile(r"^FIELDSTAT r=(\d+) seed=(\d+) n=(\d+) mean_lnK=(\S+) var_lnK=(\S+) Kmin=(\S+) Kmax=(\S+) Kmean=(\S+)")


def read_fieldstat(text):
    return {int(m.group(1)): m.groups() for m in (FS.match(ln) for ln in text.splitlines()) if m}


def fieldstat_lines(name, text):
    d = read_fieldstat(text)
    out, means, vars_ = [], [], []
    for r in sorted(d):
        _, seed, n, mean, var, kmin, kmax, kmean = d[r]
        out.append("E_LNK cfg=%s r=%d seed=%s mean_lnK=%s var_lnK=%s Kmin=%s Kmax=%s Kmean=%s" % (name, r, seed, mean, var, kmin, kmax, kmean))
        means.append(float(mean))
        vars_.append(float(var))
    if vars_:
        n = len(vars_)
        mv = sum(vars_) / n
        sv = math.sqrt(sum((x - mv) ** 2 for x in vars_) / (n - 1)) if n > 1 else float("nan")
        out.append("E_LNK_SUMMARY cfg=%s n=%d mean_lnK_mean=%.12e var_lnK_mean=%.12e var_lnK_sd=%.12e" % (name, n, sum(means) / n, mv, sv))
    else:
        out.append("E_LNK_SUMMARY cfg=%s n=0" % name)
    return out


KS = re.compile(r"K stats: min=(\S+)\s+max=(\S+)\s+mean=(\S+)")


def kstats_lines(name, fs_text, pipe_text):
    d = read_fieldstat(fs_text)
    ks = KS.findall(pipe_text)
    out, agree = [], 0
    for i, r in enumerate(sorted(d)):
        tool = (d[r][5], d[r][6], d[r][7])
        pipe = ks[i] if i < len(ks) else None
        ok = pipe is not None and tuple(pipe) == tool
        agree += int(ok)
        out.append("E_KSTATS cfg=%s r=%d agree=%d tool=%s pipeline=%s" % (name, r, int(ok), ",".join(tool), ",".join(pipe) if pipe else "missing"))
    out.append("E_KSTATS_ALL cfg=%s agree=%d/%d pipeline_lines=%d" % (name, agree, len(d), len(ks)))
    return out


# ------------------------------------------------------------------ timing
def rtimes(name, stats_dir, start_epoch, nr):
    out, prev = [], start_epoch
    for r in range(nr):
        p = os.path.join(stats_dir, "realization_%04d_timeseries.csv" % r)
        if not os.path.exists(p):
            out.append("RECORDED E_RTIME cfg=%s r=%d missing" % (name, r))
            continue
        t = os.stat(p).st_mtime
        out.append("RECORDED E_RTIME cfg=%s r=%d finished_at_s=%.3f seconds_since_previous=%.3f" % (name, r, t - start_epoch, t - prev))
        prev = t
    return out


# ------------------------------------------------------------------ paired dt vs dt/2
def read_table(path):
    with open(path, newline="") as f:
        rows = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith("#")]
    rd = list(csv.reader(rows))
    return rd[0], [[float(x) for x in r] for r in rd[1:]]


def paired(name, pa, pb, outpath, trtol):
    ha, ra = read_table(pa)
    hb, rb = read_table(pb)
    if ha != hb:
        return ["RECORDED E_PAIRED cfg=%s headers_differ a=%s b=%s" % (name, ha, hb)]
    bt = {}
    for row in rb:
        bt[row[0]] = row
    keys = sorted(bt)
    matched, unmatched = [], 0
    for row in ra:
        # join on t within trtol (the dt grid is contained in the dt/2 grid when sample_every is doubled)
        best = None
        for k in keys:
            if abs(k - row[0]) <= trtol * max(abs(k), abs(row[0]), 1e-300):
                best = k
                break
        if best is None:
            unmatched += 1
        else:
            matched.append((row, bt[best]))
    out = []
    with open(outpath, "w") as f:
        f.write("# paired differences (dt minus dt/2) of analysis/macrodispersion.csv, joined on t; RECORDED only\n")
        f.write(",".join(ha) + "\n")
        for a, b in matched:
            f.write(",".join("%.15e" % (x - y) for x, y in zip(a, b)) + "\n")
    for c, col in enumerate(ha):
        if c == 0:
            continue
        d = [(a[c] - b[c], b[c]) for a, b in matched]
        if not d:
            out.append("RECORDED E_PAIRED cfg=%s col=%s rows=0 unmatched=%d" % (name, col, unmatched))
            continue
        mabs = max(abs(x) for x, _ in d)
        rels = [abs(x) / abs(y) for x, y in d if y != 0]
        rms = math.sqrt(sum(x * x for x, _ in d) / len(d))
        out.append("RECORDED E_PAIRED cfg=%s col=%s rows=%d unmatched=%d max_abs_diff=%.6e max_rel_diff=%s rms_diff=%.6e"
                   % (name, col, len(d), unmatched, mabs, "%.6e" % max(rels) if rels else "-", rms))
    return out


# ------------------------------------------------------------------ selftest
SAMPLE_CFG = """# comment with dt: 99 and every: 7
grid:
  nx: 64
  ny: 16
  nz: 16
  dx: 5.0
stochastic:
  corr_length: 20.0
transport:
  dt: 12.5
  n_steps: 1000
  output_every: 33
  snapshot_every: 0
analysis:
  macrodispersion:
    enabled: true
    NR: 10
    lambda: 20.0
    sample_every: 10
  snapshots:
    enabled: true
    every: 10
output:
  output_dir: ./output/sf38_par2_s1
"""


def selftest():
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    t = make_copy(SAMPLE_CFG, "dt2", "dt2_s1")
    expect("dt2_dt", "  dt: 6.25\n" in t)
    expect("dt2_nsteps", "  n_steps: 2000\n" in t and "  NR: 3\n" in t.replace("    NR", "  NR"))
    expect("dt2_every", "    every: 20\n" in t and "  snapshot_every: 0\n" in t)
    expect("dt2_sample", "    sample_every: 20\n" in t and "  output_every: 66\n" in t)
    expect("dt2_outdir", "  output_dir: ./output/dt2_s1\n" in t)
    expect("dt2_comment_untouched", t.startswith("# comment with dt: 99 and every: 7\n"))
    expect("dt2_lambda_untouched", "    lambda: 20.0\n" in t)
    k = make_copy(SAMPLE_CFG, "kstats", "k_s1")
    expect("kstats", "  n_steps: 1\n" in k and "    NR: 10\n" in k and "    sample_every: 1\n" in k and "  dt: 12.5\n" in k)
    try:
        make_copy(SAMPLE_CFG.replace("  dt: 12.5\n", ""), "dt2", "x")
        expect("missing_key_rejected", False)
    except ValueError:
        expect("missing_key_rejected", True)
    try:
        make_copy(SAMPLE_CFG + "  extra:\n    dt: 1\n", "dt2", "x")
        expect("duplicate_key_rejected", False)
    except ValueError:
        expect("duplicate_key_rejected", True)
    expect("cfg_values", cfg_values(SAMPLE_CFG) == {"nx": 64, "dx": 5.0, "lam": 20.0, "dt": 12.5})

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(d + "/snapshots/r_0000")
        os.makedirs(d + "/snapshots/r_0001")
        os.makedirs(d + "/stats")
        # Lx = 320, lambda = 20 -> threshold 220; dt = 12.5
        def snap(r, step, xs):
            with open("%s/snapshots/r_%04d/step_%08d.csv" % (d, r, step), "w") as f:
                f.write("x coord,y coord,z coord\n")
                for x in xs:
                    f.write("%.15f,1.0,2.0\n" % x)
        snap(0, 10, [10.0, 20.0])
        snap(0, 20, [100.0, 219.9])
        snap(0, 30, [100.0, 220.0])
        snap(0, 40, [300.0])
        snap(1, 10, [10.0])
        snap(1, 20, [11.0])
        lines = tb_all("c", SAMPLE_CFG, d, 2)
        expect("tb_r0", "r=0 step=30 t_b=375 n_snapshots=4" in lines[0])
        expect("tb_r1_none", "r=1 step=none t_b=none n_snapshots=2" in lines[1])
        expect("tb_summary", "found=1/2 t_b_min=375 t_b_max=375" in lines[2])
        for r in range(2):
            with open("%s/stats/realization_%04d_timeseries.csv" % (d, r), "w") as f:
                f.write("t\n")
        os.makedirs(d + "/analysis")
        os.makedirs(d + "/ensemble")
        open(d + "/analysis/macrodispersion.csv", "w").close()
        open(d + "/ensemble/ensemble_timeseries.csv", "w").close()
        line, ok = outputs("c", d, 2)
        expect("outputs_ok", ok and "stats=2/2" in line and "realizations_with_snapshots=2/2" in line)
        expect("outputs_fail", not outputs("c", d, 3)[1])
        rt = rtimes("c", d + "/stats", os.stat(d + "/stats/realization_0000_timeseries.csv").st_mtime - 5.0, 2)
        expect("rtimes", "r=0 finished_at_s=5.000 seconds_since_previous=5.000" in rt[0] and len(rt) == 2)

    fs = ("FIELDSTAT r=0 seed=7001 n=100 mean_lnK=1.0e-03 var_lnK=9.9e-01 Kmin=1.1000e-01 Kmax=9.2000e+00 Kmean=1.4000e+00\n"
          "FIELDSTAT r=1 seed=7002 n=100 mean_lnK=-1.0e-03 var_lnK=1.01e+00 Kmin=1.2000e-01 Kmax=8.2000e+00 Kmean=1.5000e+00\n")
    ll = fieldstat_lines("c", fs)
    expect("fieldstat", len(ll) == 3 and "var_lnK_mean=1.000000000000e+00" in ll[2] and "r=1 seed=7002" in ll[1])
    pipe = ("  K stats: min=1.1000e-01  max=9.2000e+00  mean=1.4000e+00\n  K stats: min=1.2000e-01  max=8.2000e+00  mean=1.5001e+00\n")
    kl = kstats_lines("c", fs, pipe)
    expect("kstats_cmp", "r=0 agree=1" in kl[0] and "r=1 agree=0" in kl[1] and "agree=1/2 pipeline_lines=2" in kl[2])
    expect("kstats_missing", "pipeline=missing" in kstats_lines("c", fs, "")[0])

    with tempfile.TemporaryDirectory() as d:
        with open(d + "/a.csv", "w") as f:
            f.write("t,alpha_x,alpha_y,alpha_z,dvar_dt_x,dvar_dt_y,dvar_dt_z\n125,1,2,3,4,5,6\n250,2,2,2,2,2,2\n400,9,9,9,9,9,9\n")
        with open(d + "/b.csv", "w") as f:
            f.write("t,alpha_x,alpha_y,alpha_z,dvar_dt_x,dvar_dt_y,dvar_dt_z\n62.5,0,0,0,0,0,0\n125,1.5,2,3,4,5,6\n250,2,2,2,2,2,1\n")
        pl = paired("c", d + "/a.csv", d + "/b.csv", d + "/p.csv", 1e-9)
        expect("paired_rows", "col=alpha_x rows=2 unmatched=1 max_abs_diff=5.000000e-01 max_rel_diff=3.333333e-01" in pl[0])
        expect("paired_zero", "col=alpha_y rows=2 unmatched=1 max_abs_diff=0.000000e+00" in pl[1])
        expect("paired_file", open(d + "/p.csv").read().count("\n") == 4)
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK par2_pilot")
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    cmd = argv[1]
    o = {}
    i = 2
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            o[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            print("par2_pilot.py: bad argument %r" % argv[i], file=sys.stderr)
            return 2
    try:
        if cmd == "make-copy":
            with open(o["src"]) as f:
                text = f.read()
            with open(o["dst"], "w") as f:
                f.write(make_copy(text, o["kind"], o["outdir"]))
            return 0
        if cmd == "outputs":
            line, ok = outputs(o["cfg-name"], o["out-dir"], int(o["nr"]))
            print(line)
            return 0 if ok else 1
        if cmd == "tb":
            with open(o["cfg"]) as f:
                text = f.read()
            print("\n".join(tb_all(o["cfg-name"], text, o["out-dir"], int(o["nr"]))))
            return 0
        if cmd == "fieldstat":
            with open(o["log"]) as f:
                print("\n".join(fieldstat_lines(o["cfg-name"], f.read())))
            return 0
        if cmd == "kstats":
            with open(o["fieldstat-log"]) as f:
                fs = f.read()
            with open(o["pipeline-log"]) as f:
                pl = f.read()
            print("\n".join(kstats_lines(o["cfg-name"], fs, pl)))
            return 0
        if cmd == "rtimes":
            print("\n".join(rtimes(o["cfg-name"], o["stats-dir"], float(o["start-epoch"]), int(o["nr"]))))
            return 0
        if cmd == "paired":
            print("\n".join(paired(o["cfg-name"], o["a"], o["b"], o["out"], float(o.get("time-rtol", "1e-9")))))
            return 0
    except (KeyError, ValueError, OSError) as e:
        print("par2_pilot.py %s: %s" % (cmd, e), file=sys.stderr)
        return 2
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
