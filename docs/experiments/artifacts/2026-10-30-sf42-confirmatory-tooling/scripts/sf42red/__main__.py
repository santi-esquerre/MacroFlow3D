"""CLI: python3 -m sf42red {reduce,splitshare,prune} ... (run from the `scripts/` directory)."""
import argparse
import csv
import sys

from . import prune as prune_mod
from . import reduce as red


def _floats(s):
    a, b = (float(v) for v in s.split(","))
    return a, b


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf42red")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reduce", help="per-realization plane reduction (consumer pass/arm directories, Par2 crossing CSVs)")
    r.add_argument("--consumer", nargs="*", default=[], help="pass roots or arm directories (with realizations/)")
    r.add_argument("--par2", nargs="*", default=[], help="crossings/realization_%%04d_planes.csv files")
    r.add_argument("--par2-series", nargs="*", default=[], help="stats/realization_%%04d_timeseries.csv (secondary; needs --U-time, --par2-exit-x, --par2-dt and "
                   "<arm>/snapshots/r_%%04d/step_%%08d.csv beside stats/)")
    r.add_argument("--par2-exit-x", type=float, default=None, help="Par2 window end: x (config length units) beyond which a particle "
                   "counts as exited; t_end = the last snapshot with no particle at x >= this (SF-39 used Lx - 5 lambda; the consumer's "
                   "first exit is the outlet face x = L1)")
    r.add_argument("--par2-dt", type=float, default=None, help="Par2 time step (snapshot time = step x dt)")
    r.add_argument("--ell", type=float, default=None, help="lambda (required for Par2; consumer: checked against field.json)")
    r.add_argument("--x0", type=float, default=None, help="injection plane (required for Par2; consumer: checked against track.json)")
    r.add_argument("--d-window", type=_floats, default=(25.0, 94.0), help="d_lo,d_hi in lambda (default 25,94)")
    r.add_argument("--U-time", type=float, default=None, help="mean speed of the time-window alphas (default: qbar per realization)")
    r.add_argument("--par2-seed0", type=int, default=None, help="seed of Par2 realization 0 (SF-43): the Par2 rows carry seed = seed0 + r "
                   "(default: NaN, the SF-42 behaviour)")
    r.add_argument("--single-pass-hmax", type=int, default=None, help="hmax_div label of single-pass realization roots (SF-43; default -1)")
    r.add_argument("--pollock", action="store_true", help="SF-43: also reduce the Pollock bridge arrays (<realization>/pollock/) of every "
                   "consumer realization into code=pollock plane rows (an error if a realization has none; default off = the SF-42 behaviour)")
    r.add_argument("--root", default=None, help="manifest root (default: common parent of the inputs)")
    r.add_argument("--out", required=True)
    s = sub.add_parser("splitshare", help="particle share of the per-realization variance from a reduced table (SF-39 definition)")
    s.add_argument("table")
    p = sub.add_parser("prune", help="delete per-particle .npy of non-retained realizations (dry run unless --execute)")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--retained", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--log", default=None)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--min-retained", type=int, default=prune_mod.MIN_RETAINED,
                   help="decision G8 minimum of retained realizations per directory (default %d; SF-43 passes a lower value, down to 0, "
                   "when it prunes per segment directory against an arm-level retained list)" % prune_mod.MIN_RETAINED)
    p.add_argument("--keep-arrays", default="", help="comma-separated array names (no extension) never deleted in any realization, e.g. "
                   "seeds,weights,planes_x23,planes_tau (default: none)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "reduce":
            rows = red.reduce_all(a.consumer, a.par2, a.par2_series, a.ell, a.x0, a.d_window[0], a.d_window[1], a.U_time, a.out, a.root,
                                 par2_exit_x=a.par2_exit_x, par2_dt=a.par2_dt, par2_seed0=a.par2_seed0,
                                 single_pass_hmax=a.single_pass_hmax, pollock=a.pollock)
            print("wrote %d rows to %s" % (len(rows), a.out))
        elif a.cmd == "splitshare":
            rows = red.split_share(a.table)
            cols = ["code", "family", "weights", "basis", "set", "hmax_div", "estimator", "n", "component", "variance", "fraction", "flag"]
            wr = csv.writer(sys.stdout, lineterminator="\n")
            wr.writerow(cols)
            for row in rows:
                wr.writerow([red._fmt(row[c]) for c in cols])
        else:
            prune_mod.prune(a.arms, a.retained, a.table, a.root, a.execute, a.log, min_retained=a.min_retained,
                            keep_arrays=[k for k in a.keep_arrays.split(",") if k])
    except red.ReduceError as e:
        print("sf42red: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
