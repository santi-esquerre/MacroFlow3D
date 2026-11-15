"""python3 -m recompute43 run --campaign <C> --reduced <dir> --out <report> [--root R ...] [--primary-arm ID ...] [--rtol 1e-10] [--atol 1e-15]"""
import argparse
import os
import sys

from . import driver


def main(argv=None):
    ap = argparse.ArgumentParser(prog="recompute43", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="recompute the reduced tables from the raw moment files (read only)")
    r.add_argument("--campaign", required=True, help="campaign directory <C> (read only)")
    r.add_argument("--reduced", required=True, help="directory holding the sf42red tables, their .planes.csv and .manifest.json")
    r.add_argument("--out", required=True, help="report file (must not be inside --campaign)")
    r.add_argument("--root", action="append", default=None, help="root the tables' `source` paths are relative to (repeatable; default: campaign)")
    r.add_argument("--primary-arm", action="append", default=None,
                   help="arm id whose every consumer realization must have the plane arrays (repeatable; H21)")
    r.add_argument("--single-pass-hmax", type=int, default=None, help="hmax_div label of single-pass roots reduced with --single-pass-hmax (default: -1 only)")
    r.add_argument("--rtol", type=float, default=driver.RTOL)
    r.add_argument("--atol", type=float, default=driver.ATOL)
    a = ap.parse_args(argv)
    camp = os.path.realpath(a.campaign)
    if os.path.realpath(a.out).startswith(camp + os.sep):
        ap.error("--out must not be inside --campaign (the campaign tree is read only)")
    with open(a.out, "w") as f:
        code = driver.run(camp, os.path.realpath(a.reduced), [os.path.realpath(x) for x in (a.root or [camp])], f, a.rtol, a.atol,
                          a.primary_arm or (), a.single_pass_hmax)
    with open(a.out) as f:
        print(f.read().splitlines()[-1])
    return code


if __name__ == "__main__":
    sys.exit(main())
