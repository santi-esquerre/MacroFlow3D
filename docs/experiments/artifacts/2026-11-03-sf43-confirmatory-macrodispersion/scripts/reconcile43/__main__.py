"""python3 -m reconcile43 run --campaign <C> --out <report>"""
import argparse
import os
import sys

from . import driver


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reconcile43", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="reconcile every solver record of the campaign (read only)")
    r.add_argument("--campaign", required=True, help="campaign directory <C> (read only)")
    r.add_argument("--out", required=True, help="report file (must not be inside --campaign)")
    a = ap.parse_args(argv)
    camp = os.path.realpath(a.campaign)
    if os.path.realpath(a.out).startswith(camp + os.sep):
        ap.error("--out must not be inside --campaign (the campaign tree is read only)")
    with open(a.out, "w") as f:
        code = driver.run(camp, f)
    with open(a.out) as f:
        print(f.read().splitlines()[-1])
    return code


if __name__ == "__main__":
    sys.exit(main())
