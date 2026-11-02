"""CLI (from the scripts/ directory):

  python3 -m sf43ana run --prereg P.json --campaign DIR --reduced DIR [--gate5 auto|none|G.json] [--coverage-summary PATH|none] --out DIR
  python3 -m sf43ana check-finite --summary DIR/summary.json
"""
import argparse
import sys

from . import summary
from .config import ConfigError
from .loader import LoadError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf43ana")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="estimands, budgets, robust sign, Holm family, classes, Gate 5 and the descriptive blocks -> <out>/summary.json")
    r.add_argument("--prereg", required=True)
    r.add_argument("--campaign", required=True, help="directory with the ledger (name from the pre-registration)")
    r.add_argument("--reduced", required=True, help="directory of the sf43camp reduced tables (pre-registration `reduced_pattern`)")
    r.add_argument("--gate5", default="auto", help="auto (default): Gate 5 from the pre-registration's gate5 block; none: no Gate 5 p-values (p = 1); "
                   "or a JSON of p-values per amplitude")
    r.add_argument("--coverage-summary", default=None, help="path of the sf43cov summary.json, or 'none' (Gate 5 descriptive); default: the "
                   "pre-registration's gate5.coverage_summary")
    r.add_argument("--out", required=True)
    c = sub.add_parser("check-finite", help="structure and counts of finite values of a summary.json; never prints an estimate")
    c.add_argument("--summary", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check-finite":
        from . import finite
        return finite.run(a.summary)
    try:
        _, path = summary.run(a.prereg, a.campaign, a.reduced, a.gate5, a.out, a.coverage_summary)
    except (ConfigError, LoadError) as e:
        print("sf43ana: %s" % e, file=sys.stderr)
        return 2
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
