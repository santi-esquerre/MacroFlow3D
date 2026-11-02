"""CLI: python3 -m sf43ana run --prereg P.json --campaign DIR --reduced DIR --gate5 G.json|none --out DIR (from the scripts/ directory)."""
import argparse
import sys

from . import summary
from .config import ConfigError
from .loader import LoadError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf43ana")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Gate 4 core: estimands, budgets, robust sign, Holm family, classes -> <out>/summary.json")
    r.add_argument("--prereg", required=True)
    r.add_argument("--campaign", required=True, help="directory with the ledger (name from the pre-registration)")
    r.add_argument("--reduced", required=True, help="directory of the sf42red tables (pre-registration `reduced_pattern`)")
    r.add_argument("--gate5", required=True, help="JSON of Gate 5 p-values per amplitude, or 'none' (p = 1 in the family)")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        _, path = summary.run(a.prereg, a.campaign, a.reduced, a.gate5, a.out)
    except (ConfigError, LoadError) as e:
        print("sf43ana: %s" % e, file=sys.stderr)
        return 2
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
