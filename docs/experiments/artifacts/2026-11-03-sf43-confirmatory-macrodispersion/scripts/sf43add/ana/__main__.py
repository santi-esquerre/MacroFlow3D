"""CLI (from the repository root, PYTHONPATH = <A>/scripts:<SF42>/scripts):

  python3 -W default -m sf43add.ana run --prereg P --campaign C --frozen-reduced R --frozen-manifest M --frozen-ledger L [--frozen-segments S] [--j8-campaign J] --out O
  python3 -W default -m sf43add.ana check --prereg P --campaign C --out O
  python3 -W default -m sf43add.ana repro --campaign C --arm j1_base10 --seeds 434001[-434010] --frozen-reduced R --frozen-manifest M [--frozen-segments S]
  python3 -W default -m sf43add.ana j8-trigger --summary O/addendum_summary.json --gpu-h-used X --j8-gpu-h-projected Y --j8-end-utc ISO
"""
import argparse
import sys

from . import check, repro, summary, trigger


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf43add.ana")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="pre-registered analysis -> addendum_summary.json/.md; stdout/stderr structure only")
    r.add_argument("--prereg", required=True, help="prereg_addendum.json")
    r.add_argument("--campaign", required=True, help="addendum campaign directory (ledger.jsonl, reduced/, segments/)")
    r.add_argument("--frozen-reduced", required=True, help="frozen reduced directory <arm>/<segment>/<file>")
    r.add_argument("--frozen-manifest", required=True, help="freeze manifest of the frozen reduced directory (sha256  ./<path>)")
    r.add_argument("--frozen-ledger", required=True, help="ledger.jsonl of the frozen campaign")
    r.add_argument("--frozen-segments", default=None, help="frozen segments directory (Par2 crossing CSVs), optional")
    r.add_argument("--j8-campaign", default=None, help="campaign directory of the conditional J8 job, optional")
    r.add_argument("--out", required=True)
    c = sub.add_parser("check", help="smoke structure check (counts only)")
    c.add_argument("--prereg", required=True)
    c.add_argument("--campaign", required=True)
    c.add_argument("--out", required=True)
    p = sub.add_parser("repro", help="reproduction check of J1 (3.1.1): REPRO PASS|FAIL n_seeds n_rows max_rel csv_sha_equal=k/n")
    p.add_argument("--campaign", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--seeds", required=True, help="434001 | 434001-434010 | comma list")
    p.add_argument("--frozen-reduced", required=True)
    p.add_argument("--frozen-manifest", required=True)
    p.add_argument("--frozen-segments", default=None)
    p.add_argument("--frozen-arm", default="s4_g5p")
    t = sub.add_parser("j8-trigger", help="J8 RUN|SKIP from addendum_summary.json (read after the freeze)")
    t.add_argument("--summary", required=True)
    t.add_argument("--gpu-h-used", type=float, required=True)
    t.add_argument("--j8-gpu-h-projected", type=float, required=True)
    t.add_argument("--j8-end-utc", required=True)
    a = ap.parse_args(argv)
    return {"run": summary.main, "check": check.main, "repro": repro.main, "j8-trigger": trigger.main}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
