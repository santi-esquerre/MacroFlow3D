"""CLI: ``python3 -m sf43plan generate --variant A --campaign-dir DIR --binaries md5.json|placeholder --out OUT``."""
import argparse
import json
import os
import sys

from . import build as B
from . import design as D


def _binaries(arg):
    if arg == "placeholder":
        sys.stderr.write("WARNING: placeholder binary md5 (all zeros): the plan cannot start a campaign until regenerated with the real md5\n")
        return {"lagrangian_ensemble": B.PLACEHOLDER_MD5, "macroflow3d_pipeline": B.PLACEHOLDER_MD5}
    with open(arg) as f:
        b = json.load(f)
    if not isinstance(b, dict) or not b:
        raise SystemExit("--binaries: expected a JSON object {binary name: md5}")
    return b


def generate(args):
    if args.variant != "A":
        raise SystemExit("only variant A (owner decision H9) is implemented")
    bins = _binaries(args.binaries)
    reg = B.seeds_registry()
    B.check_seeds(reg)
    plan, _ = B.build_plan(bins, args.campaign_dir, B.analysis_hashes())
    files = {
        "plan.json": B.dumps(plan),
        "prereg.json": B.dumps(B.build_prereg()),
        "prereg_smoke.json": B.dumps(B.build_prereg(smoke=True)),
        "seeds_registry.json": B.dumps(reg),
        "design_table.md": B.design_table_md(plan, reg, all(v == B.PLACEHOLDER_MD5 for v in bins.values())),
    }
    os.makedirs(args.out, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
    print("wrote %s: %s (campaign %s, %d arms, smoke %d arms)" % (args.out, ", ".join(files), D.CAMPAIGN, len(plan["arms"]), len(plan["smoke"]["arms"])))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf43plan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write plan.json, prereg.json, prereg_smoke.json, seeds_registry.json, design_table.md")
    g.add_argument("--variant", required=True)
    g.add_argument("--campaign-dir", required=True, help="campaign_dir of the plan (relative to the repository root); the smoke dir is this + _smoke")
    g.add_argument("--binaries", required=True, help="JSON {name: md5} or the word 'placeholder'")
    g.add_argument("--out", required=True)
    return generate(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
