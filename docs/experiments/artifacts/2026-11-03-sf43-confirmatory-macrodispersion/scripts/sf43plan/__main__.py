"""CLI: ``python3 -m sf43plan generate --variant A --campaign-dir DIR --binaries md5.json|placeholder (--coverage-summary S --coverage-qualification Q | --smoke-only) --out OUT``."""
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
    cov = (args.coverage_summary, args.coverage_qualification)
    if any(cov) and not all(cov):
        raise SystemExit("--coverage-summary and --coverage-qualification go together")
    if not all(cov) and not args.smoke_only:
        raise SystemExit("a production plan needs --coverage-summary and --coverage-qualification (the files of the smoke job's coverage run, "
                         "committed under coverage/); a plan for the smoke run only is generated with --smoke-only")
    bins = _binaries(args.binaries)
    reg = B.seeds_registry()
    B.check_seeds(reg)
    prereg_text = B.dumps(B.build_prereg())
    smoke_text = B.dumps(B.build_prereg(smoke=True))
    h = B.analysis_hashes(prereg_text, B.ART_REL + "/prereg.json", cov if all(cov) else None)
    if not all(cov):
        h = B.with_coverage_placeholders(h)
    hs = B.analysis_hashes(smoke_text, B.ART_REL + "/prereg_smoke.json")
    plan, _ = B.build_plan(bins, args.campaign_dir, h, hs)
    files = {
        "plan.json": B.dumps(plan),
        "prereg.json": prereg_text,
        "prereg_smoke.json": smoke_text,
        "seeds_registry.json": B.dumps(reg),
        "design_table.md": B.design_table_md(plan, reg, all(v == B.PLACEHOLDER_MD5 for v in bins.values()), not all(cov)),
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
    g.add_argument("--coverage-summary", help="summary.json of the sf43cov run (hashed into analysis.sha256_list; committed as <A>/coverage/summary.json)")
    g.add_argument("--coverage-qualification", help="qualification.json of the same run (<A>/coverage/qualification.json)")
    g.add_argument("--smoke-only", action="store_true", help="no coverage files yet: the plan is good for --smoke only (production sha256 placeholders)")
    g.add_argument("--out", required=True)
    return generate(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
