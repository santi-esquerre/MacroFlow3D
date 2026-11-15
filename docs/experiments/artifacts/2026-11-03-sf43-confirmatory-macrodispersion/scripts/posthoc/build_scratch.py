"""post hoc (2026-11-04), SF-43 addendum 1, node A1-POSTHOC: rebuild the campaign layout of the frozen run in a scratch directory.

    python3 build_scratch.py SCRATCH

Copies results/ledger.jsonl and extracts every reduced/tables/<arm>.tar.xz and reduced/planes/<arm>.tar.xz of the committed results into
SCRATCH/camp/reduced/, then checks every extracted file against results/reduced/freeze.reduced.sha256 (the manifest of the host reduced
directory, lines ``sha256  ./<arm>/<segment>/<file>``).  Exit 1 on any mismatch or missing/extra file.  SCRATCH must be outside the repository.
"""
import hashlib
import os
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ART, "results")


def main(scratch):
    scratch = os.path.abspath(scratch)
    if scratch.startswith(ART + os.sep):
        sys.exit("scratch must be outside the artifact directory")
    camp = os.path.join(scratch, "camp")
    red = os.path.join(camp, "reduced")
    os.makedirs(red, exist_ok=True)
    shutil.copy(os.path.join(RES, "ledger.jsonl"), os.path.join(camp, "ledger.jsonl"))
    for kind in ("tables", "planes"):
        d = os.path.join(RES, "reduced", kind)
        for f in sorted(os.listdir(d)):
            with tarfile.open(os.path.join(d, f), "r:xz") as t:
                t.extractall(red, filter="data")
    want = {}
    for line in open(os.path.join(RES, "reduced", "freeze.reduced.sha256")):
        h, p = line.rstrip("\n").split("  ", 1)
        want[p[2:] if p.startswith("./") else p] = h
    got = {}
    for root, _, files in os.walk(red):
        for f in files:
            p = os.path.join(root, f)
            got[os.path.relpath(p, red)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    bad = [p for p in want if got.get(p) != want[p]]
    extra = [p for p in got if p not in want]
    print("manifest files %d, extracted %d, mismatched/missing %d, not in manifest %d" % (len(want), len(got), len(bad), len(extra)))
    for p in bad[:10] + extra[:10]:
        print("  ", p)
    return 1 if bad or extra else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
