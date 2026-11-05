"""post hoc (2026-11-04): write A/posthoc/SHA256SUMS = sha256 of every INPUT file read (frozen inputs: pre-registration, committed results,
coverage documents, the python sources of the frozen packages sf43ana, sf43cov, mcstat, sf42red) and every OUTPUT file (A/posthoc/**
except SHA256SUMS itself, and the scripts A/scripts/posthoc/*.py|sh|json).  Format of ``sha256sum -c`` (paths relative to the repository root)."""
import os

from common import ART, OUT, SF42, SCRIPTS, sha256

ROOT = os.path.abspath(os.path.join(ART, "..", "..", "..", ".."))


def walk(d, exts=None):
    for r, ds, fs in os.walk(d):
        ds[:] = [x for x in ds if x not in ("__pycache__", "tests")]
        for f in fs:
            if exts is None or f.endswith(exts):
                yield os.path.join(r, f)


def main():
    files = {os.path.join(ART, "prereg.json"), os.path.join(ART, "results", "ledger.jsonl"),
             os.path.join(ART, "results", "analysis", "summary.json"),
             os.path.join(ART, "coverage", "summary.json"), os.path.join(ART, "coverage", "qualification.json")}
    files |= set(walk(os.path.join(ART, "results", "reduced")))
    for pkg in ("sf43ana", "sf43cov"):
        files |= set(walk(os.path.join(SCRIPTS, pkg), ".py"))
    for pkg in ("mcstat", "sf42red"):
        files |= set(walk(os.path.join(SF42, pkg), ".py"))
    files |= {p for p in walk(OUT) if os.path.basename(p) != "SHA256SUMS"}
    files |= set(walk(os.path.dirname(os.path.abspath(__file__)), (".py", ".sh", ".json")))
    lines = ["%s  %s" % (sha256(p), os.path.relpath(os.path.realpath(p), os.path.realpath(ROOT))) for p in sorted(files)]
    with open(os.path.join(OUT, "SHA256SUMS"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(len(lines), "files")


if __name__ == "__main__":
    main()
