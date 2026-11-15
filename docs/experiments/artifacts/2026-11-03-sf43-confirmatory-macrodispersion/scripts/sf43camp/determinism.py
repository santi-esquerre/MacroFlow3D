"""Determinism pairs (smoke test): two arms with identical runner dicts and seeds, typically pinned to lanes A and B, must give the
same bytes.  After both arms are complete and reduced, every raw output file of every shared seed is compared by sha256, and the
reduced rows are compared value by value; the result is `DETERMINISM.txt` (PASS/FAIL per file and per row set) and a REPORT line.
A FAIL never stops the job.

Raw files compared: for a consumer seed every file under the realization's directory `out/realizations/r_XXXX/` (all passes); for a
Par2 seed `out/crossings/realization_XXXX_planes.csv`, `out/stats/realization_XXXX_*` and `out/snapshots/r_XXXX/*`.  Excluded (explicit,
extendable through the plan's `determinism_exclude`): logs and timing files, matched case-insensitively by name:
`EXCLUDE_DEFAULT`.  Reduced rows: all columns of the plane/time rows of the tables of the two arms except `source` and `r` (paths and
the index inside a segment); the sf42red tables hold no timing column.
"""
import csv
import fnmatch
import hashlib
import json
import os

EXCLUDE_DEFAULT = ("*.log", "*timing*", "*texit*", "*mem_phase*", "prune_log.csv", "retained.txt")
ROW_EXCLUDE = ("source", "r")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def seed_files(runner, segment_dir, seed):
    """{relative name: absolute path} of the raw output files of one seed of a segment."""
    r = seed - _read_json(os.path.join(segment_dir, "seedmap.json"))["seed0"]
    out = os.path.join(segment_dir, "out")
    files = {}
    if runner == "consumer":
        roots = [os.path.join(out, "realizations", "r_%04d" % r)]
    else:
        roots = [os.path.join(out, "crossings"), os.path.join(out, "stats"), os.path.join(out, "snapshots", "r_%04d" % r)]
    for root in roots:
        for dp, _, fn in os.walk(root):
            for n in fn:
                if runner == "par2" and os.path.basename(root) in ("crossings", "stats") and "realization_%04d" % r not in n:
                    continue
                rel = os.path.relpath(os.path.join(dp, n), out).replace("r_%04d" % r, "r_XXXX").replace("realization_%04d" % r, "realization_XXXX")
                files[rel] = os.path.join(dp, n)
    return files


def arm_seeds(st, cdir, arm_id):
    """{seed: segment_dir} of the completed seeds of an arm (the latest segment of a seed wins)."""
    out = {}
    for e in st.arms[arm_id].segments:
        for s in e["seeds"]:
            out[s] = os.path.join(cdir, e["segment"])
    return out


def reduced_rows(st, arm_id):
    rows = {}
    for e in st.arms[arm_id].segments:
        for t in (e.get("info") or {}).get("tables", []):
            with open(t, newline="") as f:
                for row in csv.DictReader(f):
                    key = (row["seed"], row["hmax_div"], row["family"], row["weights"], row["basis"], row["set"], row["d_lo"], row["d_hi"])
                    rows[key] = {k: v for k, v in row.items() if k not in ROW_EXCLUDE}
    return rows


def compare_pair(st, cdir, x, y, exclude):
    """-> (lines, ok): lines of DETERMINISM.txt for the pair, ok = every file and row equal and both arms complete."""
    ax, ay = st.arms[x], st.arms[y]
    runner = ax.spec["runner"]
    lines, ok = ["PAIR %s %s" % (x, y)], True
    for a in (ax, ay):
        pend = a.pending() + sorted(a.claimed)
        unred = [e["segment"] for e in a.segments if not e["reduced"]]
        if pend or unred or a.halted:
            lines.append("NOT_RUN arm %s is not complete and reduced (pending %s, unreduced %s, halted %s)" % (a.id, pend[:5], unred[:3], a.halted))
            return lines, False
    sx, sy = arm_seeds(st, cdir, x), arm_seeds(st, cdir, y)
    if set(sx) != set(sy):
        lines.append("FAIL seeds differ: only %s %s, only %s %s" % (x, sorted(set(sx) - set(sy))[:5], y, sorted(set(sy) - set(sx))[:5]))
        ok = False
    nfiles = nbad = 0
    for seed in sorted(set(sx) & set(sy)):
        fx, fy = seed_files(runner, sx[seed], seed), seed_files(runner, sy[seed], seed)
        keep = lambda n: not any(fnmatch.fnmatch(os.path.basename(n).lower(), pat) for pat in exclude)  # noqa: E731
        fx, fy = {k: v for k, v in fx.items() if keep(k)}, {k: v for k, v in fy.items() if keep(k)}
        for name in sorted(set(fx) | set(fy)):
            nfiles += 1
            if name not in fx or name not in fy:
                lines.append("FAIL seed=%d file=%s present only in %s" % (seed, name, x if name in fx else y))
                ok, nbad = False, nbad + 1
            else:
                same = sha256(fx[name]) == sha256(fy[name])
                lines.append("%s seed=%d file=%s" % ("PASS" if same else "FAIL", seed, name))
                if not same:
                    ok, nbad = False, nbad + 1
    rx, ry = reduced_rows(st, x), reduced_rows(st, y)
    nrows = nrowbad = 0
    for key in sorted(set(rx) | set(ry)):
        nrows += 1
        same = key in rx and key in ry and rx[key] == ry[key]
        if not same:
            nrowbad += 1
            ok = False
            lines.append("FAIL reduced row %s" % (key,))
    lines.append("%s reduced rows: %d compared, %d differ" % ("PASS" if not nrowbad else "FAIL", nrows, nrowbad))
    lines.append("%s %s %s: %d raw files compared, %d differ; excluded: %s" % ("PASS" if ok else "FAIL", x, y, nfiles, nbad, ",".join(exclude)))
    return lines, ok


def run(plan, st, cdir, log):
    """Compare every pair of the plan; write <campaign_dir>/DETERMINISM.txt; -> ["determinism: pair x,y PASS|FAIL|NOT_RUN ..."] for the report."""
    pairs = plan.get("determinism_pairs", [])
    if not pairs:
        return []
    exclude = tuple(EXCLUDE_DEFAULT) + tuple(plan.get("determinism_exclude", []))
    out, summary = [], []
    for x, y in pairs:
        try:
            lines, ok = compare_pair(st, cdir, x, y, exclude)
        except Exception as e:  # never stops the job
            lines, ok = ["PAIR %s %s" % (x, y), "FAIL comparison raised %s: %s" % (type(e).__name__, e)], False
        out += lines
        verdict = "NOT_RUN" if any(ln.startswith("NOT_RUN") for ln in lines) else "PASS" if ok else "FAIL"
        summary.append("determinism: pair %s,%s %s (%s)" % (x, y, verdict, lines[-1]))
        log("DETERMINISM pair=%s,%s %s" % (x, y, verdict))
    with open(os.path.join(cdir, "DETERMINISM.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    return summary
