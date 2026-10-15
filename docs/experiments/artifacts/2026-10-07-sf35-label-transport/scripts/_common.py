"""Shared helpers of the SF-35 campaign digests (standard library + numpy only).

Not a command-line tool.  Every digest imports it from its own directory.
"""
import json
import math
import os
import re
import sys

NAN = float("nan")


def load_json(path):
    """Load a driver --summary JSON (JSON null is converted by num())."""
    with open(path) as f:
        return json.load(f)


def num(x):
    """JSON number or null -> float (null / None / missing -> NaN)."""
    if x is None:
        return NAN
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def get(d, *keys, default=None):
    """Nested dict access that returns `default` when a key is missing."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def fnum(x, fmt="%.3e"):
    x = num(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return fmt % x


def order(q_coarse, q_fine):
    """Observed order log2(q_N / q_2N); NaN when undefined (non-finite or non-positive)."""
    a, b = num(q_coarse), num(q_fine)
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
        return NAN
    return math.log2(a / b)


def md_table(header, rows):
    def esc(c):
        return str(c).replace("|", "\\|")  # a literal | inside a cell must not split the cell

    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


def write_out(text, path):
    """Write `text` to `path` (if given) and always print it."""
    if path:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            f.write(text if text.endswith("\n") else text + "\n")
    print(text)


def parse_time_file(path):
    """/usr/bin/time -v file -> dict(wall_s, max_rss_kB); missing / unreadable -> NaN entries."""
    res = {"wall_s": NAN, "max_rss_kB": NAN}
    try:
        with open(path) as f:
            for line in f:
                if "Maximum resident set size" in line:
                    res["max_rss_kB"] = float(line.strip().split()[-1])
                elif "Elapsed (wall clock) time" in line:
                    t = line.strip().split()[-1]
                    s = 0.0
                    for p in [float(p) for p in t.split(":")]:
                        s = s * 60.0 + p
                    res["wall_s"] = s
    except (OSError, ValueError):
        pass
    return res


def n_of(path, js=None):
    """Grid size N of a campaign file: JSON field N, else the N<digits> of the file name."""
    if js is not None and "N" in js:
        return int(js["N"])
    m = re.search(r"N(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def log_lines(path, prefix):
    """Lines of a log that start with `prefix` (fallback source when a JSON is absent)."""
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if line.startswith(prefix):
                    out.append(line.rstrip("\n"))
    except OSError:
        pass
    return out


def kv(line):
    """key=value tokens of a driver log line (values kept as strings)."""
    return dict(re.findall(r"([A-Za-z0-9_.\-]+)=([^\s,|]+)", line))


def fail(msg, code=2):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(code)


def rel(path):
    """Path relative to the current directory (the repository root in the campaign) for printed reports."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return path
