"""jobs_r4.tsv from the `scripts/remote` job logs (slot held from the 'GPU snapshot' line to 'finished exit=').

    python3 -m pilots.jobs <kind:job:log path> ... > jobs_r4.tsv        (kind = CPU | CUDA)

Columns as ``verification/jobs_r2.tsv``: job, start_utc, end_utc, wall_s, gpu_h (wall hours holding the slot), exit, gpu, kind.
"""
import datetime
import re
import sys

_TS = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\]")


def parse(path):
    start = end = None
    exit_code = gpu = None
    with open(path, errors="replace") as f:
        for line in f:
            m = _TS.match(line)
            if not m:
                continue
            if start is None:
                start = m.group(1)
            e = re.search(r"finished exit=(\d+)", line)
            if e:
                end, exit_code = m.group(1), int(e.group(1))
            g = re.search(r"chose GPU (\d+)", line)
            if g:
                gpu = int(g.group(1))
    return start, end, exit_code, gpu


def fmt(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def rows(specs):
    out, tot = [], 0.0
    for spec in specs:
        kind, job, path = spec.split(":", 2)
        s, e, rc, gpu = parse(path)
        wall = int((fmt(e) - fmt(s)).total_seconds())
        tot += wall
        out.append("%s\t%s\t%s\t%d\t%.3f\t%d\t%s\t%s" % (job, s, e, wall, wall / 3600.0, rc, gpu, kind))
    out.append("TOTAL_R4\t\t\t%d\t%.3f" % (tot, tot / 3600.0))
    return ["job\tstart_utc\tend_utc\twall_s\tgpu_h\texit\tgpu\tkind"] + out


if __name__ == "__main__":
    print("\n".join(rows(sys.argv[1:])))
