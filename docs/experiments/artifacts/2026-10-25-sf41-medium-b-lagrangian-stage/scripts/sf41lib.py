#!/usr/bin/env python3
"""SF-41 shared readers for the verdict scripts (stdlib only; deterministic; no network).

  kv(line)            -> {key: value} of every `key=value` token of a log line (values are strings)
  lines_of(text, tok) -> the lines whose first token (after an optional `R=<r> ` prefix) is tok
  read(path)          -> file text or None
  rc_of(dir, run)     -> last recorded rc of `run` in dir/exit_codes.tsv (int) or None
  num(s)              -> float or None (nan/inf accepted)
"""
import os
import re

PREFIX = re.compile(r"^R=\d+ ")


def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def kv(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def strip_prefix(line):
    return PREFIX.sub("", line, count=1)


def lines_of(text, tok):
    out = []
    for raw in (text or "").splitlines():
        s = strip_prefix(raw)
        if s.split(None, 1)[:1] == [tok]:
            out.append(s)
    return out


def rc_of(d, run):
    rc = None
    for line in (read(os.path.join(d, "exit_codes.tsv")) or "").splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] == run:
            try:
                rc = int(p[1])
            except ValueError:
                pass
    return rc


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def verdict_word(ok, evaluable=True):
    if not evaluable:
        return "NOT_EVALUABLE"
    return "PASS" if ok else "FAIL"
