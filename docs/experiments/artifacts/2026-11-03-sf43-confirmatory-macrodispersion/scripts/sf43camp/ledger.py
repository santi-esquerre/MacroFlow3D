"""Append-only JSON-lines ledger.  One record per line, flock-serialised appends, fsync; a torn last line (SIGKILL mid-write) is ignored
on read.  path=None keeps the records in memory (dry run)."""
import fcntl
import json
import os

from .plan import utc_iso


class Ledger:
    def __init__(self, path, clock):
        self.path, self.clock, self.mem = path, clock, []

    def records(self):
        if self.path is None:
            return list(self.mem)
        out = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                for ln in f:
                    try:
                        out.append(json.loads(ln))
                    except ValueError:
                        pass  # torn tail
        return out

    def append(self, rec):
        t = self.clock()
        rec = dict(rec, t=t, utc=utc_iso(t))
        if self.path is None:
            self.mem.append(rec)
            return rec
        line = (json.dumps(rec, sort_keys=True) + "\n").encode()
        with open(self.path, "ab+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            size = f.seek(0, 2)
            if size:
                f.seek(size - 1)
                if f.read(1) != b"\n":
                    line = b"\n" + line  # terminate a torn tail so it cannot swallow this record
                f.seek(0, 2)
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return rec
