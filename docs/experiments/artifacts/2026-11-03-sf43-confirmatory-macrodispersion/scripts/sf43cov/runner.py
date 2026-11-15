"""Worker pool of the SF-43 coverage harness.  A separate module on purpose: under ``python -m sf43cov`` the entry module is
``__main__`` and a spawned worker cannot unpickle a function defined there (the initializer would fail and ``Pool`` would respawn
workers forever); a function of ``sf43cov.runner`` is importable in every interpreter."""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

from sf43cov import mcpath  # noqa: F401

from mcstat import coverage as cov


def _init_worker(out):
    from sf43cov import procs as _p  # noqa: F401  (registers the 'fieller' procedure in the spawned interpreter)
    cov._init_worker(out)


def run(out, workers, max_chunks=None, log=None):
    """Like ``mcstat.coverage.run`` but the workers register the 'fieller' procedure (spawn does not import this package)."""
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    plan = cov.load_plan(out)
    tasks = cov.pending_tasks(out, plan)
    total = sum(cov.nchunks(c) for c in plan["cells"])
    if max_chunks is not None:
        tasks = tasks[:max_chunks]
    log("sf43cov: %d of %d chunks pending, %d workers" % (len(tasks), total, workers))
    for c in plan["cells"]:
        os.makedirs(os.path.join(out, "chunks", c["id"]), exist_ok=True)
    t0, done, busy, last = time.time(), 0, 0.0, 0.0
    if workers <= 1 or not tasks:
        _init_worker(out)
        results, pool = map(cov._work, tasks), None
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(out,))
        results = pool.imap_unordered(cov._work, tasks, chunksize=1)
    try:
        for cid, j, sec in results:
            done += 1
            busy += sec
            if time.time() - last >= 60.0 or done == len(tasks):
                last = time.time()
                el = last - t0
                log("sf43cov: %d/%d chunks, %.0f s elapsed, ETA %.0f s (%s)" % (done, len(tasks), el, el / done * (len(tasks) - done), cid))
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    log("sf43cov: finished %d chunks in %.0f s (%.0f core-seconds)" % (done, time.time() - t0, busy))
    return done
