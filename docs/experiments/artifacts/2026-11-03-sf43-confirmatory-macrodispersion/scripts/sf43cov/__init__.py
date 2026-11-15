"""sf43cov: restricted coverage harness of SF-43 (decision H15; Gate 5 procedures of H5).

Qualifies the three Fieller-form procedures of ``sf43ana.gate5`` (Hall 1992 skewness-corrected t, bootstrap-t, and bootstrap-t read at
half the level) under ``normal``, ``stress``, ``stress-`` at the Holm tails of the primary family and their halves, with the SF-42
machinery (``mcstat.coverage``: plan hash, deterministic streams, restartable chunks, Clopper-Pearson verdicts of G5.3).

    python3 -m sf43cov {plan,bench,run,summarize} ...        (run from the directory that contains sf43cov/ and sf43ana/)
    bash sf43cov/run_cov.sh <out_dir> [--workers N]            (detached, resumable host job; nice 19)

Importing this package imports nothing heavy; ``sf43cov.mcpath`` puts the SF-42 ``mcstat`` package on ``sys.path``.
"""
