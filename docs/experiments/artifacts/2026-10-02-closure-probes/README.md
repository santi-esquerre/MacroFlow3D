# Streamline-closure and equation (14) versus Darcy probes — scripts and raw outputs

Artifacts for `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`.

Diagnostic numpy/scipy probes written during the 2026-10-02 roadmap audit. They are not
production code, not tests, and are not built or run by CMake/ctest. They use no project
binary: Darcy and equation (14) are solved pseudo-spectrally on small periodic grids, and
streamlines are integrated with exact trigonometric evaluation of `grad phi`.

Environment: local WSL, Python 3.13, numpy 2.5.0, scipy 1.18.0, CPU. Each case runs in
seconds to a few minutes.

| script | what it measures | raw output |
|---|---|---|
| `closure_probe.py` | return map of the face `x1 = 0` after one period (mean flux exactly `e1`) | `raw/closure.txt` |
| `long_probe.py` (imports `closure_probe`) | transverse displacement after `n` periods | `raw/long_run.txt` |
| `eq14_vs_darcy.py` (imports `closure_probe`) | converged same-index equation (14) solution versus the independent Darcy flow; `curl(c/k)` along and across `c` | `raw/eq14_vs_darcy_n16.txt`, `raw/eq14_vs_darcy_n24_n32.txt` |
| `variational_probe.py` (imports `closure_probe`) | minimization of a collocated discrete dissipation — a NEGATIVE result (the discretization is unsound, see the note) | `raw/variational_naive.txt` |

Commands are of the form `python3 <script> <field>:<eps>:<N>[:<extra>] ...`, run from
`scripts/`; the argument list of every recorded run is the first three columns of each
output line. Fields are defined in `closure_probe.py` (`FIELDS`).

Column `nonuniform_rms` of `closure.txt` is the number quoted in the note (RMS over 16
starting points of the displacement minus its mean).
