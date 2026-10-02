# SF-26 diagnostic probes — scripts and raw outputs

Artifacts for `docs/experiments/2026-10-01-sf26-pairing-correction-gates.md` and
`docs/decisions/2026-10-01-eta1-residual-floor-gauge-degeneracy.md`.

These are **diagnostic numpy/scipy probes written by the orchestrator during SF-26**.
They are not production code, not tests, and are not built or run by CMake/ctest.
They re-implement the solver's stencils independently (harmonic-mean faces,
centered gradients/Hessians, mean-zero projection) so that solver behaviour can be
separated from properties of the discrete equations. Statistics, not realizations,
match the code's fixtures: the fields come from a spectral Gaussian generator inside
the scripts (not SF-18), with `v_rms = 1` and no Darcy solve.

Environment: local WSL (Python 3.13, scipy 1.18) and the V100 host CPU
(numpy 1.26.4, scipy 1.11.4; directory `~/sf26_probe`, outside the repository
mirror; detached `scripts/remote run` jobs). No GPU is used.

## Scripts (`scripts/`) and raw outputs (`raw/`)

| script | command(s) | where | output |
|---|---|---|---|
| `jac_spectrum.py` | `python3 jac_spectrum.py {8,12,16}` | local | `jac_spectrum_12_16.txt` (the 8^3 numbers are recorded in the SF-26 bitácora row of 2026-09-30T22:40Z) |
| `solver_probe.py` (execs `jac_spectrum.py`) | `python3 solver_probe.py 12 1.0` | local | `solver_probe_12.txt` |
| `solver_probe_gauss.py` | `python3 solver_probe_gauss.py <n> <ell/h> <sigma2> <lambda> [seed]` | local (24^3) and V100 host job `sf26-numpy-probe` (24^3, 32^3) | `gauss_24_*.txt`, `gauss_probe_v100_24_32.txt` |
| `probe_long.py` (execs `solver_probe_gauss.py`) | `python3 probe_long.py` | local | `probe_long_24.txt` |
| `probe_eps.py` | `python3 probe_eps.py` | local | `probe_eps_24.txt` |
| `probe_dense_newton.py` | `python3 probe_dense_newton.py 24 8 1.0 0.1125` | V100 host job `sf26-numpy-dense` | `dense_newton_24_l0p1125.txt` (4 of 6 planned Newton steps were recorded before the job ended; the reason was not determined) |
| `probe_refine.py` | `python3 probe_refine.py 24 8 1.0 0.1125` | V100 host job `sf26-numpy-refine` | `refine_probe_v100.txt` |
| `probe_ellh.py` | `python3 probe_ellh.py <n> <ell/h> 1.0 0.1125 0 [seed]` | V100 host job `sf26-numpy-ellh` | `ellh_probe_v100.txt` |
| `probe_spectral.py` | `python3 probe_spectral.py <n> <ell/h> <sigma2> <lambda> 7 [dense]` | V100 host job `sf26-numpy-spectral` | `spectral_probe_v100.txt` |
| `probe_order.py` | `python3 probe_order.py <n> {fd2,fd2c,fd4,sp} 1.0 7 32` | V100 host job `sf26-numpy-order` | `order_probe_v100.txt` |

`raw/v100_ctest_full_extract.txt` is a grep extract of the authoritative V100
full-suite run on the integrated head `58898bf` (job `sf26-ctest-full`,
`~/sf26_ctest_full.log` for failed-test output and
`build/v100-release/Testing/Temporary/LastTest.log` for passed-test output): the
ctest summary, the per-stage tables of both heterogeneity smokes, the
`anderson_stall` / `newton_difficult` summaries, the contract-case tables, the
GMRES contract lines, and the SF-25 instrument evidence lines. The full remote
logs are not versioned and `LastTest.log` is overwritten by the next ctest run.

## Not preserved

`gmres_probe.py` (restarted vs full-recurrence GMRES on an independent 8^3
Jacobian: 2.45e-3 vs 1.9e-14; crossed Jacobian restart-10: 5.5e-11) was written
in a session temporary directory and was lost. Its numbers are recorded in the
SF-26 bitácora (2026-10-01T15:30Z) and the same phenomenon is reproduced inside
the repository by the test case `gmres_dense_lu_oracle` (host textbook restarted
GMRES 2.93e-3, full-recurrence 8.8e-11 on the code's own assembled Jacobian;
see `v100_ctest_full_extract.txt`).

## Known limitations of the instruments (see the experiment note, "Audit")

- `probe_order.py` stops a stage when the best residual has not improved by 1 %
  in 150 iterations: 27 of 100 stages stopped at iteration 201 and 8 exhausted
  the 800-iteration budget while still decreasing. Many of its numbers are
  iteration plateaus or upper bounds, not discretization floors.
- `probe_spectral.py`: the dense-Newton leg on the pseudo-spectral system is not
  valid evidence (exactly singular Jacobian solved with `rcond = 1e-13`).
- One realization (seed 7) for the order and spectral probes; no physics metric
  (`e_v`, invariance) is computed by any probe.
