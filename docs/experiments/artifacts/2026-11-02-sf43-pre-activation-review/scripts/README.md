# Scripts of the SF-43 pre-activation review

Design calculations only: no GPU, no remote host, no SF-43 data. Python 3.11+ with numpy and scipy; the SF-42 tooling where stated
(`S42=docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling`, `S39=docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study`).

| script and command (from the repository root) | what it computes | review section |
|---|---|---|
| `python3 $A/firstorder_window.py` | first-order transverse dispersivity of SF-43's plane estimand in an unbounded medium (windows `[25, 94]`, halves, `[94, 194]`) | 2.1, H3 |
| `python3 $A/firstorder_box.py` | the same in SF-43's periodic boxes (field periodic along `x` with the box length and transversally with the width) | 2.1, 3.b, H3, H17, H9 positive control |
| `python3 $A/design_cost.py` | GPU-h, two-GPU stage (lane classes), wall times of variants A, B, C from SF-42's measured per-realization costs | 3.g, 4 |
| `python3 $A/lattice_check.py --sf42-scripts $S42/scripts` | common admissible lattice sizes per amplitude and source | H10 |
| `python3 $A/gate4_power_sim.py <components\|pipeline\|mde\|forced-pL\|numerical\|alloc> --sf42-scripts $S42/scripts --sf39-reduced $S39/analysis/reduced` | Gate 4 robust-limit simulations with `mcstat` (`--R 20` for a quick run) | 3.c, 3.d, H16 |
| `python3 $A/gate5_options.py <moments\|fieller-coverage\|skewcorr\|sizing\|g7\|runtime> --sf42-scripts $S42/scripts --pilots $S42/pilots --coverage $S42/coverage` | Gate 5 pilot moments, Fieller-form re-judging of SF-42's coverage cells, skewness-corrected prototype, sizing, G7 effect sizes, runtime correlations | 3.d, 3.e, H5 |

`A=docs/experiments/artifacts/2026-11-02-sf43-pre-activation-review/scripts`. The per-realization planning seconds in `design_cost.py`
come from the SF-42 cost audit (measured `seconds.total` of the pilots' `RUN.json`, interpolated in `M^2` or adjusted with the refit's
tracing coefficient where the configuration was not measured; three values are estimates, marked in the code). `gate4_power_sim.py`'s
`sigma^2` 2.25 / 4 cells reproduce the review within Monte Carlo error (the original seeds came from a salted hash); the review quotes the
script. `gate5_options.py` uses the Holm levels of `m = 7` (decision G5.6), as the SF-42 coverage plan did; the `m = 8` tails of
decision H4 are added by the restricted coverage run (decision H15).
