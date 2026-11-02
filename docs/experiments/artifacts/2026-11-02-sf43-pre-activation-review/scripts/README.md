# Scripts of the SF-43 pre-activation review

Design calculations only: no GPU, no remote host, no SF-43 data. Python 3.11+ with numpy and scipy; the SF-42 tooling
(`docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts`) where stated.

| script | what it computes | review section |
|---|---|---|
| `firstorder_window.py` | first-order transverse dispersivity of SF-43's plane estimand (windows `[25, 94]`, halves, `[94, 194]`) | 2.1, H3 |
| `design_cost.py` | GPU-h, two-stage schedule and wall time of variants A, B, C from SF-42's measured per-realization costs | 3.g, 4 |
| `lattice_check.py --sf42-scripts <dir>` | common admissible lattice sizes per amplitude and source | H10 |
| `gate4_power_sim.py` | Gate 4 robust-limit simulations with `mcstat` on SF-39's reduced tables | 3.c, 3.d |
| `gate5_options.py` | Gate 5 pilot moments, Fieller-form coverage re-judging, skewness-corrected prototype, sizing, G7 effect sizes | 3.d, 3.e |

The per-realization planning seconds in `design_cost.py` are copied from the SF-42 cost audit (measured `seconds.total` of the pilots'
`RUN.json`, interpolated in `M^2` or adjusted with the refit's tracing coefficient where the exact configuration was not measured; SF-42
note section G).
