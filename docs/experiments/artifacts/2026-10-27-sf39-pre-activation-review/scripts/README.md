# Scripts: SF-39 pre-activation review (CPU calculations; NOT SF-39 evidence)

Used by [`../../../2026-10-27-sf39-pre-activation-review.md`](../../../2026-10-27-sf39-pre-activation-review.md) (section 5) and decision
`docs/decisions/2026-10-27-sf39-pre-activation-decisions.md`. Local CPU, seconds each, no GPU, no build. Run from the repository root.
Environment of the recorded outputs: Python 3.13.13, numpy 2.5.0, scipy 1.18.0. Fixed RNG seed (`transfer.py`); no timestamps.
No script reads or writes SF-39 data (none exists); `sign_planes.py` only reads committed SF-41 artifacts.

| script | command | expected output summary |
|---|---|---|
| `firstorder2.py` | `python3 -I docs/experiments/artifacts/2026-10-27-sf39-pre-activation-review/scripts/firstorder2.py` | first-order (Dagan-type) purely advective `alpha_L(tau)/(sigma^2 lambda)` and `alpha_T(tau)/(sigma^2 lambda)` for `C_Y = sigma^2 exp(-(r/lambda)^2)`, 3D isotropic, by quadrature in wavenumber; first line `sqrt(pi)/2 = 0.8862269254527579`; `alpha_T` peaks at `4.460e-02` (`tau = 1`) and `alpha_T tau^2 -> 0.2215 = sqrt(pi)/8`; then the `[0.75 t_b, t_b]` window means (`2.953e-05` at `t_b = 100`, i.e. `0.2953/t_b^2`) |
| `transfer.py` | `python3 -I docs/experiments/artifacts/2026-10-27-sf39-pre-activation-review/scripts/transfer.py` | Monte Carlo (4e5 draws, seed 1) pass probability of the 2026-10-23 D15 transfer rule (same sign and ratio in `[1/2, 2]` of two noisy relative differences): 0.203 / 0.192 with no effect, 0.381 / 0.737 / 0.974 with a true 5 % effect at SE 0.056-0.037 / 0.02 / 0.01, 0.110 (narrow 10 %, twin 0), 0.362 (true 1 %, SE 1 %) |
| `sign_planes.py` | `python3 -I docs/experiments/artifacts/2026-10-27-sf39-pre-activation-review/scripts/sign_planes.py docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/raw/sf41_campaign/evladder` | 18 lines (eps 1, 2 x seeds 6101-6103 x `lambda/h` 16, 32, 64): for each, the number of the 14 planes of `moments_planes.csv` where `var_x2 + var_x3` of the grid is below the `lambda/h = 128` (`g256`) reference, and the relative difference at the first / last plane; the count is the same for the three `lambda/h` of a seed (1/14, 1/14, 14/14 for eps 1; 3/14, 0/14, 5/14 for eps 2) and the relative differences shrink by about 4 per doubling (order ~2) |

Expected outputs are pasted verbatim in the note's "Observed outputs".
