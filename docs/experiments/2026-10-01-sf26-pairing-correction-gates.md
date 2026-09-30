# SF-26 — heterogeneity gates re-run on the corrected (same-index) equation (14)

- Date: 2026-10-01
- Status: running

## Question

With the coupled residual corrected to the derived same-index pairing
(`A u_i = div_h(q gbar_i) - eta q S_i`, decision 2026-09-30), do the UNCHANGED
sigma_Y^2 >= 1 heterogeneity gates of SF-21/SF-25 pass with the unchanged
implicit stack (adaptive Picard + Anderson + Newton-Krylov), and how do the
Gate 3A metrics (`e_v`, invariance, `e_div`, `|c|` percentiles) change with
resolution (32^3 vs 64^3)?

## Hypothesis (prespecified in the SF-26 spec and bitácora before any run)

The SF-21/SF-25 wall (`lambda_floor_exhausted` at `lambda = 0.5`, `eta = 1`
plateau `r_F ~ 1e-3`) was the least-squares shelf of the crossed (unsolvable)
system. Prediction: the 32^3 sigma^2 = 1 smoke reaches `lambda = 1` with every
accepted stage `r_F <= 1e-6`; the 64^3 quartet reaches `lambda = 1`; `e_v`
drops from ~2.5 % and decreases ~4x from 32^3 to 64^3 at fixed `L/ell`.
Recorded caveat (bitácora 2026-09-30T22:40Z): the corrected Jacobian carries a
near-null gauge cluster at `eta = 1`, so Newton's inner restart-10 GMRES may
budget-exhaust; Picard/Anderson carry the stage in that case (outcome recorded
either way; no gate value changed).

## Build / environment

- Integrated head: `58898bf9b28af97bdbc0eea737b2c9438e61c95f` (delivery branch
  `science/lester-sf26-source-pairing-correction`, base `b86602b` +
  harness `abd873c`).
- Remote: `v100` (2x Tesla V100-PCIE-32GB), preset `v100-release`
  (sm_70), mirror `~/MacroFlow3D`, byte-identical to the local tree
  (`find src tests apps CMakeLists.txt CMakePresets.json | md5sum` matched
  before the base-reference run).
- All long-duration runs are detached `scripts/remote run` jobs, one at a
  time (remote-v100.md Section 0).

## Config(s)

Unchanged fixtures (verbatim, no tuning): `heterogeneity_smoke_sigma025`,
`heterogeneity_smoke_sigma1` (32^3, seed 12345, ell = 8, epsilon 1e-2
degenerate leg, Anderson R5, Newton enabled per SF-25 E1);
`apps/config_streamfunctions_gaussian_smoke32.yaml`;
`apps/config_streamfunctions_gaussian_64_var{025,1,225,4}.yaml` (epsilon leg
to 1e-6; gate = `lambda = 1`).

## Commands

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run sf26-ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait sf26-ctest-full
# byte-compare (base refs generated on the untouched mirror before sync: job sf26-base-refs)
scripts/remote run sf26-bytecompare -- "..."
scripts/remote run sf26-smoke32 -- "./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_gaussian_smoke32.yaml"
scripts/remote run sf26-64var025 -- "./build/v100-release/macroflow3d_pipeline apps/config_streamfunctions_gaussian_64_var025.yaml"
scripts/remote run sf26-64var1 -- "..."; sf26-64var225; sf26-64var4   # sequential, 12 h bound each
```

## Outputs inspected

(pending)

## Result

(pending)

## Caveats

(pending)

## Next step

(pending)
