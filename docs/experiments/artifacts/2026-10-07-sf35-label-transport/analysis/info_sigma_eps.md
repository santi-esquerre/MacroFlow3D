# SF-35 informative comparison: same exp(eps Y) amplitude through two parametrizations

- A: `docs/experiments/artifacts/2026-10-07-sf35-label-transport/exports/info/sigma2.25_eps1` (field 2.25, eps 1.0)
- B: `docs/experiments/artifacts/2026-10-07-sf35-label-transport/exports/info/sigma1_eps1.5` (field 1.0, eps 1.5)

| label | max\|dpsi\| | NaN in A | NaN in B | NaN mask mismatches |
|---|---|---|---|---|
| psi1 | 2.931e-14 | 0 | 0 | 0 |
| psi2 | 1.654e-14 | 0 | 0 | 0 |

Expected `<= 1e-12` (recorded, no threshold in the spec): **met** (max over both labels: 2.931e-14)
