# Artifact: SF-41 pre-activation review — design emulations (numpy; NOT SF-41 evidence)

Used by [`../../2026-10-24-sf41-pre-activation-review.md`](../../2026-10-24-sf41-pre-activation-review.md) and decision
[`2026-10-24-sf41-sf39-planning-decisions.md`](../../../decisions/2026-10-24-sf41-sf39-planning-decisions.md), addendum E8-E12.

What this is: CPU emulations, in numpy/scipy float64 without FMA, of the mathematics fixed for SF-41 before its activation (the
not-a-knot non-periodic cubic B-spline on the cell-centred lattice, the SF-28 periodic spline, the DP5(4) arclength tracker with plane
landing, an RT0 / Pollock tracker) and of round-off probes. They predict what a correct implementation must reproduce and where a
threshold would fail for an established reason. What this is not: GPU results, SF-41 acceptance evidence, or any `alpha` statement.

## Files

| script | output | threshold | content |
|---|---|---|---|
| `scripts/a_spline.py` | (library) | A | 1-D/3-D not-a-knot and periodic splines, solver variants (reduced, dense, sparse), evaluation of value/gradient/Hessian |
| `scripts/a_ladder.py` | `analysis/a_ladder.txt` | A | scipy validation; natural-field and 300 random-field ladders; field selection (C1-C6; C3 recorded per E11); `x`-only sub-ladders; reproduction; partition of unity; CPU-vs-GPU surrogate |
| `scripts/b_roundoff.py` | `analysis/b_roundoff.txt` | B | periodic vs non-periodic representation of the same periodic samples: interior round-off floor, end-effect profile, per-knot factor; three drivers, three prefilter pairs, `lambda/h` 32 and 8 |
| `scripts/c_common.py` | (library) | C | splines, DP5(4) stepper, exact discrete layered heads, plane pass with landing, quadrature |
| `scripts/c2_tau.py` | `analysis/c2_tau.txt` | C2 | SF-38 F2 reproduction (4.620e-9, 4.771e-10); C2 candidate and variants |
| `scripts/c3_shift.py` | `analysis/c3_shift.txt` | C3 | transverse shift on four written fields, random and dyadic points |
| `scripts/c4_lobe.py` | `analysis/c4_lobe.txt` | C4 | harmonic lobe field: exact stream-function classification vs DP5 on the spline vs RT0 on analytic face fluxes; lattice and margin scan |
| — | `analysis/ab_summary.md`, `analysis/c_summary.md` | A, B, C | the workers' tables and their comparison with the reviewers' earlier in-memory numbers |

## Reproduce

From this directory, on a CPU (no GPU, no build):

```bash
for s in a_ladder b_roundoff c2_tau c3_shift c4_lobe; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 15 taskset -c 10-15 python3 -I scripts/$s.py > analysis/$s.txt
done
```

Fixed RNG seeds; no timestamps; the orchestrator re-ran every script from a separate checkout and compared the outputs byte for byte
(result in the governance PR). Environment: numpy 2.5.0, scipy 1.18.0.

## Provenance

Written by two `increment-worker` nodes in isolated worktrees from `master a3d98ae` (commits cherry-picked into the governance branch),
audited by the orchestrator. The C1 -> C3 change of the recommended A field after the `x`-closure share diagnostic (b2) was seen is
disclosed in `a_ladder.txt` and in the review note.
