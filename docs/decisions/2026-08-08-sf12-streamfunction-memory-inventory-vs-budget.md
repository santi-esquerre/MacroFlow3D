# SF-12 streamfunction memory inventory exceeds the 24.6-field budget: recorded inventory and ownership-redesign options

- Status: proposed
- Date: 2026-08-08

## Context

`docs/plans/active/lester-eq14-streamfunction-solver-plan.md` (`## Architecture
and memory constraints`) states: "the sequential Picard design must target
about 24.6 fine-grid-equivalent scalar fields (about 3.1 GiB), or about
3.6-4 GiB including `Y` and Darcy velocity", with one `256^3` double field at
128 MiB (`n=16,777,216`, 8 B/element -> 134,217,728 B/field).

`docs/plans/active/lester-eq14/increments/SF-12-public-api-workspace.md`
("Failure and rollback policy"): "If the 24.6-field budget cannot be met,
record each extra field and redesign ownership before SF-13."

SF-12 T01 (`71d882810d6e1bb5e1f739f0c15c4b6936f5439b`) implemented
`StreamfunctionFields`/`StreamfunctionWorkspace` with an exact, allocation-free
`estimate_streamfunction_memory()` and matching `allocated_device_bytes()`
introspection on every owned/sub-owned type. The estimator-equals-actual
equality was verified by direct comparison against a prepared workspace at
`16^3` and `32^3` (per-increment worker report), extended to `64^3` under
orchestrator audit, all on the local 4 GiB GPU; `128^3`/`256^3` are pure-host
estimator evaluations (the full `256^3` workspace exceeds local VRAM).

This record independently recomputes the field inventory directly from
`src/physics/streamfunctions/{ResidualEvaluator.cuh/.cu, Diagnostics.cuh/.cu,
StreamfunctionWorkspace.cuh/.cu}`, `src/numerics/solvers/pcg.cuh`, and
`src/multigrid/mg_types.hpp`. Every category below was cross-checked line by
line against `DeviceBuffer::resize()` call sites and the workspaces' own
`estimate_device_bytes()` implementations; the totals reconcile exactly to
byte precision with the totals recorded in the SF-12 T01 orchestrator audit (no
discrepancy found).

**Measured reality at `256^3`** (`n = 16,777,216` cells, `sizeof(real) = 8`,
one field = 134,217,728 B):

| Category | Owner | Bytes | Fields (bytes / 134,217,728) |
|---|---|---:|---:|
| `StreamfunctionFields` (`u1`, `u2`) | `StreamfunctionFields` | 268,435,456 | 2.000 |
| Solve-path scratch (`q`, `rhs1`, `rhs2`, `f1`, `f2`, `v_psi` U/V/W CompactMAC) | `StreamfunctionWorkspace` | 1,075,314,688 | 8.012 |
| `StreamfunctionResidualWorkspace` (SF-10) | `StreamfunctionWorkspace` (private) | 2,818,582,174 | 21.001 |
| top-level `AffinePeriodicRhsWorkspace` | `StreamfunctionWorkspace` | 2,855 | ~0.000 |
| `ProjectedPCGWorkspace` (SF-04) | `StreamfunctionWorkspace` | 671,091,471 | 5.000 |
| `MGHierarchy` (SF-05, 4 levels) | `StreamfunctionWorkspace` | 613,416,960 | 4.571 |
| `ProjectedPositiveMGPreconditioner` mean-zero workspaces | `StreamfunctionWorkspace` | 11,292 | ~0.000 |
| `StreamfunctionDiagnosticsWorkspace` (SF-11) | `StreamfunctionWorkspace` (private) | 3,623,881,887 | 27.003 |
| **Total** | | **9,070,736,783** | **67.58** |

`9,070,736,783 / 1024^3 ~= 8.45 GiB`. At `128^3`: total `1,134,065,039` B,
`67.60` fine-grid-equivalent fields (nearly identical ratio, as expected since
per-cell scratch categories dominate and scale with `n`, while the MG
hierarchy's sub-1-field coarse-level overhead is grid-size-invariant in field
units).

## Recorded inventory: budget-plausible vs. extra, with root cause

The plan's 24.6 sentence predates SF-08 through SF-11 and plausibly
decomposed (this decomposition is **inferred, not sourced from the plan
text**, and is offered only as a plausibility check): approximately 23 core
production/solver fields (`u1`, `u2`, `q`, `rhs1`, `rhs2`, one-`A`-apply
scratch, one set of 6 total gradients, one set of 3-6 nonlinear-source
scratch fields, `v_psi`) + 4 finest-level MG fields counted as 1 (only the
`q` coefficient copy anticipated, not `x`/`b`/`r`) + 0.571 coarse-level MG
overhead ~= 24.57. It did **not** anticipate the following, each confirmed in
code (status: confirmed in code):

1. **SF-08 `HessianVectorBOutput` output contract** (`ResidualEvaluator.cuh`
   lines 100-128, `.cu` lines 236-272): materializes 6 Hessian-vector-product
   scratch fields (`h2g1_{x,y,z}`, `h1g2_{x,y,z}`) as accepted, "register
   analog" outputs, even though only the 3-field `B` result feeds the
   sources. +6 fields, inside the 21-field residual workspace.
2. **`MGLevel` allocates `x`, `b`, `r`, `coefficient` at every level**
   (`mg_types.hpp` line 27-40), not just a coefficient copy. The finest level
   alone adds `x`, `b`, `r` (+3 fields) beyond the anticipated coefficient
   field.
3. **Duplicated gradient ownership**: `StreamfunctionResidualWorkspace`
   (SF-10, `ResidualEvaluator.cuh` line 156) and
   `StreamfunctionDiagnosticsWorkspace` (SF-11, `Diagnostics.cuh` line 218)
   **each** privately own their own 6 total-gradient fields (`psi{1,2}_{x,y,z}`
   / `g{1,2}{x,y,z}`), computed independently by
   `enqueue_total_streamfunction_gradients` in each module. 12 total, ~6 of
   them duplicated relative to a single shared-gradient design.
4. **SF-11 diagnostics-only scratch** (`Diagnostics.cuh` lines 217-244,
   `.cu` lines 596-628): 9 unique-face fields (`unique_vpsi_{u,v,w}`,
   `unique_vd_{u,v,w}`, `diff_{u,v,w}`) + 6 cell-centered velocity fields
   (`vpsi_c{x,y,z}`, `vd_c{x,y,z}`) + 6 cell-centered diagnostic scalar fields
   (`magnitude_field_`, `theta_field_`, `dot1_field_`, `dot2_field_`,
   `divergence_field_`, `abs_c_field_`) = 27 fields total (21 beyond its own
   6 gradients), none anticipated by the budget, which appears to have
   assumed diagnostics were cheap or reused solve-path buffers.
5. **Caller-owned residual outputs and reconstruction buffers**: `f1`/`f2`
   (+2 fields) and the `v_psi` CompactMAC triple (+~3.012 fields, the extra
   0.012 from the `(nx+1)*ny*nz` etc. duplicate-plane face sizing) held in
   `StreamfunctionWorkspace` rather than transiently.
6. `q` (+1) and the `rhs1`/`rhs2` pair (+2) are additional solve-path fields
   present in the 8.012-field scratch bucket alongside `v_psi`.

Net effect: SF-10's residual workspace alone (21 fields) is already close to
the entire original 24.6-field budget, and SF-11's diagnostics workspace (27
fields) adds a comparable amount on top, because both were accepted,
audited increments with their own independently justified numerical
contracts (status: accepted scope, each under its own increment) that were
never reconciled against the plan's aggregate memory sentence.

## Decision (proposed)

1. **Record this inventory as the accepted SF-12 outcome.** SF-12 does not
   retroactively rewrite the accepted SF-08, SF-10, or SF-11 numerical
   contracts (Hessian-vector output shape, private gradient ownership,
   diagnostics scratch shape) merely to chase the budget figure, per the
   project's single-purpose-change rule (`AGENTS.md` "Keep changes
   single-purpose"). Status: proposed for human adjudication on the SF-12 PR.

2. **Ownership-redesign options for SF-13 (not decided here; awaiting human
   review):**

   - **(a) Accept the larger footprint for the bring-up phase, defer
     optimization to SF-23 (GPU optimization increment).** `8.45 GiB` fits a
     16 GiB V100 at `256^3` with headroom; the local 4 GiB GPU caps
     production-workspace bring-up at `128^3` (`1,134,065,039 B ~= 1.06 GiB`,
     well within budget) or smaller. Does not touch any accepted contract.
     **Recommended for SF-13.**
   - **(b) Borrow, don't own: expose read-only device views from
     `StreamfunctionResidualWorkspace`'s 6 total-gradient fields so
     `StreamfunctionDiagnosticsWorkspace` can reuse them instead of computing
     and owning its own 6.** Saves ~6 fields (~9% of the total). Requires an
     additive accessor increment (new read-only view API on
     `StreamfunctionResidualWorkspace`); does not change either module's
     existing numerical contract, only its storage source. **Recommended as
     the first optimization candidate**, ahead of (c) and (d).
   - **(c) Make the SF-08 Hessian-vector scratch outputs optional in a
     revised `HessianVectorBOutput` contract** (materialize only `B`,
     discard the 6 intermediate fields without host-visible storage). Saves 6
     fields. This **changes an accepted numerical/API contract** from SF-08
     and would need its own increment plus revalidation against SF-08's
     acceptance tests, not a silent SF-13 side effect.
   - **(d) Lifetime-split the solve path from the diagnostics path**
     (prepare/allocate `StreamfunctionDiagnosticsWorkspace`'s 27 fields only
     at report time, not for the whole nonlinear loop lifetime). This
     violates the project's "no allocation in hot loops" rule
     (`AGENTS.md` "Performance rules") unless diagnostics are staged to run
     only between Picard iterations (not inside them), which would need
     explicit documentation of exactly when diagnostics run relative to the
     hot loop. Not selected without that staging design.

   Options (a) and (b) preserve every accepted contract; option (c) requires
   revising an accepted contract; option (d) requires a documented staging
   change to avoid violating the no-allocation-in-hot-loop rule. No option is
   selected as final here; SF-13 must not assume any of (b)-(d) is in effect
   until a human explicitly accepts it.

## Consequences

- **SF-13 capacity** (exact, from this inventory): `128^3` total
  `1,134,065,039 B` (~1.06 GiB, 67.60 fine-grid-equivalent fields); `256^3`
  total `9,070,736,783 B` (~8.45 GiB, 67.58 fine-grid-equivalent fields).
- **Local WSL / 4 GiB GPU**: `256^3` does not fit; `128^3` fits comfortably
  and is the practical local ceiling for full-workspace bring-up runs.
- **Remote V100 (16 GiB)**: `256^3` fits with ~7.5 GiB headroom for other
  device allocations (K/Y fields, transport, ensemble state), consistent with
  option (a) being usable through at least the SF-13 bring-up phase.
- The plan's `24.6`-field / `~3.1 GiB` budget figure in
  `docs/plans/active/lester-eq14-streamfunction-solver-plan.md` remains
  as-written; it is not silently revised by this record. A human reviewer
  must explicitly accept a revised figure (or an ownership redesign that
  restores the original figure) before it is edited.
- SF-13 must plan its `256^3` capacity assumptions against the measured
  `8.45 GiB` figure, not the plan's original `3.1-4 GiB` figure, until this
  decision is resolved.
