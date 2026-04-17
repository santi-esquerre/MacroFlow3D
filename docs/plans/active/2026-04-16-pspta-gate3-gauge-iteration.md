# Gate 3 Gauge Iteration — making invariant-pair reconstruction mismatch scientifically interpretable

- Status: active
- Owner: macroflow-agent
- Date: 2026-04-16
- Branch: science/pspta-gate3-gauge
- Parent plan: docs/plans/active/pspta-execution-plan.md (Phase 3)

## Objective

Make the recovered invariant pair `(ψ1, ψ2)` **reconstruction-ready / gauge-ready** so that
`||v − ∇ψ1 × ∇ψ2||` becomes a scientifically interpretable quality signal, then use that
cleaner signal — plus a per-mode transport/regularization energy decomposition and a
sharper spatial localization — to produce an evidence-based next move for Gate 3.

This iteration does **not** change transport, solver backends, or refinement. It only adds
diagnostics and the narrow gauge transform required to separate subspace quality from
gauge freedom.

## Scientific motivation

For any invertible 2×2 linear transform `M` acting on the invariant pair,

```
(ψ1', ψ2') = M · (ψ1, ψ2)
∇ψ1' × ∇ψ2' = det(M) · (∇ψ1 × ∇ψ2)
```

The direction of the cross product in the recovered 2D subspace is fixed up to sign;
only `det(M)` controls its magnitude. Because SLEPc returns eigenvectors with
`||ψ||₂ = 1` (a numerical normalization, not a physical one), the cross-product norm is
unrelated to `||v||`. The existing `rel_rms_mismatch ≈ 1.0` on `uniform_x` (where the
subspace is exactly correct) confirms the mismatch we measure today is dominated by a
global scale/sign gauge, not by structural subspace error.

Therefore:

1. The smallest mismatch achievable by any `(ψ1, ψ2)` in the recovered subspace is

    `residual_floor² = ||v||² − (|⟨v, c⟩| / ||c||)²`   with   `c = ∇ψ1 × ∇ψ2`

    This is the scientifically meaningful lower bound after gauge scaling.

2. The optimal scalar is `α* = ⟨v, c⟩ / ||c||²` (sign handled automatically).

3. After applying `α*` to one of the invariants (e.g. `ψ2 ← α* · ψ2`), the residual
    collapses to `residual_floor`. What remains is true subspace error.

4. Independence / orthogonality and per-mode invariance are handled by the existing
    in-subspace rotation search — that is *not* the gauge fix, but complementary to it.

The energy decomposition answers the parallel question: is `darcy_small` still mediocre
because the smoothness term is dominating, or because the transport near-nullspace itself
is shallow?

```
E_D(ψ) = ⟨ψ, D†W D ψ⟩                   (transport penalty energy)
E_L(ψ) = μ ⟨ψ, L ψ⟩                      (regularization energy)
λ(ψ)   = (E_D + E_L) / ⟨ψ, ψ⟩            (Rayleigh quotient)
```

## Scope

**In scope**

- `apps/analyze_invariant_quality.cu`: add gauge-ready scaling, modal energy decomposition, refined localization.
- `CMakeLists.txt`: add X11/Xau link to the imported `petsc` target (PETSc was rebuilt with X11 support on V100; without this the remote build fails at link time).
- New CSV artifacts under `artifacts/gate3/`:
  - `invariant_quality_gauge.csv`
  - `invariant_quality_energy.csv`
  - `invariant_quality_localization_v2.csv`
- A short Python helper under `scripts/gate3/summarize_gauge_iteration.py` to print a readable table from the three CSVs.
- An experiment record under `docs/experiments/2026-04-16-pspta-gate3-gauge.md`.

**Out of scope**

- Changing the eigensolver, operators, preconditioner, or μ sweep definition.
- Changing invariant ingestion / transport / refinement / gauge fixer in the PSPTA path.
- Any PSPTA smoke run or ensemble run.
- Refinement (deferred to the decision record).

## Non-goals

- We are **not** introducing a new physical gauge for PSPTA transport (the engine still
  uses `InletPlane` via `GaugeFixer`). The gauge transform here lives **only inside the
  diagnostic app** so mismatch becomes interpretable.
- We are **not** reshaping the base Gate 3 API or replacing existing CSVs.
- We are **not** running Gate 4 or Gate 5 work.

## Files / subsystems

| Path | Change |
|------|--------|
| `apps/analyze_invariant_quality.cu` | Add `apply_gauge(...)`, `compute_modal_energy(...)`, refined localization; emit three new CSVs. |
| `CMakeLists.txt` | Add `X11;Xau` to `petsc` imported-target INTERFACE link. |
| `scripts/gate3/summarize_gauge_iteration.py` | New helper to aggregate CSVs into a readable table. |
| `docs/experiments/2026-04-16-pspta-gate3-gauge.md` | New experiment record + diagnosis. |
| `docs/plans/active/2026-04-16-pspta-gate3-gauge-iteration.md` | This plan. |

No code under `src/physics/particles/pspta/`, `src/numerics/`, or `src/physics/flow/` is
modified. Autonomy policy: `apps/` and `docs/` edits are high-autonomy; `CMakeLists.txt`
change is scoped to imported-target link list (no semantics change).

## Gauge-ready transform definition

Given raw eigenvectors `(ψ1, ψ2)` and velocity `v` (cell-centered), the gauge-ready
pair `(ψ̃1, ψ̃2)` is constructed as follows, per (case, μ):

1. **Mean removal** (offsets are in the null direction of `A`, not physically meaningful):
    `ψi ← ψi − mean(ψi)`.
2. **Subspace rotation** (re-use current best-angle rotation so independence diagnostics
    remain monotonic across iterations — only the det-invariant rotation within the pair).
3. **Cross-product alignment**:
    - compute `c = ∇ψ̃1 × ∇ψ̃2` (cell-centered)
    - `α* = ⟨v, c⟩ / (||c||² + ε)`
    - **sign convention**: we do not flip `α` sign because negative `α` is absorbed into
      the sign of one invariant; we just record `α*` including sign.
4. **Apply scaling to one invariant**: `ψ̃2 ← α* · ψ̃2` (so `∇ψ̃1 × ∇ψ̃2 ← α* c`).

Reported per (case, μ, basis_kind):

| Field | Meaning |
|-------|---------|
| `alpha_opt` | Optimal scalar scaling. |
| `v_dot_cross` | `⟨v, c⟩`. |
| `cross_norm` | `||c||₂`. |
| `v_norm` | `||v||₂`. |
| `cos_v_cross` | `⟨v, c⟩ / (||v|| ||c||)`. |
| `rel_residual_before_gauge` | `||v − c|| / ||v||` (what we report today). |
| `rel_residual_after_gauge` | `||v − α* c|| / ||v||` (after gauge scaling). |
| `residual_floor_rel` | `sqrt(1 − cos²(v,c))` — minimum achievable given the subspace. |
| `mean_psi1`, `mean_psi2` | pre-removal means (diagnostic). |

**Acceptance signal for this iteration**: `rel_residual_after_gauge ≈ 0` on `uniform_x`
and `layered_x` (within solver tolerance). If that holds, mismatch is interpretable;
the residual floor on `darcy_small` is then the real subspace-quality signal.

## Modal energy decomposition

For each mode `i ∈ {1, 2}` and each μ:

```
apsi_D = D†WD ψi
apsi_L = L ψi
E_D_i  = ⟨ψi, apsi_D⟩
E_L_i  = μ · ⟨ψi, apsi_L⟩
E_tot  = E_D_i + E_L_i
f_D    = E_D_i / E_tot
f_L    = E_L_i / E_tot
rayleigh = E_tot / ||ψi||²   (should equal eigenvalue λi)
```

Apply `D` and `L` via existing `TransportOperator3D::apply_DTD` and
`LaplacianOperator3D::apply_L`. No new operator code.

Reported per (case, μ, mode): `lambda_i, E_D, E_L, E_tot, f_D, f_L, residual_Ax_lambda_x`.

## Refined localization

For each (case, μ, basis_kind) with basis_kind ∈ {original, best_rotation, gauge_ready}:

1. **Interior vs near-x-boundary**: split by `i ∈ [0, 1]` and `i ∈ [nx-2, nx-1]` vs rest.
2. **Per-x-slice**: for each `i`, compute RMS of `|v·∇ψ1|`, `|v·∇ψ2|`, and relative
    post-gauge mismatch. Write one row per slice.
3. **Cross magnitude quantile**: split by bottom-20% and top-20% of `|c|` (cells where the
    cross-product is small indicate near-degeneracy of the gauge itself).

Emitted as long-format rows in `invariant_quality_localization_v2.csv` with a `region`
column.

## Steps

### Step 1 — Write this plan

Already in progress (this file).

### Step 2 — Add gauge-ready CSV and helper functions in `apps/analyze_invariant_quality.cu`

- Add `struct GaugeReadyMetrics { ... };`
- Add `evaluate_gauge_ready(const Grid3D&, const HostVelocity&, const RawFieldData&,
  double angle_deg)` that applies mean-removal + rotation + α-fit and returns both
  the pre-gauge and post-gauge residuals plus the residual floor.
- Add `write_gauge_header(std::ofstream&)` and a per-row writer.

### Step 3 — Add modal energy decomposition

- Extend `SolveSummary` (or pass the solve out) with access to `ψi` on device and apply
  `TransportOperator3D::apply_DTD` and `LaplacianOperator3D::apply_L`, using existing
  `blas::dot_host` / `nrm2_host` helpers (already in use in the file).
- Write `invariant_quality_energy.csv` with columns
  `case, mu, mode, eigenvalue_solver, rayleigh_recomputed, E_D, E_L, E_total, f_D, f_L`.

### Step 4 — Add refined localization

- Add structs `InteriorBoundarySplit`, `PerSliceStats`, `CrossMagnitudeSplit`.
- Compute all three from the post-gauge pair (and also emit the same for `original` so
  the before/after contrast is in the same file).
- Write `invariant_quality_localization_v2.csv`.

### Step 5 — Fix CMakeLists X11 link

- Find `X11` and `Xau` via `find_library` (system paths).
- Add them to the `petsc` imported-target INTERFACE_LINK_LIBRARIES when `MACROFLOW3D_ENABLE_PETSC=ON`.
- Keep optional: if not found, warn but do not fail (local WSL builds without PETSc are unaffected).

### Step 6 — Remote rebuild

- `scripts/rsync_to_v100.sh` the worktree.
- Launch `cmake --build build/v100-petsc -j2 --target analyze_invariant_quality` under a
  persistent tmux session so a disconnected SSH doesn't kill the job.

### Step 7 — Remote run + pull artifacts

- Run the rebuilt binary.
- `rsync -av v100:~/MacroFlow3D/artifacts/gate3/ ./artifacts/gate3/` into the worktree.

### Step 8 — Summarize and decide

- Run `scripts/gate3/summarize_gauge_iteration.py` to get a compact table.
- Write `docs/experiments/2026-04-16-pspta-gate3-gauge.md` with:
  - before/after residual on `uniform_x`, `layered_x`, `darcy_small`
  - E_D / E_L fractions at each μ
  - localization summary
  - explicit A / B / C / D decision for Gate 3 next move.

### Step 9 — Commit

Several commits are preferred over one:

1. `feat(gate3): add gauge-ready residual to analyze_invariant_quality`
2. `feat(gate3): add modal transport vs regularization energy decomposition`
3. `feat(gate3): add refined localization diagnostics`
4. `chore(build): link X11 + Xau to imported petsc target`
5. `chore(scripts): add Gate 3 gauge iteration summarizer`
6. `docs(gate3): record gauge iteration findings and decision`

## Commands

### Local build sanity (without PETSc — only checks the file compiles cleanly where PETSc is gated off)

```bash
cmake --preset wsl-debug
cmake --build build/wsl-debug -j --target run_operator_tests macroflow3d_pipeline
ctest --test-dir build/wsl-debug --output-on-failure -R operator_tests
```

(`analyze_invariant_quality` is behind `#ifdef MACROFLOW3D_HAS_PETSC` — the local debug
build only needs to remain syntactically clean, not produce the binary.)

### Remote build and run

```bash
bash scripts/rsync_to_v100.sh
ssh v100 'tmux new -d -s gate3-gauge "cd ~/MacroFlow3D && cmake --build build/v100-petsc -j2 --target analyze_invariant_quality 2>&1 | tee logs/gate3-gauge-build.log"'
# monitor
ssh v100 'tail -f ~/MacroFlow3D/logs/gate3-gauge-build.log'
# once built
ssh v100 'cd ~/MacroFlow3D && mkdir -p artifacts/gate3 && ./build/v100-petsc/analyze_invariant_quality 2>&1 | tee logs/gate3-gauge-run.log'
rsync -av v100:~/MacroFlow3D/artifacts/gate3/ artifacts/gate3/
```

## Validation

This is a Gate 3 diagnostic-only change in `apps/`:

- Gate 1 (build) applies.
- Gate 2 (operators) is unaffected — no operator is changed.
- Gate 3 (scientific): the new diagnostics are the validation.
- Acceptance signal for this iteration is described under
  "Gauge-ready transform definition / Acceptance signal" above.

## Risks / regressions

- **X11 link change**: could make the build require X11 where it previously did not. Guard
  by keeping X11 optional (`find_library` returns NOTFOUND → warn and skip; let PETSc's
  X usage fail at link time only when a rebuilt PETSc demands it).
- **New CSV files**: no consumers yet — additive.
- **Memory**: modal energy decomposition allocates two temporary device buffers sized like
  `psi`. On the current (16³, 12³) grids this is negligible (~32 KB).
- **No risk** to `src/physics/`, `src/numerics/`, `src/runtime/`, engine, or solvers.

## Done criteria

1. `apps/analyze_invariant_quality.cu` emits three new CSVs with the columns above.
2. Remote rebuild succeeds.
3. Remote run produces complete artifacts for all three cases × five μ values.
4. Summarizer script prints per-case before/after gauge-ready residuals.
5. `uniform_x` and `layered_x` post-gauge residual drops to at most 1e-2 relative (vs 1.0
    before), confirming the subspace is right and gauge was the dominant source of
    mismatch.
6. `darcy_small` reports a genuinely informative post-gauge residual (either small → subspace
    captures most of the flow, or large → subspace is materially incomplete).
7. Modal energy decomposition and localization produce per-mode, per-μ numbers for all cases.
8. `docs/experiments/2026-04-16-pspta-gate3-gauge.md` exists and selects one of A/B/C/D
    with quantitative justification.
9. No change to PSPTA transport, engine, or solver behavior.

## Next-move decision framework (for Step 8)

Pick **one** and cite evidence:

- **A. Refinement is now justified** — only if `darcy_small` post-gauge residual is
  modest (≲ 30%), E_L fraction is already large at small μ (so smoothness is carrying
  weight we could trim), and localization does not show a structural defect.
- **B. Transport/regularization rebalance** — if μ sweep shows that residual tracks
  E_L/E_tot with strong monotonicity and there is a clear μ window where the subspace
  is better, we rework `A = D†WD + μL` weighting before refinement.
- **C. Deeper operator/discretization investigation** — if post-gauge residual remains
  high (> ~50%) with strong structural localization (e.g. boundary-dominated, or tied to
  regions of high `|c|` where cross magnitude is largest), we reopen operator quality.
- **D. Narrower blocker** — a specific, reproducible pathology emerges from the data
  that does not fit A/B/C.
