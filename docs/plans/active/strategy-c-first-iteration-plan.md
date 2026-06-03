## Strategy C First Iteration Plan

### Scientific starting point

- On `darcy_small`, Strategy A's scientifically authoritative output is the corrected first-4-mode low invariant subspace.
- No linear 2-field slice from that 4D subspace has produced a scientifically usable consumed-object gauge.
- The dominant mismatch appears before `prepare()` / stepping, so the next test must optimize the realized consumed fields themselves.

### First Strategy C hypothesis

The remaining `rel_mismatch ~ 1.068` floor is caused by a weak gauge representation, not by eigensolver fidelity or transport stepping. A consumed-object-aware alternating fit + projection loop, initialized from the first-4-mode Strategy A subspace, can reduce cross-product mismatch while preserving acceptable invariance and avoiding gauge collapse.

### A->C interface

- **Input:** the first 4 low Strategy A modes on the corrected exact small-grid path.
- **Initialization:** the best consumed-ranked candidate currently available from that 4D subspace.
- **Refinement object:** the realized `PsptaInvariantField`, not host-side gradients.

### Minimal Strategy C variant to test first

1. Start from a consumed-ranked 4D-subspace gauge candidate.
2. Alternate updates:
   - fix `psi2`, compute a local target gradient `g1*` that improves `v ~= grad(psi1) x grad(psi2)`;
   - project `g1*` to an integrable scalar update with a Laplacian solve compatible with PSPTA consumed semantics;
   - apply a relaxed update with backtracking;
   - swap roles and repeat for `psi2`.
3. After every trial update, evaluate the **realized consumed field** with `PsptaInvariantField::compute_quality()`.
4. Accept only updates that improve consumed-object quality without collapsing gradients or destroying invariance.

### What this first variant should preserve

- Strategy A subspace information remains the initialization source.
- Invariance residuals must stay acceptable; we do not accept mismatch reduction by destroying streamline invariance.
- Field gradients and ranges must remain non-degenerate.
- Controls (`uniform_x`, `layered_x`) must remain interpretable.

### Acceptance / rejection diagnostics

The refinement is scored in this order:

1. **Immediately after upload**
   - `quality_rms_r1`, `quality_rms_r2`
   - `rel_rms_mismatch`
   - degeneracy / independence
   - gradient-range collapse guards
2. **After `prepare()`**
   - prepare drift
3. **After stepping**
   - transport drift
   - Newton failure counts

Reject:

- mismatch reductions caused by gradient collapse or trivialization,
- large invariance degradation,
- candidates that look better on host metrics but not in the consumed object.

### Minimal implementation scope

- Implement the existing `RefinementAC` skeleton instead of creating a parallel refiner.
- Keep the corrected exact small-grid path authoritative.
- Use no inlet overwrite. Prefer `GaugeMethod::None` for the first iteration unless a consumed-object-neutral normalization is strictly needed.
- Reuse the existing exact-object harness in `apps/analyze_invariant_quality.cu` to:
  - build the 4D-subspace initialization,
  - run Strategy C,
  - compare pre-prepare / post-prepare / post-transport metrics.

### Scientific success / failure criteria

Success for the first Strategy C iteration means:

- consumed-object mismatch drops materially below the current `~1.068` floor on `darcy_small`,
- invariance stays acceptable,
- gradients do not collapse,
- and prepare/transport behavior does not regress badly.

Failure still produces a useful result if it isolates one of:

- the local fit is not compatible with consumed scalar realization,
- the current projection / BC semantics are too weak,
- the useful gauge requires a richer representation than the first alternating linear update can express.

### Out of scope

- Strategy B
- reviving the probed surrogate
- legacy marching changes
- treating a fixed pair as the Strategy A handoff object

## Outcome

- The first consumed-object-aware Strategy C iteration is now implemented in `RefinementAC`.
- It is scientifically informative but not yet admissible on `darcy_small`.
- Controls:
  - `uniform_x`: no accepted update; mismatch-reducing proposals are rejected for invariance growth
  - `layered_x`: no accepted update; best proposals also fail the invariance gate
- `darcy_small`:
  - no accepted update for any tested `mu` or top consumed-ranked initialization
  - best trial proposals lower mismatch only slightly (`~1.0685 -> ~1.0666..1.0678`) while increasing invariance sums by `O(10^2..10^3)`
  - therefore the current unconstrained local-gradient correction leaves the recovered Strategy A foliation
- The first Strategy C result rules out a naive alternating fit + Poisson projection as a sufficient refinement on the consumed object.
- The next Strategy C iteration should constrain or regularize the refinement against departure from the first-4-mode Strategy A subspace rather than operating as a free voxelwise correction.
