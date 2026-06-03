# PSPTA Runtime Authority Map

## Purpose

Record the current end-to-end PSPTA transport/invariant pipeline, distinguish the runtime-authoritative legacy path from the planned Strategy A/C path, and define the next execution steps in order of scientific evidence value.

## Commands executed

- `git worktree add /home/sesquerre/src/MacroFlow3D/.agents/worktrees/pspta-pipeline-map -b science/pspta-pipeline-map`
- `rg -n "transport\\.method|method:\\s*pspta|pspta" apps src/io src/runtime src/physics`
- `rg -n "march|marching|legacy|prototype|psi reconstruction|invariant" src/physics/particles/pspta src/runtime src/io apps`
- `ctest --test-dir /home/sesquerre/src/MacroFlow3D/build/wsl-debug --output-on-failure -R operator_tests`
- `/home/sesquerre/src/MacroFlow3D/build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml`
- `git submodule update --init --recursive`
- `./scripts/remote sync`
- `./scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"`
- `./scripts/remote exec -- "ctest --test-dir build/v100-release --output-on-failure -R operator_tests"`
- `./scripts/remote run pspta-small -- "./build/v100-release/macroflow3d_pipeline apps/config_pspta_small.yaml"`
- `./scripts/remote status pspta-small`
- `./scripts/remote tail pspta-small --lines 200 --no-follow`
- `./scripts/remote exec -- "find output_pspta_small -maxdepth 3 -type f | sort | sed -n '1,120p'"`
- `./scripts/remote exec -- "awk ... output_pspta_small/snapshots/step_00000005.csv output_pspta_small/snapshots/step_00000500.csv"`
- `./scripts/remote exec -- "bash src/external/scripts/build_petsc_slepc.sh clean && bash src/external/scripts/build_petsc_slepc.sh"`
- `./scripts/remote exec -- "cmake --preset v100-petsc"`
- `./scripts/remote exec -- "cmake --build build/v100-petsc -j"`
- `./scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc"`
- `./scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver"`
- `./scripts/remote exec -- "./build/v100-petsc/validate_slepc_eigensolver"`
- `cmake --preset wsl-debug`
- `cmake --build build/wsl-debug -j`
- `./scripts/remote exec -- "ctest --test-dir build/v100-petsc --output-on-failure -R 'operator_tests|smoke_test_petsc|validate_slepc_eigensolver'"`
- `./scripts/remote exec -- "cd ~/MacroFlow3D && ./build/v100-petsc/analyze_invariant_quality"`
- `./scripts/remote exec -- "cd ~/MacroFlow3D && sed -n '1,40p' artifacts/gate3/invariant_transport_consumed.csv && tail -n 20 artifacts/gate3/invariant_transport_consumed.csv"`

## Validation status of this mapping iteration

- `operator_tests` aborted after 4.38 s with `CUDA driver version is insufficient for CUDA runtime version`.
- `macroflow3d_pipeline apps/config_pspta_small.yaml` aborted immediately with the same CUDA driver/runtime mismatch.
- Remote `v100-release` build succeeded on the Tesla V100.
- Remote `operator_tests` passed in `3.26 s`.
- Remote `apps/config_pspta_small.yaml` succeeded on the runtime-authoritative legacy PSPTA path and produced `psi_refine_summary.csv`, `psi_refine_history.csv`, and particle snapshots under `output_pspta_small/`.
- Remote PETSc/SLEPc Phase 1 is now working through the repo workflow:
  - clean external build succeeded and produced `libpetsc.a` (`372M`) and `libslepc.a` (`52M`)
  - `cmake --preset v100-petsc` succeeded
  - `cmake --build build/v100-petsc -j` succeeded after importing PETSc's static link surface from `petscvariables`
  - `ctest --test-dir build/v100-petsc --output-on-failure -R smoke_test_petsc` passed in `2.52 s`
  - `ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver` passed in `2.91 s`
  - direct `validate_slepc_eigensolver` on `uniform_x` reported `||A(1)||/||1|| = 0`, converged the requested modes in `1` iteration (`379.3 ms`), with `λ[0]=λ[1]=0`, residuals `0`, and modal overlap `3.47e-18`
- The exact-object Strategy A harness now exists in `analyze_invariant_quality`, writes `artifacts/gate3/invariant_transport_consumed.csv`, and now exits cleanly on the Tesla V100 through `scripts/remote run ...` (`exit_code=0`).
- `bind_invariants()` is now exercised on the Tesla V100 with real Strategy A fields for `uniform_x`, `layered_x`, and `darcy_small`.
- The exact-object harness now searches all 2D mode planes inside the requested six-mode Strategy A subspace, transports each plane's best rotated pair, and records `mode_i/mode_j` on the consumed object.
- Control-case evidence from that exact-object harness:
  - raw Strategy A on `uniform_x` and `layered_x` preserves the bound invariants to `~1e-8` RMS drift over `8` steps with zero Newton failures, so the engine is genuinely consuming Strategy A fields rather than silently falling back to legacy marching.
  - raw Strategy A still has `rel_rms_mismatch ~ 1.0` even on the smooth controls, so transport preservation alone does not make the fields scientifically acceptable.
  - best in-subspace rotation improves independence on the exact consumed object, but does not materially change `rel_rms_mismatch`; on `darcy_small` it only changes failure counts by single digits.
  - full pair search inside the requested six-mode Strategy A subspace changes which pair is transported and materially lowers `darcy_small` failure counts, but still leaves `rel_rms_mismatch ~ 1.07` and drift maxima `~8e-6` on the winning pairs. On `uniform_x` and `layered_x`, the preferred pair moves away from `(0,1)` while mismatch stays `~1.000` / `~1.021` with zero failures.
  - period-fit affine normalization is a bad direction: it inflates mismatch and failure counts sharply on `darcy_small`.
  - amplitude-only crossfit scaling can force the fields toward constants and drive Newton failures to zero while leaving `rel_rms_mismatch` at `O(1)`. That is a degenerate numerical escape, not a valid fix.
  - raw Strategy A on `darcy_small` is still not acceptable: modal `gauge_ready=NO`, `quality_rms_r ~ 1e-2 to 1e-1`, `rel_rms_mismatch ~ 1.07`, `final drift ~ 1e-6`, and hundreds of particles with nonzero Newton failures.
  - the legacy-style inlet-plane overwrite also makes Strategy A worse on every tested control (`quality_rms_r ~ 2.3-2.7`, `rel_rms_mismatch ~ 3.2-3.4`).

Scientific conclusions in this note now depend on code-path inspection plus remote `v100-release` and `v100-petsc` evidence. They still do not prove transport correctness for the new Strategy A/C path.

## Current runtime-authoritative PSPTA path

### 1. Config selection

`transport.method=pspta` is the only selector for the PSPTA route in the runtime config layer. There is no config key for invariant source, eigensolver backend, gauge mode, or Strategy A/C selection.

Evidence:

- `src/io/config/ConfigValidator.hpp:122-149`
- `src/io/config/ConfigYaml.cpp:252-290`
- `apps/config_pspta_small.yaml:79-105`

### 2. Runtime transport wiring

When `transport.method == "pspta"`, `EnsembleRunner`:

1. allocates a compact velocity field,
2. constructs `PsptaPsiField`,
3. calls `PsptaPsiField::precompute_levelA(...)`,
4. optionally calls `PsptaPsiField::refine_psi(...)`,
5. optionally computes `PsptaPsiField::compute_psi_quality(...)`,
6. binds the deprecated `PsptaEngine::bind_psifield(...)`,
7. runs transport.

Evidence:

- `src/runtime/ensemble/EnsembleRunner.cu:438-485`

This means the runtime-authoritative mathematical object is still:

- legacy semi-Lagrangian x-marching for the seed invariants, plus
- legacy defect-correction refinement on the same field representation.

### 3. Runtime transport consumer assumptions

`PsptaEngine` already exposes the newer `bind_invariants(...)` interface, but the documented active assumptions inside the transport consumer still match the legacy inlet gauge:

- `psi1 ~ y` with self-period `Ly`
- `psi2 ~ z` with self-period `Lz`

The header explicitly labels Strategy A/C integration as future work and names inlet-gauge normalization as the current workaround.

Evidence:

- `src/physics/particles/pspta/PsptaEngine.hpp:117-157`

### 4. Runtime diagnostics currently written

The runtime CSV path only records:

- `rms/max(v·∇ψ1)`, `rms/max(v·∇ψ2)`,
- `vx_clamped` fraction from the x-marching seed,
- Newton failure counts/histogram,
- legacy refinement history.

It does not write:

- cross-product mismatch,
- independence / degeneracy,
- masked-fraction / low-velocity localization,
- modal quality at ingestion,
- construction method provenance.

Evidence:

- `src/runtime/io/CsvDiagnosticsWriter.hpp:57-180`
- remote `output_pspta_small/psi_refine_summary.csv`
- remote `output_pspta_small/psi_refine_history.csv`

## Planned authoritative PSPTA path already present in the repo

The repository already contains the intended Strategy A/C building blocks:

- `TransportOperator3D` and `CombinedOperatorA` for `A = D^T W D + mu L`
- validation and production SLEPc backends
- `PsptaInvariantField` as the new invariant container
- `PsptaInvariantField::ingest_eigenvectors(...)`
- `GaugeFixer`
- `RefinementAC`
- control-case apps for `uniform_x`, `layered_x`, and `darcy_small`

Evidence:

- `src/physics/particles/pspta/invariants/SLEPcBackend.cuh:5-28`
- `src/physics/particles/pspta/invariants/PsptaInvariantField.cu:407-470`
- `apps/validate_slepc_eigensolver.cu:81-176`
- `apps/analyze_invariant_quality.cu:335-430`

## Where the current implementation diverges from the active plan

### Divergence A: runtime transport does not consume Strategy A invariants

`PsptaInvariantField` and `bind_invariants(...)` exist, but the production runtime does not use them. The authoritative run path still binds `PsptaPsiField`.

Evidence:

- `src/runtime/ensemble/EnsembleRunner.cu:483-485`
- `src/physics/particles/pspta/PsptaEngine.hpp:147-157`
- `artifacts/gate3/invariant_transport_consumed.csv` now proves that `bind_invariants(...)` itself is a real transport path, but only inside the analysis harness, not inside production runtime.

### Divergence B: production runtime still does not emit the quality object now available in the exact-object harness

`PsptaInvariantField::compute_quality(...)` now measures invariance residuals, cross-product mismatch, and independence for the exact consumed object, but the production runtime still does not write these metrics when `transport.method=pspta` routes through legacy marching.

Evidence:

- `src/physics/particles/pspta/invariants/PsptaInvariantField.cu:346-470`
- `artifacts/gate3/invariant_transport_consumed.csv`

### Divergence C: Strategy C refinement for the new path is still a stub

`RefinementAC` returns `stop_reason = "not_implemented"` and performs no refinement.

Evidence:

- `src/physics/particles/pspta/invariants/RefinementAC.cu:34-91`

### Divergence D: gauge handling for the new path is incomplete

`GaugeFixer` only implements inlet-plane overwrite. `MeanZero` and `ScaledPeriodic` remain TODOs, and the engine still assumes legacy self-periods.

Evidence:

- `src/physics/particles/pspta/invariants/GaugeFixer.cu:40-68`
- `src/physics/particles/pspta/PsptaEngine.hpp:122-142`

### Divergence E: operator tests do not validate the same transported object

`operator_tests` include strong algebra checks for `D`, `L`, and `A`, but the `PsptaInvariantField` and `bind_invariants()` coverage is only smoke-level. The test explicitly accepts garbage invariant values as long as the computation does not crash.

Evidence:

- `src/physics/particles/pspta/invariants/OperatorTestHarness.cu:126-201`

### Divergence F: config semantics silently mix transport backend selection with invariant-construction method

`transport.method=pspta` currently means:

- use the PSPTA engine, and
- use legacy x-marching to produce the invariants consumed by that engine.

The active plan wants those concerns separated. The current config surface does not make that distinction visible.

Evidence:

- `src/io/config/ConfigValidator.hpp:122-149`
- `src/runtime/ensemble/EnsembleRunner.cu:438-485`

## Reuse vs replace

### Reuse unchanged or nearly unchanged

- `PsptaEngine` hot loop, particle SoA, wrap counters, and failure accounting.
- `bind_invariants(...)` as the transport-side entry point for the new path.
- `TransportOperator3D`, `CombinedOperatorA`, and the SLEPc backends.
- `EnsembleRunner` realization loop, allocation strategy, scheduler, profiling, and output layout.
- Existing control-case app structure in `analyze_invariant_quality` and `validate_slepc_eigensolver`.
- CSV writer pattern and manifest/config plumbing.

### Reuse only as infrastructure / baseline context

- `PsptaPsiField::precompute_levelA(...)` and `PsptaPsiField::refine_psi(...)`.
- `LegacyMarchingInvariantBuilder`.

These are useful as:

- a baseline generator,
- a gauge/periodicity reference,
- a failure-mode baseline,
- a transport-side integration template.

They are not acceptable as the final authoritative scientific method for the Lester regime.

Evidence:

- `src/physics/particles/pspta/PsptaPsiField.cuh:19-27`
- `src/physics/particles/pspta/legacy/LegacyMarchingInvariantBuilder.cu:18-65`

### Replace as authoritative math

- x-marching seed construction,
- x-marching defect-correction refinement,
- any transport path that depends on hidden legacy inlet-gauge assumptions,
- any acceptance decision based only on `v·∇ψ` residuals plus Newton fails.

## Validation/gating map today

### What is already meaningful

- Gate 2 algebra for `D`, `L`, `A`
- SLEPc validation on uniform flow
- control-case invariant analysis on `uniform_x`, `layered_x`, `darcy_small`

### What is missing for scientific trust

- end-to-end transport of Strategy A invariants through `PsptaEngine::bind_invariants(...)`
- runtime metrics for mismatch / independence / masked regions
- metadata-driven gauge semantics in transport
- convergence/sensitivity studies on the same object used in transport
- a production selector that makes legacy vs Strategy A/C explicit

## Remote evidence from the current authoritative path

- `config_pspta_small` on the Tesla V100 still enters the legacy `psi_refine` loop. Final refinement metrics were `seed_rms_r1=0.734999`, `seed_rms_r2=0.944765`, `final_rms_r1=0.734499`, `final_rms_r2=0.939298`, with `iters_done=4`, `converged=0`, and `stop_reason=max_iters`.
- The runtime still completed transport with `[pspta] active=387 exited=113 newton_stalls=0 nonzero_fail=0 max_fail=0`, which confirms runtime completion is not an admissible scientific acceptance criterion by itself.
- The smoke artifacts provide no invariant mismatch or independence evidence. Only legacy residuals and particle snapshots are available.
- A coarse particle-cloud check from `output_pspta_small/snapshots` showed `var_y: 2144.69 -> 1959.12` and `var_z: 2040.99 -> 1767.80` between `step_00000005` and `step_00000500`, but this is not a transverse-macrodispersion validation because the initial particle cloud already spans the inlet cross-section.

## Evidence-ordered execution plan

### Step 1. Establish one transported mathematical object for control cases

Build a dedicated control harness that:

1. constructs `PsptaInvariantField` from Strategy A,
2. applies the chosen gauge explicitly,
3. feeds it into `PsptaEngine::bind_invariants(...)`,
4. runs transport on `uniform_x`, `layered_x`, and `darcy_small`,
5. records invariant-preservation and Newton behavior from the same field actually consumed by transport.

Why first:

- It closes the biggest current gap: validation apps and runtime are not testing the same object.
- It avoids changing ensemble production semantics before the consumer contract is proven.

### Step 2. Complete quality metrics on `PsptaInvariantField`

Implement and expose, at minimum:

- invariance residuals,
- cross-product mismatch,
- independence / degeneracy,
- masked fraction / low-velocity localization.

Use `PsptaInvariantField` as the common reporting object for control harnesses and runtime.

Why second:

- Without this, Gate 3/Gate 4 decisions remain under-instrumented even if Strategy A reaches transport.

### Step 3. Make gauge semantics transport-safe

Remove implicit legacy gauge assumptions from the transport consumer by passing or deriving:

- self-periods,
- gauge method,
- construction provenance,
- invalid/masked-cell handling

from `PsptaInvariantField` metadata rather than from `Ly/Lz` and inlet-plane convention alone.

Why before refinement:

- Gauge ambiguity can invalidate every downstream transport result, including otherwise good eigensolver output.

### Step 4. Implement Strategy C on the new field, not on `PsptaPsiField`

Implement alternating fit + Poisson projection in `RefinementAC`, with monotone quality checks and explicit gauge reapplication.

Why after Steps 1-3:

- Refinement should improve the same object the transport engine consumes.
- Implementing refinement first would polish the wrong interface boundary.

### Step 5. Add an explicit invariant-source selector to runtime

Separate:

- `transport.method` = transport engine selection
- `transport.invariant_source` = `legacy_marching`, `strategy_a`, `strategy_ac`

Keep `legacy_marching` available only as an explicit, non-authoritative baseline path.

Why here:

- After the new path works on control harnesses, runtime can adopt it without silent method mixing.

### Step 6. Re-run Tier B/Tier C on remote PETSc/SLEPc infrastructure

Use the documented V100 workflow to run:

- `validate_slepc_eigensolver`
- control-case harnesses
- PSPTA smoke with explicit Strategy A source

Why remote before production Monte Carlo:

- The authoritative eigensolver backend is PETSc/SLEPc-backed, and the local environment currently cannot validate CUDA execution.

### Step 7. Only then move to larger smooth isotropic random-K studies

Progression:

1. `uniform_x`
2. `layered_x`
3. `darcy_small`
4. grid/time-step sensitivity on small smooth Darcy
5. larger smooth isotropic random-K runs
6. only then any macrodispersion interpretation

## Current blocker

The codebase has two incompatible PSPTA stories:

- the runtime-authoritative legacy x-marching path,
- the planned Strategy A/C path in sidecar apps and containers.

They are not yet the same transported object, and the new path’s quality/gauge/refinement stack is incomplete.

## Verdict

The new PSPTA route is not proven yet.

The authoritative runtime today is still the legacy marching prototype. The repository already contains enough infrastructure to build the planned Strategy A/C path without a broad rewrite, but the next work must unify validation and transport around `PsptaInvariantField` before any scientific macrodispersion claim is trusted.
