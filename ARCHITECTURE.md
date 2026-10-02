# ARCHITECTURE.md

## 1. Purpose

MacroFlow3D is a GPU-first research codebase for macrodispersion in 3D heterogeneous porous media. The current architecture supports:

1. stochastic generation of conductivity fields,
2. stationary Darcy flow solve,
3. face-centered velocity reconstruction,
4. particle transport through:
   - the **Par2 baseline path**, or
   - a future invariant-preserving consumer of accepted streamfunctions,
5. ongoing integration of a Lester equation (14) streamfunction solver for invariant construction,
6. ensemble statistics and macrodispersion analysis,
7. reproducible output capture.

The codebase should be read as a **scientific pipeline with numerical contracts**, not as a generic app.

---

## 2. High-level pipeline

The canonical flow is:

1. **Generate `K(x)`**
2. **Solve head `h(x)`** from the variable-coefficient Darcy problem
3. **Compute Darcy velocity `v(x)`**
4. **Run transport**
   - baseline RWPT through Par2, or
   - future invariant-preserving consumer of accepted streamfunctions
5. **Collect moments and ensemble statistics**
6. **Compute macrodispersion outputs**
7. **Write manifests, configs, CSVs, snapshots**

This is consistent with the existing entry-point and runner split:
- `apps/macroflow3d_pipeline.cu`
- `src/runtime/pipeline/PipelineRunner.*`
- `src/runtime/ensemble/EnsembleRunner.*`
- `src/runtime/analysis/AnalysisRunner.*`

---

## 3. Main architectural layers

### 3.1 Core
`src/core/`

Low-level value types and containers:
- `Grid3D`
- scalar type (`real = double`)
- device buffers / spans
- boundary spec helpers

Core code should stay boring, stable, and dependency-light.

### 3.2 Numerics
`src/numerics/`
`src/multigrid/`

Numerical kernels and operators:
- BLAS-like primitives
- variable-coefficient elliptic operators
- solver/preconditioner stack
- multigrid transfers, smoothers, V-cycle

This layer exists to provide reusable numerical mechanisms, not project policy.

Key contract:
- numerical kernels must be allocation-free in hot paths,
- host sync must be explicit and easy to audit.

### 3.3 Physics
`src/physics/`

This is the science layer.

#### Flow
`src/physics/flow/`
- head solve
- velocity-from-head
- velocity diagnostics

#### Stochastic conductivity
`src/physics/stochastic/`
- random-field generation and related workspace

#### Transport
`src/physics/particles/`
- `par2_adapter/`: baseline RWPT path
- `pspta/`: legacy invariant/transport infrastructure to audit, migrate, or retire

### 3.4 Runtime / orchestration
`src/runtime/`

This layer owns:
- CUDA context lifecycle,
- stage orchestration,
- profiling hooks,
- I/O scheduling,
- ensemble loops,
- analysis-only dispatch.

Important rule:
**orchestration belongs here, not inside physics kernels.**

### 3.5 I/O and reproducibility
`src/io/`

Owns:
- YAML config loading
- validation
- output layout
- manifest writing
- effective config serialization
- CSV writers

Runs are expected to be reproducible from:
- the committed code,
- the input config,
- the generated effective config,
- the manifest metadata.

---

## 4. Transport paths

### 4.1 Baseline path: Par2
The Par2 path is the current operational baseline.

Use it when:
- validating end-to-end pipeline behavior,
- checking non-PSPTA regressions,
- comparing statistics against legacy or known-good transport behavior.

Do not casually change this path.

### 4.2 Legacy PSPTA path
The existing PSPTA path is legacy research infrastructure.

It currently includes:
- invariant field containers,
- legacy x-marching invariant construction,
- transport-near-nullspace operator machinery,
- optional SLEPc eigensolver backend,
- refinement skeletons,
- pseudo-symplectic particle updates,
- diagnostics and failure accounting.

Do not extend the old PSPTA invariant-construction architecture. New invariant construction belongs to the Lester equation (14) streamfunction solver. Existing PSPTA code may still be useful as a compatibility surface, diagnostic source, or transport consumer while the new path is brought up.

### 4.3 Lester equation (14) streamfunction solver
The new invariant-construction direction is a solver for the coupled nonlinear Lester et al. equation (14) system.

The target is to compute `psi1`, `psi2` such that:

```math
v = grad(psi1) x grad(psi2),
v.grad(psi1) = 0,
v.grad(psi2) = 0.
```

The preferred linear subproblem uses:

```math
q = 1/k,
A psi = -div(q grad psi).
```

Then a decoupled nonlinear iteration solves:

```math
A psi1 = -q S1,
A psi2 = -q S2.
```

(same-index pairing; the paper's printed equation (14) crosses the indices —
see `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md`)

This formulation keeps a variable-coefficient diffusion structure and avoids explicitly differencing `grad(log k)`. It is the contract of the **periodic-fluctuation stack**, which is frozen (status 2026-10-02):

- the periodic-fluctuation stack (SF-02..SF-26, `src/physics/streamfunctions/`: operators, projected PCG/MG, Picard, Anderson, Newton-Krylov, continuation, diagnostics) is frozen verified infrastructure and the producer of invariants on symmetric controls (the Lester 2021 field);
- its periodic solution on generic Gaussian fields is not the Darcy flow: streamlines of a generic smooth periodic `k` do not close on the torus, so no nondegenerate affine + periodic pair represents Darcy there, and the periodic solution of (14) is the closed-streamline field nearest to Darcy (`docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`; CPU probes, amplitude <= 1, `L/ell = 4`; production-stack repeat pending in SF-30);
- the new target is the Darcy labels non-periodic in `x1` with inlet anchoring, prototyped on CPU (SF-29) before any GPU generalization (`docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`);
- planned consumer modules: `src/numerics/interpolation/` (periodic tricubic B-spline, SF-28) and `src/physics/particles/streamline_tracker/` (SF-31/32);
- the closure gate (SF-30) is the label-independent oracle;
- the legacy PSPTA engine is a possible invariant-preserving transport consumer once labels of the Darcy flow exist, but its role must be re-evaluated.

---

## 5. Invariant-consumer contract

### 5.1 Conceptual decomposition

The new architecture separates invariant construction from invariant consumption:

1. **Construct invariants**
   - solve the Lester equation (14) streamfunction system;
   - keep construction residuals, gauges, denominator regularization, and continuation state visible.

2. **Measure invariant quality**
   - residual of `v·∇ψi`
   - reconstruction mismatch
   - gradient degeneracy / collinearity
   - stagnation sensitivity

3. **Consume accepted invariants**
   - feed only accepted `psi1`, `psi2` fields into transport;
   - preserve invariant labels during particle motion;
   - track projection / transport failures separately from construction failures.

### 5.2 Why this matters

For the smooth, locally isotropic Darcy regime, the theory motivating this project says:
- the flow is helicity-free,
- two invariant streamfunctions exist locally; a global affine + periodic pair exists only for flows with closed streamlines, which a generic periodic `k` does not give (experiment note of 2026-10-02),
- streamlines remain confined to 2D streamsurfaces of the labels of the flow actually tracked (for the periodic surrogate, by construction; for Darcy, labels anchored at the inlet),
- conventional interpolation and tracking can violate those constraints and create spurious transverse dispersion.

The streamline integrator of the closure gate is an oracle independent of the labels.

So the software must not only “move particles”; it must defend those kinematic constraints. The old PSPTA engine may be adapted or replaced, but it should not dictate the new invariant-construction architecture.

---

## 6. Current scientific fault lines

These are the main places where the code can be numerically correct in a narrow sense while still being scientifically misleading.

### 6.1 Velocity field structure
A divergence-free interpolation or discretization is not automatically structure-preserving for the invariant geometry.

### 6.2 Invariant construction quality
Two fields that are algebraically small modes are not automatically useful invariants unless:
- they are actually near-invariant,
- they remain sufficiently independent,
- they are stable under refinement.

### 6.3 Tracking errors
A particle tracker can create apparent transverse spreading even when the underlying regime should forbid it asymptotically.

### 6.4 Upscaling / discretization effects
Artifacts introduced by coarse discretization, interpolation, or block-scale reformulation can masquerade as physical transverse macrodispersion.

### 6.5 Existence of the target object
A solver can converge to a small residual on an object that is not the one wanted. The periodic equation (14) has a solution on generic Gaussian fields, but it is the closed-streamline field nearest to Darcy, not the Darcy flow, because the Darcy streamlines of such fields do not close. The existence of any new target object must be checked independently of the solver (positive controls, streamline-closure oracle) before solver acceptance, and acceptance rests on a physics metric under refinement, never on a residual alone.

---

## 7. Design principles

### 7.1 The repo is the system of record
Durable context must live in versioned files:
- `AGENTS.md`
- runbooks
- validation gates
- decisions
- experiment notes

### 7.2 One layer, one responsibility
- physics decides the mathematical object,
- numerics implements the operator/mechanism,
- runtime orchestrates,
- I/O records.

### 7.3 Validation beats intuition
For the scientific core, evidence beats plausibility.

### 7.4 Baselines are preserved
Par2 is a baseline, not dead weight.
Legacy PSPTA is migration context, not a free rewrite license and not the new invariant-construction architecture.

---

## 8. What to read before editing

### If you touch new invariant construction
Read:
- `AGENTS.md`
- `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`
- `docs/theory/lester-2023-key-claims.md`
- `docs/validation/acceptance-gates.md`

### If you touch legacy PSPTA transport or macrodispersion
Read:
- `AGENTS.md`
- `docs/validation/acceptance-gates.md`
- `src/physics/particles/pspta/AGENTS.md`

### If you touch solvers or operators
Read:
- `AGENTS.md`
- `src/numerics/AGENTS.md`

### If you touch docs or workflows
Read:
- `docs/AGENTS.md`

---

## 9. Minimal mental model for new contributors

- The **baseline pipeline already works**.
- The **scientific problem is not “make legacy PSPTA compile”**.
- The scientific problem is:
  - preserve the correct kinematics,
  - separate physical transverse spreading from numerical leakage,
  - make claims that survive refinement and controlled comparisons.

If you keep that mental model, most architecture decisions become obvious.
