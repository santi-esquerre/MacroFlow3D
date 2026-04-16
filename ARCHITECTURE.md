# ARCHITECTURE.md

## 1. Purpose

MacroFlow3D is a GPU-first research codebase for macrodispersion in 3D heterogeneous porous media. The current architecture supports:

1. stochastic generation of conductivity fields,
2. stationary Darcy flow solve,
3. face-centered velocity reconstruction,
4. particle transport through either:
   - the **Par2 baseline path**, or
   - the **PSPTA / invariant-based path**,
5. ensemble statistics and macrodispersion analysis,
6. reproducible output capture.

The codebase should be read as a **scientific pipeline with numerical contracts**, not as a generic app.

---

## 2. High-level pipeline

The canonical flow is:

1. **Generate `K(x)`**
2. **Solve head `h(x)`** from the variable-coefficient Darcy problem
3. **Compute Darcy velocity `v(x)`**
4. **Run transport**
   - baseline RWPT through Par2, or
   - PSPTA / invariant-preserving path
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
- `pspta/`: invariant-based / pseudo-symplectic path

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

### 4.2 PSPTA path
The PSPTA path is the active research path.

Its job is to preserve the kinematic constraints of helicity-free, smooth, locally isotropic Darcy flow well enough to avoid spurious transverse macrodispersion in the purely advective regime.

It currently includes:
- invariant field infrastructure,
- transport operator machinery,
- optional eigensolver backend,
- refinement logic,
- pseudo-symplectic particle updates,
- diagnostics and failure accounting.

The PSPTA path is promising, but must be treated as a **validated research path**, not an assumed truth.

---

## 5. PSPTA architecture

### 5.1 Conceptual decomposition

PSPTA should be thought of as three coupled pieces:

1. **Invariant construction**
   - obtain two scalar labels `ψ1`, `ψ2`
   - target: near-invariance under `v·∇ψ = 0`
   - target: usable independence / non-degeneracy

2. **Invariant quality measurement**
   - residual of `v·∇ψi`
   - reconstruction mismatch
   - gradient degeneracy / collinearity
   - stagnation sensitivity

3. **Transport using invariants**
   - advance in `x`
   - project `(y, z)` back to the invariant manifold
   - track Newton failures and exit behavior

### 5.2 Why this matters

For the smooth, locally isotropic Darcy regime, the theory motivating this project says:
- the flow is helicity-free,
- two invariant streamfunctions exist,
- streamlines remain confined to 2D streamsurfaces,
- conventional interpolation and tracking can violate those constraints and create spurious transverse dispersion.

So the software must not only “move particles”; it must defend those kinematic constraints.

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
PSPTA is a target path, not a free rewrite license.

---

## 8. What to read before editing

### If you touch transport or macrodispersion
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
- The **scientific problem is not “make PSPTA compile”**.
- The scientific problem is:
  - preserve the correct kinematics,
  - separate physical transverse spreading from numerical leakage,
  - make claims that survive refinement and controlled comparisons.

If you keep that mental model, most architecture decisions become obvious.
