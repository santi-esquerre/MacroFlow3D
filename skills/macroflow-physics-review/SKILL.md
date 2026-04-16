# macroflow-physics-review

Review workflow for changes touching scientific or numerical behavior.

## When to use

Use this skill when reviewing or authoring changes that touch:
- `src/numerics/` — operators, solvers, BLAS, multigrid
- `src/physics/` — flow, stochastic, transport (Par2 or PSPTA)
- `src/multigrid/` — transfers, smoothers, V-cycle
- `apps/` configs that affect physics behavior
- any code path that changes numerical output

## Step 1 — Read relevant theory notes and plans

Before reviewing scientific changes, read the applicable documents:

| Changed area | Required reading |
|-------------|------------------|
| PSPTA, invariants, eigensolver, refinement, tracking, velocity reconstruction, transverse macrodispersion | `docs/plans/active/pspta-execution-plan.md` and `docs/theory/lester-2023-key-claims.md` |
| Large-domain macrodispersion, Monte Carlo design, historical baseline comparisons | `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` |

For PSPTA-related changes, verify that the change aligns with the current phase of the execution plan before accepting.

Skip if the change is Gate 0 or Gate 1 only.

## Step 2 — Classify the change

Map changed files to the required acceptance gate:

| Changed area | Minimum gate |
|--------------|-------------|
| docs / scripts / AGENTS only | Gate 0 |
| refactor, no numerical change | Gate 1 |
| operators / eigensolver / algebra | Gate 2 |
| PSPTA / invariants / tracking | Gate 3 |
| helicity-free regime correctness | Gate 4 |
| ensemble / macrodispersion output | Gate 5 |

Reference: `docs/validation/acceptance-gates.md`

## Step 3 — Check invariant risk surfaces

For any change in the scientific core, apply the Lester constraints (`docs/theory/lester-2023-key-claims.md`):

1. **Does this change the velocity field structure?**
   - Divergence-free is necessary but not sufficient for invariant geometry.

2. **Does this change invariant construction quality?**
   - Near-invariance, independence, refinement stability.

3. **Does this change tracking behavior?**
   - Newton convergence, projection accuracy, particle exit conditions.

4. **Does this change macrodispersion interpretation?**
   - Moment computation, ensemble statistics, `α_L` / `α_T` output.

5. **Could this create spurious transverse macrodispersion?**
   - In the smooth, locally isotropic, purely advective regime, positive transverse macrodispersion is NOT automatically physical.

## Step 4 — Require evidence, not intuition

For each affected gate, verify the author provides:

### Gate 1 (build + smoke)
- [ ] Configure succeeds
- [ ] Build succeeds
- [ ] Tests pass
- [ ] Smoke run completes: `./build/*/macroflow3d_pipeline apps/config_pspta_small.yaml`

### Gate 2 (operator integrity)
- [ ] `ctest -R operator_tests` passes
- [ ] `ctest -R validate_slepc_eigensolver` passes (if eigensolver touched)
- [ ] Residual norms reported
- [ ] Before/after comparison if behavior changed

### Gate 3 (PSPTA local integrity)
- [ ] `v·∇ψ1` and `v·∇ψ2` residuals inspected
- [ ] Independence / degeneracy signal checked
- [ ] Newton failure counts reported
- [ ] Particle status summary (active / exited / failed)
- [ ] No unexplained metric regression

### Gate 4 (helicity-free regime)
- [ ] Controlled pure-advection test case
- [ ] Transverse spreading explicitly characterized as physical / numerical / unresolved
- [ ] Invariant confinement behavior checked

### Gate 5 (macrodispersion)
- [ ] Baseline comparison with prior known-good run
- [ ] `α_L(t)`, `α_T1(t)`, `α_T2(t)` before/after
- [ ] Config and build artifacts recorded
- [ ] Run is reproducible

## Step 5 — Block or approve

**Block if:**
- required gate evidence is missing
- residuals materially worsened without explanation
- Newton failures exploded
- positive transverse macrodispersion is claimed as physical without control
- diagnostics were weakened silently

**Approve only when:**
- evidence matches the gate requirements
- remaining uncertainty is stated explicitly
- no silent behavior changes

## Special caution areas

- Near-stagnation regions (invariants and Newton become ill-conditioned)
- Functional dependence collapse (two modes becoming nearly dependent)
- Projection masking (numerically stable but geometrically leaky)
- Diagnostics drift (renamed, filtered, or weakened metrics)

## Report template

Use the template from `docs/validation/acceptance-gates.md`:

```md
## Scientific change report

### Scope
- files changed:
- intended effect:

### Commands run
- configure:
- build:
- tests:
- smoke:

### Evidence
- gate(s) satisfied:
- metrics inspected:
- before/after comparison:

### Remaining risk
```

## Related

- `docs/validation/acceptance-gates.md`
- `docs/theory/lester-2023-key-claims.md`
- `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`
- `src/physics/particles/pspta/AGENTS.md`
- `src/numerics/AGENTS.md`
- `docs/runbooks/autonomy-policy.md`
