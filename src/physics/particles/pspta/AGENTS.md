# PSPTA AGENTS.md

## Scope

This file applies to `src/physics/particles/pspta/`.

This is the highest-risk scientific area of the repository.

Do not treat this directory as “just another transport backend.”
It is the research path for preserving the kinematic constraints associated with the helicity-free, smooth, locally isotropic Darcy regime.

---

## Primary goal

Preserve the invariant structure well enough that purely numerical leakage is not misread as physical transverse macrodispersion.

In practice, this means protecting:

- near-invariance of `ψ1`, `ψ2`
- usable independence of `ψ1`, `ψ2`
- robust advance+project transport
- interpretable diagnostics and failure accounting

---

## Mental model

Think in three layers:

1. **construct invariants**
2. **measure invariant quality**
3. **transport particles while preserving those invariants**

A change that improves one layer by damaging another is not automatically an improvement.

---

## Files you are likely to touch

Typical PSPTA work touches some subset of:

- `PsptaPsiField.*`
- `PsptaEngine.*`
- `invariants/TransportOperator3D.*`
- `invariants/PsptaInvariantField.*`
- `invariants/EigensolverBackend.*`
- `invariants/SLEPcBackend.*`
- `invariants/RefinementAC.*`
- `invariants/GaugeFixer.*`
- `invariants/OperatorTestHarness.*`

Before editing, identify exactly which layer you are changing.

---

## Hard rules

- Plan first.
- Do not edit PSPTA code without a validation path.
- Do not claim physical improvement from a numerical change without control runs.
- Do not silently weaken diagnostics.
- Do not move allocations into the hot loop.
- Do not introduce hidden synchronizations in `step()` or equivalent hot paths.
- Do not change invariant semantics, projection semantics, or particle status semantics without documenting it.

---

## Performance rules

The hot loop must remain:
- allocation-free,
- explicit about host/device synchronization,
- compatible with large particle counts.

Any new buffer should be:
- owned clearly,
- allocated in setup / prepare,
- reused.

---

## Scientific checks

For any non-trivial PSPTA change, inspect at least:

- `v·∇ψ1`
- `v·∇ψ2`
- invariant independence / degeneracy
- Newton failure counts
- particle exit / inactive behavior
- smoke trajectory behavior on the small config

If available, also inspect:
- mismatch versus `∇ψ1 × ∇ψ2`
- refinement history
- grid sensitivity

---

## Special caution areas

### 1. Near-stagnation regions
These can make invariants and Newton projection ill-conditioned.

### 2. Functional dependence collapse
Two low modes are not useful if they become nearly dependent.

### 3. Projection masking
A projection that “works” numerically may still hide geometric leakage or unstable convergence.

### 4. Diagnostics drift
If metrics get renamed, filtered, or weakened, make the change explicit.

---

## Required validation for PSPTA changes

Minimum:
```bash
ctest --test-dir <build-dir> --output-on-failure -R operator_tests
./<build-dir>/macroflow3d_pipeline apps/config_pspta_small.yaml
```

If the eigensolver path is touched:
```bash
ctest --test-dir <build-dir> --output-on-failure -R validate_slepc_eigensolver
```

If behavior changes materially, use the acceptance gates in:
- `docs/validation/acceptance-gates.md`

---

## Done criteria

A PSPTA task is done only when:
1. the intended subsystem change is complete,
2. relevant operator / smoke validation was run,
3. diagnostics were inspected,
4. no obvious scientific regression is left unexplained,
5. the final report states remaining uncertainty explicitly.
