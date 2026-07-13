@AGENTS.md
@ARCHITECTURE.md

## Claude Code integration

This repository has a comprehensive agentic harness shared across tools (Claude Code, Codex, VS Code Copilot). Claude Code inherits it via the `@` includes above. The `AGENTS.md` files and `docs/` hierarchy are the authoritative source of truth — not this file.

### Quick orientation

**Mission:** 3D macrodispersion in heterogeneous porous media. Current focus: Lester equation (14) streamfunction solver for `psi1`, `psi2` in smooth, locally isotropic Darcy flow; existing PSPTA code is legacy compatibility/migration surface.

**Optimization priority:** correctness → reproducibility → maintainability → performance → speed.

### Routing table — read before working

| Work area | Read first |
|-----------|-----------|
| Lester equation (14) / new invariant construction | `docs/plans/active/lester-eq14-streamfunction-solver-plan.md` + `docs/theory/lester-2023-key-claims.md` |
| Legacy PSPTA audit / migration / removal | `docs/plans/archive/pspta-execution-plan.md` + `docs/theory/lester-2023-key-claims.md` |
| Macrodispersion / ensemble statistics | `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` |
| Numerics / operators / solvers | `src/numerics/AGENTS.md` |
| PSPTA transport code | `src/physics/particles/pspta/AGENTS.md` |
| Documentation changes | `docs/AGENTS.md` |
| Validation / what tier to run | `docs/validation/acceptance-gates.md` + `docs/validation/eval-tiers.md` |
| PR / branch workflow | `docs/runbooks/pr-workflow.md` + `docs/runbooks/autonomy-policy.md` |
| Remote V100 work | `docs/runbooks/remote-v100.md` |

### Local / remote split

- **Edit locally** in WSL. Local is the source of truth for code.
- **Validate locally** with `cmake --preset wsl-debug`, `ctest`, smoke runs.
- **Sync remotely** for release builds, PETSc/SLEPc, profiling, ensemble runs.
- **Never edit on the remote server.**

### Claude-native helpers

**Skills** (activated contextually):

- `macroflow-build` — local configure / build / test
- `macroflow-evals` — validation tier classification and execution
- `macroflow-physics-review` — scientific change review workflow
- `macroflow-pr-review` — branch / worktree / PR workflow
- `macroflow-remote-v100` — remote V100 build / test / run

**Agents** (subagents for delegation):

- `macroflow` — default development (full tool access)
- `research` — investigation with web access
- `readonly` — safe read-only exploration
- `review` — scientific code review (read-only + bash)

**Commands** (explicit invocation):

- `/project:build` — run local build cycle
- `/project:validate` — classify change and run correct validation tier
- `/project:sync-v100` — sync to V100 and remote build

### Hard rules (inherited from AGENTS.md)

- Read closest `AGENTS.md` before editing any file.
- Use plan mode for non-trivial tasks, especially under `src/physics/` or `src/numerics/`.
- Work in git worktrees under `.agents/worktrees/`, not main checkout.
- Do NOT treat positive transverse macrodispersion as automatically physical.
- Do NOT merge scientific-core changes without validation evidence.
