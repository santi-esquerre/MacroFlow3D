@AGENTS.md
@ARCHITECTURE.md

## Claude Code project harness

This file is intentionally **role-neutral** because custom Claude Code subagents
also load project `CLAUDE.md` context. Agent-specific behavior belongs in
`.claude/agents/*.md`; durable scientific and engineering rules remain in
`AGENTS.md`, `ARCHITECTURE.md`, and `docs/`.

The checked-in Claude Code configuration is:

- `.claude/settings.json`
  - main session agent: `orchestrator`
  - permission mode: `bypassPermissions`
  - subagent worktree base: current local `HEAD`
  - nested subagent spawning disabled
- `.claude/agents/orchestrator.md`
  - Claude Fable 5
  - `xhigh` effort
  - owns UNDERSTAND -> PLAN -> EXECUTE -> AUDIT -> CORRECT -> INTEGRATE ->
    FINAL_AUDIT -> PUBLISH_PR -> review/closure coordination
- `.claude/agents/increment-worker.md`
  - Claude Sonnet 5
  - `medium` effort
  - one implementation/corrective DAG node
  - native `isolation: worktree`
- `.claude/agents/increment-integrator.md`
  - Claude Sonnet 5
  - `medium` effort
  - integrates only orchestrator-approved commits
  - native `isolation: worktree`

A custom agent's own system prompt defines its role. Do not reinterpret a worker
as an orchestrator merely because it can read this file.

### Increment scheduling semantics

The Lester dashboard remains the authority for **which increment may run**.

- Across increments: strictly sequential.
- Inside the active increment: the orchestrator may build a DAG and run
  independent nodes concurrently.
- Every source-writing worker runs in its own native Claude Code worktree.
- Dependent workers must be given the exact approved commit hashes they depend
  on and must incorporate those commits explicitly.
- The integrator starts from the increment base and integrates only approved
  commits.
- The orchestrator performs both worker-result audit and final integration
  audit.
- The autonomous implementation deliverable is an audited GitHub PR; Claude
  never merges it.
- Human-review increments stay `awaiting_review` until explicit human approval.
  Fable then resumes the same PR and adds only the closure metadata commit that
  marks `done` and advances `NEXT`.
- The next increment does not start until that closure state is merged and
  visible on the default branch.

Ephemeral orchestration records may live under:

`.claude/orchestration/<increment-id>/`

They are runtime evidence, not the canonical project state. The versioned
increment specification, checklist, bitácora, dashboard, experiment notes, and
validation records remain authoritative.

### Quick orientation

**Mission:** 3D macrodispersion in heterogeneous porous media. Current focus:
Lester equation (14) streamfunction solver for `psi1`, `psi2` in smooth, locally
isotropic Darcy flow; existing PSPTA code is legacy compatibility/migration
surface.

**Optimization priority:** correctness -> reproducibility -> maintainability ->
performance -> development speed.

### Routing table — read before working

| Work area | Read first |
|---|---|
| Lester equation (14) / new invariant construction | `docs/plans/active/lester-eq14-streamfunction-solver-overview.md` + `docs/theory/lester-2023-key-claims.md` |
| Active Lester increment | dashboard `NEXT` spec + `docs/runbooks/lester-increment-workflow.md` |
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

Existing project skills remain valid helpers when relevant:

- `macroflow-build` — local configure / build / test
- `macroflow-evals` — validation tier classification and execution
- `macroflow-physics-review` — scientific change review workflow
- `macroflow-pr-review` — branch / worktree / PR workflow
- `macroflow-remote-v100` — remote V100 build / test / run

Explicit commands remain available:

- `/project:build`
- `/project:validate`
- `/project:sync-v100`

### Hard rules

- Read the closest `AGENTS.md` before editing any file.
- Never implement substantial increment source changes in the orchestrator's
  control checkout.
- Never let a worker or integrator push, open, or merge the increment PR.
- Do NOT treat positive transverse macrodispersion as automatically physical.
- Do NOT accept scientific-core changes without validation evidence.
