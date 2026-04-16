# docs/AGENTS.md

## Scope

This file applies to work inside `docs/`.

The job of documentation here is not decoration.
It is part of the execution harness for the project.

Write docs so that an agent can use them to:

- plan,
- execute,
- validate,
- and report work correctly.

---

## What belongs in docs

Use `docs/` for durable, versioned knowledge such as:

- runbooks
- validation rules
- decision records
- experiment records
- plan templates
- subsystem overviews

Do not bury durable operating knowledge only in prompts or chat logs.

---

## Writing rules

- Be explicit.
- Prefer commands over vague descriptions.
- Prefer checklists over prose when operational.
- Keep each file narrowly scoped.
- State what a document is for, when to use it, and what not to use it for.
- If a workflow changes, update the runbook in the same branch.

---

## Required directories

- `docs/validation/`
- `docs/runbooks/`
- `docs/plans/`
- `docs/decisions/`
- `docs/experiments/`
- `docs/theory/`

Use README or template files so directories are never context-free.

---

## Theory notes

`docs/theory/` contains distilled scientific reference notes.
These are **not** optional background reading. They are part of the execution harness.

An agent working on scientific or numerical tasks must read the relevant theory note before planning.

| Note | When to read |
|------|-------------|
| `lester-2023-key-claims.md` | PSPTA, invariants, velocity reconstruction, transverse macrodispersion interpretation, tracking geometry |
| `beaudoin-de-dreuzy-2013-key-claims.md` | Large-domain macrodispersion studies, Monte Carlo design, historical 3D baseline comparisons, longitudinal validation |

When writing experiment notes or scientific change reports, cite the relevant theory note explicitly.

---

## Plans

Use `docs/plans/` for active or archived execution plans.

Active plans are authoritative operational documents, not optional reading:

| Plan | When to read |
|------|-------------|
| `docs/plans/active/pspta-execution-plan.md` | PSPTA, invariant recovery, eigensolver, refinement, helicity-free validation, transverse macrodispersion assessment |

A good plan contains:

- objective
- scope
- exact files/subsystems
- commands
- validation
- rollback/regression concerns
- completion criteria

---

## Decisions

Use `docs/decisions/` for architecture or workflow decisions.

Each decision record should include:

- title
- status
- date
- context
- decision
- consequences

Do not write essays.
Make the reason legible.

---

## Experiments

Use `docs/experiments/` for run records and scientific comparisons.

Every experiment note should contain:

- question
- hypothesis
- exact config(s)
- exact build(s)
- exact commands
- observed outputs
- conclusion
- caveats

If a claim depends on a run and there is no experiment note, the claim is weak.

---

## Done criteria for docs changes

A docs task is done when:

1. the relevant file exists or is updated,
2. the workflow described is executable,
3. the scope is correct,
4. the doc does not contradict the current repo behavior,
5. stale instructions were removed or fixed.
