# docs/theory/

Distilled scientific reference notes for MacroFlow3D.

These are **not** optional background reading.
They are part of the execution harness: agents must read the relevant note before planning scientific or numerical work.

---

## Notes

| File | Paper | Role in repo |
|------|-------|-------------|
| `lester-2023-key-claims.md` | Lester (2023) — kinematic constraints on 3D Darcy transport | Modern constraint reference. Defines the helicity-free regime, proves two invariants exist, establishes that purely advective transverse macrodispersion is zero in smooth isotropic Darcy. Foundation for PSPTA and invariant-preserving tracking. |
| `beaudoin-de-dreuzy-2013-key-claims.md` | Beaudoin & de Dreuzy (2013) — 3D macrodispersion | Classical baseline reference. Reports positive α_T in numerical 3D Darcy, provides domain-design and Monte Carlo discipline. Must be interpreted with regime awareness after Lester. |

---

## When to read which note

| Task | Required | Optional |
|------|----------|----------|
| PSPTA / invariant work | Lester | Beaudoin |
| Velocity reconstruction | Lester | — |
| Transverse macrodispersion interpretation | Lester | Beaudoin |
| Longitudinal macrodispersion validation | Beaudoin | — |
| Large-domain / Monte Carlo design | Beaudoin | Lester |
| Historical literature comparison | Both | — |

---

## Cross-references

- `docs/validation/acceptance-gates.md` — gates informed by these notes
- `src/physics/particles/pspta/AGENTS.md` — PSPTA pre-reading requirement
- `skills/macroflow-physics-review/SKILL.md` — review workflow referencing both notes
- `skills/macroflow-evals/SKILL.md` — eval tiers referencing both notes
