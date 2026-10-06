# Experiments

Use this directory for experiment notes and validation records tied to real runs.

A valid experiment note should contain:
- question
- hypothesis
- build/config
- commands
- outputs
- conclusion
- caveats

Heavy streamfunction cases that are not ctest entries (since SF-27) are indexed in `2026-10-02-heavy-streamfunction-cases-index.md`.

SF-29 (CPU prototype of equation (14) with `x1` non-periodic and inlet labels; formulation decided 2026-10-06): `2026-10-02-sf29-inlet-labels.md`.

SF-32 (face-flux reference trackers and the paper's scalings; spurious transverse spreading of Pollock/RK vs the pseudo-symplectic tracker on surrogates with exact invariants, five label fields, V100): `2026-10-06-sf32-spurious-spreading.md` (artifacts under `artifacts/2026-10-06-sf32-spurious-spreading/`).
