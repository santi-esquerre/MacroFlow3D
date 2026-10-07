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

SF-33 (GPU inlet-label streamfunctions on the `x1`-non-periodic slab; claim (a) and the oracle established, claim (b) not established — closed `done` 2026-10-07 and superseded by the tracing constructor, `docs/decisions/2026-10-07-label-transport-constructor.md`): `2026-10-06-sf33-gpu-inlet-labels.md`.
