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
SF-33 (GPU inlet-label streamfunctions on the `x1`-non-periodic slab; claim (a) and the oracle established, claim (b) not established — closed `done` 2026-10-07 and superseded by the tracing constructor, `docs/decisions/2026-10-07-label-transport-constructor.md`): `2026-10-06-sf33-gpu-inlet-labels.md`.
SF-35 (label transport by backward streamline tracing as the production constructor of the inlet-anchored Darcy labels; claim (a) established, claim (b) established at `sigma_Y` = 0.5 and 1.0 and not established at 1.5, integration certificate not met at three coarse points — closed `done` 2026-10-17 with the owner's validity envelope `sigma_Y <= 1`): `2026-10-07-sf35-label-transport.md` (artifacts under `artifacts/2026-10-07-sf35-label-transport/`).
SF-35 follow-up (feasibility of the gridded labels at `sigma_Y^2 = 4`: 512^3 fits with P2, refinement does not extend the convergent range, the obstruction is localized in the medium; exploratory, not pre-registered; basis of `docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`): `2026-10-17-sf35-sigma4-feasibility.md` (artifacts under `artifacts/2026-10-17-sf35-sigma4-feasibility/`).
SF-36 (Lagrangian-label tracker core on the cube stage: forward DP5(4) on the spline flow, per-particle labels, covectors, accumulated divergence; positions bitwise equal to the SF-30 host integrator, exact controls, SF-35 outputs byte-identical, P2 peak 6239 -> 3771 MiB at 256^3; E/A2-I/D amended by the owner after measurement — passive I/covectors limited by the C^2 spline's C^0 Hessian): `2026-10-17-sf36-lagrangian-tracker-core.md` (artifacts under `artifacts/2026-10-17-sf36-lagrangian-tracker-core/`).
