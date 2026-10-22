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
SF-37 (verification of the Lagrangian consumer on the periodic cube at `sigma^2` in {1, 2.25, 4} with fresh seeds 3101-3103: V1 tolerance ladder met but vacuous (RR15, tol-independent step sequence; the `h/8` vs `h/32` shadow passes), V2 grid ladder met (det branch), production `lambda/h` 8 / 16 / 32 with study value 32, V4 Lagrangian `e_v` order ~2, V3a gridded labels 12/12 concordant, V3b Pollock 54 discordant / 9 concordant of 63 (non-blocking, owner), V5 named fit insufficient, blind O3 5 of 6 `accept` (one `unresolved`, `q = 1e-3` disclosed), regression R byte-identical and `ctest` 38/38; no stop rule fires per RR14; no `alpha` claim): `2026-10-19-sf37-lagrangian-cube-verification.md` (artifacts under `artifacts/2026-10-19-sf37-lagrangian-cube-verification/`).
SF-37 post-hoc review (2026-10-22, not pre-registered, no verdict changes; written for the human review of PR #56: V3b weight mechanism confirmed with the face-ownership rule (+-h/2 by the sign of the transverse velocity), Pollock closer to the fine reference than O6 at equal N, alpha-equivalent grid-error scale, plume growth superdiffusive in the cube, explanation of the Lagrangian tracker and its SF-36/SF-37 validation, feasibility in the MECOM 2025 Par2 domain: sigma^2 <= 1 feasible after SF-38, >= 2.25 not with the current stage): `2026-10-22-sf37-posthoc-review-and-mecom-feasibility.md`; scripts and outputs in `artifacts/2026-10-19-sf37-lagrangian-cube-verification/{scripts,analysis}/posthoc/`.
Lagrangian path memory and time audit after SF-37 (2026-10-22, code analysis plus existing measurements, not pre-registered, no new runs except three read-only V100 queries; evidence for the owner decision of 2026-10-22, not acceptance evidence: phase-by-phase byte model reconciled with the SF-37 512^3 peak distribution 25921 ... 27981 MiB (the 200 ms sampler misses the cuFFT work-area transients; exact-byte stage peak 212.7 B/cell), tracker 16 B/cell + 648 B/particle with 255 registers, Par2 baseline ~149 B/cell with 24 B/cell of never-read buffers, CSV-reader header bug, bitwise memory levels O1/O2/S/S+ = 92.6/84.6/68.6/60.6 B/cell and feasibility of 3328x384x384 within 30000 MiB at S+; no `alpha` claim): `2026-10-22-lagrangian-memory-audit.md` (artifacts under `artifacts/2026-10-22-lagrangian-memory-audit/`).
