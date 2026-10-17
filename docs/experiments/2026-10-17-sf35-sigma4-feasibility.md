# SF-35 follow-up: feasibility of the gridded Darcy labels at sigma_Y^2 = 4 (probes J1/J2, controls P1/P2)

- Date: 2026-10-17 (probes 13:55Z-18:12Z on V100; note written 2026-10-17 on the SF-35 closure PR #53)
- Status: `complete — exploratory, NOT pre-registered; evidence for the owner decision of 2026-10-17, not acceptance evidence of any increment`
- Context: SF-35 closed `done` with the validity envelope O1 (`sigma_Y` 0.5 and 1.0 established, 1.5 not
  established; [`SF-35-label-transport-constructor.md`](../plans/active/lester-eq14/increments/SF-35-label-transport-constructor.md),
  [`2026-10-07-sf35-label-transport.md`](2026-10-07-sf35-label-transport.md)). The owner requires the
  macrodispersion study to reach at least `sigma_Y^2 = 4` on one V100-32GB. These probes asked whether the
  gridded labels can get there.
- Decision fed: [`2026-10-17-lagrangian-consumer-sigma4.md`](../decisions/2026-10-17-lagrangian-consumer-sigma4.md)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) §3, §5 (A: representation, B: integration), Rule 1
- Artifacts: [`artifacts/2026-10-17-sf35-sigma4-feasibility/`](artifacts/2026-10-17-sf35-sigma4-feasibility/README.md)
  (`probe.patch`, `logs/`, `raw/`, `analysis/`, `scripts/`)

Abbreviations (under `A = artifacts/2026-10-17-sf35-sigma4-feasibility/`): `J0c` = `A/logs/jobs/J0c-controls.log`,
`J1a`/`J1b`/`J2a`/`J2b` = `A/logs/jobs/J1a-n128.log` etc., `PL` = `A/analysis/planes_anchor0.md`, `MEM` =
`A/analysis/memory_phases.md`, `LD20` = `A/raw/J2/label_diagnostics_eps20.md`. Every number below is read from one of
these files.

## Question

On the SF-35 production field (SF-18 Gaussian, `ell = 1/4`, seed 3001, `sigma2 = 1` normalized) at `eps = 2`
(`k = exp(2 Y)`, i.e. `sigma_Y^2 = 4`):

1. Does a 512^3 construction fit on one V100-32GB, and where is the memory peak?
2. Does refinement 256 -> 512 extend the range in `x1` over which the finite-difference metrics of the traced
   labels converge?
3. Is the loss of convergence tied to the distance travelled from the inlet anchor (cured by re-anchoring) or
   localized in the medium?

## Hypothesis

None pre-registered (exploratory). Working hypothesis carried from the SF-35 post-hoc diagnostics (SF-35 note,
"Post-hoc diagnostics", NOT pre-registered): the FD non-convergence at large amplitude comes from a small set of
vertices where the labels carry steep, localized gradients unresolved at `N <= 256`. If that is right, refinement
should shrink the affected set; it says nothing about how fast.

## Build / environment

- Probe branch `worktree-agent-a53f621a4de3d5aaf` @ `d7b74d9` on `747adc7` (local only; diff = `A/probe.patch`).
  Binary built by job `J0-build` from source `e1e376e`; `git diff e1e376e d7b74d9 -- apps src CMakeLists.txt` is
  empty. Probe changes: `--save-field DIR` (unit-amplitude `Y_cells.npy`); **P2** = release of dead `N^3`
  temporaries (`K_dev`, `v`, `w` after the SF-19 solve, `u` after the inlet samples, the SF-19 workspace and
  `lnK_dev` after the spline prefilter) in `build_production_stage`, with `MF_PROBE_KEEP_TEMPS=1` restoring the
  previous lifetimes as control; job scripts `tools/probe_sf35/`.
- V100 `Tesla V100-PCIE-32GB`, CUDA 11.4.152, GCC 9.3.1, preset `v100-release`; `ctest -R inlet_slab` 9/9 in
  `J0-build`. Remote mirror `scripts/remote --increment SF-35-feas`.

## Config(s)

`inlet_slab --construct trace --eps 2 --sigma2 1 --ell 0.25 --seed 3001 --tracer gpu --threads 32 --n N`
(SF-35 production settings: `h_max = h/8`, `tol = 1e-8`, `max_roundtrip = 1e-8`), plus `--cells FILE` for the
re-anchored fields (`Y_cells` rolled along `x1` by `k` cells, `A/probe.patch` `tools/probe_sf35/roll_field.py`).

## Commands

Exact commands: `A/logs/commands/*` and the scripts in `A/probe.patch`.

| job | what |
|---|---|
| `J0-build` | build + `ctest -R inlet_slab` |
| `J0c-controls` | P1: `--cells` of the saved field vs the Gaussian source, 64^3 and 128^3; P2: P2 on vs `MF_PROBE_KEEP_TEMPS=1`, 64^3 and 128^3; roll sanity (`k = N` identity, `k = 12` changes the inlet) |
| `J1a-n128`, `J1b-n256` | re-anchoring: field rolled so that the inlet sits at original `x1 = 0.75 ell` (`k = 24/48`) and `0.5 ell` (`k = 16/32`), 128 -> 256 |
| `J2a-n256`, `J2b-n512` | 256^3 (with `--save-field`) and 512^3 with nvidia-smi sampling (`memory.used`, ~200 ms); per-plane orders 256 -> 512; SF-35 `label_diagnostics.py` on the eps 2 exports 128/256/512 |

Job `J0b-controls` failed with exit 126 (scripts not executable; fixed by `aaed917`, then re-run as `J0c`).

## Outputs inspected

### Controls (`J0c`)

- **P1** (`--cells` reproduces the Gaussian source): max `|dpsi|` = 0.000e+00 for both labels at 64^3 and 128^3,
  no NaN; also at 128^3 in `J1a` (`N128_ctrl`).
- **P2** (release of dead temporaries is byte-neutral): the CASE/PLANE/BACKFLOW/GROWTH/RESIDUAL/TRANSPORT_STATUS/
  STATUS/FIELD lines are byte-identical (72 lines at 64^3, 136 at 128^3) and `psi1.npy`, `psi2.npy` are
  byte-identical between P2 on and off.
- Roll sanity: rolling by `k = N` is the identity (bytes equal); `k = 12` at 64^3 changes `inlet_vmin` from
  1.342e-03 to 2.012e-03 (the inlet moved).

### Q1: 512^3 fits; the peak is in the flow construction (`J2b`, `MEM`)

| N | status | max round trip | trace [s] | field [s] | stage [s] | device peak (nvidia-smi) | host max RSS (`time -v`) |
|---|---|---|---|---|---|---|---|
| 256 | `roundtrip_exceeded` (exit 21) | 1.621e-08 | 555.3 | 7.4 | 8.7 | 3769 MiB | 3.59 GiB |
| 512 | `ok` | 2.539e-09 | 14116.6 | 59.0 | 62.7 | 25925 MiB (25.32 GiB) | 24.63 GiB |

- Phases at 512^3 (`MEM`): field <= 14825 MiB; **stage (SF-19 solve + SF-28 prefilters) peaks at 25925 MiB**
  (90.5 s into the run); trace body <= 20629 MiB; end-of-run FD diagnostics <= 23385 MiB. Per cell: 202.5 B/cell at
  the peak (includes the CUDA context). The driver's own `peak_device_bytes` (cudaMemGetInfo polled only at phase
  boundaries) reads 24520949760 B and misses the in-phase peak.
- SF-19 at 512^3: 8 MG levels, the three correctors converged in 20 iterations each (`rel_residual` <= 1.914e-12);
  `div_rms` 2.862e-10. Inlet `vmin` 1.340e-03 > 0, `vD1_min` 8.865e-04 > 0, every backflow counter 0: no backflow
  and no stagnation at the vertex samples or on the traced streamlines, at 256 and 512.
- Trace time ratio 512/256 = 25.4.

### Q2: refinement does not extend the convergent range (`PL`)

Per-plane `e_v_j` observed orders at the original anchor:

| x1/ell | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|---|---|
| 128 -> 256 | 2.25 | 2.74 | 2.72 | -0.15 | -0.78 | 0.11 | -0.75 | -0.87 | -0.71 |
| 256 -> 512 | 2.05 | 2.91 | 3.23 | -0.24 | -0.68 | -0.95 | -0.84 | -0.88 | -1.07 |

`e_v_j` converges up to `x1 = 0.75 ell` and not from `1.0 ell` on, on both pairs. The invariance errors behave
differently: `e_i1_j` keeps orders 2.57-2.74 up to `2 ell` on 128 -> 256 but falls to 0.29-1.79 beyond `0.75 ell`
on 256 -> 512; `e_i2_j` is below order 1.6 beyond `0.75 ell` on both pairs.

Label values (`LD20`, section 2, common vertices): the labels self-converge in RMS (256 -> 512 order 1.99 for
`psi1`, 1.79 for `psi2`) but not in max (orders 0.90, 0.53; max `|dpsi|` 2.050e-02 and 8.943e-02 between 256 and
512). Concentration of the non-convergence of `c = grad psi1 x grad psi2` (`LD20`, section 3): half of the squared
difference comes from 376 vertices (0.0181 %) for 128 vs 256 and from 1100 vertices (0.0066 %) for 256 vs 512;
the count grows by 2.9x per doubling (between the 2x of a curve and the 4x of a sheet), and the maximum sits at
the same place on both pairs, (x1, x2, x3) = (0.96-0.98, 0.02-0.04, 0.547) in cube units, i.e. `x1 ~ 3.85-3.9 ell`,
near the outlet plane — downstream of the `x1 ~ 0.8-1.0 ell` where convergence is first lost (`LD20` uses cube units).

### Q3: the obstruction is localized in the medium (`J1b`)

128 -> 256 with the inlet moved downstream (original `x1/ell` = local + shift):

| inlet at | local 0.25 (orig.) | `e_v` order | `e_i1` | `e_i2` | local 0.5 (orig.) | `e_v` order |
|---|---|---|---|---|---|---|
| `0.75 ell` | 1.00 | 0.51 | 1.89 | 0.62 | 1.25 | -0.48 |
| `0.5 ell` | 0.75 | 2.27 | 2.35 | 2.51 | 1.00 | -0.41 |

With the inlet at `0.5 ell`, the plane at original `0.75 ell` still converges and the one at original `1.0 ell`
does not; with the inlet at `0.75 ell`, the first plane (original `1.0 ell`) already does not. The loss of
convergence follows the original position `x1 ~ 0.8-1.0 ell`, not the distance from the anchor.

## Result

Classification per `docs/AGENTS.md`.

- **Confirmed in runs.** A 512^3 construction at `sigma_Y^2 = 4` fits on one V100-32GB once the dead temporaries are
  released (P2, byte-neutral at 64/128): peak 25925 MiB, in the flow construction, not in the tracing or the FD
  diagnostics. The integration certificate B of SF-35 is met at 512^3 (2.54e-09) and not at 256^3 (1.62e-08).
- **Confirmed in runs.** Refinement 256 -> 512 does not extend the range of convergent `e_v_j`: it stays at
  `x1 <= 0.75 ell` on 128 -> 256 and on 256 -> 512 (`PL`). Uniform refinement within the memory ceiling (option O2
  of the SF-35 note) does not deliver verified gridded labels at `sigma_Y^2 = 4` on this field.
- **Confirmed in runs.** Re-anchoring does not remove the obstruction: it is tied to the position `x1 ~ 0.8-1.0 ell`
  of this field, not to the distance from the inlet (J1). Segment-wise re-anchoring (option O5 of the planning
  analysis) is refuted on this field.
- **Interpretation (hypothesis, not established).** With `v1 > 0` everywhere, the exact label map of this flow has
  no stagnation or topology change. The labels converge in RMS but not in max; the non-convergence of `c`
  concentrates in a set whose vertex count grows ~2.9x per doubling and whose maximum does not move. This reads as a
  thin layer of extreme label gradient generated where the flow is slow (`min_c` ~ 1.2e-04-1.4e-04 at 256-512;
  `e_v_j` jumps between `x1 = 0.75 ell` and `1.0 ell` by 13x / 96x / 1067x at 128 / 256 / 512) and carried downstream, because the labels are
  invariants. Its width is below what 512^3 (`h = ell/128`) resolves; a uniform resolution sufficient for it is not
  estimated here. On this reading it is a property of the object (the exact labels of this flow at
  `sigma_Y^2 = 4`), not a defect of the constructor.
- **Scale consequence (derived from the measured cost; not a run).** The long domain of the plan
  (2048x256x256 = 1.34e8 cells = the cell count of 512^3, `ell/h = 10`) fits the same 25.3 GiB (25925 MiB) peak, but its
  `ell/h` is 12.8x coarser than 512^3, where the derivatives already fail beyond `0.75 ell`. Gridded labels there
  would be unverifiable; their FD verification at `ell/128` would need ~2000x the cells of 512^3. A Lagrangian
  consumer that carries labels on particles needs only the flow and its spline (the same 25.3 GiB peak) and no
  gridded derivatives of the labels (decision 2026-10-17).

### Erratum to the runtime planning analysis

The runtime analysis `.claude/orchestration/SF-35-label-transport-constructor/analysis/sigma4-feasibility.md` (not
versioned) contained three errors, corrected there and here: (1) the memory peak is in the stage (flow + splines),
not in the FD diagnostics; the "+13 GB diagnostics" figure came from cudaMemGetInfo polls at phase boundaries;
(2) the SF-35 constructor integrates in arclength and follows and counts interior backflow and stagnation
(`SlabLabelTransport.cuh`); only inlet backflow (`v1 <= 0` on the inlet face) stops a case and `seed_backflow` seeds
get NaN, so the claim that it is "parametrized in `x1`" was wrong; (3) per realization at 1.34e8 cells the
field + flow construction (~2 min at 512^3: 59 s + 63 s, measured) is estimated to dominate a Lagrangian ensemble,
not the tracing (the tracing cost is not measured here; SF-36 criterion J measures it).

## Caveats

- One field realization (seed 3001), one `ell = 1/4`, one amplitude; the localization at `x1 ~ 0.8-1.0 ell` is a
  property of this realization. Generality is to be tested on fresh seeds (SF-37 uses 3101-3103).
- Exploratory runs, no pre-registered thresholds; the digest `PL` (128 -> 256 at the original anchor) was produced
  after the probes from committed JSON.
- nvidia-smi `memory.used` is device-wide and includes the CUDA context; the sampler period drifted to 200.6 ms
  (the phase windows of `MEM` use the effective period and are approximate to ~1 s).
- The P2 byte-neutrality was shown at 64^3 and 128^3 only; at 512^3 there is no P2-off run, and whether one would
  fit was only estimated in the unversioned planning analysis, not measured.
- The label exports and saved fields are on the V100 host only (`~/sf35_feas_probe/`), not in the repository.

## Next step

Owner decision 2026-10-17 (`docs/decisions/2026-10-17-lagrangian-consumer-sigma4.md`): Lagrangian labels carried by
forward-traced particles as the production consumer at `sigma^2 >= 4` (SF-36..SF-39); gridded labels as diagnostic
with the O1 envelope as anchor (SF-34 re-specified) and a blind, pre-registered robust metric (O3, in SF-37). P2 is
adopted into SF-36 with its byte-identity control.
