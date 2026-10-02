# Do periodic invariants exist? Streamline closure of periodic Darcy flow, and equation (14) versus Darcy

- Date: 2026-10-02
- Status: complete (CPU probes; not yet repeated on the production stack or at the paper's parameters)
- Theory: `docs/theory/lester-2023-key-claims.md` §2-§4 (the claims tested here);
  Lester et al. (2023) §3, §4, §5.1; Lester et al. (2021) §2.3, §3
- Used by: `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`

## Question

The solver plan stores `psi_i = affine + periodic fluctuation` on the 3-torus and
requires `grad psi1 x grad psi2 = v_Darcy`. Two questions that no increment had
tested independently of the streamfunction solver:

1. Does a smooth, scalar, triply periodic `k` produce a Darcy flow that ADMITS such a
   pair? A nondegenerate affine + periodic pair forces every streamline to be a closed
   curve on the torus (the level set of `(psi1, psi2)` is a compact 1-manifold; with
   `v1 > 0` the face map `(x2, x3) -> (psi1, psi2)` is a degree-one local
   diffeomorphism, hence injective). So the return map of the face `x1 = 0` must be the
   identity. This is checkable with a Darcy solve and a streamline integrator only.
2. When equation (14) (same-index pairing) is solved to a small residual with periodic
   fluctuations, is its flow the Darcy flow?

## Hypothesis

Paper (Lester 2023 §5.1): "integrable steady 3D periodic flows (such as zero helicity
flows) admit periodic streamlines"; solving (14) "yields the same velocity field as
that given by solving the flow potential". Prediction under the paper: return-map
displacement zero for every smooth scalar `k`; `e_v -> 0` with resolution.

Alternative derived before the runs: (14) is equivalent, where `c = grad psi1 x grad
psi2 != 0`, to `curl(c/k)` being parallel to `c` (two of the three components of
`curl(c/k) = 0`; the third is the helicity condition `B . c = 0`, which (14) does not
impose). It is the Euler-Lagrange system of the dissipation
`E = 1/2 <|grad psi1 x grad psi2|^2 / k>`, and
`1/2 <|c - v_D|^2 / k> = E[c] - E[v_D]` for equal mean flux. A solution of (14) is
therefore the closed-streamline field closest to the Darcy flow in the `1/k` energy
norm; it equals the Darcy flow only if the Darcy flow has closed streamlines.
Prediction: for a `k` without special symmetry the return map differs from the
identity at second order in the amplitude, and `e_v` of the converged (14) solution is
a resolution-independent `O(amplitude^2)` number.

## Build / environment

Local WSL, Python 3.13, numpy 2.5.0, scipy 1.18.0, CPU, double precision. No project
binary is involved. Scripts and raw outputs:
`docs/experiments/artifacts/2026-10-02-closure-probes/`.

## Config(s)

`k = exp(eps * f)` on `[0,1]^3`, mean flux exactly `e1` (three cell problems, mean
gradient `K_eff^-1 e1`). Fields `f` (all analytic, so every grid samples the same
continuum field):

| name | definition | property |
|---|---|---|
| `control2d` | three modes independent of `x3` | 2-D flow: streamlines must close |
| `lester2021` | Lester (2021) eq. (3.1): `sin 2πx1 cos 2πx2 sin 2πx3 + (2/5) sin 2πx1 sin 8πx3` | mirror-symmetric about `x1 = 1/4` |
| `lester_brk` | same, second term `sin(2πx1 + 0.9)` | mirror symmetry broken |
| `two_mode` | `cos 2π(x1+x2) + cos 2π(x1+x3)` | one continuous symmetry |
| `generic3d` | four oblique modes with phases | no symmetry |
| `gauss` | Gaussian-covariance random field, `ell = 1/4`, unit variance, seed 7, modes `|m|_inf <= 3` | the project's field class (`L/ell = 4` as in the 32^3 fixtures) |

Methods: pseudo-spectral Darcy solve (PCG to `1e-13`); streamlines of `grad phi`
parametrized by `x1` (`k` cancels), exact trigonometric evaluation of `grad phi`,
DOP853 at `rtol 1e-12`; pseudo-spectral same-index equation (14) with Anderson
(`m = 10`, Laplacian preconditioner), the residual of `probe_spectral.py` (SF-26).

## Commands

```bash
cd docs/experiments/artifacts/2026-10-02-closure-probes/scripts
python3 closure_probe.py <field>:<eps>:<N> ...        # return map of the face x1 = 0
python3 long_probe.py <field>:<eps>:<N>:<periods> ... # displacement after n periods
python3 eq14_vs_darcy.py <field>:<eps>:<N> ...        # eq. (14) solution vs Darcy
python3 variational_probe.py <field>:<eps>:<N>:{sp,fd2}
```

## Outputs inspected

### 1. Return map after one period (`raw/closure.txt`)

`rms` = RMS over 16 starting points of the transverse displacement minus its mean
(insensitive to a uniform drift). Integrator round trip (forward then backward):
`<= 4e-13` in every run.

| field | eps | N | displacement rms | note |
|---|---|---|---|---|
| `control2d` | 0.5 / 1.0 | 24 / 32 | 7e-13 / 2.7e-12 | closes (2-D control) |
| `lester2021` | 1.0 | 24, 32 | 8e-15, 5e-15 | closes (symmetry) |
| `lester_brk` | 1.0 | 24, 32 | 8.691e-3, 8.690e-3 | does not close |
| `two_mode` | 0.5, 0.25, 0.125, 0.0625 | 16 | 2.85e-2, 7.36e-3, 1.87e-3, 4.73e-4 | ratios 3.87, 3.93, 3.96 |
| `generic3d` | 0.5, 0.25, 0.125 | 16 | 5.40e-2, 1.35e-2, 3.38e-3 | ratios 3.99, 4.00 |
| `generic3d` | 0.5 | 16 vs 24 | 5.402e-2 vs 5.402e-2 | resolution-independent |
| `gauss` | 0.25, 0.5, 1.0 | 24 | 4.66e-3, 1.99e-2, 8.56e-2 | ratios 4.27, 4.30 |
| `gauss` | 0.5 | 16 vs 24 | 1.989e-2 vs 1.989e-2 | resolution-independent |
| `gauss` | 1.0 | 24 vs 32 | 8.559e-2 vs 8.559e-2 | resolution-independent |

`max(d phi/d x1) < 0` in every run (no backflow, no stagnation).

### 2. Displacement after `n` periods (`raw/long_run.txt`)

| field | n = 1 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| `two_mode`, eps 0.5, rms | 0.029 | 0.146 | 0.288 | 0.557 | 1.094 | 2.183 |
| `gauss`, eps 1.0, var(dx3) | 6.0e-3 | 9.0e-2 | 0.37 | 1.30 | 4.97 | |
| `gauss`, eps 1.0, var(dx2) | 1.3e-3 | 8.3e-3 | 1.3e-2 | 1.1e-2 | 1.2e-2 | |
| `control2d`, eps 1.0, var | 2e-17 | 6e-16 | 2e-15 | 9e-15 | | |
| `lester2021`, eps 1.0, var | 2e-25 | 5e-24 | 2e-23 | | | |

### 3. Equation (14) solution versus Darcy (`raw/eq14_vs_darcy_*.txt`)

`along c` / `across c`: RMS of the components of `curl(c/k)` parallel / perpendicular
to `c`, relative to RMS(`c/k`). Equation (14) removes only the perpendicular part.

| field | eps | N | `r_F` | `e_v` | `curl(c/k)` along c | across c |
|---|---|---|---|---|---|---|
| `control2d` | 0.25 | 16 / 24 | 3e-12 | 4.7e-6 / 1.6e-9 | 0 | 1.2e-4 / 6.0e-8 |
| `lester2021` | 0.25 | 32 | 2.0e-8 | 1.7e-7 | 9.8e-7 | 1.6e-6 |
| `two_mode` | 0.25 | 16 / 24 | 1e-11 | 7.743e-3 / 7.743e-3 | 6.5e-2 | 1.4e-8 / 1.5e-11 |
| `generic3d` | 0.125 | 16 / 24 | 4e-11 / 1e-11 | 3.536e-3 / 3.536e-3 | 2.5e-2 | 2.8e-6 / 2.1e-10 |
| `generic3d` | 0.25 | 16 | 4.7e-8 | 1.414e-2 | 9.6e-2 | 4.9e-5 |
| `gauss` | 0.25 | 16 / 24 | 5.4e-5 / 2.3e-6 | 4.894e-3 / 4.893e-3 | 3.7e-2 | 6.5e-4 / 2.4e-5 |
| `gauss` | 0.5 | 16 | 1.9e-3 (not converged) | 1.96e-2 | 0.14 | 1.7e-2 |

### 4. Naive energy minimization (`raw/variational_naive.txt`)

Minimizing a collocated discrete `E_h` (centered or spectral derivatives) with its exact
gradient drives `E_h` BELOW the Darcy energy (`0.4872 < 0.4950` at eps 0.25) with
`e_v = 12 %` and collapsing `min|c|`: the pointwise product `g1 x g2` is not discretely
divergence-free, and the minimizer exploits it. This probe is a negative result about
that discretization and supports no statement about residual floors.

## Result

- **R1 — Darcy streamlines of a generic smooth, scalar, periodic `k` are not closed.**
  The return map differs from the identity at second order in the amplitude (ratios
  3.9-4.3 per doubling), independently of resolution, for every field without a special
  symmetry, including the Gaussian-covariance field. The 2-D control closes to `1e-12`
  with the same code.
- **R2 — Hence no nondegenerate affine + periodic invariant pair exists for those
  flows**, and the target the solver plan has pursued since SF-02
  (`grad psi1 x grad psi2 = v_D` with periodic fluctuations) has no solution on them.
- **R3 — Equation (14) with periodic fluctuations is solvable, and its solution is not
  the Darcy flow.** With `r_F` between `1e-11` and `5e-8` the mismatch `e_v` is 3.5e-3 .. 1.4e-2,
  identical to four digits at two resolutions, scaling as amplitude squared, with
  `curl(c/k)` parallel to `c` and nonzero. On the 2-D control and on the symmetric
  Lester (2021) field `e_v` goes to zero.
- **R4 — The field used by Lester et al. (2021) §3 to show that the streamfunction and
  potential velocities converge is mirror-symmetric about `x1 = 1/4`**, which forces
  every streamline to close. Breaking the symmetry by a phase gives a displacement of
  8.7e-3 per period. That validation does not transfer to random fields.
- **R5 — The transverse displacement is not bounded.** It grows linearly with the
  number of periods for `two_mode` (all streamlines) and in one direction for the
  Gaussian field; the fluctuation of any invariant label is then unbounded, which is
  the assumption Lester (2023) §4 makes ("we assume that the fluctuations are also
  bounded") in the proof of zero transverse macrodispersion.
- **R6 — The recorded `e_v ~ 2.5 %` plateau is explained.** SF-21 (sigma^2 = 0.25,
  lambda = 1) and SF-25 "F-SAT" (sigma^2 = 1, lambda = 0.5) both sit at amplitude 0.5
  with `L/ell = 4`; the Gaussian probe field gives `e_v = 1.96e-2` at that amplitude
  from the amplitude-squared law. The plateau is the distance from the Darcy flow to
  the closed-streamline class, not truncation and not (only) the crossed pairing.

## Caveats

- Amplitudes `eps <= 1` (`sigma_Y^2 <= 1`) and `L/ell = 4`. The paper's case
  (`sigma^2 = 4`, `ell = 1/16`, 256^3) was not run; nothing here suggests an
  `O(amplitude^2)` effect disappears there, but it is not measured.
- One Gaussian realization (seed 7), band-limited to `|m|_inf <= 3`; not the SF-18
  generator, not the SF-19 Darcy solve, not the production residual.
- Periodic media only. The linear growth in R5 is a property of an exactly periodic
  cell; what it implies for the asymptotic transverse macrodispersion coefficient of a
  random, non-periodic medium is not measured here.
- `gauss` at eps 0.5 did not converge in (14) at 16^3 (`r_F = 1.9e-3`); its `e_v` is
  quoted because it follows the amplitude-squared law of the converged runs.
- Whether the production code's eta = 1 residual floor (SF-26) is related to R3 was
  not tested; the pseudo-spectral (14) converges to `1e-11` on these fields.

## Next step

Decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`:
repeat R1 on the production stack and at the paper's parameters before any further
solver work, and choose the object the tracker will preserve.
