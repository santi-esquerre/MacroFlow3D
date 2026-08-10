#pragma once

/**
 * @file PeriodicGaussianField.cuh
 * @brief Truly torus-periodic Gaussian-covariance log-conductivity generator
 * @ingroup physics_stochastic
 *
 * SF-18 (Lester eq.(14) increment). Standalone spectral generator, independent
 * of `stochastic.cuh`'s continuous-wavevector direct-sum generator (Räss,
 * Kolyukhin, Minakov 2019), which is NOT torus-periodic and is left
 * byte-untouched. This generator draws Fourier coefficients on the EXACT
 * reciprocal lattice of the periodic box and reconstructs the field with a
 * cuFFT inverse transform, so the sampled field and (numerically, to
 * discretization error) its derivatives join seamlessly across the periodic
 * boundary.
 *
 * ---------------------------------------------------------------------------
 * Locked mathematical conventions (SF-18 activation decisions 1-5; see
 * `docs/plans/active/lester-eq14/increments/SF-18-periodic-gaussian-generator.md`
 * bitácora for the authoritative record)
 * ---------------------------------------------------------------------------
 *
 * 1) Target continuous covariance and DFT convention
 *
 *    C_Y(r) = sigma2 * exp(-(r/l)^2),   l = corr_length
 *
 *    Y(x_j) = sum_m Yhat_m * exp(i * k_m . x_j),
 *    k_m = 2*pi*(m1/L1, m2/L2, m3/L3),   L_d = N_d * h_d
 *
 * 2) Periodicized spectrum (mode variance)
 *
 *    E|Yhat_m|^2 = sigma2 * pi^(3/2) * l^3 * exp(-|k_m|^2 * l^2 / 4)
 *                  / (L1*L2*L3),        for m != 0
 *
 *    This is exactly the identity obtained by periodizing (wrapping) the
 *    continuous covariance C_Y around the torus and taking its 3D Fourier
 *    series coefficients; it is the documented meaning of "periodicized
 *    Gaussian spectrum" for this generator. Each active mode is complex
 *    Gaussian with the real/imaginary parts independently distributed as
 *    N(0, S_m/2), so that E[Re^2] + E[Im^2] = S_m matches the formula above.
 *
 * 3) Zero mode / log-mean policy
 *
 *    Yhat_0 is EXCLUDED from the random sum (set to exactly 0), so the
 *    discrete sample mean of the generated fluctuation is exactly zero up to
 *    FFT round-off. This generator does not apply any log-mean shift; callers
 *    apply the project's existing `K = K_g * exp(Y)` convention (see
 *    `stochastic.cuh::generate_K_lognormal`) at exponentiation time. The
 *    realized (raw) sample mean is reported in `PeriodicGaussianFieldReport`
 *    for diagnostic purposes.
 *
 * 4) Nyquist handling and refinement-under-truncation guarantee
 *
 *    Every mode with any component m_d == -N_d/2 (the Nyquist plane of axis
 *    d) is zeroed. The active mode set is therefore exactly
 *    `|m_d| <= N_d/2 - 1` for all three axes. Because the coefficient of a
 *    surviving mode depends only on the INTEGER mode index (m1,m2,m3) and the
 *    seed (see (6)), a finer grid resolves a strict SUPERSET of a coarser
 *    grid's active modes with IDENTICAL coefficients: this generator realizes
 *    the SAME continuous random field under grid refinement via spectral
 *    truncation (as opposed to "generate at the finest grid, then restrict").
 *    Truncation error away from the coarse cutoff is controlled by the
 *    Gaussian spectral decay (negligible for corr_length small relative to
 *    the truncation wavenumber; e.g. ~exp(-153) at a 128^3 cutoff for
 *    l = L/16).
 *
 * 5) Seed-to-mode hash (stateless, order-independent, grid-independent)
 *
 *    Each surviving mode's canonical complex coefficient is produced from a
 *    STATELESS counter-based hash of (seed, m1, m2, m3) — NOT from array
 *    position or generation order, and NOT from cuRAND state arrays:
 *
 *      splitmix64(x):
 *        x += 0x9E3779B97F4A7C15
 *        x  = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
 *        x  = (x ^ (x >> 27)) * 0x94D049BB133111EB
 *        x  = x ^ (x >> 31)
 *        return x
 *
 *      hash_mode(seed, m1, m2, m3, salt):
 *        h = seed ^ 0x9E3779B97F4A7C15
 *        h = splitmix64(h ^ (int64(m1) * 0xD1B54A32D192ED03))
 *        h = splitmix64(h ^ (int64(m2) * 0xA24BAED4963EE407))
 *        h = splitmix64(h ^ (int64(m3) * 0x9FB21C651E98DF25))
 *        h = splitmix64(h ^ uint64(salt))
 *        return h
 *
 *    Two independent uniform doubles in (0,1) are derived from
 *    `hash_mode(..., salt=0)` and `hash_mode(..., salt=1)` by taking the top
 *    53 bits, adding 1, and dividing by (2^53 + 1) (strictly inside (0,1), so
 *    log(u1) never sees 0). A standard Box-Muller transform turns (u1,u2)
 *    into an independent standard-normal pair (re_std, im_std); the stored
 *    coefficient is `(re_std, im_std) * sqrt(S_m / 2)` (see (2)).
 *
 *    Determinism argument: for fixed (seed, m1, m2, m3) this is a pure
 *    function with no shared mutable state, so results are independent of
 *    launch configuration, thread scheduling, and grid size. cuFFT is
 *    deterministic for a fixed plan/input/GPU (verified empirically by the
 *    bitwise-reproducibility check in this node's validation, ahead of the
 *    formal T02 test).
 *
 * 6) Hermitian symmetry and half-spectrum (c2r) layout
 *
 *    Reality requires `Yhat_{-m} = conj(Yhat_m)`. This generator stores the
 *    half-spectrum in the layout that matches this project's field indexing
 *    convention `Grid3D::idx(i,j,k) = i + nx*(j + ny*k)` (x is the fastest,
 *    contiguous axis) -- i.e. the HALVED axis is x, not the generic "last"
 *    axis used in the increment's placeholder description. Concretely the
 *    half-spectrum is a `(nx/2+1) x ny x nz` array of `cufftDoubleComplex`
 *    with index `gx + (nx/2+1)*(gy + ny*gz)`, `gx in [0, nx/2]` holding
 *    non-negative x-frequencies `mx = gx`, and `gy in [0, ny-1]`,
 *    `gz in [0, nz-1]` holding WRAPPED signed frequencies
 *    `my = gy < ny/2 ? gy : gy - ny` (and likewise for mz). This choice keeps
 *    the complex buffer layout bit-for-bit compatible with `y_out`'s existing
 *    memory layout without any transposition, matching `cufftPlan3d(nz, ny,
 *    nx, CUFFT_Z2D)` (cuFFT's row-major convention with the LAST argument
 *    fastest maps exactly onto this project's x-fastest storage).
 *
 *    For `mx > 0` every stored entry is one representative of a {+m,-m} pair
 *    whose mirror (negative x-frequency) is NOT stored; cuFFT's c2r transform
 *    supplies it implicitly. No extra bookkeeping is required there.
 *
 *    For `mx == 0` BOTH members of a {+m,-m} pair in the (my,mz) plane are
 *    physically present in the stored array (only x is halved), so this
 *    plane is self-referential and needs an explicit canonical rule:
 *
 *      canonical(my,mz) := (my > 0) OR (my == 0 AND mz > 0)
 *
 *    The canonical member of each pair draws its coefficient directly from
 *    `hash_mode(seed, 0, my, mz, .)`; the mirror is assigned the complex
 *    conjugate of the canonical value (computed by hashing the CANONICAL
 *    index and negating the imaginary part), never drawn independently. The
 *    only self-conjugate mode at mx==0 is (my,mz)=(0,0), which is exactly
 *    the excluded zero mode (3); with the Nyquist planes zeroed (4) there is
 *    no other self-conjugate active mode to special-case.
 *
 * 7) Sample locations and the half-cell twist
 *
 *    Samples are taken at cell centers, `x = h*(i + 1/2)` (project
 *    convention, matches `stochastic.cuh`). This is implemented as an exact
 *    phase twist applied to each stored spectral coefficient BEFORE the
 *    inverse transform:
 *
 *      Yhat_m *= exp(i * (k_m1*h1 + k_m2*h2 + k_m3*h3) / 2)
 *
 *    applied using the wavevector of the coefficient's OWN stored position
 *    (after Hermitian assignment, not the canonical position it may have
 *    been mirrored from). This preserves `Yhat_{-m} = conj(Yhat_m)` for the
 *    twisted array: writing k_{-m} = -k_m, conj(twist(Yhat_m)) =
 *    conj(Yhat_m)*exp(-i k_m.h/2) = Yhat_{-m}*exp(i k_{-m}.h/2) =
 *    twist(Yhat_{-m}).
 *
 * 8) Normalization (`normalize_variance`, default true)
 *
 *    When true, the realized field is uniformly rescaled in place so its
 *    SAMPLE variance equals `sigma2` exactly (to double-precision round-off):
 *    `y *= sqrt(sigma2 / raw_variance)`. The scale is a single uniform
 *    multiplier (no mean shift), so the realized field's spectral SHAPE is
 *    preserved. Both the RAW realized mean/variance and the APPLIED scale
 *    are recorded in `PeriodicGaussianFieldReport`, together with the
 *    post-scale `final_variance`. Refinement-consistency testing (T02) uses
 *    `normalize_variance = false` so cross-grid spectral coefficients are
 *    compared directly, without a truncation-dependent rescale confounding
 *    the comparison.
 *
 * ---------------------------------------------------------------------------
 * cuFFT usage and memory accounting
 * ---------------------------------------------------------------------------
 *
 * A double-precision complex-to-real (`CUFFT_Z2D`) 3D out-of-place inverse
 * transform is used, with `cufftPlan3d(nz, ny, nx, CUFFT_Z2D)` (see (6) for
 * why the argument order matches this project's x-fastest layout) and
 * `cufftSetStream` bound to `CudaContext::cuda_stream()` so the transform is
 * stream-ordered with the rest of the pipeline. The cuFFT plan (and its
 * internal work area) is created and destroyed PER CALL in this v1 -- this is
 * a deliberate, documented simplification (no persistent plan cache yet); the
 * work area's byte size is queried via `cufftGetSize` immediately after plan
 * creation and reported in `PeriodicGaussianFieldReport::cufft_work_area_bytes`
 * even though it is freed again before `generate_periodic_gaussian_field`
 * returns. All OTHER device memory (half-spectrum buffer, reduction scratch,
 * atomic mode counter) is owned by `PeriodicGaussianFieldWorkspace` and
 * reused/resized (never shrunk) across calls; its exact byte sizes are also
 * reported so a caller can do exact-byte capacity planning, matching project
 * convention.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../runtime/CudaContext.cuh"
#include <cstddef>
#include <cufft.h>

namespace macroflow3d {
namespace physics {

/**
 * @brief Configuration for the periodic Gaussian generator.
 *
 * Distinct from and independent of `StochasticConfig` (the direct-sum
 * generator's config type). See file header for the exact mathematical
 * conventions this config parameterizes.
 */
struct PeriodicGaussianFieldConfig {
    real sigma2 = 1.0;                    ///< Target field variance sigma^2 (>= 0)
    real corr_length = 1.0;               ///< Gaussian correlation length l (> 0)
    unsigned long long seed = 0ULL;       ///< Base seed for the stateless mode hash
    bool normalize_variance = true;       ///< Rescale realized sample variance to sigma2

    PeriodicGaussianFieldConfig() = default;

    /**
     * @brief Host-side validation.
     *
     * Throws std::invalid_argument with a distinct message per violated
     * precondition:
     *   - sigma2 must be finite and >= 0
     *   - corr_length must be finite and > 0
     *   - grid extents (nx,ny,nz) must be positive
     *   - grid extents (nx,ny,nz) must be even (required by the c2r
     *     half-spectrum layout and the m_d == -N_d/2 Nyquist-zeroing rule)
     *   - grid spacings (dx,dy,dz) must be finite and > 0
     */
    void validate(const Grid3D& grid) const;
};

/**
 * @brief Exact-byte memory/statistics report for one generation call.
 *
 * All *_bytes fields are exact sums of owned device allocations (project
 * convention: no fudge factors). `cufft_work_area_bytes` is transient (freed
 * before return, see file header) but is included in `total_device_bytes`
 * because it reflects real peak device usage during the call.
 */
struct PeriodicGaussianFieldReport {
    real raw_mean = 0.0;              ///< Sample mean of the field before normalization
    real raw_variance = 0.0;          ///< Sample variance of the field before normalization
    real applied_scale = 1.0;         ///< Multiplier applied (1.0 if normalize_variance == false)
    real final_variance = 0.0;        ///< Sample variance after the applied scale
    size_t active_mode_count = 0;     ///< Number of half-spectrum entries actually drawn (non-zeroed)

    size_t spectrum_bytes = 0;            ///< Bytes owned by the half-spectrum device buffer
    size_t cufft_work_area_bytes = 0;     ///< Bytes reported by cufftGetSize for this call's plan
    size_t reduction_scratch_bytes = 0;   ///< Bytes owned by reduction/counter scratch buffers
    size_t total_device_bytes = 0;        ///< Sum of the three byte fields above
};

/**
 * @brief Owned device workspace reused across generation calls.
 *
 * Buffers are grown (never shrunk) on demand by
 * `generate_periodic_gaussian_field`. Not copyable (owns device memory via
 * DeviceBuffer); default-movable.
 */
struct PeriodicGaussianFieldWorkspace {
    DeviceBuffer<cufftDoubleComplex> spectrum;    ///< Half-spectrum, (nx/2+1)*ny*nz entries
    DeviceBuffer<real> reduce_blocks;             ///< Per-block partial sums for mean/variance
    DeviceBuffer<unsigned long long> mode_counter; ///< Single-element active-mode atomic counter

    PeriodicGaussianFieldWorkspace() = default;
};

/**
 * @brief Generate one truly periodic Gaussian-covariance realization.
 *
 * @param ctx        CUDA context; the transform and all kernels are ordered
 *                    on ctx.cuda_stream().
 * @param grid        Target grid (defines N_d, h_d, and therefore L_d).
 * @param cfg        Generator configuration (see PeriodicGaussianFieldConfig).
 * @param y_out      Output device field, size must equal grid.num_cells();
 *                    uses the project's standard x-fastest cell-index layout.
 * @param workspace  Owned scratch buffers, resized as needed.
 * @return           Exact-byte memory/statistics report for this call.
 *
 * Throws std::invalid_argument for invalid cfg/grid/y_out combinations (see
 * PeriodicGaussianFieldConfig::validate) and std::runtime_error for CUDA or
 * cuFFT failures.
 */
PeriodicGaussianFieldReport
generate_periodic_gaussian_field(const CudaContext& ctx, const Grid3D& grid,
                                 const PeriodicGaussianFieldConfig& cfg, DeviceSpan<real> y_out,
                                 PeriodicGaussianFieldWorkspace& workspace);

} // namespace physics
} // namespace macroflow3d
