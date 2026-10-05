#pragma once

/**
 * @file PeriodicTricubicBSpline.cuh
 * @brief C^2 periodic tricubic B-spline interpolation of cell-centered, triply
 *        periodic double fields: value and analytic gradient, GPU with CPU mirror.
 * @ingroup numerics
 *
 * SF-28 (Lester eq.(14) roadmap, R1). Shared by the streamline-closure oracle
 * (SF-30, Darcy potential) and the streamline tracker (SF-31, labels). The
 * authoritative record of the conventions below is the increment specification
 * `docs/plans/active/lester-eq14/increments/SF-28-periodic-tricubic-spline.md`.
 *
 * ---------------------------------------------------------------------------
 * 1) Field conventions
 * ---------------------------------------------------------------------------
 *
 *  - Samples are cell-centered, `real = double`, triply periodic, stored in the
 *    project layout `Grid3D::idx(i,j,k) = i + nx*(j + ny*k)` (x fastest).
 *  - Axis d has N_d cells, spacing h_d (grid.dx/dy/dz), period L_d = N_d h_d
 *    (grid.Lx()/Ly()/Lz()); the domain origin is 0, so the canonical period is
 *    [0, L_d) and cell i has its centre at x_i = (i + 1/2) h_d.
 *  - Per-axis spacings may differ (dx = dy = dz is NOT required here; the
 *    multigrid isotropy note of Grid3D does not bind this module).
 *  - Size rule: every N_d >= 4 (the 4-point stencil of an axis must not wrap
 *    onto itself), odd or even. Odd N_d is supported: the cuFFT D2Z/Z2D
 *    half-spectrum layout (N_x/2 + 1) x N_y x N_z is valid for any size (an
 *    odd axis simply has no Nyquist entry) and the symbol of section 3 is
 *    valid and >= 1/3 for every N >= 4. Spacings must be finite and > 0.
 *    Violations throw std::invalid_argument with distinct messages.
 *
 * ---------------------------------------------------------------------------
 * 2) Interpolant
 * ---------------------------------------------------------------------------
 *
 *    s(x) = sum_{j1,j2,j3} c_{j1 j2 j3} prod_d beta3((x_d - x_{j_d}) / h_d)
 *
 *  over the periodic coefficient lattice (indices mod N_d), beta3 = centered
 *  uniform cubic B-spline (support [-2,2], beta3(0) = 2/3, beta3(+-1) = 1/6).
 *  Piecewise cubic and C^2 by construction; periodic by construction.
 *  Coefficients share the sample index lattice (coefficient j_d sits at the
 *  cell centre x_{j_d}); there is NO half-cell phase twist anywhere.
 *
 * ---------------------------------------------------------------------------
 * 3) Prefilter (coefficient deconvolution)
 * ---------------------------------------------------------------------------
 *
 *  Interpolation s(x_i) = f_i is the periodic convolution f = b * c with
 *  b = [1/6, 2/3, 1/6] per axis. Its per-axis DFT symbol is
 *
 *    Bhat_d(m) = 2/3 + (1/3) cos(2 pi m / N_d),   min = 1/3 (Nyquist, even N),
 *
 *  so chat = fhat / (Bhat_1 Bhat_2 Bhat_3) is well posed (condition <= 27).
 *
 *  GPU (prefilter_periodic_tricubic_bspline): cuFFT D2Z of the samples into the
 *  half-spectrum (x-fastest layout gx + hx*(gy + ny*gz), hx = nx/2 + 1, plans
 *  cufftPlan3d(nz, ny, nx, ...) as in PeriodicGaussianField), one kernel that
 *  multiplies every entry by 1 / (Bhat_1(m1) Bhat_2(m2) Bhat_3(m3) N1 N2 N3)
 *  (cuFFT is unnormalized; m2, m3 are the signed wrapped frequencies; Bhat is
 *  even so the sign is immaterial), then cuFFT Z2D into the coefficient buffer.
 *
 *  CPU mirror (prefilter_periodic_tricubic_bspline_host): the same
 *  deconvolution solved exactly in double as a cyclic (periodic) tridiagonal
 *  system c_{i-1} + 4 c_i + c_{i+1} = 6 f_i along every axis line (Thomas
 *  algorithm + Sherman-Morrison correction for the two corner entries). It
 *  agrees with the GPU coefficients to roundoff (gated at 1e-13 relative).
 *
 * ---------------------------------------------------------------------------
 * 4) Evaluation (evaluate_point; identical code on host and device)
 * ---------------------------------------------------------------------------
 *
 *  Per axis, in THIS order (do not reorder; decision D-1):
 *
 *    xr = x - L floor(x / L)        canonical reduction FIRST
 *    if xr >= L: xr -= L            rounding guards only
 *    if xr <  0: xr += L
 *    t  = xr / h - 1/2,  i0 = floor(t)   (i0 in [-1, N-1])
 *    u  = t - i0                          (u in [0, 1))
 *
 *  Weights on [0,1) for the coefficients i0-1, i0, i0+1, i0+2:
 *
 *    w_{-1} = (1-u)^3/6,  w_0 = (3u^3 - 6u^2 + 4)/6,
 *    w_1 = (-3u^3 + 3u^2 + 3u + 1)/6,  w_2 = u^3/6;
 *
 *  derivative weights (multiplied by 1/h):
 *
 *    w'_{-1} = -(1-u)^2/2,  w'_0 = (3u^2 - 4u)/2,
 *    w'_1 = (-3u^2 + 2u + 1)/2,  w'_2 = u^2/2.
 *
 *  Indices i0 + k (k = -1..2) lie in [-2, N+1] and are wrapped exactly in
 *  integer arithmetic (one conditional add/subtract of N). The 4 wrapped
 *  indices, 4 weights and 4 derivative weights per axis are computed once; the
 *  sum performs 64 coefficient loads (x innermost) and returns the value and
 *  the three gradient components.
 *
 *  Why reduction-first (D-1): "s(x) and s(x + q L e_d) bitwise identical" cannot
 *  hold for every double x because x + qL is itself rounded. Reducing to
 *  [0, L) before anything else guarantees bitwise identity whenever x + qL is
 *  exactly representable (e.g. dyadic x with L = 1): the reduced coordinate is
 *  then the same double, hence the same (i0, u) and the same arithmetic. The
 *  guarantee is per platform (host vs device may differ by FMA contraction;
 *  CPU/GPU agreement is a separate normwise 1e-13 contract).
 *
 *  The CPU mirror calls the very same inline evaluate_point, so the CPU and
 *  GPU evaluation order is identical by construction.
 *
 * ---------------------------------------------------------------------------
 * 5) Batched GPU evaluation layout
 * ---------------------------------------------------------------------------
 *
 *  Structure of arrays in and out: points px[p], py[p], pz[p]; outputs
 *  value[p], gx[p], gy[p], gz[p]; all spans have the same length. The kernel
 *  allocates nothing, performs no host synchronization and is ordered on
 *  ctx.cuda_stream(); the caller synchronizes when it needs the results.
 *
 * ---------------------------------------------------------------------------
 * 6) Memory accounting and plan lifetime (D-6)
 * ---------------------------------------------------------------------------
 *
 *  The workspace owns the coefficients (N1 N2 N3 doubles) and the
 *  half-spectrum ((N1/2 + 1) N2 N3 cufftDoubleComplex); both are grown on
 *  demand and never shrunk (DeviceBuffer::resize). The report gives exact bytes
 *  of their capacities plus the sum of cufftGetSize of the D2Z and Z2D plans
 *  created in that call (transient, but part of the real peak; no fudge
 *  factors). Both plans are created, bound to ctx.cuda_stream() and destroyed
 *  inside every prefilter call (SF-18 pattern, deliberate v1 simplification:
 *  the prefilter runs once per field and is not a hot path; it synchronizes
 *  the stream once before destroying its plans). Evaluation allocates nothing,
 *  so cudaMemGetInfo is unchanged across evaluations.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"

#include <cmath>
#include <cstddef>
#include <cufft.h>

namespace macroflow3d {
namespace interpolation {

/**
 * @brief Non-owning POD view of a prefiltered coefficient lattice.
 *
 * Usable by value in any kernel and on the host. `coeff` points to device
 * memory (workspace.view()) or host memory (make_host_view()); it must hold
 * nx*ny*nz coefficients in the layout i + nx*(j + ny*k).
 */
struct PeriodicTricubicBSplineView {
    const real* coeff = nullptr;
    int nx = 0, ny = 0, nz = 0;
    real hx = 0.0, hy = 0.0, hz = 0.0; ///< Spacings
    real Lx = 0.0, Ly = 0.0, Lz = 0.0; ///< Periods (N * h)
};

namespace detail {

/// Per-axis stencil: 4 exactly wrapped indices, 4 weights, 4 derivative
/// weights (already divided by h). See file header section 4.
struct AxisStencil {
    int idx[4];
    real w[4];
    real dw[4];
};

__host__ __device__ inline void axis_stencil(real x, int n, real h, real L, AxisStencil& s) {
    // Canonical reduction FIRST (D-1); the two guards absorb rounding only.
    real xr = x - L * floor(x / L);
    if (xr >= L)
        xr -= L;
    if (xr < static_cast<real>(0.0))
        xr += L;

    const real t = xr / h - static_cast<real>(0.5);
    const real ft = floor(t);
    const int i0 = static_cast<int>(ft); // in [-1, N-1]
    const real u = t - ft;               // in [0, 1)
    const real v = static_cast<real>(1.0) - u;
    const real u2 = u * u;
    const real u3 = u2 * u;
    const real sixth = static_cast<real>(1.0) / static_cast<real>(6.0);
    const real inv_h = static_cast<real>(1.0) / h;

    s.w[0] = v * v * v * sixth;
    s.w[1] = (static_cast<real>(3.0) * u3 - static_cast<real>(6.0) * u2 + static_cast<real>(4.0)) *
             sixth;
    s.w[2] = (static_cast<real>(-3.0) * u3 + static_cast<real>(3.0) * u2 +
              static_cast<real>(3.0) * u + static_cast<real>(1.0)) *
             sixth;
    s.w[3] = u3 * sixth;

    s.dw[0] = static_cast<real>(-0.5) * v * v * inv_h;
    s.dw[1] = static_cast<real>(0.5) * (static_cast<real>(3.0) * u2 - static_cast<real>(4.0) * u) *
              inv_h;
    s.dw[2] = static_cast<real>(0.5) *
              (static_cast<real>(-3.0) * u2 + static_cast<real>(2.0) * u + static_cast<real>(1.0)) *
              inv_h;
    s.dw[3] = static_cast<real>(0.5) * u2 * inv_h;

#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int j = i0 - 1 + k; // in [-2, N+1]
        if (j < 0)
            j += n;
        if (j >= n)
            j -= n;
        s.idx[k] = j;
    }
}

} // namespace detail

/**
 * @brief Value and gradient of the periodic tricubic B-spline at (x, y, z).
 *
 * Any finite real coordinates are accepted (reduced to the canonical period
 * first). Host and device callable; allocates nothing. This is the single
 * evaluation routine used by the GPU kernel and by the CPU mirror.
 */
__host__ __device__ inline void evaluate_point(const PeriodicTricubicBSplineView& v, real x,
                                               real y, real z, real& value, real& gx, real& gy,
                                               real& gz) {
    detail::AxisStencil sx, sy, sz;
    detail::axis_stencil(x, v.nx, v.hx, v.Lx, sx);
    detail::axis_stencil(y, v.ny, v.hy, v.Ly, sy);
    detail::axis_stencil(z, v.nz, v.hz, v.Lz, sz);

    const size_t nx = static_cast<size_t>(v.nx);
    const size_t nxy = nx * static_cast<size_t>(v.ny);

    real f = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const size_t off_z = static_cast<size_t>(sz.idx[c]) * nxy;
        real f_c = 0.0, fx_c = 0.0, fy_c = 0.0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const real* row = v.coeff + off_z + static_cast<size_t>(sy.idx[b]) * nx;
            real r = 0.0, rx = 0.0;
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const real cv = row[sx.idx[a]];
                r += sx.w[a] * cv;
                rx += sx.dw[a] * cv;
            }
            f_c += sy.w[b] * r;
            fx_c += sy.w[b] * rx;
            fy_c += sy.dw[b] * r;
        }
        f += sz.w[c] * f_c;
        fx += sz.w[c] * fx_c;
        fy += sz.w[c] * fy_c;
        fz += sz.dw[c] * f_c;
    }
    value = f;
    gx = fx;
    gy = fy;
    gz = fz;
}

/**
 * @brief Exact-byte report of one GPU prefilter call (no fudge factors).
 */
struct PeriodicTricubicBSplineReport {
    size_t coefficient_bytes = 0;     ///< Capacity bytes of workspace.coefficients
    size_t spectrum_bytes = 0;        ///< Capacity bytes of workspace.spectrum
    size_t cufft_work_area_bytes = 0; ///< Sum of cufftGetSize of this call's D2Z and Z2D plans
    size_t total_device_bytes = 0;    ///< Sum of the three fields above
};

/**
 * @brief Owned device workspace: coefficients + half-spectrum. Grown, never
 *        shrunk. Not copyable; movable.
 */
struct PeriodicTricubicBSplineWorkspace {
    DeviceBuffer<real> coefficients;           ///< nx*ny*nz, layout i + nx*(j + ny*k)
    DeviceBuffer<cufftDoubleComplex> spectrum; ///< (nx/2+1)*ny*nz, layout gx + hx*(gy + ny*gz)
    Grid3D grid;                               ///< Grid of the last successful prefilter

    PeriodicTricubicBSplineWorkspace() = default;

    /// Device view of the last prefiltered coefficients. Throws
    /// std::logic_error if no prefilter has completed on this workspace.
    PeriodicTricubicBSplineView view() const;
};

/**
 * @brief GPU prefilter: device samples -> workspace.coefficients.
 *
 * @param samples  Device samples, size grid.num_cells() (std::invalid_argument
 *                 otherwise); not modified.
 * Stream-ordered on ctx.cuda_stream(); synchronizes the stream once before
 * destroying its per-call cuFFT plans (D-6). Throws std::invalid_argument for
 * an invalid grid or size, std::runtime_error for CUDA/cuFFT failures.
 */
PeriodicTricubicBSplineReport
prefilter_periodic_tricubic_bspline(const CudaContext& ctx, const Grid3D& grid,
                                    DeviceSpan<const real> samples,
                                    PeriodicTricubicBSplineWorkspace& workspace);

/**
 * @brief Batched GPU evaluation (SoA in, SoA out); always writes value and
 *        gradient. All seven spans must have the same length
 *        (std::invalid_argument otherwise). Allocates nothing; no host
 *        synchronization; ordered on ctx.cuda_stream().
 */
void evaluate_periodic_tricubic_bspline(const CudaContext& ctx,
                                        const PeriodicTricubicBSplineView& view,
                                        DeviceSpan<const real> px, DeviceSpan<const real> py,
                                        DeviceSpan<const real> pz, DeviceSpan<real> value,
                                        DeviceSpan<real> gx, DeviceSpan<real> gy,
                                        DeviceSpan<real> gz);

/**
 * @brief CPU mirror prefilter: exact cyclic tridiagonal deconvolution per axis
 *        line in double. `samples` and `coefficients` are host arrays of
 *        grid.num_cells() entries (same layout); they may alias.
 */
void prefilter_periodic_tricubic_bspline_host(const Grid3D& grid, const real* samples,
                                              real* coefficients);

/**
 * @brief View over host-resident coefficients, for use with evaluate_point on
 *        the host. Validates the grid (same rules as the prefilter).
 */
PeriodicTricubicBSplineView make_host_view(const Grid3D& grid, const real* host_coefficients);

} // namespace interpolation
} // namespace macroflow3d
