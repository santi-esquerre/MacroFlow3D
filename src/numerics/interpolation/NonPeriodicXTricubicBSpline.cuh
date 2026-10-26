#pragma once

/**
 * @file NonPeriodicXTricubicBSpline.cuh
 * @brief C^2 cubic B-spline interpolation of cell-centred double fields that are
 *        NON-periodic in x1 and periodic in x2, x3: value, analytic gradient and
 *        Hessian, GPU with CPU mirror.
 * @ingroup numerics
 *
 * SF-41 (medium (b) for the Lagrangian consumer), contract C-SPLINE. Sibling of
 * the SF-28 periodic spline (`PeriodicTricubicBSpline.cuh`, unchanged; its
 * `detail::AxisStencil{,2}` / `axis_stencil{,2}` are reused for the periodic y/z
 * axes, so those axes are bit-for-bit the SF-28 code). New types, no template
 * parameter added to the periodic ones.
 *
 * ---------------------------------------------------------------------------
 * 1) Conventions
 * ---------------------------------------------------------------------------
 *
 *  - Samples: cell-centred, `real = double`, project layout
 *    `Grid3D::idx(i,j,k) = i + nx*(j + ny*k)`, x_i = (i + 1/2) h_x (the same
 *    lattice as SF-28), L1 = Lx = nx h_x. Size rule: nx >= 8 (the reduced x
 *    system needs >= 4 interior unknowns), ny, nz >= 4, spacings finite > 0;
 *    violations throw std::invalid_argument.
 *  - Interpolant: s(x) = sum c_{a b c} beta3((x1 - x_a)/h_x) beta3(...) beta3(...),
 *    beta3 the centred uniform cubic B-spline (SF-28 weights, derivative
 *    weights and index computation, same expressions).
 *  - x1 coefficients live on a non-cyclic index range a = -2..nx+1, stored with
 *    offset 2. COEFFICIENT LAYOUT (device and host): `(nx+4)` entries per x line,
 *    `coeff[(a + 2) + (nx+4)*(j + ny*k)]`, j, k periodic indices of y and z.
 *  - x1 closure: NOT-A-KNOT on the raw samples, NO ghost samples. Unknowns
 *    c_{-1}..c_{nx} (nx+2); rows: the nx interpolation rows
 *    (c_{i-1} + 4 c_i + c_{i+1})/6 = f_i, and the two third-derivative
 *    continuity rows at the knots x_1 and x_{nx-2}
 *        -c_{-1} + 4 c_0 - 6 c_1 + 4 c_2 - c_3 = 0   (and its mirror at the end).
 *    Two continuation coefficients extend the end cubic pieces so the
 *    evaluation is defined up to a half cell outside the sample lattice:
 *        c_{-2}   = 4 c_{-1} - 6 c_0 + 4 c_1 - c_2,
 *        c_{nx+1} = 4 c_{nx} - 6 c_{nx-1} + 4 c_{nx-2} - c_{nx-3}.
 *    Reproduces cubic polynomials in x1 (constant in x2, x3); accuracy
 *    O(h^4) / O(h^3) / O(h^2) for value / gradient / Hessian up to the faces.
 *  - y, z: periodic exactly as SF-28 (symbol 2/3 + cos(2 pi m/n)/3).
 *
 * ---------------------------------------------------------------------------
 * 2) Evaluation domain (evaluate_point, evaluate_point_hessian)
 * ---------------------------------------------------------------------------
 *
 *  The x1 domain is the half-open interval [-h_x/2, L1 + h_x/2). It is decided
 *  in floating point by t = x/h_x - 1/2 (the same t the stencil uses):
 *  inside iff t >= -1 and t < nx (so NaN and +-inf are outside). Then
 *  i0 = floor(t) is in [-1, nx-1] and all four x indices (i0-1..i0+2)+2 lie in
 *  [0, nx+3]: the test is exactly the condition for staying inside the
 *  coefficient array. y or z non-finite is also reported as outside. When
 *  outside the functions return false, set all outputs to NaN and READ NO
 *  COEFFICIENT (v.coeff is not dereferenced; it may even be null).
 *  y, z accept any finite coordinate (canonical reduction first, SF-28 D-1).
 *  Inside, the functions return true.
 *
 *  evaluate_point_hessian returns value and gradient BITWISE equal (memcmp) to
 *  evaluate_point on the same platform: same stencil call, same loop nesting
 *  and accumulation order; the Hessian accumulators run alongside (as SF-28
 *  evaluate_point_hessian). Both are `__host__ __device__ inline` (no
 *  allocation) so translation units that instantiate them for the tracker can
 *  be compiled with --fmad=false.
 *
 * ---------------------------------------------------------------------------
 * 3) Prefilter
 * ---------------------------------------------------------------------------
 *
 *  Separable, applied in this ORDER: (1) x line solve on the raw samples (one
 *  per (j,k) line) producing the (nx+4)-entry lines c_{-2}..c_{nx+1}; (2) the
 *  SF-28 deconvolution in y and z applied to every one of the nx+4 coefficient
 *  planes (including the two continuation planes: the filter is linear and
 *  commutes with the x solve, so the continuation relations survive).
 *
 *  x solve ("reduced" variant, deterministic, no pivoting). The two
 *  not-a-knot rows eliminate algebraically (using rows 0..2 / the mirror) to
 *      c_1 = (8 f_1 - f_0 - f_2)/6,   c_{nx-2} = (8 f_{nx-2} - f_{nx-1} - f_{nx-3})/6,
 *  after which c_2..c_{nx-3} solve the plain diagonally dominant system
 *  [1 4 1] c = 6 f (right-hand side corrected by the two known neighbours) by
 *  the Thomas algorithm with the constant pivots d_0 = 4, d_i = 4 - 1/d_{i-1}
 *  (stored once as reciprocals); c_0, c_{-1}, c_{nx-1}, c_{nx} follow by
 *  back-substitution of the interpolation rows (c_0 = 6 f_1 - 4 c_1 - c_2,
 *  c_{-1} = 6 f_0 - 4 c_0 - c_1, and mirrored), then the two continuation
 *  coefficients. The line solve is the SAME inline code on host and device.
 *  Round-off amplification of the end rows is inherent to the closure (white
 *  noise gain about 33 at c_{-2}, 8 at c_{-1}, 1.4-1.9 elsewhere).
 *
 *  GPU (prefilter_nonperiodic_x_tricubic_bspline): reciprocal-pivot table
 *  kernel; x solve kernel (one thread per (j,k) line); batched 2-D cuFFT D2Z
 *  over the (y,z) planes of all nx+4 x indices (cufftPlanMany, rank 2, n =
 *  {nz, ny}, stride nx+4 between consecutive plane entries, distance 1
 *  between batch members, i.e. the x-fastest extended layout is used in place,
 *  no transpose); one kernel multiplying every entry by
 *  1/(Bhat(my, ny) Bhat(mz, nz) ny nz); Z2D back into the coefficient buffer
 *  (the Z2D destroys the spectrum, as in SF-28).
 *
 *  CPU mirror (prefilter_nonperiodic_x_tricubic_bspline_host): the same x
 *  solve, then cyclic tridiagonal line solves in y and z (Thomas +
 *  Sherman-Morrison, the SF-28 algorithm copied). Agrees with the GPU to
 *  round-off amplified by the end rows (threshold A: 1e-13 relative on the
 *  value).
 *
 * ---------------------------------------------------------------------------
 * 4) Memory accounting and plan lifetime
 * ---------------------------------------------------------------------------
 *
 *  The workspace owns coefficients, half-spectrum and the pivot table; all
 *  are grown on demand and never shrunk, so a second prefilter of the same
 *  (or smaller) size allocates nothing in the workspace. Exact bytes of
 *  capacities (`NonPeriodicXTricubicBSplineReport`):
 *      coefficient_bytes = 8 (nx+4) ny nz
 *      spectrum_bytes    = 16 (nx+4) (ny/2+1) nz
 *      line_table_bytes  = 8 (nx-4)
 *      cufft_work_area_bytes = cufftGetSize(D2Z) + cufftGetSize(Z2D)  (transient)
 *      total_device_bytes = sum of the four.
 *  The two cuFFT plans are created, bound to ctx.cuda_stream() and destroyed
 *  inside every prefilter call (SF-28 D-6 pattern); the call synchronizes the
 *  stream once before destroying them. Evaluation (single point and batched)
 *  allocates nothing and does not synchronize.
 */

#include "PeriodicTricubicBSpline.cuh"

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
 * @brief Non-owning POD view of a prefiltered, x-extended coefficient lattice.
 *
 * `coeff` points to device or host memory holding (nx+4)*ny*nz coefficients,
 * `coeff[(a + 2) + (nx+4)*(j + ny*k)]` for a = -2..nx+1. nx, ny, nz are the
 * SAMPLE extents; Lx = nx hx is L1.
 */
struct NonPeriodicXTricubicBSplineView {
    const real* coeff = nullptr;
    int nx = 0, ny = 0, nz = 0;
    real hx = 0.0, hy = 0.0, hz = 0.0; ///< Spacings
    real Lx = 0.0, Ly = 0.0, Lz = 0.0; ///< Lx = nx hx (the medium length L1); Ly, Lz periods
};

namespace detail {

/// Non-periodic x stencil: false (and nothing computed) when x is outside the
/// evaluation domain (section 2); otherwise idx[k] in [0, n+3] are indices of
/// the x-extended line (coefficient i0-1+k, offset 2), w/dw as SF-28
/// (identical expressions), u the fractional coordinate in [0,1).
__host__ __device__ inline bool axis_stencil_x(real x, int n, real h, AxisStencil& s, real& u_out) {
    const real t = x / h - static_cast<real>(0.5);
    if (!(t >= static_cast<real>(-1.0) && t < static_cast<real>(n)))
        return false; // also NaN / +-inf
    const real ft = floor(t);
    const int i0 = static_cast<int>(ft); // in [-1, n-1]
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
    for (int k = 0; k < 4; ++k)
        s.idx[k] = i0 + 1 + k; // (i0 - 1 + k) + 2
    u_out = u;
    return true;
}

/// axis_stencil_x plus second-derivative weights (divided by h^2); w, dw, idx
/// bitwise identical to axis_stencil_x.
__host__ __device__ inline bool axis_stencil_x2(real x, int n, real h, AxisStencil2& s) {
    AxisStencil base;
    real u;
    if (!axis_stencil_x(x, n, h, base, u))
        return false;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        s.idx[k] = base.idx[k];
        s.w[k] = base.w[k];
        s.dw[k] = base.dw[k];
    }
    const real v = static_cast<real>(1.0) - u;
    const real inv_h = static_cast<real>(1.0) / h;
    const real inv_h2 = inv_h * inv_h;
    s.d2w[0] = v * inv_h2;
    s.d2w[1] = (static_cast<real>(3.0) * u - static_cast<real>(2.0)) * inv_h2;
    s.d2w[2] = (static_cast<real>(1.0) - static_cast<real>(3.0) * u) * inv_h2;
    s.d2w[3] = u * inv_h2;
    return true;
}

__host__ __device__ inline real np_nan() { return nan(""); }

} // namespace detail

/**
 * @brief Value and gradient at (x, y, z); false when outside the evaluation
 *        domain (section 2), in which case the outputs are NaN and no
 *        coefficient is read. Host and device callable; allocates nothing.
 */
__host__ __device__ inline bool evaluate_point(const NonPeriodicXTricubicBSplineView& v, real x,
                                               real y, real z, real& value, real& gx, real& gy,
                                               real& gz) {
    detail::AxisStencil sx, sy, sz;
    real ux;
    if (!detail::axis_stencil_x(x, v.nx, v.hx, sx, ux) || !std::isfinite(y) || !std::isfinite(z)) {
        value = gx = gy = gz = detail::np_nan();
        return false;
    }
    detail::axis_stencil(y, v.ny, v.hy, v.Ly, sy);
    detail::axis_stencil(z, v.nz, v.hz, v.Lz, sz);

    const size_t nxe = static_cast<size_t>(v.nx) + 4;
    const size_t nxy = nxe * static_cast<size_t>(v.ny);

    real f = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const size_t off_z = static_cast<size_t>(sz.idx[c]) * nxy;
        real f_c = 0.0, fx_c = 0.0, fy_c = 0.0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const real* row = v.coeff + off_z + static_cast<size_t>(sy.idx[b]) * nxe;
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
    return true;
}

/**
 * @brief Value, gradient and Hessian H = (xx, yy, zz, xy, xz, yz); same domain
 *        rule as evaluate_point. value, gx, gy, gz are bitwise equal to
 *        evaluate_point's on the same platform.
 */
__host__ __device__ inline bool evaluate_point_hessian(const NonPeriodicXTricubicBSplineView& v,
                                                       real x, real y, real z, real& value,
                                                       real& gx, real& gy, real& gz, real H[6]) {
    detail::AxisStencil2 sx, sy, sz;
    if (!detail::axis_stencil_x2(x, v.nx, v.hx, sx) || !std::isfinite(y) || !std::isfinite(z)) {
        value = gx = gy = gz = detail::np_nan();
        for (int i = 0; i < 6; ++i)
            H[i] = detail::np_nan();
        return false;
    }
    detail::axis_stencil2(y, v.ny, v.hy, v.Ly, sy);
    detail::axis_stencil2(z, v.nz, v.hz, v.Lz, sz);

    const size_t nxe = static_cast<size_t>(v.nx) + 4;
    const size_t nxy = nxe * static_cast<size_t>(v.ny);

    real f = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
    real fxx = 0.0, fyy = 0.0, fzz = 0.0, fxy = 0.0, fxz = 0.0, fyz = 0.0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const size_t off_z = static_cast<size_t>(sz.idx[c]) * nxy;
        real f_c = 0.0, fx_c = 0.0, fy_c = 0.0;
        real fxx_c = 0.0, fyy_c = 0.0, fxy_c = 0.0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const real* row = v.coeff + off_z + static_cast<size_t>(sy.idx[b]) * nxe;
            real r = 0.0, rx = 0.0, rxx = 0.0;
#pragma unroll
            for (int a = 0; a < 4; ++a) {
                const real cv = row[sx.idx[a]];
                r += sx.w[a] * cv;
                rx += sx.dw[a] * cv;
                rxx += sx.d2w[a] * cv;
            }
            f_c += sy.w[b] * r;
            fx_c += sy.w[b] * rx;
            fy_c += sy.dw[b] * r;
            fxx_c += sy.w[b] * rxx;
            fxy_c += sy.dw[b] * rx;
            fyy_c += sy.d2w[b] * r;
        }
        f += sz.w[c] * f_c;
        fx += sz.w[c] * fx_c;
        fy += sz.w[c] * fy_c;
        fz += sz.dw[c] * f_c;
        fxx += sz.w[c] * fxx_c;
        fyy += sz.w[c] * fyy_c;
        fxy += sz.w[c] * fxy_c;
        fxz += sz.dw[c] * fx_c;
        fyz += sz.dw[c] * fy_c;
        fzz += sz.d2w[c] * f_c;
    }
    value = f;
    gx = fx;
    gy = fy;
    gz = fz;
    H[0] = fxx;
    H[1] = fyy;
    H[2] = fzz;
    H[3] = fxy;
    H[4] = fxz;
    H[5] = fyz;
    return true;
}

/// Exact-byte report of one GPU prefilter call (no fudge factors); formulas in
/// the file header, section 4.
struct NonPeriodicXTricubicBSplineReport {
    size_t coefficient_bytes = 0;     ///< Capacity bytes of workspace.coefficients
    size_t spectrum_bytes = 0;        ///< Capacity bytes of workspace.spectrum
    size_t line_table_bytes = 0;      ///< Capacity bytes of workspace.line_table
    size_t cufft_work_area_bytes = 0; ///< Sum of cufftGetSize of this call's D2Z and Z2D plans
    size_t total_device_bytes = 0;    ///< Sum of the four fields above
};

/// Owned device workspace. Grown, never shrunk. Not copyable; movable.
struct NonPeriodicXTricubicBSplineWorkspace {
    DeviceBuffer<real> coefficients;           ///< (nx+4)*ny*nz, layout (a+2) + (nx+4)*(j + ny*k)
    DeviceBuffer<cufftDoubleComplex> spectrum; ///< (nx+4)*(ny/2+1)*nz, layout b + (nx+4)*(gy + hy*gz)
    DeviceBuffer<real> line_table;             ///< nx-4 reciprocal Thomas pivots
    Grid3D grid;                               ///< SAMPLE grid of the last successful prefilter

    NonPeriodicXTricubicBSplineWorkspace() = default;

    /// Device view of the last prefiltered coefficients. Throws
    /// std::logic_error if no prefilter has completed on this workspace.
    NonPeriodicXTricubicBSplineView view() const;
};

/**
 * @brief GPU prefilter: device samples (nx*ny*nz, project layout, not
 *        modified) -> workspace.coefficients ((nx+4)*ny*nz, extended layout).
 *
 * Stream-ordered on ctx.cuda_stream(); synchronizes the stream once before
 * destroying its per-call cuFFT plans. Throws std::invalid_argument for an
 * invalid grid/size/null pointer, std::runtime_error for CUDA/cuFFT failures.
 */
NonPeriodicXTricubicBSplineReport
prefilter_nonperiodic_x_tricubic_bspline(const CudaContext& ctx, const Grid3D& grid,
                                         DeviceSpan<const real> samples,
                                         NonPeriodicXTricubicBSplineWorkspace& workspace);

/**
 * @brief CPU mirror prefilter. `samples` has grid.num_cells() entries,
 *        `coefficients` (nx+4)*ny*nz entries (extended layout); they must not
 *        overlap.
 */
void prefilter_nonperiodic_x_tricubic_bspline_host(const Grid3D& grid, const real* samples,
                                                   real* coefficients);

/// View over device or host extended coefficients of the SAMPLE grid `grid`
/// (validates the grid like the prefilter).
NonPeriodicXTricubicBSplineView make_nonperiodic_x_view(const Grid3D& grid, const real* coefficients);

/**
 * @brief Batched GPU evaluation (SoA in/out). `inside[p]` is 1 when point p is
 *        inside the domain (outputs finite) and 0 otherwise (outputs NaN, no
 *        coefficient read). All spans have the same length (std::invalid_argument
 *        otherwise). No allocation, no host synchronization; ordered on
 *        ctx.cuda_stream().
 */
void evaluate_nonperiodic_x_tricubic_bspline(const CudaContext& ctx,
                                             const NonPeriodicXTricubicBSplineView& view,
                                             DeviceSpan<const real> px, DeviceSpan<const real> py,
                                             DeviceSpan<const real> pz, DeviceSpan<real> value,
                                             DeviceSpan<real> gx, DeviceSpan<real> gy,
                                             DeviceSpan<real> gz, DeviceSpan<int> inside);

/// As above through evaluate_point_hessian; `hessian` has 6*n entries,
/// H[c] of point p at hessian[6*p + c] (order xx, yy, zz, xy, xz, yz).
void evaluate_nonperiodic_x_tricubic_bspline_hessian(
    const CudaContext& ctx, const NonPeriodicXTricubicBSplineView& view, DeviceSpan<const real> px,
    DeviceSpan<const real> py, DeviceSpan<const real> pz, DeviceSpan<real> value,
    DeviceSpan<real> gx, DeviceSpan<real> gy, DeviceSpan<real> gz, DeviceSpan<real> hessian,
    DeviceSpan<int> inside);

} // namespace interpolation
} // namespace macroflow3d
