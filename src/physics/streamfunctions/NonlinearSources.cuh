#pragma once

/**
 * @file NonlinearSources.cuh
 * @brief Pointwise nonlinear Lester equation (14) source terms `S1`, `S2`.
 */

#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../runtime/CudaContext.cuh"
#include "DifferentialOperators.cuh"

#include <type_traits>

namespace macroflow3d {
namespace streamfunctions {

/**
 * Read-only view of the `B = H(psi2) grad(psi1) - H(psi1) grad(psi2)` field
 * produced by `enqueue_streamfunction_hessian_vector_b`, at the same state as
 * the total gradients consumed alongside it.
 */
struct StreamfunctionBFieldView {
    DeviceSpan<const real> b_x;
    DeviceSpan<const real> b_y;
    DeviceSpan<const real> b_z;
};

// Upper bound on the number of denominator-degeneracy thresholds diagnosed
// per launch. Kept small and fixed-size so the config stays trivially
// copyable and register-resident inside the kernel.
inline constexpr int kMaxDegeneracyThresholds = 4;

/**
 * Host-provided, per-launch configuration for the nonlinear source kernel.
 *
 * `epsilon` is a dimensionless regularization coefficient applied relative to
 * `v_rms`, the reference RMS Darcy speed for the current state; the
 * regularized denominator is `d = |c|^2 + (epsilon * v_rms)^2` with
 * `c = grad(psi1) x grad(psi2)`. There is no other floor, clamp, or fallback
 * on the denominator anywhere in this kernel.
 *
 * `degeneracy_thresholds[0 .. num_degeneracy_thresholds)` are additional,
 * purely diagnostic dimensionless multipliers `tau_t` used only to count
 * cells where `|c|^2 < (tau_t * v_rms)^2`; they do not affect `S1`, `S2`.
 */
struct NonlinearSourceConfig {
    real epsilon{real{1e-2}};
    real v_rms{};
    int num_degeneracy_thresholds{0};
    real degeneracy_thresholds[kMaxDegeneracyThresholds]{};
};

/**
 * Caller-owned device buffers for the two nonlinear sources.
 */
struct NonlinearSourceOutput {
    DeviceSpan<real> s1;
    DeviceSpan<real> s2;
};

/**
 * Caller-owned device counter buffer of exact size
 * `2 + config.num_degeneracy_thresholds`, laid out as:
 *   [0]     number of cells where the computed S1 is not finite,
 *   [1]     number of cells where the computed S2 is not finite,
 *   [2 + t] number of cells where `|c|^2 < (tau_t * v_rms)^2`, evaluated in
 *           double precision with a strict `<` comparison, for
 *           `tau_t = config.degeneracy_thresholds[t]`.
 *
 * `enqueue_streamfunction_nonlinear_sources` zeroes this buffer with
 * `cudaMemsetAsync` on `ctx.cuda_stream()` before the kernel launch, so
 * callers must not rely on any prior contents surviving the call and must not
 * read the counters until work queued on that stream has completed.
 */
struct NonlinearSourceCounters {
    DeviceSpan<unsigned long long> counters;
};

static_assert(std::is_trivially_copyable<NonlinearSourceConfig>::value,
              "NonlinearSourceConfig must be safe to copy into a kernel argument");
static_assert(std::is_trivially_copyable<StreamfunctionBFieldView>::value,
              "StreamfunctionBFieldView must be safe to copy into a kernel argument");
static_assert(std::is_trivially_copyable<NonlinearSourceOutput>::value,
              "NonlinearSourceOutput must be safe to copy into a kernel argument");
static_assert(std::is_trivially_copyable<NonlinearSourceCounters>::value,
              "NonlinearSourceCounters must be safe to copy into a kernel argument");

/**
 * Enqueue the pointwise Lester equation (14) nonlinear sources `S1`, `S2`.
 *
 * At every cell, using the total gradients `g1 = grad(psi1)`, `g2 =
 * grad(psi2)` and `B = H(psi2) g1 - H(psi1) g2`, this computes entirely in
 * double-precision registers:
 *
 *   c  = g1 x g2                     (right-handed: c_x = g1_y*g2_z - g1_z*g2_y,
 *                                      c_y = g1_z*g2_x - g1_x*g2_z,
 *                                      c_z = g1_x*g2_y - g1_y*g2_x)
 *   d  = |c|^2 + (epsilon * v_rms)^2
 *   S1 = ((B x g1) . c) / d
 *   S2 = ((B x g2) . c) / d
 *
 * This kernel is purely pointwise: it does not evaluate a stencil, does not
 * recompute gradients or Hessians, and does not apply the `q` or `eta`
 * factors from later stages of the equation (14) system. It writes only
 * `output.s1`, `output.s2`, and the diagnostic counters; it does not write
 * `c` or `d` as fields.
 *
 * `config.epsilon` must be finite and `>= 0`; `config.v_rms` must be finite
 * and strictly positive; `config.num_degeneracy_thresholds` must be in
 * `[0, kMaxDegeneracyThresholds]` and every used threshold must be finite and
 * `>= 0`. The regularization term `(epsilon * v_rms)^2` is the only
 * denominator floor applied; there is no hidden additional epsilon.
 *
 * `total_gradients` (six spans) and `b` (three spans) must be non-null device
 * spans of exactly `grid.num_cells()` elements; they are read-only and may
 * overlap one another. `output.s1`, `output.s2`, and `counters.counters` must
 * be non-null, of exact size (`grid.num_cells()` for the sources,
 * `2 + config.num_degeneracy_thresholds` for the counters), and must not
 * overlap each other or any input span (byte ranges are compared explicitly
 * because the counter buffer has a different element type). Grid extents and
 * spacings must be finite and strictly positive, and the grid's cell count
 * must not overflow storage. These preconditions are checked on the host
 * before enqueueing any device work.
 *
 * Diagnostics are accumulated per thread across its grid-stride loop and
 * folded into `counters.counters` with one `atomicAdd` per counter per
 * thread after the loop, not per cell.
 *
 * This function first zeroes `counters.counters` with `cudaMemsetAsync` on
 * `ctx.cuda_stream()`, then launches the kernel on the same stream. Aside
 * from that memset, it performs no allocation, device-to-host transfer, or
 * host synchronization. All spans must remain alive and unchanged until work
 * previously queued on `ctx.cuda_stream()` has completed, and
 * `total_gradients` and `b` must correspond to the same streamfunction state.
 */
void enqueue_streamfunction_nonlinear_sources(
    CudaContext& ctx, const Grid3D& grid, const TotalStreamfunctionGradientView& total_gradients,
    const StreamfunctionBFieldView& b, const NonlinearSourceConfig& config,
    const NonlinearSourceOutput& output, const NonlinearSourceCounters& counters);

} // namespace streamfunctions
} // namespace macroflow3d
