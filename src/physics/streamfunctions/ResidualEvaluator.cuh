#pragma once

/**
 * @file ResidualEvaluator.cuh
 * @brief Coupled Lester equation (14) streamfunction residual and its
 *        dimensionless reductions.
 *
 * This module assembles, from an already-accepted set of production
 * primitives, the coupled residual pair
 *
 *   F1 = A u1 - P( rhs_affine1 - eta * q .* S2 )
 *   F2 = A u2 - P( rhs_affine2 - eta * q .* S1 )      (pairing: F1<->S2, F2<->S1)
 *
 * where:
 *   - `A u = -div_h(q grad_h u)` is `operators::LesterPositiveDiffusionOperator`,
 *     applied to the periodic fluctuations `u1`, `u2` (never to the affine
 *     total field, which is not itself a periodic input).
 *   - `rhs_affine_i` is the mean-zero-projected periodic affine RHS pair from
 *     `streamfunctions::assemble_affine_periodic_rhs`. That call already
 *     projects `rhs1`, `rhs2` in place; the diagnostics it returns (raw and
 *     projected means) are surfaced unchanged in the report.
 *   - `P` is `constraints::MeanZeroProjector` applied to the *combined* RHS
 *     `G_i = rhs_affine_i - eta*(q.*S_pair)`. Because `rhs_affine_i` is
 *     already projected, re-projecting `G_i` is idempotent on that part and
 *     only actually projects the (generally non-zero-mean) source
 *     contribution `eta*(q.*S_pair)`. This is a recorded increment decision:
 *     the residual compares `A u_i` (which is discretely mean-zero for the
 *     periodic operator `A`, since summing `-div_h(q grad_h u)` over a
 *     periodic grid telescopes to zero) against a RHS that is also forced to
 *     mean zero, so `F_i` itself is discretely mean zero by construction.
 *   - `S_i` are `streamfunctions::enqueue_streamfunction_nonlinear_sources`,
 *     fed by `enqueue_total_streamfunction_gradients` and
 *     `enqueue_streamfunction_hessian_vector_b`, all evaluated from the same
 *     fluctuation/gauge state.
 *
 * Dimensionless normalization, with `L_ref = cbrt(Lx*Ly*Lz)` from the grid
 * and `q_rms = RMS(q)`:
 *
 *   r1 = RMS(F1) * L_ref / (q_rms * v_rms)
 *   r2 = RMS(F2) * L_ref / q_rms
 *   r_F = sqrt((r1^2 + r2^2) / 2)
 *
 * `v_rms` is the same reference RMS Darcy speed supplied to the nonlinear
 * source kernel via `NonlinearSourceConfig::v_rms`; it is not recomputed
 * here.
 *
 * No new PDE discretization is introduced anywhere in this file: every
 * differential or linear-algebra primitive is reused exactly as accepted in
 * increments SF-02..SF-09.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../numerics/blas/reduction_workspace.cuh"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../../runtime/CudaContext.cuh"
#include "DifferentialOperators.cuh"
#include "NonlinearSources.cuh"
#include "affine_gauge.cuh"
#include "affine_periodic_rhs.cuh"

#include <cstddef>
#include <type_traits>

namespace macroflow3d {
namespace streamfunctions {

// Number of log-spaced |c| histogram bins, where c = grad(psi1) x grad(psi2).
inline constexpr int kResidualHistogramBins = 512;

/**
 * Host-provided configuration for the |c| histogram bin range.
 *
 * The bin range is `[c_min_rel * v_rms, c_max_rel * v_rms]`, where `v_rms`
 * is the same reference speed used by `NonlinearSourceConfig`. Both bounds
 * must be finite and `0 < c_min_rel < c_max_rel`.
 */
struct ResidualHistogramConfig {
    real c_min_rel{real{1e-10}};
    real c_max_rel{real{1e4}};
};

static_assert(std::is_trivially_copyable<ResidualHistogramConfig>::value,
              "ResidualHistogramConfig must be safe to copy into a kernel argument");

// Forward declarations needed by StreamfunctionResidualWorkspace's friend
// function declarations below.
struct StreamfunctionResidualReport;

/**
 * Reusable, exact-size storage for one coupled residual evaluation.
 *
 * `prepare(n)` allocates or resizes, exactly once for a given `n`:
 *   - the six total-gradient fields (`enqueue_total_streamfunction_gradients`),
 *   - the six Hessian-vector-product scratch fields and three `B` fields
 *     required by `enqueue_streamfunction_hessian_vector_b`'s output contract
 *     (the six HVP scratch fields are intermediate register-analog storage;
 *     only `B` feeds the sources),
 *   - the two nonlinear source fields `S1`, `S2`,
 *   - the two `A`-apply outputs `A u1`, `A u2`,
 *   - the two combined-RHS/`G` fields (the affine RHS buffers are reused
 *     in place as the `G` buffers once combined with `-eta*q.*S_pair`),
 *   - the SF-09 nonlinear-source diagnostic counter buffer, sized for the
 *     worst case `2 + kMaxDegeneracyThresholds` so any per-call
 *     `NonlinearSourceConfig::num_degeneracy_thresholds` fits without
 *     reallocation,
 *   - the affine-RHS assembly workspace and a private mean-zero workspace
 *     used to project the combined `G_i` fields,
 *   - BLAS reduction workspaces for the RMS reductions,
 *   - five device scalars for the reductions (`RMS F1`, `RMS F2`, `Linf F1`,
 *     `Linf F2`, `q_rms`, all as raw L2 norms; the RMS/`q_rms` division by
 *     `sqrt(n)` happens on the host at report time), and
 *   - the |c| histogram buffer (`kResidualHistogramBins` + underflow +
 *     overflow counters, `unsigned long long`).
 *
 * After `prepare`, evaluation performs no allocation and no host
 * synchronization.
 */
class StreamfunctionResidualWorkspace {
  public:
    StreamfunctionResidualWorkspace() = default;
    ~StreamfunctionResidualWorkspace() = default;

    StreamfunctionResidualWorkspace(const StreamfunctionResidualWorkspace&) = delete;
    StreamfunctionResidualWorkspace& operator=(const StreamfunctionResidualWorkspace&) = delete;
    StreamfunctionResidualWorkspace(StreamfunctionResidualWorkspace&&) noexcept = default;
    StreamfunctionResidualWorkspace& operator=(StreamfunctionResidualWorkspace&&) noexcept =
        default;

    void prepare(std::size_t n);
    [[nodiscard]] bool prepared_for(std::size_t n) const noexcept;

  private:
    std::size_t n_ = 0;

    // Total gradients (six components).
    DeviceBuffer<real> psi1_x_, psi1_y_, psi1_z_, psi2_x_, psi2_y_, psi2_z_;

    // Hessian-vector-product scratch (discarded) and the B field it feeds.
    DeviceBuffer<real> h2g1_x_, h2g1_y_, h2g1_z_, h1g2_x_, h1g2_y_, h1g2_z_;
    DeviceBuffer<real> b_x_, b_y_, b_z_;

    // Nonlinear sources.
    DeviceBuffer<real> s1_, s2_;
    DeviceBuffer<unsigned long long> source_counters_;

    // A-apply outputs.
    DeviceBuffer<real> a_u1_, a_u2_;

    // Affine RHS / combined-G buffers (reused in place).
    DeviceBuffer<real> g1_, g2_;

    // Sub-workspaces.
    AffinePeriodicRhsWorkspace affine_rhs_workspace_;
    constraints::MeanZeroWorkspace g_projection_workspace_;
    blas::ReductionWorkspace reduction_workspace_;

    // Reduction scalars: [RMS F1 (L2), RMS F2 (L2), Linf F1, Linf F2, q_rms (L2)].
    DeviceBuffer<real> scalars_;

    // |c| histogram: [0 .. kResidualHistogramBins) bins, then underflow, overflow.
    DeviceBuffer<unsigned long long> histogram_;

    // Diagnostics captured by the most recent enqueue call.
    AffineRhsDeviceDiagnostics last_rhs_diagnostics_{};
    bool has_last_results_ = false;

    friend void enqueue_streamfunction_residual(
        CudaContext&, const Grid3D&, DeviceSpan<const real>,
        const PeriodicStreamfunctionFluctuations&, const AffineGauge&, real,
        const NonlinearSourceConfig&, const ResidualHistogramConfig&, DeviceSpan<real>,
        DeviceSpan<real>, StreamfunctionResidualWorkspace&);
    friend StreamfunctionResidualReport synchronize_streamfunction_residual_report(
        CudaContext&, const Grid3D&, real, const NonlinearSourceConfig&,
        const ResidualHistogramConfig&, StreamfunctionResidualWorkspace&);
};

/**
 * Read-only device-side view of everything produced by one
 * `enqueue_streamfunction_residual` call, valid until the workspace is
 * reused by a subsequent enqueue.
 */
struct StreamfunctionResidualDeviceResults {
    DeviceSpan<const real> f1;
    DeviceSpan<const real> f2;
    // [RMS F1 (L2 norm), RMS F2 (L2 norm), Linf F1, Linf F2, q_rms (L2 norm)].
    DeviceSpan<const real> scalars;
    // [bins..., underflow, overflow].
    DeviceSpan<const unsigned long long> histogram;
    DeviceSpan<const unsigned long long> nonlinear_source_counters;
    AffineRhsDeviceDiagnostics rhs_mean_diagnostics;
};

/**
 * Host-assembled report for one coupled residual evaluation.
 *
 * `rms_f1`, `rms_f2`, `q_rms` are true RMS values (raw device L2 norms
 * divided by `sqrt(n)` on the host). `linf_f1`, `linf_f2` are true Linf
 * (max-abs) values. `raw_rhs_mean_*` / `projected_rhs_mean_*` are the
 * SF-06 affine-RHS compatibility diagnostics, in psi1, psi2 order.
 */
struct StreamfunctionResidualReport {
    real rms_f1{};
    real rms_f2{};
    real linf_f1{};
    real linf_f2{};
    real q_rms{};

    real r1{};
    real r2{};
    real r_F{};

    real raw_rhs_mean_psi1{};
    real raw_rhs_mean_psi2{};
    real projected_rhs_mean_psi1{};
    real projected_rhs_mean_psi2{};

    unsigned long long nonfinite_s1{};
    unsigned long long nonfinite_s2{};
    int num_degeneracy_thresholds{};
    unsigned long long degeneracy_counts[kMaxDegeneracyThresholds]{};

    unsigned long long histogram_counts[kResidualHistogramBins]{};
    unsigned long long histogram_underflow{};
    unsigned long long histogram_overflow{};
    real histogram_c_min{};
    real histogram_c_max{};

    real eta{};
    real epsilon{};
    real v_rms{};
    real L_ref{};
    std::size_t n{};
};

/**
 * Enqueue the full evaluation chain on `ctx.cuda_stream()`:
 *
 *   affine RHS -> total gradients -> Hessian-vector B -> nonlinear sources
 *   -> A.apply(u1), A.apply(u2) -> G_i = rhs_i - eta*(q.*S_pair)
 *   -> project(G_i) -> F_i = A u_i - G_i
 *   -> reductions (RMS/Linf F1, F2; q_rms) -> |c| histogram.
 *
 * `f1`, `f2` are caller-owned outputs, non-null, of exactly `grid.num_cells()`
 * elements, and must not overlap each other, `q`, or `fluctuations.u1`/`u2`.
 *
 * `q` must contain finite, strictly positive values; this precondition is
 * intentionally not reduced or synchronized here (SF-06 wording). `eta` must
 * be finite and `>= 0`. `histogram_config` must have finite
 * `0 < c_min_rel < c_max_rel`. `source_config` is validated by
 * `enqueue_streamfunction_nonlinear_sources` itself; its rules are not
 * duplicated here. `workspace` must be `prepared_for(grid.num_cells())`.
 *
 * This function performs no allocation and no host synchronization; the only
 * CUDA runtime calls it or its callees issue outside kernel launches are
 * `cudaMemsetAsync` (zeroing the SF-09 counters, the |c| histogram, and the
 * two Linf accumulator scalars) and `cudaMemcpyAsync` (the affine-RHS mean
 * diagnostics captured by `assemble_affine_periodic_rhs`), all enqueued on
 * `ctx.cuda_stream()`.
 */
void enqueue_streamfunction_residual(
    CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> q,
    const PeriodicStreamfunctionFluctuations& fluctuations, const AffineGauge& gauge, real eta,
    const NonlinearSourceConfig& source_config, const ResidualHistogramConfig& histogram_config,
    DeviceSpan<real> f1, DeviceSpan<real> f2, StreamfunctionResidualWorkspace& workspace);

/**
 * Explicit synchronization and host report assembly for the most recent
 * `enqueue_streamfunction_residual` call on `workspace`.
 *
 * `eta`, `source_config`, and `histogram_config` must match the values
 * supplied to that enqueue call; they are not re-validated against device
 * state and are only used to echo host-visible parameters (`eta`, `epsilon`,
 * `v_rms`) and to recompute the exact `c_min`/`c_max` histogram bounds for
 * the report. `grid` supplies `L_ref = cbrt(Lx*Ly*Lz)` and `n`.
 *
 * This performs exactly one `ctx.synchronize()`, after enqueueing every
 * required device-to-host copy.
 */
[[nodiscard]] StreamfunctionResidualReport synchronize_streamfunction_residual_report(
    CudaContext& ctx, const Grid3D& grid, real eta, const NonlinearSourceConfig& source_config,
    const ResidualHistogramConfig& histogram_config, StreamfunctionResidualWorkspace& workspace);

/**
 * Host helper: the upper edge of the log-spaced |c| histogram bin where the
 * cumulative fraction of samples first reaches `p`, for `p` in `(0, 1]`.
 *
 * Error bound: because bins are log-spaced with `kResidualHistogramBins`
 * bins spanning `[report.histogram_c_min, report.histogram_c_max]`, the
 * returned edge has relative value error bounded by one bin-width factor
 * `10^((log10(c_max) - log10(c_min)) / kResidualHistogramBins)`.
 *
 * If `p` is reached entirely within the underflow bucket, returns
 * `report.histogram_c_min`. If `p` is only reached once the (unbounded)
 * overflow bucket is included, returns `+infinity`.
 */
[[nodiscard]] real residual_histogram_percentile(const StreamfunctionResidualReport& report,
                                                  real p);

} // namespace streamfunctions
} // namespace macroflow3d
