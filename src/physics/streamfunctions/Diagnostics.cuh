#pragma once

/**
 * @file Diagnostics.cuh
 * @brief Physical diagnostics for the reconstructed streamfunction velocity
 *        `v_psi = grad(psi1) x grad(psi2)` on a CompactMAC face layout.
 *
 * This module reconstructs `v_psi` from an already-accepted streamfunction
 * state (periodic fluctuations plus affine gauge) and compares it against a
 * caller-supplied Darcy CompactMAC velocity field, producing the Gate 3A
 * physical acceptance metrics: velocity reconstruction error, per-component
 * correlation, magnitude error, a robust angular error, Darcy invariance,
 * reconstructed-flow divergence, and cross-gradient (`|c|`, `c = grad(psi1) x
 * grad(psi2)`) degeneracy split by Darcy speed.
 *
 * IMPORTANT SCIENTIFIC CAVEAT: the reconstruction implemented here is
 * interpolate-then-cross (SF-11 locked convention below). It is
 * *approximately*, not algebraically, divergence-free. This module reports
 * the reconstructed-flow MAC divergence as a diagnostic and never claims
 * exact discrete divergence freedom.
 *
 * Reconstruction (interpolate-then-cross, LOCKED):
 * `enqueue_total_streamfunction_gradients` (SF-07) first produces the total,
 * cell-centered gradients `g1 = grad(psi1)`, `g2 = grad(psi2)` (six fields).
 * At each face, this module averages the *tangential* components of `g1`,
 * `g2` between the two adjacent cells and forms the cross product from those
 * averages; the face-normal compact derivative of the interpolate-then-cross
 * scheme cancels algebraically out of the stored normal velocity component
 * (only the four interpolated tangential derivatives enter each face value),
 * matching the increment spec's "use the normal centered derivative and
 * consistently interpolated tangential derivatives before forming the cross
 * product" (the normal derivative contributes nothing to the retained
 * component, so it is never materialized):
 *
 *   U-face (i,j,k): a = cell(wrap(i-1),j,k), b = cell(wrap(i),j,k)
 *     t1y = 0.5*(g1y[a]+g1y[b]); t1z = 0.5*(g1z[a]+g1z[b]);
 *     t2y = 0.5*(g2y[a]+g2y[b]); t2z = 0.5*(g2z[a]+g2z[b]);
 *     U = t1y*t2z - t1z*t2y;
 *   V-face (i,j,k): a = cell(i,wrap(j-1),k), b = cell(i,wrap(j),k)
 *     t1z, t1x, t2z, t2x averaged the same way; V = t1z*t2x - t1x*t2z;
 *   W-face (i,j,k): a = cell(i,j,wrap(k-1)), b = cell(i,j,wrap(k))
 *     t1x, t1y, t2x, t2y averaged the same way; W = t1x*t2y - t1y*t2x.
 *
 * Every CompactMAC face plane is written, including the periodic duplicate
 * plane (`U` plane `nx`, `V` plane `ny`, `W` plane `nz`), which is always
 * numerically identical to plane 0 by the same wrapped formula.
 *
 * CompactMAC layout (LOCKED, matches `physics::VelocityField`):
 *   cell-centered: idx = i + nx*(j + ny*k), x-fastest, triply periodic.
 *   U: dims (nx+1, ny, nz), idx = i + j*(nx+1) + k*(nx+1)*ny.
 *   V: dims (nx, ny+1, nz), idx = i + j*nx + k*nx*(ny+1).
 *   W: dims (nx, ny, nz+1), idx = i + j*nx + k*nx*ny.
 *   U-face i lies between cells wrap(i-1) and wrap(i).
 *
 * Face-to-center averaging (pinned, applied identically to `v_psi` and the
 * Darcy input): `vx_c = 0.5*(U[i,j,k] + U[i+1,j,k])`, and analogously for
 * `vy_c` from `V` and `vz_c` from `W`.
 *
 * MAC divergence at cells (pinned, applied to the *reconstructed* field
 * only): `div = (U[i+1]-U[i])/dx + (V[j+1]-V[j])/dy + (W[k+1]-W[k])/dz`.
 *
 * Reference Darcy speed (pinned single expression): with `nu`, `nv`, `nw`
 * the raw device L2 norms (via `blas::nrm2_device`) of the three
 * cell-centered Darcy components,
 *
 *   v_rms = sqrt((nu*nu + nv*nv + nw*nw) / (double)n).
 *
 * Every kernel that needs `v_rms` (the angle-inclusion test and the
 * cross-gradient degeneracy thresholds) reads `nu`, `nv`, `nw` directly as
 * stream-ordered device scalars and recomputes `v_rms` in-kernel with this
 * exact expression; it is never floored, clamped, or otherwise hidden.
 *
 * Grid restriction: unlike the SF-02/SF-06/SF-10 evaluator chain, this
 * module does not require isotropic spacing. None of its numerical
 * dependencies -- SF-07 total gradients (already direction-independent),
 * the face-to-center averages, or the MAC divergence -- require `dx == dy
 * == dz`; each of the three grid extents must independently be finite,
 * strictly positive, and every extent must be at least 2. This is a
 * deliberate, documented SF-11 divergence from the isotropic-only
 * restriction inherited by earlier increments.
 *
 * If the measured `v_rms` is exactly zero (identically zero Darcy field),
 * every metric normalized by it becomes NaN or +infinity in the report;
 * this module never masks that with an implicit floor.
 *
 * The per-component raw-moment Pearson correlations (`corr_u/v/w`) are
 * numerically unstable, not just undefined, for exactly-degenerate
 * (zero-variance) component inputs: the formula's numerator and the
 * matching variance term can reduce to the same algebraic quantity computed
 * by two different reduction kernels (`blas::sum_device` vs
 * `blas::dot_device`), so the ratio collapses to that quantity's sign.
 * Whether the underlying quantity rounds to exactly zero (yielding NaN) or
 * to a tiny nonzero value (yielding a spurious +-1) depends on cross-kernel
 * rounding and grid size; both outcomes have been observed on identical
 * degenerate inputs at different grid extents. Nothing is floored or
 * masked; the correlation is meaningful only for non-degenerate inputs.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../numerics/blas/reduction_workspace.cuh"
#include "../../runtime/CudaContext.cuh"
#include "NonlinearSources.cuh"
#include "affine_gauge.cuh"

#include <cstddef>
#include <type_traits>

namespace macroflow3d {
namespace streamfunctions {

/**
 * Read-only CompactMAC view of the Darcy velocity field used as the
 * comparison target. `u`, `v`, `w` must be exactly `(nx+1)*ny*nz`,
 * `nx*(ny+1)*nz`, `nx*ny*(nz+1)` elements respectively, and every plane
 * (including the periodic duplicate) is expected to already be populated by
 * the caller.
 */
struct CompactMacVelocityConstView {
    DeviceSpan<const real> u;
    DeviceSpan<const real> v;
    DeviceSpan<const real> w;
};

/**
 * Caller-owned CompactMAC output view for the reconstructed velocity
 * `v_psi`. Every plane is written by `enqueue_streamfunction_physical_diagnostics`,
 * including the periodic duplicate planes.
 */
struct CompactMacVelocityView {
    DeviceSpan<real> u;
    DeviceSpan<real> v;
    DeviceSpan<real> w;
};

/**
 * Host-provided, per-launch configuration.
 *
 * `angle_exclusion_rel` and `low_speed_rel` are dimensionless multipliers of
 * the measured Darcy `v_rms`; both must be finite and strictly positive.
 * `degeneracy_thresholds[0 .. num_degeneracy_thresholds)` are additional
 * dimensionless multipliers of `v_rms` used only to split the cross-gradient
 * (`|c|`) degeneracy counts; each must be finite, strictly positive, and the
 * list must be strictly increasing. `num_degeneracy_thresholds` must be in
 * `[0, kMaxDegeneracyThresholds]`.
 */
struct PhysicalDiagnosticsConfig {
    real angle_exclusion_rel{real{1e-8}};
    real low_speed_rel{real{1e-2}};
    int num_degeneracy_thresholds{0};
    real degeneracy_thresholds[kMaxDegeneracyThresholds]{};
};

static_assert(std::is_trivially_copyable<PhysicalDiagnosticsConfig>::value,
              "PhysicalDiagnosticsConfig must be safe to copy into a kernel argument");
static_assert(std::is_trivially_copyable<CompactMacVelocityConstView>::value,
              "CompactMacVelocityConstView must be safe to copy into a kernel argument");
static_assert(std::is_trivially_copyable<CompactMacVelocityView>::value,
              "CompactMacVelocityView must be safe to copy into a kernel argument");

// Forward declaration needed by StreamfunctionDiagnosticsWorkspace's friend
// function declarations below.
struct PhysicalDiagnosticsReport;

/**
 * Reusable, exact-size storage for one physical-diagnostics evaluation.
 *
 * `prepare(grid)` allocates or resizes, exactly once for a given grid:
 *   - the six SF-07 total-gradient fields (`g1`, `g2`),
 *   - per-CompactMAC-component (U, V, W) unique-face scratch: the
 *     reconstructed value, the Darcy value, and their difference, each sized
 *     `grid.num_cells()` (unique faces only; see the file header),
 *   - the six cell-centered velocity fields (`v_psi` and Darcy, three
 *     components each),
 *   - six cell-centered scalar diagnostic fields (magnitude error, robust
 *     angle, the two Darcy-invariance dot products, MAC divergence of the
 *     reconstructed field, and `|c| = |grad(psi1) x grad(psi2)|`),
 *   - a BLAS reduction workspace shared by every `nrm2`/`dot`/`sum` call in
 *     this module (prepared once for `sum` at `grid.num_cells()`),
 *   - a fixed-layout device scalar buffer for every raw reduction result,
 *     and a device counter buffer sized `2 + 2*kMaxDegeneracyThresholds`
 *     (leading `[included, excluded]` angle counts, then
 *     `[total_t, low_speed_t]` per configured degeneracy threshold).
 *
 * After `prepare`, evaluation performs no allocation and no host
 * synchronization.
 */
class StreamfunctionDiagnosticsWorkspace {
  public:
    StreamfunctionDiagnosticsWorkspace() = default;
    ~StreamfunctionDiagnosticsWorkspace() = default;

    StreamfunctionDiagnosticsWorkspace(const StreamfunctionDiagnosticsWorkspace&) = delete;
    StreamfunctionDiagnosticsWorkspace& operator=(const StreamfunctionDiagnosticsWorkspace&) =
        delete;
    StreamfunctionDiagnosticsWorkspace(StreamfunctionDiagnosticsWorkspace&&) noexcept = default;
    StreamfunctionDiagnosticsWorkspace& operator=(StreamfunctionDiagnosticsWorkspace&&) noexcept =
        default;

    void prepare(const Grid3D& grid);
    [[nodiscard]] bool prepared_for(const Grid3D& grid) const noexcept;

  private:
    int nx_ = 0, ny_ = 0, nz_ = 0;
    std::size_t n_ = 0;

    // SF-07 total gradients (six components), cell-centered.
    DeviceBuffer<real> g1x_, g1y_, g1z_, g2x_, g2y_, g2z_;

    // Unique-face scratch (one entry per unique CompactMAC face, indexed by
    // cell layout), per component.
    DeviceBuffer<real> unique_vpsi_u_, unique_vpsi_v_, unique_vpsi_w_;
    DeviceBuffer<real> unique_vd_u_, unique_vd_v_, unique_vd_w_;
    DeviceBuffer<real> diff_u_, diff_v_, diff_w_;

    // Cell-centered velocity components (v_psi and Darcy).
    DeviceBuffer<real> vpsi_cx_, vpsi_cy_, vpsi_cz_;
    DeviceBuffer<real> vd_cx_, vd_cy_, vd_cz_;

    // Cell-centered scalar diagnostic fields.
    DeviceBuffer<real> magnitude_field_;
    DeviceBuffer<real> theta_field_;
    DeviceBuffer<real> dot1_field_;
    DeviceBuffer<real> dot2_field_;
    DeviceBuffer<real> divergence_field_;
    DeviceBuffer<real> abs_c_field_;

    blas::ReductionWorkspace reduction_workspace_;

    // Fixed-layout device scalars; see kScalar* constants in Diagnostics.cu.
    DeviceBuffer<real> scalars_;

    // [included, excluded, (total_t, low_speed_t) * kMaxDegeneracyThresholds].
    DeviceBuffer<unsigned long long> counters_;

    bool has_last_results_ = false;

    friend void enqueue_streamfunction_physical_diagnostics(
        CudaContext&, const Grid3D&, const PeriodicStreamfunctionFluctuations&, const AffineGauge&,
        const CompactMacVelocityConstView&, const PhysicalDiagnosticsConfig&,
        const CompactMacVelocityView&, StreamfunctionDiagnosticsWorkspace&);
    friend PhysicalDiagnosticsReport synchronize_streamfunction_physical_diagnostics_report(
        CudaContext&, const Grid3D&, const PhysicalDiagnosticsConfig&,
        StreamfunctionDiagnosticsWorkspace&);
};

/**
 * Host-assembled report for one physical-diagnostics evaluation.
 *
 * `_rel` fields are the corresponding raw metric divided by `v_d_rms`. All
 * metrics normalized by `v_d_rms` or by counts that may legitimately be zero
 * (e.g. `angle_included_count`) become NaN/+infinity rather than being
 * hidden behind an implicit floor; see the file header.
 */
struct PhysicalDiagnosticsReport {
    // Per-component (U, V, W) unique-face velocity reconstruction error.
    real rms_u{}, rms_v{}, rms_w{};
    real linf_u{}, linf_v{}, linf_w{};
    real rms_u_rel{}, rms_v_rel{}, rms_w_rel{};
    real linf_u_rel{}, linf_v_rel{}, linf_w_rel{};
    real e_v{};

    // Per-component Pearson correlation over unique faces. For
    // exactly-degenerate (zero-variance) component inputs this raw-moment
    // formula is numerically unstable and may report NaN or a spurious +-1
    // (sign determined by cross-kernel rounding of an algebraically shared
    // quantity; see the file header); the value is meaningful only for
    // non-degenerate inputs.
    real corr_u{}, corr_v{}, corr_w{};

    // Cell-centered magnitude error m = |v_psi_c| - |v_D_c|.
    real rms_magnitude{}, linf_magnitude{};
    real rms_magnitude_rel{}, linf_magnitude_rel{};

    // Robust angular error (radians), excluding near-zero pairs. rms_theta
    // and max_theta are NaN when angle_included_count == 0: undefined over an
    // empty included set, consistent with the no-hidden-values rule (no
    // silent zero standing in for an undefined average/maximum).
    real rms_theta{}, max_theta{};
    unsigned long long angle_included_count{};
    unsigned long long angle_excluded_count{};

    // Darcy invariance v_D . grad(psi_i).
    real invariance_raw_rms_psi1{}, invariance_raw_rms_psi2{};
    real invariance_grad_rms_psi1{}, invariance_grad_rms_psi2{};
    real invariance_e_psi1{}, invariance_e_psi2{};

    // MAC divergence of the reconstructed field (NOT algebraically zero).
    real rms_div{}, linf_div{}, e_div{};

    // |c| = |grad(psi1) x grad(psi2)| at cell centers.
    real c_min{}, c_max{}, c_mean{};
    int num_degeneracy_thresholds{};
    real degeneracy_thresholds[kMaxDegeneracyThresholds]{};
    unsigned long long degeneracy_total[kMaxDegeneracyThresholds]{};
    unsigned long long degeneracy_low_speed[kMaxDegeneracyThresholds]{};
    unsigned long long degeneracy_unexplained[kMaxDegeneracyThresholds]{};

    // Echoes.
    real v_d_rms{};
    real L_ref{};
    real angle_exclusion_rel{};
    real low_speed_rel{};
    std::size_t n{};
};

/**
 * Enqueue the full evaluation chain on `ctx.cuda_stream()`:
 *
 *   total gradients (SF-07) -> face reconstruction + unique-face scratch
 *   -> face-to-center averaging (v_psi, Darcy) -> Darcy speed norms
 *   -> cell-centered metrics (magnitude, angle, invariance, divergence,
 *      |c|, degeneracy) -> reductions.
 *
 * `v_psi.u/v/w` are caller-owned outputs of exact CompactMAC size and must
 * not overlap each other, `darcy.u/v/w`, or `fluctuations.u1`/`u2`.
 * `darcy.u/v/w` must not alias each other. `fluctuations.u1` and `u2` may
 * overlap each other because both are read-only.
 *
 * `config` is validated as documented on `PhysicalDiagnosticsConfig`. `grid`
 * must have every extent `>= 2` and finite, strictly positive, independent
 * spacings (see the file header; this module does not require isotropic
 * spacing). `workspace` must be `prepared_for(grid)`.
 *
 * This function performs no allocation and no host synchronization; the
 * only CUDA runtime calls it issues outside kernel launches are two
 * `cudaMemsetAsync` calls on `ctx.cuda_stream()` (zeroing the fixed-layout
 * scalar buffer and the counter buffer). A dedicated single-thread kernel
 * launch (not a CUDA runtime call) then sets the `|c|` minimum accumulator
 * to `+infinity`, because `cudaMemsetAsync` cannot express that bit
 * pattern.
 */
void enqueue_streamfunction_physical_diagnostics(
    CudaContext& ctx, const Grid3D& grid, const PeriodicStreamfunctionFluctuations& fluctuations,
    const AffineGauge& gauge, const CompactMacVelocityConstView& darcy,
    const PhysicalDiagnosticsConfig& config, const CompactMacVelocityView& v_psi,
    StreamfunctionDiagnosticsWorkspace& workspace);

/**
 * Explicit synchronization and host report assembly for the most recent
 * `enqueue_streamfunction_physical_diagnostics` call on `workspace`.
 *
 * `config` must match the value supplied to that enqueue call; it is only
 * used to echo host-visible parameters and threshold counts, not
 * re-validated against device state. `grid` supplies `L_ref =
 * cbrt(Lx*Ly*Lz)` and `n = grid.num_cells()`.
 *
 * This performs exactly one `ctx.synchronize()`, after enqueueing every
 * required device-to-host copy.
 */
[[nodiscard]] PhysicalDiagnosticsReport synchronize_streamfunction_physical_diagnostics_report(
    CudaContext& ctx, const Grid3D& grid, const PhysicalDiagnosticsConfig& config,
    StreamfunctionDiagnosticsWorkspace& workspace);

} // namespace streamfunctions
} // namespace macroflow3d
