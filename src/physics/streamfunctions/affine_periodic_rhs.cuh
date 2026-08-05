#pragma once

/**
 * @file affine_periodic_rhs.cuh
 * @brief Asynchronous assembly of periodic affine streamfunction right-hand sides.
 */

#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Grid3D.hpp"
#include "../../core/Scalar.hpp"
#include "../../numerics/constraints/MeanZeroProjector.cuh"
#include "../../runtime/CudaContext.cuh"
#include "affine_gauge.cuh"

#include <array>
#include <cstddef>

namespace macroflow3d {
namespace streamfunctions {

/** Device-resident compatibility diagnostics, in psi1, psi2 order. */
struct AffineRhsDeviceDiagnostics {
    DeviceSpan<const real> raw_means;
    DeviceSpan<const real> projected_means;
};

/** Host copy of the affine RHS compatibility diagnostics, in psi1, psi2 order. */
struct AffineRhsHostDiagnostics {
    std::array<real, 2> raw_means{};
    std::array<real, 2> projected_means{};
};

/**
 * Reusable storage for affine RHS projection and diagnostics.
 *
 * Construction may reserve the reduction result scalar. prepare() may query
 * the reduction backend and reserve or resize its temporary storage and the
 * diagnostic buffers. After exact-size preparation, assembly neither
 * allocates nor synchronizes the host. The workspace and returned diagnostics
 * are overwritten by the next assembly and are not safe for concurrent streams
 * or host threads.
 */
class AffinePeriodicRhsWorkspace {
  public:
    AffinePeriodicRhsWorkspace() = default;
    ~AffinePeriodicRhsWorkspace() = default;

    AffinePeriodicRhsWorkspace(const AffinePeriodicRhsWorkspace&) = delete;
    AffinePeriodicRhsWorkspace& operator=(const AffinePeriodicRhsWorkspace&) = delete;
    AffinePeriodicRhsWorkspace(AffinePeriodicRhsWorkspace&&) noexcept = default;
    AffinePeriodicRhsWorkspace& operator=(AffinePeriodicRhsWorkspace&&) noexcept = default;

    void prepare(std::size_t n);
    [[nodiscard]] bool prepared_for(std::size_t n) const noexcept;

  private:
    constraints::MeanZeroWorkspace mean_zero_;
    DeviceBuffer<real> raw_means_;
    DeviceBuffer<real> projected_means_;

    friend AffineRhsDeviceDiagnostics assemble_affine_periodic_rhs(
        CudaContext&, const Grid3D&, DeviceSpan<const real>, const AffineGauge&, DeviceSpan<real>,
        DeviceSpan<real>, AffinePeriodicRhsWorkspace&);
};

/**
 * Enqueue both div_h(q*gbar) right-hand sides and their mean-zero projections.
 *
 * q, rhs1, rhs2, workspace, and returned device diagnostics must remain alive
 * until queued work on ctx.cuda_stream() completes. q must contain finite,
 * strictly positive values; this precondition is intentionally not reduced or
 * synchronized here. This function performs no allocation or host sync.
 */
[[nodiscard]] AffineRhsDeviceDiagnostics assemble_affine_periodic_rhs(
    CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> q, const AffineGauge& gauge,
    DeviceSpan<real> rhs1, DeviceSpan<real> rhs2, AffinePeriodicRhsWorkspace& workspace);

/**
 * Explicit diagnostic transfer. Enqueues four scalar copies and performs
 * exactly one CudaContext synchronization before returning their host values.
 */
[[nodiscard]] AffineRhsHostDiagnostics synchronize_affine_rhs_diagnostics(
    CudaContext& ctx, const AffineRhsDeviceDiagnostics& diagnostics);

} // namespace streamfunctions
} // namespace macroflow3d
