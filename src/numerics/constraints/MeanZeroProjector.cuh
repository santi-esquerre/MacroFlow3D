#pragma once

/**
 * @file MeanZeroProjector.cuh
 * @brief Explicit GPU projection onto the zero-mean scalar-field subspace.
 *
 * `real` is double precision.  The projector applies the unweighted
 * cell-centered operation P(x) = x - mean(x) in place; it has no grid or pin
 * policy and is therefore reusable for every periodic field with the prepared
 * number of cells.  The caller owns the workspace and must keep it, and the
 * field, alive until queued work on the supplied CudaContext stream completes.
 */

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"
#include "../blas/reduction_workspace.cuh"

#include <cstddef>

namespace macroflow3d {
namespace constraints {

/**
 * Caller-owned storage for a mean-zero projection.
 *
 * Construction may reserve the reduction result scalar.  prepare() may query
 * the reduction backend and reserve or resize its temporary storage.  Once
 * prepared, project() and mean_device() neither allocate nor synchronize the
 * host.  A workspace is valid only for its exact prepared size and is not safe
 * to use concurrently on multiple host threads or streams.
 */
class MeanZeroWorkspace {
  public:
    MeanZeroWorkspace() = default;
    ~MeanZeroWorkspace() = default;

    MeanZeroWorkspace(const MeanZeroWorkspace&) = delete;
    MeanZeroWorkspace& operator=(const MeanZeroWorkspace&) = delete;
    MeanZeroWorkspace(MeanZeroWorkspace&&) noexcept = default;
    MeanZeroWorkspace& operator=(MeanZeroWorkspace&&) noexcept = default;

    // Prepare exactly n values, querying the backend and reserving or resizing
    // temporary reduction storage outside the projector hot path as needed.
    void prepare(std::size_t n);

    [[nodiscard]] bool prepared_for(std::size_t n) const noexcept;
    [[nodiscard]] std::size_t prepared_size() const noexcept { return prepared_n_; }
    [[nodiscard]] std::size_t temporary_storage_capacity_bytes() const noexcept;

  private:
    blas::ReductionWorkspace reduction_;
    std::size_t prepared_n_ = 0;
    bool prepared_ = false;

    friend class MeanZeroProjector;
};

/**
 * Stateless in-place projector P = I - 11^T/n.
 *
 * project() and mean_device() only enqueue work on ctx.cuda_stream(); neither
 * allocates device storage nor synchronizes the host.  mean_host() is a
 * diagnostic operation: it performs one asynchronous scalar copy followed by
 * the explicit CudaContext synchronization.
 */
class MeanZeroProjector {
  public:
    // Enqueue x <- x - mean(x). x must not alias the workspace result scalar.
    void project(CudaContext& ctx, DeviceSpan<real> x, MeanZeroWorkspace& workspace) const;

    // Enqueue a double-precision mean. The returned scalar is workspace-owned
    // and is overwritten by the next projector operation using that workspace.
    [[nodiscard]] DeviceSpan<const real> mean_device(CudaContext& ctx,
                                                      DeviceSpan<const real> x,
                                                      MeanZeroWorkspace& workspace) const;

    // Diagnostic-only mean query. This operation synchronizes ctx explicitly.
    [[nodiscard]] real mean_host(CudaContext& ctx, DeviceSpan<const real> x,
                                 MeanZeroWorkspace& workspace) const;
};

} // namespace constraints
} // namespace macroflow3d
