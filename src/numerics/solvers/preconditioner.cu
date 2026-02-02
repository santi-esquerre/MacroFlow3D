/**
 * @file preconditioner.cu
 * @brief Implementation of basic preconditioners
 */

#include "preconditioner.cuh"
#include "../blas/blas.cuh"

namespace rwpt {
namespace solvers {

void IdentityPreconditioner::apply(
    CudaContext& ctx,
    DeviceSpan<const real> r,
    DeviceSpan<real> z
) const {
    // z = r (identity preconditioner)
    rwpt::blas::copy(ctx, r, z);
}

} // namespace solvers
} // namespace rwpt
