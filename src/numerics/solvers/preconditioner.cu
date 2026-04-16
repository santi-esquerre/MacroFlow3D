/**
 * @file preconditioner.cu
 * @brief Implementation of basic preconditioners
 */

#include "../blas/blas.cuh"
#include "preconditioner.cuh"

namespace macroflow3d {
namespace solvers {

void IdentityPreconditioner::apply(CudaContext& ctx, DeviceSpan<const real> r,
                                   DeviceSpan<real> z) const {
    // z = r (identity preconditioner)
    macroflow3d::blas::copy(ctx, r, z);
}

} // namespace solvers
} // namespace macroflow3d
