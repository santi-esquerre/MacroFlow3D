#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// Scale vector: x = a*x
void scal(CudaContext& ctx, DeviceSpan<real> x, real a);

} // namespace blas
} // namespace rwpt
