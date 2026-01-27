#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// Axpby: y = a*x + b*y
void axpby(CudaContext& ctx, real a, DeviceSpan<const real> x, real b, DeviceSpan<real> y);

} // namespace blas
} // namespace rwpt
