#pragma once

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"
#include "../../runtime/CudaContext.cuh"

namespace macroflow3d {
namespace blas {

// Axpby: y = a*x + b*y
void axpby(CudaContext& ctx, real a, DeviceSpan<const real> x, real b, DeviceSpan<real> y);

} // namespace blas
} // namespace macroflow3d
