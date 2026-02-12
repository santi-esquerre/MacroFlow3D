#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace macroflow3d {
namespace blas {

// Axpy: y = a*x + y
void axpy(CudaContext& ctx, real a, DeviceSpan<const real> x, DeviceSpan<real> y);

} // namespace blas
} // namespace macroflow3d
