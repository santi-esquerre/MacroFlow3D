#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace macroflow3d {
namespace blas {

// Copy vector: y = x
void copy(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y);

} // namespace blas
} // namespace macroflow3d
