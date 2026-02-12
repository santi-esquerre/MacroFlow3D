#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace macroflow3d {
namespace blas {

// Fill vector with constant value
void fill(CudaContext& ctx, DeviceSpan<real> y, real value);

} // namespace blas
} // namespace macroflow3d
