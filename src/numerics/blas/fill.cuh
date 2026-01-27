#pragma once

#include "../../runtime/CudaContext.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace blas {

// Fill vector with constant value
void fill(CudaContext& ctx, DeviceSpan<real> y, real value);

} // namespace blas
} // namespace rwpt
