#pragma once

#include "../../core/DeviceBuffer.cuh"
#include "../../core/Scalar.hpp"
#include <cstddef>

namespace rwpt {
namespace blas {

struct ReductionWorkspace {
    DeviceBuffer<std::byte> temp;
    size_t temp_bytes = 0;
    
    // Device scalar for results (no host sync in hot-path)
    DeviceBuffer<real> d_scalar;
    
    ReductionWorkspace() {
        d_scalar.resize(1);
    }
    
    void ensure_bytes(size_t required_bytes) {
        if (temp_bytes < required_bytes) {
            temp.resize(required_bytes);
            temp_bytes = required_bytes;
        }
    }
    
    void ensure_scalar() {
        if (d_scalar.size() < 1) {
            d_scalar.resize(1);
        }
    }
};

} // namespace blas
} // namespace rwpt
