#pragma once

#include "../../core/DeviceBuffer.cuh"
#include "../../core/Scalar.hpp"
#include <cstddef>

namespace macroflow3d {
namespace blas {

struct ReductionWorkspace {
    // Caller-owned temporary storage for reductions (std::byte is not C++14).
    // Its capacity may exceed the bytes required by the currently prepared operation.
    DeviceBuffer<unsigned char> temp;
    size_t temp_bytes = 0;

    // Device scalar for results (no host sync in hot-path)
    DeviceBuffer<real> d_scalar;

    // Exact preparation metadata for CUB double-precision sums.  Keeping this
    // separate from temp_bytes lets the hot path reject a changed input size
    // rather than silently resizing or re-querying CUB.
    bool sum_is_prepared = false;
    size_t sum_prepared_size = 0;
    size_t sum_prepared_bytes = 0;

    ReductionWorkspace() { d_scalar.resize(1); }

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
} // namespace macroflow3d
