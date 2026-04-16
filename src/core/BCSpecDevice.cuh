#pragma once

#include "BCSpec.hpp"
#include "Scalar.hpp"
#include <cstdint>
#include <cuda_runtime.h>

namespace macroflow3d {

// Device-friendly POD for BC specification
// Passed by value to kernels (cheap: 6 bytes + 6*8 = 54 bytes)
struct BCSpecDevice {
    // Type for each face (order: xmin, xmax, ymin, ymax, zmin, zmax)
    uint8_t type[6];

    // Value for each face (same order)
    real value[6];

    __host__ __device__ BCSpecDevice() {
        for (int i = 0; i < 6; ++i) {
            type[i] = static_cast<uint8_t>(BCType::Dirichlet);
            value[i] = 0.0;
        }
    }
};

// Convert BCSpec to device view (host-only, trivial)
inline BCSpecDevice to_device(const BCSpec& bc) {
    BCSpecDevice dev;

    // Order: xmin=0, xmax=1, ymin=2, ymax=3, zmin=4, zmax=5
    dev.type[0] = static_cast<uint8_t>(bc.xmin.type);
    dev.value[0] = bc.xmin.value;

    dev.type[1] = static_cast<uint8_t>(bc.xmax.type);
    dev.value[1] = bc.xmax.value;

    dev.type[2] = static_cast<uint8_t>(bc.ymin.type);
    dev.value[2] = bc.ymin.value;

    dev.type[3] = static_cast<uint8_t>(bc.ymax.type);
    dev.value[3] = bc.ymax.value;

    dev.type[4] = static_cast<uint8_t>(bc.zmin.type);
    dev.value[4] = bc.zmin.value;

    dev.type[5] = static_cast<uint8_t>(bc.zmax.type);
    dev.value[5] = bc.zmax.value;

    return dev;
}

} // namespace macroflow3d
