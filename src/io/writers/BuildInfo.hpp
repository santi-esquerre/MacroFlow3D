#pragma once

/**
 * @file BuildInfo.hpp
 * @brief GPU and build metadata for manifest enrichment (Etapa 8).
 *
 * Queries CUDA runtime for device info. Build flags are injected
 * via CMake compile definitions (MACROFLOW3D_GIT_HASH, MACROFLOW3D_BUILD_TYPE, etc.).
 */

#include <string>
#include <cuda_runtime.h>

namespace macroflow3d {
namespace io {

/**
 * @brief GPU device information snapshot.
 */
struct GPUInfo {
    std::string device_name;
    int compute_major = 0;
    int compute_minor = 0;
    size_t global_mem_bytes = 0;
    int cuda_runtime_version = 0;   // e.g. 12090
    int cuda_driver_version  = 0;

    /// Query the current CUDA device and populate fields.
    static GPUInfo query() {
        GPUInfo info;

        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return info;

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return info;

        info.device_name      = prop.name;
        info.compute_major    = prop.major;
        info.compute_minor    = prop.minor;
        info.global_mem_bytes = prop.totalGlobalMem;

        cudaRuntimeGetVersion(&info.cuda_runtime_version);
        cudaDriverGetVersion(&info.cuda_driver_version);

        return info;
    }

    std::string compute_capability() const {
        return std::to_string(compute_major) + "." + std::to_string(compute_minor);
    }

    /// VRAM in MiB (human-readable).
    int vram_mib() const {
        return static_cast<int>(global_mem_bytes / (1024ULL * 1024ULL));
    }
};

/**
 * @brief Compile-time build information.
 *
 * Values are populated from CMake -D defines. If not defined,
 * sensible fallbacks are used.
 */
struct BuildInfo {
#ifdef MACROFLOW3D_GIT_HASH
    static constexpr const char* git_hash = MACROFLOW3D_GIT_HASH;
#else
    static constexpr const char* git_hash = "unknown";
#endif

#ifdef MACROFLOW3D_BUILD_TYPE
    static constexpr const char* build_type = MACROFLOW3D_BUILD_TYPE;
#else
    static constexpr const char* build_type = "unknown";
#endif

#ifdef MACROFLOW3D_CUDA_ARCHITECTURES
    static constexpr const char* cuda_architectures = MACROFLOW3D_CUDA_ARCHITECTURES;
#else
    static constexpr const char* cuda_architectures = "unknown";
#endif

#ifdef __NVCC__
    static constexpr int nvcc_version = __CUDACC_VER_MAJOR__ * 1000
                                      + __CUDACC_VER_MINOR__ * 10;
#else
    static constexpr int nvcc_version = 0;
#endif
};

} // namespace io
} // namespace macroflow3d
