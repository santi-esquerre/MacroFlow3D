#pragma once

/**
 * @file CsvParticleSnapshotWriter.hpp
 * @brief Host-side writer for particle snapshot CSV files.
 *
 * Receives host-arrays (already staged from device) and flushes to disk.
 * No CUDA dependency — pure C++17.
 */

#include "../output_layout.hpp"
#include <cstdio>
#include <string>
#include <filesystem>

namespace macroflow3d {
namespace io {

/**
 * @brief Host-side arrays for a particle snapshot (non-owning spans).
 */
template <typename T>
struct HostParticleSnapshot {
    const T* x = nullptr;
    const T* y = nullptr;
    const T* z = nullptr;
    int n = 0;              // number of particles to write
    T time = T(0);
    int step = 0;

    // Optional unwrapped positions
    const T* x_u = nullptr;
    const T* y_u = nullptr;
    const T* z_u = nullptr;
    bool has_unwrapped = false;

    // Stride: write every `stride` particles (1 = all)
    int stride = 1;

    int precision = 10;
};

/**
 * @brief Stateless CSV snapshot writer (host-side only).
 */
class CsvParticleSnapshotWriter {
public:

    template <typename T>
    static bool write(const std::string& filename,
                      const HostParticleSnapshot<T>& snap)
    {
        // Ensure parent directory exists
        auto parent = std::filesystem::path(filename).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        FILE* f = std::fopen(filename.c_str(), "w");
        if (!f) return false;

        const int s = (snap.stride > 0) ? snap.stride : 1;

        // Version + Header
        // std::fprintf(f, "# format_version=%d\n", kOutputFormatVersion);
        if (snap.has_unwrapped) {
            std::fprintf(f, "t,id,x,y,z,x_u,y_u,z_u\n");
        } else {
            std::fprintf(f, "t,id,x,y,z\n");
        }

        char fmt_wrap[128];
        char fmt_unwrap[128];
        std::snprintf(fmt_wrap, sizeof(fmt_wrap),
                      "%%.%de,%%d,%%.%de,%%.%de,%%.%de\n",
                      snap.precision, snap.precision,
                      snap.precision, snap.precision);
        std::snprintf(fmt_unwrap, sizeof(fmt_unwrap),
                      "%%.%de,%%d,%%.%de,%%.%de,%%.%de,%%.%de,%%.%de,%%.%de\n",
                      snap.precision, snap.precision,
                      snap.precision, snap.precision,
                      snap.precision, snap.precision, snap.precision);

        for (int i = 0; i < snap.n; i += s) {
            if (snap.has_unwrapped) {
                std::fprintf(f, fmt_unwrap,
                             (double)snap.time, i,
                             (double)snap.x[i], (double)snap.y[i], (double)snap.z[i],
                             (double)snap.x_u[i], (double)snap.y_u[i], (double)snap.z_u[i]);
            } else {
                std::fprintf(f, fmt_wrap,
                             (double)snap.time, i,
                             (double)snap.x[i], (double)snap.y[i], (double)snap.z[i]);
            }
        }

        std::fclose(f);
        return true;
    }
};

} // namespace io
} // namespace macroflow3d
