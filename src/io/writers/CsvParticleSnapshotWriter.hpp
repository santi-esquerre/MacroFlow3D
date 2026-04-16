#pragma once

/**
 * @file CsvParticleSnapshotWriter.hpp
 * @brief Host-side writer for particle snapshot CSV files.
 *
 * Receives host-arrays (already staged from device) and flushes to disk.
 * No CUDA dependency — pure C++17.
 */

#include "../output_layout.hpp"
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

namespace macroflow3d {
namespace io {

/**
 * @brief Host-side arrays for a particle snapshot (non-owning spans).
 */
template <typename T> struct HostParticleSnapshot {
    const T* x = nullptr;
    const T* y = nullptr;
    const T* z = nullptr;
    int n = 0;
    T time = T(0);
    int step = 0;

    // Optional unwrapped positions
    const T* x_u = nullptr;
    const T* y_u = nullptr;
    const T* z_u = nullptr;
    bool has_unwrapped = false;

    // Optional particle status (uint8: 0=active, non-zero=inactive/exited)
    const uint8_t* status = nullptr;
    bool has_status = false;

    // Optional periodic wrap counters
    const int32_t* wrapX = nullptr;
    const int32_t* wrapY = nullptr;
    const int32_t* wrapZ = nullptr;
    bool has_wrap_counts = false;

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
    static bool write(const std::string& filename, const HostParticleSnapshot<T>& snap) {
        // Ensure parent directory exists
        auto parent = std::filesystem::path(filename).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);

        FILE* f = std::fopen(filename.c_str(), "w");
        if (!f)
            return false;

        const int s = (snap.stride > 0) ? snap.stride : 1;

        // ── Header ────────────────────────────────────────────────────────
        std::string hdr = "t,id,x,y,z";
        if (snap.has_status)
            hdr += ",status";
        if (snap.has_wrap_counts)
            hdr += ",wrapX,wrapY,wrapZ";
        if (snap.has_unwrapped)
            hdr += ",x_u,y_u,z_u";
        hdr += "\n";
        std::fputs(hdr.c_str(), f);

        // ── Row format strings ─────────────────────────────────────────────
        // fmt_base: t, id, x, y, z  (no trailing newline)
        // fmt_uw:   ,x_u, y_u, z_u  (no trailing newline)
        char fmt_base[256], fmt_uw[128];
        std::snprintf(fmt_base, sizeof(fmt_base), "%%.%de,%%d,%%.%de,%%.%de,%%.%de", snap.precision,
                      snap.precision, snap.precision, snap.precision);
        std::snprintf(fmt_uw, sizeof(fmt_uw), ",%%.%de,%%.%de,%%.%de", snap.precision,
                      snap.precision, snap.precision);

        // ── Rows ──────────────────────────────────────────────────────────
        for (int i = 0; i < snap.n; i += s) {
            std::fprintf(f, fmt_base, (double)snap.time, i, (double)snap.x[i], (double)snap.y[i],
                         (double)snap.z[i]);
            if (snap.has_status && snap.status)
                std::fprintf(f, ",%d", (int)snap.status[i]);
            if (snap.has_wrap_counts && snap.wrapX)
                std::fprintf(f, ",%d,%d,%d", snap.wrapX[i], snap.wrapY[i], snap.wrapZ[i]);
            if (snap.has_unwrapped && snap.x_u)
                std::fprintf(f, fmt_uw, (double)snap.x_u[i], (double)snap.y_u[i],
                             (double)snap.z_u[i]);
            std::fputc('\n', f);
        }

        std::fclose(f);
        return true;
    }
};

} // namespace io
} // namespace macroflow3d
