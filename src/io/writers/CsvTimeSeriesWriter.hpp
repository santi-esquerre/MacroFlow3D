#pragma once

/**
 * @file CsvTimeSeriesWriter.hpp
 * @brief Host-side writer for particle-moments time-series CSV.
 * @ingroup io_writers
 *
 * Receives a batch of samples and flushes to disk in one call.
 * No CUDA, no device pointers — receives host-side data only.
 *
 * Thread-safety: single-writer (no internal locking).
 */

#include "../output_layout.hpp"
#include <cstdio>
#include <string>
#include <vector>

namespace macroflow3d {
namespace io {

/**
 * One row of the time-series CSV: moments at a given simulation time.
 */
template <typename T>
struct TimeSeriesPoint {
    T  time;
    T  mean[3];   // x, y, z
    T  var[3];    // x, y, z
    int active;
};

/**
 * @brief Writes time-series CSV (one file per realization).
 *
 * File is opened/closed per flush — no persistent FILE* held open.
 */
class CsvTimeSeriesWriter {
public:
    /**
     * @brief Write a full series (append=false → overwrite).
     *
     * @param filename  Path to output CSV.
     * @param points    Host-side time-series data.
     * @return true on success.
     */
    template <typename T>
    static bool write(const std::string& filename,
                      const std::vector<TimeSeriesPoint<T>>& points)
    {
        FILE* f = std::fopen(filename.c_str(), "w");
        if (!f) return false;

        std::fprintf(f, "# format_version=%d\n", kOutputFormatVersion);
        std::fprintf(f, "t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n");
        for (const auto& p : points) {
            std::fprintf(f,
                "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%d\n",
                (double)p.time,
                (double)p.mean[0], (double)p.mean[1], (double)p.mean[2],
                (double)p.var[0],  (double)p.var[1],  (double)p.var[2],
                p.active);
        }
        std::fclose(f);
        return true;
    }

    /**
     * @brief Write ensemble-mean CSV from multiple realizations.
     */
    template <typename T>
    static bool write_ensemble_mean(
        const std::string& filename,
        const std::vector<std::vector<TimeSeriesPoint<T>>>& all_series)
    {
        const int NR = static_cast<int>(all_series.size());
        if (NR == 0) return false;
        const int ns = static_cast<int>(all_series[0].size());
        if (ns == 0) return false;

        FILE* f = std::fopen(filename.c_str(), "w");
        if (!f) return false;

        std::fprintf(f, "# format_version=%d\n", kOutputFormatVersion);
        std::fprintf(f, "t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n");
        for (int i = 0; i < ns; ++i) {
            T t_i = all_series[0][i].time;
            T m[3] = {}, v[3] = {};
            T act = T(0);
            for (int r = 0; r < NR; ++r) {
                for (int k = 0; k < 3; ++k) {
                    m[k] += all_series[r][i].mean[k];
                    v[k] += all_series[r][i].var[k];
                }
                act += T(all_series[r][i].active);
            }
            for (int k = 0; k < 3; ++k) { m[k] /= NR; v[k] /= NR; }
            act /= NR;
            std::fprintf(f,
                "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.0f\n",
                (double)t_i,
                (double)m[0], (double)m[1], (double)m[2],
                (double)v[0], (double)v[1], (double)v[2],
                (double)act);
        }
        std::fclose(f);
        return true;
    }
};

} // namespace io
} // namespace macroflow3d
