#pragma once

/**
 * @file MacrodispersionAnalysis.hpp
 * @brief Compute macrodispersivity α(t) from particle moment time-series.
 *
 * Pure CPU — no CUDA dependency. Can consume:
 *   (A) In-memory time-series from the pipeline
 *   (B) CSV files written by the scheduler (offline analysis)
 *
 * Design: analysis is SEPARATE from runtime. The pipeline may call it
 * inline after all realizations, or it can be run offline later.
 */

#include "../../io/writers/CsvTimeSeriesWriter.hpp"
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace macroflow3d {
namespace analysis {

/**
 * @brief Row of macrodispersion output.
 */
template <typename T>
struct MacrodispersionRow {
    T time;
    T alpha[3];      // α_x, α_y, α_z
    T dvar_dt[3];    // dVar/dt  (before scaling)
};

/**
 * @brief Compute macrodispersivity from variance time-series.
 *
 * α_k(t) = (1 / (2 λ ||<v>||)) · d(Var_k)/dt
 *
 * Uses central differences (forward/backward at edges).
 * Averages dVar/dt across NR realizations.
 *
 * @param all_series  Per-realization time-series (moments).
 * @param lambda      Correlation length.
 * @param vmean_norm  Mean velocity norm ||<v>||.
 * @return  Vector of MacrodispersionRow (one per time step).
 */
template <typename T>
std::vector<MacrodispersionRow<T>> compute_macrodispersion(
    const std::vector<std::vector<io::TimeSeriesPoint<T>>>& all_series,
    T lambda, T vmean_norm)
{
    std::vector<MacrodispersionRow<T>> result;

    const int NR = static_cast<int>(all_series.size());
    if (NR == 0) return result;
    const int ns = static_cast<int>(all_series[0].size());
    if (ns < 2) return result;

    // Validate equal length
    for (int r = 0; r < NR; ++r) {
        if (static_cast<int>(all_series[r].size()) != ns) return result;
    }

    const T factor = T(1) / (T(2) * lambda * vmean_norm);
    result.resize(ns);

    for (int i = 0; i < ns; ++i) {
        result[i].time = all_series[0][i].time;
        T dvar_dt_avg[3] = {T(0), T(0), T(0)};

        for (int r = 0; r < NR; ++r) {
            const auto& s = all_series[r];
            for (int k = 0; k < 3; ++k) {
                T dv;
                if (i == 0) {
                    T dt01 = s[1].time - s[0].time;
                    dv = (dt01 > T(0)) ? (s[1].var[k] - s[0].var[k]) / dt01 : T(0);
                } else if (i == ns - 1) {
                    T dt = s[ns-1].time - s[ns-2].time;
                    dv = (dt > T(0)) ? (s[ns-1].var[k] - s[ns-2].var[k]) / dt : T(0);
                } else {
                    T dt = s[i+1].time - s[i-1].time;
                    dv = (dt > T(0)) ? (s[i+1].var[k] - s[i-1].var[k]) / dt : T(0);
                }
                dvar_dt_avg[k] += dv;
            }
        }

        for (int k = 0; k < 3; ++k) {
            dvar_dt_avg[k] /= T(NR);
            result[i].alpha[k]   = factor * dvar_dt_avg[k];
            result[i].dvar_dt[k] = dvar_dt_avg[k];
        }
    }

    return result;
}

/**
 * @brief Write macrodispersion results to CSV.
 */
template <typename T>
bool write_macrodispersion_csv(const std::string& filename,
                               const std::vector<MacrodispersionRow<T>>& rows) {
    FILE* f = std::fopen(filename.c_str(), "w");
    if (!f) return false;

    std::fprintf(f, "t,alpha_x,alpha_y,alpha_z,dvar_dt_x,dvar_dt_y,dvar_dt_z\n");
    for (const auto& row : rows) {
        std::fprintf(f, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n",
                     (double)row.time,
                     (double)row.alpha[0], (double)row.alpha[1], (double)row.alpha[2],
                     (double)row.dvar_dt[0], (double)row.dvar_dt[1], (double)row.dvar_dt[2]);
    }
    std::fclose(f);
    return true;
}

/**
 * @brief Read a single realization's time-series from CSV (offline analysis).
 *
 * Reads CSV with header: t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active
 *
 * @param filename  Path to CSV.
 * @return  Parsed time-series points.
 */
template <typename T>
std::vector<io::TimeSeriesPoint<T>> read_timeseries_csv(const std::string& filename) {
    std::vector<io::TimeSeriesPoint<T>> pts;

    std::ifstream ifs(filename);
    if (!ifs.is_open()) return pts;

    std::string line;
    // Skip header
    if (!std::getline(ifs, line)) return pts;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        io::TimeSeriesPoint<T> p;
        // Parse: t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active
        char* end = nullptr;
        const char* s = line.c_str();

        p.time    = static_cast<T>(std::strtod(s, &end)); s = end + 1;
        p.mean[0] = static_cast<T>(std::strtod(s, &end)); s = end + 1;
        p.mean[1] = static_cast<T>(std::strtod(s, &end)); s = end + 1;
        p.mean[2] = static_cast<T>(std::strtod(s, &end)); s = end + 1;
        p.var[0]  = static_cast<T>(std::strtod(s, &end)); s = end + 1;
        p.var[1]  = static_cast<T>(std::strtod(s, &end)); s = end + 1;
        p.var[2]  = static_cast<T>(std::strtod(s, &end)); s = end + 1;
        p.active  = static_cast<int>(std::strtol(s, &end, 10));

        pts.push_back(p);
    }

    return pts;
}

} // namespace analysis
} // namespace macroflow3d
