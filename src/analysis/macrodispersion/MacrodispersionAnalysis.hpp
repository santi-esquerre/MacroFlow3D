#pragma once

/**
 * @file MacrodispersionAnalysis.hpp
 * @brief Compute dimensionless macrodispersivity α*(t) from particle moment time-series.
 * @ingroup analysis
 *
 * Returns \f$\alpha^*_k(t) =
 * \frac{1}{2\lambda\|\langle\mathbf{v}\rangle\|}\frac{d\mathrm{Var}_k}{dt}\f$ which is the
 * **dimensionless** macrodispersivity (\f$[-]\f$). The dimensional macrodispersivity is \f$\alpha_k
 * = \lambda\,\alpha^*_k\f$ \f$[L]\f$.
 *
 * Pure CPU — no CUDA dependency. Can consume:
 *   (A) In-memory time-series from the pipeline
 *   (B) CSV files written by the scheduler (offline analysis)
 *
 * Design: analysis is SEPARATE from runtime. The pipeline may call it
 * inline after all realizations, or it can be run offline later.
 */

#include "../../io/writers/CsvTimeSeriesWriter.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace analysis {

/**
 * @brief Row of macrodispersion output.
 *
 * `alpha[k]` stores the **dimensionless** macrodispersivity
 * \f$\alpha^*_k = \frac{1}{2\lambda\|\langle\mathbf{v}\rangle\|}\frac{d\mathrm{Var}_k}{dt}\f$.
 * Dimensional macrodispersivity: \f$\alpha_k = \lambda\,\alpha^*_k\f$ \f$[L]\f$.
 */
template <typename T> struct MacrodispersionRow {
    T time;
    T alpha[3];   // α*_x, α*_y, α*_z  (dimensionless)
    T dvar_dt[3]; // dVar/dt  [L²/T]  (before scaling)
};

/**
 * @brief Compute dimensionless macrodispersivity from variance time-series.
 *
 * \f$\alpha^*_k(t) =
 * \frac{1}{2\,\lambda\,\|\langle\mathbf{v}\rangle\|}\;\frac{d(\mathrm{Var}_k)}{dt}\f$
 *
 * This is the **dimensionless** macrodispersivity \f$[-]\f$.
 * Uses central differences (forward/backward at edges).
 * Averages dVar/dt across NR realizations.
 *
 * @param all_series  Per-realization time-series (moments).
 * @param lambda      Correlation length \f$\lambda\f$ \f$[L]\f$.
 * @param vmean_norm  Mean velocity norm \f$\|\langle\mathbf{v}\rangle\|\f$ \f$[L/T]\f$.
 * @return  Vector of MacrodispersionRow (one per time step); `alpha` is \f$\alpha^*\f$ \f$[-]\f$.
 */
template <typename T>
std::vector<MacrodispersionRow<T>>
compute_macrodispersion(const std::vector<std::vector<io::TimeSeriesPoint<T>>>& all_series,
                        T lambda, T vmean_norm) {
    std::vector<MacrodispersionRow<T>> result;

    const int NR = static_cast<int>(all_series.size());
    if (NR == 0)
        return result;
    const int ns = static_cast<int>(all_series[0].size());
    if (ns < 2)
        return result;

    // Validate equal length
    for (int r = 0; r < NR; ++r) {
        if (static_cast<int>(all_series[r].size()) != ns)
            return result;
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
                    T dt = s[ns - 1].time - s[ns - 2].time;
                    dv = (dt > T(0)) ? (s[ns - 1].var[k] - s[ns - 2].var[k]) / dt : T(0);
                } else {
                    T dt = s[i + 1].time - s[i - 1].time;
                    dv = (dt > T(0)) ? (s[i + 1].var[k] - s[i - 1].var[k]) / dt : T(0);
                }
                dvar_dt_avg[k] += dv;
            }
        }

        for (int k = 0; k < 3; ++k) {
            dvar_dt_avg[k] /= T(NR);
            result[i].alpha[k] = factor * dvar_dt_avg[k];
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
    if (!f)
        return false;

    std::fprintf(f, "t,alpha_x,alpha_y,alpha_z,dvar_dt_x,dvar_dt_y,dvar_dt_z\n");
    for (const auto& row : rows) {
        std::fprintf(f, "%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", (double)row.time,
                     (double)row.alpha[0], (double)row.alpha[1], (double)row.alpha[2],
                     (double)row.dvar_dt[0], (double)row.dvar_dt[1], (double)row.dvar_dt[2]);
    }
    const bool bad = std::ferror(f) != 0; // fclose flushes: both must be clean
    return std::fclose(f) == 0 && !bad;
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
    if (!ifs.is_open())
        return pts;

    std::string line;
    // Format contract (CsvTimeSeriesWriter): leading '#' comment lines, one column-header line, rows.
    while (std::getline(ifs, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] != '#')
            break; // `line` is the column header (or EOF: empty series)
    }
    if (!ifs)
        return pts;

    while (std::getline(ifs, line)) {
        if (line.empty())
            continue;

        io::TimeSeriesPoint<T> p;
        // Parse: t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active; a malformed field throws (never a silent 0)
        char* end = nullptr;
        const char* s = line.c_str();
        const auto bad = [&] {
            throw std::runtime_error("malformed numeric field in " + filename + ": '" + line + "'");
        };
        const auto next = [&]() -> T {
            const double v = std::strtod(s, &end);
            if (end == s || *end != ',')
                bad();
            s = end + 1;
            return static_cast<T>(v);
        };

        p.time = next();
        p.mean[0] = next();
        p.mean[1] = next();
        p.mean[2] = next();
        p.var[0] = next();
        p.var[1] = next();
        p.var[2] = next();
        p.active = static_cast<int>(std::strtol(s, &end, 10));
        if (end == s || line.find_first_not_of(" \t\r", static_cast<std::size_t>(end - line.c_str())) !=
                            std::string::npos)
            bad();

        pts.push_back(p);
    }

    return pts;
}

} // namespace analysis
} // namespace macroflow3d
