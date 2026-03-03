/**
 * @file CsvDiagnosticsWriter.hpp
 * @brief Append-mode CSV writers for PSPTA runtime diagnostics.
 *
 * Both writers follow the same "header-once" pattern:
 *   - Files are opened in append mode.
 *   - If the file looks empty (tellp == 0), the header row is written first.
 *   - Each call appends exactly one data row.
 *
 * Thread safety: not thread-safe; callers must serialize if needed.
 */

#pragma once

#include <cstdio>
#include <cmath>
#include <string>
#include <fstream>
#include <stdexcept>

#include "../../physics/particles/pspta/PsptaPsiField.cuh"
#include "../../physics/particles/pspta/PsptaEngine.hpp"

namespace macroflow3d {
namespace runtime {

// ============================================================================
// Internal helper
// ============================================================================

namespace detail {

/// Open/create an append-mode text file.  Throws std::runtime_error on failure.
inline std::ofstream open_append(const std::string& path) {
    std::ofstream f(path, std::ios::app);
    if (!f.is_open())
        throw std::runtime_error("[CsvDiagnosticsWriter] Cannot open: " + path);
    return f;
}

/// Return true if the file exists and has non-zero content.
inline bool file_has_content(const std::string& path) {
    std::ifstream f(path, std::ios::ate);
    return f.is_open() && (f.tellg() > 0);
}

} // namespace detail

// ============================================================================
// CsvDiagnosticsWriter
// ============================================================================

class CsvDiagnosticsWriter {
public:

    // ── ψ quality ─────────────────────────────────────────────────────────────

    /**
     * @brief Append one row to `psi_quality.csv`.
     *
     * Schema:
     *   realization_id, n_cells,
     *   rms_r1, max_r1, rms_r2, max_r2,
     *   vx_clamped, vx_clamped_frac
     *
     * @param path         Absolute or relative path to the CSV file.
     * @param realization  Zero-based realization index.
     * @param pre_rep      PsptaPrecomputeReport (level A output).
     * @param qual_rep     PsiQualityReport from compute_psi_quality().
     */
    static void write_psi_quality_row(
        const std::string&                         path,
        int                                        realization,
        const pspta::PsptaPrecomputeReport&        pre_rep,
        const pspta::PsiQualityReport&             qual_rep)
    {
        const bool write_header = !detail::file_has_content(path);
        auto f = detail::open_append(path);

        if (write_header)
            f << "realization_id,n_cells,"
                 "rms_r1,max_r1,rms_r2,max_r2,"
                 "vx_clamped,vx_clamped_frac\n";

        const double clamp_frac = (pre_rep.n_total > 0)
            ? static_cast<double>(pre_rep.n_vx_clamped) /
              static_cast<double>(pre_rep.n_total)
            : 0.0;

        f << realization          << ','
          << qual_rep.n_cells     << ','
          << qual_rep.rms_r1      << ','
          << qual_rep.max_r1      << ','
          << qual_rep.rms_r2      << ','
          << qual_rep.max_r2      << ','
          << pre_rep.n_vx_clamped << ','
          << clamp_frac           << '\n';
    }

    // ── Newton robustness ─────────────────────────────────────────────────────

    /**
     * @brief Append one row to `newton_fail_summary.csv`.
     *
     * Schema:
     *   realization_id, N,
     *   n_nonzero, frac_nonzero,
     *   total_fail, mean_fail, max_fail,
     *   hist_0, hist_1, hist_2, hist_3_4, hist_5_8, hist_9_16, hist_17p
     *
     * @param path         Absolute or relative path to the CSV file.
     * @param realization  Zero-based realization index.
     * @param N            Total number of particles.
     * @param ts           TransportStats from compute_transport_stats().
     */
    static void write_newton_fail_row(
        const std::string&                     path,
        int                                    realization,
        int                                    N,
        const pspta::PsptaEngine::TransportStats& ts)
    {
        const bool write_header = !detail::file_has_content(path);
        auto f = detail::open_append(path);

        if (write_header)
            f << "realization_id,N,"
                 "n_nonzero,frac_nonzero,"
                 "total_fail,mean_fail,max_fail,"
                 "hist_0,hist_1,hist_2,hist_3_4,hist_5_8,hist_9_16,hist_17p\n";

        const double dN = (N > 0) ? static_cast<double>(N) : 1.0;
        const double frac_nonzero = static_cast<double>(ts.n_nonzero_fail) / dN;
        const double mean_fail    = static_cast<double>(ts.total_fail) / dN;

        f << realization          << ','
          << N                    << ','
          << ts.n_nonzero_fail    << ','
          << frac_nonzero         << ','
          << ts.total_fail        << ','
          << mean_fail            << ','
          << ts.max_fail_count    << ',';

        for (int b = 0; b < pspta::PSPTA_FAIL_HIST_BINS; ++b) {
            f << ts.hist[b];
            if (b < pspta::PSPTA_FAIL_HIST_BINS - 1) f << ',';
        }
        f << '\n';
    }
};

} // namespace runtime
} // namespace macroflow3d
