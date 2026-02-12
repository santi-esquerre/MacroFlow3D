/**
 * @file AnalysisRunner.cpp
 * @brief Offline analysis — reads CSVs, computes macrodispersion.
 *
 * Etapa 10: RunMode::AnalysisOnly
 *
 * Pure CPU — compiles with g++ (no nvcc needed).
 * Uses read_timeseries_csv() from MacrodispersionAnalysis.hpp to
 * load previously-written realization CSVs, then recomputes
 * ensemble mean + macrodispersivity α(t).
 */

#include "AnalysisRunner.hpp"

#include "../../io/output_layout.hpp"
#include "../../io/writers/CsvTimeSeriesWriter.hpp"
#include "../../analysis/macrodispersion/MacrodispersionAnalysis.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>

namespace macroflow3d {
namespace analysis_runner {

int run_analysis(const io::AppConfig& cfg) {
    const auto& mac = cfg.analysis.macrodispersion;
    if (!mac.enabled) {
        std::fprintf(stderr, "ERROR: analysis_only mode requires macrodispersion.enabled = true\n");
        return EXIT_FAILURE;
    }

    const int NR = mac.NR;
    io::OutputLayout layout(cfg.output.output_dir);

    std::printf("[analysis_only] Reading %d realization CSVs from %s\n",
                NR, layout.stats_dir().c_str());

    // ── Read all realization time-series from disk ──────────────────
    std::vector<std::vector<io::TimeSeriesPoint<real>>> all_series;
    all_series.reserve(NR);

    for (int r = 0; r < NR; ++r) {
        std::string fname = layout.realization_timeseries(r);
        if (!std::filesystem::exists(fname)) {
            std::fprintf(stderr, "ERROR: missing %s\n", fname.c_str());
            return EXIT_FAILURE;
        }
        auto pts = analysis::read_timeseries_csv<real>(fname);
        if (pts.empty()) {
            std::fprintf(stderr, "ERROR: empty or unparseable %s\n", fname.c_str());
            return EXIT_FAILURE;
        }
        all_series.push_back(std::move(pts));
        if ((r + 1) % 50 == 0 || r == NR - 1) {
            std::printf("       Loaded %d / %d\n", r + 1, NR);
        }
    }

    // ── Ensure output dirs exist ───────────────────────────────────
    layout.ensure_all_dirs();

    // ── Ensemble mean time-series ──────────────────────────────────
    std::string mean_path = layout.ensemble_timeseries();
    io::CsvTimeSeriesWriter::write_ensemble_mean(mean_path, all_series);
    std::printf("       Wrote %s\n", mean_path.c_str());

    // ── Macrodispersivity α(t) ─────────────────────────────────────
    auto alpha_rows = analysis::compute_macrodispersion(
        all_series, mac.lambda, mac.vmean_norm);

    std::string alpha_path = layout.macrodispersion_csv();
    analysis::write_macrodispersion_csv(alpha_path, alpha_rows);
    std::printf("       Wrote %s\n", alpha_path.c_str());

    std::printf("[analysis_only] Done. NR=%d, %d time-points, output=%s\n",
                NR, (int)alpha_rows.size(), cfg.output.output_dir.c_str());

    return EXIT_SUCCESS;
}

} // namespace analysis_runner
} // namespace macroflow3d
