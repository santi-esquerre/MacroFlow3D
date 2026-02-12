#pragma once

/**
 * @file AnalysisRunner.hpp
 * @brief Offline analysis runner — reads existing CSVs, recomputes α(t).
 *
 * Etapa 10: RunMode::AnalysisOnly entry point.
 *
 * Pure CPU — no CUDA includes or GPU dependency.
 * Reads realization_XXXX_timeseries.csv from the output directory,
 * computes ensemble mean + macrodispersion, writes results.
 */

#include "../../io/config/Config.hpp"

namespace macroflow3d {
namespace analysis_runner {

/**
 * @brief Run offline macrodispersion analysis on existing output CSVs.
 *
 * Requires cfg.output.output_dir to contain stats/realization_XXXX_timeseries.csv
 * files from a previous ensemble run. No GPU is needed.
 *
 * @param cfg  Fully-validated config (only output + analysis sections used).
 * @return EXIT_SUCCESS on completion, EXIT_FAILURE if files missing.
 */
int run_analysis(const io::AppConfig& cfg);

} // namespace analysis_runner
} // namespace macroflow3d
