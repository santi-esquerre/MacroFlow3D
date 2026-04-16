#pragma once

/**
 * @file stats_types.hpp
 * @brief Par2-agnostic stats sample struct.
 *
 * Only the StatsSample<T> POD lives here — used internally by Par2StatsAdapter.
 * Post-processing (macrodispersion, ensemble CSV) has moved to:
 *   - src/analysis/macrodispersion/MacrodispersionAnalysis.hpp
 *   - src/io/writers/CsvTimeSeriesWriter.hpp
 */

#include "../../core/Scalar.hpp"

namespace macroflow3d {

template <typename T> struct StatsSample {
    T time;
    T mean[3]; // mean_x, mean_y, mean_z
    T var[3];  // var_x, var_y, var_z  (biased or unbiased per config)
    int active;
};

} // namespace macroflow3d
