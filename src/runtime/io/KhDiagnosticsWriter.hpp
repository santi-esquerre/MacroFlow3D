#pragma once

/**
 * @file KhDiagnosticsWriter.hpp
 * @brief CSV writers for KH reconstruction experiment diagnostics.
 *
 * These writers are deliberately transport-backend neutral and do not include
 * PSPTA headers. They append rows and write headers once.
 */

#include "../../core/Scalar.hpp"
#include "../../io/writers/CsvTimeSeriesWriter.hpp"
#include "../../physics/flow/velocity_diagnostics.cuh"

#include <fstream>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace runtime {

namespace kh_detail {

inline std::ofstream open_append(const std::string& path) {
    std::ofstream f(path, std::ios::app);
    if (!f.is_open())
        throw std::runtime_error("[KhDiagnosticsWriter] Cannot open: " + path);
    return f;
}

inline bool file_has_content(const std::string& path) {
    std::ifstream f(path, std::ios::ate);
    return f.is_open() && (f.tellg() > 0);
}

} // namespace kh_detail

class KhDiagnosticsWriter {
  public:
    static void write_field_row(const std::string& path,
                                const physics::VelocityEvalDiagnosticsSummary& s) {
        const bool write_header = !kh_detail::file_has_content(path);
        auto f = kh_detail::open_append(path);
        if (write_header) {
            f << "realization_id,backend,n_samples,sample_stride,finite_count,invalid_count,"
                 "speed_mean,speed_max,div_abs_mean,div_abs_max,curl_mag_mean,curl_mag_max,"
                 "helicity_mean,helicity_abs_mean,helicity_abs_max,helicity_norm_mean,"
                 "helicity_norm_std,helicity_norm_p50,helicity_norm_p95,helicity_norm_max,"
                 "k_interp_min,k_interp_max,k_interp_mean,k_interp_nonpositive_count,"
                 "k_interp_clamped_count,logk_interp_min,logk_interp_max\n";
        }
        f << s.realization_id << ',' << s.backend << ',' << s.n_samples << ',' << s.sample_stride
          << ',' << s.finite_count << ',' << s.invalid_count << ',' << s.speed_mean << ','
          << s.speed_max << ',' << s.div_abs_mean << ',' << s.div_abs_max << ',' << s.curl_mag_mean
          << ',' << s.curl_mag_max << ',' << s.helicity_mean << ',' << s.helicity_abs_mean << ','
          << s.helicity_abs_max << ',' << s.helicity_norm_mean << ',' << s.helicity_norm_std << ','
          << s.helicity_norm_p50 << ',' << s.helicity_norm_p95 << ',' << s.helicity_norm_max << ','
          << s.k_interp_min << ',' << s.k_interp_max << ',' << s.k_interp_mean << ','
          << s.k_interp_nonpositive_count << ',' << s.k_interp_clamped_count << ','
          << s.logk_interp_min << ',' << s.logk_interp_max << '\n';
    }

    static void write_comparison_row(const std::string& path,
                                     const physics::VelocityBackendComparisonSummary& s) {
        const bool write_header = !kh_detail::file_has_content(path);
        auto f = kh_detail::open_append(path);
        if (write_header) {
            f << "realization_id,backend,n_samples,sample_stride,finite_count,invalid_count,"
                 "rel_l2_diff,diff_mean,diff_std,diff_p50,diff_p95,diff_max,"
                 "rel_diff_mean,rel_diff_p95,rel_diff_max,"
                 "vector_correlation\n";
        }
        f << s.realization_id << ',' << s.backend << ',' << s.n_samples << ',' << s.sample_stride
          << ',' << s.finite_count << ',' << s.invalid_count << ',' << s.rel_l2_diff << ','
          << s.diff_mean << ',' << s.diff_std << ',' << s.diff_p50 << ',' << s.diff_p95 << ','
          << s.diff_max << ',' << s.rel_diff_mean << ',' << s.rel_diff_p95 << ',' << s.rel_diff_max
          << ',' << s.vector_correlation << '\n';
    }

    static void write_transport_row(const std::string& path, int realization_id,
                                    const std::string& backend, int n_particles,
                                    const io::TimeSeriesPoint<real>* final_sample) {
        const bool write_header = !kh_detail::file_has_content(path);
        auto f = kh_detail::open_append(path);
        if (write_header) {
            f << "realization_id,backend,n_particles,active,problematic,"
                 "final_time,var_x,var_y,var_z\n";
        }
        const int active = final_sample ? final_sample->active : n_particles;
        const int problematic = n_particles - active;
        f << realization_id << ',' << backend << ',' << n_particles << ',' << active << ','
          << problematic << ',';
        if (final_sample) {
            f << final_sample->time << ',' << final_sample->var[0] << ',' << final_sample->var[1]
              << ',' << final_sample->var[2] << '\n';
        } else {
            f << "nan,nan,nan,nan\n";
        }
    }

    static void write_runtime_row(const std::string& path, int realization_id,
                                  const std::string& backend, double transport_seconds,
                                  int n_particles, int n_steps) {
        const bool write_header = !kh_detail::file_has_content(path);
        auto f = kh_detail::open_append(path);
        if (write_header) {
            f << "realization_id,backend,transport_seconds,n_particles,n_steps,"
                 "particle_steps,particle_steps_per_second\n";
        }
        const double particle_steps =
            static_cast<double>(n_particles) * static_cast<double>(n_steps);
        const double rate = transport_seconds > 0.0 ? particle_steps / transport_seconds : 0.0;
        f << realization_id << ',' << backend << ',' << transport_seconds << ',' << n_particles
          << ',' << n_steps << ',' << particle_steps << ',' << rate << '\n';
    }
};

} // namespace runtime
} // namespace macroflow3d
