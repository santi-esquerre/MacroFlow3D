#pragma once

/**
 * @file closure_statistics.hpp
 * @brief SF-30 closure-gate displacement statistics (header-only, host-only
 *        C++17).
 *
 * For one-period return displacements `d_i = (d2_i, d3_i)` with optional
 * non-negative weights `w_i` (unweighted = all weights 1):
 *
 *   mean   = sum_i w_i d_i / W,                         W = sum_i w_i
 *   R      = sqrt( sum_i w_i |d_i - mean|^2 / W )       (non-uniform RMS displacement;
 *                                                         the decision statistic)
 *   rms    = sqrt( sum_i w_i |d_i|^2 / W ),   max = max_i |d_i| (over w_i > 0)
 *   var_d2 = sum_i w_i (d2_i - mean_2)^2 / W, var_d3 likewise (population,
 *            about the weighted mean)
 *
 * so that `R^2 = var_d2 + var_d3`.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace closure_gate {

struct DisplacementStatistics {
    std::size_t count = 0;
    double weight_sum = 0.0;
    double mean[2] = {0.0, 0.0};
    double R = 0.0;
    double rms_abs = 0.0;
    double max_abs = 0.0;
    double var_d2 = 0.0;
    double var_d3 = 0.0;
};

inline DisplacementStatistics displacement_statistics(const std::vector<std::array<double, 2>>& d,
                                                      const std::vector<double>& w) {
    if (d.empty()) {
        throw std::invalid_argument("closure_gate: displacement_statistics requires >= 1 point");
    }
    if (w.size() != d.size()) {
        throw std::invalid_argument("closure_gate: displacement_statistics weight size mismatch");
    }
    DisplacementStatistics st;
    st.count = d.size();
    double W = 0.0, m2 = 0.0, m3 = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        if (!(w[i] >= 0.0) || !std::isfinite(w[i])) {
            throw std::invalid_argument(
                "closure_gate: displacement_statistics weights must be finite and >= 0");
        }
        W += w[i];
        m2 += w[i] * d[i][0];
        m3 += w[i] * d[i][1];
    }
    if (!(W > 0.0)) {
        throw std::invalid_argument("closure_gate: displacement_statistics weight sum must be > 0");
    }
    m2 /= W;
    m3 /= W;
    double v2 = 0.0, v3 = 0.0, sq = 0.0, mx = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        const double a = d[i][0] - m2;
        const double b = d[i][1] - m3;
        v2 += w[i] * a * a;
        v3 += w[i] * b * b;
        const double n2 = d[i][0] * d[i][0] + d[i][1] * d[i][1];
        sq += w[i] * n2;
        if (w[i] > 0.0) {
            mx = std::max(mx, std::sqrt(n2));
        }
    }
    st.weight_sum = W;
    st.mean[0] = m2;
    st.mean[1] = m3;
    st.var_d2 = v2 / W;
    st.var_d3 = v3 / W;
    st.R = std::sqrt((v2 + v3) / W);
    st.rms_abs = std::sqrt(sq / W);
    st.max_abs = mx;
    return st;
}

inline DisplacementStatistics displacement_statistics(const std::vector<std::array<double, 2>>& d) {
    return displacement_statistics(d, std::vector<double>(d.size(), 1.0));
}

} // namespace closure_gate
} // namespace macroflow3d
