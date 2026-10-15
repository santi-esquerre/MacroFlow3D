/**
 * @file SlabOracle.cu
 * @brief SF-33 N3: non-template parts of the production oracle (GPU label evaluation at the feet,
 *        report formatting). See SlabOracle.cuh.
 */

#include "SlabOracle.cuh"

#include "../../../core/DeviceBuffer.cuh"
#include "../../../runtime/cuda_check.cuh"

#include <cstdio>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

const char* to_string(SlabOracleStatus s) {
    switch (s) {
    case SlabOracleStatus::ok:
        return "ok";
    case SlabOracleStatus::oracle_roundtrip_fail:
        return "oracle_roundtrip_fail";
    }
    return "unknown";
}

std::string SlabOracleResult::table() const {
    std::string out;
    char b[256];
    for (const SlabOraclePlane& P : planes) {
        if (P.j == 0)
            continue;
        std::snprintf(b, sizeof(b),
                      "  oracle plane j=%3d x1=%.4f nfev=%9lld nfev_rt=%9lld roundtrip=%.3e "
                      "non_ok(back,fwd)=(%lld,%lld) backflow_enc=%lld%s\n",
                      P.j, P.x1, P.nfev, P.nfev_rt, P.roundtrip, P.back_non_ok, P.fwd_non_ok,
                      P.backflow_encounters, P.flagged ? " FLAGGED" : "");
        out += b;
    }
    std::snprintf(b, sizeof(b),
                  "  oracle tol=%.1e status=%s max_roundtrip=%.3e non_ok=%lld t_trace=%.2fs "
                  "t_labels=%.3fs\n",
                  tol, to_string(status), max_roundtrip, non_ok, seconds_trace, seconds_labels);
    out += b;
    return out;
}

namespace detail {

void oracle_evaluate_labels(CudaContext& ctx, const InletSlabGrid& grid, const InletLabels& labels,
                            const std::vector<double>& foot_y, const std::vector<double>& foot_z,
                            DeviceSpan<real> psi1, DeviceSpan<real> psi2) {
    const int N = grid.n;
    const std::size_t n2 = grid.plane_size();
    const std::size_t total = grid.field_size();
    if (foot_y.size() != total || foot_z.size() != total) {
        throw std::invalid_argument("oracle_evaluate_labels: foot arrays must have N^3 entries");
    }
    // points: plane 0 vertices first, then the feet of planes 1..N (full-array order)
    std::vector<real> hy(grid.full_size()), hz(grid.full_size());
    for (int m2 = 0; m2 < N; ++m2)
        for (int m3 = 0; m3 < N; ++m3) {
            const std::size_t p = grid.plane_index(m2, m3);
            hy[p] = grid.coord(m2);
            hz[p] = grid.coord(m3);
        }
    std::copy(foot_y.begin(), foot_y.end(), hy.begin() + static_cast<std::ptrdiff_t>(n2));
    std::copy(foot_z.begin(), foot_z.end(), hz.begin() + static_cast<std::ptrdiff_t>(n2));
    DeviceBuffer<real> dy(hy.size()), dz(hz.size());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(dy.data(), hy.data(), hy.size() * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(dz.data(), hz.data(), hz.size() * sizeof(real), cudaMemcpyHostToDevice));
    // SF-33 C2: legacy-stream copy must land before ctx-stream work
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    labels.evaluate_labels(ctx, DeviceSpan<const real>(dy.data(), dy.size()),
                           DeviceSpan<const real>(dz.data(), dz.size()), psi1, psi2);
    ctx.synchronize();
}

} // namespace detail

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
