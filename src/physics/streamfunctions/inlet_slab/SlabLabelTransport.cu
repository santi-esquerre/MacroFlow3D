/**
 * @file SlabLabelTransport.cu
 * @brief SF-35 N1: non-template parts of the label-transport constructor (see
 *        SlabLabelTransport.cuh).
 */

#include "SlabLabelTransport.cuh"

#include "NpyIo.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

const char* to_string(TracerBackend b) {
    switch (b) {
    case TracerBackend::host:
        return "host";
    case TracerBackend::gpu:
        return "gpu";
    }
    return "unknown";
}

const char* to_string(LabelTransportStatus s) {
    switch (s) {
    case LabelTransportStatus::ok:
        return "ok";
    case LabelTransportStatus::trace_fail:
        return "trace_fail";
    case LabelTransportStatus::roundtrip_exceeded:
        return "roundtrip_exceeded";
    }
    return "unknown";
}

std::string LabelTransportResult::table() const {
    std::string out;
    char b[512];
    for (const LabelTransportPlane& P : planes) {
        if (P.j == 0)
            continue;
        std::snprintf(b, sizeof(b),
                      "  PLANE j=%3d x1=%.4f nfev=%9lld nfev_rt=%9lld roundtrip=%.3e "
                      "non_ok(back,fwd)=(%lld,%lld) backflow_enc=%lld hist(ok,sb,cap,stag,uflow,"
                      "land,kinv)=(%lld,%lld,%lld,%lld,%lld,%lld,%lld) landings_discarded=%lld "
                      "min_g1_hat=%.6e%s\n",
                      P.j, P.x1, P.nfev, P.nfev_rt, P.roundtrip, P.back_non_ok, P.fwd_non_ok,
                      P.backflow_encounters, P.status_hist[0], P.status_hist[1], P.status_hist[2],
                      P.status_hist[3], P.status_hist[4], P.status_hist[5], P.status_hist[6],
                      P.landings_discarded, P.min_g1_hat, P.flagged ? " FLAGGED" : "");
        out += b;
    }
    std::snprintf(b, sizeof(b),
                  "  TRANSPORT backend=%s tol=%.1e h_max=%.6e threads=%d status=%s "
                  "max_roundtrip=%.3e non_ok=%lld seed_backflow=%lld backflow_enc=%lld "
                  "landings_discarded=%lld min_g1_hat=%.6e t_trace=%.2fs t_labels=%.3fs\n",
                  to_string(backend), tol, h_max_effective, threads, to_string(status),
                  max_roundtrip, non_ok, seed_backflow, backflow_encounters, landings_discarded,
                  min_g1_hat, seconds_trace, seconds_labels);
    out += b;
    return out;
}

nlohmann::json to_json(const LabelTransportResult& r) {
    using nlohmann::json;
    json j;
    j["status"] = to_string(r.status);
    j["backend"] = to_string(r.backend);
    j["tol"] = r.tol;
    j["h_max"] = r.h_max_effective;
    j["threads"] = r.threads;
    j["non_ok"] = r.non_ok;
    j["seed_backflow"] = r.seed_backflow;
    j["backflow_encounters"] = r.backflow_encounters;
    j["landings_discarded"] = r.landings_discarded;
    j["min_g1_hat"] = r.min_g1_hat;
    j["max_roundtrip"] = r.max_roundtrip;
    j["seconds_trace"] = r.seconds_trace;
    j["seconds_labels"] = r.seconds_labels;
    j["status_hist_names"] =
        json::array({"ok", "seed_backflow", "cap_exceeded", "stagnation", "step_underflow",
                     "landing_failed", "invalid_conductivity"});
    long long tot[7] = {0, 0, 0, 0, 0, 0, 0};
    json rt = json::array(), ph = json::array();
    for (const LabelTransportPlane& P : r.planes) {
        if (P.j == 0)
            continue;
        rt.push_back(P.roundtrip);
        json h = json::array();
        for (int s = 0; s < 7; ++s) {
            h.push_back(P.status_hist[s]);
            tot[s] += P.status_hist[s];
        }
        ph.push_back(h);
    }
    j["roundtrip_per_plane"] = rt;
    j["status_hist_per_plane"] = ph;
    j["status_hist"] = json::array({tot[0], tot[1], tot[2], tot[3], tot[4], tot[5], tot[6]});
    return j;
}

LabelTransportResult construct_labels_by_tracing(CudaContext& ctx, const ProductionStage& stage,
                                                 const LabelTransportOptions& opt,
                                                 DeviceSpan<real> psi1, DeviceSpan<real> psi2) {
    if (!stage.ok()) {
        throw std::logic_error("construct_labels_by_tracing: production stage is not ok");
    }
    if (opt.backend == TracerBackend::gpu) {
        throw std::runtime_error("construct_labels_by_tracing: gpu tracer not built (SF-35 N6)");
    }
    const SlabSplineDirectionField fld = stage.direction_field();
    return construct_labels_by_tracing(ctx, stage.grid, fld, stage.labels, opt, psi1, psi2);
}

void save_labels(const std::string& dir, const InletSlabGrid& grid,
                 const std::vector<real>& psi1_host, const std::vector<real>& psi2_host,
                 const nlohmann::json& metadata) {
    require_valid_grid(grid, "save_labels");
    if (psi1_host.size() != grid.full_size() || psi2_host.size() != grid.full_size()) {
        throw std::invalid_argument("save_labels: psi arrays must have (N+1) N^2 entries");
    }
    std::filesystem::create_directories(dir);
    const std::vector<std::size_t> shape = {static_cast<std::size_t>(grid.n) + 1,
                                            static_cast<std::size_t>(grid.n),
                                            static_cast<std::size_t>(grid.n)};
    const std::filesystem::path d(dir);
    write_npy((d / "psi1.npy").string(), shape, psi1_host.data(), psi1_host.size());
    write_npy((d / "psi2.npy").string(), shape, psi2_host.data(), psi2_host.size());
    nlohmann::json m = metadata;
    m["shape"] = {grid.n + 1, grid.n, grid.n};
    m["layout"] = "idx = m3 + N (m2 + N j)";
    m["format"] = "full labels psi1 = x2 + u1, psi2 = x3 + u2";
    std::ofstream f((d / "labels.json").string());
    if (!f) {
        throw std::runtime_error("save_labels: cannot write labels.json in " + dir);
    }
    f << m.dump(2) << "\n";
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
