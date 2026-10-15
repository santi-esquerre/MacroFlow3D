/**
 * @file slab_trace_gpu_tests.cu
 * @brief SF-35 N6: the GPU tracer (SlabTraceGpu.cuh) reproduces the host tracer (feet and labels
 *        <= 1e-12, spec threshold F) on k = 1, generic3d (eps 0.25) and an SF-18 Gaussian field
 *        (sigma2 1, ell 0.25, seed 3001, eps 0.5), production settings (h/8, 1e-8).
 *
 * Per field and N: (a) feet / labels differences (NaN positions coincide); (b) per-plane
 * nfev, nfev_rt, status_hist, non_ok, backflow_encounters, landings_discarded identical, and the
 * number of seeds whose backward nfev differs from the host trace_to_plane (INFO; the asserted
 * criterion is (a)); (c) two GPU runs bitwise identical; (d) no device allocation inside trace
 * after prepare (cudaMemGetInfo free bytes and storage pointers); (e) max round trip <= 1e-8.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabLabelTransport.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabOracle.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabTraceGpu.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
namespace cg = macroflow3d::closure_gate;

namespace {

struct TestReport {
    bool overall_pass = true;
    void check(bool cond, const std::string& name, const std::string& detail = "") {
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        std::fflush(stdout);
        overall_pass = overall_pass && cond;
    }
};

std::string fmt(const char* f, double v) {
    char b[160];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
bool same_vec(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i]))
            return false;
    return true;
}

struct Run {
    sl::LabelTransportResult res;
    std::vector<real> psi1, psi2;
};

Run run_stage(CudaContext& ctx, const sl::ProductionStage& st, const sl::LabelTransportOptions& opt) {
    DeviceBuffer<real> p1(st.grid.full_size()), p2(st.grid.full_size());
    Run r;
    r.res = sl::construct_labels_by_tracing(ctx, st, opt, DeviceSpan<real>(p1.data(), p1.size()),
                                            DeviceSpan<real>(p2.data(), p2.size()));
    r.psi1 = download(p1.data(), p1.size());
    r.psi2 = download(p2.data(), p2.size());
    return r;
}

/// max |a - b| over positions finite in both; false in `nan_match` if NaN positions differ.
double max_diff(const std::vector<double>& a, const std::vector<double>& b, bool& nan_match) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const bool na = std::isnan(a[i]), nb = std::isnan(b[i]);
        if (na != nb) {
            nan_match = false;
            continue;
        }
        if (!na)
            m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

void run_case(TestReport& rep, CudaContext& ctx, const std::string& name,
              const sl::SlabFieldSource& src, double eps, int N) {
    const std::string tag = name + " N=" + std::to_string(N) + ": ";
    const sl::InletSlabGrid grid = sl::InletSlabGrid::make(N);
    sl::ProductionStage st = sl::build_production_stage(ctx, grid, src, eps);
    rep.check(st.ok(), tag + "production stage ok", sl::to_string(st.report.status));
    if (!st.ok())
        return;

    sl::LabelTransportOptions oh;
    oh.threads = 4;
    oh.backend = sl::TracerBackend::host;
    sl::LabelTransportOptions og = oh;
    og.backend = sl::TracerBackend::gpu;
    const Run H = run_stage(ctx, st, oh);
    const Run G1 = run_stage(ctx, st, og);
    const Run G2 = run_stage(ctx, st, og);
    std::printf("  %stimings: host t_trace=%.3fs  gpu t_trace=%.3fs (kernel + copies)  gpu "
                "t_labels=%.3fs  [threads host %d, gpu %d]\n",
                tag.c_str(), H.res.seconds_trace, G1.res.seconds_trace, G1.res.seconds_labels,
                H.res.threads, G1.res.threads);

    // (a)
    bool nan_ok = true;
    const double dfy = max_diff(H.res.foot_y, G1.res.foot_y, nan_ok);
    const double dfz = max_diff(H.res.foot_z, G1.res.foot_z, nan_ok);
    const double dp1 = max_diff(H.psi1, G1.psi1, nan_ok);
    const double dp2 = max_diff(H.psi2, G1.psi2, nan_ok);
    const double drt = max_diff(H.res.roundtrip, G1.res.roundtrip, nan_ok);
    rep.check(nan_ok && std::max(dfy, dfz) <= 1e-12,
              tag + "feet: max |dfoot_y|, |dfoot_z| <= 1e-12 (NaN positions coincide)",
              fmt("dfoot_y %.3e", dfy) + fmt(" dfoot_z %.3e", dfz));
    rep.check(std::max(dp1, dp2) <= 1e-12, tag + "labels: max |dpsi1|, |dpsi2| <= 1e-12 (all vertices)",
              fmt("dpsi1 %.3e", dp1) + fmt(" dpsi2 %.3e", dp2));
    std::printf("  %sINFO max |d roundtrip| = %.3e (host %.3e, gpu %.3e)\n", tag.c_str(), drt,
                H.res.max_roundtrip, G1.res.max_roundtrip);

    // (b) per-plane accounting
    bool plane_same = H.res.planes.size() == G1.res.planes.size();
    int planes_diff = 0;
    for (std::size_t p = 0; plane_same && p < H.res.planes.size(); ++p) {
        const sl::LabelTransportPlane &a = H.res.planes[p], &b = G1.res.planes[p];
        bool same = a.nfev == b.nfev && a.nfev_rt == b.nfev_rt && a.non_ok == b.non_ok &&
                    a.backflow_encounters == b.backflow_encounters &&
                    a.landings_discarded == b.landings_discarded &&
                    a.back_non_ok == b.back_non_ok && a.fwd_non_ok == b.fwd_non_ok;
        for (int s = 0; s < 7; ++s)
            same = same && a.status_hist[s] == b.status_hist[s];
        if (!same)
            ++planes_diff;
    }
    const bool stage_same = H.res.non_ok == G1.res.non_ok &&
                            H.res.landings_discarded == G1.res.landings_discarded &&
                            H.res.seed_backflow == G1.res.seed_backflow;
    rep.check(plane_same && planes_diff == 0 && stage_same,
              tag + "per-plane nfev, nfev_rt, status_hist, non_ok, backflow_encounters, "
                    "landings_discarded identical to host",
              "planes differing: " + std::to_string(planes_diff));

    // seeds whose step sequence (backward nfev) differs: INFO
    {
        sl::SlabGpuTracer tr;
        tr.prepare(ctx, st);
        cg::IntegratorOptions base;
        base.tol = 1e-8;
        base.h_max = grid.h / 8.0;
        base.min_step = 1e-13;
        base.max_arclength_per_period = 100.0;
        base.n_periods = 1;
        sl::SlabGpuTraceOutputs T;
        tr.trace(ctx, grid, base, T);
        const sl::SlabSplineDirectionField fld = st.direction_field();
        const std::size_t n2 = grid.plane_size();
        long long ndiff_b = 0, ndiff_f = 0;
        for (std::size_t i = 0; i < grid.field_size(); ++i) {
            const int j = static_cast<int>(i / n2) + 1;
            const std::size_t rem = i % n2;
            const int m2 = static_cast<int>(rem / static_cast<std::size_t>(N));
            const int m3 = static_cast<int>(rem % static_cast<std::size_t>(N));
            const std::array<double, 3> seed = {grid.coord(j), grid.coord(m2), grid.coord(m3)};
            cg::IntegratorOptions ob = base;
            ob.sigma = -1;
            const cg::StreamlineResult rb = sl::trace_to_plane(fld, seed, 0.0, ob);
            if (rb.field_evaluations != T.nfev_b[i])
                ++ndiff_b;
            if (rb.status == cg::StreamlineStatus::ok) {
                cg::IntegratorOptions of = base;
                of.sigma = +1;
                const cg::StreamlineResult rf = sl::trace_to_plane(
                    fld, {0.0, rb.records[0].x2, rb.records[0].x3}, seed[0], of);
                if (rf.field_evaluations != T.nfev_f[i])
                    ++ndiff_f;
            }
        }
        std::printf("  %sINFO seeds with different nfev (step sequence): backward %lld, forward "
                    "%lld of %zu\n",
                    tag.c_str(), ndiff_b, ndiff_f, grid.field_size());

        // (d) no allocation inside trace
        std::vector<const void*> before;
        for (int k = 0; k < 11; ++k)
            before.push_back(tr.storage_ptr(k));
        std::size_t free0 = 0, free1 = 0, freeb = 0, total = 0;
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free0, &total));
        tr.trace(ctx, grid, base, T);
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free1, &total));
        tr.trace(ctx, grid, base, T);
        MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&freeb, &total));
        std::printf("  %sINFO free bytes: before %zu, after 1st %zu, after 2nd %zu\n", tag.c_str(),
                    free0, free1, freeb);
        bool ptr_same = true;
        for (int k = 0; k < 11; ++k)
            ptr_same = ptr_same && before[static_cast<std::size_t>(k)] == tr.storage_ptr(k);
        rep.check(free0 == free1 && free1 == freeb && ptr_same, tag + "no device allocation inside trace() after prepare",
                  "free bytes before " + std::to_string(free0) + ", after " + std::to_string(free1));
    }

    // (c)
    rep.check(same_vec(G1.psi1, G2.psi1) && same_vec(G1.psi2, G2.psi2) &&
                  same_vec(G1.res.foot_y, G2.res.foot_y) && same_vec(G1.res.foot_z, G2.res.foot_z) &&
                  same_vec(G1.res.roundtrip, G2.res.roundtrip),
              tag + "two consecutive GPU runs bitwise identical (feet, round trips, labels)");
    // (e) status ok / <= 1e-8 where the HOST tracer attains it (the round trip at (h/8, 1e-8) is a
    // property of the field; the GPU result must equal the host one: see (a))
    const bool host_ok = H.res.status == sl::LabelTransportStatus::ok && H.res.max_roundtrip <= 1e-8;
    rep.check(G1.res.status == H.res.status &&
                  std::abs(G1.res.max_roundtrip - H.res.max_roundtrip) <= 1e-12,
              tag + "GPU status and max round trip equal the host's",
              std::string(sl::to_string(G1.res.status)) + fmt(" gpu %.3e", G1.res.max_roundtrip) +
                  fmt(" host %.3e", H.res.max_roundtrip));
    if (host_ok) {
        rep.check(G1.res.status == sl::LabelTransportStatus::ok && G1.res.max_roundtrip <= 1e-8,
                  tag + "GPU status ok, max round trip <= 1e-8", fmt("%.3e", G1.res.max_roundtrip));
    } else {
        std::printf("  %sINFO host round trip %.3e > 1e-8 at (h/8, 1e-8): criterion (e) not "
                    "applicable to this field/N\n",
                    tag.c_str(), H.res.max_roundtrip);
    }
}

} // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "quick"; // skip N = 32
    TestReport rep;
    CudaContext ctx(0);
    const auto t0 = std::chrono::steady_clock::now();
    auto lap = [&](const char* what) {
        std::printf("  [time] %s: %.1fs elapsed\n", what,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        std::fflush(stdout);
    };

    {
        const int N = 16;
        const sl::SlabFieldSource src = sl::SlabFieldSource::spectral(
            ctx, "uniform", N, std::vector<real>(static_cast<std::size_t>(N) * N * N, 0.0));
        run_case(rep, ctx, "k = 1", src, 1.0, N);
        lap("k = 1 N=16");
    }
    {
        const sl::SlabFieldSource src = sl::SlabFieldSource::analytic("generic3d", 16);
        run_case(rep, ctx, "generic3d eps 0.25", src, 0.25, 16);
        lap("generic3d N=16");
    }
    {
        sl::ProductionFieldSpec spec;
        spec.N = 16;
        spec.sigma2 = 1.0;
        spec.ell = 0.25;
        spec.seed = 3001ULL;
        const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
        run_case(rep, ctx, "gaussian s2=1 ell=0.25 seed=3001 eps 0.5", src, 0.5, 16);
        lap("gaussian N=16");
    }
    if (!quick) {
        sl::ProductionFieldSpec spec;
        spec.N = 32;
        spec.sigma2 = 1.0;
        spec.ell = 0.25;
        spec.seed = 3001ULL;
        const sl::SlabFieldSource src = sl::SlabFieldSource::gaussian(ctx, spec);
        run_case(rep, ctx, "gaussian s2=1 ell=0.25 seed=3001 eps 0.5", src, 0.5, 32);
        lap("gaussian N=32");
    }

    std::printf("\n%s\n", rep.overall_pass ? "ALL PASS" : "SOME FAILED");
    return rep.overall_pass ? 0 : 1;
}
