/**
 * @file LongBoxStage.cu
 * @brief SF-40 L3: lean parametric Grid3D stage of the Lagrangian-label path. See LongBoxStage.cuh.
 *
 * Host orchestration only (no device code of its own): every kernel and library call used here is
 * the one of the current stage (SF-18 generator, SF-28 prefilter, SF-19 lean solver, InletLabels).
 * Includes only LagrangianTracker.cuh among the tracker headers (the tracker templates and their
 * --fmad=false instantiations live in LagrangianTracker.cu).
 */

#include "LongBoxStage.cuh"

#include "../../../numerics/blas/sum.cuh"
#include "../../../numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "../../../runtime/cuda_check.cuh"
#include "../../flow/AffinePeriodicFlowSolver.cuh"
#include "../../stochastic/PeriodicGaussianField.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <unistd.h>

namespace macroflow3d {
namespace lagrangian_labels {

namespace sl = streamfunctions::inlet_slab;

int box_auto_mg_levels(int nx, int ny, int nz) {
    const auto can_coarsen = [](int n) { return n % 2 == 0 && n / 2 >= 4 && (n / 2) % 2 == 0; };
    int levels = 1;
    int a = nx, b = ny, c = nz;
    while (can_coarsen(a) && can_coarsen(b) && can_coarsen(c)) {
        a /= 2;
        b /= 2;
        c /= 2;
        ++levels;
    }
    return levels;
}

namespace {

using Clock = std::chrono::steady_clock;

double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::size_t device_used_now() {
    std::size_t free_b = 0, total_b = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    return total_b - free_b;
}

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    if (n > 0)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

/// C-SYNC: legacy-stream upload followed by a device-wide synchronization.
template <class T> void upload_sync(T* dst, const T* src, std::size_t n) {
    if (n == 0)
        return;
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
}

template <class F> void in_phase(const char* phase, F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        std::string m = e.what();
        if (m.find("phase=") == std::string::npos)
            m += std::string(" (phase=") + phase + ")";
        throw std::runtime_error("LongBoxStage: " + m);
    }
}

void release(std::vector<real>& v) {
    std::vector<real>().swap(v);
}

} // namespace

DeviceSplineFlow LongBoxStage::device_flow() const {
    DeviceSplineFlow f;
    f.grid = grid;
    if (fused_coefficients) {
        f.coef_interleaved = coef_interleaved.data();
    } else {
        f.coef_pot = coef_pot.data();
        f.coef_logk = coef_logk.data();
    }
    for (int d = 0; d < 3; ++d)
        f.G[d] = G[d];
    f.labels = (labels_available && labels.device_prepared()) ? &labels : nullptr;
    return f;
}

LongBoxStage build_long_box_stage(CudaContext& ctx, const LongBoxSpec& spec) {
    const int nx = spec.nx, ny = spec.ny, nz = spec.nz;
    if (nx < 4 || ny < 4 || nz < 4 || nx % 2 != 0 || ny % 2 != 0 || nz % 2 != 0) {
        throw std::invalid_argument("build_long_box_stage: nx, ny, nz must be even and >= 4");
    }
    if (ny != nz) {
        throw std::invalid_argument("build_long_box_stage: ny must equal nz (InletLabels takes one "
                                    "transverse N; h = 1/ny)");
    }
    if (!std::isfinite(spec.eps)) {
        throw std::invalid_argument("build_long_box_stage: eps must be finite");
    }
    const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
    if (spec.source == LongBoxSourceKind::cells && spec.cells.size() != n) {
        throw std::invalid_argument("build_long_box_stage: cells.size() != nx*ny*nz");
    }
    if (spec.label_chunk < 1) {
        throw std::invalid_argument("build_long_box_stage: label_chunk must be >= 1");
    }

    LongBoxStage S;
    const real h = 1.0 / static_cast<real>(ny);
    S.grid = Grid3D(nx, ny, nz, h, h, h);
    S.fused_coefficients = spec.fused_coefficients;
    S.device_used_start = device_used_now();
    const int levels = spec.mg_levels == 0 ? box_auto_mg_levels(nx, ny, nz) : spec.mg_levels;

    sl::ProductionStageReport& R = S.report;
    R.field = !spec.field_name.empty()
                  ? spec.field_name
                  : (spec.source == LongBoxSourceKind::gaussian ? "gaussian" : "cells");
    R.eps = spec.eps;
    R.N = nx;
    R.mg_levels = levels;

    // ---- MEM_GEOM
    {
        LongBoxGeometry& g = S.geometry;
        g.pid = static_cast<long long>(::getpid());
        g.nx = nx;
        g.ny = ny;
        g.nz = nz;
        g.n = n;
        g.levels = levels;
        for (int l = 0; l < levels; ++l) {
            const std::size_t nl = static_cast<std::size_t>(nx >> l) *
                                   static_cast<std::size_t>(ny >> l) *
                                   static_cast<std::size_t>(nz >> l);
            g.level_cells.push_back(nl);
            g.cub_sum_bytes.push_back(blas::sum_workspace_bytes(nl));
        }
        g.coarsest[0] = nx >> (levels - 1);
        g.coarsest[1] = ny >> (levels - 1);
        g.coarsest[2] = nz >> (levels - 1);
        const std::size_t nu = static_cast<std::size_t>(nx + 1) * ny * nz;
        const std::size_t nv = static_cast<std::size_t>(nx) * (ny + 1) * nz;
        const std::size_t nw = static_cast<std::size_t>(nx) * ny * (nz + 1);
        g.face_max = std::max(nu, std::max(nv, nw));
        g.bhat_mapped = spec.bhat_mapped_host;
        g.bhat_host_bytes = spec.bhat_mapped_host ? n * sizeof(real) : 0;
    }

    // ---- F: SF-18 field (gaussian) -> host Y ---------------------------------------------------
    std::vector<real> Y;
    {
        LongBoxPhaseRecord& rec = S.phase_F;
        rec.phase = "F";
        rec.present = true;
        const auto t0 = Clock::now();
        if (spec.source == LongBoxSourceKind::gaussian) {
            in_phase("F", [&] {
                // exactly SlabFieldSource::gaussian's SF-18 call
                physics::PeriodicGaussianFieldConfig cfg;
                cfg.sigma2 = spec.sigma2;
                cfg.corr_length = spec.ell;
                cfg.seed = spec.seed;
                cfg.normalize_variance = spec.normalize_variance;
                DeviceBuffer<real> y(n);
                physics::PeriodicGaussianFieldWorkspace ws;
                R.sf18 = physics::generate_periodic_gaussian_field(ctx, S.grid, cfg, y.span(), ws);
                R.has_sf18 = true;
                ctx.synchronize();
                Y = download(y.data(), n);
                rec.items.emplace_back("y", y.capacity() * sizeof(real));
                rec.items.emplace_back("sf18_spectrum",
                                       ws.spectrum.capacity() * sizeof(cufftDoubleComplex));
                rec.items.emplace_back("sf18_mode_counter",
                                       ws.mode_counter.capacity() * sizeof(unsigned long long));
                rec.items.emplace_back("sf18_reduce_blocks",
                                       ws.reduce_blocks.capacity() * sizeof(real));
                rec.cufft_work = R.sf18.cufft_work_area_bytes;
                rec.cufft_named.emplace_back("cufft_gen", R.sf18.cufft_work_area_bytes);
                rec.device_used = device_used_now();
            });
        } else {
            Y = spec.cells;
            rec.device_used = S.device_used_start;
        }
        rec.seconds = since(t0);
        if (spec.keep_host_field)
            S.host_field = Y;
    }

    // ---- S1: lnK = eps Y, K = exp(lnK) on the host; prefilter of lnK -> host coefficients -------
    std::vector<real> K(n), coef_logk_host;
    {
        LongBoxPhaseRecord& rec = S.phase_S1;
        rec.phase = "S1";
        rec.present = true;
        const auto t0 = Clock::now();
        in_phase("S1", [&] {
            std::vector<real> lnK(n);
            for (std::size_t c = 0; c < n; ++c) { // build_production_stage :478-482
                lnK[c] = spec.eps * Y[c];
                K[c] = std::exp(lnK[c]);
            }
            release(Y);
            DeviceBuffer<real> lnk_dev(n);
            upload_sync(lnk_dev.data(), lnK.data(), n);
            release(lnK);
            interpolation::PeriodicTricubicBSplineWorkspace ws;
            const auto prep = interpolation::prefilter_periodic_tricubic_bspline(
                ctx, S.grid, DeviceSpan<const real>(lnk_dev.data(), n), ws);
            ctx.synchronize();
            coef_logk_host = download(ws.coefficients.data(), n);
            rec.items.emplace_back("lnk", lnk_dev.capacity() * sizeof(real));
            rec.items.emplace_back("coef_logk", ws.coefficients.capacity() * sizeof(real));
            rec.items.emplace_back("prefilter_spectrum",
                                   ws.spectrum.capacity() * sizeof(cufftDoubleComplex));
            rec.cufft_work = prep.cufft_work_area_bytes;
            rec.cufft_named.emplace_back("cufft_prefilter_logk", prep.cufft_work_area_bytes);
            rec.device_used = device_used_now();
        });
        rec.seconds = since(t0);
    }

    // ---- S2: lean SF-19 (S+), G, inlet samples, D-1 labels -------------------------------------
    physics::AffinePeriodicFlowLeanResult flow;
    {
        LongBoxPhaseRecord& rec = S.phase_S2;
        rec.phase = "S2";
        rec.present = true;
        const auto t0 = Clock::now();
        in_phase("S2", [&] {
            physics::AffinePeriodicFlowConfig fcfg;
            fcfg.qbar[0] = 1.0;
            fcfg.qbar[1] = 0.0;
            fcfg.qbar[2] = 0.0;
            fcfg.linear.rtol = spec.pcg_rtol;
            if (spec.pcg_max_iter != -1)
                fcfg.linear.max_iter = spec.pcg_max_iter;
            fcfg.mg.num_levels = levels;
            physics::AffinePeriodicFlowLeanOptions lo;
            lo.bhat_mapped_host = spec.bhat_mapped_host;
            lo.keep_face_fluxes_host = spec.keep_face_fluxes;
            flow = physics::solve_affine_periodic_flow_lean(ctx, S.grid, K, fcfg, lo);
            ctx.synchronize();
            R.darcy = flow.report;
            R.darcy_converged = R.darcy.corrector_results[0].converged &&
                                R.darcy.corrector_results[1].converged &&
                                R.darcy.corrector_results[2].converged;
            rec.items = {};
            for (const auto& it : flow.memory.items)
                rec.items.emplace_back(it.name, it.bytes);
            rec.cufft_work = 0;
            rec.device_used =
                std::max(flow.memory.device_used_at_peak, flow.memory.device_used_after_solve);
            const auto& c = flow.memory.categories;
            char b[512];
            std::snprintf(b, sizeof(b),
                          "correctors:%zu,rhs_scratch:%zu,flux_scratch:%zu,reduction_scratch:%zu,"
                          "affine_rhs_workspace:%zu,pcg_workspace:%zu,mg_hierarchy:%zu,"
                          "mg_preconditioner:%zu",
                          c.correctors_bytes, c.rhs_scratch_bytes, c.flux_scratch_bytes,
                          c.reduction_scratch_bytes, c.affine_rhs_workspace_bytes,
                          c.pcg_workspace_bytes, c.mg_hierarchy_bytes, c.mg_preconditioner_bytes);
            rec.categories = b;
            S.geometry.bhat_host_bytes = flow.memory.bhat_host_bytes;
        });
        rec.seconds = since(t0);
        release(K);
    }
    if (!R.darcy_converged) {
        R.status = sl::SlabProductionStatus::darcy_failed;
        return S;
    }
    for (int d = 0; d < 3; ++d)
        S.G[d] = R.darcy.G[d];
    if (spec.keep_face_fluxes) {
        S.face_flux_u = std::move(flow.face_u);
        S.face_flux_v = std::move(flow.face_v);
        S.face_flux_w = std::move(flow.face_w);
    }
    bool backflow = false;
    in_phase("S2", [&] {
        S.inlet_v1_samples = std::move(flow.inlet_u_plane);
        const std::size_t N = static_cast<std::size_t>(ny);
        real sum = 0.0;
        for (std::size_t j = 0; j < N; ++j) // loop order of build_production_stage :529-537
            for (std::size_t k = 0; k < N; ++k)
                sum += S.inlet_v1_samples[k + N * j];
        R.inlet_mean = sum / static_cast<real>(N * N);
        try {
            S.labels = sl::InletLabels::build(S.inlet_v1_samples, ny, 0.5, 0.5);
        } catch (const sl::InletBackflowError& e) {
            R.inlet_vmin = e.vmin();
            R.status = sl::SlabProductionStatus::inlet_backflow;
            backflow = true; // as require_inlet_labels = false: record and continue
        }
        S.labels_available = !backflow;
        if (!backflow) {
            R.inlet_vmin = S.labels.vmin();
            R.Q0 = S.labels.Q0();
        }
    });

    // ---- S3: coef_logk to the device, label tables, prefilter of h_tilde -> coef_pot -----------
    {
        LongBoxPhaseRecord& rec = S.phase_S3;
        rec.phase = "S3";
        rec.present = true;
        const auto t0 = Clock::now();
        in_phase("S3", [&] {
            DeviceBuffer<real> logk_dev(n);
            upload_sync(logk_dev.data(), coef_logk_host.data(), n);
            release(coef_logk_host);
            if (S.labels_available)
                S.labels.prepare_device(spec.label_chunk);
            interpolation::PeriodicTricubicBSplineWorkspace ws;
            const auto prep = interpolation::prefilter_periodic_tricubic_bspline(
                ctx, S.grid, DeviceSpan<const real>(flow.h_tilde.data(), n), ws);
            ctx.synchronize();
            rec.items.emplace_back("h_tilde", flow.h_tilde.capacity() * sizeof(real));
            rec.items.emplace_back("coef_logk", logk_dev.capacity() * sizeof(real));
            if (S.labels_available && S.labels.device_prepared())
                rec.items.emplace_back("labels_device", S.labels.device_bytes());
            rec.items.emplace_back("coef_pot", ws.coefficients.capacity() * sizeof(real));
            rec.items.emplace_back("prefilter_spectrum",
                                   ws.spectrum.capacity() * sizeof(cufftDoubleComplex));
            rec.cufft_work = prep.cufft_work_area_bytes;
            rec.cufft_named.emplace_back("cufft_prefilter_pot", prep.cufft_work_area_bytes);
            rec.device_used = device_used_now();

            S.coef_pot = std::move(ws.coefficients);
            ws.spectrum.reset();
            flow.h_tilde.reset();
            S.coef_logk = std::move(logk_dev);

            if (spec.fused_coefficients) {
                // host interleave of {pot, logk} (pure data movement), then one device array
                std::vector<real> hp = download(S.coef_pot.data(), n);
                std::vector<real> hl = download(S.coef_logk.data(), n);
                S.coef_pot.reset();
                S.coef_logk.reset();
                std::vector<double2> inter(n);
                for (std::size_t i = 0; i < n; ++i)
                    inter[i] = make_double2(hp[i], hl[i]);
                release(hp);
                release(hl);
                S.coef_interleaved = DeviceBuffer<double2>(n);
                upload_sync(S.coef_interleaved.data(), inter.data(), n);
            }
        });
        rec.seconds = since(t0);
    }
    R.status = backflow ? sl::SlabProductionStatus::inlet_backflow : sl::SlabProductionStatus::ok;
    return S;
}

namespace {

std::string fmt_items(const std::vector<std::pair<std::string, std::size_t>>& items) {
    if (items.empty())
        return "-";
    std::string s;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i > 0)
            s += ",";
        s += items[i].first + ":" + std::to_string(items[i].second);
    }
    return s;
}

std::string phase_line(const LongBoxPhaseRecord& r) {
    std::string s = "MEM_PHASE phase=" + r.phase + " exact_bytes=" + std::to_string(r.exact_bytes()) +
                    " cufft_work=" + std::to_string(r.cufft_work);
    for (const auto& w : r.cufft_named)
        s += " " + w.first + "=" + std::to_string(w.second);
    s += " device_used=" + std::to_string(r.device_used);
    char b[64];
    std::snprintf(b, sizeof(b), " seconds=%.6f", r.seconds);
    s += b;
    if (!r.categories.empty())
        s += " categories=" + r.categories;
    s += " items=" + fmt_items(r.items);
    return s;
}

std::string join_sizes(const std::vector<std::size_t>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0)
            s += ",";
        s += std::to_string(v[i]);
    }
    return s;
}

} // namespace

std::vector<std::string> mem_lines(const LongBoxStage& stage, const LagrangianTracker* tracker,
                                   double tracker_prepare_seconds) {
    std::vector<std::string> out;
    const LongBoxGeometry& g = stage.geometry;
    out.push_back("MEM_GEOM pid=" + std::to_string(g.pid) + " nx=" + std::to_string(g.nx) +
                  " ny=" + std::to_string(g.ny) + " nz=" + std::to_string(g.nz) +
                  " n=" + std::to_string(g.n) + " levels=" + std::to_string(g.levels) +
                  " level_cells=" + join_sizes(g.level_cells) +
                  " coarsest=" + std::to_string(g.coarsest[0]) + "x" +
                  std::to_string(g.coarsest[1]) + "x" + std::to_string(g.coarsest[2]) +
                  " face_max=" + std::to_string(g.face_max) +
                  " bhat=" + (g.bhat_mapped ? "mapped" : "device") +
                  " bhat_host_bytes=" + std::to_string(g.bhat_host_bytes) +
                  " cub_sum_bytes=" + join_sizes(g.cub_sum_bytes));
    for (const LongBoxPhaseRecord* r :
         {&stage.phase_F, &stage.phase_S1, &stage.phase_S2, &stage.phase_S3}) {
        if (r->present)
            out.push_back(phase_line(*r));
    }
    if (tracker != nullptr && tracker->prepared() && stage.flow_ready()) {
        LongBoxPhaseRecord t;
        t.phase = "T";
        t.present = true;
        if (stage.fused_coefficients) {
            t.items.emplace_back("coef_interleaved",
                                 stage.coef_interleaved.capacity() * sizeof(double2));
        } else {
            t.items.emplace_back("coef_pot", stage.coef_pot.capacity() * sizeof(real));
            t.items.emplace_back("coef_logk", stage.coef_logk.capacity() * sizeof(real));
        }
        if (stage.labels_available && stage.labels.device_prepared())
            t.items.emplace_back("labels_device", stage.labels.device_bytes());
        for (const auto& it : tracker->device_buffer_bytes())
            t.items.emplace_back("tracker_" + it.first, it.second);
        t.device_used = device_used_now();
        t.seconds = tracker_prepare_seconds;
        out.push_back(phase_line(t));
    }
    return out;
}

} // namespace lagrangian_labels
} // namespace macroflow3d
