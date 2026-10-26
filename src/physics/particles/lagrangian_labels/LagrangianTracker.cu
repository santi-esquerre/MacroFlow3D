/**
 * @file LagrangianTracker.cu
 * @brief SF-36 N4: forward Lagrangian-label tracker on the cube production stage (spline field),
 *        host helpers (validation, lattice, weights, reductions) and the output writer.
 *        This TU is compiled with --fmad=false (CMakeLists.txt). See LagrangianTracker.cuh.
 */

#include "LagrangianTracker.cuh"
#include "FusedSplineLagrangianField.cuh"
#include "LagrangianTrackerImpl.cuh"
#include "NonPeriodicSplineLagrangianField.cuh"

#include "../../../external/nlohmann/json.hpp"
#include "../../streamfunctions/inlet_slab/NpyIo.hpp"

#include "apps/closure_gate/streamline_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace lagrangian_labels {

namespace sl = streamfunctions::inlet_slab;

// ================================================================================================
// Host helpers
// ================================================================================================

void validate_track_setup(const InjectionSpec& inj, const TrackOptions& opt) {
    auto bad = [](const std::string& m) { throw std::invalid_argument("lagrangian_labels: " + m); };
    if (!std::isfinite(inj.X0))
        bad("injection X0 must be finite");
    if (inj.M < 1)
        bad("injection M must be >= 1");
    if (!(std::isfinite(inj.w2a) && std::isfinite(inj.w2b) && inj.w2a < inj.w2b))
        bad("injection window requires finite w2a < w2b");
    if (!(std::isfinite(inj.w3a) && std::isfinite(inj.w3b) && inj.w3a < inj.w3b))
        bad("injection window requires finite w3a < w3b");
    if (!(opt.tol > 0.0) || !std::isfinite(opt.tol))
        bad("tol must be finite and > 0");
    if (!(opt.h_max > 0.0) || !std::isfinite(opt.h_max) || !(opt.h_max < 1.0))
        bad("h_max must be set explicitly, finite, 0 < h_max < 1 (period)");
    if (!(opt.min_step > 0.0) || !(opt.min_step < opt.h_max))
        bad("min_step must satisfy 0 < min_step < h_max");
    if (!std::isfinite(opt.max_arclength))
        bad("max_arclength must be finite (<= 0 selects the default)");
    if (opt.block < 1 || opt.block > 1024)
        bad("block must be in [1, 1024]");
    if (opt.planes.empty() && opt.times.empty())
        bad("at least one plane or one snapshot time is required");
    double prev = inj.X0;
    for (double X : opt.planes) {
        if (!std::isfinite(X) || !(X > prev))
            bad("planes must be finite, strictly increasing and > X0");
        prev = X;
    }
    prev = 0.0;
    for (double t : opt.times) {
        if (!std::isfinite(t) || !(t > prev))
            bad("times must be finite, strictly increasing and > 0");
        prev = t;
    }
}

double default_max_arclength(const InjectionSpec& inj, const TrackOptions& opt) {
    const double xl = opt.planes.empty() ? 0.0 : opt.planes.back() - inj.X0;
    const double tl = opt.times.empty() ? 0.0 : opt.times.back();
    return 4.0 * std::max(xl, tl) + 10.0;
}

std::vector<double> make_seed_lattice(const InjectionSpec& inj) {
    const std::size_t M = static_cast<std::size_t>(inj.M);
    std::vector<double> s(M * M * 3);
    for (std::size_t m2 = 0; m2 < M; ++m2) {
        for (std::size_t m3 = 0; m3 < M; ++m3) {
            const std::size_t p = m2 * M + m3;
            s[3 * p] = inj.X0;
            s[3 * p + 1] =
                inj.w2a + (inj.w2b - inj.w2a) * ((static_cast<double>(m2) + 0.5) / inj.M);
            s[3 * p + 2] =
                inj.w3a + (inj.w3b - inj.w3a) * ((static_cast<double>(m3) + 0.5) / inj.M);
        }
    }
    return s;
}

void normalize_flux_weights(const std::vector<double>& flux, std::vector<double>& weights,
                            long long& seed_backflow) {
    weights.assign(flux.size(), 0.0);
    seed_backflow = 0;
    double total = 0.0;
    for (std::size_t p = 0; p < flux.size(); ++p) {
        if (std::isfinite(flux[p]) && flux[p] > 0.0) {
            total += flux[p];
        } else {
            ++seed_backflow;
        }
    }
    if (!(total > 0.0)) {
        throw std::runtime_error("lagrangian_labels: no injected seed has positive flux");
    }
    for (std::size_t p = 0; p < flux.size(); ++p) {
        if (std::isfinite(flux[p]) && flux[p] > 0.0) {
            weights[p] = flux[p] / total;
        }
    }
}

namespace {

void summarize_pass(const std::vector<PassStats>& st, PassTotals& t) {
    t = PassTotals{};
    t.min_g1_hat = std::numeric_limits<double>::infinity();
    for (const PassStats& s : st) {
        if (s.status == kStatusExitInlet) // SF-41: not in the 7-slot histogram
            ++t.exit_inlet;
        else if (s.status == kStatusExitOutlet)
            ++t.exit_outlet;
        else
            ++t.status_hist[static_cast<std::size_t>(s.status >= 0 && s.status < 7 ? s.status : 6)];
        t.nfev += s.nfev;
        t.accepted_steps += s.accepted_steps;
        t.rejected_steps += s.rejected_steps;
        t.landing_steps += s.landing_steps;
        t.landing_rejected += s.landing_rejected;
        t.landings_discarded += s.landings_discarded;
        t.recrossings += s.recrossings;
        t.backflow_encounters += s.backflow_encounter;
        if (s.nfev > 0)
            t.min_g1_hat = std::min(t.min_g1_hat, s.min_g1_hat);
    }
}

/// Weighted mean and biased variance of values v(p) over the selected particles (index order).
template <class Sel, class Val>
void weighted_mean_var(const std::vector<double>& w, std::size_t Np, Sel sel, Val val,
                       double& mean, double& var, int& count) {
    double W = 0.0, S = 0.0;
    count = 0;
    for (std::size_t p = 0; p < Np; ++p) {
        if (!sel(p))
            continue;
        ++count;
        W += w[p];
        S += w[p] * val(p);
    }
    if (count == 0 || !(W > 0.0)) {
        mean = var = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    mean = S / W;
    double V = 0.0;
    for (std::size_t p = 0; p < Np; ++p) {
        if (!sel(p))
            continue;
        const double d = val(p) - mean;
        V += w[p] * d * d;
    }
    var = V / W;
}

} // namespace

void finalize_track_result(LagrangianTrackResult& r) {
    const std::size_t Np = static_cast<std::size_t>(r.Np);
    const std::size_t P = static_cast<std::size_t>(r.P), T = static_cast<std::size_t>(r.T);
    r.status_plane.resize(Np);
    r.status_time.resize(Np);
    for (std::size_t p = 0; p < Np; ++p) {
        r.status_plane[p] = static_cast<std::uint8_t>(r.stats_plane[p].status);
        r.status_time[p] = static_cast<std::uint8_t>(r.stats_time[p].status);
    }
    summarize_pass(r.stats_plane, r.totals_plane);
    summarize_pass(r.stats_time, r.totals_time);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    r.moments_time.assign(T, io::TimeSeriesPoint<double>{});
    for (std::size_t n = 0; n < T; ++n) {
        io::TimeSeriesPoint<double>& pt = r.moments_time[n];
        pt.time = r.times[n];
        auto sel = [&](std::size_t p) {
            return static_cast<std::size_t>(r.stats_time[p].n_reached) > n;
        };
        int cnt = 0;
        for (int k = 0; k < 3; ++k) {
            auto val = [&](std::size_t p) { return r.times_x[(p * T + n) * 3 + k]; };
            double m = nan, v = nan;
            weighted_mean_var(r.weights, Np, sel, val, m, v, cnt);
            pt.mean[k] = m;
            pt.var[k] = v;
        }
        pt.active = cnt;
    }

    r.moments_planes.assign(P, PlaneMoments{});
    for (std::size_t m = 0; m < P; ++m) {
        PlaneMoments& pm = r.moments_planes[m];
        pm.x1 = r.planes[m];
        auto sel = [&](std::size_t p) {
            return static_cast<std::size_t>(r.stats_plane[p].n_reached) > m;
        };
        int cnt = 0;
        weighted_mean_var(
            r.weights, Np, sel, [&](std::size_t p) { return r.planes_x23[(p * P + m) * 2]; },
            pm.mean_x2, pm.var_x2, cnt);
        weighted_mean_var(
            r.weights, Np, sel, [&](std::size_t p) { return r.planes_x23[(p * P + m) * 2 + 1]; },
            pm.mean_x3, pm.var_x3, cnt);
        weighted_mean_var(
            r.weights, Np, sel, [&](std::size_t p) { return r.planes_tau[p * P + m]; },
            pm.mean_tau, pm.var_tau, cnt);
        pm.contributing = cnt;
        double W = 0.0, S = 0.0, mx = 0.0;
        for (std::size_t p = 0; p < Np; ++p) {
            if (!sel(p))
                continue;
            const double e = 1.0 - std::exp(-r.planes_I[p * P + m]);
            W += r.weights[p];
            S += r.weights[p] * e * e;
            mx = std::max(mx, std::abs(e));
        }
        pm.rms_ev = (cnt > 0 && W > 0.0) ? std::sqrt(S / W) : nan;
        pm.max_ev = cnt > 0 ? mx : nan;
        int non_ok = 0;
        for (std::size_t p = 0; p < Np; ++p)
            if (r.stats_plane[p].status != 0)
                ++non_ok;
        pm.non_ok = non_ok;
    }

    if (r.nonperiodic_x) { // SF-41: exits before each plane (time), as the Pollock comparator
        const auto count = [&](const std::vector<PassStats>& st, std::size_t n, int code) {
            std::vector<long long> c(n, 0);
            for (const PassStats& s : st)
                if (s.status == code)
                    for (std::size_t m = static_cast<std::size_t>(s.n_reached); m < n; ++m)
                        ++c[m]; // counted from the first plane (time) not reached
            return c;
        };
        r.exit_inlet_plane = count(r.stats_plane, P, kStatusExitInlet);
        r.exit_outlet_plane = count(r.stats_plane, P, kStatusExitOutlet);
        r.exit_inlet_time = count(r.stats_time, T, kStatusExitInlet);
        r.exit_outlet_time = count(r.stats_time, T, kStatusExitOutlet);
    }
}

// ================================================================================================
// LagrangianTracker (cube spline field)
// ================================================================================================

struct LagrangianTracker::Impl {
    DeviceBuffer<real> coef_pot, coef_logk;
    LagrangianTrackerT<SplineLagrangianField> core;
    // SF-40 L6 (TrackOptions::fused_splines): interleaved {pot, logk} array and its tracker; the
    // separate arrays above are not created in that mode.
    DeviceBuffer<double2> coef_fused;
    LagrangianTrackerT<FusedSplineLagrangianField> core_fused;
    // SF-41 N3b: non-periodic-in-x1 flows (non-owning: the caller owns the coefficient arrays)
    LagrangianTrackerT<NonPeriodicSplineLagrangianField> core_np;
    LagrangianTrackerT<FusedNonPeriodicSplineLagrangianField> core_np_fused;
    enum class Kind { plain, fused, np, np_fused };
    Kind kind = Kind::plain;

    /// Calls fn on the tracker core selected by the last prepare().
    template <class Self, class Fn> static decltype(auto) visit(Self& self, Fn&& fn) {
        switch (self.kind) {
        case Kind::fused:
            return fn(self.core_fused);
        case Kind::np:
            return fn(self.core_np);
        case Kind::np_fused:
            return fn(self.core_np_fused);
        default:
            return fn(self.core);
        }
    }
};

LagrangianTracker::LagrangianTracker() : impl_(std::make_unique<Impl>()) {}
LagrangianTracker::~LagrangianTracker() = default;
LagrangianTracker::LagrangianTracker(LagrangianTracker&&) noexcept = default;
LagrangianTracker& LagrangianTracker::operator=(LagrangianTracker&&) noexcept = default;

void LagrangianTracker::prepare(CudaContext& ctx, const sl::ProductionStage& stage,
                                const InjectionSpec& inj, const TrackOptions& opt) {
    if (!stage.flow_ready()) {
        throw std::logic_error("LagrangianTracker::prepare: production stage flow is not ready");
    }
    const sl::SlabSplineDirectionField hf = stage.direction_field();
    const std::size_t nc = stage.potential_coefficients.size();
    if (stage.logk_coefficients.size() != nc) {
        throw std::logic_error("LagrangianTracker::prepare: coefficient arrays differ in size");
    }
    SplineLagrangianField f;
    f.potential = hf.potential;
    f.log_conductivity = hf.log_conductivity;
    for (int d = 0; d < 3; ++d)
        f.G[d] = hf.G[d];
    const sl::InletLabels* labels =
        (stage.labels_available && stage.labels.device_prepared()) ? &stage.labels : nullptr;
    impl_->kind = opt.fused_splines ? Impl::Kind::fused : Impl::Kind::plain;
    if (opt.fused_splines) {
        const std::vector<double2> inter = interleave_coefficients(
            stage.potential_coefficients.data(), stage.logk_coefficients.data(), nc);
        impl_->coef_fused = DeviceBuffer<double2>(nc);
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(impl_->coef_fused.data(), inter.data(),
                                          nc * sizeof(double2), cudaMemcpyHostToDevice));
        // SF-33 C2: legacy-stream copies must land before ctx-stream work.
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        impl_->core_fused.prepare(ctx, make_fused_field(f, impl_->coef_fused.data()), inj, opt,
                                  labels);
        return;
    }
    impl_->coef_pot = DeviceBuffer<real>(nc);
    impl_->coef_logk = DeviceBuffer<real>(nc);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(impl_->coef_pot.data(), stage.potential_coefficients.data(),
                                      nc * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(impl_->coef_logk.data(), stage.logk_coefficients.data(),
                                      nc * sizeof(real), cudaMemcpyHostToDevice));
    // SF-33 C2: legacy-stream copies must land before ctx-stream work.
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    f.potential.coeff = impl_->coef_pot.data();
    f.log_conductivity.coeff = impl_->coef_logk.data();
    impl_->core.prepare(ctx, f, inj, opt, labels);
}

void LagrangianTracker::prepare(CudaContext& ctx, const DeviceSplineFlow& flow,
                                const InjectionSpec& inj, const TrackOptions& opt) {
    const bool fused = opt.fused_splines;
    if (fused ? flow.coef_interleaved == nullptr
              : (flow.coef_pot == nullptr || flow.coef_logk == nullptr)) {
        throw std::logic_error(
            "LagrangianTracker::prepare: DeviceSplineFlow lacks the coefficient array(s) needed "
            "by TrackOptions::fused_splines");
    }
    if (flow.labels != nullptr && !flow.labels->device_prepared()) {
        throw std::logic_error("LagrangianTracker::prepare: DeviceSplineFlow labels not "
                               "device-prepared");
    }
    // Geometry only in the fused case (make_fused_field ignores the coefficient pointers).
    const real* geom_ptr =
        fused ? reinterpret_cast<const real*>(flow.coef_interleaved) : flow.coef_pot;
    SplineLagrangianField f;
    f.potential = interpolation::make_host_view(flow.grid, geom_ptr);
    f.log_conductivity = interpolation::make_host_view(flow.grid, geom_ptr);
    for (int d = 0; d < 3; ++d)
        f.G[d] = flow.G[d];
    impl_->kind = fused ? Impl::Kind::fused : Impl::Kind::plain;
    if (fused) {
        impl_->core_fused.prepare(ctx, make_fused_field(f, flow.coef_interleaved), inj, opt,
                                  flow.labels);
        return;
    }
    f.potential.coeff = flow.coef_pot;
    f.log_conductivity.coeff = flow.coef_logk;
    impl_->core.prepare(ctx, f, inj, opt, flow.labels);
}

void LagrangianTracker::prepare(CudaContext& ctx, const DeviceSplineFlowNonPeriodicX& flow,
                                const InjectionSpec& inj, const TrackOptions& opt) {
    const bool fused = opt.fused_splines;
    if (fused ? flow.coef_interleaved == nullptr
              : (flow.coef_pot == nullptr || flow.coef_logk == nullptr)) {
        throw std::logic_error(
            "LagrangianTracker::prepare: DeviceSplineFlowNonPeriodicX lacks the coefficient "
            "array(s) needed by TrackOptions::fused_splines");
    }
    // Geometry (and grid validation) from the same call in both modes; the coefficient pointers
    // are set below.
    NonPeriodicSplineLagrangianField f;
    f.potential = interpolation::make_nonperiodic_x_view(
        flow.grid, fused ? reinterpret_cast<const real*>(flow.coef_interleaved) : flow.coef_pot);
    f.log_conductivity = f.potential;
    for (int d = 0; d < 3; ++d)
        f.G[d] = flow.G[d];
    impl_->kind = fused ? Impl::Kind::np_fused : Impl::Kind::np;
    if (fused) {
        impl_->core_np_fused.prepare(ctx, make_fused_np_field(f, flow.coef_interleaved), inj, opt,
                                     nullptr);
        return;
    }
    f.log_conductivity.coeff = flow.coef_logk;
    impl_->core_np.prepare(ctx, f, inj, opt, nullptr);
}

std::vector<std::pair<std::string, std::size_t>> LagrangianTracker::device_buffer_bytes() const {
    if (!impl_ || !prepared())
        return {};
    auto v = Impl::visit(*impl_, [](const auto& c) { return c.device_buffer_bytes(); });
    v.emplace_back("prepare_transient",
                   Impl::visit(*impl_, [](const auto& c) { return c.prepare_transient_bytes(); }));
    return v;
}

const LagrangianTrackResult& LagrangianTracker::track(CudaContext& ctx, int block) {
    return Impl::visit(
        *impl_, [&](auto& c) -> const LagrangianTrackResult& { return c.track(ctx, block); });
}

bool LagrangianTracker::prepared() const {
    return impl_ && Impl::visit(*impl_, [](const auto& c) { return c.prepared(); });
}

std::vector<const void*> LagrangianTracker::device_storage_pointers() const {
    std::vector<const void*> v =
        Impl::visit(*impl_, [](const auto& c) { return c.device_storage_pointers(); });
    if (impl_->kind == Impl::Kind::fused) {
        v.push_back(impl_->coef_fused.data());
    } else if (impl_->kind == Impl::Kind::plain) {
        v.push_back(impl_->coef_pot.data());
        v.push_back(impl_->coef_logk.data());
    }
    return v;
}

// ================================================================================================
// Output
// ================================================================================================

namespace {

/// Closes a FILE* written with fprintf; throws if any write or the final flush failed (ENOSPC).
void close_or_throw(std::FILE* f, const std::string& what) {
    const bool bad = std::ferror(f) != 0;
    if (std::fclose(f) != 0 || bad)
        throw std::runtime_error("write_track_outputs: write failed for " + what);
}

void write_npy_u8(const std::string& path, const std::vector<std::size_t>& shape,
                  const std::uint8_t* data, std::size_t count) {
    sl::NpyArray tmp;
    tmp.shape = shape;
    if (tmp.count() != count)
        throw std::invalid_argument("write_npy_u8: data count does not match shape");
    const std::string dict =
        "{'descr': '|u1', 'fortran_order': False, 'shape': " + tmp.shape_string() + ", }";
    const std::size_t total = 10 + dict.size() + 1;
    const std::size_t pad = (64 - total % 64) % 64;
    const std::string hdr = dict + std::string(pad, ' ') + "\n";
    std::string out("\x93NUMPY", 6);
    out.push_back('\x01');
    out.push_back('\x00');
    const std::uint16_t hl = static_cast<std::uint16_t>(hdr.size());
    out.push_back(static_cast<char>(hl & 0xFFu));
    out.push_back(static_cast<char>((hl >> 8) & 0xFFu));
    out += hdr;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("write_track_outputs: cannot write " + path);
    f.write(out.data(), static_cast<std::streamsize>(out.size()));
    if (count > 0)
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count));
    f.close();
    if (!f)
        throw std::runtime_error("write_track_outputs: write failed for " + path);
}

void npy(const std::string& dir, const char* name, const std::vector<std::size_t>& shape,
         const std::vector<double>& v) {
    sl::write_npy(dir + "/" + name, shape, v.data(), v.size());
}

nlohmann::json hist_json(const PassTotals& t) {
    nlohmann::json h = nlohmann::json::object();
    for (int s = 0; s < 7; ++s) {
        h[closure_gate::to_string(static_cast<closure_gate::StreamlineStatus>(s))] =
            t.status_hist[static_cast<std::size_t>(s)];
    }
    return h;
}

nlohmann::json totals_json(const PassTotals& t) {
    nlohmann::json j;
    j["status_hist"] = hist_json(t);
    j["nfev"] = t.nfev;
    j["accepted_steps"] = t.accepted_steps;
    j["rejected_steps"] = t.rejected_steps;
    j["landing_steps"] = t.landing_steps;
    j["landing_rejected"] = t.landing_rejected;
    j["landings_discarded"] = t.landings_discarded;
    j["recrossings"] = t.recrossings;
    j["backflow_encounters"] = t.backflow_encounters;
    j["min_g1_hat"] = t.min_g1_hat;
    return j;
}

} // namespace

void write_track_outputs(const std::string& dir, const LagrangianTrackResult& r,
                         const TrackMeta& meta) {
    std::filesystem::create_directories(dir);
    const std::size_t Np = static_cast<std::size_t>(r.Np);
    const std::size_t P = static_cast<std::size_t>(r.P), T = static_cast<std::size_t>(r.T);

    npy(dir, "seeds.npy", {Np, 3}, r.seeds);
    npy(dir, "weights.npy", {Np}, r.weights);
    npy(dir, "labels.npy", {Np, 2}, r.labels);
    write_npy_u8(dir + "/status_plane.npy", {Np}, r.status_plane.data(), r.status_plane.size());
    write_npy_u8(dir + "/status_time.npy", {Np}, r.status_time.data(), r.status_time.size());
    npy(dir, "planes_x23.npy", {Np, P, 2}, r.planes_x23);
    npy(dir, "planes_tau.npy", {Np, P}, r.planes_tau);
    npy(dir, "planes_I.npy", {Np, P}, r.planes_I);
    npy(dir, "times_x.npy", {Np, T, 3}, r.times_x);
    npy(dir, "times_I.npy", {Np, T}, r.times_I);
    if (r.covectors) {
        npy(dir, "planes_cov.npy", {Np, P, 6}, r.planes_cov);
        npy(dir, "times_cov.npy", {Np, T, 6}, r.times_cov);
    }

    if (!io::CsvTimeSeriesWriter::write(dir + "/moments_time.csv", r.moments_time)) {
        throw std::runtime_error("write_track_outputs: cannot write moments_time.csv");
    }
    {
        std::FILE* f = std::fopen((dir + "/moments_planes.csv").c_str(), "w");
        if (!f)
            throw std::runtime_error("write_track_outputs: cannot write moments_planes.csv");
        std::fprintf(f, "x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_tau,var_tau,"
                        "rms_ev,max_ev\n");
        for (const PlaneMoments& m : r.moments_planes) {
            std::fprintf(f, "%.15e,%d,%d,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", m.x1,
                         m.contributing, m.non_ok, m.mean_x2, m.var_x2, m.mean_x3, m.var_x3,
                         m.mean_tau, m.var_tau, m.rms_ev, m.max_ev);
        }
        close_or_throw(f, "moments_planes.csv");
    }

    nlohmann::json j;
    j["field"] = meta.field;
    j["eps"] = meta.eps;
    j["N"] = meta.N;
    j["h"] = meta.h;
    j["commit"] = meta.commit;
    j["X0"] = r.injection.X0;
    j["M"] = r.injection.M;
    j["window"] = {r.injection.w2a, r.injection.w2b, r.injection.w3a, r.injection.w3b};
    j["planes"] = r.planes;
    j["times"] = r.times;
    j["tol"] = r.tol;
    j["h_max"] = r.h_max;
    j["min_step"] = r.min_step;
    j["max_arclength"] = r.max_arclength;
    j["covectors"] = r.covectors;
    j["Np"] = r.Np;
    j["seed_backflow"] = r.seed_backflow;
    j["plane_pass"] = totals_json(r.totals_plane);
    j["time_pass"] = totals_json(r.totals_time);
    if (r.nonperiodic_x) { // SF-41: new keys / files only for non-periodic-in-x1 runs
        j["nonperiodic_x"] = true;
        j["domain_L1"] = r.domain_L1;
        j["exit_status_codes"] = {{"exit_inlet", kStatusExitInlet}, {"exit_outlet", kStatusExitOutlet}};
        j["plane_pass"]["exit_inlet"] = r.totals_plane.exit_inlet;
        j["plane_pass"]["exit_outlet"] = r.totals_plane.exit_outlet;
        j["time_pass"]["exit_inlet"] = r.totals_time.exit_inlet;
        j["time_pass"]["exit_outlet"] = r.totals_time.exit_outlet;
        const auto write_exits = [&](const char* file, const char* head, const std::vector<double>& axis,
                                     const std::vector<long long>& in, const std::vector<long long>& out) {
            std::FILE* f = std::fopen((dir + "/" + file).c_str(), "w");
            if (!f)
                throw std::runtime_error(std::string("write_track_outputs: cannot write ") + file);
            std::fprintf(f, "%s,exit_inlet,exit_outlet\n", head);
            for (std::size_t i = 0; i < axis.size(); ++i)
                std::fprintf(f, "%.15e,%lld,%lld\n", axis[i], i < in.size() ? in[i] : 0LL,
                             i < out.size() ? out[i] : 0LL);
            close_or_throw(f, file);
        };
        write_exits("exits_planes.csv", "x1", r.planes, r.exit_inlet_plane, r.exit_outlet_plane);
        write_exits("exits_time.csv", "t", r.times, r.exit_inlet_time, r.exit_outlet_time);
    }
    j["timings_s"] = {{"plane_kernel", r.seconds_plane_kernel},
                      {"time_kernel", r.seconds_time_kernel},
                      {"download", r.seconds_download},
                      {"moments", r.seconds_moments}};
    if (r.single_launch) // SF-40 L6: key only with --single-launch (plane/time kernel are null)
        j["timings_s"]["single_kernel"] = r.seconds_single_kernel;
    std::ofstream jf(dir + "/track.json", std::ios::trunc);
    if (!jf)
        throw std::runtime_error("write_track_outputs: cannot write track.json");
    jf << j.dump(2) << "\n";
    jf.close();
    if (!jf)
        throw std::runtime_error("write_track_outputs: write failed for track.json");
}

} // namespace lagrangian_labels
} // namespace macroflow3d
