/**
 * @file EnsembleRunner.cu
 * @brief SF-38 N3: Lagrangian ensemble runner (contract C-RUNNER). See EnsembleRunner.cuh.
 *
 * Host orchestration only: every numerical call is an existing, read-only library entry point
 * (LongBoxStage, LagrangianTracker, finalize_track_result, PeriodicFaceFluxes, track_pollock_lattice,
 * compute_macrodispersion). The lines printed per realization mirror those of
 * apps/inlet_slab/inlet_slab_driver.cuh (run_track_forward_lean / track_forward_tail /
 * run_pollock_comparator_faces), prefixed by `R=<r> `.
 */

#include "EnsembleRunner.cuh"

#include "../../../analysis/macrodispersion/MacrodispersionAnalysis.hpp"
#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Grid3D.hpp"
#include "../../../external/nlohmann/json.hpp"
#include "../../../io/output_layout.hpp"
#include "../../../io/writers/CsvTimeSeriesWriter.hpp"
#include "../../stochastic/PeriodicGaussianField.cuh"
#include "../../streamfunctions/inlet_slab/NpyIo.hpp"
#include "FaceFluxRepack.cuh"
#include "LagrangianTracker.cuh"
#include "LongBoxStage.cuh"
#include "PollockComparator.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>

namespace macroflow3d {
namespace lagrangian_labels {

namespace {

namespace fs = std::filesystem;
namespace sl = streamfunctions::inlet_slab;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

void out_line(const std::string& s) {
    std::fputs(s.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::string join_list(const std::vector<double>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i)
        s += fmt(i == 0 ? "%.10g" : ",%.10g", v[i]);
    return s;
}

std::string join_list17(const std::vector<double>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i)
        s += fmt(i == 0 ? "%.17g" : ",%.17g", v[i]);
    return s;
}

// ================================================================================================
// Command line
// ================================================================================================

double parse_double(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || !std::isfinite(v))
        throw UsageError(opt + ": '" + s + "' is not a finite number");
    return v;
}

long long parse_int(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || end != s.c_str() + s.size())
        throw UsageError(opt + ": '" + s + "' is not an integer");
    return v;
}

unsigned long long parse_u64(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    if (s.empty() || s[0] == '-' || s[0] == '+')
        throw UsageError(opt + ": '" + s + "' is not an unsigned integer");
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (end != s.c_str() + s.size())
        throw UsageError(opt + ": '" + s + "' is not an unsigned integer");
    return v;
}

std::vector<double> parse_list(const std::string& opt, const std::string& s) {
    std::vector<double> v;
    std::size_t a = 0;
    while (true) {
        const std::size_t b = s.find(',', a);
        v.push_back(parse_double(opt, s.substr(a, b == std::string::npos ? b : b - a)));
        if (b == std::string::npos)
            break;
        a = b + 1;
    }
    return v;
}

bool parse_onoff(const std::string& opt, const std::string& s) {
    if (s == "on")
        return true;
    if (s == "off")
        return false;
    throw UsageError(opt + ": '" + s + "' must be 'on' or 'off'");
}

InjectionSpec make_injection(const EnsembleConfig& c) {
    InjectionSpec inj;
    inj.X0 = c.x0;
    inj.M = c.np_side;
    inj.w2a = c.window[0];
    inj.w2b = c.window[1];
    inj.w3a = c.window[2];
    inj.w3b = c.window[3];
    return inj;
}

// ================================================================================================
// File helpers
// ================================================================================================

void write_json(const std::string& path, const json& j) {
    std::ofstream f(path, std::ios::trunc);
    if (!f)
        throw std::runtime_error("cannot write " + path);
    f << j.dump(2) << "\n";
    if (!f)
        throw std::runtime_error("write failed for " + path);
}

/// write_track_outputs embeds the tracker's wall-clock timings in track.json (and is read-only for
/// this increment); every non-timings file must be byte-reproducible, so the key is removed here
/// and the values are recorded in timings.json instead.
void strip_track_timings(const std::string& path) {
    json j;
    {
        std::ifstream f(path);
        if (!f)
            throw std::runtime_error("cannot read " + path);
        f >> j;
    }
    j.erase("timings_s");
    write_json(path, j);
}

/// C-order `.npy` v1.0 of any little-endian element type (header padded to 64 bytes as NpyIo).
void write_npy_raw(const std::string& path, const std::string& descr,
                   const std::vector<std::size_t>& shape, const void* data, std::size_t count,
                   std::size_t elem_bytes) {
    sl::NpyArray tmp;
    tmp.shape = shape;
    if (tmp.count() != count)
        throw std::invalid_argument("write_npy_raw: data count does not match shape for " + path);
    const std::string dict =
        "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + tmp.shape_string() + ", }";
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
        throw std::runtime_error("cannot write " + path);
    f.write(out.data(), static_cast<std::streamsize>(out.size()));
    if (count > 0)
        f.write(static_cast<const char*>(data), static_cast<std::streamsize>(count * elem_bytes));
    if (!f)
        throw std::runtime_error("write failed for " + path);
}

void write_f8(const std::string& path, const std::vector<std::size_t>& shape,
              const std::vector<double>& v) {
    sl::write_npy(path, shape, v.data(), v.size());
}

void write_tracker_planes_csv(const std::string& path, const std::vector<PlaneMoments>& pm) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_tau,var_tau,"
                    "rms_ev,max_ev\n");
    for (const PlaneMoments& m : pm)
        std::fprintf(f, "%.15e,%d,%d,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", m.x1,
                     m.contributing, m.non_ok, m.mean_x2, m.var_x2, m.mean_x3, m.var_x3,
                     m.mean_tau, m.var_tau, m.rms_ev, m.max_ev);
    std::fclose(f);
}

void write_pollock_planes_csv(const std::string& path, const std::vector<PollockPlaneMoments>& pm) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f)
        throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_t,var_t\n");
    for (const PollockPlaneMoments& m : pm)
        std::fprintf(f, "%.15e,%d,%d,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n", m.x1, m.contributing,
                     m.non_ok, m.mean_x2, m.var_x2, m.mean_x3, m.var_x3, m.mean_t, m.var_t);
    std::fclose(f);
}

void write_series_csv(const std::string& path, const std::vector<io::TimeSeriesPoint<double>>& s) {
    if (!io::CsvTimeSeriesWriter::write(path, s))
        throw std::runtime_error("cannot write " + path);
}

// ================================================================================================
// Paths and atomicity
// ================================================================================================

std::string rdir_of(const EnsembleConfig& c, int r) {
    return fmt("%s/realizations/r_%04d", c.out.c_str(), r);
}
std::string stats_tmp(const io::OutputLayout& L, int r) { return L.realization_timeseries(r) + ".tmp"; }
std::string twin_stats_path(const EnsembleConfig& c, int r) {
    return fmt("%s/stats_twin/realization_%04d_timeseries.csv", c.out.c_str(), r);
}

std::string done_text(const EnsembleConfig& c, int r) {
    return "signature: " + ensemble_signature(c) + "\n" +
           fmt("r=%d\nseed=%llu\n", r, c.seed0 + static_cast<unsigned long long>(r));
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void test_abort(const EnsembleConfig& c, int r, const char* phase) {
    if (c.test_abort_r == r && c.test_abort_phase == phase) {
        out_line(fmt("R=%d TEST_ABORT phase=%s (simulated crash, exit %d)", r, phase, kExitTestAbort));
        std::_Exit(kExitTestAbort);
    }
}

/// Renames the finished temporaries into place and writes DONE last. `stats` are (tmp, final) pairs.
void commit_realization(const EnsembleConfig& c, int r, const std::string& tmpdir,
                        const std::vector<std::pair<std::string, std::string>>& stats) {
    test_abort(c, r, "tmp");
    for (const auto& p : stats)
        fs::rename(p.first, p.second);
    const std::string fin = rdir_of(c, r);
    fs::rename(tmpdir, fin);
    test_abort(c, r, "rename");
    const std::string done_tmp = fin + "/DONE.tmp";
    {
        std::ofstream f(done_tmp, std::ios::trunc);
        f << done_text(c, r);
        if (!f)
            throw std::runtime_error("cannot write " + done_tmp);
    }
    fs::rename(done_tmp, fin + "/DONE");
}

// ================================================================================================
// Pollock bridge twin: uniform weights, the comparator's reduction rules (PollockComparator.cu)
// ================================================================================================

template <class Sel, class Val>
void weighted_mean_var(const std::vector<double>& w, std::size_t Np, Sel sel, Val val, double& mean,
                       double& var, int& count) {
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

void pollock_twin_moments(const PollockLatticeResult& Q, std::vector<PollockPlaneMoments>& planes,
                          std::vector<io::TimeSeriesPoint<double>>& time) {
    const std::size_t Np = static_cast<std::size_t>(Q.Np);
    const std::size_t P = static_cast<std::size_t>(Q.P), T = static_cast<std::size_t>(Q.T);
    const std::vector<double> w(Np, 1.0 / static_cast<double>(Np));
    planes.assign(P, PollockPlaneMoments{});
    for (std::size_t m = 0; m < P; ++m) {
        PollockPlaneMoments& pm = planes[m];
        pm.x1 = Q.planes[m];
        auto sel = [&](std::size_t p) { return static_cast<std::size_t>(Q.reached_plane[p]) > m; };
        int cnt = 0;
        weighted_mean_var(
            w, Np, sel, [&](std::size_t p) { return Q.planes_x23[(p * P + m) * 2]; }, pm.mean_x2,
            pm.var_x2, cnt);
        weighted_mean_var(
            w, Np, sel, [&](std::size_t p) { return Q.planes_x23[(p * P + m) * 2 + 1]; },
            pm.mean_x3, pm.var_x3, cnt);
        weighted_mean_var(
            w, Np, sel, [&](std::size_t p) { return Q.planes_t[p * P + m]; }, pm.mean_t, pm.var_t,
            cnt);
        pm.contributing = cnt;
        int non_ok = 0;
        for (std::size_t p = 0; p < Np; ++p)
            if (Q.status_plane[p] != physics::particles::streamline_tracker::kStatusActive)
                ++non_ok;
        pm.non_ok = non_ok;
    }
    time.assign(T, io::TimeSeriesPoint<double>{});
    for (std::size_t n = 0; n < T; ++n) {
        io::TimeSeriesPoint<double>& pt = time[n];
        pt.time = Q.times[n];
        auto sel = [&](std::size_t p) { return static_cast<std::size_t>(Q.reached_time[p]) > n; };
        int cnt = 0;
        for (int k = 0; k < 3; ++k) {
            double m = 0.0, v = 0.0;
            weighted_mean_var(
                w, Np, sel, [&](std::size_t p) { return Q.times_x[(p * T + n) * 3 + k]; }, m, v, cnt);
            pt.mean[k] = m;
            pt.var[k] = v;
        }
        pt.active = cnt;
    }
}

std::string pollock_hist_string(const std::array<long long, kPollockStatusSlots>& h) {
    std::string s;
    for (std::size_t k = 0; k < h.size(); ++k) {
        if (h[k] == 0 && k != 0)
            continue;
        s += fmt(s.empty() ? "%zu:%lld" : ",%zu:%lld", k, h[k]);
    }
    return s;
}

json field_json(const EnsembleConfig& c, unsigned long long seed,
                const physics::PeriodicGaussianFieldReport& f) {
    return {{"kind", "gaussian"},
            {"grid", {c.nx, c.ny, c.nz}},
            {"sigma2", c.sigma2},
            {"ell", c.ell},
            {"seed", seed},
            {"normalize_variance", true},
            {"raw_mean", f.raw_mean},
            {"raw_variance", f.raw_variance},
            {"applied_scale", f.applied_scale},
            {"final_variance", f.final_variance},
            {"active_modes", f.active_mode_count}};
}

std::string build_commit() {
#ifdef MACROFLOW3D_GIT_HASH
    return MACROFLOW3D_GIT_HASH;
#else
    return "unknown";
#endif
}

// ================================================================================================
// One realization: --field-only
// ================================================================================================

int run_field_only(CudaContext& ctx, const EnsembleConfig& c, int r) {
    const auto t_all = Clock::now();
    const unsigned long long seed = c.seed0 + static_cast<unsigned long long>(r);
    const std::string tmpdir = rdir_of(c, r) + ".tmp";
    fs::remove_all(tmpdir);
    fs::create_directories(tmpdir);

    const real h = 1.0 / static_cast<real>(c.ny);
    const Grid3D grid(c.nx, c.ny, c.nz, h, h, h);
    const std::size_t n = static_cast<std::size_t>(c.nx) * c.ny * c.nz;
    physics::PeriodicGaussianFieldConfig fc; // exactly LongBoxStage phase F
    fc.sigma2 = c.sigma2;
    fc.corr_length = c.ell;
    fc.seed = seed;
    fc.normalize_variance = true;
    physics::PeriodicGaussianFieldReport rep;
    const auto t0 = Clock::now();
    {
        DeviceBuffer<real> y(n);
        physics::PeriodicGaussianFieldWorkspace ws;
        rep = physics::generate_periodic_gaussian_field(ctx, grid, fc, y.span(), ws);
        ctx.synchronize();
    }
    const double field_s = since(t0);
    out_line(fmt("R=%d FIELD gaussian (SF-18) N=%d grid=%dx%dx%d sigma2=%g ell=%g seed=%llu "
                 "normalize_variance=1 applied_scale=%.12e raw_variance=%.12e final_variance=%.12e "
                 "active_modes=%zu (field-only: no flow)",
                 r, c.nx, c.nx, c.ny, c.nz, c.sigma2, c.ell, seed, rep.applied_scale,
                 rep.raw_variance, rep.final_variance, rep.active_mode_count));
    test_abort(c, r, "stage");
    write_json(tmpdir + "/field.json", field_json(c, seed, rep));
    write_json(tmpdir + "/timings.json",
               {{"r", r}, {"seed", seed}, {"commit", build_commit()},
                {"seconds", {{"field", field_s}, {"total", since(t_all)}}}});
    commit_realization(c, r, tmpdir, {});
    out_line(fmt("R=%d STATUS ok", r));
    out_line(fmt("TIMING r=%d seconds=%.3f", r, since(t_all)));
    return kExitOk;
}

// ================================================================================================
// One realization: full path
// ================================================================================================

int run_realization(CudaContext& ctx, const EnsembleConfig& c, int r, const io::OutputLayout& layout,
                    EnsembleRunSummary* sum) {
    const auto t_all = Clock::now();
    const unsigned long long seed = c.seed0 + static_cast<unsigned long long>(r);
    const std::string tmpdir = rdir_of(c, r) + ".tmp";
    fs::remove_all(tmpdir);
    fs::create_directories(tmpdir);
    auto P = [&](const std::string& s) { out_line(fmt("R=%d ", r) + s); };
    auto fail = [&](const std::string& status, int code) {
        P("STATUS " + status);
        fs::remove_all(tmpdir);
        return code;
    };
    json TJ; // timings.json
    TJ["r"] = r;
    TJ["seed"] = seed;
    TJ["commit"] = build_commit();

    // ---- stage ------------------------------------------------------------------------------
    LongBoxSpec spec;
    spec.nx = c.nx;
    spec.ny = c.ny;
    spec.nz = c.nz;
    spec.source = LongBoxSourceKind::gaussian;
    spec.sigma2 = c.sigma2;
    spec.ell = c.ell;
    spec.seed = seed;
    spec.normalize_variance = true;
    spec.field_name = "gaussian";
    spec.eps = c.eps;
    spec.pcg_rtol = c.pcg_rtol;
    spec.bhat_mapped_host = true;
    spec.keep_face_fluxes = c.pollock_bridge;
    spec.fused_coefficients = c.fused_splines;

    const auto t_stage = Clock::now();
    auto stage = std::make_unique<LongBoxStage>(build_long_box_stage(ctx, spec));
    const double stage_s = since(t_stage);
    const sl::ProductionStageReport& rep = stage->report;
    const real h = stage->grid.dx;

    P(fmt("FIELD gaussian (SF-18) N=%d grid=%dx%dx%d sigma2=%g ell=%g seed=%llu normalize_variance=1 "
          "applied_scale=%.12e raw_variance=%.12e final_variance=%.12e active_modes=%zu | stage "
          "field k = exp(eps Y), eps = %g",
          c.nx, c.nx, c.ny, c.nz, c.sigma2, c.ell, seed, rep.sf18.applied_scale,
          rep.sf18.raw_variance, rep.sf18.final_variance, rep.sf18.active_mode_count, c.eps));
    {   // SETUP block: the stage summary with its two vertex-vD lines replaced by explicit lean lines
        std::string block;
        const std::string s = rep.summary();
        std::size_t a = 0;
        while (a < s.size()) {
            std::size_t b = s.find('\n', a);
            if (b == std::string::npos)
                b = s.size();
            const std::string ln = s.substr(a, b - a);
            if (ln.rfind("  inlet:", 0) == 0)
                block += fmt("  inlet: vmin=%.6e mean(U plane 0)=%.15f (mean-1=%.3e) Q0-1=%.3e "
                             "v_rms=n/a (lean stage: vertex vD not built)\n",
                             rep.inlet_vmin, rep.inlet_mean, rep.inlet_mean - 1.0, rep.Q0 - 1.0);
            else if (ln.rfind("  v1(U-face)", 0) == 0)
                block += "  v1(U-face) vs spline flow k g1: n/a (lean stage); spline potential GPU "
                         "vs host prefilter: n/a\n";
            else
                block += ln + "\n";
            a = b + 1;
        }
        std::istringstream is(block);
        for (std::string ln; std::getline(is, ln);)
            P("SETUP " + ln);
    }
    P(fmt("SETUP stage eps=%g built in %.3fs (lean stage, Grid3D %dx%dx%d, bhat=mapped)", c.eps,
          stage_s, c.nx, c.ny, c.nz));
    P(fmt("STAGE status=%s grid=%dx%dx%d mg_levels=%d G=%.12e,%.12e,%.12e darcy_converged=%d "
          "flow_ready=%d seconds=%.3f",
          sl::to_string(rep.status), c.nx, c.ny, c.nz, rep.mg_levels, stage->G[0], stage->G[1],
          stage->G[2], rep.darcy_converged ? 1 : 0, stage->flow_ready() ? 1 : 0, stage_s));
    if (!stage->flow_ready()) {
        for (const std::string& l : mem_lines(*stage, nullptr))
            P(l);
        P(fmt("STAGE_FAILED status=darcy_failed eps=%g inlet_vmin=%.6e: the SF-19 flow did not "
              "converge or the splines were not built (stage status %s)",
              c.eps, rep.inlet_vmin, sl::to_string(rep.status)));
        return fail("darcy_failed", kExitStageFailed);
    }
    const bool labels_available = stage->labels_available;
    const double inlet_vmin = rep.inlet_vmin;
    const std::string stage_status = sl::to_string(rep.status);
    const json field_j = field_json(c, seed, rep.sf18);
    test_abort(c, r, "stage");

    // ---- tracker ----------------------------------------------------------------------------
    const InjectionSpec inj = make_injection(c);
    TrackOptions topt;
    topt.tol = c.tol;
    topt.max_arclength = c.max_arclength;
    topt.h_max = h / static_cast<double>(c.hmax_div);
    topt.planes = c.planes;
    topt.times = c.times;
    topt.fused_splines = c.fused_splines;
    topt.single_launch = c.single_launch;
    auto tracker = std::make_unique<LagrangianTracker>();
    const auto t_prep = Clock::now();
    tracker->prepare(ctx, stage->device_flow(), inj, topt);
    const double prepare_s = since(t_prep);
    std::vector<std::string> mems = mem_lines(*stage, tracker.get(), prepare_s);
    for (const std::string& l : mems)
        P(l);
    const auto t_track = Clock::now();
    const LagrangianTrackResult res = tracker->track(ctx);
    const double track_s = since(t_track);

    P(fmt("TRACK field=gaussian eps=%g N=%d h=%.10g x0=%.10g M=%d Np=%d planes=%s times=%s "
          "tol=%.1e h_max=%.6e (h/%d) min_step=%.1e max_arclength=%g covectors=%d "
          "labels_available=%d inlet_vmin=%.6e",
          c.eps, c.nx, static_cast<double>(h), inj.X0, inj.M, res.Np, join_list(res.planes).c_str(),
          join_list(res.times).c_str(), res.tol, res.h_max, c.hmax_div, res.min_step,
          res.max_arclength, res.covectors ? 1 : 0, labels_available ? 1 : 0, inlet_vmin));
    P(fmt("BACKFLOW field=gaussian eps=%g N=%d inlet_vmin=%.6e labels_available=%d seed_backflow=%lld "
          "backflow_enc_plane=%lld backflow_enc_time=%lld (inlet backflow is recorded, not an exit "
          "condition)",
          c.eps, c.nx, inlet_vmin, labels_available ? 1 : 0, res.seed_backflow,
          res.totals_plane.backflow_encounters, res.totals_time.backflow_encounters));
    for (std::size_t m = 0; m < res.moments_planes.size(); ++m) {
        const PlaneMoments& q = res.moments_planes[m];
        P(fmt("TPLANE m=%zu x1=%.10g n_ok=%d mean2=%.12e mean3=%.12e var2=%.12e var3=%.12e "
              "mean_tau=%.12e var_tau=%.12e ev_rms=%.6e ev_max=%.6e",
              m, q.x1, q.contributing, q.mean_x2, q.mean_x3, q.var_x2, q.var_x3, q.mean_tau,
              q.var_tau, q.rms_ev, q.max_ev));
    }
    for (std::size_t n = 0; n < res.moments_time.size(); ++n) {
        const auto& q = res.moments_time[n];
        P(fmt("TTIME n=%zu t=%.10g active=%d mean1=%.12e mean2=%.12e mean3=%.12e var1=%.12e "
              "var2=%.12e var3=%.12e",
              n, q.time, q.active, q.mean[0], q.mean[1], q.mean[2], q.var[0], q.var[1], q.var[2]));
    }
    for (int pass = 0; pass < 2; ++pass) {
        const PassTotals& t = pass == 0 ? res.totals_plane : res.totals_time;
        P(fmt("TSTATUS pass=%s hist=%lld,%lld,%lld,%lld,%lld,%lld,%lld landings_discarded=%lld "
              "backflow_enc=%lld nfev=%lld",
              pass == 0 ? "plane" : "time", t.status_hist[0], t.status_hist[1], t.status_hist[2],
              t.status_hist[3], t.status_hist[4], t.status_hist[5], t.status_hist[6],
              t.landings_discarded, t.backflow_encounters, t.nfev));
    }
    // Non-ok particles are recorded (status.json, STATUS line), never an exit condition: the moments
    // are already taken over the particles that reached each plane / time.
    long long nonok_plane = 0, nonok_time = 0;
    for (std::size_t s = 2; s < 7; ++s) {
        nonok_plane += res.totals_plane.status_hist[s];
        nonok_time += res.totals_time.status_hist[s];
    }

    // ---- release the stage and the tracker before the bridge (peak = max, not sum) -----------
    std::vector<real> fu, fv, fw;
    if (c.pollock_bridge) {
        fu = std::move(stage->face_flux_u);
        fv = std::move(stage->face_flux_v);
        fw = std::move(stage->face_flux_w);
    }
    const json tracker_timings = {{"plane_kernel", res.seconds_plane_kernel},
                                  {"time_kernel", res.seconds_time_kernel},
                                  {"download", res.seconds_download},
                                  {"moments", res.seconds_moments},
                                  {"single_kernel", res.seconds_single_kernel}};
    tracker.reset();
    stage.reset();

    // ---- twin series ------------------------------------------------------------------------
    std::vector<io::TimeSeriesPoint<double>> twin_time;
    std::vector<PlaneMoments> twin_planes;
    std::vector<double> twin_weights;
    if (c.twin) {
        LagrangianTrackResult tw = res;
        tw.weights.assign(static_cast<std::size_t>(tw.Np), 1.0 / static_cast<double>(tw.Np));
        finalize_track_result(tw);
        twin_time = std::move(tw.moments_time);
        twin_planes = std::move(tw.moments_planes);
        twin_weights = std::move(tw.weights);
        P(fmt("TWIN Np=%d weights=1/Np (every particle, backflow seeds included) planes=%zu times=%zu",
              res.Np, twin_planes.size(), twin_time.size()));
    }

    // ---- Pollock bridge ---------------------------------------------------------------------
    PollockLatticeResult Q;
    std::vector<PollockPlaneMoments> pq_twin_planes;
    std::vector<io::TimeSeriesPoint<double>> pq_twin_time;
    double repack_s = 0.0;
    long long n_fail = 0; // positive-weight Pollock particles not active in a pass (recorded)
    if (c.pollock_bridge) {
        const auto t_rep = Clock::now();
        {
            PeriodicFaceFluxes faces =
                PeriodicFaceFluxes::upload_and_repack(ctx, fu, fv, fw, c.nx, c.ny, c.nz, h);
            repack_s = since(t_rep);
            std::vector<real>().swap(fu);
            std::vector<real>().swap(fv);
            std::vector<real>().swap(fw);
            physics::particles::streamline_tracker::PollockParams prm;
            prm.max_cells_per_call = 10000000; // the SF-32 documented default
            Q = track_pollock_lattice(ctx, faces.view(), inj, c.planes, c.times, prm);
        } // periodic faces freed here
        if (c.twin)
            pollock_twin_moments(Q, pq_twin_planes, pq_twin_time);
        for (int p = 0; p < Q.Np; ++p) {
            const std::size_t u = static_cast<std::size_t>(p);
            if (Q.weights[u] > 0.0 && (Q.status_plane[u] != 0 || Q.status_time[u] != 0))
                ++n_fail;
        }
        P(fmt("POLLOCK Np=%d seed_nonpos=%lld seeds_equal_tracker=%d planes=%d times=%d "
              "hist_plane=%s hist_time=%s cells_plane=%llu cells_time=%llu weighted_not_active=%lld "
              "repack_seconds=%.3f weights=rt0_v1_at_seed",
              Q.Np, Q.seed_nonpos, Q.seeds == res.seeds ? 1 : 0, Q.P, Q.T,
              pollock_hist_string(Q.status_hist_plane).c_str(),
              pollock_hist_string(Q.status_hist_time).c_str(), Q.cells_plane, Q.cells_time, n_fail,
              repack_s));
    }

    // ---- write into the temporary directory ---------------------------------------------------
    const auto t_write = Clock::now();
    TrackMeta meta;
    meta.field = "gaussian";
    meta.eps = c.eps;
    meta.N = c.nx;
    meta.h = h;
    meta.commit = "unknown"; // the build commit is wall-clock-like provenance: timings.json
    write_track_outputs(tmpdir, res, meta);
    strip_track_timings(tmpdir + "/track.json");
    test_abort(c, r, "partial");
    write_json(tmpdir + "/field.json", field_j);
    {   // deterministic per-realization status record (no wall-clock values)
        auto pass_json = [](const PassTotals& t, long long ge2) {
            return json{{"status_hist", std::vector<long long>(t.status_hist.begin(), t.status_hist.end())},
                        {"status_ge2", ge2},
                        {"backflow_encounters", t.backflow_encounters},
                        {"landings_discarded", t.landings_discarded}};
        };
        json sj = {{"Np", res.Np},
                   {"seed_backflow", res.seed_backflow},
                   {"tracker_plane_pass", pass_json(res.totals_plane, nonok_plane)},
                   {"tracker_time_pass", pass_json(res.totals_time, nonok_time)}};
        if (c.pollock_bridge)
            sj["pollock"] = {{"seed_nonpos", Q.seed_nonpos},
                             {"status_hist_plane", Q.status_hist_plane},
                             {"status_hist_time", Q.status_hist_time},
                             {"weighted_not_active", n_fail}};
        write_json(tmpdir + "/status.json", sj);
    }
    if (c.twin) {
        fs::create_directories(tmpdir + "/twin");
        write_tracker_planes_csv(tmpdir + "/twin/moments_planes.csv", twin_planes);
        write_series_csv(tmpdir + "/twin/moments_time.csv", twin_time);
    }
    if (c.pollock_bridge) {
        const std::string pd = tmpdir + "/pollock";
        fs::create_directories(pd);
        const std::size_t Np = static_cast<std::size_t>(Q.Np);
        const std::size_t PP = static_cast<std::size_t>(Q.P), TT = static_cast<std::size_t>(Q.T);
        write_f8(pd + "/pollock_weights.npy", {Np}, Q.weights);
        write_npy_raw(pd + "/pollock_status_plane.npy", "|u1", {Np}, Q.status_plane.data(),
                      Q.status_plane.size(), 1);
        write_npy_raw(pd + "/pollock_status_time.npy", "|u1", {Np}, Q.status_time.data(),
                      Q.status_time.size(), 1);
        write_f8(pd + "/pollock_planes_x23.npy", {Np, PP, 2}, Q.planes_x23);
        write_f8(pd + "/pollock_planes_t.npy", {Np, PP}, Q.planes_t);
        write_f8(pd + "/pollock_times_x.npy", {Np, TT, 3}, Q.times_x);
        write_npy_raw(pd + "/pollock_cells.npy", "<u8", {Np, 2}, Q.cells.data(), Q.cells.size(),
                      sizeof(std::uint64_t));
        write_series_csv(pd + "/pollock_moments_time.csv", Q.moments_time);
        write_pollock_planes_csv(pd + "/pollock_moments_planes.csv", Q.moments_planes);
        if (c.twin) {
            write_series_csv(pd + "/pollock_twin_moments_time.csv", pq_twin_time);
            write_pollock_planes_csv(pd + "/pollock_twin_moments_planes.csv", pq_twin_planes);
        }
        json pj = {{"comparator", "pollock"},
                   {"weights", "rt0_v1_at_seed"},
                   {"Np", Q.Np},
                   {"seed_nonpos", Q.seed_nonpos},
                   {"seeds_equal_tracker", Q.seeds == res.seeds},
                   {"max_cells_per_call", Q.max_cells_per_call},
                   {"planes", Q.planes},
                   {"times", Q.times},
                   {"status_hist_plane", Q.status_hist_plane},
                   {"status_hist_time", Q.status_hist_time},
                   {"cells_plane", Q.cells_plane},
                   {"cells_time", Q.cells_time},
                   {"twin", c.twin}};
        write_json(pd + "/pollock.json", pj);
    }
    std::vector<std::pair<std::string, std::string>> stats;
    write_series_csv(stats_tmp(layout, r), res.moments_time);
    stats.emplace_back(stats_tmp(layout, r), layout.realization_timeseries(r));
    if (c.twin) {
        write_series_csv(twin_stats_path(c, r) + ".tmp", twin_time);
        stats.emplace_back(twin_stats_path(c, r) + ".tmp", twin_stats_path(c, r));
    }
    const double write_s = since(t_write);

    TJ["seconds"] = {{"stage", stage_s},     {"prepare", prepare_s},      {"track", track_s},
                     {"bridge_repack", repack_s},
                     {"bridge_track", c.pollock_bridge ? Q.seconds_plane_kernel + Q.seconds_time_kernel
                                                       : 0.0},
                     {"write", write_s},     {"total", since(t_all)}};
    TJ["tracker_timings_s"] = tracker_timings;
    if (c.pollock_bridge)
        TJ["pollock_timings_s"] = {{"plane_kernel", Q.seconds_plane_kernel},
                                   {"time_kernel", Q.seconds_time_kernel},
                                   {"download", Q.seconds_download},
                                   {"moments", Q.seconds_moments}};
    TJ["stage_status"] = stage_status;
    TJ["mem_lines"] = mems;
    write_json(tmpdir + "/timings.json", TJ);

    commit_realization(c, r, tmpdir, stats);

    if (sum) {
        sum->flux_series[static_cast<std::size_t>(r)] = res.moments_time;
        sum->twin_series[static_cast<std::size_t>(r)] = twin_time;
        sum->twin_weights[static_cast<std::size_t>(r)] = twin_weights;
        sum->computed[static_cast<std::size_t>(r)] = true;
    }
    P(fmt("STATUS ok tracker_non_ok_plane=%lld tracker_non_ok_time=%lld seed_backflow=%lld "
          "pollock_weighted_not_active=%lld",
          nonok_plane, nonok_time, res.seed_backflow, n_fail));
    out_line(fmt("TIMING r=%d seconds=%.3f", r, since(t_all)));
    return kExitOk;
}

// ================================================================================================
// After the last realization: in-process analysis
// ================================================================================================

int run_analysis_outputs(const EnsembleConfig& c, const io::OutputLayout& L) {
    // The series are read back from the stats CSVs (not taken from memory) so that an
    // interrupted-and-restarted run and an uninterrupted one feed the analysis identical values.
    std::vector<std::vector<io::TimeSeriesPoint<double>>> all;
    for (int r = 0; r < c.nr; ++r) {
        all.push_back(analysis::read_timeseries_csv<double>(L.realization_timeseries(r)));
        if (all.back().empty())
            throw std::runtime_error("empty or unparseable " + L.realization_timeseries(r));
    }
    fs::create_directories(L.ensemble_dir());
    fs::create_directories(L.analysis_dir());
    if (!io::CsvTimeSeriesWriter::write_ensemble_mean(L.ensemble_timeseries(), all))
        throw std::runtime_error("cannot write " + L.ensemble_timeseries());
    const auto rows = analysis::compute_macrodispersion<double>(all, c.ell, 1.0);
    if (rows.empty())
        throw std::runtime_error("compute_macrodispersion returned no rows");
    if (!analysis::write_macrodispersion_csv(L.macrodispersion_csv(), rows))
        throw std::runtime_error("cannot write " + L.macrodispersion_csv());
    {
        std::ofstream f(c.out + "/analysis_config.yaml", std::ios::trunc);
        f << "# written by lagrangian_ensemble (SF-38 C-RUNNER); box units: lambda = ell, qbar = 1.\n"
          << "# output_dir is relative: run as  cd DIR && macroflow3d_pipeline analysis_config.yaml\n"
          << "# (reads stats/, rewrites ensemble/ and analysis/ in DIR; byte-identical for any DIR).\n"
          << "run_mode: analysis_only\n"
          << "output:\n"
          << "  output_dir: \".\"\n"
          << "analysis:\n"
          << "  macrodispersion:\n"
          << "    enabled: true\n"
          << "    NR: " << c.nr << "\n"
          << "    lambda: " << fmt("%.17g", c.ell) << "\n"
          << "    vmean_norm: 1\n";
        if (!f)
            throw std::runtime_error("cannot write analysis_config.yaml");
    }
    out_line(fmt("ANALYSIS nr=%d rows=%zu lambda=%.17g vmean_norm=1 wrote %s %s", c.nr, rows.size(),
                 c.ell, L.ensemble_timeseries().c_str(), L.macrodispersion_csv().c_str()));
    return kExitOk;
}

} // namespace

// ================================================================================================
// Public API
// ================================================================================================

std::string ensemble_usage() {
    return "usage: lagrangian_ensemble --grid nx,ny,nz --ell L --eps E --seed0 S --out DIR [options]\n"
           "  --grid nx,ny,nz        box [0, nx/ny] x [0,1]^2, h = 1/ny (ny == nz, even, >= 4)\n"
           "  --sigma2 V             SF-18 variance (default 1)\n"
           "  --ell L                correlation length in box units (= lambda)\n"
           "  --eps E                k = exp(E Y) (required unless --field-only)\n"
           "  --seed0 S              realization r uses seed S + r\n"
           "  --nr N                 number of realizations (default 1)\n"
           "  --x0 X                 injection plane (default 0)\n"
           "  --window a,b,c,d       injection window w2a,w2b,w3a,w3b (default 0.1,0.9,0.1,0.9)\n"
           "  --np-side M            M x M seed lattice (default 100)\n"
           "  --planes <list>|auto   unwrapped x1 planes (auto: x0+m*ell, m=1,2,4,.. <= x0+L1, plus\n"
           "                         x0+L1-10*ell and x0+L1)\n"
           "  --times <list>|auto    snapshot times (auto: n*ell/2, n=1..ceil(2*L1/ell))\n"
           "  --tol T                integrator tolerance (default 1e-8)\n"
           "  --hmax-div D           h_max = h/D (default 8)\n"
           "  --pcg-rtol R           SF-19 PCG relative tolerance (default 1e-10)\n"
           "  --max-arclength A      cap on the arclength per pass (default: tracker rule); particles\n"
           "                         exceeding it get a non-ok status (recorded, never an exit)\n"
           "  --fused-splines | --single-launch\n"
           "  --twin on|off          uniform-weight twin series (default on)\n"
           "  --pollock-bridge on|off  Pollock comparator on the SF-19 faces (default off)\n"
           "  --field-only           SF-18 only: FIELD line + field.json per realization\n"
           "  --out DIR              output directory\n"
           "exit codes: 0 ok (every realization completed; non-ok particles are recorded in status.json\n"
           "            and the STATUS line, not an exit condition), 1 exception, 2 usage, 3 seed on a\n"
           "            cell face, 4 stage failure, 7 DONE realization under different options\n"
           "analysis_config.yaml (written in DIR, output_dir \".\"): cd DIR && macroflow3d_pipeline analysis_config.yaml\n";
}

std::vector<double> auto_planes(double x0, double L1, double lambda) {
    std::vector<double> v;
    for (long long m = 1;; m *= 2) {
        const double x = x0 + static_cast<double>(m) * lambda;
        if (x > x0 + L1)
            break;
        v.push_back(x);
    }
    v.push_back(x0 + L1 - 10.0 * lambda);
    v.push_back(x0 + L1);
    std::sort(v.begin(), v.end());
    // shortcut: near-equal planes (|dx| <= 1e-12 max(1,|x|), rounding of x0+m*ell vs x0+L1) are one plane.
    std::vector<double> u;
    for (double x : v)
        if (u.empty() || x - u.back() > 1e-12 * std::max(1.0, std::fabs(x)))
            u.push_back(x);
    return u;
}

std::vector<double> auto_times(double L1, double lambda) {
    // 1e-9 guards ceil() against 2*L1/lambda = k + rounding when the ratio is an exact integer.
    const long long nmax = static_cast<long long>(std::ceil(2.0 * L1 / lambda - 1e-9));
    std::vector<double> t;
    for (long long n = 1; n <= nmax; ++n)
        t.push_back(static_cast<double>(n) * lambda / 2.0);
    return t;
}

bool find_seed_on_face(const InjectionSpec& inj, int n_transverse, std::string* what) {
    const std::vector<double> s = make_seed_lattice(inj);
    const std::size_t Np = s.size() / 3;
    for (std::size_t p = 0; p < Np; ++p) {
        for (int k = 1; k <= 2; ++k) {
            const double v = static_cast<double>(n_transverse) * s[3 * p + static_cast<std::size_t>(k)];
            if (!(std::fabs(v - std::round(v)) > 1e-6)) {
                if (what)
                    *what = fmt("seed p=%zu (m2=%zu, m3=%zu) x%d=%.17g: n_T*x = %.17g is within 1e-6 "
                                "of the integer %.0f (transverse cell face)",
                                p, p / static_cast<std::size_t>(inj.M),
                                p % static_cast<std::size_t>(inj.M), k + 1,
                                s[3 * p + static_cast<std::size_t>(k)], v, std::round(v));
                return true;
            }
        }
    }
    return false;
}

std::string ensemble_signature(const EnsembleConfig& c) {
    return fmt("grid=%dx%dx%d sigma2=%.17g ell=%.17g eps=%.17g seed0=%llu x0=%.17g "
               "window=%.17g,%.17g,%.17g,%.17g M=%d planes=%s times=%s tol=%.17g hmax_div=%d "
               "max_arclength=%.17g pcg_rtol=%.17g fused=%d single_launch=%d twin=%d bridge=%d field_only=%d",
               c.nx, c.ny, c.nz, c.sigma2, c.ell, c.eps, c.seed0, c.x0, c.window[0], c.window[1],
               c.window[2], c.window[3], c.np_side, join_list17(c.planes).c_str(),
               join_list17(c.times).c_str(), c.tol, c.hmax_div, c.max_arclength, c.pcg_rtol, c.fused_splines ? 1 : 0,
               c.single_launch ? 1 : 0, c.twin ? 1 : 0, c.pollock_bridge ? 1 : 0,
               c.field_only ? 1 : 0);
}

EnsembleConfig parse_ensemble_args(int argc, const char* const* argv) {
    static const std::set<std::string> value_opts = {
        "--grid",   "--sigma2",  "--ell",     "--eps",      "--seed0",           "--nr",
        "--x0",     "--window",  "--np-side", "--planes",   "--times",           "--tol",
        "--hmax-div", "--twin",  "--pollock-bridge", "--out", "--pcg-rtol", "--max-arclength", "--test-abort-after"};
    static const std::set<std::string> flag_opts = {"--fused-splines", "--single-launch",
                                                    "--field-only", "--help"};
    EnsembleConfig c;
    std::map<std::string, std::string> val;
    std::set<std::string> flags;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        c.command_line += (i > 1 ? " " : "") + a;
        if (flag_opts.count(a)) {
            if (!flags.insert(a).second)
                throw UsageError("option " + a + " given twice");
        } else if (value_opts.count(a)) {
            if (val.count(a))
                throw UsageError("option " + a + " given twice");
            if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
                throw UsageError("option " + a + " needs a value");
            val[a] = argv[++i];
            c.command_line += " " + val[a];
        } else {
            throw UsageError("unknown option '" + a + "'");
        }
    }
    if (flags.count("--help"))
        throw UsageError(ensemble_usage());
    c.field_only = flags.count("--field-only") != 0;
    c.fused_splines = flags.count("--fused-splines") != 0;
    c.single_launch = flags.count("--single-launch") != 0;
    if (c.field_only) {
        for (const char* o : {"--eps", "--x0", "--window", "--np-side", "--planes", "--times", "--tol",
                              "--hmax-div", "--twin", "--pollock-bridge", "--pcg-rtol", "--max-arclength",
                              "--fused-splines", "--single-launch"})
            if (val.count(o) || flags.count(o))
                throw UsageError(std::string("--field-only contradicts ") + o +
                                 " (field-only runs SF-18 only: no flow, no tracking)");
    }
    for (const char* o : {"--grid", "--ell", "--seed0", "--out"})
        if (!val.count(o))
            throw UsageError(std::string("missing required option ") + o);
    if (!c.field_only && !val.count("--eps"))
        throw UsageError("missing required option --eps");

    {
        const std::vector<double> g = parse_list("--grid", val["--grid"]);
        if (g.size() != 3)
            throw UsageError("--grid: expected nx,ny,nz");
        for (double d : g)
            if (d != std::floor(d) || d < 4 || d > 1e6)
                throw UsageError("--grid: extents must be integers >= 4");
        c.nx = static_cast<int>(g[0]);
        c.ny = static_cast<int>(g[1]);
        c.nz = static_cast<int>(g[2]);
        if (c.ny != c.nz)
            throw UsageError("--grid: ny must equal nz (the lean stage takes one transverse N)");
        if (c.nx % 2 != 0 || c.ny % 2 != 0)
            throw UsageError("--grid: extents must be even");
    }
    if (val.count("--sigma2"))
        c.sigma2 = parse_double("--sigma2", val["--sigma2"]);
    if (!(c.sigma2 >= 0.0))
        throw UsageError("--sigma2: must be >= 0");
    c.ell = parse_double("--ell", val["--ell"]);
    if (!(c.ell > 0.0))
        throw UsageError("--ell: must be > 0");
    if (val.count("--eps"))
        c.eps = parse_double("--eps", val["--eps"]);
    c.seed0 = parse_u64("--seed0", val["--seed0"]);
    if (val.count("--nr")) {
        const long long nr = parse_int("--nr", val["--nr"]);
        if (nr < 1 || nr > 100000)
            throw UsageError("--nr: must be in [1, 100000]");
        c.nr = static_cast<int>(nr);
    }
    if (val.count("--x0"))
        c.x0 = parse_double("--x0", val["--x0"]);
    if (val.count("--window")) {
        const std::vector<double> w = parse_list("--window", val["--window"]);
        if (w.size() != 4)
            throw UsageError("--window: expected w2a,w2b,w3a,w3b");
        for (int k = 0; k < 4; ++k)
            c.window[k] = w[static_cast<std::size_t>(k)];
    }
    if (!(c.window[0] < c.window[1]) || !(c.window[2] < c.window[3]))
        throw UsageError("--window: need w2a < w2b and w3a < w3b");
    if (val.count("--np-side")) {
        const long long m = parse_int("--np-side", val["--np-side"]);
        if (m < 1 || m > 4096)
            throw UsageError("--np-side: must be in [1, 4096]");
        c.np_side = static_cast<int>(m);
    }
    if (val.count("--tol"))
        c.tol = parse_double("--tol", val["--tol"]);
    if (!(c.tol > 0.0))
        throw UsageError("--tol: must be > 0");
    if (val.count("--hmax-div")) {
        const long long d = parse_int("--hmax-div", val["--hmax-div"]);
        if (d < 1)
            throw UsageError("--hmax-div: must be >= 1");
        c.hmax_div = static_cast<int>(d);
    }
    if (val.count("--max-arclength")) {
        c.max_arclength = parse_double("--max-arclength", val["--max-arclength"]);
        if (!(c.max_arclength > 0.0))
            throw UsageError("--max-arclength: must be > 0 (omit for the tracker default)");
    }
    if (val.count("--pcg-rtol"))
        c.pcg_rtol = parse_double("--pcg-rtol", val["--pcg-rtol"]);
    if (!(c.pcg_rtol > 0.0))
        throw UsageError("--pcg-rtol: must be > 0");
    if (val.count("--twin"))
        c.twin = parse_onoff("--twin", val["--twin"]);
    if (val.count("--pollock-bridge"))
        c.pollock_bridge = parse_onoff("--pollock-bridge", val["--pollock-bridge"]);
    c.out = val["--out"];
    if (c.out.empty())
        throw UsageError("--out: empty");
    if (val.count("--test-abort-after")) {
        const std::string& s = val["--test-abort-after"];
        const std::size_t comma = s.find(',');
        if (comma == std::string::npos)
            throw UsageError("--test-abort-after: expected r,phase");
        c.test_abort_r = static_cast<int>(parse_int("--test-abort-after", s.substr(0, comma)));
        c.test_abort_phase = s.substr(comma + 1);
        if (c.test_abort_phase != "stage" && c.test_abort_phase != "partial" &&
            c.test_abort_phase != "tmp" && c.test_abort_phase != "rename")
            throw UsageError("--test-abort-after: phase must be stage|partial|tmp|rename");
        if (c.field_only && c.test_abort_phase == "partial")
            throw UsageError("--test-abort-after: phase 'partial' does not exist with --field-only");
    }

    if (!c.field_only) {
        const double L1 = static_cast<double>(c.nx) / static_cast<double>(c.ny);
        const bool pa = !val.count("--planes") || val["--planes"] == "auto";
        const bool ta = !val.count("--times") || val["--times"] == "auto";
        c.planes_auto = pa;
        c.times_auto = ta;
        if (pa) {
            if (!(L1 > 10.0 * c.ell))
                throw UsageError("--planes auto: needs L1 > 10 ell for the twin plane x0+L1-10*ell "
                                 "(give --planes explicitly)");
            c.planes = auto_planes(c.x0, L1, c.ell);
        } else {
            c.planes = parse_list("--planes", val["--planes"]);
        }
        c.times = ta ? auto_times(L1, c.ell) : parse_list("--times", val["--times"]);
        for (std::size_t i = 0; i < c.planes.size(); ++i)
            if (!(c.planes[i] > c.x0) || (i > 0 && !(c.planes[i] > c.planes[i - 1])))
                throw UsageError("--planes: must be strictly increasing and > x0");
        for (std::size_t i = 0; i < c.times.size(); ++i)
            if (!(c.times[i] > 0.0) || (i > 0 && !(c.times[i] > c.times[i - 1])))
                throw UsageError("--times: must be strictly increasing and > 0");
        if (c.times.size() < 2)
            throw UsageError("--times: at least 2 snapshot times are needed (macrodispersion "
                             "differentiates the series)");
    }
    return c;
}

int run_ensemble(CudaContext& ctx, const EnsembleConfig& c, EnsembleRunSummary* sum) {
    io::OutputLayout layout(c.out);
    if (!c.field_only) {
        std::string what;
        if (find_seed_on_face(make_injection(c), c.ny, &what)) {
            std::fprintf(stderr, "ERROR: seed-face assertion (RR12) failed before tracking: %s\n",
                         what.c_str());
            out_line("SEED_FACE_FAILED " + what);
            return kExitSeedFace;
        }
        out_line(fmt("RESOLVED grid=%dx%dx%d L1=%.10g lambda=%.10g planes(%s)=%s times(%s)=%s",
                     c.nx, c.ny, c.nz, static_cast<double>(c.nx) / c.ny, c.ell,
                     c.planes_auto ? "auto" : "list", join_list(c.planes).c_str(),
                     c.times_auto ? "auto" : "list", join_list(c.times).c_str()));
    }
    if (sum) {
        const std::size_t n = static_cast<std::size_t>(c.nr);
        sum->flux_series.assign(n, {});
        sum->twin_series.assign(n, {});
        sum->twin_weights.assign(n, {});
        sum->computed.assign(n, false);
    }

    fs::create_directories(c.out + "/realizations");
    if (!c.field_only) {
        fs::create_directories(layout.stats_dir());
        if (c.twin)
            fs::create_directories(c.out + "/stats_twin");
    }
    // restart: delete stale temporaries and unfinished realization directories
    for (const std::string& d : {c.out + "/realizations", layout.stats_dir(), c.out + "/stats_twin"}) {
        if (!fs::exists(d))
            continue;
        std::vector<fs::path> stale;
        for (const auto& e : fs::directory_iterator(d))
            if (e.path().extension() == ".tmp")
                stale.push_back(e.path());
        for (const fs::path& p : stale)
            fs::remove_all(p);
    }
    for (int r = 0; r < c.nr; ++r) {
        const std::string d = rdir_of(c, r);
        if (fs::exists(d) && !fs::exists(d + "/DONE"))
            fs::remove_all(d);
    }

    for (int r = 0; r < c.nr; ++r) {
        const std::string d = rdir_of(c, r);
        if (fs::exists(d + "/DONE")) {
            if (slurp(d + "/DONE") != done_text(c, r)) {
                std::fprintf(stderr,
                             "ERROR: %s/DONE was written under different options; refusing to mix "
                             "(use a fresh --out)\n",
                             d.c_str());
                return kExitConfigMismatch;
            }
            out_line(fmt("R=%d SKIP done (DONE present)", r));
            continue;
        }
        const int rc = c.field_only ? run_field_only(ctx, c, r) : run_realization(ctx, c, r, layout, sum);
        if (rc != kExitOk)
            return rc;
    }
    if (!c.field_only)
        run_analysis_outputs(c, layout);
    out_line(fmt("RUN_DONE nr=%d", c.nr));
    return kExitOk;
}

} // namespace lagrangian_labels
} // namespace macroflow3d
