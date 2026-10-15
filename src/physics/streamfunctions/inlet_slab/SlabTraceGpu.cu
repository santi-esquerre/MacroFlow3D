/**
 * @file SlabTraceGpu.cu
 * @brief SF-35 N6: GPU tracer (device copies of the SF-30 integrator, one thread per seed) and
 *        compute_oracle_gpu. See SlabTraceGpu.cuh. This TU is compiled with --fmad=false.
 */

#include "SlabTraceGpu.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace trace_dev {

using closure_gate::StreamlineStatus;

// Device copy of closure_gate::detail::DP5 (same constexpr expressions).
struct DP5 {
    static constexpr double c2 = 1.0 / 5.0, c3 = 3.0 / 10.0, c4 = 4.0 / 5.0, c5 = 8.0 / 9.0;
    static constexpr double a21 = 1.0 / 5.0;
    static constexpr double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
    static constexpr double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
    static constexpr double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0,
                            a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
    static constexpr double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0,
                            a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0;
    static constexpr double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0,
                            b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    static constexpr double e1 = 71.0 / 57600.0, e3 = -71.0 / 16695.0, e4 = 71.0 / 1920.0,
                            e5 = -17253.0 / 339200.0, e6 = 22.0 / 525.0, e7 = -1.0 / 40.0;
};

enum class EvalCode { good, invalid_direction, invalid_conductivity, nonpositive_g1 };
enum class LandingOutcome { landed, backflow, invalid_direction, invalid_conductivity, underflow };

constexpr int kDim = 4;

/// Device mirror of the StreamlineResult fields used by the oracle.
struct DevResult {
    int status = 0;
    int periods_completed = 0;
    bool backflow_encounter = false;
    double min_g1_hat = 0.0;
    double backflow_arclength = 0.0;
    long long accepted_steps = 0, rejected_steps = 0, landing_steps = 0, landing_rejected = 0;
    long long landings_discarded = 0, field_evaluations = 0;
    double rec_x2 = 0.0, rec_x3 = 0.0; ///< landed (x2, x3) at the plane
};

__device__ inline bool finite3(const double g[3]) {
    return isfinite(g[0]) && isfinite(g[1]) && isfinite(g[2]);
}

// std::max(a, b) = (a < b) ? b : a ; std::min(a, b) = (b < a) ? b : a (host semantics).
__device__ inline double hmax(double a, double b) {
    return (a < b) ? b : a;
}
__device__ inline double hmin(double a, double b) {
    return (b < a) ? b : a;
}

template <class Field> struct ArclengthRhs {
    const Field& field;
    double sigma;
    long long& nfev;
    double last_g1 = 0.0;
    double last_g1_hat = 0.0;

    __device__ EvalCode operator()(double /*s*/, const double y[kDim], double dy[kDim]) {
        const double x[3] = {y[0], y[1], y[2]};
        double g[3] = {0.0, 0.0, 0.0};
        double k = 0.0;
        field(x, g, k);
        ++nfev;
        if (!finite3(g)) {
            return EvalCode::invalid_direction;
        }
        const double gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (!(gn > 0.0) || !isfinite(gn)) {
            return EvalCode::invalid_direction;
        }
        if (!isfinite(k) || !(k > 0.0)) {
            return EvalCode::invalid_conductivity;
        }
        dy[0] = sigma * g[0] / gn;
        dy[1] = sigma * g[1] / gn;
        dy[2] = sigma * g[2] / gn;
        dy[3] = 1.0 / (k * gn);
        last_g1 = g[0];
        last_g1_hat = g[0] / gn;
        return EvalCode::good;
    }
};

template <class Field> struct LandingRhs {
    const Field& field;
    double sigma;
    long long& nfev;

    __device__ EvalCode operator()(double x1, const double y[kDim], double dy[kDim]) {
        const double x[3] = {x1, y[0], y[1]};
        double g[3] = {0.0, 0.0, 0.0};
        double k = 0.0;
        field(x, g, k);
        ++nfev;
        if (!finite3(g)) {
            return EvalCode::invalid_direction;
        }
        const double gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (!(gn > 0.0) || !isfinite(gn)) {
            return EvalCode::invalid_direction;
        }
        if (!isfinite(k) || !(k > 0.0)) {
            return EvalCode::invalid_conductivity;
        }
        if (!(g[0] > 0.0)) {
            return EvalCode::nonpositive_g1;
        }
        dy[0] = g[1] / g[0];
        dy[1] = g[2] / g[0];
        dy[2] = sigma * gn / g[0];
        dy[3] = sigma / (k * g[0]);
        return EvalCode::good;
    }
};

template <class Rhs>
__device__ EvalCode dp5_step(Rhs& rhs, double t, double t_end, const double y[kDim],
                             const double k1[kDim], double h, int n_err, double tol,
                             double y5[kDim], double k7[kDim], double& err) {
    using C = DP5;
    double k2[kDim], k3[kDim], k4[kDim], k5[kDim], k6[kDim], yt[kDim];
    EvalCode code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a21 * k1[i]);
    if ((code = rhs(t + C::c2 * h, yt, k2)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a31 * k1[i] + C::a32 * k2[i]);
    if ((code = rhs(t + C::c3 * h, yt, k3)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a41 * k1[i] + C::a42 * k2[i] + C::a43 * k3[i]);
    if ((code = rhs(t + C::c4 * h, yt, k4)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a51 * k1[i] + C::a52 * k2[i] + C::a53 * k3[i] + C::a54 * k4[i]);
    if ((code = rhs(t + C::c5 * h, yt, k5)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        yt[i] = y[i] + h * (C::a61 * k1[i] + C::a62 * k2[i] + C::a63 * k3[i] + C::a64 * k4[i] +
                            C::a65 * k5[i]);
    if ((code = rhs(t_end, yt, k6)) != EvalCode::good)
        return code;

    for (int i = 0; i < kDim; ++i)
        y5[i] = y[i] + h * (C::b1 * k1[i] + C::b3 * k3[i] + C::b4 * k4[i] + C::b5 * k5[i] +
                            C::b6 * k6[i]);
    if ((code = rhs(t_end, y5, k7)) != EvalCode::good)
        return code;

    err = 0.0;
    for (int i = 0; i < n_err; ++i) {
        const double e = h * (C::e1 * k1[i] + C::e3 * k3[i] + C::e4 * k4[i] + C::e5 * k5[i] +
                              C::e6 * k6[i] + C::e7 * k7[i]);
        err = hmax(err, fabs(e) / tol);
    }
    return EvalCode::good;
}

__device__ inline double step_factor(double err) {
    if (!(err > 0.0)) {
        return 5.0;
    }
    return hmin(5.0, hmax(0.2, 0.9 * pow(err, -0.2)));
}

__device__ inline int status_from(EvalCode c) {
    return c == EvalCode::invalid_conductivity
               ? static_cast<int>(StreamlineStatus::invalid_conductivity)
               : static_cast<int>(StreamlineStatus::stagnation);
}

__device__ inline LandingOutcome map_code(EvalCode c) {
    switch (c) {
    case EvalCode::nonpositive_g1:
        return LandingOutcome::backflow;
    case EvalCode::invalid_conductivity:
        return LandingOutcome::invalid_conductivity;
    default:
        return LandingOutcome::invalid_direction;
    }
}

template <class Field>
__device__ LandingOutcome land_on_plane(const Field& field, const closure_gate::IntegratorOptions& opt,
                                        double x1_a, const double ya[kDim], double X,
                                        double y_out[kDim], DevResult& res) {
    LandingRhs<Field> rhs{field, static_cast<double>(opt.sigma), res.field_evaluations};
    double t = x1_a;
    double y[kDim] = {ya[0], ya[1], ya[2], ya[3]};
    double k1[kDim];
    EvalCode code = rhs(t, y, k1);
    if (code != EvalCode::good) {
        return map_code(code);
    }
    const double dir = (X - t) >= 0.0 ? 1.0 : -1.0;
    double dt_mag = hmin(opt.h_max, fabs(X - t));
    long long guard = 0;
    while (true) {
        const double remaining = X - t;
        bool last = false;
        double dt = dir * dt_mag;
        if (dt_mag >= fabs(remaining)) {
            dt = remaining;
            last = true;
        }
        const double t_end = last ? X : t + dt;
        double y5[kDim], k7[kDim], err = 0.0;
        code = dp5_step(rhs, t, t_end, y, k1, dt, 2, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            return map_code(code);
        }
        const double fac = step_factor(err);
        if (err <= 1.0) {
            ++res.landing_steps;
            for (int i = 0; i < kDim; ++i) {
                y[i] = y5[i];
                k1[i] = k7[i];
            }
            if (last) {
                for (int i = 0; i < kDim; ++i)
                    y_out[i] = y[i];
                return LandingOutcome::landed;
            }
            t = t_end;
            dt_mag = hmin(opt.h_max, fabs(dt) * fac);
        } else {
            ++res.landing_rejected;
            dt_mag = fabs(dt) * fac;
            if (dt_mag < opt.min_step) {
                return LandingOutcome::underflow;
            }
        }
        if (++guard > 100000000LL) {
            return LandingOutcome::underflow;
        }
    }
}

/// Device copy of inlet_slab::trace_to_plane (options pre-validated on the host).
template <class Field>
__device__ DevResult trace_to_plane(const Field& field, const double seed[3], double X,
                                    const closure_gate::IntegratorOptions& opt) {
    DevResult res;
    const double sigma = static_cast<double>(opt.sigma);
    ArclengthRhs<Field> rhs{field, sigma, res.field_evaluations};

    double y[kDim] = {seed[0], seed[1], seed[2], 0.0};
    double s = 0.0;
    double f[kDim];

    EvalCode code = rhs(0.0, y, f);
    if (code != EvalCode::good) {
        res.status = status_from(code);
        return res;
    }
    res.min_g1_hat = rhs.last_g1_hat;
    if (!(rhs.last_g1 > 0.0)) {
        res.status = static_cast<int>(StreamlineStatus::seed_backflow);
        return res;
    }

    const double s_plane = 0.0;
    double h = 0.1 * opt.h_max;

    while (true) {
        double y5[kDim], k7[kDim], err = 0.0;
        code = dp5_step(rhs, s, s + h, y, f, h, 3, opt.tol, y5, k7, err);
        if (code != EvalCode::good) {
            res.status = status_from(code);
            return res;
        }
        const double g1_end = rhs.last_g1;
        const double g1_hat_end = rhs.last_g1_hat;
        const double fac = step_factor(err);

        if (!(err <= 1.0)) {
            ++res.rejected_steps;
            h = h * fac;
            if (h < opt.min_step) {
                res.status = static_cast<int>(StreamlineStatus::step_underflow);
                return res;
            }
            continue;
        }

        const bool crossing = sigma * (y5[0] - X) >= 0.0;
        if (crossing) {
            const double ya[kDim] = {y[1], y[2], s, y[3]};
            double yl[kDim];
            const LandingOutcome out = land_on_plane(field, opt, y[0], ya, X, yl, res);
            if (out == LandingOutcome::backflow) {
                ++res.landings_discarded;
                h = 0.5 * h;
                if (h < opt.min_step) {
                    res.status = static_cast<int>(StreamlineStatus::landing_failed);
                    return res;
                }
                continue;
            }
            if (out == LandingOutcome::invalid_direction) {
                res.status = static_cast<int>(StreamlineStatus::stagnation);
                return res;
            }
            if (out == LandingOutcome::invalid_conductivity) {
                res.status = static_cast<int>(StreamlineStatus::invalid_conductivity);
                return res;
            }
            if (out == LandingOutcome::underflow) {
                res.status = static_cast<int>(StreamlineStatus::landing_failed);
                return res;
            }
            if (yl[2] - s_plane > opt.max_arclength_per_period) {
                res.status = static_cast<int>(StreamlineStatus::cap_exceeded);
                return res;
            }
            y[0] = X;
            y[1] = yl[0];
            y[2] = yl[1];
            y[3] = yl[3];
            s = yl[2];
            res.rec_x2 = y[1];
            res.rec_x3 = y[2];
            res.periods_completed = 1;
            code = rhs(s, y, f);
            if (code != EvalCode::good) {
                res.status = status_from(code);
                return res;
            }
            res.min_g1_hat = hmin(res.min_g1_hat, rhs.last_g1_hat);
            if (!(rhs.last_g1 > 0.0)) {
                res.backflow_encounter = true;
            }
            res.status = static_cast<int>(StreamlineStatus::ok);
            return res;
        }

        ++res.accepted_steps;
        for (int i = 0; i < kDim; ++i) {
            y[i] = y5[i];
            f[i] = k7[i];
        }
        s += h;
        res.min_g1_hat = hmin(res.min_g1_hat, g1_hat_end);
        if (!(g1_end > 0.0)) {
            res.backflow_encounter = true;
            res.backflow_arclength += h;
        }
        if (s - s_plane > opt.max_arclength_per_period) {
            res.status = static_cast<int>(StreamlineStatus::cap_exceeded);
            return res;
        }
        h = hmin(opt.h_max, h * fac);
    }
}

} // namespace trace_dev

namespace {

struct TraceKernelArgs {
    int N;
    std::size_t total, n2;
    InletSlabGrid grid;
    closure_gate::IntegratorOptions base;
    double* foot_y;
    double* foot_z;
    double* roundtrip;
    long long *nfev_b, *nfev_f, *ldisc_b;
    std::uint8_t *st_b, *st_f, *bfe;
    double* ming_b;
};

__global__ void trace_kernel(const DeviceSplineDirectionField field, const TraceKernelArgs a) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + static_cast<std::size_t>(threadIdx.x);
    if (i >= a.total) {
        return;
    }
    const int j = static_cast<int>(i / a.n2) + 1;
    const std::size_t rem = i % a.n2;
    const int m2 = static_cast<int>(rem / static_cast<std::size_t>(a.N));
    const int m3 = static_cast<int>(rem % static_cast<std::size_t>(a.N));
    const double seed[3] = {a.grid.coord(j), a.grid.coord(m2), a.grid.coord(m3)};

    const double nan = __longlong_as_double(0x7ff8000000000000LL);
    a.foot_y[i] = nan;
    a.foot_z[i] = nan;
    a.roundtrip[i] = nan;
    a.nfev_f[i] = 0;
    a.st_f[i] = 0;

    closure_gate::IntegratorOptions ob = a.base;
    ob.sigma = -1;
    const trace_dev::DevResult rb = trace_dev::trace_to_plane(field, seed, 0.0, ob);
    a.nfev_b[i] = rb.field_evaluations;
    a.st_b[i] = static_cast<std::uint8_t>(rb.status);
    a.ldisc_b[i] = rb.landings_discarded;
    a.ming_b[i] = rb.min_g1_hat;
    a.bfe[i] = 0;
    if (rb.status != static_cast<int>(closure_gate::StreamlineStatus::ok)) {
        return;
    }
    const double fy = rb.rec_x2, fz = rb.rec_x3;
    closure_gate::IntegratorOptions of = a.base;
    of.sigma = +1;
    const double fseed[3] = {0.0, fy, fz};
    const trace_dev::DevResult rf = trace_dev::trace_to_plane(field, fseed, seed[0], of);
    a.nfev_f[i] = rf.field_evaluations;
    a.st_f[i] = static_cast<std::uint8_t>(rf.status);
    a.bfe[i] = (rb.backflow_encounter || rf.backflow_encounter) ? 1 : 0;
    if (rf.status != static_cast<int>(closure_gate::StreamlineStatus::ok)) {
        return;
    }
    a.foot_y[i] = fy;
    a.foot_z[i] = fz;
    a.roundtrip[i] = trace_dev::hmax(fabs(rf.rec_x2 - seed[1]), fabs(rf.rec_x3 - seed[2]));
}

template <class T> void download_vec(std::vector<T>& h, const T* d, std::size_t n) {
    h.resize(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
}

} // namespace

void SlabGpuTracer::prepare(const CudaContext& /*ctx*/, const ProductionStage& stage) {
    if (!stage.ok()) {
        throw std::logic_error("SlabGpuTracer::prepare: production stage is not ok");
    }
    require_valid_grid(stage.grid, "SlabGpuTracer::prepare");
    const SlabSplineDirectionField hf = stage.direction_field();
    const std::size_t nc = stage.potential_coefficients.size();
    if (stage.logk_coefficients.size() != nc) {
        throw std::logic_error("SlabGpuTracer::prepare: coefficient arrays differ in size");
    }
    n_ = stage.grid.n;
    coef_pot_ = DeviceBuffer<real>(nc);
    coef_logk_ = DeviceBuffer<real>(nc);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(coef_pot_.data(), stage.potential_coefficients.data(),
                                      nc * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(coef_logk_.data(), stage.logk_coefficients.data(),
                                      nc * sizeof(real), cudaMemcpyHostToDevice));
    field_.potential = hf.potential;
    field_.potential.coeff = coef_pot_.data();
    field_.log_conductivity = hf.log_conductivity;
    field_.log_conductivity.coeff = coef_logk_.data();
    for (int d = 0; d < 3; ++d)
        field_.G[d] = hf.G[d];

    const std::size_t total = stage.grid.field_size();
    d_foot_y_ = DeviceBuffer<double>(total);
    d_foot_z_ = DeviceBuffer<double>(total);
    d_rt_ = DeviceBuffer<double>(total);
    d_ming_ = DeviceBuffer<double>(total);
    d_nfev_b_ = DeviceBuffer<long long>(total);
    d_nfev_f_ = DeviceBuffer<long long>(total);
    d_ldisc_ = DeviceBuffer<long long>(total);
    d_st_b_ = DeviceBuffer<std::uint8_t>(total);
    d_st_f_ = DeviceBuffer<std::uint8_t>(total);
    d_bfe_ = DeviceBuffer<std::uint8_t>(total);
    // SF-33 C2: legacy-stream copies must land before ctx-stream work.
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    prepared_ = true;
}

const void* SlabGpuTracer::storage_ptr(int i) const {
    switch (i) {
    case 0: return coef_pot_.data();
    case 1: return coef_logk_.data();
    case 2: return d_foot_y_.data();
    case 3: return d_foot_z_.data();
    case 4: return d_rt_.data();
    case 5: return d_ming_.data();
    case 6: return d_nfev_b_.data();
    case 7: return d_nfev_f_.data();
    case 8: return d_ldisc_.data();
    case 9: return d_st_b_.data();
    case 10: return d_st_f_.data();
    default: return d_bfe_.data();
    }
}

void SlabGpuTracer::trace(CudaContext& ctx, const InletSlabGrid& grid,
                          const closure_gate::IntegratorOptions& base, SlabGpuTraceOutputs& out) {
    if (!prepared_) {
        throw std::logic_error("SlabGpuTracer::trace: prepare() not called");
    }
    require_valid_grid(grid, "SlabGpuTracer::trace");
    if (grid.n != n_) {
        throw std::invalid_argument("SlabGpuTracer::trace: grid does not match the prepared stage");
    }
    closure_gate::IntegratorOptions b = base;
    b.n_periods = 1;
    closure_gate::detail::validate_options(b);

    const std::size_t total = grid.field_size();
    TraceKernelArgs a;
    a.N = grid.n;
    a.total = total;
    a.n2 = grid.plane_size();
    a.grid = grid;
    a.base = b;
    a.foot_y = d_foot_y_.data();
    a.foot_z = d_foot_z_.data();
    a.roundtrip = d_rt_.data();
    a.nfev_b = d_nfev_b_.data();
    a.nfev_f = d_nfev_f_.data();
    a.ldisc_b = d_ldisc_.data();
    a.st_b = d_st_b_.data();
    a.st_f = d_st_f_.data();
    a.bfe = d_bfe_.data();
    a.ming_b = d_ming_.data();

    const auto t0 = std::chrono::steady_clock::now();
    const unsigned blocks = static_cast<unsigned>((total + kBlock - 1) / kBlock);
    trace_kernel<<<blocks, kBlock, 0, ctx.cuda_stream()>>>(field_, a);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    ctx.synchronize();
    out.seconds_kernel = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    download_vec(out.foot_y, d_foot_y_.data(), total);
    download_vec(out.foot_z, d_foot_z_.data(), total);
    download_vec(out.roundtrip, d_rt_.data(), total);
    download_vec(out.st_b, d_st_b_.data(), total);
    download_vec(out.st_f, d_st_f_.data(), total);
    download_vec(out.bfe, d_bfe_.data(), total);
    download_vec(out.nfev_b, d_nfev_b_.data(), total);
    download_vec(out.nfev_f, d_nfev_f_.data(), total);
    download_vec(out.ldisc_b, d_ldisc_.data(), total);
    download_vec(out.ming_b, d_ming_.data(), total);
}

SlabOracleResult compute_oracle_gpu(CudaContext& ctx, const InletSlabGrid& grid,
                                    const ProductionStage& stage, SlabGpuTracer& tracer,
                                    const SlabOracleOptions& opt, DeviceSpan<real> psi1_out,
                                    DeviceSpan<real> psi2_out) {
    require_valid_grid(grid, "compute_oracle_gpu");
    if (psi1_out.size() != grid.full_size() || psi2_out.size() != grid.full_size()) {
        throw std::invalid_argument("compute_oracle_gpu: psi outputs must have (N+1) N^2 entries");
    }
    if (!stage.labels.device_prepared()) {
        throw std::logic_error("compute_oracle_gpu: inlet labels not device-prepared");
    }
    const int N = grid.n;
    const std::size_t n2 = grid.plane_size();

    closure_gate::IntegratorOptions base;
    base.tol = opt.tol;
    base.h_max = opt.h_max > 0.0 ? opt.h_max : grid.h;
    base.min_step = opt.min_step;
    base.max_arclength_per_period = opt.max_arclength;
    base.n_periods = 1;
    closure_gate::detail::validate_options(base);

    SlabOracleResult R;
    R.tol = opt.tol;

    const auto t0 = std::chrono::steady_clock::now();
    SlabGpuTraceOutputs T;
    tracer.trace(ctx, grid, base, T);
    R.seconds_trace = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    R.foot_y = std::move(T.foot_y);
    R.foot_z = std::move(T.foot_z);
    R.roundtrip = std::move(T.roundtrip);
    const std::vector<long long>& nfev_b = T.nfev_b;
    const std::vector<long long>& nfev_f = T.nfev_f;
    const std::vector<std::uint8_t>& st_b = T.st_b;
    const std::vector<std::uint8_t>& st_f = T.st_f;
    const std::vector<std::uint8_t>& bfe = T.bfe;
    const std::vector<long long>& ldisc_b = T.ldisc_b;
    const std::vector<double>& ming_b = T.ming_b;

    // per-plane table (sequential, index order) -- body copied from compute_oracle
    R.planes.resize(static_cast<std::size_t>(N) + 1);
    R.planes[0].j = 0;
    R.planes[0].x1 = 0.0;
    for (int j = 1; j <= N; ++j) {
        SlabOraclePlane& P = R.planes[static_cast<std::size_t>(j)];
        P.j = j;
        P.x1 = grid.coord(j);
        for (std::size_t r = 0; r < n2; ++r) {
            const std::size_t i = static_cast<std::size_t>(j - 1) * n2 + r;
            P.nfev += nfev_b[i];
            P.nfev_rt += nfev_f[i];
            ++P.status_hist[st_b[i] < 7 ? st_b[i] : 6];
            P.landings_discarded += ldisc_b[i];
            P.min_g1_hat = std::min(P.min_g1_hat, ming_b[i]);
            if (st_b[i] != 0) {
                ++P.back_non_ok;
            } else if (st_f[i] != 0) {
                ++P.fwd_non_ok;
            } else {
                P.roundtrip = std::max(P.roundtrip, R.roundtrip[i]);
                if (bfe[i])
                    ++P.backflow_encounters;
            }
        }
        P.flagged = P.back_non_ok > 0 || P.fwd_non_ok > 0 || P.roundtrip > opt.max_roundtrip;
        R.non_ok += P.back_non_ok + P.fwd_non_ok;
        R.seed_backflow += P.status_hist[1];
        R.landings_discarded += P.landings_discarded;
        R.min_g1_hat = std::min(R.min_g1_hat, P.min_g1_hat);
        R.max_roundtrip = std::max(R.max_roundtrip, P.roundtrip);
        if (P.flagged)
            R.status = SlabOracleStatus::oracle_roundtrip_fail;
    }

    const auto t1 = std::chrono::steady_clock::now();
    detail::oracle_evaluate_labels(ctx, grid, stage.labels, R.foot_y, R.foot_z, psi1_out,
                                   psi2_out);
    R.seconds_labels = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    return R;
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
