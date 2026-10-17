/**
 * @file SlabTraceGpu.cu
 * @brief SF-35 N6: GPU tracer (device copies of the SF-30 integrator, one thread per seed) and
 *        compute_oracle_gpu. See SlabTraceGpu.cuh. This TU is compiled with --fmad=false.
 */

#include "SlabTraceGpu.cuh"
#include "SlabTraceDevice.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

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
