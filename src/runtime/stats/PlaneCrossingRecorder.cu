/**
 * @file PlaneCrossingRecorder.cu
 * @brief SF-42 N1: device detection and host reduction of first plane crossings (see the header).
 */

#include "PlaneCrossingRecorder.cuh"

#include "../cuda_check.cuh"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace runtime {

using physics::particles::ConstParticlesSoA;

namespace {

constexpr int kBlock = 256;

inline int blocks_for(int n) {
    return (n + kBlock - 1) / kBlock;
}

// Same expression as Par2's unwrap kernel (x + wrap * L); the wrap array may be null.
__device__ inline double unwrapped(double v, const int32_t* wrap, int i, double L) {
    return wrap != nullptr ? v + static_cast<double>(wrap[i]) * L : v;
}

__global__ void capture_kernel(ConstParticlesSoA<real> p, double Lx, double Ly, double Lz,
                               double* prev_x, double* prev_y, double* prev_z, double* inj_y,
                               double* inj_z, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const double x = unwrapped(p.x[i], p.wrapX, i, Lx);
    const double y = unwrapped(p.y[i], p.wrapY, i, Ly);
    const double z = unwrapped(p.z[i], p.wrapZ, i, Lz);
    prev_x[i] = x;
    prev_y[i] = y;
    prev_z[i] = z;
    inj_y[i] = y;
    inj_z[i] = z;
}

__global__ void crossing_kernel(ConstParticlesSoA<real> p, double Lx, double Ly, double Lz,
                                const double* planes, int P, double t_n, double dt, double* prev_x,
                                double* prev_y, double* prev_z, double* rec_t, double* rec_y,
                                double* rec_z, unsigned char* rec_flag, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const double x0 = prev_x[i], y0 = prev_y[i], z0 = prev_z[i];
    const double x1 = unwrapped(p.x[i], p.wrapX, i, Lx);
    const double y1 = unwrapped(p.y[i], p.wrapY, i, Ly);
    const double z1 = unwrapped(p.z[i], p.wrapZ, i, Lz);
    if (x1 > x0) {
        const double dx = __dsub_rn(x1, x0);
        for (int k = 0; k < P; ++k) {
            const double X = planes[k];
            if (X > x1)
                break; // planes are increasing
            const size_t rk = static_cast<size_t>(k) * static_cast<size_t>(n) + i;
            if (x0 < X && !rec_flag[rk]) {
                const double f = __ddiv_rn(__dsub_rn(X, x0), dx);
                rec_t[rk] = __dadd_rn(t_n, __dmul_rn(f, dt));
                rec_y[rk] = __dadd_rn(y0, __dmul_rn(f, __dsub_rn(y1, y0)));
                rec_z[rk] = __dadd_rn(z0, __dmul_rn(f, __dsub_rn(z1, z0)));
                rec_flag[rk] = 1;
            }
        }
    }
    prev_x[i] = x1;
    prev_y[i] = y1;
    prev_z[i] = z1;
}

// Two-pass centred biased moments of (a, b) over the selected entries.
struct Moments2 {
    double ma = 0, mb = 0, va = 0, vb = 0, cab = 0;
};

Moments2 moments2(const std::vector<double>& a, const std::vector<double>& b) {
    Moments2 m;
    const size_t n = a.size();
    if (n == 0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan, nan, nan, nan};
    }
    double sa = 0, sb = 0;
    for (size_t i = 0; i < n; ++i) {
        sa += a[i];
        sb += b[i];
    }
    m.ma = sa / static_cast<double>(n);
    m.mb = sb / static_cast<double>(n);
    double qa = 0, qb = 0, qab = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - m.ma, db = b[i] - m.mb;
        qa += da * da;
        qb += db * db;
        qab += da * db;
    }
    m.va = qa / static_cast<double>(n);
    m.vb = qb / static_cast<double>(n);
    m.cab = qab / static_cast<double>(n);
    return m;
}

} // namespace

PlaneCrossingRecorder::PlaneCrossingRecorder(int n_particles, const std::vector<double>& planes,
                                             double Lx, double Ly, double Lz)
    : n_(n_particles), P_(static_cast<int>(planes.size())), Lx_(Lx), Ly_(Ly), Lz_(Lz),
      h_planes_(planes) {
    if (n_ <= 0 || P_ <= 0)
        throw std::invalid_argument("PlaneCrossingRecorder: n_particles and planes must be > 0");
    const size_t n = static_cast<size_t>(n_), pn = n * static_cast<size_t>(P_);
    d_planes_.resize(P_);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d_planes_.data(), h_planes_.data(), P_ * sizeof(double),
                                      cudaMemcpyHostToDevice));
    prev_x_.resize(n);
    prev_y_.resize(n);
    prev_z_.resize(n);
    inj_y_.resize(n);
    inj_z_.resize(n);
    rec_t_.resize(pn);
    rec_y_.resize(pn);
    rec_z_.resize(pn);
    rec_flag_.resize(pn);
    h_inj_y_.resize(n);
    h_inj_z_.resize(n);
}

size_t PlaneCrossingRecorder::device_bytes() const {
    const size_t n = static_cast<size_t>(n_), pn = n * static_cast<size_t>(P_);
    return P_ * sizeof(double) + 5 * n * sizeof(double) + 3 * pn * sizeof(double) + pn;
}

void PlaneCrossingRecorder::begin(const ConstParticlesSoA<real>& p, cudaStream_t stream) {
    if (p.n != n_)
        throw std::invalid_argument("PlaneCrossingRecorder::begin: particle count mismatch");
    const size_t pn = static_cast<size_t>(n_) * static_cast<size_t>(P_);
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(rec_flag_.data(), 0, pn, stream));
    capture_kernel<<<blocks_for(n_), kBlock, 0, stream>>>(p, Lx_, Ly_, Lz_, prev_x_.data(),
                                                          prev_y_.data(), prev_z_.data(),
                                                          inj_y_.data(), inj_z_.data(), n_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void PlaneCrossingRecorder::update(const ConstParticlesSoA<real>& p, int step, double dt,
                                   cudaStream_t stream) {
    const double t_n = static_cast<double>(step - 1) * dt;
    crossing_kernel<<<blocks_for(n_), kBlock, 0, stream>>>(
        p, Lx_, Ly_, Lz_, d_planes_.data(), P_, t_n, dt, prev_x_.data(), prev_y_.data(),
        prev_z_.data(), rec_t_.data(), rec_y_.data(), rec_z_.data(), rec_flag_.data(), n_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

std::vector<PlaneCrossingRow> PlaneCrossingRecorder::finish(cudaStream_t stream) {
    const size_t n = static_cast<size_t>(n_);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_inj_y_.data(), inj_y_.data(), n * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_inj_z_.data(), inj_z_.data(), n * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
    std::vector<double> ht(n), hy(n), hz(n);
    std::vector<unsigned char> hf(n);
    std::vector<PlaneCrossingRow> rows(static_cast<size_t>(P_));
    std::vector<double> t, x2, x3, d2, d3, zero;
    for (int k = 0; k < P_; ++k) {
        const size_t off = static_cast<size_t>(k) * n;
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(ht.data(), rec_t_.data() + off, n * sizeof(double),
                                               cudaMemcpyDeviceToHost, stream));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(hy.data(), rec_y_.data() + off, n * sizeof(double),
                                               cudaMemcpyDeviceToHost, stream));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(hz.data(), rec_z_.data() + off, n * sizeof(double),
                                               cudaMemcpyDeviceToHost, stream));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(hf.data(), rec_flag_.data() + off, n,
                                               cudaMemcpyDeviceToHost, stream));
        MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));
        t.clear();
        x2.clear();
        x3.clear();
        d2.clear();
        d3.clear();
        for (size_t i = 0; i < n; ++i) {
            if (!hf[i])
                continue;
            t.push_back(ht[i]);
            x2.push_back(hy[i]);
            x3.push_back(hz[i]);
            d2.push_back(hy[i] - h_inj_y_[i]);
            d3.push_back(hz[i] - h_inj_z_[i]);
        }
        PlaneCrossingRow& r = rows[k];
        r.x1 = h_planes_[k];
        r.n_injected = n_;
        r.n_crossed = static_cast<long long>(t.size());
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (t.empty()) {
            r.mean_t = r.var_t = r.mean_x2 = r.mean_x3 = r.var_x2 = r.var_x3 = r.cov_x2x3 = nan;
            r.mean_dx2 = r.mean_dx3 = r.var_dx2 = r.var_dx3 = r.cov_dx2dx3 = nan;
            continue;
        }
        zero.assign(t.size(), 0.0);
        const Moments2 mt = moments2(t, zero);
        r.mean_t = mt.ma;
        r.var_t = mt.va;
        const Moments2 mx = moments2(x2, x3);
        r.mean_x2 = mx.ma;
        r.mean_x3 = mx.mb;
        r.var_x2 = mx.va;
        r.var_x3 = mx.vb;
        r.cov_x2x3 = mx.cab;
        const Moments2 md = moments2(d2, d3);
        r.mean_dx2 = md.ma;
        r.mean_dx3 = md.mb;
        r.var_dx2 = md.va;
        r.var_dx3 = md.vb;
        r.cov_dx2dx3 = md.cab;
    }
    return rows;
}

bool write_plane_crossings_csv(const std::string& path, const std::vector<PlaneCrossingRow>& rows) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f)
        return false;
    std::fprintf(f, "# format_version=1 sf42_plane_crossings weights=uniform time_origin=injection\n");
    std::fprintf(f, "x1,n_injected,n_crossed,mean_t,var_t,mean_x2,mean_x3,var_x2,var_x3,cov_x2x3,"
                    "mean_dx2,mean_dx3,var_dx2,var_dx3,cov_dx2dx3\n");
    for (const auto& r : rows) {
        std::fprintf(f, "%.17g,%lld,%lld,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                        "%.17g,%.17g\n",
                     r.x1, r.n_injected, r.n_crossed, r.mean_t, r.var_t, r.mean_x2, r.mean_x3,
                     r.var_x2, r.var_x3, r.cov_x2x3, r.mean_dx2, r.mean_dx3, r.var_dx2, r.var_dx3,
                     r.cov_dx2dx3);
    }
    const bool bad = std::ferror(f) != 0;
    return std::fclose(f) == 0 && !bad;
}

bool write_injection_csv(const std::string& path, double x1, const std::vector<double>& x2,
                         const std::vector<double>& x3) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f)
        return false;
    std::fprintf(f, "x1,x2,x3\n");
    for (size_t i = 0; i < x2.size(); ++i)
        std::fprintf(f, "%.17g,%.17g,%.17g\n", x1, x2[i], x3[i]);
    const bool bad = std::ferror(f) != 0;
    return std::fclose(f) == 0 && !bad;
}

} // namespace runtime
} // namespace macroflow3d
