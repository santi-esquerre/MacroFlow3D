/**
 * @file SlabMetrics.cu
 * @brief SF-33 N0: metrics kernels, numpy-default percentiles and the CASE line (see
 * SlabMetrics.cuh).
 */

#include "SlabMetrics.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

constexpr int kPointSums = 11; // see metrics_point_kernel
constexpr int kOutDiv = 11;    // out_[11] = sum div^2
constexpr int kOutGather = 12; // out_[12..29] = gathered order statistics
constexpr int kOutSize = 32;
constexpr real kPercentiles[4] = {0.1, 1.0, 5.0, 50.0};

struct PercentileIndex {
    std::size_t lo = 0;
    std::size_t hi = 0;
    real gamma = 0.0;
};

/// numpy `_quantile` (method "linear"): virtual index (n - 1) * (pct / 100), floor / floor + 1,
/// clipped to the array, gamma = virtual - previous.
PercentileIndex percentile_index(std::size_t n, real pct) {
    const real q = pct / 100.0;
    const real virt = static_cast<real>(n - 1) * q;
    PercentileIndex r;
    real prev = std::floor(virt);
    std::size_t lo = static_cast<std::size_t>(prev);
    std::size_t hi = lo + 1;
    if (virt >= static_cast<real>(n - 1)) {
        lo = hi = n - 1;
        prev = -1.0; // numpy sets previous index -1 (the last element); gamma uses that value
    }
    if (virt < 0.0) {
        lo = hi = 0;
        prev = 0.0;
    }
    r.lo = lo;
    r.hi = hi;
    r.gamma = virt - prev;
    return r;
}

/// numpy `_lerp`.
real numpy_lerp(real a, real b, real t) {
    const real diff = b - a;
    if (t >= 0.5)
        return b - diff * (1.0 - t);
    return a + diff * t;
}

struct GatherIdx {
    unsigned long long lo[4];
    unsigned long long hi[4];
};

/**
 * Pointwise metrics pass over all (N+1) N^2 vertices. Sums:
 *  0 |c - vD|^2   1 |vD|^2   2 (vD.g1)^2   3 |g1|^2   4 (vD.g2)^2   5 |g2|^2
 *  6 (psi1 - psi1_or)^2   7 (psi2 - psi2_or)^2   8 (psi1_or - x2)^2   9 (psi2_or - x3)^2
 *  10 count of vertices with non-finite |c| or |vD|
 */
__global__ void metrics_point_kernel(InletSlabGrid g, SlabStencilView st,
                                     const real* __restrict__ U1, const real* __restrict__ U2,
                                     const real* __restrict__ v0, const real* __restrict__ v1,
                                     const real* __restrict__ v2, const real* __restrict__ por1,
                                     const real* __restrict__ por2, real* __restrict__ c0,
                                     real* __restrict__ c1, real* __restrict__ c2,
                                     real* __restrict__ cn, real* __restrict__ vn, real* partials,
                                     int nblocks) {
    const std::size_t nfull = g.full_size();
    const std::size_t np = g.plane_size();
    real acc[kPointSums];
    for (int k = 0; k < kPointSums; ++k)
        acc[k] = 0.0;
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < nfull; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int j = static_cast<int>(idx / np);
        const std::size_t rem = idx % np;
        const int m2 = static_cast<int>(rem / static_cast<std::size_t>(g.n));
        const int m3 = static_cast<int>(rem % static_cast<std::size_t>(g.n));
        real du1[3], du2[3];
        grad4_at(U1, st, g, j, m2, m3, du1);
        grad4_at(U2, st, g, j, m2, m3, du2);
        const real g1[3] = {du1[0], 1.0 + du1[1], du1[2]};
        const real g2[3] = {du2[0], du2[1], 1.0 + du2[2]};
        const real c[3] = {g1[1] * g2[2] - g1[2] * g2[1], g1[2] * g2[0] - g1[0] * g2[2],
                           g1[0] * g2[1] - g1[1] * g2[0]};
        const real v[3] = {v0[idx], v1[idx], v2[idx]};
        c0[idx] = c[0];
        c1[idx] = c[1];
        c2[idx] = c[2];
        const real cnv = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        const real vnv = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        cn[idx] = cnv;
        vn[idx] = vnv;
        const real e0 = c[0] - v[0], e1 = c[1] - v[1], e2 = c[2] - v[2];
        acc[0] += e0 * e0 + e1 * e1 + e2 * e2;
        acc[1] += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        const real vg1 = v[0] * g1[0] + v[1] * g1[1] + v[2] * g1[2];
        const real vg2 = v[0] * g2[0] + v[1] * g2[1] + v[2] * g2[2];
        acc[2] += vg1 * vg1;
        acc[3] += g1[0] * g1[0] + g1[1] * g1[1] + g1[2] * g1[2];
        acc[4] += vg2 * vg2;
        acc[5] += g2[0] * g2[0] + g2[1] * g2[1] + g2[2] * g2[2];
        if (por1 != nullptr) {
            const real x2 = g.coord(m2);
            const real x3 = g.coord(m3);
            const real p1 = x2 + U1[idx];
            const real p2 = x3 + U2[idx];
            const real a1 = p1 - por1[idx];
            const real a2 = p2 - por2[idx];
            const real d1 = por1[idx] - x2;
            const real d2 = por2[idx] - x3;
            acc[6] += a1 * a1;
            acc[7] += a2 * a2;
            acc[8] += d1 * d1;
            acc[9] += d2 * d2;
        }
        if (!(isfinite(cnv) && isfinite(vnv)))
            acc[10] += 1.0;
    }
    detail::block_reduce_store<kPointSums>(acc, partials, nblocks);
}

__global__ void metrics_div_kernel(InletSlabGrid g, SlabStencilView st, const real* __restrict__ c0,
                                   const real* __restrict__ c1, const real* __restrict__ c2,
                                   real* partials, int nblocks) {
    const std::size_t nfull = g.full_size();
    const std::size_t np = g.plane_size();
    real acc[1] = {0.0};
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < nfull; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int j = static_cast<int>(idx / np);
        const std::size_t rem = idx % np;
        const int m2 = static_cast<int>(rem / static_cast<std::size_t>(g.n));
        const int m3 = static_cast<int>(rem % static_cast<std::size_t>(g.n));
        const real div = d1_fd4_at(c0, st, g, j, m2, m3) + dp4_at(c1, g, j, m2, m3, 2) +
                         dp4_at(c2, g, j, m2, m3, 3);
        acc[0] += div * div;
    }
    detail::block_reduce_store<1>(acc, partials, nblocks);
}

__global__ void gather_order_stats_kernel(const real* __restrict__ cs, const real* __restrict__ vs,
                                          GatherIdx gi, real* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    for (int p = 0; p < 4; ++p) {
        out[2 * p] = cs[gi.lo[p]];
        out[2 * p + 1] = cs[gi.hi[p]];
        out[9 + 2 * p] = vs[gi.lo[p]];
        out[9 + 2 * p + 1] = vs[gi.hi[p]];
    }
    out[8] = cs[0];
    out[17] = vs[0];
}

__global__ void labels_to_periodic_kernel(InletSlabGrid g, const real* __restrict__ p1,
                                          const real* __restrict__ p2, real* __restrict__ u1,
                                          real* __restrict__ u2) {
    const std::size_t nfull = g.full_size();
    const std::size_t np = g.plane_size();
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < nfull; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t rem = idx % np;
        const int m2 = static_cast<int>(rem / static_cast<std::size_t>(g.n));
        const int m3 = static_cast<int>(rem % static_cast<std::size_t>(g.n));
        u1[idx] = p1[idx] - g.coord(m2);
        u2[idx] = p2[idx] - g.coord(m3);
    }
}

void require_size(std::size_t got, std::size_t want, const char* what) {
    if (got != want) {
        throw std::invalid_argument(std::string("inlet_slab metrics: ") + what + " has size " +
                                    std::to_string(got) + ", expected " + std::to_string(want));
    }
}

std::string fmt_e3(real x) {
    if (std::isnan(x))
        return "nan";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3e", x);
    return buf;
}

} // namespace

real percentile_linear_sorted(const real* sorted, std::size_t n, real pct) {
    if (n == 0)
        throw std::invalid_argument("percentile_linear_sorted: empty array");
    const PercentileIndex pi = percentile_index(n, pct);
    return numpy_lerp(sorted[pi.lo], sorted[pi.hi], pi.gamma);
}

void SlabMetricsWorkspace::prepare(const InletSlabGrid& g) {
    require_valid_grid(g, "SlabMetricsWorkspace::prepare");
    if (!table_.built_for(g))
        table_.build(g);
    const std::size_t nfull = g.full_size();
    for (auto& b : c_)
        b.resize(nfull);
    cn_.resize(nfull);
    vn_.resize(nfull);
    cn_sorted_.resize(nfull);
    vn_sorted_.resize(nfull);
    nblocks_ = detail::slab_reduce_blocks(nfull);
    partials_.resize(static_cast<std::size_t>(kPointSums + 1) * static_cast<std::size_t>(nblocks_));
    out_.resize(kOutSize);
    std::size_t bytes = 0;
    MACROFLOW3D_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        static_cast<void*>(nullptr), bytes, static_cast<const real*>(nullptr),
        static_cast<real*>(nullptr), static_cast<int>(nfull)));
    if (bytes > sort_temp_.capacity())
        sort_temp_.resize(bytes);
    sort_temp_bytes_ = bytes;
    n_ = g.n;
}

std::size_t SlabMetricsWorkspace::allocated_bytes() const {
    std::size_t r = table_.allocated_bytes() + sort_temp_.capacity();
    for (const auto& b : c_)
        r += b.capacity() * sizeof(real);
    r += (cn_.capacity() + vn_.capacity() + cn_sorted_.capacity() + vn_sorted_.capacity() +
          partials_.capacity() + out_.capacity()) *
         sizeof(real);
    return r;
}

std::vector<const void*> SlabMetricsWorkspace::storage_pointers() const {
    return {c_[0].data(),      c_[1].data(),      c_[2].data(),     cn_.data(),  vn_.data(),
            cn_sorted_.data(), vn_sorted_.data(), partials_.data(), out_.data(), sort_temp_.data()};
}

SlabMetrics evaluate_metrics(CudaContext& ctx, const InletSlabGrid& grid, DeviceSpan<const real> U1,
                             DeviceSpan<const real> U2, const SlabReferenceData& ref,
                             SlabMetricsWorkspace& ws) {
    require_valid_grid(grid, "evaluate_metrics");
    if (!ws.prepared_for(grid)) {
        throw std::logic_error(
            "evaluate_metrics: workspace not prepared for this grid (call prepare first)");
    }
    const std::size_t nfull = grid.full_size();
    require_size(U1.size(), nfull, "U1");
    require_size(U2.size(), nfull, "U2");
    if (!ref.has_vD)
        throw std::invalid_argument("evaluate_metrics: reference vD is required");
    for (int k = 0; k < 3; ++k)
        require_size(ref.vD[k].size(), nfull, "vD");
    if (ref.has_psi_or) {
        require_size(ref.psi_or[0].size(), nfull, "psi_or[0]");
        require_size(ref.psi_or[1].size(), nfull, "psi_or[1]");
    }

    const cudaStream_t stream = ctx.cuda_stream();
    const SlabStencilView st = ws.table_.device_view();
    const int nb = ws.nblocks_;
    real* partials_point = ws.partials_.data();
    real* partials_div = ws.partials_.data() + static_cast<std::size_t>(kPointSums) * nb;

    metrics_point_kernel<<<nb, detail::kSlabBlock, 0, stream>>>(
        grid, st, U1.data(), U2.data(), ref.vD[0].data(), ref.vD[1].data(), ref.vD[2].data(),
        ref.has_psi_or ? ref.psi_or[0].data() : nullptr,
        ref.has_psi_or ? ref.psi_or[1].data() : nullptr, ws.c_[0].data(), ws.c_[1].data(),
        ws.c_[2].data(), ws.cn_.data(), ws.vn_.data(), partials_point, nb);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    metrics_div_kernel<<<nb, detail::kSlabBlock, 0, stream>>>(
        grid, st, ws.c_[0].data(), ws.c_[1].data(), ws.c_[2].data(), partials_div, nb);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    detail::finalize_partials_kernel<kPointSums>
        <<<1, detail::kSlabBlock, 0, stream>>>(partials_point, nb, ws.out_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    detail::finalize_partials_kernel<1>
        <<<1, detail::kSlabBlock, 0, stream>>>(partials_div, nb, ws.out_.data() + kOutDiv);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // Sorted copies of |c| and |vD| (preallocated CUB temporary storage; no allocation here).
    std::size_t bytes = ws.sort_temp_bytes_;
    MACROFLOW3D_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        static_cast<void*>(ws.sort_temp_.data()), bytes, static_cast<const real*>(ws.cn_.data()),
        ws.cn_sorted_.data(), static_cast<int>(nfull), 0, static_cast<int>(sizeof(real) * 8),
        stream));
    bytes = ws.sort_temp_bytes_;
    MACROFLOW3D_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        static_cast<void*>(ws.sort_temp_.data()), bytes, static_cast<const real*>(ws.vn_.data()),
        ws.vn_sorted_.data(), static_cast<int>(nfull), 0, static_cast<int>(sizeof(real) * 8),
        stream));
    PercentileIndex pidx[4];
    GatherIdx gi;
    for (int p = 0; p < 4; ++p) {
        pidx[p] = percentile_index(nfull, kPercentiles[p]);
        gi.lo[p] = pidx[p].lo;
        gi.hi[p] = pidx[p].hi;
    }
    gather_order_stats_kernel<<<1, 1, 0, stream>>>(ws.cn_sorted_.data(), ws.vn_sorted_.data(), gi,
                                                   ws.out_.data() + kOutGather);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(ws.host_out_, ws.out_.data(), kOutSize * sizeof(real),
                                           cudaMemcpyDeviceToHost, stream));
    // The single documented host synchronization of evaluate_metrics.
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));

    const real* o = ws.host_out_;
    const real n = static_cast<real>(nfull);
    SlabMetrics m;
    m.v_rms = std::sqrt(o[1] / n);
    m.e_v = std::sqrt(o[0] / n) / m.v_rms;
    m.e_i1 = std::sqrt(o[2] / n) / (m.v_rms * std::sqrt(o[3] / n));
    m.e_i2 = std::sqrt(o[4] / n) / (m.v_rms * std::sqrt(o[5] / n));
    m.e_div = std::sqrt(o[kOutDiv] / n) / m.v_rms;
    m.nonfinite = static_cast<long long>(o[10]);
    const real* gc = o + kOutGather;     // |c|: (lo, hi) x 4, min
    const real* gv = o + kOutGather + 9; // |vD|: (lo, hi) x 4, min
    real pc[4], pv[4];
    for (int p = 0; p < 4; ++p) {
        pc[p] = numpy_lerp(gc[2 * p], gc[2 * p + 1], pidx[p].gamma);
        pv[p] = numpy_lerp(gv[2 * p], gv[2 * p + 1], pidx[p].gamma);
    }
    if (m.nonfinite == 0) {
        m.min_c = gc[8];
        m.p0_1 = pc[0];
        m.p1 = pc[1];
        m.p5 = pc[2];
        m.p50 = pc[3];
        m.vD_min = gv[8];
        for (int p = 0; p < 4; ++p)
            m.vD_p[p] = pv[p];
    }
    if (ref.has_psi_or) {
        const real den[2] = {std::sqrt(o[8] / n), std::sqrt(o[9] / n)};
        const real dref = std::max(den[0], den[1]);
        const real a[2] = {std::sqrt(o[6] / n), std::sqrt(o[7] / n)};
        real e[2], used[2];
        for (int i = 0; i < 2; ++i) {
            const real d = den[i] > kEPsiDenRel * dref ? den[i] : dref;
            used[i] = d;
            e[i] = d > 0.0 ? a[i] / d : a[i];
        }
        m.has_psi = true;
        m.a_psi1 = a[0];
        m.a_psi2 = a[1];
        m.den_used1 = used[0];
        m.den_used2 = used[1];
        m.e_psi1 = e[0];
        m.e_psi2 = e[1];
        m.e_psi = std::max(e[0], e[1]);
    }
    return m;
}

void labels_to_periodic_parts(CudaContext& ctx, const InletSlabGrid& grid,
                              DeviceSpan<const real> psi1, DeviceSpan<const real> psi2,
                              DeviceSpan<real> U1, DeviceSpan<real> U2) {
    require_valid_grid(grid, "labels_to_periodic_parts");
    const std::size_t nfull = grid.full_size();
    require_size(psi1.size(), nfull, "psi1");
    require_size(psi2.size(), nfull, "psi2");
    require_size(U1.size(), nfull, "U1");
    require_size(U2.size(), nfull, "U2");
    labels_to_periodic_kernel<<<detail::slab_reduce_blocks(nfull), detail::kSlabBlock, 0,
                                ctx.cuda_stream()>>>(grid, psi1.data(), psi2.data(), U1.data(),
                                                     U2.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

std::string format_case_line(const std::string& field, real eps, int N, const std::string& cand,
                             const SlabMetrics& m, const CaseLineExtras& extras) {
    char buf[64];
    std::string line = "CASE field=" + field;
    std::snprintf(buf, sizeof(buf), "%g", eps);
    line += std::string(" eps=") + buf;
    std::snprintf(buf, sizeof(buf), "%d", N);
    line += std::string(" N=") + buf + " cand=" + cand;
    line += " | r_F=" + (extras.r_F ? fmt_e3(*extras.r_F) : std::string("nan"));
    if (extras.its) {
        std::snprintf(buf, sizeof(buf), "%d", *extras.its);
        line += std::string(" its=") + buf;
    } else {
        line += " its=nan";
    }
    line += " | e_v=" + fmt_e3(m.e_v) + " e_psi=" + fmt_e3(m.e_psi) + " e_i=(" + fmt_e3(m.e_i1) +
            "," + fmt_e3(m.e_i2) + ") e_div=" + fmt_e3(m.e_div) + " min_c=" + fmt_e3(m.min_c) +
            " p0.1=" + fmt_e3(m.p0_1) + " p1=" + fmt_e3(m.p1) + " p5=" + fmt_e3(m.p5) +
            " p50=" + fmt_e3(m.p50);
    std::snprintf(buf, sizeof(buf), "%.1f", extras.t);
    line += std::string(" | t=") + buf;
    if (m.has_psi)
        line += " | e_psi1=" + fmt_e3(m.e_psi1) + " e_psi2=" + fmt_e3(m.e_psi2);
    return line;
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
