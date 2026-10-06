/**
 * @file SlabResidual.cu
 * @brief SF-33 N0: residual kernel of the inlet slab (see SlabResidual.cuh for the contract).
 */

#include "SlabResidual.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

constexpr int kResidualSums =
    4; // sum F1^2, sum F2^2, sum q^2 (planes 1..N-1), sum outlet defect^2 (plane N)

__global__ void slab_residual_kernel(InletSlabGrid g, SlabStencilView st,
                                     const real* __restrict__ U1, const real* __restrict__ U2,
                                     const real* __restrict__ q, const real* __restrict__ gl0,
                                     const real* __restrict__ gl1, const real* __restrict__ gl2,
                                     const real* __restrict__ v2in, const real* __restrict__ v3in,
                                     real* __restrict__ E, real* partials, int nblocks) {
    const std::size_t nf = g.field_size();
    const std::size_t np = g.plane_size();
    real acc[kResidualSums] = {0.0, 0.0, 0.0, 0.0};
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < nf; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int j = 1 + static_cast<int>(idx / np);
        const std::size_t rem = idx % np;
        const int m2 = static_cast<int>(rem / static_cast<std::size_t>(g.n));
        const int m3 = static_cast<int>(rem % static_cast<std::size_t>(g.n));
        const std::size_t fi = g.full_index(j, m2, m3);
        const real qv = q[fi];
        real du1[3], du2[3], g1[3], g2[3];
        if (j < g.n) {
            real H1[6], H2[6];
            derivs4_at(U1, st, g, j, m2, m3, du1, H1);
            derivs4_at(U2, st, g, j, m2, m3, du2, H2);
            slab_label_gradients(du1, du2, g1, g2);
            const real glnk[3] = {gl0[fi], gl1[fi], gl2[fi]};
            SlabPointState s;
            slab_equation_point(g1, g2, H1, H2, glnk, qv, s);
            E[idx] = s.F1;
            E[nf + idx] = s.F2;
            acc[0] += s.F1 * s.F1;
            acc[1] += s.F2 * s.F2;
            acc[2] += qv * qv;
        } else {
            grad4_at(U1, st, g, j, m2, m3, du1);
            grad4_at(U2, st, g, j, m2, m3, du2);
            slab_label_gradients(du1, du2, g1, g2);
            const std::size_t pi = g.plane_index(m2, m3);
            real e1, e2, d2, d3;
            slab_outlet_point(g1, g2, v2in[pi], v3in[pi], qv, g.h, e1, e2, d2, d3);
            E[idx] = e1;
            E[nf + idx] = e2;
            acc[3] += d2 * d2 + d3 * d3;
        }
    }
    if (partials != nullptr)
        detail::block_reduce_store<kResidualSums>(acc, partials, nblocks);
}

__global__ void q_from_lnk_kernel(const real* __restrict__ lnk, real* __restrict__ q,
                                  std::size_t n) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        q[i] = 1.0 / exp(lnk[i]);
    }
}

void require_size(std::size_t got, std::size_t want, const char* what) {
    if (got != want) {
        throw std::invalid_argument(std::string("inlet_slab: ") + what + " has size " +
                                    std::to_string(got) + ", expected " + std::to_string(want));
    }
}

} // namespace

void SlabResidualWorkspace::prepare(const InletSlabGrid& g) {
    require_valid_grid(g, "SlabResidualWorkspace::prepare");
    if (!table_.built_for(g))
        table_.build(g);
    nblocks_ = detail::slab_reduce_blocks(g.field_size());
    partials_.resize(static_cast<std::size_t>(kResidualSums) * static_cast<std::size_t>(nblocks_));
    sums_.resize(kResidualSums);
    n_ = g.n;
}

std::size_t SlabResidualWorkspace::allocated_bytes() const {
    return table_.allocated_bytes() + (partials_.capacity() + sums_.capacity()) * sizeof(real);
}

void evaluate_residual(CudaContext& ctx, const InletSlabGrid& grid, const SlabStageInputs& inputs,
                       DeviceSpan<const real> U1, DeviceSpan<const real> U2, DeviceSpan<real> E_out,
                       SlabResidualWorkspace& ws, SlabResidualNorms* norms_out) {
    require_valid_grid(grid, "evaluate_residual");
    if (!ws.prepared_for(grid)) {
        throw std::logic_error(
            "evaluate_residual: workspace not prepared for this grid (call prepare first)");
    }
    inputs.check(grid);
    require_size(U1.size(), grid.full_size(), "U1");
    require_size(U2.size(), grid.full_size(), "U2");
    require_size(E_out.size(), grid.unknown_size(), "E_out");

    const cudaStream_t stream = ctx.cuda_stream();
    const bool reduce = norms_out != nullptr;
    slab_residual_kernel<<<ws.nblocks_, detail::kSlabBlock, 0, stream>>>(
        grid, ws.table_.device_view(), U1.data(), U2.data(), inputs.q.data(),
        inputs.grad_lnk[0].data(), inputs.grad_lnk[1].data(), inputs.grad_lnk[2].data(),
        inputs.vperp_in[0].data(), inputs.vperp_in[1].data(), E_out.data(),
        reduce ? ws.partials_.data() : nullptr, ws.nblocks_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    if (!reduce)
        return;

    detail::finalize_partials_kernel<kResidualSums>
        <<<1, detail::kSlabBlock, 0, stream>>>(ws.partials_.data(), ws.nblocks_, ws.sums_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(ws.host_sums_, ws.sums_.data(),
                                           kResidualSums * sizeof(real), cudaMemcpyDeviceToHost,
                                           stream));
    // The single documented host synchronization of evaluate_residual (norms requested).
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));

    const real n_eq = static_cast<real>(grid.n - 1) * static_cast<real>(grid.plane_size());
    const real n_out = static_cast<real>(grid.plane_size());
    const real mean_f1 = ws.host_sums_[0] / n_eq;
    const real mean_f2 = ws.host_sums_[1] / n_eq;
    const real q_rms = std::sqrt(ws.host_sums_[2] / n_eq);
    norms_out->r_F = std::sqrt((mean_f1 + mean_f2) / 2.0) / q_rms;
    norms_out->r_out = std::sqrt(ws.host_sums_[3] / n_out) / inputs.v_rms;
}

void assemble_full_planes(CudaContext& ctx, const InletSlabGrid& grid, DeviceSpan<const real> u_vec,
                          const SlabStageInputs& inputs, DeviceSpan<real> U1, DeviceSpan<real> U2) {
    require_valid_grid(grid, "assemble_full_planes");
    require_size(u_vec.size(), grid.unknown_size(), "u_vec");
    require_size(U1.size(), grid.full_size(), "U1");
    require_size(U2.size(), grid.full_size(), "U2");
    require_size(inputs.u0[0].size(), grid.plane_size(), "u0[0]");
    require_size(inputs.u0[1].size(), grid.plane_size(), "u0[1]");
    const cudaStream_t s = ctx.cuda_stream();
    const std::size_t np = grid.plane_size();
    const std::size_t nf = grid.field_size();
    real* dst[2] = {U1.data(), U2.data()};
    for (int f = 0; f < 2; ++f) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(dst[f], inputs.u0[f].data(), np * sizeof(real),
                                               cudaMemcpyDeviceToDevice, s));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(dst[f] + np,
                                               u_vec.data() + static_cast<std::size_t>(f) * nf,
                                               nf * sizeof(real), cudaMemcpyDeviceToDevice, s));
    }
}

void extract_unknowns(CudaContext& ctx, const InletSlabGrid& grid, DeviceSpan<const real> U1,
                      DeviceSpan<const real> U2, DeviceSpan<real> u_vec) {
    require_valid_grid(grid, "extract_unknowns");
    require_size(u_vec.size(), grid.unknown_size(), "u_vec");
    require_size(U1.size(), grid.full_size(), "U1");
    require_size(U2.size(), grid.full_size(), "U2");
    const cudaStream_t s = ctx.cuda_stream();
    const std::size_t np = grid.plane_size();
    const std::size_t nf = grid.field_size();
    const real* src[2] = {U1.data(), U2.data()};
    for (int f = 0; f < 2; ++f) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(u_vec.data() + static_cast<std::size_t>(f) * nf,
                                               src[f] + np, nf * sizeof(real),
                                               cudaMemcpyDeviceToDevice, s));
    }
}

void fill_q_from_lnk(CudaContext& ctx, const InletSlabGrid& grid, SlabStageInputs& inputs) {
    require_valid_grid(grid, "fill_q_from_lnk");
    require_size(inputs.lnk.size(), grid.full_size(), "lnk");
    require_size(inputs.q.size(), grid.full_size(), "q");
    const std::size_t n = grid.full_size();
    const int blocks = detail::slab_reduce_blocks(n);
    q_from_lnk_kernel<<<blocks, detail::kSlabBlock, 0, ctx.cuda_stream()>>>(inputs.lnk.data(),
                                                                            inputs.q.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
