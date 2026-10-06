/**
 * @file SlabJacobianVectorProduct.cu
 * @brief SF-33 N1: analytic Jacobian-vector product of the inlet-slab residual (see
 * SlabJacobianVectorProduct.cuh for the derivation and the contract).
 */

#include "SlabJacobianVectorProduct.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

__global__ void slab_jvp_kernel(InletSlabGrid g, SlabStencilView st, const real* __restrict__ U1,
                                const real* __restrict__ U2, const real* __restrict__ dU1,
                                const real* __restrict__ dU2, const real* __restrict__ q,
                                const real* __restrict__ gl0, const real* __restrict__ gl1,
                                const real* __restrict__ gl2, real* __restrict__ out) {
    const std::size_t nf = g.field_size();
    const std::size_t np = g.plane_size();
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < nf; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int j = 1 + static_cast<int>(idx / np);
        const std::size_t rem = idx % np;
        const int m2 = static_cast<int>(rem / static_cast<std::size_t>(g.n));
        const int m3 = static_cast<int>(rem % static_cast<std::size_t>(g.n));
        const std::size_t fi = g.full_index(j, m2, m3);
        const real qv = q[fi];
        real du1[3], du2[3], g1[3], g2[3], dg1[3], dg2[3];
        if (j < g.n) {
            real H1[6], H2[6], dH1[6], dH2[6];
            derivs4_at(U1, st, g, j, m2, m3, du1, H1);
            derivs4_at(U2, st, g, j, m2, m3, du2, H2);
            slab_label_gradients(du1, du2, g1, g2);
            const real glnk[3] = {gl0[fi], gl1[fi], gl2[fi]};
            SlabPointState s;
            slab_equation_point(g1, g2, H1, H2, glnk, qv, s);
            derivs4_at(dU1, st, g, j, m2, m3, dg1, dH1);
            derivs4_at(dU2, st, g, j, m2, m3, dg2, dH2);
            real dF1, dF2;
            slab_equation_point_jvp(g1, g2, H1, H2, glnk, qv, s, dg1, dg2, dH1, dH2, dF1, dF2);
            out[idx] = dF1;
            out[nf + idx] = dF2;
        } else {
            grad4_at(U1, st, g, j, m2, m3, du1);
            grad4_at(U2, st, g, j, m2, m3, du2);
            slab_label_gradients(du1, du2, g1, g2);
            grad4_at(dU1, st, g, j, m2, m3, dg1);
            grad4_at(dU2, st, g, j, m2, m3, dg2);
            real dE1, dE2;
            slab_outlet_point_jvp(g1, g2, dg1, dg2, qv, g.h, dE1, dE2);
            out[idx] = dE1;
            out[nf + idx] = dE2;
        }
    }
}

void require_size(std::size_t got, std::size_t want, const char* what) {
    if (got != want) {
        throw std::invalid_argument(std::string("SlabJvpWorkspace: ") + what + " has size " +
                                    std::to_string(got) + ", expected " + std::to_string(want));
    }
}

bool overlaps(const void* a, std::size_t na, const void* b, std::size_t nb) {
    const char* pa = static_cast<const char*>(a);
    const char* pb = static_cast<const char*>(b);
    return pa < pb + nb && pb < pa + na;
}

} // namespace

void SlabJvpWorkspace::prepare(const InletSlabGrid& g) {
    require_valid_grid(g, "SlabJvpWorkspace::prepare");
    if (!table_.built_for(g))
        table_.build(g);
    for (int f = 0; f < 2; ++f) {
        base_U_[f].resize(g.full_size());
        dir_U_[f].resize(g.full_size());
        // Plane 0 of the direction arrays is the Dirichlet inlet: identically zero. Nothing ever
        // writes it afterwards (apply copies planes 1..N only). Preparation step: synchronous.
        MACROFLOW3D_CUDA_CHECK(cudaMemset(dir_U_[f].data(), 0, g.plane_size() * sizeof(real)));
    }
    // SF-33 C2: legacy-stream memset must land before ctx-stream work
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    base_inputs_ = nullptr;
    n_ = g.n;
}

void SlabJvpWorkspace::prepare_base(CudaContext& ctx, const InletSlabGrid& grid,
                                    const SlabStageInputs& inputs, DeviceSpan<const real> U1,
                                    DeviceSpan<const real> U2) {
    require_valid_grid(grid, "SlabJvpWorkspace::prepare_base");
    if (!prepared_for(grid)) {
        throw std::logic_error(
            "SlabJvpWorkspace::prepare_base: workspace not prepared for this grid");
    }
    inputs.check(grid);
    require_size(U1.size(), grid.full_size(), "U1");
    require_size(U2.size(), grid.full_size(), "U2");
    const cudaStream_t s = ctx.cuda_stream();
    const real* src[2] = {U1.data(), U2.data()};
    for (int f = 0; f < 2; ++f) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(base_U_[f].data(), src[f],
                                               grid.full_size() * sizeof(real),
                                               cudaMemcpyDeviceToDevice, s));
    }
    base_inputs_ = &inputs;
}

void SlabJvpWorkspace::apply(CudaContext& ctx, const InletSlabGrid& grid,
                             DeviceSpan<const real> direction, DeviceSpan<real> out) {
    require_valid_grid(grid, "SlabJvpWorkspace::apply");
    if (!prepared_for(grid)) {
        throw std::logic_error("SlabJvpWorkspace::apply: workspace not prepared for this grid");
    }
    if (base_inputs_ == nullptr) {
        throw std::logic_error(
            "SlabJvpWorkspace::apply: base state not frozen (call prepare_base first)");
    }
    require_size(direction.size(), grid.unknown_size(), "direction");
    require_size(out.size(), grid.unknown_size(), "out");
    if (overlaps(direction.data(), direction.size() * sizeof(real), out.data(),
                 out.size() * sizeof(real))) {
        throw std::invalid_argument("SlabJvpWorkspace::apply: direction and out overlap");
    }
    const cudaStream_t s = ctx.cuda_stream();
    const std::size_t np = grid.plane_size();
    const std::size_t nf = grid.field_size();
    for (int f = 0; f < 2; ++f) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(dir_U_[f].data() + np,
                                               direction.data() + static_cast<std::size_t>(f) * nf,
                                               nf * sizeof(real), cudaMemcpyDeviceToDevice, s));
    }
    const SlabStageInputs& in = *base_inputs_;
    const int blocks = detail::slab_reduce_blocks(nf);
    slab_jvp_kernel<<<blocks, detail::kSlabBlock, 0, s>>>(
        grid, table_.device_view(), base_U_[0].data(), base_U_[1].data(), dir_U_[0].data(),
        dir_U_[1].data(), in.q.data(), in.grad_lnk[0].data(), in.grad_lnk[1].data(),
        in.grad_lnk[2].data(), out.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    // No host synchronization: the result is stream-ordered on ctx.cuda_stream().
}

std::size_t SlabJvpWorkspace::allocated_bytes() const {
    std::size_t b = table_.allocated_bytes();
    for (int f = 0; f < 2; ++f)
        b += (base_U_[f].capacity() + dir_U_[f].capacity()) * sizeof(real);
    return b;
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
