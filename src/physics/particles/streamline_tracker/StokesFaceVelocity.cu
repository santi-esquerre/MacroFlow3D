/**
 * @file StokesFaceVelocity.cu
 * @brief SF-32 Stokes face fluxes of the label velocity - implementation
 *        (validation, edge and face kernels, host mirror, divergence diagnostic).
 *
 * See StokesFaceVelocity.cuh for the full specification (periodic
 * decomposition, layout, orientation, quadrature exactness, determinism).
 */

#include "../../../runtime/cuda_check.cuh"
#include "StokesFaceVelocity.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

namespace {

constexpr int kBlockSize = 256;

/// One thread per edge; direction d = blockIdx.y (0: ex, 1: ey, 2: ez).
__global__ void kernel_stokes_edges(real* __restrict__ ex, real* __restrict__ ey,
                                    real* __restrict__ ez, int nx, int ny, int nz, real dx, real dy,
                                    real dz, const SplineLabelPair labels) {
    const size_t n = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const size_t c = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n)
        return;
    const int d = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(c % static_cast<size_t>(nx));
    const size_t r = c / static_cast<size_t>(nx);
    const int j = static_cast<int>(r % static_cast<size_t>(ny));
    const int k = static_cast<int>(r / static_cast<size_t>(ny));
    const real e = stokes_detail::grid_edge(labels, d, i, j, k, dx, dy, dz);
    real* out = d == 0 ? ex : (d == 1 ? ey : ez);
    out[c] = e;
}

/// One thread per cell: its three face averages.
__global__ void kernel_stokes_faces(const real* __restrict__ ex, const real* __restrict__ ey,
                                    const real* __restrict__ ez, real* __restrict__ u,
                                    real* __restrict__ v, real* __restrict__ w, int nx, int ny,
                                    int nz, real dx, real dy, real dz, real cb0, real cb1,
                                    real cb2) {
    const size_t n = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const size_t c = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n)
        return;
    const int i = static_cast<int>(c % static_cast<size_t>(nx));
    const size_t r = c / static_cast<size_t>(nx);
    const int j = static_cast<int>(r % static_cast<size_t>(ny));
    const int k = static_cast<int>(r / static_cast<size_t>(ny));
    const real cbar[3] = {cb0, cb1, cb2};
    real uu, vv, ww;
    stokes_detail::cell_faces(ex, ey, ez, nx, ny, nz, dx, dy, dz, cbar, i, j, k, uu, vv, ww);
    u[c] = uu;
    v[c] = vv;
    w[c] = ww;
}

void validate_inputs(const SplineLabelPair& labels, const Grid3D& grid, const char* who) {
    const std::string w(who);
    if (grid.nx < 1 || grid.ny < 1 || grid.nz < 1) {
        throw std::invalid_argument(w + ": Pollock grid cell counts must be >= 1");
    }
    const real sp[3] = {grid.dx, grid.dy, grid.dz};
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(sp[d]) || !(sp[d] > 0.0)) {
            throw std::invalid_argument(w + ": Pollock grid spacing " + std::to_string(d) +
                                        " must be finite and > 0");
        }
    }
    if (labels.s1.coeff == nullptr || labels.s2.coeff == nullptr) {
        throw std::invalid_argument(w + ": label spline coefficient pointer is null");
    }
    const real Lg[3] = {grid.Lx(), grid.Ly(), grid.Lz()};
    for (int d = 0; d < 3; ++d) {
        const real Ll = labels.L[d];
        if (!std::isfinite(Ll) || !(Ll > 0.0)) {
            throw std::invalid_argument(w + ": label period " + std::to_string(d) +
                                        " must be finite and > 0");
        }
        if (!(std::fabs(Lg[d] - Ll) <= 1e-12 * Ll)) {
            throw std::invalid_argument(w + ": Pollock grid period " + std::to_string(d) +
                                        " differs from the label period by more than 1e-12 "
                                        "relative");
        }
    }
}

void mean_cross(const SplineLabelPair& labels, real cbar[3]) {
    cross3(labels.gbar1, labels.gbar2, cbar);
}

} // namespace

PeriodicFaceFluxView StokesFaceFluxWorkspace::view() const {
    const size_t n = grid.num_cells();
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || u.data() == nullptr || u.size() != n ||
        v.size() != n || w.size() != n) {
        throw std::logic_error("StokesFaceFluxWorkspace::view: no compute_stokes_face_fluxes call "
                               "has completed on this workspace");
    }
    PeriodicFaceFluxView fv{};
    fv.u = u.data();
    fv.v = v.data();
    fv.w = w.data();
    fv.nx = grid.nx;
    fv.ny = grid.ny;
    fv.nz = grid.nz;
    fv.dx = grid.dx;
    fv.dy = grid.dy;
    fv.dz = grid.dz;
    fv.Lx = grid.Lx();
    fv.Ly = grid.Ly();
    fv.Lz = grid.Lz();
    return fv;
}

StokesFaceFluxReport compute_stokes_face_fluxes(cudaStream_t stream, const SplineLabelPair& labels,
                                                const Grid3D& grid, StokesFaceFluxWorkspace& ws) {
    validate_inputs(labels, grid, "compute_stokes_face_fluxes");
    const size_t n = grid.num_cells();
    ws.ex.resize(n);
    ws.ey.resize(n);
    ws.ez.resize(n);
    ws.u.resize(n);
    ws.v.resize(n);
    ws.w.resize(n);

    const size_t blocks = (n + kBlockSize - 1) / kBlockSize;
    const dim3 edge_grid(static_cast<unsigned int>(blocks), 3u, 1u);
    kernel_stokes_edges<<<edge_grid, kBlockSize, 0, stream>>>(
        ws.ex.data(), ws.ey.data(), ws.ez.data(), grid.nx, grid.ny, grid.nz, grid.dx, grid.dy,
        grid.dz, labels);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    real cbar[3];
    mean_cross(labels, cbar);
    kernel_stokes_faces<<<static_cast<unsigned int>(blocks), kBlockSize, 0, stream>>>(
        ws.ex.data(), ws.ey.data(), ws.ez.data(), ws.u.data(), ws.v.data(), ws.w.data(), grid.nx,
        grid.ny, grid.nz, grid.dx, grid.dy, grid.dz, cbar[0], cbar[1], cbar[2]);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    ws.grid = grid;

    StokesFaceFluxReport rep{};
    rep.device_bytes = (ws.ex.capacity() + ws.ey.capacity() + ws.ez.capacity() + ws.u.capacity() +
                        ws.v.capacity() + ws.w.capacity()) *
                       sizeof(real);
    rep.quadrature_nodes_per_interval = stokes_detail::kGaussNodes;
    return rep;
}

void compute_stokes_face_fluxes_host(const SplineLabelPair& labels, const Grid3D& grid,
                                     std::vector<real>& u, std::vector<real>& v,
                                     std::vector<real>& w) {
    validate_inputs(labels, grid, "compute_stokes_face_fluxes_host");
    const size_t n = grid.num_cells();
    std::vector<real> e[3];
    for (int d = 0; d < 3; ++d) {
        e[d].resize(n);
    }
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const size_t c = stokes_detail::lin(i, j, k, grid.nx, grid.ny);
                for (int d = 0; d < 3; ++d) {
                    e[d][c] =
                        stokes_detail::grid_edge(labels, d, i, j, k, grid.dx, grid.dy, grid.dz);
                }
            }
        }
    }
    real cbar[3];
    mean_cross(labels, cbar);
    u.resize(n);
    v.resize(n);
    w.resize(n);
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const size_t c = stokes_detail::lin(i, j, k, grid.nx, grid.ny);
                stokes_detail::cell_faces(e[0].data(), e[1].data(), e[2].data(), grid.nx, grid.ny,
                                          grid.nz, grid.dx, grid.dy, grid.dz, cbar, i, j, k, u[c],
                                          v[c], w[c]);
            }
        }
    }
}

real max_relative_divergence(const PeriodicFaceFluxView& fv) {
    if (fv.u == nullptr || fv.v == nullptr || fv.w == nullptr) {
        throw std::invalid_argument("max_relative_divergence: face array pointer is null");
    }
    if (fv.nx < 1 || fv.ny < 1 || fv.nz < 1) {
        throw std::invalid_argument("max_relative_divergence: cell counts must be >= 1");
    }
    const real ax = fv.dy * fv.dz;
    const real ay = fv.dz * fv.dx;
    const real az = fv.dx * fv.dy;
    real max_div = 0.0;
    real max_u = 0.0;
    for (int k = 0; k < fv.nz; ++k) {
        const int kp = (k + 1 == fv.nz) ? 0 : k + 1;
        for (int j = 0; j < fv.ny; ++j) {
            const int jp = (j + 1 == fv.ny) ? 0 : j + 1;
            for (int i = 0; i < fv.nx; ++i) {
                const int ip = (i + 1 == fv.nx) ? 0 : i + 1;
                const size_t c = stokes_detail::lin(i, j, k, fv.nx, fv.ny);
                const real div = (fv.u[stokes_detail::lin(ip, j, k, fv.nx, fv.ny)] - fv.u[c]) * ax +
                                 (fv.v[stokes_detail::lin(i, jp, k, fv.nx, fv.ny)] - fv.v[c]) * ay +
                                 (fv.w[stokes_detail::lin(i, j, kp, fv.nx, fv.ny)] - fv.w[c]) * az;
                max_div = std::max(max_div, std::fabs(div));
                max_u = std::max(max_u, std::fabs(fv.u[c]));
            }
        }
    }
    const real denom = max_u * ax;
    if (denom == 0.0) {
        return 0.0;
    }
    return max_div / denom;
}

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
