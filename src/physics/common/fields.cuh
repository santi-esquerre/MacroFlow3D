#pragma once

/**
 * @file fields.cuh
 * @brief Field types for physics modules (cell-centered and staggered)
 * 
 * These types own their GPU memory via DeviceBuffer and provide
 * dimension-aware access. No virtual functions, no runtime polymorphism.
 * 
 * Layout conventions (from legacy compute_velocity_from_head_v1.cu):
 *   - K, h: cell-centered, dims (nx, ny, nz)
 *   - U: face-centered in x, dims (nx+1, ny, nz)
 *   - V: face-centered in y, dims (nx, ny+1, nz)
 *   - W: face-centered in z, dims (nx, ny, nz+1)
 * 
 * Memory layout: column-major (x fastest), i.e. idx = i + j*stride_j + k*stride_k
 */

#include "../../core/Grid3D.hpp"
#include "../../core/DeviceBuffer.cuh"
#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

namespace rwpt {
namespace physics {

// ============================================================================
// Cell-centered scalar field (K, h, etc.)
// ============================================================================

/**
 * @brief Cell-centered scalar field
 * 
 * Owns GPU memory for a 3D scalar field defined at cell centers.
 * Dimensions match the grid: (nx, ny, nz)
 */
struct ScalarField {
    int nx = 0, ny = 0, nz = 0;
    real dx = 1.0, dy = 1.0, dz = 1.0;
    DeviceBuffer<real> data;
    
    ScalarField() = default;
    
    explicit ScalarField(const Grid3D& grid)
        : nx(grid.nx), ny(grid.ny), nz(grid.nz)
        , dx(grid.dx), dy(grid.dy), dz(grid.dz)
        , data(grid.num_cells())
    {}
    
    ScalarField(int nx_, int ny_, int nz_, real dx_ = 1.0, real dy_ = 1.0, real dz_ = 1.0)
        : nx(nx_), ny(ny_), nz(nz_)
        , dx(dx_), dy(dy_), dz(dz_)
        , data(static_cast<size_t>(nx_) * ny_ * nz_)
    {}
    
    // Resize (reallocates if needed)
    void resize(const Grid3D& grid) {
        nx = grid.nx; ny = grid.ny; nz = grid.nz;
        dx = grid.dx; dy = grid.dy; dz = grid.dz;
        data.resize(grid.num_cells());
    }
    
    void resize(int nx_, int ny_, int nz_) {
        nx = nx_; ny = ny_; nz = nz_;
        data.resize(static_cast<size_t>(nx_) * ny_ * nz_);
    }
    
    // Accessors
    size_t size() const { return static_cast<size_t>(nx) * ny * nz; }
    bool empty() const { return data.size() == 0; }
    
    DeviceSpan<real> span() { return data.span(); }
    DeviceSpan<const real> span() const { return data.span(); }
    
    real* device_ptr() { return data.data(); }
    const real* device_ptr() const { return data.data(); }
    
    // Convert to Grid3D (useful for passing to existing APIs)
    Grid3D grid() const { return Grid3D(nx, ny, nz, dx, dy, dz); }
    
    // Linear index (host-side helper, not for kernels)
    size_t idx(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * nx + static_cast<size_t>(k) * nx * ny;
    }
};

// Type aliases for semantic clarity
using KField = ScalarField;      ///< Conductivity field (cell-centered)
using HeadField = ScalarField;   ///< Hydraulic head field (cell-centered)

// ============================================================================
// Staggered velocity field (U, V, W on faces)
// ============================================================================

/**
 * @brief Staggered velocity field (face-centered components)
 * 
 * Legacy layout (compute_velocity_from_head_v1.cu):
 *   U[idx] where idx = ix + iy*(Nx+1) + iz*(Nx+1)*Ny, dims (Nx+1, Ny, Nz)
 *   V[idx] where idx = ix + iy*Nx + iz*Nx*(Ny+1),     dims (Nx, Ny+1, Nz)
 *   W[idx] where idx = ix + iy*Nx + iz*Nx*Ny,         dims (Nx, Ny, Nz+1)
 * 
 * Note: U is x-velocity at x-faces, V is y-velocity at y-faces, etc.
 */
struct VelocityField {
    // Base grid dimensions (cell-centered)
    int nx = 0, ny = 0, nz = 0;
    real dx = 1.0, dy = 1.0, dz = 1.0;
    
    // Face-centered components (staggered)
    DeviceBuffer<real> U;  // dims: (nx+1, ny, nz)
    DeviceBuffer<real> V;  // dims: (nx, ny+1, nz)
    DeviceBuffer<real> W;  // dims: (nx, ny, nz+1)
    
    VelocityField() = default;
    
    explicit VelocityField(const Grid3D& grid)
        : nx(grid.nx), ny(grid.ny), nz(grid.nz)
        , dx(grid.dx), dy(grid.dy), dz(grid.dz)
    {
        allocate();
    }
    
    VelocityField(int nx_, int ny_, int nz_, real dx_ = 1.0, real dy_ = 1.0, real dz_ = 1.0)
        : nx(nx_), ny(ny_), nz(nz_)
        , dx(dx_), dy(dy_), dz(dz_)
    {
        allocate();
    }
    
    // Allocate buffers based on current dims
    void allocate() {
        if (nx > 0 && ny > 0 && nz > 0) {
            U.resize(size_U());
            V.resize(size_V());
            W.resize(size_W());
        }
    }
    
    // Resize (reallocates if needed)
    void resize(const Grid3D& grid) {
        nx = grid.nx; ny = grid.ny; nz = grid.nz;
        dx = grid.dx; dy = grid.dy; dz = grid.dz;
        allocate();
    }
    
    void resize(int nx_, int ny_, int nz_) {
        nx = nx_; ny = ny_; nz = nz_;
        allocate();
    }
    
    // Size helpers
    size_t size_U() const { return static_cast<size_t>(nx + 1) * ny * nz; }
    size_t size_V() const { return static_cast<size_t>(nx) * (ny + 1) * nz; }
    size_t size_W() const { return static_cast<size_t>(nx) * ny * (nz + 1); }
    size_t total_size() const { return size_U() + size_V() + size_W(); }
    
    bool empty() const { return U.size() == 0; }
    
    // Spans for kernel access
    DeviceSpan<real> U_span() { return U.span(); }
    DeviceSpan<real> V_span() { return V.span(); }
    DeviceSpan<real> W_span() { return W.span(); }
    
    DeviceSpan<const real> U_span() const { return U.span(); }
    DeviceSpan<const real> V_span() const { return V.span(); }
    DeviceSpan<const real> W_span() const { return W.span(); }
    
    // Pointers for kernel access
    real* U_ptr() { return U.data(); }
    real* V_ptr() { return V.data(); }
    real* W_ptr() { return W.data(); }
    
    const real* U_ptr() const { return U.data(); }
    const real* V_ptr() const { return V.data(); }
    const real* W_ptr() const { return W.data(); }
    
    // Strides for indexing (host-side helpers)
    // U: idx = i + j*(nx+1) + k*(nx+1)*ny
    int stride_U_j() const { return nx + 1; }
    int stride_U_k() const { return (nx + 1) * ny; }
    
    // V: idx = i + j*nx + k*nx*(ny+1)
    int stride_V_j() const { return nx; }
    int stride_V_k() const { return nx * (ny + 1); }
    
    // W: idx = i + j*nx + k*nx*ny
    int stride_W_j() const { return nx; }
    int stride_W_k() const { return nx * ny; }
    
    // Linear index helpers (host-side, not for kernels)
    size_t idx_U(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * stride_U_j() + static_cast<size_t>(k) * stride_U_k();
    }
    size_t idx_V(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * stride_V_j() + static_cast<size_t>(k) * stride_V_k();
    }
    size_t idx_W(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * stride_W_j() + static_cast<size_t>(k) * stride_W_k();
    }
    
    // Convert to Grid3D (base cell-centered grid)
    Grid3D grid() const { return Grid3D(nx, ny, nz, dx, dy, dz); }
};

} // namespace physics
} // namespace rwpt
