/**
 * @file PollockTracker.cu
 * @brief SF-32 Pollock-type (RT0) semi-analytical cell tracker - GPU engine.
 *
 * See PollockTracker.cuh for the velocity model, the well-conditioned exit
 * forms, the state representation, the drivers and the status codes. The
 * kernels call the very same __host__ __device__ core on the device face
 * arrays.
 */

#include "../../../runtime/cuda_check.cuh"
#include "PollockTracker.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

namespace {

constexpr int kBlockSize = 128;

int grid_size(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

/// Device arrays of the engine state (axis a at offset a * n).
struct StateArrays {
    int32_t* cell;
    int32_t* wrap;
    real* rel;
    real* clock;
    uint32_t* cells;
    int n;
};

__device__ inline void load_state(const StateArrays& s, int i, PollockState& st) {
    for (int a = 0; a < 3; ++a) {
        const size_t o = static_cast<size_t>(a) * static_cast<size_t>(s.n) + static_cast<size_t>(i);
        st.cell[a] = s.cell[o];
        st.w[a] = s.wrap[o];
        st.r[a] = s.rel[o];
    }
    st.t = s.clock[i];
}

__device__ inline void store_state(const StateArrays& s, int i, const PollockState& st) {
    for (int a = 0; a < 3; ++a) {
        const size_t o = static_cast<size_t>(a) * static_cast<size_t>(s.n) + static_cast<size_t>(i);
        s.cell[o] = st.cell[a];
        s.wrap[o] = st.w[a];
        s.rel[o] = st.r[a];
    }
    s.clock[i] = st.t;
}

/// Wrapped SoA position of a state: xi = fma(cell, D, r) reduced to [0, L) by
/// wrap_position, wraps = cell-wrap counters plus the removed periods.
__device__ inline void write_back(const PeriodicFaceFluxView& f, const PollockState& st,
                                  const ParticlesSoA<real>& p, int i) {
    real xi[3];
    int32_t w[3];
    for (int a = 0; a < 3; ++a) {
        xi[a] = fma(static_cast<real>(st.cell[a]), pollock_detail::axis_d(f, a), st.r[a]);
        w[a] = st.w[a];
    }
    const real L[3] = {f.Lx, f.Ly, f.Lz};
    wrap_position(xi, w, L);
    p.x[i] = xi[0];
    p.y[i] = xi[1];
    p.z[i] = xi[2];
    p.wrapX[i] = w[0];
    p.wrapY[i] = w[1];
    p.wrapZ[i] = w[2];
}

__global__ void kernel_prepare(PeriodicFaceFluxView f, ParticlesSoA<real> p, StateArrays s) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= s.n)
        return;
    const real xi[3] = {p.x[i], p.y[i], p.z[i]};
    const int32_t w[3] = {p.wrapX[i], p.wrapY[i], p.wrapZ[i]};
    PollockState st;
    const uint8_t code = pollock_init_state(f, xi, w, st);
    s.cells[i] = 0u;
    if (code != kStatusActive) {
        // Keep the input position; mark the particle and store a neutral state.
        for (int a = 0; a < 3; ++a) {
            st.cell[a] = 0;
            st.w[a] = 0;
            st.r[a] = static_cast<real>(0.0);
        }
        st.t = static_cast<real>(0.0);
        store_state(s, i, st);
        p.status[i] = code;
        return;
    }
    store_state(s, i, st);
}

/// kToX1: pollock_advance_to_x1(a); otherwise pollock_advance_to_time(a).
template <bool kToX1>
__global__ void kernel_advance(PeriodicFaceFluxView f, PollockParams prm, ParticlesSoA<real> p,
                               StateArrays s, real a) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= s.n)
        return;
    if (p.status[i] != kStatusActive)
        return;
    PollockState st;
    load_state(s, i, st);
    PollockCounters cnt{0u};
    uint8_t code;
    if (kToX1) {
        code = pollock_advance_to_x1(f, st, a, prm, cnt);
    } else {
        code = pollock_advance_to_time(f, st, a, prm, cnt);
    }
    store_state(s, i, st);
    s.cells[i] += cnt.cells;
    write_back(f, st, p, i);
    if (code != kStatusActive) {
        p.status[i] = code;
    }
}

void require_wraps(const ParticlesSoA<real>& p, const char* who) {
    if (p.wrapX == nullptr || p.wrapY == nullptr || p.wrapZ == nullptr) {
        throw std::invalid_argument(std::string(who) +
                                    ": a wrap-counter pointer (wrapX, wrapY or wrapZ) is null "
                                    "(wrap arrays are required: triply periodic tracker)");
    }
}

} // namespace

// ============================================================================
// Construction and configuration
// ============================================================================

PollockTracker::PollockTracker(cudaStream_t stream, uint64_t inject_seed)
    : stream_(stream), inject_seed_(inject_seed) {}

void PollockTracker::configure(const PollockConfig& cfg) {
    if (cfg.max_cells_per_call < 1) {
        throw std::invalid_argument("PollockTracker::configure: max_cells_per_call must be >= 1");
    }
    cfg_ = cfg;
    configured_ = true;
}

void PollockTracker::bind_fluxes(const PeriodicFaceFluxView& fluxes) {
    if (fluxes.u == nullptr || fluxes.v == nullptr || fluxes.w == nullptr) {
        throw std::invalid_argument(
            "PollockTracker::bind_fluxes: a face array pointer (u, v or w) is null");
    }
    if (fluxes.nx < 1 || fluxes.ny < 1 || fluxes.nz < 1) {
        throw std::invalid_argument("PollockTracker::bind_fluxes: cell counts must be >= 1");
    }
    const real sp[3] = {fluxes.dx, fluxes.dy, fluxes.dz};
    const real per[3] = {fluxes.Lx, fluxes.Ly, fluxes.Lz};
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(sp[d]) || !(sp[d] > 0.0)) {
            throw std::invalid_argument(
                "PollockTracker::bind_fluxes: a spacing is not finite or not > 0");
        }
        if (!std::isfinite(per[d]) || !(per[d] > 0.0)) {
            throw std::invalid_argument(
                "PollockTracker::bind_fluxes: a period is not finite or not > 0");
        }
    }
    fluxes_ = fluxes;
    fluxes_bound_ = true;
    prepared_ = false;
}

void PollockTracker::bind_particles(ParticlesSoA<real>& p) {
    if (p.x == nullptr || p.y == nullptr || p.z == nullptr) {
        throw std::invalid_argument(
            "PollockTracker::bind_particles: a position pointer (x, y or z) is null");
    }
    if (p.status == nullptr) {
        throw std::invalid_argument("PollockTracker::bind_particles: status pointer is null");
    }
    if (p.n < 0) {
        throw std::invalid_argument("PollockTracker::bind_particles: p.n < 0");
    }
    p_ = p;
    particles_bound_ = true;
    prepared_ = false;
}

void PollockTracker::inject_box(real x0, real y0, real z0, real x1, real y1, real z1, int first,
                                int count) {
    if (!particles_bound_) {
        throw std::logic_error("PollockTracker::inject_box: called before bind_particles");
    }
    streamline_tracker::inject_box(stream_, p_, x0, y0, z0, x1, y1, z1, first, count, inject_seed_);
}

void PollockTracker::ensure_tracking() {
    if (!particles_bound_) {
        throw std::logic_error("PollockTracker::ensure_tracking: called before bind_particles");
    }
    require_wraps(p_, "PollockTracker::ensure_tracking");
}

// ============================================================================
// prepare
// ============================================================================

void PollockTracker::prepare() {
    if (!configured_) {
        throw std::logic_error("PollockTracker::prepare: called before configure");
    }
    if (!fluxes_bound_) {
        throw std::logic_error("PollockTracker::prepare: called before bind_fluxes");
    }
    if (!particles_bound_) {
        throw std::logic_error("PollockTracker::prepare: called before bind_particles");
    }
    require_wraps(p_, "PollockTracker::prepare");

    const size_t n = static_cast<size_t>(p_.n);
    cell_.resize(3 * n);
    wrap_.resize(3 * n);
    rel_.resize(3 * n);
    clock_.resize(n);
    cell_count_.resize(n);

    if (p_.n > 0) {
        StateArrays s{cell_.data(),  wrap_.data(),       rel_.data(),
                      clock_.data(), cell_count_.data(), p_.n};
        kernel_prepare<<<grid_size(p_.n), kBlockSize, 0, stream_>>>(fluxes_, p_, s);
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }
    t_target_ = 0.0;
    prepared_n_ = p_.n;
    prepared_ = true;
}

void PollockTracker::require_prepared(const char* who) const {
    if (!prepared_) {
        throw std::logic_error(std::string(who) + ": called before prepare");
    }
}

// ============================================================================
// Hot path: kernel launches only (no allocation, no synchronization)
// ============================================================================

void PollockTracker::step(real dt) {
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("PollockTracker::step: dt must be finite and >= 0");
    }
    require_prepared("PollockTracker::step");
    t_target_ += dt;
    if (prepared_n_ == 0)
        return;
    const PollockParams prm{cfg_.max_cells_per_call};
    StateArrays s{cell_.data(),  wrap_.data(),       rel_.data(),
                  clock_.data(), cell_count_.data(), prepared_n_};
    kernel_advance<false>
        <<<grid_size(prepared_n_), kBlockSize, 0, stream_>>>(fluxes_, prm, p_, s, t_target_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void PollockTracker::step_to_x1(real x1_target_unwrapped) {
    if (!std::isfinite(x1_target_unwrapped)) {
        throw std::invalid_argument("PollockTracker::step_to_x1: target must be finite");
    }
    require_prepared("PollockTracker::step_to_x1");
    if (prepared_n_ == 0)
        return;
    const PollockParams prm{cfg_.max_cells_per_call};
    StateArrays s{cell_.data(),  wrap_.data(),       rel_.data(),
                  clock_.data(), cell_count_.data(), prepared_n_};
    kernel_advance<true><<<grid_size(prepared_n_), kBlockSize, 0, stream_>>>(fluxes_, prm, p_, s,
                                                                             x1_target_unwrapped);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Contract accessors
// ============================================================================

void PollockTracker::synchronize() {
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

ConstParticlesSoA<real> PollockTracker::particles() const {
    ConstParticlesSoA<real> c;
    c.x = p_.x;
    c.y = p_.y;
    c.z = p_.z;
    c.n = p_.n;
    c.status = p_.status;
    c.wrapX = p_.wrapX;
    c.wrapY = p_.wrapY;
    c.wrapZ = p_.wrapZ;
    return c;
}

void PollockTracker::compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream) {
    if (!fluxes_bound_) {
        throw std::logic_error("PollockTracker::compute_unwrapped: called before bind_fluxes");
    }
    if (!particles_bound_) {
        throw std::logic_error("PollockTracker::compute_unwrapped: called before bind_particles");
    }
    const real L[3] = {fluxes_.Lx, fluxes_.Ly, fluxes_.Lz};
    streamline_tracker::compute_unwrapped(stream, particles(), L, uw);
}

// ============================================================================
// Stats (host; synchronizes; not for the hot loop)
// ============================================================================

PollockStats PollockTracker::compute_stats() {
    require_prepared("PollockTracker::compute_stats");
    const int n = prepared_n_;
    PollockStats out{};
    out.n_particles = n;
    if (n == 0) {
        return out;
    }
    const size_t nn = static_cast<size_t>(n);
    std::vector<uint8_t> status(nn);
    std::vector<uint32_t> cells(nn);
    std::vector<real> clock(nn);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(status.data(), p_.status, nn * sizeof(uint8_t),
                                           cudaMemcpyDeviceToHost, stream_));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(cells.data(), cell_count_.data(), nn * sizeof(uint32_t),
                                           cudaMemcpyDeviceToHost, stream_));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(clock.data(), clock_.data(), nn * sizeof(real),
                                           cudaMemcpyDeviceToHost, stream_));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream_));

    out.min_clock = clock[0];
    out.max_clock = clock[0];
    for (size_t i = 0; i < nn; ++i) {
        switch (status[i]) {
        case kStatusActive:
            ++out.n_active;
            break;
        case kStatusPollockStagnation:
            ++out.n_stagnation;
            break;
        case kStatusSubstepLimit:
            ++out.n_substep_limit;
            break;
        case kStatusNonFinite:
            ++out.n_nonfinite;
            break;
        default:
            ++out.n_other;
            break;
        }
        out.total_cells += cells[i];
        if (cells[i] > out.max_cells)
            out.max_cells = cells[i];
        if (clock[i] < out.min_clock)
            out.min_clock = clock[i];
        if (clock[i] > out.max_clock)
            out.max_clock = clock[i];
    }
    return out;
}

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
