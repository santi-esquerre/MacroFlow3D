#pragma once

/**
 * @file PinnedHostBuffer.hpp
 * @brief RAII wrapper for CUDA pinned (page-locked) host memory.
 *
 * Allocated once in prepare/init phase, reused across steps.
 * Supports async D2H copies via cudaMemcpyAsync.
 *
 * HPC contract: NO allocations in the hot loop.
 */

#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace macroflow3d {
namespace runtime {

/**
 * @brief RAII pinned host buffer.
 *
 * Use: allocate once (in prepare), memcpy async per step, read after sync.
 */
template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() : ptr_(nullptr), n_(0) {}

    explicit PinnedHostBuffer(size_t n) : ptr_(nullptr), n_(0) {
        allocate(n);
    }

    ~PinnedHostBuffer() { free(); }

    // Non-copyable
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    // Movable
    PinnedHostBuffer(PinnedHostBuffer&& o) noexcept
        : ptr_(o.ptr_), n_(o.n_) { o.ptr_ = nullptr; o.n_ = 0; }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& o) noexcept {
        if (this != &o) {
            free();
            ptr_ = o.ptr_; n_ = o.n_;
            o.ptr_ = nullptr; o.n_ = 0;
        }
        return *this;
    }

    /// Allocate (call ONCE in prepare phase, not in hot loop)
    void allocate(size_t n) {
        if (n > n_) {
            free();
            if (n > 0) {
                cudaHostAlloc(&ptr_, n * sizeof(T), cudaHostAllocDefault);
                n_ = n;
            }
        }
    }

    /// Async copy from device to this pinned buffer
    void copy_from_device_async(const T* d_src, size_t count,
                                 cudaStream_t stream) {
        cudaMemcpyAsync(ptr_, d_src, count * sizeof(T),
                        cudaMemcpyDeviceToHost, stream);
    }

    T*       data()       { return ptr_; }
    const T* data() const { return ptr_; }
    size_t   size() const { return n_; }

    T& operator[](size_t i)       { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

private:
    void free() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            n_ = 0;
        }
    }

    T*     ptr_;
    size_t n_;
};

/**
 * @brief Pre-allocated host-side snapshot staging buffers.
 *
 * Holds pinned memory for positions (wrapped + optional unwrapped).
 * Allocated once, reused across all snapshot events.
 */
template <typename T>
struct SnapshotStaging {
    PinnedHostBuffer<T> x, y, z;
    PinnedHostBuffer<T> x_u, y_u, z_u;  // unwrapped (optional)
    int capacity = 0;
    bool has_unwrapped = false;

    /// Allocate all buffers (call ONCE in prepare)
    void allocate(int n_particles, bool need_unwrap) {
        capacity = n_particles;
        has_unwrapped = need_unwrap;
        x.allocate(n_particles);
        y.allocate(n_particles);
        z.allocate(n_particles);
        if (need_unwrap) {
            x_u.allocate(n_particles);
            y_u.allocate(n_particles);
            z_u.allocate(n_particles);
        }
    }

    /// Async copy wrapped positions from device
    void stage_wrapped_async(const T* dx, const T* dy, const T* dz,
                             int n, cudaStream_t stream) {
        x.copy_from_device_async(dx, n, stream);
        y.copy_from_device_async(dy, n, stream);
        z.copy_from_device_async(dz, n, stream);
    }

    /// Async copy unwrapped positions from device
    void stage_unwrapped_async(const T* dxu, const T* dyu, const T* dzu,
                               int n, cudaStream_t stream) {
        x_u.copy_from_device_async(dxu, n, stream);
        y_u.copy_from_device_async(dyu, n, stream);
        z_u.copy_from_device_async(dzu, n, stream);
    }
};

} // namespace runtime
} // namespace macroflow3d
