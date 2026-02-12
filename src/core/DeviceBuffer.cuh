#pragma once

#include "DeviceSpan.cuh"
#include "../runtime/cuda_check.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

namespace macroflow3d {

// RAII wrapper for device memory allocation with capacity
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), n_(0), capacity_(0) {}

    explicit DeviceBuffer(size_t n) : ptr_(nullptr), n_(n), capacity_(0) {
        if (n > 0) {
            MACROFLOW3D_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
            capacity_ = n;
        }
    }

    ~DeviceBuffer() {
        reset();
    }

    // Delete copy semantics
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), n_(other.n_), capacity_(other.capacity_) {
        other.ptr_ = nullptr;
        other.n_ = 0;
        other.capacity_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            n_ = other.n_;
            capacity_ = other.capacity_;
            other.ptr_ = nullptr;
            other.n_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    
    size_t size() const { return n_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return n_ == 0; }

    DeviceSpan<T> span() {
        return DeviceSpan<T>(ptr_, n_);
    }

    DeviceSpan<const T> span() const {
        return DeviceSpan<const T>(ptr_, n_);
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            // Best effort free, no exceptions (safe for destructor)
            cudaFree(ptr_);
            ptr_ = nullptr;
            n_ = 0;
            capacity_ = 0;
        }
    }

    void resize(size_t n) {
        // Only reallocate if required size exceeds capacity
        if (n > capacity_) {
            reset();
            if (n > 0) {
                MACROFLOW3D_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
                capacity_ = n;
            }
        }
        n_ = n;
    }

    // Alias for clarity: guarantee room for at least n elements
    void ensure_capacity(size_t n) { resize(n); }
    
    void swap(DeviceBuffer& other) noexcept {
        T* tmp_ptr = ptr_;
        size_t tmp_n = n_;
        size_t tmp_cap = capacity_;
        
        ptr_ = other.ptr_;
        n_ = other.n_;
        capacity_ = other.capacity_;
        
        other.ptr_ = tmp_ptr;
        other.n_ = tmp_n;
        other.capacity_ = tmp_cap;
    }

private:
    T* ptr_;
    size_t n_;
    size_t capacity_;
};

} // namespace macroflow3d
