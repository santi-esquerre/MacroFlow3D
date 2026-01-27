#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace rwpt {

// Non-owning view into device memory
template<typename T>
class DeviceSpan {
public:
    DeviceSpan() : ptr_(nullptr), n_(0) {}

    DeviceSpan(T* ptr, size_t n) : ptr_(ptr), n_(n) {}

    T* data() const { return ptr_; }
    
    size_t size() const { return n_; }

    // Device-only to prevent dangerous host access to device memory
    __device__
    T& operator[](size_t i) const {
        return ptr_[i];
    }

private:
    T* ptr_;
    size_t n_;
};

} // namespace rwpt
