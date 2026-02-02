#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>

namespace rwpt {

// Non-owning view into device memory
template<typename T>
class DeviceSpan {
public:
    DeviceSpan() : ptr_(nullptr), n_(0) {}

    DeviceSpan(T* ptr, size_t n) : ptr_(ptr), n_(n) {}
    
    // Conversion from non-const to const (safe direction)
    template<typename U = T>
    __host__ __device__
    DeviceSpan(const DeviceSpan<typename std::remove_const<U>::type>& other,
               typename std::enable_if<std::is_const<U>::value>::type* = nullptr)
        : ptr_(const_cast<T*>(other.data())), n_(other.size()) {}

    __host__ __device__
    T* data() const { return ptr_; }
    
    __host__ __device__
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
