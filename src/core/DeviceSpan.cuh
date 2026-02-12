#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>

namespace macroflow3d {

// Non-owning view into device memory.
// DeviceSpan<T> for mutable access, DeviceSpan<const T> for read-only.
// Implicit conversion from DeviceSpan<T> → DeviceSpan<const T> is safe.
template<typename T>
class DeviceSpan {
public:
    DeviceSpan() : ptr_(nullptr), n_(0) {}

    DeviceSpan(T* ptr, size_t n) : ptr_(ptr), n_(n) {}
    
    // Conversion from non-const to const (safe direction, no cast needed)
    template<typename U = T,
             typename = typename std::enable_if<
                 std::is_const<U>::value &&
                 std::is_same<U, T>::value>::type>
    __host__ __device__
    DeviceSpan(const DeviceSpan<typename std::remove_const<U>::type>& other)
        : ptr_(other.data()), n_(other.size()) {}

    __host__ __device__
    T* data() const { return ptr_; }
    
    __host__ __device__
    size_t size() const { return n_; }

    __host__ __device__
    bool empty() const { return n_ == 0; }

    // Device-only to prevent dangerous host access to device memory
    __device__
    T& operator[](size_t i) const {
        return ptr_[i];
    }

private:
    T* ptr_;
    size_t n_;
};

} // namespace macroflow3d
