#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <cstdio>

namespace rwpt {

// CUDA runtime error checking
inline void cuda_check_impl(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::string msg = std::string("CUDA error at ") + file + ":" + 
                         std::to_string(line) + " - " + cudaGetErrorString(err);
        throw std::runtime_error(msg);
    }
}

#define RWPT_CUDA_CHECK(expr) ::rwpt::cuda_check_impl((expr), __FILE__, __LINE__)

// cuBLAS error checking
inline void cublas_check_impl(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::string msg = std::string("cuBLAS error at ") + file + ":" + 
                         std::to_string(line) + " - code " + std::to_string(status);
        throw std::runtime_error(msg);
    }
}

#define RWPT_CUBLAS_CHECK(expr) ::rwpt::cublas_check_impl((expr), __FILE__, __LINE__)

} // namespace rwpt
