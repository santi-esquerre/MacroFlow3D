cmake_minimum_required(VERSION 3.18)
project(rwpt-mdisp-simulator CUDA CXX)

# CUDA setup
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)  # Adjust for your GPU

# Find CUDA toolkit for cuBLAS
find_package(CUDAToolkit REQUIRED)

# Source files
set(RUNTIME_SOURCES
    src/runtime/CudaContext.cu
)

set(BLAS_SOURCES
    src/numerics/blas/fill.cu
    src/numerics/blas/copy.cu
    src/numerics/blas/scal.cu
    src/numerics/blas/axpy.cu
    src/numerics/blas/axpby.cu
    src/numerics/blas/dot.cu
    src/numerics/blas/nrm2.cu
    src/numerics/blas/cg_kernels.cu
)

set(OPERATOR_SOURCES
    src/numerics/operators/poisson3d_operator.cu
)

set(MULTIGRID_SOURCES
    src/multigrid/transfer/restrict_3d.cu
    src/multigrid/transfer/prolong_3d.cu
    src/multigrid/smoothers/residual_3d.cu
    src/multigrid/smoothers/gsrb_3d.cu
    src/multigrid/cycle/v_cycle.cu
)

# Library
add_library(rwpt_lib STATIC
    ${RUNTIME_SOURCES}
    ${BLAS_SOURCES}
    ${OPERATOR_SOURCES}
    ${MULTIGRID_SOURCES}
)

target_include_directories(rwpt_lib PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(rwpt_lib PUBLIC CUDA::cudart CUDA::cublas)

# Apps
add_executable(smoke_device apps/smoke_device.cu)
target_link_libraries(smoke_device rwpt_lib)

add_executable(smoke_core apps/smoke_core.cu)
target_link_libraries(smoke_core rwpt_lib)

add_executable(cg_smoke apps/cg_smoke.cu)
target_link_libraries(cg_smoke rwpt_lib)

add_executable(mg_smoke apps/mg_smoke.cu)
target_link_libraries(mg_smoke rwpt_lib)

# Compile options
set_target_properties(rwpt_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(rwpt_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
