cmake_minimum_required(VERSION 3.10)
project(flashattention LANGUAGES C CXX CUDA)

enable_testing()

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread -std=c++17 --expt-relaxed-constexpr -Xcompiler=-fPIE -Xcudafe --diag_suppress=611")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo -std=c++17 --extended-lambda -Xcudafe --diag_suppress=611")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -std=c++17 --extended-lambda -Xcudafe --diag_suppress=611 -g -G")

set(LIBTORCH_PATH "/home/weimin.chen/Downloads/libtorch")

include_directories(
    /usr/include/python3.10/
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/layers
    ${LIBTORCH_PATH}/include
    ${LIBTORCH_PATH}/include/torch/csrc/api/include
)
find_package(GTest REQUIRED)
find_package(pybind11 REQUIRED)
link_directories(${LIBTORCH_PATH}/lib)
#-----------------------------------------------#
#                   CUTLASS                     #
#-----------------------------------------------#
find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS cutlass/include)
find_path(CUTLASS_UTILS_INCLUDE_DIR cutlass/util/host_tensor.h HINTS cutlass/tools/util/include)

# Find cnpy if installed system-wide
find_path(CNPY_INCLUDE_DIR cnpy.h PATHS /usr/local/include /usr/include)
find_library(CNPY_LIBRARY cnpy PATHS /usr/local/lib /usr/lib)

if(NOT CNPY_INCLUDE_DIR OR NOT CNPY_LIBRARY)
    message(FATAL_ERROR "Could not find cnpy. Did you install it correctly?")
endif()


add_executable(unittest unittest.cu)
target_include_directories(unittest PRIVATE ${CNPY_INCLUDE_DIR} ${GTEST_INCLUDE_DIRS})
target_link_libraries(unittest PRIVATE GTest::gtest_main cudadevrt)

add_executable(rmsnorm my_layers/RMSNorm.cu)
target_link_libraries(rmsnorm PRIVATE cudadevrt ${CNPY_LIBRARY})

add_executable(selfAttention my_layers/selfAttention.cu)
target_link_libraries(selfAttention PRIVATE cudadevrt ${CNPY_LIBRARY})

add_executable(GEMM_cutlass_splitK my_layers/GEMM_cutlass_splitK.cu)
target_link_libraries(GEMM_cutlass_splitK PRIVATE 
    torch
    c10
    torch_cpu
    torch_cuda
)
target_include_directories(GEMM_cutlass_splitK PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
target_compile_options(GEMM_cutlass_splitK PRIVATE --expt-relaxed-constexpr)

add_executable(GEMM_cutlass_default my_layers/GEMM_cutlass_default.cu)
target_link_libraries(GEMM_cutlass_default PRIVATE 
    torch
    c10
    torch_cpu
    torch_cuda
)
target_include_directories(GEMM_cutlass_default PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
target_compile_options(GEMM_cutlass_default PRIVATE --expt-relaxed-constexpr)

pybind11_add_module(my_cutlass my_layers/adaptor.cu)
target_include_directories(my_cutlass PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
target_link_libraries(my_cutlass PRIVATE ${TORCH_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(unittest)
