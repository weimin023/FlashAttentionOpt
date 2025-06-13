cmake_minimum_required(VERSION 3.10)
project(flashattention LANGUAGES C CXX CUDA)

enable_testing()

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream per-thread -std=c++17 --expt-relaxed-constexpr -Xcompiler=-fPIE -Xcudafe --diag_suppress=611")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo -std=c++17 --extended-lambda -Xcudafe --diag_suppress=611")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -std=c++17 --extended-lambda -Xcudafe --diag_suppress=611 -g -G")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/layers
)

find_package(GTest REQUIRED)

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

include(GoogleTest)
gtest_discover_tests(unittest)
