#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cnpy.h"
#include "cublas_v2.h"

#define CHECK_CUBLAS(call)                                                                                                                                                                               \
    do {                                                                                                                                                                                               \
        cublasStatus_t err = call;                                                                                                                                                                        \
        if (err != CUBLAS_STATUS_SUCCESS) {                                                                                                                                                                      \
            fprintf(stderr, "cuBLAS failed with error code %s at line %d in file %s\n", cublasGetStatusName(err), __LINE__, __FILE__);                                                                    \
            exit(EXIT_FAILURE);                                                                                                                                                                        \
        }                                                                                                                                                                                              \
    } while (0)

inline void save_npy(const thrust::device_vector<float> &d_to_save, int batch, int M, int N, std::string fname) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    thrust::host_vector<float> h_out = d_to_save;
    cnpy::npy_save(fname, h_out.data(), {static_cast<size_t>(batch), static_cast<size_t>(M), static_cast<size_t>(N)}, "w");
}