#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cnpy.h"

inline void save_npy(const thrust::device_vector<float> &d_to_save, int N, std::string fname) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    thrust::host_vector<float> h_out = d_to_save;
    cnpy::npy_save(fname, h_out.data(), {static_cast<size_t>(N)}, "w");
}
