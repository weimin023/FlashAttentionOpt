#include <cuda_runtime.h>
#include <cstdio>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cnpy.h"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        return; \
    }

#define WARP_SIZE 32

template<int BLOCK_DIM>
__global__ void RMSNorm_kernel(float *d_out, const float *d_src, const float *w, int N, float eps) {
    __shared__ float sum_shared;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: 每個 thread 局部平方求和
    float thread_sum = 0.0f;
    for (int i = gid; i < N; i += BLOCK_DIM * gridDim.x) {
        float val = d_src[i];
        thread_sum += val * val;
    }

    // Step 2: warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Step 3: 將每個 warp 的第0 lane 結果寫入 shared
    if (tid % 32 == 0) atomicAdd(&sum_shared, thread_sum);

    __syncthreads();

    // Step 4: 取出 scale
    float scale = rsqrtf(sum_shared / N + eps);

    // Step 5: 正規化輸出
    if (gid < N) {
        float val = d_src[gid];
        d_out[gid] = val * scale * w[gid];
    }
}

int main() {
    int N = 128;
    thrust::host_vector<float> h_src(N), h_w(N);
    for (int i = 0; i < N; ++i) {
        h_src[i] = i;
        h_w[i] = 1.0f;
    }

    thrust::device_vector<float> d_src = h_src;
    thrust::device_vector<float> d_w = h_w;
    thrust::device_vector<float> d_out(N);

    float eps = 1e-5;

    constexpr int BLOCK_DIM = 128;
    dim3 block(BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM);

    // launch
    RMSNorm_kernel<BLOCK_DIM><<<grid, block>>>(thrust::raw_pointer_cast(d_out.data()),
                                               thrust::raw_pointer_cast(d_src.data()),
                                               thrust::raw_pointer_cast(d_w.data()),
                                               N, eps);

    cudaDeviceSynchronize();  // 檢查 kernel 是否 crash
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    thrust::host_vector<float> h_out = d_out;
    cnpy::npy_save("../layers/npy_verify/rmsnorm.npy", h_out.data(), {static_cast<size_t>(N)}, "w");

    return 0;
}