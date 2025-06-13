#include <cuda_runtime.h>
#include "common.h"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        return; \
    }

#define WARP_SIZE 32

template<int BLOCK_DIM>
__global__ void RMSNorm_warp_single_block_kernel(float *d_out, const float *d_src, const float *w, int N, float eps) {

    __shared__ float warp_sums[BLOCK_DIM / 32];
    __shared__ float total_sum;

    int tid = threadIdx.x;

    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    float threadSum = 0;
    for (int i = tid; i < N; i += BLOCK_DIM) {
        float val = d_src[i];
        threadSum += val * val;
    }
    __syncthreads();

    for (int offset = 16; offset > 0; offset /= 2) {
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    }
    __syncthreads();

    if (laneId == 0) {
        warp_sums[warpId] = threadSum;
    }

    // Step 4: 第一个 warp 对所有 warp 结果求和
    if (warpId == 0) {
        float warp_sum = (laneId < (BLOCK_DIM / 32)) ? warp_sums[laneId] : 0.0f;
        
        // warp 内归约
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (laneId == 0) {
            total_sum = warp_sum;
        }
    }
    __syncthreads();
    
    // Step 5: 所有线程使用相同的 total_sum 进行正规化
    float scale = rsqrtf(total_sum / N + eps);
    for (int i = tid; i < N; i += BLOCK_DIM) {  // 使用 BLOCK_DIM
        d_out[i] = d_src[i] * scale * w[i];
    }
}

int main() {
    int N = 128*128;
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
    thrust::device_vector<float> d_out2(N);

    // ========================= launch warp opt.
    RMSNorm_warp_single_block_kernel<BLOCK_DIM><<<1, 128>>>(thrust::raw_pointer_cast(d_out2.data()),
                                                thrust::raw_pointer_cast(d_src.data()),
                                                thrust::raw_pointer_cast(d_w.data()),
                                                N, eps);


    save_npy(d_out2, N, "../layers/npy_verify/rmsnorm_warp_opt.npy");

    return 0;
}