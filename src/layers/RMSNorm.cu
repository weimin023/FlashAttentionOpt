#include <cuda_runtime.h>
#include "common.h"

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

template<int BLOCK_DIM>
__global__ void RMSNorm_warp_single_block_kernel(float *d_out, const float *d_src, const float *w, int N, float eps) {
    int tid = threadIdx.x;

    float threadSum = 0;
    for (int i = tid; i < N; i += WARP_SIZE) {
        float val = d_src[i];
        threadSum += val * val;
    }
    __syncwarp();

    for (int offset = 16; offset > 0; offset /= 2) {
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    }
    __syncwarp();

    threadSum = __shfl_sync(0xffffffff, threadSum, 0);

    for (int i = tid; i < N; i += WARP_SIZE) {
        d_out[i] = d_src[i] * rsqrtf(threadSum/N + eps) * w[i];
    }
}

template<int BLOCK_DIM>
__global__ void RMSNorm_warp_optimized_kernel(float *d_out, const float *d_src, const float *w, int N, float eps) {
    __shared__ float warp_sums[BLOCK_DIM / 32];  // 存储每个 warp 的结果
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Step 1: 每个线程局部计算
    float thread_sum = 0.0f;
    for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
        float val = d_src[i];
        thread_sum += val * val;
    }
    
    // Step 2: warp 内归约
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Step 3: 每个 warp 的第 0 lane 写入 shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Step 4: 第一个 warp 对所有 warp 结果求和
    if (warp_id == 0) {
        // 让所有 32 个线程都参与，但只有需要的线程加载数据
        float warp_sum = (lane_id < (BLOCK_DIM / 32)) ? warp_sums[lane_id] : 0.0f;
        
        // 现在整个 warp 都参与 shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;  // 存储 block 总和
        }
    }
    __syncthreads();
    
    // Step 5: 计算 scale 并输出
    float scale = rsqrtf(warp_sums[0] / N + eps);
    if (gid < N) {
        d_out[gid] = d_src[gid] * scale * w[gid];
    }
}

int main() {
    int N = 512*512;
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

    // ========================= launch
    RMSNorm_kernel<BLOCK_DIM><<<grid, block>>>(thrust::raw_pointer_cast(d_out.data()),
                                               thrust::raw_pointer_cast(d_src.data()),
                                               thrust::raw_pointer_cast(d_w.data()),
                                               N, eps);

    save_npy(d_out, N, "../layers/npy_verify/rmsnorm.npy");

    thrust::device_vector<float> d_out2(N);

    // ========================= launch warp opt.
    RMSNorm_warp_single_block_kernel<BLOCK_DIM><<<1, 32>>>(thrust::raw_pointer_cast(d_out2.data()),
                                                thrust::raw_pointer_cast(d_src.data()),
                                                thrust::raw_pointer_cast(d_w.data()),
                                                N, eps);


    save_npy(d_out2, N, "../layers/npy_verify/rmsnorm_warp_opt.npy");

    thrust::device_vector<float> d_out3(N);

    // ========================= launch warp opt.
    RMSNorm_warp_optimized_kernel<BLOCK_DIM><<<grid, block>>>(thrust::raw_pointer_cast(d_out3.data()),
                                                    thrust::raw_pointer_cast(d_src.data()),
                                                    thrust::raw_pointer_cast(d_w.data()),
                                                    N, eps);

    save_npy(d_out3, N, "../layers/npy_verify/rmsnorm_warp2.npy");

    return 0;
}