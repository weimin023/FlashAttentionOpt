#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"
#include <cfloat>
#include <torch/extension.h>


template<int TILE_K>
__global__ void fmha_kernel_multihead(
    const float* __restrict__ Q, // [B, H, S_q, D_head]
    const float* __restrict__ K, // [B, H, S_k, D_head]
    const float* __restrict__ V, // [B, H, S_k, D_head]
    float* __restrict__ O,       // [B, H, S_q, D_head]
    int B, int H, int S_q, int S_k, int D_head
) {
    extern __shared__ float shm[];  // size: 2 * TILE_K * D_head floats

    float* Ks = shm;
    float* Vs = shm + TILE_K * D_head;

    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_idx = blockIdx.x;  // one block per query row
    int tid = threadIdx.x;

    // --- 初始化最大值與sum ---
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Load Q vector for this (b,h,q_idx)
    const float* q_ptr = Q + b*H*S_q*D_head + h*S_q*D_head + q_idx*D_head;
    float q_vec[128]; // assume max D_head <= 128

    // 初始化 q_vec 全為0 避免未初始化風險
#pragma unroll
    for (int i = 0; i < D_head; ++i) {
        q_vec[i] = 0.0f;
    }
    for (int d = tid; d < D_head; d += blockDim.x) {
        q_vec[d] = q_ptr[d];
    }
    __syncthreads();

    // 初始化 output accumulator o_vec
    float o_vec[128];
#pragma unroll
    for (int i = 0; i < D_head; ++i) {
        o_vec[i] = 0.0f;
    }

    // 總是要用 float scale 為 softmax 穩定性
    for (int t = 0; t < S_k; t += TILE_K) {
        int tile_size = min(TILE_K, S_k - t);

        // Load K and V tile
        int k_idx = t + threadIdx.y; // 假設 blockDim.y == TILE_K
        if (k_idx < S_k && tid < D_head) {
            Ks[threadIdx.y * D_head + tid] = K[b*H*S_k*D_head + h*S_k*D_head + k_idx*D_head + tid];
            Vs[threadIdx.y * D_head + tid] = V[b*H*S_k*D_head + h*S_k*D_head + k_idx*D_head + tid];
        } else if (tid < D_head) {
            // 對超出部分給0，避免nan
            Ks[threadIdx.y * D_head + tid] = 0.0f;
            Vs[threadIdx.y * D_head + tid] = 0.0f;
        }
        __syncthreads();

        // 計算 dot product 和 softmax online 更新
        for (int ki = 0; ki < tile_size; ++ki) {
            float dot = 0.0f;
            for (int d = 0; d < D_head; ++d) {
                dot += q_vec[d] * Ks[ki * D_head + d];
            }
            dot /= sqrtf((float)D_head);

            // online softmax 算法
            float prev_m = m_i;
            m_i = fmaxf(m_i, dot);

            // scale factor for numerically stable softmax
            float exp_diff = expf(dot - m_i);
            float scale = 0.0f;

            if (m_i == prev_m) {
                scale = l_i / (l_i + exp_diff);
                l_i = l_i + exp_diff;
            } else {
                scale = (l_i * expf(prev_m - m_i)) / (l_i * expf(prev_m - m_i) + exp_diff);
                l_i = l_i * expf(prev_m - m_i) + exp_diff;
            }

            // 更新輸出 o_vec
            for (int d = 0; d < D_head; ++d) {
                o_vec[d] = o_vec[d] * scale + Vs[ki * D_head + d] * (exp_diff / l_i);
            }
        }
        __syncthreads();
    }

    // 寫回 global memory
    for (int d = tid; d < D_head; d += blockDim.x) {
        O[b*H*S_q*D_head + h*S_q*D_head + q_idx*D_head + d] = o_vec[d];
    }
}


void cu_flash_attn_multihead_v0(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int B, int H, int S_q, int S_k, int D_head
) {
    constexpr int TILE_K = 8;  // 你可調整tile大小

    dim3 grid(S_q, H, B);   // x=seq_q, y=num_heads, z=batch
    dim3 block(128, TILE_K); // 128 thread x TILE_K blockDim.y

    size_t shared_mem = 2 * TILE_K * D_head * sizeof(float);

    fmha_kernel_multihead<TILE_K><<<grid, block, shared_mem>>>(
        d_Q, d_K, d_V, d_O, B, H, S_q, S_k, D_head
    );
}


template<int TILE_SIZE> __global__ void gemm_AB_kernel(const float *dA, const float *dB, float *dC, int M, int K, int N) {
    int c = threadIdx.x;
    int r = threadIdx.y;
    
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    int offset_A = blockIdx.z * M * K;
    int offset_B = blockIdx.z * N * K;
    int offset_C = blockIdx.z * M * N;

    __shared__ float SA[TILE_SIZE][TILE_SIZE];
    __shared__ float SB[TILE_SIZE][TILE_SIZE];

    float reg_tile = 0;
    for (int t = 0; t < K; t += TILE_SIZE) {

        if (row < M && (t + c) < K) {
            SA[r][c] = dA[offset_A + row * K + (t + c)];
        } else {
            SA[r][c] = 0;
        }

        // Load B with transposed access
        // Original: SB[r][c] = dB[(t + r) * N + col];
        // Transposed: Bᵗ[col][t + r] == B[t + r][col]
        if (col < N && (t + r) < K) {
            SB[r][c] = dB[offset_B + (t + r) * N + col];  // Notice the change
        } else {
            SB[r][c] = 0.0f;
        }

        __syncthreads();
        
        // accumulate sum
        // global idx = i * N + j;
        for (int k = 0; k < TILE_SIZE; ++k) {
            reg_tile += SA[r][k] * SB[k][c];
        }
        
    }

    if (row < M && col < N) {
        dC[offset_C + row * N + col] = reg_tile;
    }
}

void cu_gemm_AB(int batch,
    float *out,
    const float *Q, // (batch, seq_len_q, d_k)
    const float *K, // (batch, d_k, seq_len_k)
    int seq_len_q,
    int seq_len_k,
    int d_k) {

    dim3 threads(32, 32);  // TILE_SIZE x TILE_SIZE
    dim3 grid((seq_len_k + threads.x - 1)/threads.x, (seq_len_q + threads.y - 1)/threads.y, batch);
    gemm_AB_kernel<32><<<grid, threads>>>(Q, K, out, seq_len_q, d_k, seq_len_k);
}