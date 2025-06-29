#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"
#include <cfloat>
#include <torch/extension.h>


template<int TILE_K>
__global__ void cu_fmha_kernel_multihead(
    const float* __restrict__ Q, // [B, H, S_q, D_head]
    const float* __restrict__ K, // [B, H, S_k, D_head]
    const float* __restrict__ V, // [B, H, S_k, D_head]
    float* __restrict__ O,       // [B, H, S_q, D_head]
    int B, int H, int S_q, int S_k, int D_head
) {
    extern __shared__ float shm[];

    float* Ks = shm;
    float* Vs = shm + TILE_K * D_head;
    float* Qs = Vs + TILE_K * D_head;

    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_idx = blockIdx.x;  // one block per query row
    int tid = threadIdx.x;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    const float *q_ptr = Q + b * H * S_q * D_head + h * S_q * D_head + q_idx * D_head;

    if (tid >= D_head) return;

    Qs[tid] = q_ptr[tid];

    float o = 0.0f;
    for (int t = 0; t < S_k; t += TILE_K) {
        int tile_size = min(TILE_K, S_k - t);

        // Load K and V tile
        int k_idx = t + threadIdx.y; // 假設 blockDim.y == TILE_K
        if (k_idx < S_k && tid < D_head) {
            Ks[threadIdx.y * D_head + tid] = K[b*H*S_k*D_head + h*S_k*D_head + k_idx*D_head + tid];
            Vs[threadIdx.y * D_head + tid] = V[b*H*S_k*D_head + h*S_k*D_head + k_idx*D_head + tid];
        } else if (tid < D_head) {
            Ks[threadIdx.y * D_head + tid] = 0.0f;
            Vs[threadIdx.y * D_head + tid] = 0.0f;
        }
        __syncthreads();

        // 計算 dot product 和 softmax online 更新
        for (int ki = 0; ki < tile_size; ++ki) {
            float dot = 0.0f;
            for (int d = 0; d < D_head; ++d) {
                dot += Qs[d] * Ks[ki * D_head + d];
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

            // 更新輸出 Os
            o = o * scale + Vs[ki * D_head + tid] * (exp_diff / l_i);
        }
        __syncthreads();
    }

    // 寫回 global memory
    O[b*H*S_q*D_head + h*S_q*D_head + q_idx*D_head + tid] = o;
}


void test_cu_fmha_kernel_multihead() {
    constexpr int TILE_K = 8;

    int batch = 10;
    int head = 5;
    int seq_len_q = 32;
    int d_head = 16;
    int seq_len_k = 32;

    thrust::host_vector<float> h_Q(batch*head*seq_len_q*d_head); // [batch, head, seq_len_q, d_head]
    thrust::host_vector<float> h_K(batch*head*seq_len_k*d_head); // [batch, head, seq_len_k, d_head]
    thrust::host_vector<float> h_V(batch*head*seq_len_k*d_head); // [batch, head, seq_len_k, d_head]

    // Q[b,h,q,d] = b + h + 1 + 0.01*d
    for (int b = 0; b < batch; ++b)
        for (int h_ = 0; h_ < head; ++h_)
            for (int q = 0; q < seq_len_q; ++q)
                for (int d = 0; d < d_head; ++d) {
                    int idx = (((b * head + h_) * seq_len_q + q) * d_head) + d;
                    h_Q[idx] = static_cast<float>(b + h_ + 1.0 + 0.01 * d);
                }

    // K[b,h,k,d] = sin(k + d + b + h)
    for (int b = 0; b < batch; ++b)
        for (int h_ = 0; h_ < head; ++h_)
            for (int k = 0; k < seq_len_k; ++k)
                for (int d = 0; d < d_head; ++d) {
                    int idx = (((b * head + h_) * seq_len_k + k) * d_head) + d;
                    h_K[idx] = sinf(static_cast<float>(k + d + b + h_));
                }

    // V[b,h,k,d] = (b+1)*(h+1)*(k%d == d)
    for (int b = 0; b < batch; ++b)
        for (int h_ = 0; h_ < head; ++h_)
            for (int k = 0; k < seq_len_k; ++k)
                for (int d = 0; d < d_head; ++d) {
                    int idx = (((b * head + h_) * seq_len_k + k) * d_head) + d;
                    h_V[idx] = (k % d_head == d) ? float((b + 1) * (h_ + 1)) : 0.0f;
                }

    thrust::device_vector<float> d_Q = h_Q;
    thrust::device_vector<float> d_K = h_K;
    thrust::device_vector<float> d_V = h_V;
    thrust::device_vector<float> d_O(batch*head*seq_len_q*d_head); // [batch, head, seq_len_q, d_head]

    dim3 grid(seq_len_q, head, batch);   // x=seq_q, y=num_heads, z=batch
    dim3 block(128, TILE_K); // 128 thread x TILE_K blockDim.y

    size_t shared_mem = 3 * TILE_K * d_head * sizeof(float);

    cu_fmha_kernel_multihead<TILE_K><<<grid, block, shared_mem>>>(
        d_Q.data().get(), d_K.data().get(), d_V.data().get(), d_O.data().get(), batch, head, seq_len_q, seq_len_k, d_head
    );

    save_npy(d_O, batch*head, seq_len_q, d_head, "../my_layers/npy_verify/cu_fmha_kernel_multihead.npy");
}

int main() {
    test_cu_fmha_kernel_multihead();

    return 0;
}