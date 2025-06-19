#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"

#define WARP_SIZE 32
#define theta 10000.0f

template<int TILE_SIZE> __global__ void gemm_ABt_scale_kernel(const float *dA, const float *dB, float *dC, int M, int K, int N, float scale) {

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
            SB[r][c] = dB[offset_B + col * K + (t + r)];  // Notice the change
        } else {
            SB[r][c] = 0.0f;
        }

        // accumulate sum
        // global idx = i * N + j;
        for (int k = 0; k < TILE_SIZE; ++k) {
            reg_tile += SA[r][k] * SB[k][c];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        dC[offset_C + row * N + col] = reg_tile / scale;
    }
    
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

        // accumulate sum
        // global idx = i * N + j;
        for (int k = 0; k < TILE_SIZE; ++k) {
            reg_tile += SA[r][k] * SB[k][c];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        dC[offset_C + row * N + col] = reg_tile;
    }
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int NUM_THREADS = 1024>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template <const int NUM_THREADS = 1024>
__global__ void cu_softmax_online_kernel(float *y, float *x, int len_q, int len_k) {
    const int tid = threadIdx.x;
    const int seq_idx = blockIdx.x;    // 当前处理的 query 序列位置 (0 to len_q-1)
    const int batch_idx = blockIdx.z;  // 当前处理的 batch (0 to batch-1)
    
    // 计算当前行的起始位置
    // 数据布局：[batch][seq_q][seq_k]
    int offset = batch_idx * len_q * len_k + seq_idx * len_k;
    
    // 每个 thread 处理该行的一个元素
    float exp_val = (tid < len_k) ? expf(x[offset + tid]) : 0.0f;
    
    // Block 内所有 threads 的 exp 值求和
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    
    // 归一化并写回
    if (tid < len_k) {
        y[offset + tid] = exp_val / exp_sum;
    }
}

__global__ void cu_rope_f32_kernel(float2 *d_out, float2 *d_src, int seq_len, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * N) return;
    
    float2 v = d_src[idx];

    int token_pos = idx / N;
    int token_idx = idx % N;

    float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2));
    float sin_v = sinf(token_pos * exp_v);
    float cos_v = cosf(token_pos * exp_v);

    float2 r;
    r.x = v.x * cos_v - v.y * sin_v;
    r.y = v.x * sin_v + v.y * cos_v;

    d_out[idx] = r;
}

void cu_rope(float *d_out, float *d_src, int seq_len, int N) {

    int complex_pairs = seq_len * N;
    dim3 threads(256);
    int grids = (complex_pairs + threads.x - 1)/threads.x;
    cu_rope_f32_kernel<<<grids, threads>>>(reinterpret_cast<float2*>(d_out), reinterpret_cast<float2*>(d_src), seq_len, N);
}

void cu_gemm_ABt_scale(int batch,
                        float *out,
                        const float *Q, // (batch, seq_len_q, d_k)
                        const float *K, // (batch, seq_len_k, d_k)
                        int seq_len_q,
                        int seq_len_k,
                        int d_k) {

    dim3 threads(32, 32);  // TILE_SIZE x TILE_SIZE
    dim3 grid((seq_len_k + threads.x - 1)/threads.x, (seq_len_q + threads.y - 1)/threads.y, batch);
    gemm_ABt_scale_kernel<32><<<grid, threads>>>(Q, K, out, seq_len_q, d_k, seq_len_k, (float)sqrt(d_k));
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

void cu_softmax_online(float *d_out, float *d_src, int len_q, int len_k, int batch) {
    dim3 grid(len_q, 1, batch);
    cu_softmax_online_kernel<<<grid, 1024>>>(d_out, d_src, len_q, len_k);
}

void cu_scaled_dot_product_attention(float *out,
                                        const float *Q, // (batch, seq_len_q, d_k)
                                        const float *K, // (batch, seq_len_k, d_k)
                                        const float *V, // (batch, seq_len_k, d_v)
                                        int seq_len_q,
                                        int seq_len_k,
                                        int d_k,
                                        int d_v,
                                        int batch) {

    thrust::device_vector<float> d_QKT(batch * seq_len_q * seq_len_k);
    thrust::device_vector<float> d_P(batch * seq_len_q * seq_len_k);
    thrust::device_vector<float> d_O(batch * seq_len_q * d_v);

    // step1: QK^T
    cu_gemm_ABt_scale(batch, d_QKT.data().get(), Q, K, seq_len_q, seq_len_k, d_k);

    // step2: softmax
    cu_softmax_online(d_P.data().get(), d_QKT.data().get(), seq_len_q, seq_len_k, batch);

    save_npy(d_P, batch, seq_len_q, seq_len_k, "../my_layers/npy_verify/tmp.npy");

    // step3: QK^T*V
    cu_gemm_AB(batch, d_O.data().get(), d_P.data().get(), V, seq_len_q, seq_len_k, d_v);

    cudaMemcpy(out, d_O.data().get(), sizeof(float) * batch * seq_len_q * d_v, cudaMemcpyDeviceToDevice);
}

void test_cu_gemm_ABt_scale() {
    int batch = 10;
    int M = 32;
    int K = 32;
    int N = 32;

    thrust::host_vector<float> h_A(batch*M*K);
    thrust::host_vector<float> h_B(batch*N*K);

    for (int i_b = 0; i_b < batch; ++i_b) {
        for (int i=0;i<M;++i) {
            for (int j=0;j<K;++j) {
                h_A[i_b*M*K + i*K+j] = i*K+j;
            }
        }
    
        for (int i=0;i<N;++i) {
            for (int j=0;j<K;++j) {
                h_B[i_b*K*N + i*K+j] = i*K+j;
            }
        }
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(batch*M*N);

    cu_gemm_ABt_scale(batch, d_C.data().get(), d_A.data().get(), d_B.data().get(), M, N, K);
    save_npy(d_C, batch, M, N, "../my_layers/npy_verify/cu_gemm_ABt_scale.npy");
}

void test_cu_gemm_AB() {
    int batch = 2;
    int M = 512;
    int K = 128;
    int N = 512;

    thrust::host_vector<float> h_A(batch*M*K);
    thrust::host_vector<float> h_B(batch*K*N);

    for (int i_b = 0; i_b < batch; ++i_b) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                h_A[i_b * M * K + i * K + j] = static_cast<float>(i * K + j + i_b);
            }
        }
    
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                h_B[i_b * K * N + i * N + j] = static_cast<float>(i * N + j + i_b);
            }
        }
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(batch*M*N);

    cu_gemm_AB(batch, d_C.data().get(), d_A.data().get(), d_B.data().get(), M, N, K);
    save_npy(d_C, batch, M, N, "../my_layers/npy_verify/cu_gemm_AB.npy");
}

void test_cu_rope() {
    int seq_len = 4;
    int N = 8;
    int total_complex = seq_len * N;

    thrust::host_vector<float> h_in(total_complex * 2);

    for (int i = 0; i < total_complex; ++i) {
        h_in[i * 2 + 0] = i;       // real
        h_in[i * 2 + 1] = -i;      // imag
    }

    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out(total_complex * 2);

    cu_rope(d_out.data().get(), d_in.data().get(), seq_len, N);

    save_npy(d_out, seq_len, N, 2, "../my_layers/npy_verify/cu_rope.npy");
}

void test_cu_softmax_online() {
    int batch = 10;
    int N = 1024;
    thrust::host_vector<float> x(batch*N);

    for (int b = 0; b < batch; ++b) {
        for (int i=0;i<N;++i) {
            x[b*N + i] = (float)i/100 + 0.3;
        }
    }

    thrust::device_vector<float> d_x = x;
    thrust::device_vector<float> d_out(batch*N);
    cu_softmax_online(d_out.data().get(), d_x.data().get(), 1, N, batch);

    save_npy(d_out, batch, N, 1, "../my_layers/npy_verify/cu_softmax_online.npy");
}

void test_cu_scaled_dot_product_attention() {
    int batch = 10;
    int seq_len_q = 32;
    int seq_len_k = 32;
    int d_k = 32;
    int d_v = 32;

    thrust::host_vector<float> h_Q(batch*seq_len_q*d_k);
    thrust::host_vector<float> h_K(batch*seq_len_k*d_k);
    thrust::host_vector<float> h_V(batch*seq_len_k*d_v);

    int M = seq_len_q;
    int K = d_k;
    int N = seq_len_k;

    for (int i_b = 0; i_b < batch; ++i_b) {
        for (int i=0;i<seq_len_q*d_k;++i) {
            h_Q[i_b*M*K + i] = (i % 100) * 0.01 - 0.5;
        }
    
        for (int i=0;i<seq_len_k*d_k;++i) {
            h_K[i_b*N*K + i] = (i % 100) * 0.02 - 1;
        }
    
        for (int i=0;i<seq_len_k*d_v;++i) {
            h_V[i_b*N*d_v + i] = (i % 100) * 0.01;
        }
    }

    thrust::device_vector<float> d_Q = h_Q;
    thrust::device_vector<float> d_K = h_K;
    thrust::device_vector<float> d_V = h_V;
    thrust::device_vector<float> d_out(batch*seq_len_q*d_v);

    cu_scaled_dot_product_attention(d_out.data().get(), d_Q.data().get(), d_K.data().get(), d_V.data().get(), seq_len_q, seq_len_k, d_k, d_v, batch);

    save_npy(d_out, batch, seq_len_q, d_v, "../my_layers/npy_verify/cu_scaled_dot_product_attention.npy");
}

int main() {
    test_cu_softmax_online();
    test_cu_gemm_ABt_scale();
    test_cu_gemm_AB();
    test_cu_scaled_dot_product_attention();
}