#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"

#define WARP_SIZE 32

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

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int NUM_THREADS = 256>
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

template <const int NUM_THREADS = 256>
__global__ void cu_softmax_online_kernel(float *y, float *x, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    int offset = blockIdx.z * N;

    float exp_val = (idx < N) ? expf(x[offset + idx]) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // block sum

    if (idx < N)
        y[offset + idx] = exp_val / exp_sum;
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
    gemm_ABt_scale_kernel<32><<<grid, threads>>>(Q, K, out, seq_len_q, d_k, seq_len_k, sqrt(d_k));
}

void cu_softmax_online(float *d_out, float *d_src, int N, int batch) {
    dim3 grid((N + 256 - 1)/256, 1, batch);
    cu_softmax_online_kernel<<<grid, 256>>>(d_out, d_src, N);
}

/*void cu_scaled_dot_product_attention(float *out,
                                        const float *Q, // (batch, seq_len_q, d_k)
                                        const float *K, // (batch, seq_len_k, d_k)
                                        const float *V, // (batch, seq_len_k, d_v)
                                        int seq_len_q,
                                        int seq_len_k,
                                        int d_k,
                                        int d_v,
                                        int batch) {

    thrust::device_vector<float> d_QKT(batch * seq_len_q * seq_len_k);

    // step1: QK^T
    cu_gemm_ABt_scale(batch, d_QKT.data().get(), Q, K, seq_len_q, seq_len_k, d_k);

    // step2: softmax
    cu_softmax_online(d_QKT.data().get(), );

    // step3: QK^T*V

}*/

void test_cu_gemm_ABt_scale() {
    int batch = 10;
    int M = 512;
    int K = 128;
    int N = 512;

    thrust::host_vector<float> h_A(batch*M*K);
    thrust::host_vector<float> h_B(batch*K*N);

    for (int i_b = 0; i_b < batch; ++i_b) {
        for (int i=0;i<M;++i) {
            for (int j=0;j<K;++j) {
                h_A[i_b*M*K + i*K+j] = i;
            }
        }
    
        for (int i=0;i<N;++i) {
            for (int j=0;j<K;++j) {
                h_B[i_b*K*N + i*K+j] = i;
            }
        }
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(batch*M*N);

    cu_gemm_ABt_scale(batch, d_C.data().get(), d_A.data().get(), d_B.data().get(), M, N, K);
    save_npy(d_C, batch*M*N, "../my_layers/npy_verify/cu_gemm_ABt_scale.npy");
}

void test_cu_softmax_online() {
    int batch = 10;
    int N = 256;
    thrust::host_vector<float> x(batch*N);

    for (int b = 0; b < batch; ++b) {
        for (int i=0;i<N;++i) {
            x[b*N + i] = (float)i/100 + 0.3;
        }
    }

    thrust::device_vector<float> d_x = x;
    thrust::device_vector<float> d_out(batch*N);
    cu_softmax_online(d_out.data().get(), d_x.data().get(), N, batch);

    save_npy(d_out, batch*N, "../my_layers/npy_verify/cu_softmax_online.npy");
}

/*void test_cu_scaled_dot_product_attention() {
    int batch = 10;
    int seq_len_q = 512;
    int seq_len_k = 512;
    int d_k = 128;
    int d_v = 256;

    thrust::host_vector<float> h_Q(batch*seq_len_q*d_k);
    thrust::host_vector<float> h_K(batch*seq_len_k*d_k);
    thrust::host_vector<float> h_V(batch*seq_len_k*d_v);

    for (int i_b = 0; i_b < batch; ++i_b) {
        for (int i=0;i<seq_len_q*d_k;++i) {
            h_Q[i_b*M*K + i] = i * 0.2 + 3.5;
        }
    
        for (int i=0;i<seq_len_k*d_k;++i) {
            h_K[i_b*N*K + i] = i * 0.3 + 1.2;
        }
    
        for (int i=0;i<seq_len_k*d_v;++i) {
            h_V[i_b*N*d_v + i] = i * 0.7 + 6.2;
        }
    }

    thrust::device_vector<float> d_Q = h_Q;
    thrust::device_vector<float> d_K = h_K;
    thrust::device_vector<float> d_V = h_V;
    thrust::device_vector<float> d_out(seq_len_q*seq_len_k);

    cu_scaled_dot_product_attention(d_out.data().get(), d_Q.data().get(), d_K.data().get(), d_V.data().get(), seq_len_q, seq_len_k, d_k, d_v);

    save_npy(d_out, seq_len_q*seq_len_k, "../my_layers/npy_verify/cu_scaled_dot_product_attention.npy");
}*/

int main() {
    //test_cu_scaled_dot_product_attention();
    test_cu_softmax_online();
    test_cu_gemm_ABt_scale();
}