#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"

#define WARP_SIZE 32

template<int TILE_SIZE> __global__ void gemm_ABt_scale_kernel(const float *dA, const float *dB, float *dC, int M, int K, int N) {

    int c = threadIdx.x;
    int r = threadIdx.y;
    
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float SA[TILE_SIZE][TILE_SIZE];
    __shared__ float SB[TILE_SIZE][TILE_SIZE];

    float reg_tile = 0;
    for (int t = 0; t < K; t += TILE_SIZE) {

        if (row < M && (t + c) < K) {
            SA[r][c] = dA[row * K + (t + c)];
        } else {
            SA[r][c] = 0;
        }

        // Load B with transposed access
        // Original: SB[r][c] = dB[(t + r) * N + col];
        // Transposed: Bᵗ[col][t + r] == B[t + r][col]
        if (col < N && (t + r) < K) {
            SB[r][c] = dB[col * K + (t + r)];  // Notice the change
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
        dC[row * N + col] = reg_tile / sqrtf(K);
    }
    
}

template <int NUM_THREADS = 256>
__device__ float warp_reduce_sum(float val) {
    for (int mask = WARP_SIZE/2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <int NUM_THREADS = 256>
__device__ float block_reduce_sum(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    static __shared__ float shm[NUM_WARPS];

    float value = warp_reduce_sum<NUM_THREADS>(val);
    if (laneId == 0) shm[warpId] = value;
    __syncthreads();

    value = (laneId < NUM_WARPS)? shm[laneId]:0;
    value = warp_reduce_sum<NUM_THREADS>(value);

    value = __shfl_sync(0xffffffff, value, 0);
    return value;

}

template <int NUM_THREADS = 256>
__global__ void cu_softmax_online_kernel(float *d_out, const float *d_src, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float exp_val = (idx < N)? expf(d_src[idx]):0.0f;
    float exp_sum = block_reduce_sum<NUM_THREADS>(exp_val);

    if (idx < N) {
        d_out[idx] = exp_val/exp_sum;
    }
}

void cu_gemm_ABt_scale(float *out,
                        const float *Q, // (batch, seq_len_q, d_k)
                        const float *K, // (batch, seq_len_k, d_k)
                        int seq_len_q,
                        int seq_len_k,
                        int d_k) {

    dim3 threads(32, 32);  // TILE_SIZE x TILE_SIZE
    dim3 grid((seq_len_k + threads.x - 1)/threads.x, (seq_len_q + threads.y - 1)/threads.y);
    gemm_ABt_scale_kernel<32><<<grid, threads>>>(Q, K, out, seq_len_q, d_k, seq_len_k);
}

void cu_softmax_online(float *d_out, const float *d_src, int N) {
    dim3 grid(N/256);
    cu_softmax_online_kernel<256><<<grid, 256>>>(d_out, d_src, N);
}

void cu_scaled_dot_product_attention(float *out,
                                        const float *Q, // (batch, seq_len_q, d_k)
                                        const float *K, // (batch, seq_len_k, d_k)
                                        const float *V, // (batch, seq_len_k, d_v)
                                        int seq_len_q,
                                        int seq_len_k,
                                        int d_k,
                                        int d_v) {

    // step1: QK^T
    // cu_gemm_ABt_scale(out, Q, K, V, seq_len_q, seq_len_k, d_k, d_v);

    // step2: softmax

    // step3: QK^T*V

}

void test_cu_gemm_ABt_scale() {
    int M = 512;
    int K = 128;
    int N = 512;

    thrust::host_vector<float> h_A(M*K);
    thrust::host_vector<float> h_B(K*N);

    for (int i=0;i<M;++i) {
        for (int j=0;j<K;++j) {
            h_A[i*K+j] = i;
        }
    }

    for (int i=0;i<N;++i) {
        for (int j=0;j<K;++j) {
            h_B[i*K+j] = i;
        }
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(M*N);

    cu_gemm_ABt_scale(d_C.data().get(), d_A.data().get(), d_B.data().get(), M, N, K);
    save_npy(d_C, M*N, "../my_layers/npy_verify/cu_gemm_ABt_scale.npy");
}

void test_cu_softmax_online() {
    int N = 1024;
    thrust::host_vector<float> x(N);

    for (int i=0;i<N;++i) {
        x[i] = i * 2.2 + 0.3;
    }

    thrust::device_vector<float> d_x = x;
    thrust::device_vector<float> d_out(N);
    cu_softmax_online(d_out.data().get(), d_x.data().get(), N);

    save_npy(d_out, N, "../my_layers/npy_verify/cu_softmax_online.npy");
}

void test_cu_scaled_dot_product_attention() {
    int batch = 1;
    int seq_len_q = 512;
    int seq_len_k = 512;
    int d_k = 128;
    int d_v = 256;

    thrust::host_vector<float> h_Q(seq_len_q*d_k);
    thrust::host_vector<float> h_K(seq_len_k*d_k);
    thrust::host_vector<float> h_V(seq_len_k*d_v);

    for (int i=0;i<seq_len_q*d_k;++i) {
        h_Q[i] = i * 0.2 + 3.5;
    }

    for (int i=0;i<seq_len_k*d_k;++i) {
        h_K[i] = i * 0.3 + 1.2;
    }

    for (int i=0;i<seq_len_k*d_v;++i) {
        h_V[i] = i * 0.7 + 6.2;
    }

    thrust::device_vector<float> d_Q = h_Q;
    thrust::device_vector<float> d_K = h_K;
    thrust::device_vector<float> d_V = h_V;
    thrust::device_vector<float> d_out(seq_len_q*seq_len_k);

    cu_scaled_dot_product_attention(d_out.data().get(), d_Q.data().get(), d_K.data().get(), d_V.data().get(), seq_len_q, seq_len_k, d_k, d_v);

    save_npy(d_out, seq_len_q*seq_len_k, "../my_layers/npy_verify/cu_scaled_dot_product_attention.npy");
}

int main() {
    test_cu_scaled_dot_product_attention();
    test_cu_softmax_online();
    test_cu_gemm_ABt_scale();
}