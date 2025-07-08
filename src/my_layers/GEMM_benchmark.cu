#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"
#include <float.h>

#define OFFSET(r, c, ld) ((r) * (ld) + (c))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, cublasHandle_t handle = nullptr);
float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat, cublasHandle_t handle = nullptr);

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

void sgemm_cublas(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K, cublasHandle_t handle) {

    // a: [M, K] row-major
    // b: [K, N] row-major
    // c: [M, N] row-major
    // Compute: c = a * b

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b, N,
        a, K,
        &beta,
        c, N
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM failed! Error code = %d\n", stat);
    }

}

__global__ void sgemm_naive(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += a[OFFSET(row, k, K)] * b[OFFSET(k, col, N)];
            }
            c[OFFSET(row, col, N)] = acc;
        }
}

template<int BLOCK_SIZE>
__global__ void sgemm_naive_coalescing(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

        int br = blockIdx.y;
        int bc = blockIdx.x;
        int tr = threadIdx.y;
        int tc = threadIdx.x;

        int row = br * BLOCK_SIZE + tr;
        int col = bc * BLOCK_SIZE + tc;

        if (row < M && col < N) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += a[OFFSET(row, k, K)] * b[OFFSET(k, col, N)];
            }
            c[OFFSET(row, col, N)] = acc;
        }
}

template<int BM, int BN, int BK>
__global__ void sgemm_2D_tile(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    int br = blockIdx.y;
    int bc = blockIdx.x;
    int tr = threadIdx.y;
    int tc = threadIdx.x;

    int row = br * BM + tr;
    int col = bc * BN + tc;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float acc = 0;
    for (int bk = 0; bk < (K + BK - 1)/BK; ++bk) {
        int a_row = br * BM + tr;
        int a_col = bk * BK + tc;
        if (a_row < M && a_col < K) {
            s_a[tr][tc] = a[OFFSET(a_row, a_col, K)];
        } else {
            s_a[tr][tc] = 0;
        }

        int b_row = bk * BK + tr;
        int b_col = bc * BN + tc;
        if (b_row < K && b_col < N) {
            s_b[tr][tc] = b[OFFSET(b_row, b_col, N)];
        } else {
            s_b[tr][tc] = 0;
        }
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            acc += s_a[tr][k] * s_b[k][tc];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        c[OFFSET(row, col, N)] = acc;
    }
}

template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_register_tile(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    int br = blockIdx.y;
    int bc = blockIdx.x;
    
    int block_row_thread = BM / TM;
    int block_col_thread = BN / TN;
    int thread_num = block_row_thread * block_col_thread;

    int tr = (threadIdx.x / block_row_thread) * TM;
    int tc = (threadIdx.x % block_row_thread) * TN;

    __shared__ float s_a[BM * BK];
    __shared__ float s_b[BK * BN];

    a = &a[br * BM * K];
    b = &b[bc * BN];
    c = &c[br * BM * N + bc * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float tmp[TM][TN] = {0};
    float reg_a[TM] = {0};
    float reg_b[TN] = {0};

    #pragma unroll
    for (int bk = 0; bk < K; bk += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            s_a[OFFSET(a_tile_row + i, a_tile_col, BK)] = a[OFFSET(a_tile_row + i, a_tile_col, K)];
        }

        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            s_b[OFFSET(b_tile_row + i, b_tile_col, BN)] = b[OFFSET(b_tile_row + i, b_tile_col, N)];
        }
        __syncthreads();

        a += BK;
        b += BK * N;

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                reg_a[j] = s_a[OFFSET(tr + j, i, BK)];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = s_b[OFFSET(i, tc + j, BN)];
            }
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                #pragma unroll
                for (int k = 0; k < TN; ++k) {
                    tmp[j][k] += reg_a[j] * reg_b[k];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        for (int k = 0; k < TN; ++k) {
            c[OFFSET(tr + j, tc + k, N)] = tmp[j][k];
        }
    }
    
}

template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_2D_tile_vectorized(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
    int br = blockIdx.y;
    int bc = blockIdx.x;
    
    const int block_row_thread = BM / TN;
    const int block_col_thread = BN / TM;
    const int thread_num = block_row_thread * block_col_thread;

    int tr = (threadIdx.x / block_row_thread) * TM;
    int tc = (threadIdx.x % block_row_thread) * TN;

    __shared__ float s_a[BK * BM];
    __shared__ float s_b[BK * BN];

    float acc[TM][TN] = {0};
    float frag_a[TM] = {0};
    float frag_b[TN] = {0};

    a = &a[br * BM * K];
    b = &b[bc * BN];
    c = &c[br * BM * N + bc * BN];

    const int ldg_a_num = BM * BK / thread_num / 4;
    const int ldg_b_num = BN * BK / thread_num / 4;

    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num; 
    
    float ldg_a_reg[4 * ldg_a_num] = {0};
    #pragma unroll
    for (int bk = 0; bk < K; bk += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;
            FLOAT4(ldg_a_reg[ldg_index]) = FLOAT4(a[OFFSET(a_tile_row + i, a_tile_col, K)]);
            s_a[OFFSET(a_tile_col, a_tile_row + i, BM)] = ldg_a_reg[ldg_index];
            s_a[OFFSET(a_tile_col + 1, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 1];
            s_a[OFFSET(a_tile_col + 2, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 2];
            s_a[OFFSET(a_tile_col + 3, a_tile_row + i, BM)] = ldg_a_reg[ldg_index + 3];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FLOAT4(s_b[OFFSET(b_tile_row + i, b_tile_col, BN)]) = FLOAT4(b[OFFSET(b_tile_row + i, b_tile_col, N)]);
        }
        __syncthreads();

        a += BK;
        b += BK * N;

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int j = 0; j < TM; j += 4) {
                FLOAT4(frag_a[j]) = FLOAT4(s_a[OFFSET(i, tr + j, BM)]);
            }
            #pragma unroll
            for (int j = 0; j < TN; j += 4) {
                FLOAT4(frag_b[j]) = FLOAT4(s_b[OFFSET(i, tc + j, BN)]);
            }
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                #pragma unroll
                for (int k = 0; k < TN; ++k) {
                    acc[j][k] += frag_a[j] * frag_b[k];
                }
            }
        }
        __syncthreads();
    }
    for (int m = 0; m < TM; ++m) {
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FLOAT4(c[OFFSET(tr + m, tc + n, N)]);
            ctmp.x = acc[m][n];
            ctmp.y = acc[m][n + 1];
            ctmp.z = acc[m][n + 2];
            ctmp.w = acc[m][n + 3];
            FLOAT4(c[OFFSET(tr + m, tc + n, N)]) = ctmp;
        }
    }
}

__global__ void sgemm_V1(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

__global__ void sgemm_V2(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    int br = blockIdx.y;
    int bc = blockIdx.x;
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    int tid = tr * blockDim.x + tc;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = br * BM + load_a_smem_m;
    int load_b_gmem_n = bc * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1)/BK; ++bk) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BK; ++tk) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][tr * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][tr * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tc * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tc * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int store_c_gmem_m = br * BM + tr * TM / 2 + i;
        int store_c_gmem_n = bc * BN + tc * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int store_c_gmem_m = br * BM + BM / 2 + tr * TM / 2 + i;
        int store_c_gmem_n = bc * BN + tc * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

int main(void) {

    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("\nKernal = sgemm_2D_tile_vectorized\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = sgemm_2D_tile_vectorized<BM, BN, 32, TM, TN>/*, sgemm_naive_coalescing<32>*/;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim((BM/TM) * (BN/TN));
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    
    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim((BM/TM) * (BN/TN));
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }

    cublasDestroy(handle);
    return 0;
}


float testError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, cublasHandle_t handle) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    // gpuSgemm(d_a, d_b, d_c, M, N, K, handle);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat, cublasHandle_t handle) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        // gpuSgemm(d_a, d_b, d_c, M, N, K, handle);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}