#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"

#include "cutlass_common.cuh"
#include <cutlass/gemm/device/gemm_batched.h>

void cu_gemm_cutlass_batch(int batch, float *C, const float *A, const float *B, int M, int K, int N) {
    using Element = float;
    using Layout = cutlass::layout::RowMajor;

    using GemmBatched_AB = cutlass::gemm::device::GemmBatched<
        Element, Layout,
        Element, Layout,
        Element, Layout,
        Element
    >;

    GemmBatched_AB::Arguments args(
        {M, N, K},                         // M, N, K
        {A, K},                           // A (Q)
        M*K,
        {B, N},                           // B (K)
        K*N,
        {C, N}, // C
        M*N,
        {C, N}, // D
        M*N,
        {1.0f, 0.0f},                                        // alpha, beta
        batch                                                // batch count
    );

    GemmBatched_AB gemm;
    CUTLASS_CHECK(gemm(args));
}

void test_cu_gemm_cutlass_batch() {
    
    int M = 32;
    int K = 32;
    int N = 32;
    int batch = 2;

    thrust::host_vector<float> h_A(batch*M*K);
    thrust::host_vector<float> h_B(batch*K*N);

    for (int i=0;i<batch*M*K;++i) {
        h_A[i] = i;
        h_B[i] = i;
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(batch*M*N);

    cu_gemm_cutlass_batch(batch, d_C.data().get(), d_A.data().get(), d_B.data().get(), M, K, N);

    save_npy(d_C, batch, M, N, "../my_layers/npy_verify/test_cu_gemm_cutlass_batch.npy");
}

int main() {
    test_cu_gemm_cutlass_batch();
}