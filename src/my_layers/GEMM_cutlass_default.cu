#include "GEMM_cutlass_default.cuh"

void CutlassGEMMDefault::init() {
    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 1, ElementInputA(4), ElementInputA(-4), 0);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 1, ElementInputB(4), ElementInputB(-4), 0);
    cutlass::reference::host::TensorFill(out.host_view(), ElementOutput(0));
    cutlass::reference::host::TensorFill(out_ref.host_view(), ElementOutput(0));

    A.sync_device();
    B.sync_device();
    out.sync_device();
    out_ref.sync_device();

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    args = typename Gemm::Arguments(
        problem_size,
        A.device_ref(),
        B.device_ref(),
        out.device_ref(),
        out.device_ref(),
        {1.0f, 0.0f}
    );

    CUTLASS_CHECK(gemm_op.initialize(args));
}

bool CutlassGEMMDefault::correctness_check() {
    Gemm_ref gemm_ref;
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    CUTLASS_CHECK(gemm_op(args));

    // REF Value
    gemm_ref(problem_size,
        1,
        A.device_ref(),
        B.device_ref(),
        0,
        out.device_ref(),
        out_ref.device_ref());

    cudaDeviceSynchronize();

    out.sync_host();
    out_ref.sync_host();

    bool passed = cutlass::reference::host::TensorRelativelyEquals(out.host_view(), out_ref.host_view(), 1e-3f, 1e-2f);

    return passed;
}

float CutlassGEMMDefault::benchmark(int iterations) {
    // warmup
    CUTLASS_CHECK(gemm_op(args, nullptr));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iterations; ++i) {
        CUTLASS_CHECK(gemm_op(args, nullptr));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_ms / iterations;
}

int main() {
    std::vector<int> sizes = {16, 32, 64, 128, 256, 512, 1024};

    for (int size : sizes) {
        CutlassGEMMDefault gemm(size, size, size);
        gemm.init();

        int M = size;
        int N = size;
        int K = size;
        float elapsed_ms = gemm.benchmark(10);

        double tflops = 2.0 * M * N * K / (elapsed_ms * 1e6);
        if (gemm.correctness_check()) {
            printf("size: %d, TFLOPS: %f\n", size, tflops);
        }
    }

    return 0;
}