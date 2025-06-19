#include "GEMM_cutlass_splitK.cuh"

void CutlassGEMMWrapper::init() {
    cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 1, ElementInputA(4), ElementInputA(-4), 0);
    cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 1, ElementInputB(4), ElementInputB(-4), 0);
    cutlass::reference::host::TensorFill(C.host_view());

    A.sync_device();
    B.sync_device();
    C.sync_device();
    D.sync_device();
    D_ref.sync_device();

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    args = typename Gemm::Arguments(
        problem_size,
        A.device_ref(),
        B.device_ref(),
        C.device_ref(),
        D.device_ref(),
        {1.0f, 0.0f},
        split_k_slices
    );

    size_t workspace_size = Gemm::get_workspace_size(args);
    workspace.reset(workspace_size);

    auto status = gemm_op.initialize(args, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS initialize failed");
    }
}

bool CutlassGEMMWrapper::correctness_check() {
    Gemm_ref gemm_ref;
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    CUTLASS_CHECK(gemm_op(args, workspace.get()));

    // REF Value
    gemm_ref(problem_size,
             1,
             A.device_ref(),
             B.device_ref(),
             0,
             C.device_ref(),
             D_ref.device_ref());

    cudaDeviceSynchronize();

    D.sync_host();
    D_ref.sync_host();

    bool passed = cutlass::reference::host::TensorRelativelyEquals(D.host_view(), D_ref.host_view(), 1e-3f, 1e-2f);

    return passed;
}

float CutlassGEMMWrapper::benchmark(int iterations) {
    // warmup
    gemm_op();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iterations; ++i) {
        gemm_op();
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
        CutlassGEMMWrapper gemm(size, size, size, 1);
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