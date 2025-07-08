#include "GEMM_cutlass_splitK.cuh"
#include "cutlass_common.cuh"

void CutlassGEMMSplitk::init() {
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

void CutlassGEMMSplitk::load_data_from_host(const float* a_host, const float* b_host) {
    // copy a_host (size M x K) into A
    std::memcpy(A.host_data(), a_host, sizeof(float) * M * K);
    // copy b_host (size K x N) into B
    std::memcpy(B.host_data(), b_host, sizeof(float) * K * N);

    A.sync_device();
    B.sync_device();
}

bool CutlassGEMMSplitk::correctness_check() {
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

float CutlassGEMMSplitk::benchmark(int iterations) {
    // warmup
    CUTLASS_CHECK(gemm_op(args, workspace.get()));
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iterations; ++i) {
        CUTLASS_CHECK(gemm_op(args, workspace.get()));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_ms / iterations;
}

int main() {
    std::vector<int> sizes = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};

    for (int size : sizes) {
        CutlassGEMMSplitk gemm(size, size, size, 1);
        gemm.init();

        int M = size;
        int N = size;
        int K = size;
        float elapsed_ms = gemm.benchmark(10);

        double gflops = 2.0 * M * N * K / (elapsed_ms * 1e-3) / 1024 / 1024 / 1024;
        if (gemm.correctness_check()) {
            printf("size: %d, elapsed_ms: %f, GFLOPS: %f\n", size, elapsed_ms, gflops);
        }
    }

    return 0;
}