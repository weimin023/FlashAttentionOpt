#include <torch/extension.h>
#include "GEMM_cutlass_splitK.cuh"
#include <cutlass/half.h>

/*void run_gemm_cutlass_splitK(int M, int N, int K,
                            cutlass::half_t* A_device, cutlass::half_t* B_device, float* D_device) {

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    constexpr int split_k_slices = 1;

    cutlass::TensorRef<cutlass::half_t, LayoutInputA> A_ref(A_device, LayoutInputA::packed({M, K}).stride(0));
    cutlass::TensorRef<cutlass::half_t, LayoutInputB> B_ref(B_device, LayoutInputB::packed({K, N}).stride(0));
    cutlass::TensorRef<float, LayoutOutput> C_ref(D_device, LayoutOutput::packed({M, N}).stride(0));
    cutlass::TensorRef<float, LayoutOutput> D_ref(D_device, LayoutOutput::packed({M, N}).stride(0));

    typename Gemm::Arguments args(
        problem_size,  // problem size
        A_ref,
        B_ref,
        C_ref,
        D_ref,
        {1.0f, 0.0f},
        split_k_slices);

    size_t workspace_size = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    Gemm gemm_op;

    CUTLASS_CHECK(gemm_op.initialize(args, workspace.get()));
    CUTLASS_CHECK(gemm_op());
}

torch::Tensor cutlass_gemm_splitK(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A.shape[1] != B.shape[0]");

    auto D = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    run_gemm_cutlass_splitK(
        M, N, K,
        reinterpret_cast<cutlass::half_t*>(A.data_ptr<at::Half>()),
        reinterpret_cast<cutlass::half_t*>(B.data_ptr<at::Half>()),
        D.data_ptr<float>()
    );

    return D;
}

PYBIND11_MODULE(my_cutlass, m) {
    m.def("gemm_splitK", &cutlass_gemm_splitK, "CUTLASS GEMM SplitK (half x half -> float)");
}*/
