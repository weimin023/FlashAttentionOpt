#include <torch/extension.h>

void cu_flash_attn_multihead_v0(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int S_q, int S_k, int D
);

void cu_gemm_AB(int batch,
    float *out,
    const float *Q, // (batch, seq_len_q, d_k)
    const float *K, // (batch, d_k, seq_len_k)
    int seq_len_q,
    int seq_len_k,
    int d_k);

torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "All inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");

    int B = Q.size(0);
    int H = Q.size(1);
    int S_q = Q.size(2);
    int D = Q.size(3);
    int S_k = K.size(2);

    auto O = torch::empty({B, H, S_q, D}, Q.options());

    cu_flash_attn_multihead_v0(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, S_q, S_k, D
    );

    return O;
}

torch::Tensor gemm_AB(torch::Tensor Q, torch::Tensor K) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "Only float32 supported");

    Q = Q.contiguous();
    K = K.contiguous();
    
    int batch = Q.size(0);
    int seq_len_q = Q.size(1);
    int d_k = Q.size(2);
    int seq_len_k = K.size(2);

    auto O = torch::empty({batch, seq_len_q, seq_len_k}, Q.options());

    // 呼叫你的 cu_gemm_AB kernel wrapper
    cu_gemm_AB(batch,
               O.data_ptr<float>(),
               Q.data_ptr<float>(),
               K.data_ptr<float>(),
               seq_len_q,
               seq_len_k,
               d_k);

    return O;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "Flash Attention Forward (CUDA)");
    m.def("gemm_test", &gemm_AB, "gemm test");
}