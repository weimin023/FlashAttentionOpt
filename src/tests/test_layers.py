import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from my_layers.RMSNorm import RMSNorm

sys.path.append("../")
import cuFMHA
                
def test_rmsnorm():
    N = 128*128
    eps = 1e-5
    x = torch.ones(N)
    for i in range(N):
        x[i] = i

    # Instantiate both modules
    custom_rms = RMSNorm(N, eps=eps)
    torch_rms = nn.RMSNorm(N, eps=eps)

    # Copy the weights to match exactly
    with torch.no_grad():
        torch_rms.weight.copy_(custom_rms.w)

    # Get outputs
    out_custom = custom_rms(x)
    out_torch = torch_rms(x)

    # npy from CUDA implementation
    #out_cuda_warp = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/rmsnorm_warp_opt.npy")
    #out_cuda_warp = torch.from_numpy(out_cuda_warp)

    # Compare outputs
    #torch.testing.assert_close(out_custom, out_torch, rtol=1e-5, atol=1e-6)
    #torch.testing.assert_close(out_cuda_warp, out_torch, rtol=1e-5, atol=1e-6)

    raise NotImplementedError

def test_cu_gemm_cutlass_batch():
    batch = 2
    M = N = K = 32

    a = torch.arange(batch*M*K).reshape(batch, M, K).float()
    b = torch.arange(batch*N*K).reshape(batch, M, K).float()

    c = torch.matmul(a, b)

    out_cutlass = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/test_cu_gemm_cutlass_batch.npy")
    out_cutlass = torch.from_numpy(out_cutlass)

    # Compare outputs
    torch.testing.assert_close(out_cutlass, c, rtol=1e-5, atol=1e-6)

def test_gemm_ABt_scale():
    batch = 2
    M = 128
    K = 128
    N = 128

    a = torch.zeros(batch, M, K)
    b = torch.zeros(batch, N, K)

    # Create batched tensors
    for i_b in range(batch):
        for i in range(M):
            for j in range(K):
                a[i_b, i, j] = i * K + j

        for i in range(N):
            for j in range(K):
                b[i_b, i, j] = i * K + j

    torch_out = torch.matmul(a, b.transpose(-1, -2)) / np.sqrt(K)

    cuda_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_gemm_ABt_scale.npy")
    cuda_out = torch.from_numpy(cuda_out).reshape((batch, M, N))

    torch.testing.assert_close(torch_out, cuda_out, rtol=1e-5, atol=1e-6)
    
def test_gemm_AB():
    batch = 2
    M = 128
    K = 128
    N = 128

    a = torch.zeros(batch, M, K)
    b = torch.zeros(batch, K, N)

    # Create batched tensors
    for i_b in range(batch):
        for i in range(M):
            for j in range(K):
                a[i_b, i, j] = i * K + j + i_b

        for i in range(K):
            for j in range(N):
                b[i_b, i, j] = i * N + j + i_b

    torch_out = torch.matmul(a, b)

    cuda_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_gemm_AB.npy")
    cuda_out = torch.from_numpy(cuda_out).reshape((batch, M, N))

    torch.testing.assert_close(torch_out, cuda_out, rtol=1e-5, atol=1e-6)

def test_online_softmax():
    batch = 10
    N = 1024

    x = torch.arange(N, dtype=torch.float32) / 100 + 0.3    # shape: (N,)
    x = x.expand(batch, -1).clone()                         # shape: (batch, N)

    torch_softmax = nn.Softmax(dim=1)
    torch_out = torch_softmax(x)

    cuda_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_softmax_online.npy")
    cuda_out = torch.from_numpy(cuda_out).reshape((batch, N))

    torch.testing.assert_close(cuda_out, torch_out, rtol=1e-5, atol=1e-6)

def test_scaled_dot_product_attention():
    
    batch = 8
    seq_len_q = 128
    seq_len_k = 2048
    d_k = 128
    d_v = 128

    # shape: (batch, seq_len, dim)
    h_Q = torch.zeros((batch, seq_len_q, d_k), dtype=torch.float32)
    h_K = torch.zeros((batch, seq_len_k, d_k), dtype=torch.float32)
    h_V = torch.zeros((batch, seq_len_k, d_v), dtype=torch.float32)

    for i_b in range(batch):
        for i in range(seq_len_q):
            for j in range(d_k):
                h_Q[i_b, i, j] = ((i * d_k + j) % 100) * 0.01 - 0.5  # roughly in [-0.5, 0.5]

        for i in range(seq_len_k):
            for j in range(d_k):
                h_K[i_b, i, j] = ((i * d_k + j) % 100) * 0.02 - 1.0  # roughly in [-1.0, 1.0]

        for i in range(seq_len_k):
            for j in range(d_v):
                h_V[i_b, i, j] = ((i * d_v + j) % 100) * 0.01  # roughly in [0, 1.0]

    # PyTorch S-DPA (non-causal)
    torch_func_out = F.scaled_dot_product_attention(h_Q, h_K, h_V, attn_mask=None, is_causal=False)

    # load CUDA kernel output
    cuda_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_scaled_dot_product_attention.npy")
    cuda_out = torch.from_numpy(cuda_out).reshape((batch, seq_len_q, d_v))

    '''tmp = torch.matmul(h_Q, h_K.transpose(1, 2))/np.sqrt(d_k)
    torch_softmax = nn.Softmax(dim=2)
    torch_func_out = torch_softmax(tmp)
    torch_func_out = torch.matmul(torch_func_out, h_V)'''

    cutlass_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_scaled_dot_product_attention_cutlass_batched.npy")
    cutlass_out = torch.from_numpy(cutlass_out).reshape((batch, seq_len_q, d_v))

    # Compare outputs
    torch.testing.assert_close(cutlass_out, cuda_out, rtol=1e-5, atol=1e-6)
    #torch.testing.assert_close(cutlass_out, torch_func_out, rtol=1e-5, atol=1e-6)

def test_cu_fmha_kernel_multihead():
    import math

    B, H, S_q, S_k, D_head = 10, 5, 32, 32, 16

    # Q[b,h,q,d] = b + h + 1 + 0.01*d
    Q = torch.zeros(B, H, S_q, D_head)
    for b in range(B):
        for h in range(H):
            for d in range(D_head):
                Q[b, h, :, d] = b + h + 1.0 + 0.01 * d

    # K[b,h,k,d] = sin(k + d + b + h)
    K = torch.zeros(B, H, S_k, D_head)
    for b in range(B):
        for h in range(H):
            for k in range(S_k):
                for d in range(D_head):
                    K[b, h, k, d] = torch.sin(torch.tensor(k + d + b + h, dtype=torch.float32))

    # V[b,h,k,d] = (b+1)*(h+1)*(k%d_head == d)
    V = torch.zeros(B, H, S_k, D_head)
    for b in range(B):
        for h in range(H):
            for k in range(S_k):
                d = k % D_head
                V[b, h, k, d] = (b + 1) * (h + 1)

    # F.scaled_dot_product_attention 需要 [B*H, S, D]
    Q_ = Q.reshape(B * H, S_q, D_head)
    K_ = K.reshape(B * H, S_k, D_head)
    V_ = V.reshape(B * H, S_k, D_head)

    O = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)

    # 轉回 [B, H, S_q, D_head]
    O = O.reshape(B, H, S_q, D_head)

    cuda_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_fmha_kernel_multihead.npy")
    cuda_out = torch.from_numpy(cuda_out).reshape((B, H, S_q, D_head))

    torch.testing.assert_close(O, cuda_out, rtol=1e-5, atol=1e-6)

def test_gemm_pybind():
    # 模擬輸入參數
    batch = 2
    seq_len_q = 512
    seq_len_k = 128
    d_k = 32

    torch.manual_seed(0)

    # Q shape: [batch, seq_len_q, d_k]
    Q = torch.randn(batch, seq_len_q, d_k, device='cuda', dtype=torch.float32)
    # K shape: [batch, d_k, seq_len_k]
    K = torch.randn(batch, d_k, seq_len_k, device='cuda', dtype=torch.float32)

    # 你的 CUDA gemm_AB 函數計算結果
    out_cuda = cuFMHA.gemm_test(Q, K)  # shape: [batch, seq_len_q, seq_len_k]

    # PyTorch 參考計算
    out_torch = torch.bmm(Q, K)

    torch.testing.assert_close(out_cuda, out_torch, rtol=1e-5, atol=1e-6)