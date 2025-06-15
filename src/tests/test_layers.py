import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from my_layers.RMSNorm import RMSNorm

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

def test_gemm_ABt_scale():
    batch = 10
    M = 32
    K = 32
    N = 32

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
    M = 512
    K = 128
    N = 512

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
    
    batch = 10
    seq_len_q = 32
    seq_len_k = 32
    d_k = 32
    d_v = 32

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
    torch_out = F.scaled_dot_product_attention(h_Q, h_K, h_V, attn_mask=None, is_causal=False)

    # load CUDA kernel output
    cuda_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_scaled_dot_product_attention.npy")
    cuda_out = torch.from_numpy(cuda_out).reshape((batch, seq_len_q, d_v))

    # Compare outputs
    torch.testing.assert_close(cuda_out, torch_out, rtol=1e-5, atol=1e-6)
