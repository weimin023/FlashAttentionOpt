import torch
import torch.nn as nn
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
    M = 512
    K = 128
    N = 512

    # Create batched tensors
    a = torch.arange(M, dtype=torch.float32).view(1, M, 1).expand(batch, M, K)
    b = torch.arange(N, dtype=torch.float32).view(1, N, 1).expand(batch, N, K)

    torch_out = torch.matmul(a, b.transpose(-1, -2)) / np.sqrt(K)

    cuda_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_gemm_ABt_scale.npy")
    cuda_out = torch.from_numpy(cuda_out).reshape((batch, M, N))

    torch.testing.assert_close(torch_out, cuda_out, rtol=1e-5, atol=1e-6)
    

def test_online_softmax():
    batch = 10
    N = 256

    x = torch.arange(N, dtype=torch.float32) / 100 + 0.3    # shape: (N,)
    x = x.expand(batch, -1).clone()                         # shape: (batch, N)

    torch_softmax = nn.Softmax(dim=1)
    torch_out = torch_softmax(x)

    cuda_out = np.load("/home/weimin.chen/Desktop/FlashAttentionOpt/src/my_layers/npy_verify/cu_softmax_online.npy")
    cuda_out = torch.from_numpy(cuda_out).reshape((batch, N))

    torch.testing.assert_close(cuda_out, torch_out, rtol=1e-5, atol=1e-6)

def test_scaled_dot_product_attention():
    
    batch = 1
    seq_len_q = 512
    seq_len_k = 512
    d_k = 128
    d_v = 256

    # shape: (batch, seq_len, dim)
    h_Q = torch.zeros((batch, seq_len_q, d_k), dtype=torch.float32)
    h_K = torch.zeros((batch, seq_len_k, d_k), dtype=torch.float32)
    h_V = torch.zeros((batch, seq_len_k, d_v), dtype=torch.float32)

    for i in range(seq_len_q):
        for j in range(d_k):
            h_Q[0, i, j] = (i * d_k + j) * 0.2 + 3.5

    for i in range(seq_len_k):
        for j in range(d_k):
            h_K[0, i, j] = (i * d_k + j) * 0.3 + 1.2

    for i in range(seq_len_k):
        for j in range(d_v):
            h_V[0, i, j] = (i * d_v + j) * 0.7 + 6.2

    torch_out = nn.functional.scaled_dot_product_attention(h_Q, h_K, h_V)
    print(torch_out.shape)
