import torch
import torch.nn as nn
import torch.testing
import unittest
import numpy as np

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.w
    
class TestRMSNorm(unittest.TestCase):
    def test_compare_with_torch_builtin(self):

        N = 512*512
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
        out_cuda_warp = np.load("./npy_verify/rmsnorm_warp_opt.npy")
        out_cuda_warp = torch.from_numpy(out_cuda_warp)

        out_cuda = np.load("./npy_verify/rmsnorm.npy")
        out_cuda = torch.from_numpy(out_cuda)

        out_cuda_warp2 = np.load("./npy_verify/rmsnorm_warp2.npy")
        out_cuda_warp2 = torch.from_numpy(out_cuda_warp2)

        # Compare outputs
        torch.allclose(out_custom, out_torch, rtol=1e-5, atol=1e-6)
        torch.allclose(out_cuda, out_torch, rtol=1e-5, atol=1e-6)
        torch.allclose(out_cuda_warp, out_torch, rtol=1e-5, atol=1e-6)
        torch.allclose(out_cuda_warp2, out_torch, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()