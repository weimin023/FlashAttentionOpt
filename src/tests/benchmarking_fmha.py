import torch
import torch.nn.functional as F
import time
torch.set_float32_matmul_precision('high')

torch.manual_seed(42)
device = 'cuda'

@torch.no_grad()
def run_attention(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=False)

@torch.no_grad()
def benchmark(name, func, warmup=10, repeat=100):
    # Warm-up
    for _ in range(warmup):
        func()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        func()
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000 / repeat
    print(f"{name:<20} | Avg Time: {avg_ms:.3f} ms")

def create_inputs(B, H, S_q, S_k, D_head):
    Q = torch.randn(B * H, S_q, D_head, device=device)
    K = torch.randn(B * H, S_k, D_head, device=device)
    V = torch.randn(B * H, S_k, D_head, device=device)
    return Q, K, V

def main():
    H = 8
    S_q = S_k = 128
    D_head = 64

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    for B in batch_sizes:
        print(f"\n=== B={B}, H={H}, S_q=S_k={S_q}, D_head={D_head} ===")
        Q, K, V = create_inputs(B, H, S_q, S_k, D_head)

        # Eager
        benchmark("Eager", lambda: run_attention(Q, K, V))

        # Graph (compile with mode='max-autotune')
        compiled_fn = torch.compile(run_attention, mode='max-autotune', fullgraph=True)
        benchmark("Torch Compile", lambda: compiled_fn(Q, K, V))

if __name__ == "__main__":
    main()
