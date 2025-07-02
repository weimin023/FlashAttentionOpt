import torch
import torch.nn.functional as F
import time

def prepare_data(batch, seq_len_q, seq_len_k, d_k=128, d_v=128):
    h_Q = torch.zeros((batch, seq_len_q, d_k), dtype=torch.float32, device='cuda')
    h_K = torch.zeros((batch, seq_len_k, d_k), dtype=torch.float32, device='cuda')
    h_V = torch.zeros((batch, seq_len_k, d_v), dtype=torch.float32, device='cuda')

    # 初始化資料
    for i_b in range(batch):
        for i in range(seq_len_q):
            for j in range(d_k):
                h_Q[i_b, i, j] = ((i * d_k + j) % 100) * 0.01 - 0.5

        for i in range(seq_len_k):
            for j in range(d_k):
                h_K[i_b, i, j] = ((i * d_k + j) % 100) * 0.02 - 1.0

        for i in range(seq_len_k):
            for j in range(d_v):
                h_V[i_b, i, j] = ((i * d_v + j) % 100) * 0.01

    return h_Q, h_K, h_V


def run_attention(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=False)

# 用 torch.compile 加速
compiled_attention = torch.compile(run_attention)

# Benchmark
seq_q = [64, 128, 128, 256, 512]
seq_k = [512, 1024, 2048, 4096, 8192]
warmup = 3
repeat = 10

print(f"{'Batch':>6} | {'Eager Time (ms)':>15} | {'Compiled Time (ms)':>18}")
print("-" * 45)

for i in range(5):
    Q, K, V = prepare_data(8, seq_q[i], seq_k[i])

    # Warmup
    for _ in range(warmup):
        run_attention(Q, K, V)
        compiled_attention(Q, K, V)

    # Benchmark eager
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        run_attention(Q, K, V)
    torch.cuda.synchronize()
    eager_time = (time.time() - t0) / repeat * 1000

    # Benchmark compiled
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        compiled_attention(Q, K, V)
    torch.cuda.synchronize()
    compiled_time = (time.time() - t0) / repeat * 1000

    print(f"{seq_q[i]:6} | {seq_k[i]:6}| {eager_time:15.3f} | {compiled_time:18.3f}")
