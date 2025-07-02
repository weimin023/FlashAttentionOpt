import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


'''# 建立資料
data = {
    'batch': [1, 4, 8, 16, 32] * 4,
    'time_ms': [0.133, 0.462, 0.897, 1.793, 3.628, 
                0.153, 0.188, 0.23, 0.362, 0.694,
                0.073, 0.216, 0.332, 0.605, 1.235,
                0.139, 0.216, 0.384, 0.546, 1.238],
    'method': ['CUDA Custom Kernel (3 Kernels)'] * 5 + ['CUTLASS Custom Kernel (2 GemmBatched + 1 Softmax)'] * 5 + ['Torch (Eager Mode)'] * 5 + ['Torch (Graph Mode)'] * 5
}

df = pd.DataFrame(data)

# 畫圖
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='batch', y='time_ms', hue='method', marker='o')

# 標題與標籤
plt.title(r'$\mathbf{Scaled\ Dot\ Product\ Attention}$' + '\nExecution Time vs Batch Size (seq_q=128, seq_k=2048, d_k=d_v=128)')
plt.xlabel('Batch Size')
plt.ylabel('Elapsed Time (ms)')
plt.yscale('linear')  # 或 'log' 也可視情況
plt.xticks([1, 4, 8, 16, 32])
plt.tight_layout()
plt.legend(title=None)

plt.show()'''

'''# 建立資料
seq_shapes = ['64x512', '128x1024', '128x2048', '256x4096', '512x8192']
data = {
    'seq_shape': seq_shapes * 4,
    'time_ms': [0.127, 0.462, 0.897, 3.456, 12.621,
                0.071, 0.128, 0.23, 0.604, 2.048,
                0.078, 0.151, 0.358, 0.976, 3.797,
                0.115, 0.160, 0.366, 0.98, 3.812],
    'method': ['CUDA Custom Kernel (3 Kernels)'] * 5 + ['CUTLASS Custom Kernel (2 GemmBatched + 1 Softmax)'] * 5 + ['Torch (Eager Mode)'] * 5 + ['Torch (Graph Mode)'] * 5
}

df = pd.DataFrame(data)

# 畫圖
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='seq_shape', y='time_ms', hue='method', marker='o')

# 粗體第一行標題
plt.title(r'$\mathbf{Scaled\ Dot\ Product\ Attention}$' + '\nExecution Time vs Sequence Shape (batch=8, d_k=d_v=128)')
plt.xlabel('Sequence Shape (seq_q × seq_k)')
plt.ylabel('Elapsed Time (ms)')
plt.yscale('linear')  # 或 'log' 看變化曲線也不錯
plt.legend(title=None)
plt.tight_layout()

plt.show()'''

batch_sizes = ['1', '2', '4', '8', '16', '32', '64']

data = {
    'batch_size': batch_sizes * 4,
    'time_ms': [
        5.519, 10.942, 19.776, 39.598, 79.505, 159.298, 320.349,   # CUDA custom
        0.0066, 0.00828, 0.01034, 0.01669, 0.0282, 0.05198, 0.148, # CUTLASS
        0.120, 0.120, 0.118, 0.120, 0.231, 0.788, 1.653,            # Torch Eager
        0.090, 0.090, 0.089, 0.089, 0.197, 0.604, 1.541             # Torch Compile
    ],
    'method': (
        ['CUDA Custom Kernel (Per Block Process a Q)'] * 7 +
        ['CUTLASS Kernel'] * 7 +
        ['Torch (Eager Mode)'] * 7 +
        ['Torch (Compile Mode)'] * 7
    )
}

df = pd.DataFrame(data)

# 畫圖
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='batch_size', y='time_ms', hue='method', marker='o')

plt.title(r'$\mathbf{Fused\ Multi\ Head\ Attention}$' + '\nExecution Time vs Batch Size (H=8, S_q=S_k=128, D_head=64)')
plt.xlabel('Batch Size')
plt.ylabel('Elapsed Time (ms)')
plt.yscale('log')  # 用 log scale 更容易看出差距
plt.legend(title=None)
plt.tight_layout()
plt.show()