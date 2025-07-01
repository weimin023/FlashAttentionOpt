import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

'''
# 建立資料
data = {
    'batch': [1, 4, 8, 16, 32] * 2,
    'time_ms': [0.133, 0.462, 0.897, 1.793, 3.628, 
                0.153, 0.188, 0.23, 0.362, 0.694],
    'method': ['CUDA Custom Kernel (3 Kernels)'] * 5 + ['CUTLASS Custom Kernel (2 GemmBatched + 1 Softmax)'] * 5
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

# 建立資料
seq_shapes = ['64x512', '128x1024', '128x2048', '256x4096', '512x8192']
data = {
    'seq_shape': seq_shapes * 2,
    'time_ms': [0.127, 0.462, 0.897, 3.456, 12.621,
                0.071, 0.128, 0.23, 0.604, 2.048],
    'method': ['CUDA Custom Kernel (3 Kernels)'] * 5 + ['CUTLASS Custom Kernel (2 GemmBatched + 1 Softmax)'] * 5
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

plt.show()