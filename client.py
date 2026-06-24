import torch
import torch.nn as nn
import numpy as np
import sys
import flwr as fl

# =========================
# モデル定義
# =========================
model = nn.Linear(1, 1)
print("model created")

# =========================
# client番号取得
# =========================
client_id = int(sys.argv[1])

# =========================
# データ作成
# =========================
x = np.random.rand(100, 1)

# clientごとにデータ分割（非IID）
if client_id == 1:
    x = x[x < 0.33]
elif client_id == 2:
    x = x[(x >= 0.33) & (x < 0.66)]
else:
    x = x[x >= 0.66]

# データ0件防止
if len(x) == 0:
    x = np.random.rand(10, 1)

# shape補正（超重要）
x = x.reshape(-1, 1)

# 出力
y = 3 * x + 2

# =========================
# train/test分割
# =========================
split_index = int(0.8 * len(x))

x_train_np = x[:split_index]
y_train_np = y[:split_index]

x_test_np = x[split_index:]
y_test_np = y[split_index:]

# torch変換
x_train = torch.tensor(x_train_np, dtype=torch.float32)
