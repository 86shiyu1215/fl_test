import torch  #pytorch を使うための準備
import torch.nn as nn  #ニューラルネットワーク用

# シンプルなモデル（1入力→1出力）
model = nn.Linear(1, 1)  #入力１つ，出力１つ，のモデルで行くで

print("model created") #動作確認
import numpy as np

# データ作成（100個のデータ）
x = np.random.rand(100, 1)  # 入力データ（0〜1のランダム）
y = 3 * x + 2               # 出力データ（関係式）

# torch用に変換
x_train = torch.tensor(x, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

print("data created")
