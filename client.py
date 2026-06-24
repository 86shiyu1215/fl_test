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

# データ0件防止（重要）
if len(x) == 0:
    x = np.random.rand(10, 1)

# shape補正（超重要）
x = np.array(x).reshape(-1, 1)

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
y_train = torch.tensor(y_train_np, dtype=torch.float32)

x_test = torch.tensor(x_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

print("data created")

# =========================
# 学習設定
# =========================
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# =========================
# Flower クライアント
# =========================
class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.detach().numpy() for val in model.parameters()]

    def set_parameters(self, parameters):
        for param, new_val in zip(model.parameters(), parameters):
            param.data = torch.tensor(new_val, dtype=torch.float32)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        print(f"Client {client_id} training start")

        for epoch in range(50):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()

        return self.get_parameters(config), len(x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        with torch.no_grad():
            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)

        print(f"Client {client_id} test loss: {loss.item()}")

        return float(loss), len(x_test), {}

# =========================
# Flower 接続（最重要🔥）
# =========================
print("starting Flower connection")

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
