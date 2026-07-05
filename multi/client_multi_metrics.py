import torch
import torch.nn as nn
import numpy as np
import sys
import flwr as fl

# =========================
# 再現性のための乱数固定
# =========================
client_id = int(sys.argv[1])

np.random.seed(42 + client_id)
torch.manual_seed(42 + client_id)

# =========================
# モデル定義
# 3入力 → 1出力
# =========================
model = nn.Linear(3, 1)
print("model created")

# =========================
# データ作成：3入力の多変量回帰データ
# =========================
x = np.random.rand(100, 3)

# clientごとにデータ分割（非IID）
# 1つ目の入力 x[:, 0] を基準にして分割
if client_id == 1:
    x = x[x[:, 0] < 0.33]
elif client_id == 2:
    x = x[(x[:, 0] >= 0.33) & (x[:, 0] < 0.66)]
else:
    x = x[x[:, 0] >= 0.66]

# データ0件防止
if len(x) == 0:
    x = np.random.rand(10, 3)

# 出力 y を作成
# y = 2*x1 + 3*x2 + 5*x3 + 1
y = (
    2 * x[:, 0]
    + 3 * x[:, 1]
    + 5 * x[:, 2]
    + 1
)

# yをPyTorchで扱いやすい形にする
y = y.reshape(-1, 1)

print("x shape:", x.shape)
print("y shape:", y.shape)

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

            mse = loss_fn(y_pred, y_test)
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(y_pred - y_test))

            ss_res = torch.sum((y_test - y_pred) ** 2)
            ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)

            if ss_tot.item() == 0:
                r2 = torch.tensor(0.0)
            else:
                r2 = 1 - ss_res / ss_tot

        print(
            f"Client {client_id} | "
            f"MSE: {mse.item()} | "
            f"RMSE: {rmse.item()} | "
            f"MAE: {mae.item()} | "
            f"R2: {r2.item()}"
        )

        return mse.item(), len(x_test), {
            "mse": mse.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
            "r2": r2.item(),
        }

# =========================
# Flower 接続
# =========================
print("starting Flower connection")

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
