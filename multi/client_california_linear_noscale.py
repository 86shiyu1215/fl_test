import sys
import numpy as np
import torch
import torch.nn as nn
import flwr as fl

from sklearn.datasets import fetch_california_housing

# =========================
# client番号取得
# =========================
client_id = int(sys.argv[1])

# =========================
# 再現性のための乱数固定
# =========================
np.random.seed(42 + client_id)
torch.manual_seed(42 + client_id)

# =========================
# California Housing データ読み込み
# =========================
housing = fetch_california_housing()

# 入力：8特徴量
# 出力：住宅価格
x_raw = housing.data
y_raw = housing.target.reshape(-1, 1)

feature_names = housing.feature_names

print("feature names:", feature_names)
print("original x shape:", x_raw.shape)
print("original y shape:", y_raw.shape)

# =========================
# 非IID分割用の基準
# =========================
# MedInc（median income）を使って、
# 低所得・中所得・高所得の3 clientに分ける
medinc = x_raw[:, 0]

q1 = np.quantile(medinc, 1 / 3)
q2 = np.quantile(medinc, 2 / 3)

if client_id == 1:
    indices = medinc <= q1
elif client_id == 2:
    indices = (medinc > q1) & (medinc <= q2)
else:
    indices = medinc > q2

# =========================
# 標準化なし
# =========================
# 今回は、特徴量スケールの影響を確認するため、
# StandardScalerを使わずにそのまま学習する。
x = x_raw[indices]
y = y_raw[indices]

print(f"Client {client_id} data size: {len(x)}")
print("x shape:", x.shape)
print("y shape:", y.shape)

# =========================
# train/test分割
# =========================
num_samples = len(x)
shuffled_indices = np.random.permutation(num_samples)

split_index = int(0.8 * num_samples)

train_indices = shuffled_indices[:split_index]
test_indices = shuffled_indices[split_index:]

x_train_np = x[train_indices]
y_train_np = y[train_indices]

x_test_np = x[test_indices]
y_test_np = y[test_indices]

# =========================
# torch Tensor化
# =========================
x_train = torch.tensor(x_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

x_test = torch.tensor(x_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

print("train x shape:", x_train.shape)
print("test x shape:", x_test.shape)

# =========================
# モデル定義
# =========================
# California Housingは8入力 → 1出力
# 線形モデル版
model = nn.Linear(8, 1)

print("model created: Linear(8, 1), no scaling")

# =========================
# 学習設定
# =========================
# 注意：
# 標準化なしで SGD(lr=0.01) を使うと発散する可能性が高い。
# まずは標準化あり版と比較するため、同じ条件で実行する。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# =========================
# Flower Client
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
# Flower接続
# =========================
print("starting Flower connection")

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
