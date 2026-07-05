import sys
import numpy as np
import torch
import torch.nn as nn
import flwr as fl

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

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
# Irisデータ読み込み
# =========================
iris = load_iris()

# Irisの特徴量は4つ
# 0: sepal length
# 1: sepal width
# 2: petal length
# 3: petal width
data = iris.data
labels = iris.target

# =========================
# 回帰問題として使う
# =========================
# 入力：3項目
#  - sepal length
#  - sepal width
#  - petal length
#
# 出力：1項目
#  - petal width
x = data[:, 0:3]
y = data[:, 3].reshape(-1, 1)

# =========================
# 標準化
# =========================
# 今回は公開データを使う基礎実験なので、
# 全体データで標準化している。
# 実データ・機密データでは扱いに注意。
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y)

# =========================
# clientごとに非IID分割
# =========================
# Irisのクラスラベルを使って、
# clientごとに異なる品種データを持たせる。
#
# client1: class 0
# client2: class 1
# client3: class 2
if client_id == 1:
    indices = labels == 0
elif client_id == 2:
    indices = labels == 1
else:
    indices = labels == 2

x = x[indices]
y = y[indices]

print(f"Client {client_id} data size: {len(x)}")
print("x shape:", x.shape)
print("y shape:", y.shape)

# =========================
# train/test分割
# =========================
# 各clientのデータを80% train, 20% testに分ける
num_samples = len(x)
indices = np.random.permutation(num_samples)

split_index = int(0.8 * num_samples)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

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

# =========================
# モデル定義
# =========================
# 3入力 → 1出力
model = nn.Linear(3, 1)
print("model created")

# =========================
# 学習設定
# =========================
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
