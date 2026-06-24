import torch  #pytorch を使うための準備
import torch.nn as nn  #ニューラルネットワーク用

# シンプルなモデル（1入力→1出力）
model = nn.Linear(1, 1)  #入力１つ，出力１つ，のモデルで行くで

print("model created") #動作確認
import numpy as np

# データ作成（100個のデータ）
x = np.random.rand(100, 1) # 入力データ（0〜1のランダム）
y = 3 * x + 2 # 出力データ（関係式）

# データ分割（80%:train, 20%:test）
split_index = int(0.8 * len(x))

x_train_np = x[:split_index]
y_train_np = y[:split_index]

x_test_np = x[split_index:]
y_test_np = y[split_index:]

# torch用に変換
x_train = torch.tensor(x_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)

x_test = torch.tensor(x_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

print("data created")

# 学習の準備
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) #torch.optim.SGD モデルをどう賢くするかのルール
loss_fn = nn.MSELoss() #誤差の測り方

# 学習ループ
for epoch in range(200): #学習回数10回に設定
    optimizer.zero_grad()  # 勾配リセット
    
    y_pred = model(x_train)  # モデルの予測　lossが小さくなることで学習が成功したことになる
    
    loss = loss_fn(y_pred, y_train)  # 予測と正解の誤差
    
    loss.backward()  # 誤差をもとに，どこが間違ってるか計算
    
    optimizer.step()  # パラメータ更新
    
    print(f"epoch {epoch}, loss: {loss.item()}")

# テストデータで評価
with torch.no_grad():
    y_test_pred = model(x_test)
    test_loss = loss_fn(y_test_pred, y_test)

print(f"Test loss: {test_loss.item()}")
import flwr as fl

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.detach().numpy() for val in model.parameters()]

    def set_parameters(self, parameters):
        for param, new_val in zip(model.parameters(), parameters):
            param.data = torch.tensor(new_val)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # 学習（今までのコード）
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

        return float(loss), len(x_test), {}


# Flower client起動
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
