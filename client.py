# =========================
# 残りのtorch変換（続き）
# =========================
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
# Flower クライアント定義
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

            print(f"Client {client_id} epoch {epoch}, loss: {loss.item()}")

        return self.get_parameters(config), len(x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        with torch.no_grad():
            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)

        print(f"Client {client_id} test loss: {loss.item()}")

        return float(loss), len(x_test), {}

# =========================
# Flower 接続（超重要🔥）
# =========================
print("starting Flower connection")

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
``
