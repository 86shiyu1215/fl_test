import flwr as fl
import matplotlib

# 画面表示なしで画像保存できるようにする
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import csv

# =========================
# 設定
# =========================
NUM_ROUNDS = 20

# =========================
# Flower server 起動
# =========================
history = fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
)

# =========================
# Roundごとのlossを取得
# =========================
losses = history.losses_distributed

print("==== Distributed Loss History ====")
print(losses)

# =========================
# CSVとグラフに保存
# =========================
if len(losses) > 0:
    rounds = [r for r, loss in losses]
    loss_values = [loss for r, loss in losses]

    # CSV保存
    with open("federated_loss_multi.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss"])
        for r, loss in zip(rounds, loss_values):
            writer.writerow([r, loss])

    # グラフ保存
    plt.figure()
    plt.plot(rounds, loss_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Federated Learning Loss Curve Multi Regression")
    plt.grid(True)
    plt.savefig("federated_loss_curve_multi.png", dpi=300)

    print("Saved: federated_loss_multi.csv")
    print("Saved: federated_loss_curve_multi.png")

else:
    print("No distributed loss was recorded.")
