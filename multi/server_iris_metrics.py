import flwr as fl
import matplotlib

# 画面表示なしで画像保存できるようにする
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import csv

# =========================
# 設定
# =========================
NUM_ROUNDS = 40

# =========================
# 評価指標の集約関数
# =========================
def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    mse = sum(num_examples * m["mse"] for num_examples, m in metrics) / total_examples
    rmse = sum(num_examples * m["rmse"] for num_examples, m in metrics) / total_examples
    mae = sum(num_examples * m["mae"] for num_examples, m in metrics) / total_examples
    r2 = sum(num_examples * m["r2"] for num_examples, m in metrics) / total_examples

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

# =========================
# Strategy設定
# =========================
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)

# =========================
# Flower server 起動
# =========================
history = fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

# =========================
# 結果取得
# =========================
losses = history.losses_distributed
metrics = history.metrics_distributed

print("==== Iris Distributed Loss History ====")
print(losses)

print("==== Iris Distributed Metrics History ====")
print(metrics)

# =========================
# CSV保存用リスト作成
# =========================
rounds = []
mse_values = []
rmse_values = []
mae_values = []
r2_values = []

if "mse" in metrics:
    for r, value in metrics["mse"]:
        rounds.append(r)
        mse_values.append(value)

if "rmse" in metrics:
    rmse_values = [value for r, value in metrics["rmse"]]

if "mae" in metrics:
    mae_values = [value for r, value in metrics["mae"]]

if "r2" in metrics:
    r2_values = [value for r, value in metrics["r2"]]

# =========================
# CSV保存
# =========================
with open("federated_metrics_iris.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["round", "mse", "rmse", "mae", "r2"])

    for i in range(len(rounds)):
        writer.writerow([
            rounds[i],
            mse_values[i],
            rmse_values[i],
            mae_values[i],
            r2_values[i],
        ])

print("Saved: federated_metrics_iris.csv")

# =========================
# グラフ保存
# =========================
if len(rounds) > 0:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(rounds, mse_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("MSE")
    plt.title("Iris Regression MSE")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(rounds, rmse_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("RMSE")
    plt.title("Iris Regression RMSE")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(rounds, mae_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("MAE")
    plt.title("Iris Regression MAE")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(rounds, r2_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("R2")
    plt.title("Iris Regression R2")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("federated_metrics_iris.png", dpi=300)

    print("Saved: federated_metrics_iris.png")

else:
    print("No metrics were recorded.")
