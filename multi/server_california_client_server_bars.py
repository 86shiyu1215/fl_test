
import os
import csv
import pandas as pd
import flwr as fl
import matplotlib

# 画面表示なしで画像保存できるようにする
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# =========================
# 設定
# =========================
NUM_ROUNDS = 80

# server側のRoundごとの集約結果
SERVER_CSV = "server_metrics_california_round80.csv"
SERVER_PNG = "server_metrics_california_round80.png"

# client側のRoundごとの評価結果
CLIENT1_CSV = "client1_metrics_california.csv"
CLIENT2_CSV = "client2_metrics_california.csv"
CLIENT3_CSV = "client3_metrics_california.csv"

# 最終Round後のclient/server比較結果
FINAL_COMPARISON_CSV = "final_client_server_comparison_california.csv"
FINAL_COMPARISON_PNG = "final_client_server_comparison_california.png"

# =========================
# 古い結果が残らないように削除
# =========================
for file in [
    SERVER_CSV,
    SERVER_PNG,
    FINAL_COMPARISON_CSV,
    FINAL_COMPARISON_PNG,
]:
    if os.path.exists(file):
        os.remove(file)

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
# 各Round番号をclientへ渡す
# =========================
def evaluate_config(server_round):
    return {"round": server_round}

# =========================
# Strategy設定
# =========================
# 3client全員が揃うまで待ち、fit/evaluateの両方で3client全員を使用する
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_evaluate_config_fn=evaluate_config,
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
# server集約結果を取得
# =========================
metrics = history.metrics_distributed

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
# server集約結果CSV保存
# =========================
with open(SERVER_CSV, "w", newline="") as f:
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

print(f"Saved: {SERVER_CSV}")

# =========================
# server側 Roundごとの推移グラフ保存
# =========================
if len(rounds) > 0:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(rounds, mse_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("MSE")
    plt.title("Server Aggregated MSE")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(rounds, rmse_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("RMSE")
    plt.title("Server Aggregated RMSE")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(rounds, mae_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("MAE")
    plt.title("Server Aggregated MAE")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(rounds, r2_values, marker="o")
    plt.xlabel("Round")
    plt.ylabel("R2")
    plt.title("Server Aggregated R2")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(SERVER_PNG, dpi=300)

    print(f"Saved: {SERVER_PNG}")
else:
    print("No server round metrics were recorded.")

# =========================
# 最終Roundのclient/server比較表を作成
# =========================
def load_final_row(file_name, label):
    df = pd.read_csv(file_name)

    # roundが-1などの初期評価を含む可能性があるため、正のroundだけ使う
    df = df[df["round"] > 0]

    final_row = df.loc[df["round"].idxmax()]

    return {
        "name": label,
        "round": int(final_row["round"]),
        "mse": float(final_row["mse"]),
        "rmse": float(final_row["rmse"]),
        "mae": float(final_row["mae"]),
        "r2": float(final_row["r2"]),
    }

results = []

# client側CSV
for file_name, label in [
    (CLIENT1_CSV, "client1"),
    (CLIENT2_CSV, "client2"),
    (CLIENT3_CSV, "client3"),
]:
    if os.path.exists(file_name):
        results.append(load_final_row(file_name, label))
    else:
        print(f"Warning: {file_name} not found.")

# server側CSV
if os.path.exists(SERVER_CSV):
    results.append(load_final_row(SERVER_CSV, "server"))
else:
    print(f"Warning: {SERVER_CSV} not found.")

# =========================
# 最終Round比較CSV保存
# =========================
with open(FINAL_COMPARISON_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "round", "mse", "rmse", "mae", "r2"])

    for row in results:
        writer.writerow([
            row["name"],
            row["round"],
            row["mse"],
            row["rmse"],
            row["mae"],
            row["r2"],
        ])

print(f"Saved: {FINAL_COMPARISON_CSV}")

# =========================
# client1, client2, client3, server の棒グラフ作成
# =========================
if len(results) > 0:
    names = [row["name"] for row in results]
    mse_plot = [row["mse"] for row in results]
    rmse_plot = [row["rmse"] for row in results]
    mae_plot = [row["mae"] for row in results]
    r2_plot = [row["r2"] for row in results]

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.bar(names, mse_plot)
    plt.ylabel("MSE")
    plt.title("Final Round MSE")
    plt.grid(axis="y")

    plt.subplot(2, 2, 2)
    plt.bar(names, rmse_plot)
    plt.ylabel("RMSE")
    plt.title("Final Round RMSE")
    plt.grid(axis="y")

    plt.subplot(2, 2, 3)
    plt.bar(names, mae_plot)
    plt.ylabel("MAE")
    plt.title("Final Round MAE")
    plt.grid(axis="y")

    plt.subplot(2, 2, 4)
    plt.bar(names, r2_plot)
    plt.ylabel("R2")
    plt.title("Final Round R2")
    plt.grid(axis="y")

    plt.tight_layout()
    plt.savefig(FINAL_COMPARISON_PNG, dpi=300)

    print(f"Saved: {FINAL_COMPARISON_PNG}")
else:
    print("No final comparison results were recorded.")
