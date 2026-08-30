from pathlib import Path

import pandas as pd
import torch

# ============================================================
# matplotlib
# Ubuntuのサーバー環境でも画面表示なしでPNG保存できるようにする
# ============================================================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
)

from flwr.serverapp import (
    Grid,
    ServerApp,
)

from flwr.serverapp.strategy import FedAvg


from model import CoFDNN

from task import (
    TARGET_COLUMN,
    evaluate_model,
    load_csv_data,
    set_seed,
)


# ============================================================
# Flower ServerApp
# ============================================================

app = ServerApp()


# ============================================================
# Server処理
# ============================================================

@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:

    set_seed()

    # --------------------------------------------------------
    # pyproject.tomlから設定を取得
    # --------------------------------------------------------

    num_rounds = int(
        context.run_config[
            "num-server-rounds"
        ]
    )

    learning_rate = float(
        context.run_config[
            "learning-rate"
        ]
    )

    local_epochs = int(
        context.run_config[
            "local-epochs"
        ]
    )

    batch_size = int(
        context.run_config[
            "batch-size"
        ]
    )

    fraction_train = float(
        context.run_config[
            "fraction-train"
        ]
    )

    test_csv = Path(
        str(
            context.run_config[
                "test-data-path"
            ]
        )
    )

    result_dir = Path(
        str(
            context.run_config[
                "result-dir"
            ]
        )
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TEST 32件を読み込む
    #
    # この32件は学習には使用しない。
    # Global modelの評価だけに使用する。
    # --------------------------------------------------------

    x_test, y_test, test_df = load_csv_data(
        test_csv
    )

    print(
        "========================================"
    )

    print(
        "STEP3 DNN Federated Learning"
    )

    print(
        "Standardization: OFF"
    )

    print(
        "========================================"
    )

    print(
        f"Test samples : {len(x_test)}"
    )

    print(
        f"Rounds       : {num_rounds}"
    )

    print(
        f"Local epochs : {local_epochs}"
    )

    print(
        f"Batch size   : {batch_size}"
    )

    print(
        f"Learning rate: {learning_rate}"
    )

    print()

    # ========================================================
    # Global DNN初期化
    # ========================================================

    global_model = CoFDNN()

    initial_arrays = ArrayRecord(
        global_model.state_dict()
    )

    # ========================================================
    # Roundごとの評価指標履歴
    # ========================================================

    metric_history = []

    metrics_csv_path = (
        result_dir
        / "server_round_metrics_no_scaling.csv"
    )

    # ========================================================
    # Global model評価関数
    #
    # Round 0
    #   FL開始前
    #
    # Round 1～30
    #   FedAvg後
    # ========================================================

    def global_evaluate(
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord:

        model = CoFDNN()

        model.load_state_dict(
            arrays.to_torch_state_dict()
        )

        metrics, _ = evaluate_model(
            model,
            x_test,
            y_test,
        )

        row = {
            "round": int(
                server_round
            ),
            "mse": float(
                metrics["mse"]
            ),
            "rmse": float(
                metrics["rmse"]
            ),
            "mae": float(
                metrics["mae"]
            ),
            "r2": float(
                metrics["r2"]
            ),
        }

        metric_history.append(
            row
        )

        # ----------------------------------------------------
        # 毎Round CSVを更新
        #
        # 実験途中で停止しても
        # そこまでの結果を残す
        # ----------------------------------------------------

        pd.DataFrame(
            metric_history
        ).to_csv(
            metrics_csv_path,
            index=False,
        )

        print(
            f"Global Round {server_round:2d} | "
            f"MSE={metrics['mse']:.8f} | "
            f"RMSE={metrics['rmse']:.8f} | "
            f"MAE={metrics['mae']:.8f} | "
            f"R2={metrics['r2']:.8f}"
        )

        return MetricRecord(
            {
                "mse": float(
                    metrics["mse"]
                ),
                "rmse": float(
                    metrics["rmse"]
                ),
                "mae": float(
                    metrics["mae"]
                ),
                "r2": float(
                    metrics["r2"]
                ),
            }
        )

    # ========================================================
    # FedAvg
    # ========================================================

    strategy = FedAvg(

        fraction_train=(
            fraction_train
        ),

        # Client側では評価しない
        fraction_evaluate=0.0,

        # 毎Round必ず3 Client参加
        min_train_nodes=3,

        min_available_nodes=3,
    )

    # ========================================================
    # Clientへ送る学習条件
    # ========================================================

    train_config = ConfigRecord(
        {
            "learning-rate": (
                learning_rate
            ),
            "local-epochs": (
                local_epochs
            ),
            "batch-size": (
                batch_size
            ),
        }
    )

    # ========================================================
    # Federated Learning開始
    # ========================================================

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # ========================================================
    # 最終Global model保存
    # ========================================================

    final_state_dict = (
        result.arrays.to_torch_state_dict()
    )

    model_path = (
        result_dir
        / "global_model_no_scaling.pt"
    )

    torch.save(
        final_state_dict,
        model_path,
    )

    # ========================================================
    # Client側Train LossをCSV保存
    # ========================================================

    train_loss_rows = []

    for server_round, metrics in sorted(
        result.train_metrics_clientapp.items()
    ):

        train_loss_rows.append(
            {
                "round": int(
                    server_round
                ),
                "train_loss": float(
                    metrics[
                        "train_loss"
                    ]
                ),
            }
        )

    train_loss_csv_path = (
        result_dir
        / "client_train_loss_no_scaling.csv"
    )

    pd.DataFrame(
        train_loss_rows
    ).to_csv(
        train_loss_csv_path,
        index=False,
    )

    # ========================================================
    # 最終Global modelをTEST32件で評価
    # ========================================================

    final_model = CoFDNN()

    final_model.load_state_dict(
        final_state_dict
    )

    final_metrics, final_predictions = (
        evaluate_model(
            final_model,
            x_test,
            y_test,
        )
    )

    # ========================================================
    # TEST32件
    # 実測CoF vs 予測CoF
    # CSV保存
    # ========================================================

    prediction_df = pd.DataFrame(
        {
            "data_id": (
                test_df[
                    "data_id"
                ]
            ),

            "actual_cof": (
                test_df[
                    TARGET_COLUMN
                ]
            ),

            "predicted_cof": (
                final_predictions
            ),
        }
    )

    # --------------------------------------------------------
    # 残差
    #
    # residual
    # = Actual - Predicted
    # --------------------------------------------------------

    prediction_df[
        "residual"
    ] = (
        prediction_df[
            "actual_cof"
        ]
        -
        prediction_df[
            "predicted_cof"
        ]
    )

    prediction_csv_path = (
        result_dir
        / "test_predictions_no_scaling.csv"
    )

    prediction_df.to_csv(
        prediction_csv_path,
        index=False,
    )

    # ========================================================
    # Actual CoF vs Predicted CoF
    #
    # TEST32件の散布図
    # ＋ 理想線 y = x
    # ========================================================

    actual = prediction_df[
        "actual_cof"
    ].to_numpy()

    predicted = prediction_df[
        "predicted_cof"
    ].to_numpy()

    # --------------------------------------------------------
    # ActualとPredictedの両方を含む
    # 共通の軸範囲を作る
    # --------------------------------------------------------

    plot_min = min(
        actual.min(),
        predicted.min(),
    )

    plot_max = max(
        actual.max(),
        predicted.max(),
    )

    margin = (
        plot_max
        - plot_min
    ) * 0.05

    plot_min = (
        plot_min
        - margin
    )

    plot_max = (
        plot_max
        + margin
    )

    # --------------------------------------------------------
    # 散布図を作成
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 6)
    )

    # TEST32件の
    # Actual vs Predicted
    plt.scatter(
        actual,
        predicted,
        s=50,
    )

    # --------------------------------------------------------
    # 理想線
    #
    # Actual = Predicted
    # y = x
    # --------------------------------------------------------

    plt.plot(
        [
            plot_min,
            plot_max,
        ],
        [
            plot_min,
            plot_max,
        ],
        linestyle="--",
        label="y = x",
    )

    # --------------------------------------------------------
    # 軸
    # --------------------------------------------------------

    plt.xlabel(
        "Actual CoF"
    )

    plt.ylabel(
        "Predicted CoF"
    )

    # --------------------------------------------------------
    # タイトル
    # R²も表示
    # --------------------------------------------------------

    plt.title(
        "Actual vs Predicted CoF\n"
        f"No Standardization, "
        f"R² = "
        f"{final_metrics['r2']:.3f}"
    )

    # --------------------------------------------------------
    # x軸・y軸を同じ範囲にする
    # --------------------------------------------------------

    plt.xlim(
        plot_min,
        plot_max,
    )

    plt.ylim(
        plot_min,
        plot_max,
    )

    # --------------------------------------------------------
    # x軸とy軸の縮尺を1:1にする
    #
    # y=xからの距離を
    # 視覚的に正しく見るため重要
    # --------------------------------------------------------

    plt.gca().set_aspect(
        "equal",
        adjustable="box",
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # PNG保存
    # --------------------------------------------------------

    scatter_path = (
        result_dir
        / "actual_vs_predicted_no_scaling.png"
    )

    plt.savefig(
        scatter_path,
        dpi=300,
        bbox_inches="tight",
    )

    # メモリ解放
    plt.close()

    # ========================================================
    # 最終評価指標をCSV保存
    # ========================================================

    final_metrics_df = pd.DataFrame(
        [
            {
                "round": (
                    num_rounds
                ),

                "mse": float(
                    final_metrics[
                        "mse"
                    ]
                ),

                "rmse": float(
                    final_metrics[
                        "rmse"
                    ]
                ),

                "mae": float(
                    final_metrics[
                        "mae"
                    ]
                ),

                "r2": float(
                    final_metrics[
                        "r2"
                    ]
                ),
            }
        ]
    )

    final_metrics_csv_path = (
        result_dir
        / "final_metrics_no_scaling.csv"
    )

    final_metrics_df.to_csv(
        final_metrics_csv_path,
        index=False,
    )

    # ========================================================
    # 完了表示
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "Federated Learning completed"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Saved:"
    )

    print(
        metrics_csv_path
    )

    print(
        train_loss_csv_path
    )

    print(
        prediction_csv_path
    )

    print(
        final_metrics_csv_path
    )

    print(
        scatter_path
    )

    print(
        model_path
    )

    print()

    print(
        "Final Test Metrics"
    )

    print(
        f"MSE  : "
        f"{final_metrics['mse']:.8f}"
    )

    print(
        f"RMSE : "
        f"{final_metrics['rmse']:.8f}"
    )

    print(
        f"MAE  : "
        f"{final_metrics['mae']:.8f}"
    )

    print(
        f"R2   : "
        f"{final_metrics['r2']:.8f}"
    )
