from pathlib import Path

import pandas as pd
import torch

# ============================================================
# matplotlib
# Ubuntuでも画面表示なしでPNG保存できるようにする
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

from experiment_utils import (
    initialize_experiment,
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

    # ========================================================
    # pyproject.tomlから実験条件を取得
    # ========================================================

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

    # --------------------------------------------------------
    # experiment-id
    #
    # pyproject.tomlに設定がなければ
    # no-scaling baselineとして保存
    # --------------------------------------------------------

    experiment_id = str(
        context.run_config.get(
            "experiment-id",
            "exp_001_no_scaling_baseline",
        )
    )

    # --------------------------------------------------------
    # standardization
    #
    # 現在は none
    # 後で zscore に変更する
    # --------------------------------------------------------

    standardization = str(
        context.run_config.get(
            "standardization",
            "none",
        )
    )

    # ========================================================
    # STEP3フォルダ
    # ========================================================

    base_dir = Path(
        __file__
    ).resolve().parent

    # ========================================================
    # TEST32件を読み込む
    # ========================================================

    x_test, y_test, test_df = load_csv_data(
        test_csv
    )

    # ========================================================
    # 今回の実験条件
    #
    # config.csvとして自動保存する
    # ========================================================

    experiment_config = {
        "experiment_id": (
            experiment_id
        ),

        "standardization": (
            standardization
        ),

        "model": (
            "5-8-4-1"
        ),

        "input_features": (
            5
        ),

        "activation": (
            "ReLU"
        ),

        "output_activation": (
            "None"
        ),

        "optimizer": (
            "Adam"
        ),

        "loss_function": (
            "MSELoss"
        ),

        "learning_rate": (
            learning_rate
        ),

        "batch_size": (
            batch_size
        ),

        "local_epochs": (
            local_epochs
        ),

        "server_rounds": (
            num_rounds
        ),

        "num_clients": (
            3
        ),

        "samples_client1": (
            35
        ),

        "samples_client2": (
            35
        ),

        "samples_client3": (
            35
        ),

        "test_samples": (
            len(x_test)
        ),

        "aggregation": (
            "FedAvg"
        ),

        "fraction_train": (
            fraction_train
        ),

        "random_seed": (
            42
        ),

        "test_data_path": (
            str(test_csv)
        ),
    }

    # ========================================================
    # 実験保存フォルダを自動作成
    #
    # config.csv
    # code_snapshot/
    # results/
    #
    # を作成する
    # ========================================================

    experiment_info = initialize_experiment(
        base_dir=base_dir,
        experiment_id=experiment_id,
        config=experiment_config,
    )

    # --------------------------------------------------------
    # この実験専用のresultsフォルダ
    # --------------------------------------------------------

    result_dir = Path(
        experiment_info[
            "result_dir"
        ]
    )

    # ========================================================
    # 実験開始表示
    # ========================================================

    print(
        "========================================"
    )

    print(
        "STEP3 DNN Federated Learning"
    )

    print(
        "========================================"
    )

    print(
        f"Experiment ID  : "
        f"{experiment_id}"
    )

    print(
        f"Standardization: "
        f"{standardization}"
    )

    print(
        f"Test samples   : "
        f"{len(x_test)}"
    )

    print(
        f"Rounds         : "
        f"{num_rounds}"
    )

    print(
        f"Local epochs   : "
        f"{local_epochs}"
    )

    print(
        f"Batch size     : "
        f"{batch_size}"
    )

    print(
        f"Learning rate  : "
        f"{learning_rate}"
    )

    print(
        f"Result dir     : "
        f"{result_dir}"
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
        / "server_round_metrics.csv"
    )

    # ========================================================
    # Global model評価関数
    #
    # Round 0
    #   学習開始前
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
                metrics[
                    "mse"
                ]
            ),

            "rmse": float(
                metrics[
                    "rmse"
                ]
            ),

            "mae": float(
                metrics[
                    "mae"
                ]
            ),

            "r2": float(
                metrics[
                    "r2"
                ]
            ),
        }

        metric_history.append(
            row
        )

        # ----------------------------------------------------
        # 毎Round保存
        #
        # 途中停止しても
        # そこまでの結果が残る
        # ----------------------------------------------------

        pd.DataFrame(
            metric_history
        ).to_csv(
            metrics_csv_path,
            index=False,
        )

        print(
            f"Global Round "
            f"{server_round:2d} | "
            f"MSE="
            f"{metrics['mse']:.8f} | "
            f"RMSE="
            f"{metrics['rmse']:.8f} | "
            f"MAE="
            f"{metrics['mae']:.8f} | "
            f"R2="
            f"{metrics['r2']:.8f}"
        )

        return MetricRecord(
            {
                "mse": float(
                    metrics[
                        "mse"
                    ]
                ),

                "rmse": float(
                    metrics[
                        "rmse"
                    ]
                ),

                "mae": float(
                    metrics[
                        "mae"
                    ]
                ),

                "r2": float(
                    metrics[
                        "r2"
                    ]
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

        fraction_evaluate=0.0,

        # 毎Round3 Clientすべて参加
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
    # 最終Global model
    # ========================================================

    final_state_dict = (
        result.arrays.to_torch_state_dict()
    )

    model_path = (
        result_dir
        / "global_model.pt"
    )

    torch.save(
        final_state_dict,
        model_path,
    )

    # ========================================================
    # Client側 Train Loss
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
        / "client_train_loss.csv"
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
    #
    # Actual CoF
    # Predicted CoF
    # Residual
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
        / "test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_csv_path,
        index=False,
    )

    # ========================================================
    # Actual CoF vs Predicted CoF
    # 散布図 + y=x
    # ========================================================

    actual = prediction_df[
        "actual_cof"
    ].to_numpy()

    predicted = prediction_df[
        "predicted_cof"
    ].to_numpy()

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
    # Scatter plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        actual,
        predicted,
        s=50,
    )

    # --------------------------------------------------------
    # 理想線 y=x
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

    plt.xlabel(
        "Actual CoF"
    )

    plt.ylabel(
        "Predicted CoF"
    )

    plt.title(
        "Actual vs Predicted CoF\n"
        f"{experiment_id}, "
        f"R² = "
        f"{final_metrics['r2']:.3f}"
    )

    plt.xlim(
        plot_min,
        plot_max,
    )

    plt.ylim(
        plot_min,
        plot_max,
    )

    plt.gca().set_aspect(
        "equal",
        adjustable="box",
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    scatter_path = (
        result_dir
        / "actual_vs_predicted.png"
    )

    plt.savefig(
        scatter_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # 最終評価指標CSV
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
        / "final_metrics.csv"
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
        f"Experiment ID: "
        f"{experiment_id}"
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
