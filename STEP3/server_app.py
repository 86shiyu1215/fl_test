from pathlib import Path

import pandas as pd
import torch

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
    # この32件は学習には一切使用しない。
    # 各Round後のGlobal model評価だけに使う。
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

    # --------------------------------------------------------
    # Global DNN初期化
    # --------------------------------------------------------

    global_model = CoFDNN()

    initial_arrays = ArrayRecord(
        global_model.state_dict()
    )

    # --------------------------------------------------------
    # 評価指標履歴
    #
    # Roundごとに
    # MSE / RMSE / MAE / R2
    # をここへ蓄積する。
    # --------------------------------------------------------

    metric_history = []

    metrics_csv_path = (
        result_dir
        / "server_round_metrics_no_scaling.csv"
    )

    # --------------------------------------------------------
    # Global model評価関数
    #
    # Round 0:
    #   FL開始前の初期DNN
    #
    # Round 1～30:
    #   各FedAvg後のGlobal DNN
    # --------------------------------------------------------

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
            "round": int(server_round),
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

        metric_history.append(row)

        # ----------------------------------------------------
        # 毎Round CSVを更新
        #
        # 途中で実験が止まっても、
        # そこまでの結果を残せる。
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

    # --------------------------------------------------------
    # FedAvg
    # --------------------------------------------------------

    strategy = FedAvg(
        fraction_train=fraction_train,

        # Client側の評価は今回は行わない。
        # 共通TEST32件をServer側で毎Round評価する。
        fraction_evaluate=0.0,

        # 毎Round必ず3 Client全て参加
        min_train_nodes=3,
        min_available_nodes=3,
    )

    # --------------------------------------------------------
    # Clientへ送る学習条件
    # --------------------------------------------------------

    train_config = ConfigRecord(
        {
            "learning-rate": learning_rate,
            "local-epochs": local_epochs,
            "batch-size": batch_size,
        }
    )

    # --------------------------------------------------------
    # Federated Learning開始
    # --------------------------------------------------------

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # --------------------------------------------------------
    # 最終Global model保存
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RoundごとのClient学習LossもCSVへ保存
    #
    # Flower/FedAvgが3 Clientのtrain_lossを
    # num-examplesで重み付き平均した値。
    #
    # 今回は35・35・35なので単純平均と同じ。
    # --------------------------------------------------------

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
                    metrics["train_loss"]
                ),
            }
        )

    pd.DataFrame(
        train_loss_rows
    ).to_csv(
        result_dir
        / "client_train_loss_no_scaling.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 最終Global modelをTEST32件で評価
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 実測CoF vs 予測CoF
    #
    # 将来R2散布図を作るためのCSV
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "data_id": test_df["data_id"],
            "actual_cof": test_df[
                TARGET_COLUMN
            ],
            "predicted_cof": (
                final_predictions
            ),
        }
    )

    prediction_df["residual"] = (
        prediction_df["actual_cof"]
        - prediction_df["predicted_cof"]
    )

    prediction_df.to_csv(
        result_dir
        / "test_predictions_no_scaling.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 最終評価指標
    # --------------------------------------------------------

    final_metrics_df = pd.DataFrame(
        [
            {
                "round": num_rounds,
                "mse": float(
                    final_metrics["mse"]
                ),
                "rmse": float(
                    final_metrics["rmse"]
                ),
                "mae": float(
                    final_metrics["mae"]
                ),
                "r2": float(
                    final_metrics["r2"]
                ),
            }
        ]
    )

    final_metrics_df.to_csv(
        result_dir
        / "final_metrics_no_scaling.csv",
        index=False,
    )

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

    print(
        "Saved:"
    )

    print(
        metrics_csv_path
    )

    print(
        result_dir
        / "client_train_loss_no_scaling.csv"
    )

    print(
        result_dir
        / "test_predictions_no_scaling.csv"
    )

    print(
        result_dir
        / "final_metrics_no_scaling.csv"
    )

    print(
        model_path
    )
