from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

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

from flwr.serverapp.strategy import (
    FedAvg,
)


from model import CoFDNN

from task import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    apply_zscore,
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

    # ========================================================
    # Seed
    # ========================================================

    set_seed()

    # ========================================================
    # pyproject.tomlから条件取得
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

    standardization = str(
        context.run_config.get(
            "standardization",
            "none",
        )
    )

    # ========================================================
    # STEP3本体のディレクトリ
    #
    # test.csv
    #   parents[0] = splits
    #   parents[1] = data
    #   parents[2] = STEP3
    # ========================================================

    base_dir = (
        test_csv.parents[2]
    )

    # ========================================================
    # TEST32件
    # ========================================================

    x_test, y_test, test_df = (
        load_csv_data(
            test_csv
        )
    )

    # ========================================================
    # Z-score用パラメータ
    # ========================================================

    scaler_mean = None
    scaler_scale = None

    # ========================================================
    # Z-score
    #
    # Client1～3のTRAIN105件だけから
    # mean / scaleを計算
    #
    # TEST32は計算に使わない
    # ========================================================

    if standardization == "zscore":

        split_dir = (
            test_csv.parent
        )

        train_feature_arrays = []

        for client_id in range(
            1,
            4,
        ):

            client_csv = (
                split_dir
                / f"client{client_id}_train.csv"
            )

            client_df = pd.read_csv(
                client_csv
            )

            client_x = (
                client_df[
                    FEATURE_COLUMNS
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            train_feature_arrays.append(
                client_x
            )

        # ----------------------------------------------------
        # 35 + 35 + 35 = 105件
        # ----------------------------------------------------

        x_train_all = np.vstack(
            train_feature_arrays
        )

        # ----------------------------------------------------
        # TRAIN105件の平均
        # ----------------------------------------------------

        scaler_mean = (
            x_train_all.mean(
                axis=0
            )
        )

        # ----------------------------------------------------
        # TRAIN105件の標準偏差
        #
        # ddof=0
        # sklearn StandardScalerと同じ定義
        # ----------------------------------------------------

        scaler_scale = (
            x_train_all.std(
                axis=0,
                ddof=0,
            )
        )

        if np.any(
            scaler_scale == 0
        ):

            raise ValueError(
                "Standard deviation is zero "
                "for at least one feature."
            )

        # ----------------------------------------------------
        # TEST32件にも同じmean/scaleを適用
        # ----------------------------------------------------

        x_test = apply_zscore(
            x_test,
            scaler_mean,
            scaler_scale,
        )

    elif standardization != "none":

        raise ValueError(
            f"Unknown standardization: "
            f"{standardization}"
        )

    # ========================================================
    # 実験条件
    # ========================================================

    experiment_config = {

        "standardization": (
            standardization
        ),

        "scaler_fit_scope": (
            "train_105_only"
            if standardization == "zscore"
            else "none"
        ),

        "scaler_ddof": (
            0
            if standardization == "zscore"
            else "none"
        ),

        "target_standardization": (
            "none"
        ),

        "model": (
            "5-16-8-1"
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
    # exp_001, exp_002...
    # 自動採番
    # ========================================================

    experiment_info = (
        initialize_experiment(
            base_dir=base_dir,
            config=experiment_config,
        )
    )

    experiment_id = str(
        experiment_info[
            "experiment_id"
        ]
    )

    result_dir = Path(
        experiment_info[
            "result_dir"
        ]
    )

    # ========================================================
    # Z-scoreパラメータ保存
    # ========================================================

    if standardization == "zscore":

        scaler_df = pd.DataFrame(
            {
                "feature": (
                    FEATURE_COLUMNS
                ),
                "mean": (
                    scaler_mean
                ),
                "scale": (
                    scaler_scale
                ),
            }
        )

        scaler_csv_path = (
            result_dir
            / "scaler_parameters.csv"
        )

        scaler_df.to_csv(
            scaler_csv_path,
            index=False,
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
        f"Experiment ID   : "
        f"{experiment_id}"
    )

    print(
        f"Standardization : "
        f"{standardization}"
    )

    print(
        f"Test samples    : "
        f"{len(x_test)}"
    )

    print(
        f"Rounds          : "
        f"{num_rounds}"
    )

    print(
        f"Local epochs    : "
        f"{local_epochs}"
    )

    print(
        f"Batch size      : "
        f"{batch_size}"
    )

    print(
        f"Learning rate   : "
        f"{learning_rate}"
    )

    print(
        f"Result dir      : "
        f"{result_dir}"
    )

    print()

    # ========================================================
    # Global model
    # ========================================================

    global_model = (
        CoFDNN()
    )

    initial_arrays = (
        ArrayRecord(
            global_model.state_dict()
        )
    )

    # ========================================================
    # Global評価履歴
    # ========================================================

    metric_history = []

    metrics_csv_path = (
        result_dir
        / "server_round_metrics.csv"
    )

    # ========================================================
    # Global評価
    # ========================================================

    def global_evaluate(
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord:

        model = CoFDNN()

        model.load_state_dict(
            arrays.to_torch_state_dict()
        )

        metrics, _ = (
            evaluate_model(
                model,
                x_test,
                y_test,
            )
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
        # Roundごとに保存
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

        min_train_nodes=3,

        min_available_nodes=3,
    )

    # ========================================================
    # Clientへ送る設定
    # ========================================================

    train_config_dict = {

        "learning-rate": (
            learning_rate
        ),

        "local-epochs": (
            local_epochs
        ),

        "batch-size": (
            batch_size
        ),

        "standardization": (
            standardization
        ),
    }

    # ========================================================
    # Z-scoreの場合
    #
    # 全Clientへ同じmean / scaleを送信
    # ========================================================

    if standardization == "zscore":

        train_config_dict[
            "zscore-mean-json"
        ] = json.dumps(
            scaler_mean.tolist()
        )

        train_config_dict[
            "zscore-scale-json"
        ] = json.dumps(
            scaler_scale.tolist()
        )

    train_config = ConfigRecord(
        train_config_dict
    )

    # ========================================================
    # Federated Learning開始
    # ========================================================

    result = strategy.start(

        grid=grid,

        initial_arrays=(
            initial_arrays
        ),

        train_config=(
            train_config
        ),

        num_rounds=(
            num_rounds
        ),

        evaluate_fn=(
            global_evaluate
        ),
    )

    # ========================================================
    # 最終Global model
    # ========================================================

    final_state_dict = (
        result.arrays
        .to_torch_state_dict()
    )

    # ========================================================
    # Global model保存
    # ========================================================

    model_path = (
        result_dir
        / "global_model.pt"
    )

    torch.save(
        final_state_dict,
        model_path,
    )

    # ========================================================
    # Client train loss
    # ========================================================

    train_loss_rows = []

    for (
        server_round,
        metrics,
    ) in sorted(
        result
        .train_metrics_clientapp
        .items()
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
    # 最終Global model評価
    # ========================================================

    final_model = (
        CoFDNN()
    )

    final_model.load_state_dict(
        final_state_dict
    )

    (
        final_metrics,
        final_predictions,
    ) = evaluate_model(
        final_model,
        x_test,
        y_test,
    )

    # ========================================================
    # TEST予測結果
    # ========================================================

    prediction_df = (
        pd.DataFrame(
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
    )

    # ========================================================
    # 残差
    # ========================================================

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
    # Actual vs Predicted
    # ========================================================

    actual = (
        prediction_df[
            "actual_cof"
        ].to_numpy()
    )

    predicted = (
        prediction_df[
            "predicted_cof"
        ].to_numpy()
    )

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

    plot_min -= margin
    plot_max += margin

    # ========================================================
    # Scatter
    # ========================================================

    plt.figure(
        figsize=(6, 6)
    )

    plt.scatter(
        actual,
        predicted,
        s=50,
    )

    # ========================================================
    # y=x
    # ========================================================

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
        f"{standardization}, "
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
    # 最終評価CSV
    # ========================================================

    final_metrics_df = (
        pd.DataFrame(
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

    print(
        f"Standardization: "
        f"{standardization}"
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

    if standardization == "zscore":

        print(
            scaler_csv_path
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
