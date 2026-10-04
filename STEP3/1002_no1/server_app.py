from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from flwr.app import (
    ArrayRecord,
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

from experiment_utils import (
    create_experiment,
)

from model import CoFDNN

from task import (
    evaluate_model,
    load_xy,
    set_seed,
)


app = ServerApp()


@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:

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


    batch_size = int(
        context.run_config[
            "batch-size"
        ]
    )


    local_epochs = int(
        context.run_config[
            "local-epochs"
        ]
    )


    seed = int(
        context.run_config[
            "seed"
        ]
    )


    num_clients = int(
        context.run_config[
            "num-clients"
        ]
    )


    data_dir = Path(
        str(
            context.run_config[
                "data-dir"
            ]
        )
    ).resolve()


    test_csv = Path(
        str(
            context.run_config[
                "test-data-path"
            ]
        )
    ).resolve()


    base_dir = (
        test_csv.parents[2]
    )


    set_seed(
        seed
    )


    # ========================================================
    # Client sample sizes
    # ========================================================

    client_sizes = []


    for client_id in range(
        1,
        num_clients + 1,
    ):

        client_df = pd.read_csv(
            data_dir
            / f"client{client_id}_train.csv"
        )


        client_sizes.append(
            len(client_df)
        )


    # ========================================================
    # Test
    # ========================================================

    test_df, x_test, y_test = load_xy(
        test_csv
    )


    # ========================================================
    # Experiment folder
    # ========================================================

    experiment_config = {

        "dataset":
            "collaboration_preprocessed",

        "total_samples":
            92,

        "train_samples":
            sum(client_sizes),

        "test_samples":
            len(test_df),

        "num_clients":
            num_clients,

        "client1_samples":
            client_sizes[0],

        "client2_samples":
            client_sizes[1],

        "num_features":
            26,

        "model":
            "26-16-8-1",

        "activation":
            "ReLU",

        "optimizer":
            "Adam",

        "loss":
            "MSELoss",

        "learning_rate":
            learning_rate,

        "batch_size":
            batch_size,

        "local_epochs":
            local_epochs,

        "server_rounds":
            num_rounds,

        "aggregation":
            "FedAvg",

        "additional_standardization":
            "none",

        "preprocessing":
            "NIMS-FL preprocessed dataset",

        "seed":
            seed,

        "test_data_path":
            str(
                test_csv
            ),
    }


    (
        experiment_id,
        experiment_dir,
        results_dir,
    ) = create_experiment(
        base_dir=base_dir,
        config=experiment_config,
    )


    print(
        "========================================"
    )

    print(
        f"Experiment: {experiment_id}"
    )

    print(
        f"Experiment directory: "
        f"{experiment_dir}"
    )

    print(
        "========================================"
    )


    # ========================================================
    # Initial model
    # ========================================================

    set_seed(
        seed
    )


    global_model = CoFDNN()


    initial_arrays = ArrayRecord(
        global_model.state_dict()
    )


    server_history = []


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ========================================================
    # Global evaluation
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
            model=model,
            x=x_test,
            y=y_test,
            device=device,
        )


        server_history.append(
            {
                "round":
                    server_round,

                "mse":
                    metrics["mse"],

                "rmse":
                    metrics["rmse"],

                "mae":
                    metrics["mae"],

                "r2":
                    metrics["r2"],
            }
        )


        print(
            f"Round {server_round:02d} | "
            f"MSE={metrics['mse']:.8f} | "
            f"RMSE={metrics['rmse']:.8f} | "
            f"MAE={metrics['mae']:.8f} | "
            f"R2={metrics['r2']:.8f}"
        )


        return MetricRecord(
            {
                "mse":
                    metrics["mse"],

                "rmse":
                    metrics["rmse"],

                "mae":
                    metrics["mae"],

                "r2":
                    metrics["r2"],
            }
        )


    # ========================================================
    # FedAvg
    # ========================================================

    strategy = FedAvg(

        fraction_train=1.0,

        fraction_evaluate=0.0,

        min_train_nodes=2,

        min_available_nodes=2,

        weighted_by_key="num-examples",
    )


    result = strategy.start(

        grid=grid,

        initial_arrays=initial_arrays,

        num_rounds=num_rounds,

        evaluate_fn=global_evaluate,
    )


    # ========================================================
    # Save round metrics
    # ========================================================

    pd.DataFrame(
        server_history
    ).to_csv(
        results_dir
        / "server_round_metrics.csv",
        index=False,
    )


    # ========================================================
    # Train loss
    # ========================================================

    train_loss_rows = []


    for server_round, metrics in sorted(
        result.train_metrics_clientapp.items()
    ):

        train_loss_rows.append(
            {
                "round":
                    server_round,

                "train_loss":
                    metrics.get(
                        "train_loss",
                        np.nan,
                    ),
            }
        )


    pd.DataFrame(
        train_loss_rows
    ).to_csv(
        results_dir
        / "client_train_loss.csv",
        index=False,
    )


    # ========================================================
    # Final model
    # ========================================================

    final_state_dict = (
        result.arrays
        .to_torch_state_dict()
    )


    torch.save(
        final_state_dict,
        results_dir
        / "global_model.pt",
    )


    final_model = CoFDNN()


    final_model.load_state_dict(
        final_state_dict
    )


    final_metrics, predictions = evaluate_model(

        model=final_model,

        x=x_test,

        y=y_test,

        device=device,
    )


    pd.DataFrame(
        [
            {
                "round":
                    num_rounds,

                "mse":
                    final_metrics[
                        "mse"
                    ],

                "rmse":
                    final_metrics[
                        "rmse"
                    ],

                "mae":
                    final_metrics[
                        "mae"
                    ],

                "r2":
                    final_metrics[
                        "r2"
                    ],
            }
        ]
    ).to_csv(
        results_dir
        / "final_metrics.csv",
        index=False,
    )


    # ========================================================
    # Predictions
    # ========================================================

    prediction_df = pd.DataFrame(
        {
            "actual_cof":
                y_test.reshape(-1),

            "predicted_cof":
                predictions,
        }
    )


    if "source_row" in test_df.columns:

        prediction_df.insert(
            0,
            "source_row",
            test_df[
                "source_row"
            ].to_numpy(),
        )


    if "reference_group" in test_df.columns:

        prediction_df.insert(
            1,
            "reference_group",
            test_df[
                "reference_group"
            ].to_numpy(),
        )


    prediction_df[
        "residual"
    ] = (
        prediction_df[
            "actual_cof"
        ]
        - prediction_df[
            "predicted_cof"
        ]
    )


    prediction_df.to_csv(
        results_dir
        / "test_predictions.csv",
        index=False,
    )


    # ========================================================
    # Scatter
    # ========================================================

    plt.figure(
        figsize=(7, 7)
    )


    plt.scatter(
        prediction_df[
            "actual_cof"
        ],
        prediction_df[
            "predicted_cof"
        ],
    )


    min_value = min(
        prediction_df[
            "actual_cof"
        ].min(),
        prediction_df[
            "predicted_cof"
        ].min(),
    )


    max_value = max(
        prediction_df[
            "actual_cof"
        ].max(),
        prediction_df[
            "predicted_cof"
        ].max(),
    )


    plt.plot(
        [
            min_value,
            max_value,
        ],
        [
            min_value,
            max_value,
        ],
        linestyle="--",
    )


    plt.xlabel(
        "Actual CoF"
    )


    plt.ylabel(
        "Predicted CoF"
    )


    plt.title(
        f"{experiment_id} | "
        f"R² = {final_metrics['r2']:.3f}"
    )


    plt.tight_layout()


    plt.savefig(
        results_dir
        / "actual_vs_predicted.png",
        dpi=300,
    )


    plt.close()


    print(
        ""
    )

    print(
        "========================================"
    )

    print(
        "Federated Learning completed"
    )

    print(
        f"Experiment : {experiment_id}"
    )

    print(
        f"MSE  : {final_metrics['mse']:.8f}"
    )

    print(
        f"RMSE : {final_metrics['rmse']:.8f}"
    )

    print(
        f"MAE  : {final_metrics['mae']:.8f}"
    )

    print(
        f"R2   : {final_metrics['r2']:.8f}"
    )

    print(
        f"Results: {results_dir}"
    )

    print(
        "========================================"
    )
