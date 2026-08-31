from pathlib import Path
import json

import numpy as np
import torch

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from flwr.clientapp import ClientApp


from model import CoFDNN

from task import (
    apply_zscore,
    create_dataloader,
    load_csv_data,
    set_seed,
    train_model,
)


# ============================================================
# Flower ClientApp
# ============================================================

app = ClientApp()


# ============================================================
# Client側の学習処理
# ============================================================

@app.train()
def train(
    msg: Message,
    context: Context,
) -> Message:

    # ========================================================
    # Client ID
    #
    # partition-id
    # 0 -> Client1
    # 1 -> Client2
    # 2 -> Client3
    # ========================================================

    partition_id = int(
        context.node_config[
            "partition-id"
        ]
    )

    client_id = (
        partition_id
        + 1
    )

    # ========================================================
    # Round番号
    # ========================================================

    server_round = int(
        msg.content[
            "config"
        ].get(
            "server-round",
            0,
        )
    )

    # ========================================================
    # 再現性のためseed設定
    #
    # Clientごと・Roundごとに少し変える
    # ========================================================

    seed = (
        42
        + client_id * 1000
        + server_round
    )

    set_seed(
        seed
    )

    # ========================================================
    # ClientのCSVパス
    # ========================================================

    if (
        "data-path"
        in context.node_config
    ):

        csv_path = Path(
            str(
                context.node_config[
                    "data-path"
                ]
            )
        )

    else:

        data_dir = Path(
            str(
                context.run_config[
                    "data-dir"
                ]
            )
        )

        csv_path = (
            data_dir
            / f"client{client_id}_train.csv"
        )

    # ========================================================
    # Clientデータ読み込み
    # ========================================================

    x_train, y_train, _ = (
        load_csv_data(
            csv_path
        )
    )

    # ========================================================
    # Serverから送られた学習条件
    # ========================================================

    train_config = msg.content[
        "config"
    ]

    learning_rate = float(
        train_config[
            "learning-rate"
        ]
    )

    local_epochs = int(
        train_config[
            "local-epochs"
        ]
    )

    batch_size = int(
        train_config[
            "batch-size"
        ]
    )

    # ========================================================
    # 標準化方式
    # ========================================================

    standardization = str(
        train_config.get(
            "standardization",
            "none",
        )
    )

    # ========================================================
    # Z-score標準化
    #
    # ServerがTRAIN105件から求めた
    # 共通mean / scaleを全Clientで使用する
    # ========================================================

    if standardization == "zscore":

        scaler_mean = np.asarray(
            json.loads(
                str(
                    train_config[
                        "zscore-mean-json"
                    ]
                )
            ),
            dtype=np.float32,
        )

        scaler_scale = np.asarray(
            json.loads(
                str(
                    train_config[
                        "zscore-scale-json"
                    ]
                )
            ),
            dtype=np.float32,
        )

        x_train = apply_zscore(
            x_train,
            scaler_mean,
            scaler_scale,
        )

    elif standardization != "none":

        raise ValueError(
            f"Unknown standardization: "
            f"{standardization}"
        )

    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = create_dataloader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    # ========================================================
    # Global modelを受け取る
    # ========================================================

    model = CoFDNN()

    arrays = msg.content[
        "arrays"
    ]

    state_dict = (
        arrays.to_torch_state_dict()
    )

    model.load_state_dict(
        state_dict
    )

    # ========================================================
    # Local training
    # ========================================================

    epoch_losses = train_model(
        model,
        train_loader,
        epochs=local_epochs,
        learning_rate=learning_rate,
    )

    final_train_loss = float(
        epoch_losses[-1]
    )

    print(
        f"Client {client_id} | "
        f"Round {server_round} | "
        f"Train Loss = "
        f"{final_train_loss:.8f}"
    )

    # ========================================================
    # 学習後の重み
    # ========================================================

    model_record = ArrayRecord(
        model.state_dict()
    )

    # ========================================================
    # Serverへ返すMetric
    # ========================================================

    metrics = MetricRecord(
        {
            "train_loss": (
                final_train_loss
            ),
            "num-examples": (
                len(x_train)
            ),
        }
    )

    # ========================================================
    # Serverへ返す内容
    # ========================================================

    content = RecordDict(
        {
            "arrays": (
                model_record
            ),
            "metrics": (
                metrics
            ),
        }
    )

    return Message(
        content=content,
        reply_to=msg,
    )
