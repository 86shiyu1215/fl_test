import json
import numpy as np

from pathlib import Path

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
    RANDOM_SEED,
    create_dataloader,
    load_csv_data,
    set_seed,
    train_model,
    apply_zscore,
)


# ============================================================
# Flower ClientApp
# ============================================================

app = ClientApp()


# ============================================================
# Clientが使用するCSVを決める
# ============================================================

def get_train_csv(context: Context) -> Path:
    """
    Standalone simulation:
        partition-id 0 -> client1_train.csv
        partition-id 1 -> client2_train.csv
        partition-id 2 -> client3_train.csv

    Future deployment:
        各研究拠点で node-config に data-path を渡せば、
        その拠点のローカルCSVだけを読み込める。
    """

    # 将来の実機分散用
    if "data-path" in context.node_config:
        return Path(
            str(context.node_config["data-path"])
        )

    # 今回のB上でのSimulation用
    partition_id = int(
        context.node_config["partition-id"]
    )

    client_id = partition_id + 1

    data_dir = Path(
        str(context.run_config["data-dir"])
    )

    return (
        data_dir
        / f"client{client_id}_train.csv"
    )


# ============================================================
# ローカル学習
# ============================================================

@app.train()
def train(
    msg: Message,
    context: Context,
):
    """
    ServerからGlobal modelを受信し、
    各Clientの35件で10 epoch学習して、
    更新したモデル重みをServerへ返す。
    """

    # --------------------------------------------------------
    # Client ID
    # --------------------------------------------------------

    partition_id = int(
        context.node_config["partition-id"]
    )

    client_id = partition_id + 1

    # --------------------------------------------------------
    # Serverから送られた学習条件
    # --------------------------------------------------------

    config = msg.content["config"]

    learning_rate = float(
        config["learning-rate"]
    )

    local_epochs = int(
        config["local-epochs"]
    )

    batch_size = int(
        config["batch-size"]
    )

    server_round = int(
        config["server-round"]
    )

    # --------------------------------------------------------
    # 再現性のためseedを固定
    #
    # Client・Roundごとに少し変えることで
    # batchの並びは毎Round変化するが、
    # 同じ実験をすれば再現できる。
    # --------------------------------------------------------

    client_seed = (
        RANDOM_SEED
        + client_id * 100
        + server_round
    )

    set_seed(client_seed)

    # --------------------------------------------------------
    # 各Clientの35件を読み込む
    # --------------------------------------------------------

    train_csv = get_train_csv(context)

    x_train, y_train, _ = load_csv_data(
        train_csv
    )
    # ============================================================
# 標準化方式
# ============================================================

standardization = str(
    train_config.get(
        "standardization",
        "none",
    )
)

# ============================================================
# Z-score標準化
#
# Serverから受け取った
# 共通mean / scaleを全Clientで使用する
# ============================================================

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

    # --------------------------------------------------------
    # 7件 × 5 batch
    # --------------------------------------------------------

    train_loader = create_dataloader(
        x_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=client_seed,
    )

    # --------------------------------------------------------
    # DNNを作成
    # --------------------------------------------------------

    model = CoFDNN()

    # Serverから受け取ったGlobal modelの重みをセット
    model.load_state_dict(
        msg.content[
            "arrays"
        ].to_torch_state_dict()
    )

    # --------------------------------------------------------
    # Local training
    # --------------------------------------------------------

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
        f"Train Loss = {final_train_loss:.8f}"
    )

    # --------------------------------------------------------
    # 学習後のDNN重み
    # --------------------------------------------------------

    model_record = ArrayRecord(
        model.state_dict()
    )

    # --------------------------------------------------------
    # Serverへ返す情報
    #
    # num-examples=35をFedAvgの重みに使う。
    # 今回は全Client35件なので実質等重み。
    # --------------------------------------------------------

    metrics = MetricRecord(
        {
            "train_loss": final_train_loss,
            "num-examples": len(x_train),
        }
    )

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metrics,
        }
    )

    return Message(
        content=content,
        reply_to=msg,
    )
