from pathlib import Path

import torch

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from flwr.clientapp import ClientApp

from exp1005_no1_model import CoFDNN

from exp1005_no1_task import (
    load_xy,
    make_train_loader,
    set_seed,
    train_model,
)


app = ClientApp()


@app.train()
def train(
    msg: Message,
    context: Context,
) -> Message:

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


    data_dir = Path(
        str(
            context.run_config[
                "data-dir"
            ]
        )
    )


    partition_id = int(
        context.node_config[
            "partition-id"
        ]
    )


    client_id = (
        partition_id
        + 1
    )


    client_csv = (
        data_dir
        / f"client{client_id}_train.csv"
    )


    set_seed(
        seed
    )


    model = CoFDNN()


    model.load_state_dict(
        msg.content[
            "arrays"
        ].to_torch_state_dict()
    )


    _, x_train, y_train = load_xy(
        client_csv
    )


    train_loader = make_train_loader(
        x_train,
        y_train,
        batch_size=batch_size,
        seed=seed,
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    train_loss = train_model(
        model=model,
        train_loader=train_loader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )


    print(
        f"Client {client_id} | "
        f"samples={len(y_train)} | "
        f"train_loss={train_loss:.8f}"
    )


    model_record = ArrayRecord(
        model.state_dict()
    )


    metric_record = MetricRecord(
        {
            "train_loss":
                float(
                    train_loss
                ),

            "num-examples":
                int(
                    len(y_train)
                ),
        }
    )


    content = RecordDict(
        {
            "arrays":
                model_record,

            "metrics":
                metric_record,
        }
    )


    return Message(
        content=content,
        reply_to=msg,
    )
