import random

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from torch import nn
from torch.optim import Adam

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


FEATURE_COLUMNS = [

    "temperature_per_1000c",
    "load_per_10n",
    "log10p1_speed_per_3",
    "sliding_time_per_60min",

    "disk__40crnimoa",
    "disk__alloy_800ht",
    "disk__gh2132",
    "disk__gh4169",
    "disk__gh605",
    "disk__inco718",
    "disk__inco939_additive_as_built",
    "disk__inco939_additive_heat_treated",
    "disk__inco939_cast",
    "disk__naf10",
    "disk__naf15",
    "disk__naf5",
    "disk__rene88",
    "disk__ta_w",

    "pin__2344_h13_skd61",
    "pin__alloy_800ht",
    "pin__ha188",
    "pin__ha25",
    "pin__inco718",

    "ambient__he",

    "test_type__pin_on_disk",
    "test_type__pin_on_flat",
]


TARGET_COLUMN = "target_ave_cof"


def set_seed(
    seed: int,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def load_xy(
    csv_path,
):

    df = pd.read_csv(
        csv_path
    )


    missing_columns = [
        column
        for column in (
            FEATURE_COLUMNS
            + [TARGET_COLUMN]
        )
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            f"Missing columns: "
            f"{missing_columns}"
        )


    x = (
        df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    y = (
        df[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=np.float32
        )
        .reshape(
            -1,
            1,
        )
    )


    return (
        df,
        x,
        y,
    )


def make_train_loader(
    x,
    y,
    batch_size,
    seed,
):

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32,
    )


    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )


    dataset = TensorDataset(
        x_tensor,
        y_tensor,
    )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )


def train_model(
    model,
    train_loader,
    local_epochs,
    learning_rate,
    device,
):

    model.to(
        device
    )

    model.train()


    criterion = nn.MSELoss()


    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )


    final_epoch_loss = None


    for _ in range(
        local_epochs
    ):

        running_loss = 0.0

        sample_count = 0


        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )


            optimizer.zero_grad()


            prediction = model(
                x_batch
            )


            loss = criterion(
                prediction,
                y_batch,
            )


            loss.backward()


            optimizer.step()


            batch_n = len(
                x_batch
            )


            running_loss += (
                loss.item()
                * batch_n
            )


            sample_count += (
                batch_n
            )


        final_epoch_loss = (
            running_loss
            / sample_count
        )


    return float(
        final_epoch_loss
    )


def evaluate_model(
    model,
    x,
    y,
    device,
):

    model.to(
        device
    )

    model.eval()


    x_tensor = torch.tensor(
        x,
        dtype=torch.float32,
    ).to(
        device
    )


    with torch.no_grad():

        predictions = (
            model(
                x_tensor
            )
            .cpu()
            .numpy()
            .reshape(-1)
        )


    actual = (
        np.asarray(
            y
        )
        .reshape(-1)
    )


    mse = mean_squared_error(
        actual,
        predictions,
    )


    rmse = np.sqrt(
        mse
    )


    mae = mean_absolute_error(
        actual,
        predictions,
    )


    r2 = r2_score(
        actual,
        predictions,
    )


    metrics = {

        "mse":
            float(mse),

        "rmse":
            float(rmse),

        "mae":
            float(mae),

        "r2":
            float(r2),
    }


    return (
        metrics,
        predictions,
    )
