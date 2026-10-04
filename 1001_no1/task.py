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
    "hardness_gpa",
    "normal_load_n",
    "sliding_velocity_m_s",
    "sliding_distance_m",
    "film_thickness_nm",
]


TARGET_COLUMN = "cof"


# ============================================================
# Seed
# ============================================================

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


# ============================================================
# Load CSV
# ============================================================

def load_xy(
    csv_path,
):

    df = pd.read_csv(
        csv_path
    )

    x = (
        df[FEATURE_COLUMNS]
        .to_numpy(
            dtype=np.float32
        )
    )

    y = (
        df[TARGET_COLUMN]
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


# ============================================================
# Z-score
# ============================================================

def apply_zscore(
    x,
    mean,
    scale,
):

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    scale = np.asarray(
        scale,
        dtype=np.float32,
    )

    safe_scale = np.where(
        scale == 0.0,
        1.0,
        scale,
    )

    return (
        x - mean
    ) / safe_scale


# ============================================================
# DataLoader
# ============================================================

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


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    return loader


# ============================================================
# Local training
# ============================================================

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


# ============================================================
# Evaluation
# ============================================================

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
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


    return (
        metrics,
        predictions,
    )
