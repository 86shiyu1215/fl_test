from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from model import CoFDNN


# ============================================================
# 基本設定
# ============================================================

RANDOM_SEED = 42

FEATURE_COLUMNS = [
    "hardness_gpa",
    "normal_load_n",
    "sliding_velocity_m_s",
    "sliding_distance_m",
    "film_thickness_nm",
]

TARGET_COLUMN = "cof"

BATCH_SIZE = 7
LOCAL_EPOCHS = 10
LEARNING_RATE = 0.001


# ============================================================
# 乱数seedを固定
# ============================================================

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CSVを読み込む
# ============================================================

def load_csv_data(csv_path):

    df = pd.read_csv(csv_path)

    x = df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    y = df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32
    )

    # [35] ではなく [35, 1] の形にする
    y = y.reshape(-1, 1)

    return x, y, df


# ============================================================
# PyTorchのDataLoaderを作る
# ============================================================

def create_dataloader(
    x,
    y,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED,
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
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )

    return dataloader


# ============================================================
# DNNを学習する
# ============================================================

def train_model(
    model,
    train_loader,
    epochs=LOCAL_EPOCHS,
    learning_rate=LEARNING_RATE,
):

    # 誤差の測り方
    criterion = torch.nn.MSELoss()

    # 重みの更新方法
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # 学習モード
    model.train()

    epoch_losses = []

    for epoch in range(epochs):

        batch_losses = []

        for x_batch, y_batch in train_loader:

            # ① 前回計算した勾配をリセット
            optimizer.zero_grad()

            # ② DNNで摩擦係数を予測
            predictions = model(x_batch)

            # ③ 実測CoFとのMSEを計算
            loss = criterion(
                predictions,
                y_batch,
            )

            # ④ 誤差逆伝播
            loss.backward()

            # ⑤ AdamでDNNの重みを更新
            optimizer.step()

            batch_losses.append(
                loss.item()
            )

        # そのepochにおける平均Loss
        epoch_loss = np.mean(
            batch_losses
        )

        epoch_losses.append(
            epoch_loss
        )

    return epoch_losses


# ============================================================
# DNNを評価する
# ============================================================

def evaluate_model(
    model,
    x,
    y,
):

    # 評価モード
    model.eval()

    x_tensor = torch.tensor(
        x,
        dtype=torch.float32,
    )

    # 評価時は重みを更新しない
    with torch.no_grad():

        predictions = model(
            x_tensor
        ).cpu().numpy()

    actual = y.reshape(-1)
    predicted = predictions.reshape(-1)

    mse = mean_squared_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    return metrics, predicted


# ============================================================
# task.py単体で動作確認
# ============================================================

if __name__ == "__main__":

    set_seed()

    base_dir = Path(
        __file__
    ).resolve().parent

    # 今回は動作確認としてClient1を使用
    train_csv = (
        base_dir
        / "data"
        / "splits"
        / "client1_train.csv"
    )

    # 共通TEST 32件
    test_csv = (
        base_dir
        / "data"
        / "splits"
        / "test.csv"
    )

    print(
        "========================================"
    )
    print(
        "STEP3 local DNN training test"
    )
    print(
        "Standardization: OFF"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # データ読み込み
    # --------------------------------------------------------

    x_train, y_train, _ = load_csv_data(
        train_csv
    )

    x_test, y_test, _ = load_csv_data(
        test_csv
    )

    print(
        f"Train samples : {len(x_train)}"
    )

    print(
        f"Test samples  : {len(x_test)}"
    )

    # --------------------------------------------------------
    # DataLoader作成
    # --------------------------------------------------------

    train_loader = create_dataloader(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(
        f"Batch size    : {BATCH_SIZE}"
    )

    print(
        f"Batches/epoch : {len(train_loader)}"
    )

    print(
        f"Local epochs  : {LOCAL_EPOCHS}"
    )

    print(
        f"Learning rate : {LEARNING_RATE}"
    )

    # --------------------------------------------------------
    # DNN作成
    # --------------------------------------------------------

    model = CoFDNN()

    print()
    print(model)

    # --------------------------------------------------------
    # 学習
    # --------------------------------------------------------

    epoch_losses = train_model(
        model,
        train_loader,
        epochs=LOCAL_EPOCHS,
        learning_rate=LEARNING_RATE,
    )

    print()
    print(
        "========================================"
    )
    print(
        "Training Loss"
    )
    print(
        "========================================"
    )

    for epoch, loss in enumerate(
        epoch_losses,
        start=1,
    ):
        print(
            f"Epoch {epoch:2d}: "
            f"MSELoss = {loss:.8f}"
        )

    # --------------------------------------------------------
    # TESTデータで評価
    # --------------------------------------------------------

    metrics, predictions = evaluate_model(
        model,
        x_test,
        y_test,
    )

    print()
    print(
        "========================================"
    )
    print(
        "Test Metrics"
    )
    print(
        "========================================"
    )

    print(
        f"MSE  : {metrics['mse']:.8f}"
    )

    print(
        f"RMSE : {metrics['rmse']:.8f}"
    )

    print(
        f"MAE  : {metrics['mae']:.8f}"
    )

    print(
        f"R2   : {metrics['r2']:.8f}"
    )
