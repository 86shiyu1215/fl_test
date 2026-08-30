import torch
import torch.nn as nn


class CoFDNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),

            nn.Linear(8, 4),
            nn.ReLU(),

            nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# model.pyを直接実行した場合の動作確認
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    model = CoFDNN()

    print("========================================")
    print("CoF DNN model")
    print("========================================")
    print(model)
    print()

    # 例として4件×5特徴量の疑似入力を作る
    dummy_x = torch.randn(
        4,
        5,
    )

    dummy_output = model(dummy_x)

    print(
        "入力shape :",
        dummy_x.shape,
    )

    print(
        "出力shape :",
        dummy_output.shape,
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "学習パラメータ数:",
        parameter_count,
    )
