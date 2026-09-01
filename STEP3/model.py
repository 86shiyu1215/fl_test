import torch
from torch import nn


# ============================================================
# CoF prediction DNN
#
# Input:
#   5 features
#
# Hidden layers:
#   16
#   8
#
# Output:
#   1 CoF
#
# Architecture:
#   5 -> 16 -> 8 -> 1
# ============================================================

class CoFDNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # ------------------------------------------------
            # Input 5 -> Hidden 16
            # ------------------------------------------------

            nn.Linear(
                5,
                16,
            ),

            nn.ReLU(),

            # ------------------------------------------------
            # Hidden 16 -> Hidden 8
            # ------------------------------------------------

            nn.Linear(
                16,
                8,
            ),

            nn.ReLU(),

            # ------------------------------------------------
            # Hidden 8 -> Output 1
            #
            # CoF回帰なので
            # 出力層には活性化関数を入れない
            # ------------------------------------------------

            nn.Linear(
                8,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# 単体確認
# ============================================================

if __name__ == "__main__":

    model = CoFDNN()

    print(model)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{total_params}"
    )

    dummy_x = torch.randn(
        4,
        5,
    )

    dummy_y = model(
        dummy_x
    )

    print(
        f"Input shape : "
        f"{dummy_x.shape}"
    )

    print(
        f"Output shape: "
        f"{dummy_y.shape}"
    )
