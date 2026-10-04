import torch
from torch import nn


class CoFDNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                26,
                16,
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                8,
            ),

            nn.ReLU(),

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


if __name__ == "__main__":

    model = CoFDNN()

    print(
        model
    )


    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


    print(
        f"Trainable parameters: "
        f"{total_params}"
    )


    dummy_x = torch.randn(
        4,
        26,
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
