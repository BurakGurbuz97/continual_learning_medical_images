import torch
import torch.nn as nn

from typing import Tuple


class VanillaMLP(nn.Module):
    def __init__(self, input_size: Tuple, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(VanillaMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size[0] * input_size[1] * input_size[2] , 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU()
        )

        self.classifier = nn.Linear(1000, self.output_size)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, self.input_size[0] * self.input_size[1] * self.input_size[2])
        feats = self.hidden_layers(x)
        out = self.classifier(feats)

        return out, feats