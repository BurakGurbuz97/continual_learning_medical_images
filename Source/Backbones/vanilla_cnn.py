import torch
import torch.nn as nn

from typing import Tuple


class VanillaCNN(nn.Module):
    def __init__(self, input_size: Tuple, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(VanillaCNN, self).__init__()
        self.input_size = input_size
        self.conv2lin_size = 32*9*9
        self.output_size = output_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0], 32, 3, stride=1,padding=1, dilation=1, groups=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )

        self.hidden_layers = nn.Sequential(
            nn.Linear(self.conv2lin_size, 1000),
            nn.ReLU()
        )

        self.classifier = nn.Linear(1000, self.output_size)
        self.net = nn.Sequential(self.conv, self.hidden_layers, self.classifier)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = self.conv(x)
        x = x.view(-1, self.conv2lin_size)
        feats = self.hidden_layers(x)
        out = self.classifier(feats)

        return out, feats
