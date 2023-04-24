import torch
import torch.nn as nn
from argparse import Namespace
import copy

from typing import Tuple, List

from Source.Backbones.utils import SparseConv2d, SparseLinear


class CNN_Small(nn.Module):
    def __init__(self, input_size: Tuple, output_size: int, args: Namespace) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        self.conv2d = nn.Conv2d if args.method != 'nispa_replay_plus'  else SparseConv2d
        self.linear = nn.Linear if args.method != 'nispa_replay_plus'  else SparseLinear

        super(CNN_Small, self).__init__()
        self.input_size = input_size
        self.conv2lin_size = 128*8*8
        self.conv2lin_mapping_size = 8*8
        self.output_size = output_size

        self.conv = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()

        # Add convolutional layers
        self.conv.append(self.conv2d(input_size[0], 64, 3, stride=1,padding=1, dilation=1, groups=1, bias=True))
        self.conv.append(nn.ReLU())
        self.conv.append(self.conv2d(64, 64, 3, stride=1,padding=1, dilation=1, groups=1, bias=True))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(2))

        self.conv.append(self.conv2d(64, 128, 3, stride=1,padding=1, dilation=1, groups=1, bias=True))
        self.conv.append(nn.ReLU())
        self.conv.append(self.conv2d(128, 128, 3, stride=1,padding=1, dilation=1, groups=1, bias=True))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(2))
        
        # Add hidden layers
        self.hidden_layers.append(self.linear(self.conv2lin_size, 2000))
        self.hidden_layers.append(nn.ReLU())

        # Add classifier
        self.classifier = self.linear(2000, self.output_size)
        self._initialize_weights()


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        for module in self.conv:
            x = module(x)

        # Flatten the output of the convolutional layers
        x = x.view(-1, self.conv2lin_size)

        for module in self.hidden_layers:
            x = module(x)
        
        feats = x.detach().clone()

        out = self.classifier(x)

        return out, feats
    

    def set_masks(self, weight_masks: List[torch.Tensor] , bias_masks: List[torch.Tensor]) -> None:
        i = 0
        for m in self.modules():
            if isinstance(m,(SparseLinear, SparseConv2d)):
                m.set_mask(weight_masks[i],bias_masks[i])
                i = i + 1


    def forward_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        # forward x and save activations after relu
        for module in self.conv:
            x = module(x)
            if isinstance(module, nn.ReLU):
                activations.append(x)
        x = x.view(-1, self.conv2lin_size)
        for module in self.hidden_layers:
            x = module(x)
            if isinstance(module, nn.ReLU):
                activations.append(x)
        
        return activations
    
    def get_weight_bias_masks_numpy(self) -> List[Tuple]:
        weights = []
        for module in self.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
                weight_mask, bias_mask = module.get_mask()  # type: ignore
                weights.append((copy.deepcopy(weight_mask).cpu().numpy(),
                                copy.deepcopy(bias_mask).cpu().numpy())) # type: ignore
        return weights