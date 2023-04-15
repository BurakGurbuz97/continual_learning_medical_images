import torch
import torch.nn as nn

from typing import Tuple, List
import copy
from argparse import Namespace

from Source.Backbones.utils import SparseConv2d, SparseLinear

class VanillaMLP(nn.Module):
    def __init__(self, input_size: Tuple, output_size: int, args: Namespace) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(VanillaMLP, self).__init__()
        self.linear = nn.Linear if args.method != 'nispa_replay_plus'  else SparseLinear

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = nn.ModuleList()

        # Add hidden layers
        self.hidden_layers.append(self.linear(input_size[0] * input_size[1] * input_size[2] , 1000))
        self.hidden_layers.append(nn.ReLU())
        self.hidden_layers.append(self.linear(1000, 1000))
        self.hidden_layers.append(nn.ReLU())

        self.classifier = self.linear(1000, self.output_size)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, self.input_size[0] * self.input_size[1] * self.input_size[2])
        for module in self.hidden_layers:
            x = module(x)
        feats = x.detach().clone()
        out = self.classifier(x)

        return out, feats
    
    def forward_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        activations = []
        # forward x and save activations after relu
        x = x.view(-1, self.input_size[0] * self.input_size[1] * self.input_size[2])
        for module in self.hidden_layers:
            x = module(x)
            if isinstance(module, nn.ReLU):
                activations.append(x)
        
        return activations
    

    def set_masks(self, weight_masks: List[torch.Tensor] , bias_masks: List[torch.Tensor]) -> None:
        i = 0
        for m in self.modules():
            if isinstance(m,(SparseLinear, SparseConv2d)):
                m.set_mask(weight_masks[i],bias_masks[i])
                i = i + 1


    def get_weight_bias_masks_numpy(self) -> List[Tuple]:
        weights = []
        for module in self.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
                weight_mask, bias_mask = module.get_mask()  # type: ignore
                weights.append((copy.deepcopy(weight_mask).cpu().numpy(),
                                copy.deepcopy(bias_mask).cpu().numpy())) # type: ignore
        return weights