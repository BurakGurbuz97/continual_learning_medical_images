import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Tuple, Optional

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def to_var(x, requires_grad = False, volatile = False) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().to(get_device())
    else:
        x = torch.tensor(x).to(get_device())
    return Variable(x, requires_grad = requires_grad, volatile = volatile)

class SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias_flag=True, layer_name = ""):
        super(SparseLinear, self).__init__(in_features, out_features, True)
        self.bias_flag = bias_flag
        self.layer_name = layer_name
        
    def set_mask(self, weight_mask: torch.Tensor, bias_mask: torch.Tensor) -> None:
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data
        
        self.bias_mask = to_var(bias_mask, requires_grad=False)
        self.bias.data = self.bias.data * self.bias_mask.data

    def get_mask(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.weight_mask, self.bias_mask

    def forward(self, x) -> torch.Tensor:
        weight = self.weight * self.weight_mask
        bias = self.bias * self.bias_mask
        return F.linear(x, weight, bias if self.bias_flag else None)
    
class SparseConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, layer_name = ""):
        super(SparseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bias_flag = bias
        self.layer_name = layer_name

    def set_mask(self, weight_mask: torch.Tensor, bias_mask: torch.Tensor) -> None:
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data
        if self.bias_flag:
            self.bias_mask = to_var(bias_mask, requires_grad=False)
            self.bias.data = self.bias.data * self.bias_mask.data # type: ignore

    def get_mask(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.weight_mask, self.bias_mask if self.bias_flag else None

    def forward(self, x) -> torch.Tensor:
        weight = self.weight * self.weight_mask
        bias = self.bias * self.bias_mask  if self.bias_flag else self.bias
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)