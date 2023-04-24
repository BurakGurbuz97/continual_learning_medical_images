import torch
import torch.nn as nn
from argparse import Namespace
import copy

from typing import Tuple, List

from Source.Backbones.utils import SparseConv2d, SparseLinear

from .vanilla_cnn import VanillaCNN


class VGG11(VanillaCNN):
    def __init__(self, input_size: Tuple, output_size: int, args: Namespace) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        self.conv2d = nn.Conv2d if args.method != 'nispa_replay_plus'  else SparseConv2d
        self.linear = nn.Linear if args.method != 'nispa_replay_plus'  else SparseLinear

        super(VGG11, self).__init__(input_size, output_size, args)
        self.input_size = input_size

        if input_size[2] == 28:
            padding = 3
        else:
            padding = 1
        self.conv2lin_size = 512
        self.output_size = output_size

        self.conv = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()

        # Add convolutional layers
        self.conv.append(self.conv2d(input_size[0], 64, 3,padding=padding))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))
        
        self.conv.append(self.conv2d(64, 128, 3, padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))

        self.conv.append(self.conv2d(128, 256, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(self.conv2d(256, 256, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))

        self.conv.append(self.conv2d(256, 512, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(self.conv2d(512, 512, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))

        self.conv.append(self.conv2d(512, 512, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(self.conv2d(512, 512, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))
        # self.conv.append(nn.AdaptiveAvgPool2d(output_size=(7,7)))

        # Add hidden layers
        self.hidden_layers.append(self.linear(512, 4096))
        self.hidden_layers.append(nn.ReLU())
        self.hidden_layers.append(nn.Dropout(0.5))
        self.hidden_layers.append(self.linear(4096, 4096))
        self.hidden_layers.append(nn.ReLU())
        self.hidden_layers.append(nn.Dropout(0.5))

        # Add classifier
        self.classifier = self.linear(4096, self.output_size)
        self.net = nn.Sequential(self.conv, self.hidden_layers, self.classifier)



    def forward(self, x: torch.Tensor) :
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        for module in self.conv:
            x = module(x)

        # Flatten the output of the convolutional layers
        # x = x.view(-1, self.conv2lin_size)
        x = x.flatten(start_dim=1)

        for module in self.hidden_layers:
            x = module(x)
        
        feats = x.detach().clone()

        out = self.classifier(x)

        return out, feats



class VGG11_head(VanillaCNN):
    def __init__(self, input_size: Tuple, output_size: int, args: Namespace) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        self.conv2d = nn.Conv2d if args.method != 'nispa_replay_plus'  else SparseConv2d
        self.linear = nn.Linear if args.method != 'nispa_replay_plus'  else SparseLinear

        super(VGG11_head, self).__init__(input_size, output_size, args)
        self.input_size = input_size

        if input_size[2] == 28:
            padding = 3
        else:
            padding = 1
        self.conv2lin_size = 512
        self.output_size = output_size

        self.conv = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()

        # Add convolutional layers
        self.conv.append(self.conv2d(input_size[0], 64, 3,padding=padding))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))
        
        self.conv.append(self.conv2d(64, 128, 3, padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))

        self.conv.append(self.conv2d(128, 256, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(self.conv2d(256, 256, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))

        self.conv.append(self.conv2d(256, 512, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(self.conv2d(512, 512, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))


    def forward(self, x: torch.Tensor) :
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        for module in self.conv:
            x = module(x)

        return x, None



class VGG11_tail(VanillaCNN):
    def __init__(self, input_size: Tuple, output_size: int, args: Namespace) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        self.conv2d = nn.Conv2d if args.method != 'nispa_replay_plus'  else SparseConv2d
        self.linear = nn.Linear if args.method != 'nispa_replay_plus'  else SparseLinear

        super(VGG11_tail, self).__init__(input_size, output_size, args)
        self.input_size = input_size

        if input_size[2] == 28:
            padding = 3
        else:
            padding = 1
        self.conv2lin_size = 512
        self.output_size = output_size

        self.conv = nn.ModuleList()
        self.hidden_layers = nn.ModuleList()

        # Add convolutional layers
        self.conv.append(self.conv2d(512, 512, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(self.conv2d(512, 512, 3,padding=1))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool2d(kernel_size= 2, stride=2))
        # self.conv.append(nn.AdaptiveAvgPool2d(output_size=(7,7)))

        # Add hidden layers
        self.hidden_layers.append(self.linear(512, 4096))
        self.hidden_layers.append(nn.ReLU())
        self.hidden_layers.append(nn.Dropout(0.5))
        self.hidden_layers.append(self.linear(4096, 4096))
        self.hidden_layers.append(nn.ReLU())
        self.hidden_layers.append(nn.Dropout(0.5))

        # Add classifier
        self.classifier = self.linear(4096, self.output_size)
        self.net = nn.Sequential(self.conv, self.hidden_layers, self.classifier)

    def forward(self, x: torch.Tensor) :
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        for module in self.conv:
            x = module(x)

        # Flatten the output of the convolutional layers
        # x = x.view(-1, self.conv2lin_size)
        x = x.flatten(start_dim=1)

        for module in self.hidden_layers:
            x = module(x)
        
        feats = x.detach().clone()

        out = self.classifier(x)

        return out, feats


class vgg11_wrapper(nn.Module):
    def __init__(self, input_size, output_size, args):
        super(vgg11_wrapper, self).__init__()
        self.head = VGG11_head(input_size, output_size, args) 
        self.tail = VGG11_tail(input_size, output_size, args) 

    def forward(self, images):
        out , _ = self.head(images)
        out , feats = self.tail(out)
        return out, feats