import torch
from argparse import Namespace
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from avalanche.benchmarks import GenericCLScenario, TCLExperience
import math
import copy

from Source.utils import get_device

from Source.naive_continual_learner import NaiveContinualLearner

# TODO: Later remove the imports not used..

class MemoryAwareSynapses(NaiveContinualLearner):
    def __init__(self, args: Namespace, backbone: torch.nn.Module, scenario: GenericCLScenario, task2classes: Dict):
        super(MemoryAwareSynapses, self).__init__(args, backbone, scenario, task2classes)
