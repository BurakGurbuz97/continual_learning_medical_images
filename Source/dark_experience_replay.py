from argparse import Namespace
from torch.utils.data import DataLoader
import torch
import numpy as np
from typing import Dict, List
from avalanche.benchmarks import GenericCLScenario, TCLExperience
from Source.naive_continual_learner import NaiveContinualLearner
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, Pad
from Source.utils import get_device
from copy import deepcopy
from typing import Tuple

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

class Buffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size, *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

    def is_empty(self) -> bool:
        return self.num_seen_examples == 0

    def empty(self) -> None:
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


class DarkExperienceReplay(NaiveContinualLearner):

    def __init__(self, args: Namespace, backbone: torch.nn.Module, scenario: GenericCLScenario, task2classes: Dict):
        super(DarkExperienceReplay, self).__init__(args, backbone, scenario, task2classes)
        self.args = args
        self.backbone = backbone
        self.scenario = scenario
        self.task2classes = task2classes
        self.transforms = RandomHorizontalFlip(0.3)
        self.buffer = Buffer(self.args.batch_size_memory, get_device())

    # Overwrite this method
    def begin_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        return None


    # Overwrite this method
    def learn_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # Delete unused parameters. we accept this parameter for compatibility. 
        del val_dataset
        # Create optimizer and loss
        self.optim_obj = getattr(torch.optim, self.args.optimizer)
        optimizer = self.optim_obj(self.backbone.parameters(), lr= self.args.learning_rate,
                                   weight_decay= self.args.weight_decay)
        loss = torch.nn.CrossEntropyLoss()
        # Create data loaders
        train_loader = DataLoader(train_dataset.dataset, batch_size = self.args.batch_size, 
                                  shuffle=True)
        print("*****  TASK: {} has classes {}  *****".format(current_task_index,
                                                             self.task2classes[current_task_index]))
        epoch_losses = []
        for epoch_id in range(self.args.epochs):
            print("*****  Epoch: {}/{}  *****".format(epoch_id+1, self.args.epochs))
            self.backbone.train()
            for data, target, _ in train_loader:
                data_aug = self.transforms(data).to(get_device())
                data = data.to(get_device())
                target = target.to(get_device())
                optimizer.zero_grad()
                output, _ = self.backbone(data_aug)
                batch_loss = loss(output, target.long())
                if not self.buffer.is_empty():
                    buf_inputs, buf_logits = self.buffer.get_data(self.args.batch_size, transform=self.transforms)
                    buf_outputs, _ = self.backbone(buf_inputs)
                    batch_loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                batch_loss.backward()
                epoch_losses.append(batch_loss.item())
                optimizer.step()
                self.buffer.add_data(data, logits = output.data)

        return None



    # Overwrite this method
    def end_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        return None
