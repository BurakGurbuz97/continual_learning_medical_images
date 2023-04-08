from argparse import Namespace
from torch.utils.data import DataLoader
import torch
from typing import Dict, List
from avalanche.benchmarks import GenericCLScenario, TCLExperience

from Source.utils import get_device

class NaiveContinualLearner():
    """
        Simple Class to inherit. Train model on scenario without any tricks.    
    """

    def __init__(self, args: Namespace, backbone: torch.nn.Module, scenario: GenericCLScenario, task2classes: Dict):
        self.args = args
        self.backbone = backbone
        self.scenario = scenario
        self.task2classes = task2classes


    # Overwrite this method
    def begin_task(self) -> None:
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
        for epoch_id in range(self.args.epochs):
            print("*****  Epoch: {}/{}  *****".format(epoch_id+1, self.args.epochs))
            self.backbone.train()
            for data, target, _ in train_loader:
                data = data.to(get_device())
                target = target.to(get_device())
                optimizer.zero_grad()
                output, _ = self.backbone(data)
                batch_loss = loss(output, target.long())
                batch_loss.backward()
                optimizer.step()

        return None



    # Overwrite this method
    def end_task(self) -> None:
        return None
    

    def _test(self, data_loader: DataLoader, class_in_this_task = None) -> float:
        self.backbone.eval()
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for data, target, _ in data_loader:
                data = data.to(get_device())
                target = target.to(get_device())
                output, _ = self.backbone.forward(data)
                if class_in_this_task:
                    output_temp = torch.zeros_like(output)
                    output_temp[:, class_in_this_task] = output[:, class_in_this_task]
                    output = output_temp
                predicted_class = output.argmax(dim=1)
                correct_predictions += (predicted_class == target).sum().item()
                total_predictions += target.size(0)
        return correct_predictions / total_predictions


    # Overwrite this method if needed
    def accuracies_on_previous_task(self, current_task_index: int, use_task_labels: bool, verbose = True) -> List:
        accuracies = []
        for task_id, test_task in enumerate(self.scenario.test_stream[:current_task_index], 1):
            test_loader = DataLoader(test_task.dataset, self.args.batch_size)
            acc = self._test(test_loader, self.task2classes[task_id] if use_task_labels else None)
            if verbose:
                print("Current Task: {} --> Accuracy on Task-{} is {}  (Scenario: {})".format(
                    current_task_index,
                    task_id,
                    acc,
                    "TIL" if use_task_labels else "CIL"
                ))
            accuracies.append(acc)
        return accuracies


