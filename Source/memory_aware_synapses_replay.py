import torch
from argparse import Namespace
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from avalanche.benchmarks import GenericCLScenario, TCLExperience
# import math
# import copy

from Source.utils import get_device

from Source.naive_continual_learner import NaiveContinualLearner

# TODO: Later remove the imports not used..

def get_n_sample_per_class(dataset: TCLExperience, n: int) -> List:
    indices = {i: [] for i in dataset.classes_in_this_experience}
    for i, (_, y, _) in enumerate(dataset.dataset):
        indices[y].append(i)

    subsets = []
    for i in dataset.classes_in_this_experience:
        dataloader = DataLoader(Subset(dataset.dataset, indices[i][:n]), batch_size=n)
        samples, _, _  = next(iter(dataloader))
        subsets.append((samples, i))
    return subsets

class MemoryAwareSynapsesReplay(NaiveContinualLearner):
    def __init__(self, args: Namespace, backbone: torch.nn.Module, scenario: GenericCLScenario, task2classes: Dict):
        super(MemoryAwareSynapsesReplay, self).__init__(args, backbone, scenario, task2classes)
        self.lambda_val = args.lambda_val # The regression weightage
        self.update_size = args.update_size
        self.omega = dict()
        self.prev_weights = dict()
        self.class_in_this_task = None
        self.prev_count = 0
        self.use_task_labels = args.scenario == "TIL"
        self.memory = MemoryBuffer(args, task2classes, self.backbone.input_size) # type: ignore
        
        for name, param in self.backbone.named_parameters():
            # print(name, param.size())
            self.omega[name] = torch.zeros_like(param).to(get_device())
            self.prev_weights[name] = torch.zeros_like(param).to(get_device())


    # Overwrite this method
    def begin_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # Based on the task number, and if we are doing TIL or CIL, apply masking ..
        self.class_in_this_task = self.task2classes[current_task_index] if self.use_task_labels else None
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
        # print("*****  TASK: {} has classes {}  *****".format(current_task_index,
        #                                                      self.task2classes[current_task_index]))
        # Based on the task number, and if we are doing TIL or CIL, apply masking ..
        for epoch_id in range(self.args.epochs):
            # print("*****  Epoch: {}/{}  *****".format(epoch_id+1, self.args.epochs))
            self.backbone.train()
            for data, target, _ in train_loader:
                # print('data', data)
                data = data.to(get_device())
                target = target.to(get_device())
                optimizer.zero_grad()
                stream_output, _ = self.backbone(data)
                
                # Addition to incorporate Replay
                memory=self.memory if current_task_index > 1 else None
                if memory is not None:
                    memo_samples, memo_labels = memory.sample_n(n = self.args.batch_size_memory,
                                                                current_task_index = current_task_index)
                    memo_samples, memo_labels = torch.tensor(memo_samples).to(get_device()), torch.tensor(memo_labels).to(get_device())
                    memo_output, _ = self.backbone.forward(memo_samples)
                    stream_output = torch.cat((stream_output, memo_output), dim=0)
                    target = torch.cat((target, memo_labels), dim=0)

                batch_loss = loss(stream_output, target)
                reg_loss = torch.tensor(0.0).to(get_device())
                if current_task_index != 1:
                    for name, param in self.backbone.named_parameters():
                        reg_loss += torch.sum(self.omega[name] * torch.square(param - self.prev_weights[name]))
                loss_t = batch_loss + self.lambda_val * reg_loss
                # print(batch_loss.data, reg_loss.data, loss_t.data)
                loss_t.backward()
                optimizer.step()

        return None


    # Overwrite this method
    def end_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # Update the Omega based on some images from the current task
        # Select few images, TODO: add a hyperparamter to select the number of images...
        update_loader = DataLoader(train_dataset.dataset, batch_size = 1, shuffle=True)

        # Create a batch of these images, define the L2 loss function and call backward to calculate the gradients.
        count = 0
        temp_dict = dict()
        for data, target, _ in update_loader:
            self.backbone.zero_grad()
            data = next(iter(update_loader))[0].to(get_device())
            # print('data', data)
            output, _ = self.backbone(data)
            if self.class_in_this_task:
                output_temp = torch.zeros_like(output)
                output_temp[:, self.class_in_this_task] = output[:, self.class_in_this_task]
                output = output_temp
            l2_loss = torch.norm(output, dim=None)
            l2_loss.backward()
            
            for name, param in self.backbone.named_parameters():
                if name not in temp_dict:
                    temp_dict[name] = torch.abs(param.grad)
                else:
                    temp_dict[name] += torch.abs(param.grad)

            count += 1
        # Normalize the gradients and store them in omega
        for name in temp_dict:
            self.omega[name] = (self.omega[name] * self.prev_count + temp_dict[name]) / (self.prev_count + count)
        # print("self.omega['classifier.bias']", self.omega['classifier.bias'])

        # Update the weights after the training of the previous task
        for name, param in self.backbone.named_parameters():
            self.prev_weights[name] = param.detach().clone().to(get_device())
        # print("classifier.bias", self.prev_weights['classifier.bias'])
        self.prev_count += count

        # update memory
        subsets = get_n_sample_per_class(train_dataset, self.args.memory_per_class)
        all_samples = [samples.to(get_device()) for samples, _ in subsets]
        labels = [label for _, label in subsets]
        self.memory.insert_samples(all_samples, labels)

        return None
    

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
    

class MemoryBuffer():

    def __init__(self, args: Namespace,  task2classes: Dict, representation_size: int):
        self.args = args
        self.memory = {} # mapping from what to what? Tasks -> Dict(classes -> samples)
        self.task2classes = task2classes
        self.class2task = {}
        for task, classes in task2classes.items():
            for class_ in classes:
                self.class2task[class_] = task

        self.representation_size = representation_size
        for task_index, classes in self.task2classes.items():
            if self.memory.get(task_index, None) is None:
                self.memory[task_index] = {}
            for class_ in classes:
                self.memory[task_index][class_] = None

    # Inserts the samples into the memory
    def insert_samples(self, all_samples: List[torch.Tensor], labels: List) -> None:
        for label, class_samples in zip(labels, all_samples):
            self.memory[self.class2task[int(label)]][int(label)] = class_samples
        return None
            
    # Returns n samples with its labels. n is generally batch_size_memory.
    # The samples are fetched uniformly from each class seen till now
    def sample_n(self, n: int, current_task_index: int) -> Tuple[NDArray, NDArray]:
        tasks = list(range(1, current_task_index))
        samples_per_class = self.get_samples_per_class(n, current_task_index)
        samples = []
        labels = []
        i = 0
        for task in tasks:
            for class_ in self.memory[task]:
                class_samples = self.memory[task][class_].cpu()
                try:
                    samples.extend(class_samples[np.random.choice(class_samples.shape[0],
                                                        samples_per_class[i], replace=False)])
                except:
                    # If not enough memory samples, sample with replacement.
                    samples.extend(class_samples[np.random.choice(class_samples.shape[0],
                                                        samples_per_class[i], replace=True)])
                labels.extend(list([class_])*samples_per_class[i])
                i = i + 1
        return np.stack(samples), np.array(labels)
    
    # def get_n_from_classes(self, n: int, current_task_index: int) -> Tuple:
    #     tasks = list(range(1, current_task_index))
    #     labels = []
    #     samples = []
    #     time = 0
    #     ages = []
    #     for task in reversed(tasks):
    #         for class_ in self.memory[task]:
    #             class_samples = self.memory[task][class_].cpu()
    #             try:
    #                 samples.append(class_samples[np.random.choice(class_samples.shape[0],n, replace=False)])
    #             except:
    #                 # If not enough memory samples, sample with replacement.
    #                 samples.append(class_samples[np.random.choice(class_samples.shape[0],n, replace=True)])
    #             labels.append(class_)
    #             ages.append(time)
    #         time = time + 1
    #     return samples, labels, ages
    
    # Return number of samples for each class
    def get_samples_per_class(self, n: int, current_task_index: int) -> List:
        tasks = list(range(1, current_task_index))
        number_of_classes = self.task2numclasses(tasks) # number of classes seen till now
        a =  int(n / number_of_classes) # a is the number of samples per class
        remaining = n % number_of_classes
        samples_per_class = [a for _ in range(number_of_classes)]
        for i in range(remaining):
            samples_per_class[i] = samples_per_class[i] + 1
        return samples_per_class
    
    def task2numclasses(self, tasks: List) -> int:
        numclasses = 0
        for task in tasks:
            numclasses = numclasses + len(self.task2classes[task]) 
        return numclasses