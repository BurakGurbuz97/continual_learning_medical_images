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
        self.lambda_val = args.lambda_val # The regression weightage
        self.update_size = args.update_size
        self.omega = dict()
        self.prev_weights = dict()
        self.class_in_this_task = None
        self.prev_count = 0
        self.use_task_labels = args.scenario == "TIL"
        
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
                output, _ = self.backbone(data)
                batch_loss = loss(output, target.long())
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
                output_temp = torch.zeros_like(output, device=get_device())
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
    


# From MAS Github repo
# class MAS_Omega_update(torch.optim.SGD):
#     """
#     Update the paramerter importance using the gradient of the function output norm. To be used at deployment time.
#     reg_params:parameters omega to be updated
#     batch_index,batch_size:used to keep a running average over the seen samples
#     """

#     def __init__(self, params, lr=0.001, momentum=0, dampening=0,
#                  weight_decay=0, nesterov=False):
#         super(MAS_Omega_update, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        
#     def __setstate__(self, state):
#         super(MAS_Omega_update, self).__setstate__(state)

#     def step(self, reg_params, batch_index, batch_size, closure=None):
#         """
#         Performs a single parameters importance update setp
#         """
#         #print('************************DOING A STEP************************')
#         loss = None
#         if closure is not None:
#             loss = closure()
             
#         for group in self.param_groups:
   
#             #if the parameter has an omega to be updated
#             for p in group['params']:
          
#                 #print('************************ONE PARAM************************')
                
#                 if p.grad is None:
#                     continue
               
#                 if p in reg_params:
#                     d_p = p.grad.data
                    
#                     #HERE MAS IMPOERANCE UPDATE GOES
#                     #get the gradient
#                     unreg_dp = p.grad.data.clone()
#                     reg_param=reg_params.get(p)

#                     zero=torch.FloatTensor(p.data.size()).zero_()
#                     #get parameter omega
#                     omega=reg_param.get('omega')
#                     omega=omega.cuda()

#                     #sum up the magnitude of the gradient
#                     prev_size=batch_index*batch_size
#                     curr_size=(batch_index+1)*batch_size
#                     omega=omega.mul(prev_size)

#                     omega=omega.add(unreg_dp.abs_())
#                     #update omega value
#                     omega=omega.div(curr_size)
#                     if omega.equal(zero.cuda()):
#                         print('omega after zero')

#                     reg_param['omega']=omega

#                     reg_params[p]=reg_param
#                     #HERE MAS IMPOERANCE UPDATE ENDS
#         return loss