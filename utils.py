import numpy as np
import torch
from torch.backends import cudnn 
from typing import List
import csv
import os

import argparse
import random

from config import LOG_PATH

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Experiment')
    # Logging params
    parser.add_argument('--experiment_name', type=str, default = 'REMIND_MNIST_CIL') # This is the name of the log directory

    # Dataset params
    parser.add_argument('--dataset', type=str, default = 'Microscopic', choices=["MNIST", "CIFAR10", "EMNIST", "Microscopic", "Radiological", 
                                                                                 "Interhospital" , "Interdepartmental"]) # Name of the datasets to partition
    parser.add_argument('--number_of_tasks', type=int, default = 5)
    parser.add_argument('--scenario', type=str, default = "CIL", choices=["TIL", "CIL"])

    # Architectural params
    parser.add_argument('--backbone', type=str, default = 'cnn_small') # DNN backbone
    # Number of tasks = 1 ==> naive_continual_learner = joint learner (upper bound)
    # Number of tasks > 1 ==> naive_continual_learner = standard SGD (lower bound)
    parser.add_argument('--method', type=str, default = 'remind') # Continual learning method name

    # System params
    parser.add_argument('--seed', type=int,  default=0) # for reproducibility
    # setting seed is not enough for some operations, if 1, all non-deterministic operations fails to ensure reproducibility
    parser.add_argument('--deterministic', type=int,  default=1) 

    # Learning params
    parser.add_argument('--optimizer', type=str, default = 'SGD') # Anything under torch.optim works. e.g., 'SGD' and 'Adam'
    parser.add_argument('--sgd_momentum', type=float, default = 0.90)
    parser.add_argument('--learning_rate', type=float, default = 0.001)
    parser.add_argument('--batch_size', type=int, default = 256)
    parser.add_argument('--batch_size_memory', type=int, default = 256) # needed for some replay methods
    parser.add_argument('--weight_decay', type=float, default =0.0) # l2 regularization
    parser.add_argument('--epochs', type=int, default = 5) # num epochs per task
    parser.add_argument('--memory_per_class', type=int, default = 100) # needed for some replay methods

    # ADDITIONAL PARAMS: ADD ANYTHING NEEDED FOR SPECIFIC METHOD
    # MAKE SURE YOU PROVIDE DEFAULT VALUE, SO THAT OTHERS CAN JUST IGNORE
    # MAS params
    parser.add_argument('--lambda_val', type=float, default=5.0)
    parser.add_argument('--update_size', type=int, default=50) # The number of samples over which we calculate the omega values

    # NISPA params
    parser.add_argument('--min_activation_perc', type=float, default=60.0)
    parser.add_argument('--phase_epochs', type=int, default = 1)
    parser.add_argument("--num_phases", type=int, default=20)
    parser.add_argument('--prune_perc', type=float, default=80.0)

    # REMIND params:
    parser.add_argument('--return_idx', type=bool, default = False)
    parser.add_argument('--num_scans_per_class', type=float, default = 100) # CIFAR100: 4.096, EMNIST: 0.0784
    parser.add_argument('--num_codebooks', type=int, default = 32)
    parser.add_argument('--codebook_size', type=int, default = 256)
    parser.add_argument('--load_path', type=str, default = None) #"./cifar10_pretrain_g.pth")
    parser.add_argument('--store_path', type=str, default = "./pretrain_g")
    parser.add_argument('--pretrain_lr', type=float, default = 0.001)
    parser.add_argument('--num_epochs', type=int, default = 2)
    parser.add_argument('--pretrain_epochs', type=int, default = 2)
    parser.add_argument('--verbose', type=int, default = True)


    # DER params
    parser.add_argument('--alpha', type=float, default = 0.5)
    parser.add_argument('--beta', type=float, default = 0.5)
    parser.add_argument('--pp', action="store_true", help="Use DER++")
    return parser.parse_args()


def log(args: argparse.Namespace, all_accuracies: List[List], btw_list , ftw_list, suffix= "Acc") -> None:
    with open(os.path.join(LOG_PATH, 'results_{}.csv'.format(args.experiment_name)), mode='w', newline='') as csvfile:
        column_headers = ['Task{}'.format(i) for i, _ in enumerate(all_accuracies, 1)]
        row_headers = ['{}'.format(i) for i, _ in enumerate(all_accuracies, 1)]
        csv_writer = csv.writer(csvfile)
        
        # Write column headers
        csv_writer.writerow(['After task'] + column_headers)  # Add an empty cell before column headers for row headers
        
        # Write row headers and data
        for row_header, row_data in zip(row_headers, all_accuracies):
            csv_writer.writerow([row_header] + row_data)

        csv_writer.writerow(['btw'] + btw_list)
        csv_writer.writerow(['ftw'] + ftw_list)


def get_backward_transfer(acc):
    score = []
    N = len(acc)
    for i in range(0,N):
        aaa = acc[N-1][i] - acc[i][i]
        score.append(aaa)
    score.append(sum(score)/(N-1))
    return score

def get_forward_transfer(acc, b ):
    score = []
    N = len(acc)
    for i in range(1,N):
        aaa = acc[i-1][i] - b[i]
        score.append(aaa)
    score.append(sum(score)/(N-1))
    return score
