import argparse
from typing import Tuple, Dict
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import torch

from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.classic import SplitCIFAR10, SplitMNIST
from avalanche.benchmarks.generators import benchmark_with_validation_stream, nc_benchmark
from avalanche.benchmarks.datasets import EMNIST

from config import DATASET_PATH

import medmnist


class CustomDataset(Dataset):
    def __init__(self, split, dataset_name, label_padding = 0, return_idx= False):
        self.dataset = getattr(medmnist, dataset_name)(split=split, download=True, root=DATASET_PATH)
        self.data = self.dataset.imgs
        self.targets = list(map(lambda x: int(x), self.dataset.labels + label_padding))
        self.return_idx = return_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the example at the given index
        example = self.data[idx]
        target = self.targets[idx]

        # Check if the example has the third dimension (n_channels)
        if len(example.shape) == 2:
            example = np.expand_dims(example, axis=-1)

        # Pad the example with zeros to make it 32x32
        example_padded = np.pad(example, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

        # Check if the example has a single channel, if so, copy it along the third dimension to make it 3 channels
        if example_padded.shape[-1] == 1:
            example_padded = np.repeat(example_padded, 3, axis=2)

        # Normalize the pixel values to be between 0 and 1
        example_normalized = example_padded / 255.0
        example_transposed = np.transpose(example_normalized, (2, 0, 1))

        if self.return_idx == False:
            return torch.from_numpy(example_transposed.astype(np.float32)), torch.tensor(target, dtype=torch.long)
        else:
            return torch.from_numpy(example_transposed.astype(np.float32)), torch.tensor(target, dtype=torch.long) , idx


def get_experience_streams(args: argparse.Namespace) -> Tuple[GenericCLScenario, Tuple, int, Dict]:
    # Example of how to add Pytorch dataset. Here EMNIST is pytorch dataset object.
    if args.dataset == "EMNIST": # All letters (26 classes in total)
        emnist_train = EMNIST(root=DATASET_PATH, train = True, split='letters', download= True)
        emnist_train.targets = emnist_train.targets - 1
        emnist_test = EMNIST(root=DATASET_PATH, train = False, split='letters', download= True)
        emnist_test.targets = emnist_test.targets - 1
        stream = nc_benchmark(train_dataset=emnist_train,test_dataset=emnist_test, # type: ignore
                              n_experiences=args.number_of_tasks, task_labels = False, shuffle = False,
                              seed = args.seed, fixed_class_order=list(map(lambda x: int(x), emnist_train.targets.unique())),
                              train_transform = transforms.ToTensor(),
                              eval_transform=transforms.ToTensor())
        
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.3, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, (1, 28, 28), 26, task2classes)
    
    if args.dataset == "MNIST":
        stream = SplitMNIST(n_experiences = args.number_of_tasks,
                            seed = args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(10)))
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.8, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, (1, 28, 28), 10, task2classes)
    

    if args.dataset == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        stream = SplitCIFAR10(n_experiences = args.number_of_tasks,
                              seed = args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(10)),
                              train_transform=transform, eval_transform=transform)
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.3, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, (3, 32, 32), 10, task2classes)
    

    if args.dataset == "Microscopic":
        # Create PathMNIST dataset
        path_train = CustomDataset(split="train", dataset_name="PathMNIST" , return_idx= args.return_idx )
        path_test = CustomDataset(split="test", dataset_name="PathMNIST", return_idx= args.return_idx)
        num_classes_path = len(np.unique(path_train.targets))

        # Create BloodMNIST dataset and pad the labels
        blood_train = CustomDataset(split="train", dataset_name="BloodMNIST", label_padding=num_classes_path, return_idx= args.return_idx)
        blood_test = CustomDataset(split="test", dataset_name="BloodMNIST", label_padding=num_classes_path, return_idx= args.return_idx)
        num_classes_blood = len(np.unique(blood_train.targets))

        # Create TissueMNIST dataset and pad the labels
        tissue_train = CustomDataset(split="train", dataset_name="TissueMNIST", label_padding=num_classes_path + num_classes_blood, return_idx= args.return_idx)
        tissue_test = CustomDataset(split="test", dataset_name="TissueMNIST", label_padding=num_classes_path + num_classes_blood, return_idx= args.return_idx)
        num_classes_tissue = len(np.unique(tissue_train.targets))

        # Create Scenario
        stream = nc_benchmark(train_dataset=[path_train, blood_train, tissue_train], # type: ignore
                              test_dataset=[path_test, blood_test, tissue_test], # type: ignore
                              n_experiences=3,
                              per_exp_classes={0: num_classes_path, 1: num_classes_blood, 2: num_classes_tissue},
                              task_labels = False, shuffle = False,
                              seed = args.seed, fixed_class_order=list([l for l in range(num_classes_path + num_classes_blood + num_classes_tissue)]),
                              train_transform = torch.nn.Identity(),
                              eval_transform=torch.nn.Identity())
        
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        
        total_classes = num_classes_path + num_classes_blood + num_classes_tissue

        return (stream_with_val, (3, 32, 32), total_classes, task2classes)
    
    raise Exception("Dataset {} is not defined!".format(args.dataset))