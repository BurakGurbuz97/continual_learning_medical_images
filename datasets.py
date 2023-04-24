import argparse
from typing import Tuple, Dict
from torchvision import transforms

from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.classic import SplitCIFAR10, SplitMNIST
from avalanche.benchmarks.generators import benchmark_with_validation_stream, nc_benchmark
from avalanche.benchmarks.datasets import EMNIST

from config import DATASET_PATH



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
    
    raise Exception("Dataset {} is not defined!".format(args.dataset))