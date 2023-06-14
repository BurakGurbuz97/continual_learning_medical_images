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
        self.return_idx = False #return_idx

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


class CustomTensorDataset(Dataset):
    def __init__(self, file_path, label_padding = 0, return_idx= False):
        dataset = np.load(file_path)
        self.data = dataset['imgs']
        self.targets = list(map(lambda x: int(x), np.array(dataset['targets']) + label_padding))
        self.return_idx = False #return_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the example at the given index
        example = self.data[idx]
        target = self.targets[idx]

        # Check if the example has the third dimension (n_channels)
        if len(example.shape) == 2:
            example = np.expand_dims(example, axis=-1)

        # Check if the example has a single channel, if so, copy it along the third dimension to make it 3 channels
        if example.shape[-1] == 1:
            example = np.repeat(example, 3, axis=2)

        # Normalize the pixel values to be between 0 and 1
        example_normalized = example / 255.0
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
                              n_experiences=3 if args.number_of_tasks != 1 else 1,
                              per_exp_classes={0: num_classes_path, 1: num_classes_blood, 2: num_classes_tissue}
                              if args.number_of_tasks != 1 else {0: num_classes_path + num_classes_blood + num_classes_tissue},
                              task_labels = False, shuffle = False,
                              seed = args.seed, fixed_class_order=list([l for l in range(num_classes_path + num_classes_blood + num_classes_tissue)]),
                              train_transform = torch.nn.Identity(),
                              eval_transform=torch.nn.Identity())
        
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        
        total_classes = num_classes_path + num_classes_blood + num_classes_tissue

        return (stream_with_val, (3, 32, 32), total_classes, task2classes)
 
    if args.dataset == "Radiological":
        # OrganMNIST, 11 class classification (5, 2, 2, 2), we have four tasks
        organ_train = CustomDataset(split="train", dataset_name="OrganCMNIST",
                                    label_padding = 0, return_idx= args.return_idx)
        organ_test = CustomDataset(split="test", dataset_name="OrganCMNIST",
                                   label_padding = 0, return_idx= args.return_idx)
        num_classes_organ = len(np.unique(organ_train.targets))

        # BreastMNIST, binary classification
        breast_train = CustomDataset(split="train", dataset_name="BreastMNIST",
                                     label_padding= num_classes_organ, return_idx= args.return_idx)
        breast_test = CustomDataset(split="test", dataset_name="BreastMNIST",
                                    label_padding= num_classes_organ, return_idx= args.return_idx)
        num_classes_breast = len(np.unique(breast_train.targets))

        # PneumoniaMNIST, binary classification
        pneumonia_train = CustomDataset(split="train", dataset_name="PneumoniaMNIST",
                                        label_padding=(num_classes_organ+num_classes_breast),
                                        return_idx= args.return_idx)
        pneumonia_test = CustomDataset(split="test", dataset_name="PneumoniaMNIST",
                                       label_padding=(num_classes_organ+num_classes_breast),
                                         return_idx= args.return_idx)
        num_classes_pneumonia = len(np.unique(pneumonia_train.targets))

        
        stream = nc_benchmark(train_dataset=[organ_train, breast_train, pneumonia_train], # type: ignore
                        test_dataset=[organ_test, breast_test, pneumonia_test], # type: ignore
                        n_experiences=6 if args.number_of_tasks != 1 else 1,
                        per_exp_classes={0: 5, 1: 2, 2:2, 3:2,
                                         4: num_classes_breast, 5: num_classes_pneumonia}
                                         if args.number_of_tasks != 1 else {0: num_classes_organ + num_classes_breast + num_classes_pneumonia},
                        task_labels = False, shuffle = False,
                        seed = args.seed, fixed_class_order=list([l
                                            for l in range(num_classes_organ + num_classes_breast + num_classes_pneumonia)]),
                        train_transform = torch.nn.Identity(),
                        eval_transform=torch.nn.Identity())

        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        
        total_classes = num_classes_organ + num_classes_breast + num_classes_pneumonia

        return (stream_with_val, (3, 32, 32), total_classes, task2classes)
    
    if args.dataset == "Interdepartmental":
        pad_value = 0
        # Create PathMNIST dataset
        path_train = CustomDataset(split="train", dataset_name="PathMNIST" , label_padding= pad_value, return_idx= args.return_idx )
        path_test = CustomDataset(split="test", dataset_name="PathMNIST", label_padding= pad_value,return_idx= args.return_idx)
        num_classes_path = len(np.unique(path_train.targets))
        pad_value = num_classes_path
        
        # PneumoniaMNIST, binary classification
        pneumonia_train = CustomDataset(split="train", dataset_name="PneumoniaMNIST",
                                        label_padding=pad_value,
                                        return_idx= args.return_idx)
        pneumonia_test = CustomDataset(split="test", dataset_name="PneumoniaMNIST",
                                       label_padding=pad_value,
                                         return_idx= args.return_idx)
        num_classes_pneumonia = len(np.unique(pneumonia_train.targets))
        pad_value += num_classes_pneumonia

        # Create DermaMNIST dataset
        derma_train = CustomDataset(split="train", dataset_name="DermaMNIST" , return_idx= args.return_idx )
        derma_test = CustomDataset(split="test", dataset_name="DermaMNIST", return_idx= args.return_idx)
        num_classes_derma = len(np.unique(derma_train.targets))
        pad_value += num_classes_derma
        
        # Create RetinaMNIST dataset and pad the labels
        retina_train = CustomDataset(split="train", dataset_name="RetinaMNIST", label_padding=pad_value, return_idx= args.return_idx)
        retina_test = CustomDataset(split="test", dataset_name="RetinaMNIST", label_padding=pad_value, return_idx= args.return_idx)
        num_classes_retina = len(np.unique(retina_train.targets))
        pad_value += num_classes_retina
        
        total_classes = num_classes_derma + num_classes_retina + num_classes_path + num_classes_pneumonia

        # Create Scenario
        stream = nc_benchmark(train_dataset=[path_train, pneumonia_train, derma_train, retina_train ], # type: ignore
                              test_dataset=[path_test, pneumonia_test , derma_test, retina_test], # type: ignore
                              n_experiences=4 if args.number_of_tasks != 1 else 1,
                              per_exp_classes={0:num_classes_path , 1:num_classes_pneumonia , 2: num_classes_derma, 3:num_classes_retina}
                              if args.number_of_tasks != 1 else {0: total_classes},
                              task_labels = False, shuffle = False,
                              seed = args.seed, fixed_class_order=list([l for l in range(total_classes)]),
                              train_transform = torch.nn.Identity(),
                              eval_transform=torch.nn.Identity())
        
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        
        return (stream_with_val, (3, 32, 32), total_classes, task2classes)
    

    if args.dataset == "Interhospital_6":
        chexpert_train = CustomTensorDataset("/home/amrit/pipeline/in_progress/continual_learning/datasets/inter_hosp_data/chexpert_32.npz",
                                    label_padding = 0, return_idx= args.return_idx)
        num_classes_chexpert = len(np.unique(chexpert_train.targets))

        cxr14_train = CustomTensorDataset("/home/amrit/pipeline/in_progress/continual_learning/datasets/inter_hosp_data/cxr14_32.npz",
                                    label_padding = num_classes_chexpert, return_idx= args.return_idx)
        num_classes_cxr14 = len(np.unique(cxr14_train.targets))

        vinbig_train = CustomTensorDataset("/home/amrit/pipeline/in_progress/continual_learning/datasets/inter_hosp_data/vinbig_32.npz",
                                    label_padding = num_classes_chexpert+ num_classes_cxr14, return_idx= args.return_idx)
        num_classes_vinbig = len(np.unique(vinbig_train.targets))


        total_classes = num_classes_vinbig + num_classes_chexpert + num_classes_cxr14

        stream = nc_benchmark(train_dataset=[chexpert_train, cxr14_train, vinbig_train], # type: ignore
                              test_dataset=[chexpert_train, cxr14_train, vinbig_train],
                        n_experiences=6 if args.number_of_tasks != 1 else 1,
                        per_exp_classes={0: 2, 1: 3, 2:2, 3:3, 4: 2, 5: 3}
                                         if args.number_of_tasks != 1 else {0: total_classes},
                        task_labels = False, shuffle = False,
                        seed = args.seed, fixed_class_order=list([l
                                            for l in range(total_classes)]),
                        train_transform = torch.nn.Identity(),
                        eval_transform=torch.nn.Identity())

        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        
     
        return (stream_with_val, (3, 32, 32), total_classes, task2classes)
    
    if args.dataset == "Interhospital":
        chexpert_train = CustomTensorDataset("/home/amrit/pipeline/in_progress/continual_learning/datasets/medical imaging/chexpert_v1_64.npz",
                                    label_padding = 0, return_idx= args.return_idx)
        num_classes_chexpert = len(np.unique(chexpert_train.targets))

        cxr14_train = CustomTensorDataset("/home/amrit/pipeline/in_progress/continual_learning/datasets/medical imaging/cxr14_v1_64.npz",
                                    label_padding = num_classes_chexpert, return_idx= args.return_idx)
        num_classes_cxr14 = len(np.unique(cxr14_train.targets))

        vinbig_train = CustomTensorDataset("/home/amrit/pipeline/in_progress/continual_learning/datasets/medical imaging/vinbig_v1_64.npz",
                                    label_padding = num_classes_chexpert+ num_classes_cxr14, return_idx= args.return_idx)
        num_classes_vinbig = len(np.unique(vinbig_train.targets))

        total_classes = num_classes_vinbig + num_classes_chexpert + num_classes_cxr14

        stream = nc_benchmark(train_dataset=[chexpert_train, cxr14_train, vinbig_train], # type: ignore
                              test_dataset=[chexpert_train, cxr14_train, vinbig_train],
                        n_experiences=3 if args.number_of_tasks != 1 else 1,
                        per_exp_classes={0: 4, 1:4, 2:4}
                                         if args.number_of_tasks != 1 else {0: total_classes},
                        task_labels = False, shuffle = False,
                        seed = args.seed, fixed_class_order=list([l
                                            for l in range(total_classes)]),
                        train_transform = torch.nn.Identity(),
                        eval_transform=torch.nn.Identity())

        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        
     
        return (stream_with_val, (3, 64, 64), total_classes, task2classes)
    
    
    if args.dataset == "all":
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

        # OrganMNIST, 11 class classification (5, 2, 2, 2), we have four tasks
        organ_train = CustomDataset(split="train", dataset_name="OrganCMNIST",
                                    label_padding = 0, return_idx= args.return_idx)
        organ_test = CustomDataset(split="test", dataset_name="OrganCMNIST",
                                   label_padding = 0, return_idx= args.return_idx)
        num_classes_organ = len(np.unique(organ_train.targets))

        # BreastMNIST, binary classification
        breast_train = CustomDataset(split="train", dataset_name="BreastMNIST",
                                     label_padding= num_classes_organ, return_idx= args.return_idx)
        breast_test = CustomDataset(split="test", dataset_name="BreastMNIST",
                                    label_padding= num_classes_organ, return_idx= args.return_idx)
        num_classes_breast = len(np.unique(breast_train.targets))

        # PneumoniaMNIST, binary classification
        pneumonia_train = CustomDataset(split="train", dataset_name="PneumoniaMNIST",
                                        label_padding=(num_classes_organ+num_classes_breast),
                                        return_idx= args.return_idx)
        pneumonia_test = CustomDataset(split="test", dataset_name="PneumoniaMNIST",
                                       label_padding=(num_classes_organ+num_classes_breast),
                                         return_idx= args.return_idx)
        num_classes_pneumonia = len(np.unique(pneumonia_train.targets))

        total_classes =  num_classes_path + num_classes_blood + num_classes_tissue + num_classes_organ + num_classes_breast + num_classes_pneumonia
        
        stream = nc_benchmark(train_dataset=[path_train, blood_train, tissue_train,organ_train, breast_train, pneumonia_train], # type: ignore
                        test_dataset=[path_test, blood_test, tissue_test, organ_test, breast_test, pneumonia_test], # type: ignore
                        n_experiences=6 + 3  if args.number_of_tasks != 1 else 1,
                        per_exp_classes={0: num_classes_path, 1: num_classes_blood, 2: num_classes_tissue, 3: 5, 4: 2, 5:2, 6:2,
                                         7: num_classes_breast, 8: num_classes_pneumonia}
                                         if args.number_of_tasks != 1 else {0: total_classes},
                        task_labels = False, shuffle = False,
                        seed = args.seed, fixed_class_order=list([l
                                            for l in range(total_classes)]),
                        train_transform = torch.nn.Identity(),
                        eval_transform=torch.nn.Identity())

        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.1, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        
        return (stream_with_val, (3, 32, 32), total_classes, task2classes)
    
    raise Exception("Dataset {} is not defined!".format(args.dataset))