from argparse import Namespace

from typing import Dict
from avalanche.benchmarks import GenericCLScenario, TCLExperience
import torch
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import faiss
import numpy as np
import time
import torch.nn.functional as F

from typing import Tuple, List

from continuum import rehearsal
from continuum.tasks import TaskSet
from torch.utils.data import TensorDataset
from numpy.typing import NDArray


from .Backbones.cnn_small import REMIND_F, REMIND_G
from .naive_continual_learner import NaiveContinualLearner

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_f_and_g(args: Namespace, pretraining = False):
    if args.dataset == "Microscopic":
        return REMIND_F(output_size= 9 if pretraining else 25).to(get_device()), REMIND_G().to(get_device()), 9, 25
    elif args.dataset == "Radiological":
        return REMIND_F(output_size= 11 if pretraining else 15).to(get_device()), REMIND_G().to(get_device()), 11, 15
    else:
        raise NotImplementedError("Dataset not implemented: {}".format(args.dataset))
    

def get_num_feats_and_channels(args: Namespace):
    if args.dataset == "EMNIST":
        return 7, 16
    elif args.dataset == "CIFAR100":
        return 8, 64
    elif args.dataset == "Microscopic":
        return 16, 64
    elif args.dataset == "Radiological":
        return 16, 64
    else:
        raise NotImplementedError("Dataset not implemented: {}".format(args.dataset))


class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """

    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1

def _bytes_to_mb(byte_value):
    mb_value = byte_value / (1024 * 1024)
    return mb_value

class Memory:
    def __init__(self, num_channels, num_codebooks, num_feats, nbits, memo_size_mb, nb_total_classes):
        self.memo_size_mb = memo_size_mb
        pq_size = num_codebooks * (2 ** nbits) * (num_channels // num_codebooks) * 4
        size_per_quan = num_feats * num_feats * num_codebooks
        memory_size_samples = memo_size_mb #int((self.memo_size_mb  - _bytes_to_mb(pq_size)) // _bytes_to_mb(size_per_quan))
        print("Memory size in samples: ", memory_size_samples) 

        self.memory = rehearsal.RehearsalMemory(
                memory_size=memory_size_samples,
                herding_method="random",
                fixed_memory=False,
                nb_total_classes = nb_total_classes
            )

    def push_samples(self, codes: NDArray, labels: NDArray, task_id: NDArray):
        #Get samples function applies transformation to samples
        #Use only normalization otherwise we will store augmented samples
        self.memory.add(codes, labels, task_id, z = False)


    def get_random_samples(self, n):
        l = len(self.memory)
        try:
            index = np.random.choice(list(range(l)), size = n, replace = False)
        except:
            index = np.random.choice(list(range(l)), size = n, replace = True)
        mem_x, mem_y, mem_t = self.memory.get()
        return mem_x[index], mem_y[index], mem_t[index]

    #Use this to get pytorch dataset for memory samples
    def create_dataset(self):
        mem_x, mem_y, _ = self.memory.get()
        return  TensorDataset(torch.Tensor(mem_x),torch.Tensor(mem_y))


class Remind(NaiveContinualLearner):

    def __init__(self, args: Namespace, scenario: GenericCLScenario, task2classes: Dict):
        self.args = args
        self.original_scenario = scenario
        self.tas2classes = task2classes
        self.classifier_F, self.feature_extract_G, self.num_classes_pretrain, self.num_classes_cl = get_f_and_g(args, pretraining=True)
        self.num_feats, self.num_channels = get_num_feats_and_channels(args)
        self.pq = faiss.ProductQuantizer(self.num_channels, args.num_codebooks, int(np.log2(args.codebook_size)))
        self.memory = None # added after pretraining



    def train_pq(self, pretrain_task: Dataset):
        feats_base_init = []  # This should be casted to numpy array
        labels_base_init = []  # This should be casted to numpy array

        # Create DataLoader for pretrain_task
        pretrain_loader = DataLoader(pretrain_task, batch_size=self.args.batch_size, shuffle=False)

        # Set feature extractor to evaluation mode
        self.feature_extract_G.eval()

        # Iterate over pretrain_loader and extract features and labels
        with torch.no_grad():
            for inputs, labels ,_  in pretrain_loader:
                inputs = inputs.to(get_device())

                # Extract features using feature_extract_G
                features = self.feature_extract_G(inputs)

                # Append features and labels to the respective lists
                feats_base_init.extend(features.detach().cpu().numpy())
                labels_base_init.extend(labels.detach().cpu().numpy())

        # Convert lists to numpy arrays
        feats_base_init = np.array(feats_base_init)
        labels_base_init = np.array(labels_base_init)
        print('\nTraining Product Quantizer')
        start = time.time()
        train_data_base_init = np.transpose(feats_base_init, (0, 2, 3, 1))
        train_data_base_init = np.reshape(train_data_base_init, (-1, self.num_channels))
        self.pq.train(train_data_base_init) # type: ignore
        print("Completed in {} secs".format(time.time() - start))


    def pretrain(self, pretrain_task: Dataset, pretrain_test_task: Dataset):

        # Pretraining of the model
        if self.args.load_path != None:
            print("Loading pretrained feature_extract_G from {}".format(self.args.load_path))
            # Load weights for feature_extract_G and freeze it
            self.feature_extract_G.load_state_dict(torch.load(self.args.load_path))
            for param in self.feature_extract_G.parameters():
                param.requires_grad = False
        else: # FileNotFoundError:
            print("Weights not found. Pretraining feature_extract_G")
            # Pretrain classifier_F and feature_extract_G on pretrain_task
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(list(self.feature_extract_G.parameters()) + list(self.classifier_F.parameters()),
                                        lr=self.args.pretrain_lr,
                                        momentum=0.90)
            train_loader = DataLoader(pretrain_task, batch_size=self.args.batch_size,shuffle=True)

            for epoch in range(self.args.pretrain_epochs):
                for _, (data, target,_ ) in enumerate(train_loader):
                    optimizer.zero_grad()
                    data = data.to(get_device())
                    target = target.to(get_device())

                    features = self.feature_extract_G(data)
                    output = self.classifier_F(features)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                print(f"Pretrain Epoch: {epoch+1}/{self.args.pretrain_epochs}, Loss: {loss.item()}")

            # Save feature_extract_G to store_path
            torch.save(self.feature_extract_G.state_dict(), self.args.store_path+self.args.experiment_name+'.pth')
            for param in self.feature_extract_G.parameters():
                param.requires_grad = False

            # Evaluate test accuracy
            self.feature_extract_G.eval()
            self.classifier_F.eval()

            test_loader = DataLoader(pretrain_test_task, batch_size=self.args.batch_size, shuffle=False)
            correct, total = 0, 0
            with torch.no_grad():
                for data, target , _ in test_loader:
                    data = data.to(get_device())
                    target = target.to(get_device())

                    features = self.feature_extract_G(data)
                    output = self.classifier_F(features)
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            print(f"Test Accuracy: {accuracy}%")

        # Reinitialize classifier_F with correct output units
        self.classifier_F, _, _, _ = get_f_and_g(self.args, pretraining=False)
        # print("Feature Extractor: {}".format(self.feature_extract_G))
        # print("Classifier: {}".format(self.classifier_F))

            # Overwrite this method
    def begin_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        if current_task_index == 1:
            self.pretrain(train_dataset.dataset, val_dataset.dataset)
            # Creating Memory
            print('\nCreating Memory')
            self.memory = Memory(self.num_channels, self.args.num_codebooks, self.num_feats,
                                int(np.log2(self.args.codebook_size)),
                                self.args.num_scans_per_class*self.args.num_classes, self.num_classes_cl)
    
        self.train_pq(train_dataset.dataset)

    def learn_task(self, train_task: TCLExperience,
                           val_task: TCLExperience, episode_index: int):
        print("****** Learning Episode-{}   Classes: {} ******".format(episode_index, train_task.classes_in_this_experience))
        # Create loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.classifier_F.parameters(),
                                        lr=self.args.learning_rate,
                                        momentum=0.90)
        # Create DataLoader for train_task, val_task and test_task
        train_loader = DataLoader(train_task.dataset, batch_size=self.args.batch_size,shuffle=True, drop_last=True)
        test_loader = DataLoader(val_task.dataset, batch_size=self.args.batch_size,shuffle=False, drop_last=True)


        self.classifier_F.train()
        self.feature_extract_G.eval()
        total_loss = CMA()
        for epoch in range(self.args.num_epochs):
            for batch_images, batch_labels ,_ in train_loader: # type: ignore
                data_batch = self.feature_extract_G(batch_images.to(get_device())).cpu().numpy() # type: ignore
                data_batch = np.transpose(data_batch, (0, 2, 3, 1))
                data_batch = np.reshape(data_batch, (-1, self.num_channels))
                codes = self.pq.compute_codes(data_batch) # type: ignore
                codes = np.reshape(codes, (-1, self.num_feats, self.num_feats, self.args.num_codebooks))

                # Concat Memory codes with current codes
                if episode_index != 1:
                    memory_codes, memory_labels, _ = self.memory.get_random_samples(n = self.args.batch_size_memory)
                    codes = np.concatenate((codes, memory_codes), axis=0)
                    batch_labels = torch.tensor(np.concatenate((batch_labels, memory_labels), axis=0))

                # Reconstruct codes
                codes = np.reshape(codes, (
                        codes.shape[0] * self.num_feats * self.num_feats, self.args.num_codebooks))
                data_batch_reconstructed = self.pq.decode(codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                          (-1, self.num_feats, self.num_feats,
                                                           self.num_channels))
                data_batch_reconstructed = torch.from_numpy(
                        np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).cuda()
                
                # fit on replay mini-batch plus new sample
                output = self.classifier_F(data_batch_reconstructed)
                batch_labels = batch_labels.to(get_device())
                loss = criterion(output, batch_labels)

                optimizer.zero_grad() 
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                total_loss.update(loss.item())

            print(f"Epoch: {epoch+1}/{self.args.num_epochs}, Loss: {total_loss.avg}")
            if self.args.verbose:
                _, _, accuracy_train = self.predict(train_loader)
                _, _, accuracy_test = self.predict(test_loader)
                print(f"Epoch: {epoch+1}/{self.args.num_epochs}, Accuracy Train: {accuracy_train}")
                print(f"Epoch: {epoch+1}/{self.args.num_epochs}, Accuracy Test: {accuracy_test}")

        # Push codes from traininig data to memory
        print("Pushing codes to memory")
        self.update_memory(train_loader, episode_index)


    def update_memory(self, train_loader, episode_index) -> None:
        # Update memory with train_task
        all_codes = []
        all_labels = []
        for batch_images, batch_labels,_ in train_loader:
            # get features from G and latent codes from PQ
            data_batch = self.feature_extract_G(batch_images.cuda()).cpu().numpy()
            data_batch = np.transpose(data_batch, (0, 2, 3, 1))
            data_batch = np.reshape(data_batch, (-1, self.num_channels))
            codes = self.pq.compute_codes(data_batch) # type: ignore
            codes = np.reshape(codes, (-1, self.num_feats, self.num_feats, self.args.num_codebooks))
            all_codes.extend(codes)
            all_labels.extend(batch_labels)
        all_codes = np.array(all_codes)
        all_labels = np.array(all_labels)
        task_labels = np.array([episode_index]*len(all_labels))
        self.memory.push_samples(all_codes, all_labels, task_labels)

    # Overwrite this method
    def end_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        return None

    def predict(self, loader) -> Tuple[np.ndarray, np.ndarray, float]:
        with torch.no_grad():
            self.classifier_F.eval()
            self.feature_extract_G.eval()

            probas = torch.zeros((len(loader.dataset), self.num_classes_cl))
            all_lbls = torch.zeros((len(loader.dataset)))
            start_ix = 0
            for batch_x, batch_lbls,_ in loader:
                batch_x = batch_x.to(get_device())
                batch_lbls = batch_lbls.to(get_device())

                # get G features
                data_batch = self.feature_extract_G(batch_x).detach().cpu().numpy()

                # quantize test data so features are in the same space as training data
                data_batch = np.transpose(data_batch, (0, 2, 3, 1))
                data_batch = np.reshape(data_batch, (-1, self.num_channels))
                codes = self.pq.compute_codes(data_batch) # type: ignore
                data_batch_reconstructed = self.pq.decode(codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                      (-1, self.num_feats, self.num_feats, self.num_channels))
                data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).cuda()

                logits = self.classifier_F(data_batch_reconstructed)
                end_ix = start_ix + len(batch_x)
                probas[start_ix:end_ix] = F.softmax(logits.data, dim=1)
                all_lbls[start_ix:end_ix] = batch_lbls.squeeze()
                start_ix = end_ix

            preds = probas.data.max(1)[1]

        accuracy = float(torch.eq(preds, all_lbls).float().mean())
        return preds.numpy(), all_lbls.int().numpy(), accuracy
    
    # Overwrite this method if needed
    def accuracies_on_previous_task(self, current_task_index: int, verbose = True) -> List:
        accuracies = []
        for task_id, test_task in enumerate(self.original_scenario.test_stream[:current_task_index], 1):
            test_loader = DataLoader(test_task.dataset, self.args.batch_size)
            _, _, acc = self.predict(test_loader)
            print("Current Task: {} --> Accuracy on Task-{} is {:.2f}  (Scenario: {})".format(
                current_task_index,
                task_id,
                acc,
                "CIL"
            ))
            accuracies.append(acc)
        return accuracies
            
