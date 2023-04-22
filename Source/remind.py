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
from collections import defaultdict

import argparse
import time
import torch.optim as optim
import sys
import random
import faiss

# torch.multiprocessing.set_sharing_strategy('file_system')
sys.setrecursionlimit(10000)



import copy

# ClassifierF
class ResNet18_StartAt_Layer4_1(nn.Module):
    def __init__(self, backbone):
        super(ResNet18_StartAt_Layer4_1, self).__init__()
        self.model = copy.deepcopy(backbone)

        del self.model.conv1
        del self.model.bn1
        del self.model.layer1
        del self.model.layer2
        del self.model.layer3
        del self.model.layer4[0]

    def forward(self, x):
        out = self.model.layer4(x)
        out = F.avg_pool2d(out, out.size()[3])
        final_embedding = out.view(out.size(0), -1)
        out = self.model.fc(final_embedding)
        return out , None # , final_embedding

# ClassifierG

class BaseResNet18ClassifyAfterLayer4(nn.Module):
    def __init__(self, backbone , num_del=0):
        super(BaseResNet18ClassifyAfterLayer4, self).__init__()
        self.model = copy.deepcopy(backbone)
        for _ in range(0, num_del):
            del self.model.layer4[-1]
    def forward(self, x):
        out = self.model(x)
        return out, None


class ResNet18ClassifyAfterLayer4_1(BaseResNet18ClassifyAfterLayer4):
    def __init__(self, backbone ,  num_classes=None):
        super(ResNet18ClassifyAfterLayer4_1, self).__init__(backbone , num_del=0)




def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """

    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    # print(model)
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))


def randint(max_val, num_samples):
    """
    return num_samples random integers in the range(max_val)
    """
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break

        if len(rand_vals) >= _num_samples:
            break
    return rand_vals.keys()



class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        if self.return_single:
            return output_vals[0], None
        else:
            return output_vals, None





class Remind(NaiveContinualLearner):

    def __init__(self, args: Namespace, backbone: torch.nn.Module, scenario: GenericCLScenario, task2classes: Dict):
        super(Remind, self).__init__(args, backbone, scenario, task2classes)

        self.classifier_F ='ResNet18_StartAt_Layer4_1'
        self.classifier_ckpt = "/data/remind/best_ResNet18ClassifyAfterLayer4_1_100_orig.pth"
        self.classifier_G = 'ResNet18ClassifyAfterLayer4_1'
        self.extract_features_from = 'layer4.1'
        self.start_lr = 0.1
        self.end_lr = 0.001
        self.batch_size = args.batch_size
        self.max_buffer_size = 9595
        
        self.num_feats = 7 # spatial_feat_dim
        self.num_codebooks = 32
        self.codebook_size = 256
        self.num_channels = 512

        self.num_samples = 50
        self.mixup_alpha = 0.1

        self.lr_mode = 'step_lr_per_class'
        self.lr_step_size = 100
        
        
        self.REPLAY_SAMPLES=50

        self.num_classes = args.num_classes

        self.use_mixup = False
        self.use_random_resize_crops = False


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # make the classifier
        self.classifier_F = ResNet18_StartAt_Layer4_1(backbone = backbone).to(self.device)
        core_model = ResNet18ClassifyAfterLayer4_1(backbone = backbone).to(self.device)
        self.classifier_G = ModelWrapper(core_model.model, output_layer_names=[self.extract_features_from], return_single=True)

        # make the optimizer
        trainable_params = []
        for k, v in self.classifier_F.named_parameters():
            trainable_params.append({'params': v, 'lr': self.start_lr})

        self.optimizer = optim.SGD(trainable_params, momentum=0.9, weight_decay=1e-5)


    def extract_features(self, model, data_loader, data_len, num_channels=512, spatial_feat_dim=7):
        """
        Extract image features and put them into arrays.
        :param model: pre-trained model to extract features
        :param data_loader: data loader of images for which we want features (images, labels, item_ixs)
        :param data_len: number of images for which we want features
        :param num_channels: number of channels in desired features
        :param spatial_feat_dim: spatial dimension of desired features
        :return: numpy arrays of features, labels, item_ixs
        """

        model.eval()
        model.to(self.device) 

        # allocate space for features and labels
        features_data = np.empty((data_len, num_channels, spatial_feat_dim, spatial_feat_dim), dtype=np.float32)
        labels_data = np.empty((data_len, 1), dtype=np.int)
        item_ixs_data = np.empty((data_len, 1), dtype=np.int)

        # put features and labels into arrays
        start_ix = 0
        for batch_ix, (batch_x, batch_y, batch_item_ixs) in enumerate(data_loader):
            # print(batch_ix)
            batch_feats , _ = model(batch_x.to(self.device))
            end_ix = start_ix + len(batch_feats)
            features_data[start_ix:end_ix] = batch_feats.detach().cpu().numpy()
            labels_data[start_ix:end_ix] = np.atleast_2d(batch_y.numpy().astype(np.int)).transpose()
            item_ixs_data[start_ix:end_ix] = np.atleast_2d(batch_item_ixs.numpy().astype(np.int)).transpose()
            start_ix = end_ix
        return features_data, labels_data, item_ixs_data



    def fit_pq(self, feats_base_init, labels_base_init, item_ix_base_init, num_codebooks,
            codebook_size, num_channels=512, spatial_feat_dim=7 , batch_size=128):
        """
        Fit the PQ model and then quantize and store the latent codes of the data used to train the PQ in a dictionary to 
        be used later as a replay buffer.
        :param feats_base_init: numpy array of base init features that will be used to train the PQ
        :param labels_base_init: numpy array of the base init labels used to train the PQ
        :param item_ix_base_init: numpy array of the item_ixs used to train the PQ
        :param num_channels: number of channels in desired features
        :param spatial_feat_dim: spatial dimension of desired features
        :param num_codebooks: number of codebooks for PQ
        :param codebook_size: size of each codebook for PQ
        :param batch_size: batch size used to extract PQ features
        :return: (trained PQ object, dictionary of latent codes, list of item_ixs for latent codes, dict of visited classes
        and associated item_ixs)
        """

        counter = 0 #count how many latent codes are in the replay buffer/dict

        train_data_base_init = np.transpose(feats_base_init, (0, 2, 3, 1))
        train_data_base_init = np.reshape(train_data_base_init, (-1, num_channels))
        num_samples = len(train_data_base_init)

        print('\nTraining Product Quantizer')
        start = time.time()
        nbits = int(np.log2(codebook_size))
        pq = faiss.ProductQuantizer(num_channels, num_codebooks, nbits)
        pq.train(train_data_base_init)
        print("Completed in {} secs".format(time.time() - start))
        del train_data_base_init

        print('\nEncoding and Storing Base Init Codes')
        start_time = time.time()
        latent_dict = {}
        class_id_to_item_ix_dict = defaultdict(list)
        rehearsal_ixs = []
        mb = min(batch_size, num_samples)
        for i in range(0, num_samples, mb):
            start = i
            end = min(start + mb, num_samples)
            data_batch = feats_base_init[start:end]
            batch_labels = labels_base_init[start:end]
            batch_item_ixs = item_ix_base_init[start:end]

            data_batch = np.transpose(data_batch, (0, 2, 3, 1))
            data_batch = np.reshape(data_batch, (-1, num_channels))
            codes = pq.compute_codes(data_batch)
            codes = np.reshape(codes, (-1, spatial_feat_dim, spatial_feat_dim, num_codebooks))

            # put codes and labels into buffer (dictionary)
            for j in range(len(batch_labels)):
                ix = int(batch_item_ixs[j])
                latent_dict[ix] = [codes[j], batch_labels[j]]
                rehearsal_ixs.append(ix)
                class_id_to_item_ix_dict[int(batch_labels[j])].append(ix)
                counter += 1

        print("Completed in {} secs".format(time.time() - start_time))
        return pq, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict
    

    # Overwrite this method
    def begin_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        
        print("Getting dataloaders")

        self.train_loader = DataLoader(train_dataset.dataset, batch_size = self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset.dataset, batch_size = self.batch_size, shuffle=False)


        print("Extracting features")
        feat_data, label_data, item_ix_data = self.extract_features(self.classifier_G, self.train_loader,
                                                                                len(self.train_loader.dataset))
        

        print("Training feature quantizer")
        self.pq, self.latent_dict, self.rehearsal_ixs, self.class_id_to_item_ix_dict = self.fit_pq(feat_data, label_data, item_ix_data,
                                    num_codebooks = self.num_codebooks, codebook_size= self.codebook_size, 
                                    num_channels=self.num_channels, spatial_feat_dim=self.num_feats , 
                                    batch_size=self.batch_size)

    # Overwrite this method
    def learn_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # fit model with rehearsal
        # pass
        self.fit_incremental_batch(self.train_loader, self.latent_dict, self.pq, rehearsal_ixs=self.rehearsal_ixs,
                                        class_id_to_item_ix_dict=self.class_id_to_item_ix_dict)
        


    def fit_incremental_batch(self, curr_loader, latent_dict, pq, rehearsal_ixs=None, class_id_to_item_ix_dict=None,
                            verbose=True):
        """
        Fit REMIND on samples from a data loader one at a time.
        :param curr_loader: the data loader of new samples to be fit (returns (images, labels, item_ixs)
        :param latent_dict: dictionary containing latent codes for replay samples
        :param pq: trained PQ object for decoding latent codes
        :param rehearsal_ixs: list of item_ixs eligible for replay
        :param class_id_to_item_ix_dict: dictionary of visited classes with associated item_ixs visited
        :param verbose: true for printing loss to console
        :return: None
        """
        print("Starting training session")

        # put classifiers on GPU and set plastic portion of network to train
        classifier_F = self.classifier_F.train().to(self.device) #.cuda()
        classifier_G = self.classifier_G.eval().to(self.device) #.cuda()

        criterion = nn.CrossEntropyLoss(reduction='none')
        
        counter = 0 #track how many samples are in buffer
        total_loss = 0
        for batch_images, batch_labels, batch_item_ixs in curr_loader:
            print(counter)
            # get features from G and latent codes from PQ
            data_batch , _ = classifier_G(batch_images.to(self.device))
            data_batch = data_batch.cpu().numpy()

            data_batch = np.transpose(data_batch, (0, 2, 3, 1))
            data_batch = np.reshape(data_batch, (-1, self.num_channels))
            codes = pq.compute_codes(data_batch)
            codes = np.reshape(codes, (-1, self.num_feats, self.num_feats, self.num_codebooks))

            # train REMIND on one new sample at a time
            for x, y, item_ix in zip(codes, batch_labels, batch_item_ixs):
               
                # gather previous data for replay
                data_codes = np.empty(
                    (self.num_samples + 1, self.num_feats, self.num_feats, self.num_codebooks),
                    dtype=np.uint8)
                data_labels = torch.empty((self.num_samples + 1), dtype=torch.long).to(self.device) #.cuda()
                data_codes[0] = x
                data_labels[0] = y
                ixs = randint(len(rehearsal_ixs), self.num_samples)
                ixs = [rehearsal_ixs[_curr_ix] for _curr_ix in ixs]
                for ii, v in enumerate(ixs):
                    data_codes[ii + 1] = latent_dict[v][0]
                    data_labels[ii + 1] = torch.from_numpy(latent_dict[v][1])

                # reconstruct/decode samples with PQ
                data_codes = np.reshape(data_codes, (
                    (self.num_samples + 1) * self.num_feats * self.num_feats, self.num_codebooks))
                data_batch_reconstructed = pq.decode(data_codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                        (-1, self.num_feats, self.num_feats,
                                                        self.num_channels))
                data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))) #.cuda()

                # fit on replay mini-batch plus new sample
                self.optimizer.zero_grad()
                output, _ = classifier_F(data_batch_reconstructed.to(self.device))
                loss = criterion(output, data_labels)
                loss = loss.mean()
                # zero out grads before backward pass because they are accumulated
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                # c += 1

                # since we have visited item_ix, it is now eligible for replay
                rehearsal_ixs.append(int(item_ix.numpy()))
                latent_dict[int(item_ix.numpy())] = [x, y.numpy()]
                class_id_to_item_ix_dict[int(y.numpy())].append(int(item_ix.numpy()))

                # if buffer is full, randomly replace previous example from class with most samples
                if self.max_buffer_size is not None and counter >= self.max_buffer_size:
                    # class with most samples and random item_ix from it
                    max_key = max(class_id_to_item_ix_dict, key=lambda x: len(class_id_to_item_ix_dict[x]))
                    max_class_list = class_id_to_item_ix_dict[max_key]
                    rand_item_ix = random.choice(max_class_list)

                    # remove the random_item_ix from all buffer references
                    max_class_list.remove(rand_item_ix)
                    latent_dict.pop(rand_item_ix)
                    rehearsal_ixs.remove(rand_item_ix)
                else:
                    counter += 1


    def mixup_data(self, x1, y1, x2, y2, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_x = lam * x1 + (1 - lam) * x2
        y_a, y_b = y1, y2
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a.squeeze()) + (1 - lam) * criterion(pred, y_b.squeeze())

    # Overwrite this method
    def end_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # perform inference
        # test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
        #                                 batch_size=args.batch_size)
        pass



    def _test(self, data_loader: DataLoader, class_in_this_task = None) -> float:
        
        with torch.no_grad():
            # put classifiers on GPU and set plastic portion of network to train
            classifier_F = self.classifier_F.train().to(self.device) 
            classifier_G = self.classifier_G.eval().to(self.device) 

            correct_predictions = 0
            total_predictions = 0
            
            for batch_images, batch_labels, batch_item_ixs in data_loader:
                # get features from G and latent codes from PQ
                
                # data_batch = classifier_G(batch_images.cuda()).cpu().numpy()
                data_batch , _ = classifier_G(batch_images.to(self.device))
                data_batch = data_batch.cpu().numpy()

                data_batch = np.transpose(data_batch, (0, 2, 3, 1))
                data_batch = np.reshape(data_batch, (-1, self.num_channels))
                codes = self.pq.compute_codes(data_batch)
                codes = np.reshape(codes, (-1, self.num_feats, self.num_feats, self.num_codebooks))
                # codes = torch.from_numpy(codes).to(self.device) #.cuda(
                
                # reconstruct/decode samples with PQ
                # data_codes = np.reshape(data_codes, ((self.num_samples + 1) * self.num_feats * self.num_feats, self.num_codebooks))
                data_batch_reconstructed = self.pq.decode(codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                        (-1, self.num_feats, self.num_feats,
                                                        self.num_channels))
                data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))) #.cuda()

                # fit on replay mini-batch plus new sample
                output, _ = classifier_F(data_batch_reconstructed)

                if class_in_this_task:
                    output_temp = torch.zeros_like(output)
                    output_temp[:, class_in_this_task] = output[:, class_in_this_task]
                    output = output_temp
                predicted_class = output.argmax(dim=1)
                correct_predictions += (predicted_class == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)
            
            return correct_predictions / total_predictions
