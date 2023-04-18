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

from torchvision.models.resnet import resnet18
# torch.multiprocessing.set_sharing_strategy('file_system')
sys.setrecursionlimit(10000)




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
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))


def test_resnet18():
    output_layer_names = ['layer1.0.bn1', 'layer4.0', 'fc']
    in_tensor = torch.ones((2, 3, 224, 224))

    core_model = resnet18()
    wrapper = ModelWrapper(core_model, output_layer_names)
    y1, y2, y3 = wrapper(in_tensor)
    assert y1.shape[0] == 2
    assert y1.shape[2] == 56
    assert y2.shape[2] == 7
    assert y3.shape[1] == 1000



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
            return output_vals[0]
        else:
            return output_vals

class Remind(NaiveContinualLearner):

    def __init__(self, args: Namespace, backbone: torch.nn.Module, scenario: GenericCLScenario, task2classes: Dict):
        super(Remind, self).__init__(args, backbone, scenario, task2classes)

        args.classifier_F ='ResNet18_StartAt_Layer4_1'
        args.classifier_ckpt = "/data/remind/best_ResNet18ClassifyAfterLayer4_1_100_orig.pth"
        args.classifier_G = 'ResNet18ClassifyAfterLayer4_1'
        # args.num_classes
        args.extract_features_from = 'model.layer4.0'
        args.start_lr = 0.1
        args.end_lr = 0.001
        args.batch_size = 32
        args.max_buffer_size = 9595
        # args.num_channels,
        # args.spatial_feat_dim

        args.num_codebooks = 32
        args.codebook_size = 256


        self.args = args

        self.device = args.device
        
        # make the classifier
        classifier_F = self.build_classifier(args.classifier_F, args.classifier_ckpt, num_classes=args.num_classes)
        core_model = self.build_classifier(args.classifier_G, args.classifier_ckpt, num_classes=None)
        self.classifier_G = ModelWrapper(core_model, output_layer_names=[args.extract_features_from], return_single=True)

        # make the optimizer
        trainable_params = []
        for k, v in classifier_F.named_parameters():
            trainable_params.append({'params': v, 'lr': args.start_lr})

        self.optimizer = optim.SGD(trainable_params, momentum=0.9, weight_decay=1e-5)

    
    def build_classifier(self, classifier, classifier_ckpt, num_classes):
        classifier = eval(classifier)(num_classes=num_classes)

        if classifier_ckpt is None:
            print("Will not resume any checkpoints!")
        else:
            resumed = torch.load(classifier_ckpt , map_location=torch.device('cpu'))
            if 'state_dict' in resumed:
                state_dict_key = 'state_dict'
            else:
                state_dict_key = 'model_state'
            print("Resuming with {}".format(classifier_ckpt))
            self.safe_load_dict(classifier, resumed[state_dict_key], should_resume_all_params=True)
        return classifier
    

    def safe_load_dict(self , model, new_model_state, should_resume_all_params=False):
        old_model_state = model.state_dict()
        c = 0
        if should_resume_all_params:
            for old_name, old_param in old_model_state.items():
                assert old_name in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(
                    old_name)
        for name, param in new_model_state.items():
            n = name.split('.')
            beg = n[0]
            end = n[1:]
            if beg == 'module':
                name = '.'.join(end)
            if name not in old_model_state:
                # print('%s not found in old model.' % name)
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            c += 1
            if old_model_state[name].shape != param.shape:
                print('Shape mismatch...ignoring %s' % name)
                continue
            else:
                old_model_state[name].copy_(param)
        if c == 0:
            raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')

    
    def get_trainable_params(self, classifier, start_lr):
        trainable_params = []
        for k, v in classifier.named_parameters():
            trainable_params.append({'params': v, 'lr': start_lr})
        return trainable_params


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
            print(batch_ix)
            batch_feats = model(batch_x) #.cuda())
            end_ix = start_ix + len(batch_feats)
            features_data[start_ix:end_ix] = batch_feats.cpu().numpy()
            labels_data[start_ix:end_ix] = np.atleast_2d(batch_y.numpy().astype(np.int)).transpose()
            item_ixs_data[start_ix:end_ix] = np.atleast_2d(batch_item_ixs.numpy().astype(np.int)).transpose()
            start_ix = end_ix
        return features_data, labels_data, item_ixs_data



    def fit_pq(feats_base_init, labels_base_init, item_ix_base_init, num_codebooks,
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

        self.train_loader = DataLoader(train_dataset.dataset, batch_size = self.args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset.dataset, batch_size = self.args.batch_size, shuffle=False)


        args = self.args

        feat_data, label_data, item_ix_data = self.extract_features(self.classifier_G, self.train_loader,
                                                                                len(self.train_loader.dataset))
        

        self.pq, self.latent_dict, self.rehearsal_ixs, self.class_id_to_item_ix_dict = self.fit_pq(feat_data, label_data, item_ix_data,
                                                                            args.num_codebooks,
                                                                            args.codebook_size)

    # Overwrite this method
    def learn_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # fit model with rehearsal
        
        self.fit_incremental_batch(self.train_loader, self.latent_dict, self.pq, rehearsal_ixs=self.rehearsal_ixs,
                                        class_id_to_item_ix_dict=self.class_id_to_item_ix_dict,
                                    )
        self.predict(self.val_loader, self.pq)
    


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

        ongoing_class = None

        # put classifiers on GPU and set plastic portion of network to train
        classifier_F = self.classifier_F.to(self.device) #.cuda()
        classifier_F.train()
        classifier_G = self.classifier_G.to(self.device) #.cuda()
        classifier_G.eval()

        counter = 0 #track how many samples are in buffer
        total_loss = 0
        c = 0
        for batch_images, batch_labels, batch_item_ixs in curr_loader:

            # get features from G and latent codes from PQ
            
            # data_batch = classifier_G(batch_images.cuda()).cpu().numpy()
            data_batch = classifier_G(batch_images).to(self.device).numpy()

            data_batch = np.transpose(data_batch, (0, 2, 3, 1))
            data_batch = np.reshape(data_batch, (-1, self.num_channels))
            codes = pq.compute_codes(data_batch)
            codes = np.reshape(codes, (-1, self.num_feats, self.num_feats, self.num_codebooks))

            # train REMIND on one new sample at a time
            for x, y, item_ix in zip(codes, batch_labels, batch_item_ixs):
                if self.lr_mode == 'step_lr_per_class' and (ongoing_class is None or ongoing_class != y):
                    ongoing_class = y

                if self.use_mixup:
                    # gather two batches of previous data for mixup and replay
                    data_codes = np.empty(
                        (2 * self.num_samples + 1, self.num_feats, self.num_feats, self.num_codebooks),
                        dtype=np.uint8)
                    data_labels = torch.empty((2 * self.num_samples + 1), dtype=torch.int).to(self.device) #.cuda()
                    data_codes[0] = x
                    data_labels[0] = y
                    ixs = randint(len(rehearsal_ixs), 2 * self.num_samples)
                    ixs = [rehearsal_ixs[_curr_ix] for _curr_ix in ixs]
                    for ii, v in enumerate(ixs):
                        data_codes[ii + 1] = latent_dict[v][0]
                        data_labels[ii + 1] = torch.from_numpy(latent_dict[v][1])

                    # reconstruct/decode samples with PQ
                    data_codes = np.reshape(data_codes, (
                        (2 * self.num_samples + 1) * self.num_feats * self.num_feats, self.num_codebooks))
                    data_batch_reconstructed = pq.decode(data_codes)
                    data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                          (-1, self.num_feats, self.num_feats,
                                                           self.num_channels))
                    data_batch_reconstructed = torch.from_numpy(
                        np.transpose(data_batch_reconstructed, (0, 3, 1, 2))) #.cuda()

                    # perform random resize crop augmentation on each tensor
                    if self.use_random_resize_crops:
                        transform_data_batch = torch.empty_like(data_batch_reconstructed)
                        for tens_ix, tens in enumerate(data_batch_reconstructed):
                            transform_data_batch[tens_ix] = self.random_resize_crop(tens)
                        data_batch_reconstructed = transform_data_batch

                    # MIXUP: Do mixup between two batches of previous data
                    x_prev_mixed, prev_labels_a, prev_labels_b, lam = self.mixup_data(
                        data_batch_reconstructed[1:1 + self.num_samples],
                        data_labels[1:1 + self.num_samples],
                        data_batch_reconstructed[1 + self.num_samples:],
                        data_labels[1 + self.num_samples:],
                        alpha=self.mixup_alpha)

                    data = torch.empty((self.num_samples + 1, self.num_channels, self.num_feats, self.num_feats))
                    data[0] = data_batch_reconstructed[0]
                    data[1:] = x_prev_mixed.clone()
                    labels_a = torch.zeros(self.num_samples + 1).long()
                    labels_b = torch.zeros(self.num_samples + 1).long()
                    labels_a[0] = y.squeeze()
                    labels_b[0] = y.squeeze()
                    labels_a[1:] = prev_labels_a
                    labels_b[1:] = prev_labels_b

                    # fit on replay mini-batch plus new sample
                    
                    # output = classifier_F(data.cuda())
                    output = classifier_F(data.to(self.device))
                    
                    loss = self.mixup_criterion(self.criterion, output, labels_a.to(self.device), labels_b.to(self.device), lam)
                else:
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
                    data_batch_reconstructed = torch.from_numpy(
                        np.transpose(data_batch_reconstructed, (0, 3, 1, 2))) #.cuda()

                    # perform random resize crop augmentation on each tensor
                    if self.use_random_resize_crops:
                        transform_data_batch = torch.empty_like(data_batch_reconstructed)
                        for tens_ix, tens in enumerate(data_batch_reconstructed):
                            transform_data_batch[tens_ix] = self.random_resize_crop(tens)
                        data_batch_reconstructed = transform_data_batch

                    # fit on replay mini-batch plus new sample
                    output = classifier_F(data_batch_reconstructed)
                    loss = self.criterion(output, data_labels)

                loss = loss.mean()
                self.optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
                loss.backward()

                # if gradient clipping is desired
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(classifier_F.parameters(), self.grad_clip)

                self.optimizer.step()

                total_loss += loss.item()
                c += 1

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

                # update lr scheduler
                if self.lr_scheduler_per_class is not None:
                    self.lr_scheduler_per_class[int(y)].step()

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

        val_loader = DataLoader(val_dataset.dataset, batch_size = self.args.batch_size, shuffle=False)

        _, probas, y_test = self.predict(val_loader, self.pq)
      

    def predict(self, data_loader, pq):
        """
        Perform inference with REMIND.
        :param data_loader: data loader of test images (images, labels)
        :param pq: trained PQ model
        :return: (label predictions, probabilities, ground truth labels)
        """
        with torch.no_grad():
            self.classifier_F.eval()
            self.classifier_F #.cuda()
            self.classifier_G.eval()
            self.classifier_G #.cuda()

            probas = torch.zeros((len(data_loader.dataset), self.num_classes))
            all_lbls = torch.zeros((len(data_loader.dataset)))
            start_ix = 0
            for batch_ix, batch in enumerate(data_loader):
                batch_x, batch_lbls = batch[0], batch[1]
                batch_x = batch_x #.cuda()

                # get G features
                data_batch = self.classifier_G(batch_x).cpu().numpy()

                # quantize test data so features are in the same space as training data
                data_batch = np.transpose(data_batch, (0, 2, 3, 1))
                data_batch = np.reshape(data_batch, (-1, self.num_channels))
                codes = pq.compute_codes(data_batch)
                data_batch_reconstructed = pq.decode(codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                      (-1, self.num_feats, self.num_feats, self.num_channels))
                data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))) #.cuda()

                batch_lbls = batch_lbls #.cuda()
                logits = self.classifier_F(data_batch_reconstructed)
                end_ix = start_ix + len(batch_x)
                probas[start_ix:end_ix] = F.softmax(logits.data, dim=1)
                all_lbls[start_ix:end_ix] = batch_lbls.squeeze()
                start_ix = end_ix

            preds = probas.data.max(1)[1]

        return preds.numpy(), probas.numpy(), all_lbls.int().numpy()