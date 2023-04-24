from argparse import Namespace
from torch.utils.data import DataLoader
import torch
import numpy as np
from typing import Dict, List
from avalanche.benchmarks import GenericCLScenario, TCLExperience
from Source.naive_continual_learner import NaiveContinualLearner
import torch.nn as nn

from Source.utils import get_device
from collections import defaultdict

import time
import torch.optim as optim
import sys
import random
import faiss

sys.setrecursionlimit(10000)
torch.use_deterministic_algorithms(True)

import copy
from tqdm import tqdm


class Remind(NaiveContinualLearner):

    def __init__(self, args: Namespace, backbone: torch.nn.Module, scenario: GenericCLScenario, task2classes: Dict):
        super(Remind, self).__init__(args, backbone, scenario, task2classes)

        # self.classifier_F ='ResNet18_StartAt_Layer4_1'
        # self.classifier_G = 'ResNet18ClassifyAfterLayer4_1'
        # self.extract_features_from = 'layer4.1'

        self.extract_features_from = 'model.15'


        self.lr = args.remind_learning_rate
        self.batch_size = args.batch_size

        self.max_buffer_size = int(args.memory_per_class * args.num_classes)#args.max_buffer_size
        self.spatial_feat_dim = args.spatial_feat_dim# spatial_feat_dim
        self.num_codebooks = args.num_codebooks
        self.codebook_size = args.codebook_size
        self.num_channels = args.num_channels
        self.num_samples =  min(int(args.replay_percentage * self.max_buffer_size ) ,50)

        if args.overfit_batches != None:
            self.overfit_batches = args.overfit_batches
        else:
            self.overfit_batches = 100000


        if args.pretrain_overfit_batches != None:
            self.pretrain_overfit_batches = args.pretrain_overfit_batches
        else:
            self.pretrain_overfit_batches = 100000

        self.max_pretrain_epoch = args.pretrain_epochs
        
        # self.REPLAY_SAMPLES=50

        self.num_classes = args.num_classes

        self.use_mixup = False
        self.use_random_resize_crops = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

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
        
        for batch_id  , (batch_x, batch_y , batch_idx , _)  in  enumerate(tqdm(data_loader , desc=f" Training Feature quantizer")):        
            batch_feats , _= model(batch_x.to(self.device))
            end_ix = start_ix + len(batch_feats)
            features_data[start_ix:end_ix] = batch_feats.detach().cpu().numpy()
            labels_data[start_ix:end_ix] = np.atleast_2d(batch_y.numpy().astype(np.int)).transpose()
            item_ixs_data[start_ix:end_ix] = np.atleast_2d(batch_idx.numpy().astype(np.int)).transpose()
            start_ix = end_ix
        return features_data, labels_data, item_ixs_data

    def fit_pq(self, feats_base_init, labels_base_init,item_ix_base_init, num_codebooks,
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
        
        print("current_task_index ", current_task_index)
        if current_task_index == 1:
            train_loader = DataLoader(train_dataset.dataset, batch_size = self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset.dataset, batch_size = self.batch_size, shuffle=False)

            # Train both models jointly
            print("Learning entire model together")
            self.fit_both_models_jointly(train_loader, val_loader , self.max_pretrain_epoch)
        else:
            pass

    # Overwrite this method
    def learn_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # fit model with rehearsal
        train_loader = DataLoader(train_dataset.dataset, batch_size = self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset.dataset, batch_size = self.batch_size, shuffle=False)

        print("Extracting features and Training feature quantizer")
        feat_data, label_data, item_ix_data = self.extract_features(self.classifier_G, train_loader,
                                                                                len(train_loader.dataset),
                                                                                num_channels=self.num_channels,
                                                                                spatial_feat_dim = self.spatial_feat_dim)        

        self.pq, self.latent_dict , self.rehearsal_ixs, self.class_id_to_item_ix_dict= self.fit_pq(feat_data, label_data, item_ix_data,
                                    num_codebooks = self.num_codebooks, codebook_size= self.codebook_size, 
                                    num_channels=self.num_channels, spatial_feat_dim=self.spatial_feat_dim , 
                                    batch_size=self.batch_size)

        self.fit_incremental_batch(train_loader, self.latent_dict, self.pq, rehearsal_ixs=self.rehearsal_ixs,
                                        class_id_to_item_ix_dict=self.class_id_to_item_ix_dict,
                                    current_task_index= current_task_index)
        
    def fit_both_models_jointly(self, train_loader, val_loader, max_pretrain_epoch):
        """
        Fit entire network together and divide it into classifier F and classfier G: Done only during first task
        :return: None
        """
        model = self.backbone
        optimizer = optim.Adam(model.parameters(), lr=self.lr) #, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(reduction='none')

        print("Starting model pretraining") 
        
        if self.args.pretrain_overfit_batches != None:
            print(f" Overfiting {self.pretrain_overfit_batches} batches, every epoch")
                    
        for epoch_id  in range(max_pretrain_epoch):
            model.train()
            for batch_id  , (images, target, idx, _)  in  enumerate(tqdm(train_loader , desc=f" Train Epoch {epoch_id}/{self.args.pretrain_epochs}")):        
                if batch_id >= self.pretrain_overfit_batches and self.args.pretrain_overfit_batches != None:
                    break
                else:
                    # print(batch_id, self.overfit_batches)                    
                    images = images.to(self.device)
                    target =target.to(self.device)
                    output, _ = model(images)
                    train_loss = criterion(output, target)
                    optimizer.zero_grad()
                    train_loss = train_loss.mean()
                    train_loss.backward()
                    optimizer.step()

            # switch to evaluate mode
            model.eval()
            with torch.no_grad():
                for batch_id  , (images, target, idx, _)  in  enumerate(tqdm(val_loader , desc=f" Val Epoch {epoch_id}/{self.args.pretrain_epochs}")):        
                # for images, target, idx , _ in val_loader:
                    if batch_id >= self.pretrain_overfit_batches and self.args.pretrain_overfit_batches != None:
                        break
                    else:
                        images = images.to(self.device)
                        target =target.to(self.device)
                        output, _= model(images)
                        val_loss = criterion(output, target)
                        val_loss = val_loss.mean()                
            print(f"Epoch {epoch_id}: Train loss {train_loss}, Val loss  {val_loss}")


        self.backbone = model
        print("Dividing model in fixed and plastic part")
        self.classifier_G = model.head.to(self.device)
        self.classifier_F = model.tail.to(self.device)


    def fit_incremental_batch(self, curr_loader, latent_dict, pq, rehearsal_ixs=None, class_id_to_item_ix_dict=None,
                            verbose=True , current_task_index=0):
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
        classifier_G = self.classifier_G.to(self.device) #.cuda()
        classifier_G.eval()
        classifier_F = self.classifier_F.to(self.device) #.cuda()
        classifier_F.train()

        optimizer = optim.Adam(classifier_F.parameters(), lr = self.lr) #, weight_decay=1e-5)

        criterion = nn.CrossEntropyLoss(reduction='none')
        
        counter = 0 #track how many samples are in buffer
        total_loss = 0

        for epoch_id  in range(self.args.epochs):
            for batch_id , (batch_images, batch_labels, batch_item_ixs, _)  in  enumerate(tqdm(curr_loader , desc=f" Epoch {epoch_id}/{self.args.epochs}")):
                if batch_id >= self.overfit_batches and self.args.overfit_batches != None:
                    print(f" Overfiting {self.overfit_batches} batches, every epoch")
                    break
                else:
                    # get features from G and latent codes from PQ
                    batch_images = batch_images.to(self.device)
                    data_batch , _= classifier_G(batch_images)
                    data_batch = data_batch.detach().cpu().numpy()

                    data_batch = np.transpose(data_batch, (0, 2, 3, 1))
                    data_batch = np.reshape(data_batch, (-1, self.num_channels))
                    codes = pq.compute_codes(data_batch)
                    codes = np.reshape(codes, (-1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_codebooks))

                    # train REMIND on one new sample at a time
                    for x, y, item_ix in zip(codes, batch_labels, batch_item_ixs):
                        # gather previous data for replay
                        data_codes = np.empty( (self.num_samples + 1, self.spatial_feat_dim, self.spatial_feat_dim, self.num_codebooks),dtype=np.uint8)
                        data_labels = torch.empty((self.num_samples + 1), dtype=torch.long).to(self.device) #.cuda()
                        data_codes[0] = x
                        data_labels[0] = y
                        
                        ixs = np.random.choice(list(latent_dict.keys()), size=self.num_samples) 

                        for ii, v in enumerate(ixs):
                            data_codes[ii + 1] = latent_dict[v][0]
                            data_labels[ii + 1] = torch.from_numpy(latent_dict[v][1])
                        
                        # reconstruct/decode samples with PQ
                        data_codes = np.reshape(data_codes, ((self.num_samples + 1) * self.spatial_feat_dim * self.spatial_feat_dim, self.num_codebooks))
                        data_batch_reconstructed = pq.decode(data_codes)
                        data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                                (-1, self.spatial_feat_dim, self.spatial_feat_dim,
                                                                self.num_channels))
                        data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))) #.cuda()


                        # fit on replay mini-batch plus new sample
                        output , _ = classifier_F(data_batch_reconstructed.to(self.device))
                    
                        # print(output, data_labels)
                    
                        loss = criterion(output, data_labels)

                        loss = loss.mean()
                        optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

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
                            # print("removing", rand_item_ix)

                            # remove the random_item_ix from all buffer references
                            max_class_list.remove(rand_item_ix)
                            try:
                                latent_dict.pop(rand_item_ix)
                            except:
                                pass
                            rehearsal_ixs.remove(rand_item_ix)
                        else:
                            counter += 1

            print("epoch" , epoch_id , "train loss" , loss.item())


    # Overwrite this method
    def end_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # perform inference
        val_loader = DataLoader(val_dataset.dataset, batch_size = self.batch_size, shuffle=False)
        self._test(val_loader)
        pass

    def _test(self, data_loader: DataLoader, class_in_this_task = None) -> float:
        
        with torch.no_grad():
            # put classifiers on GPU and set plastic portion of network to train
            classifier_F = self.classifier_F.train().to(self.device) 
            classifier_G = self.classifier_G.eval().to(self.device) 

            correct_predictions = 0
            total_predictions = 0
            
            for batch_images, batch_labels, batch_item_ixs, _ in data_loader:
                # get features from G and latent codes from PQ
                data_batch , _ = classifier_G(batch_images.to(self.device))
                data_batch = data_batch.cpu().numpy()

                data_batch = np.transpose(data_batch, (0, 2, 3, 1))
                data_batch = np.reshape(data_batch, (-1, self.num_channels))
                codes = self.pq.compute_codes(data_batch)
                data_batch_reconstructed = self.pq.decode(codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                        (-1, self.spatial_feat_dim, self.spatial_feat_dim,
                                                        self.num_channels))
                data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))) #.cuda()

                # fit on replay mini-batch plus new sample
                output, _ = classifier_F(data_batch_reconstructed.to(self.device))

                if class_in_this_task:
                    output_temp = torch.zeros_like(output)
                    output_temp[:, class_in_this_task] = output[:, class_in_this_task]
                    output = output_temp
                predicted_class = output.argmax(dim=1)
                correct_predictions += (predicted_class == batch_labels.to(self.device)).sum().item()
                total_predictions += batch_labels.size(0)
            
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

