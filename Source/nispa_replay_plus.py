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


class NispaReplayPlus(NaiveContinualLearner):

    def __init__(self, args: Namespace, backbone: torch.nn.Module, scenario: GenericCLScenario, task2classes: Dict):
        super(NispaReplayPlus, self).__init__(args, backbone, scenario, task2classes)
        self.args = args
        self.backbone = random_prune(backbone, args.prune_perc) # prune backbone before training
        self.scenario = scenario
        self.task2classes = task2classes
        self.memory = MemoryBuffer(args, task2classes, self.backbone.input_size) # type: ignore
        self.tau_schedule = lambda t: 0.5 * (1 + math.cos(t * math.pi / 40.0)) # cosine annealing based schedule
        
        # Unit types
        self.classes_seen_so_far = []
        self.current_plastic_units = [[]] + [list(range(param.shape[0])) for param in backbone.parameters() if len(param.shape) != 1]
        self.current_candidate_units = [[] for param in backbone.parameters() if len(param.shape) != 1] + [[]]
        self.current_stable_units = copy.deepcopy(self.current_candidate_units)
        self.current_stable_units[0] = list(range(next(self.backbone.parameters()).shape[1])) # type: ignore
        self.freeze_masks = None


    # Overwrite this method
    def begin_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        self.classes_seen_so_far = list(set(self.classes_seen_so_far + train_dataset.classes_in_this_experience))
        self.current_stable_units[-1] = self.classes_seen_so_far
        self.current_task_index = current_task_index
        self.current_plastic_units[-1] = list(set(range(self.backbone.classifier.weight.shape[0])) - set(self.classes_seen_so_far))


    def learn_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # Create optimizer and loss
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim_obj = getattr(torch.optim, self.args.optimizer)
        print("****** Learning Task-{}   Classes: {} ******".format(current_task_index, train_dataset.classes_in_this_experience))
        # Create data loaders
        train_loader = DataLoader(train_dataset.dataset, batch_size = self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset.dataset, batch_size = self.args.batch_size, shuffle=False)

        # Train
        for phase_index in range(1, self.args.num_phases + 1):
            print('Selecting Units')
            selection_perc = max(self.tau_schedule(phase_index) * 100, self.args.min_activation_perc)
            print('Selection percentage: {}'.format(selection_perc))
            plastic_units, candidate_units = self.select_candidate_stable_units(train_dataset, selection_perc) # type: ignore
            self.set_platic_units(plastic_units)
            self.set_candidate_untis(candidate_units)
            print('Dropping connections')
            connection_quota_grow = self.drop_plastic_to_others()
            # connection_quota_drop_mag = self.drop_mag_pruning()
            # Total connections to grow
            #connection_quota_grow = [drop1+drop2 for drop1, drop2 in zip(connection_quota_drop_plastic, connection_quota_drop_mag)]
            # Grow connections
            self.backbone, _ = self.grow_connections(connection_quota_grow)

            print("Sparsity phase-{}: {:.2f}".format(phase_index, _compute_weight_sparsity(self.backbone)))
            optimizer = self.optim_obj(self.backbone.parameters(), lr= self.args.learning_rate , weight_decay= 0.0)
            self.backbone = self.train_task(self.backbone, self.loss,
                                       optimizer, train_loader, val_loader, self.memory if current_task_index > 1 else None)

    def reinit_plastic_units(self):
        i = 0
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d): 
                m.weight.data[self.current_plastic_units[i+1],:] = nn.init.kaiming_normal_(m.weight.data[self.current_plastic_units[i+1],:],
                                                                                    mode='fan_out', nonlinearity='relu')
                m.bias.data[self.current_plastic_units[i+1]] = nn.init.constant_(m.bias.data[self.current_plastic_units[i+1]], 0.0) # type: ignore
                i += 1
        
    def end_task(self, train_dataset: TCLExperience, val_dataset: TCLExperience, current_task_index: int) -> None:
        # update stable units
        self.current_stable_units = [list(set(stable).union(candidate))
                                    for stable, candidate in zip(self.current_stable_units,self.current_candidate_units)]
        # update freeze masks
        self.update_freeze_masks()
        # update memory
        subsets = get_n_sample_per_class(train_dataset, self.args.memory_per_class)
        all_samples = [samples.to(get_device()) for samples, _ in subsets]
        labels = [label for _, label in subsets]
        self.memory.insert_samples(all_samples, labels)

        # Remaning plastic units counts
        print("Remaining plastic units: {}".format([len(plastic) for plastic in self.current_plastic_units]))


    
    
    def train_task(self, backbone, loss, optimizer, train_loader, val_loader, memory):
        for _ in range(self.args.phase_epochs):
            backbone.train()
            epoch_output_loss = []
            for data, target, _ in train_loader:
                optimizer.zero_grad()
                data, target = data.to(get_device()), target.to(get_device())
                stream_output, _ = backbone.forward(data)
                if memory is not None:
                    memo_samples, memo_labels = memory.sample_n(n = self.args.batch_size_memory,
                                                                current_task_index = self.current_task_index)
                    memo_samples, memo_labels = torch.tensor(memo_samples).to(get_device()), torch.tensor(memo_labels).to(get_device())
                    memo_output, _ = backbone.forward(memo_samples)
                    stream_output = torch.cat((stream_output, memo_output), dim=0)
                    target = torch.cat((target, memo_labels), dim=0)
                

                loss_value = loss(stream_output, target)
                loss_value.backward()
                if self.freeze_masks:
                    backbone = self.reset_frozen_gradients(backbone)
                optimizer.step()
                epoch_output_loss.append(loss_value.item())
            print("Average training loss: {}".format(sum(epoch_output_loss) / len(epoch_output_loss)))
        return backbone

    def reset_frozen_gradients(self, backbone: nn.Module) -> nn.Module:
        mask_index = 0
        for module in backbone.modules():
            if isinstance(module, nn.Linear) or  isinstance(module, nn.Conv2d):
                module.weight.grad[self.freeze_masks[mask_index][0]] = 0
                if module.bias_flag:
                    module.bias.grad[self.freeze_masks[mask_index][1]] = 0    # type: ignore
                mask_index = mask_index + 1
        return backbone

    def select_candidate_stable_units(self, train_task: TCLExperience, selection_perc: float) -> Tuple[List, List]:
        def _pick_top_neurons(average_layer_activation: NDArray[np.float32], stable_selection_perc: float) -> List[int]:
            total = sum(average_layer_activation)
            accumulate = 0
            indices = []
            sort_indices = np.argsort(-average_layer_activation)
            for index in sort_indices:
                accumulate = accumulate + average_layer_activation[index]
                indices.append(index)
                if accumulate >= total * stable_selection_perc / 100:
                    break
            return indices
        
        total_activations = []
        with torch.no_grad():
            train_loader = DataLoader(train_task.dataset, batch_size = self.args.batch_size, shuffle=True)
            for data, _, _ in train_loader:
                data = data.to(get_device())
                activations = [activation.detach().cpu().numpy()
                               for activation in self.backbone.forward_activations(data)] # type: ignore
                batch_sum_activation = [np.sum(activation, axis = (0, 2, 3)) if len(activation.shape) != 2 else  np.sum(activation, axis = 0)
                                    for activation in activations]  
                total_activations =  [total_activations[i]+activation
                                for i, activation in enumerate(batch_sum_activation)] if total_activations else batch_sum_activation
                
        selected_candidate_units = []
        for average_layer_activation in total_activations:
            selected_candidate_units.append(_pick_top_neurons(average_layer_activation, selection_perc))
        
        selected_candidate_units = selected_candidate_units
        selected_plastic_units = []
        for layer_index, candidates in enumerate(selected_candidate_units, 1):
            layer_stable_units = self.current_stable_units[layer_index]
            all_units = set(list(range(total_activations[layer_index-1].shape[0])))
            selected_plastic_units.append(list(all_units - set(layer_stable_units) - set(candidates)))

        selected_candidate_units = [[]] + selected_candidate_units + [[]]
        selected_plastic_units = [[]] + selected_plastic_units + [list(set(range(self.backbone.classifier.weight.shape[0])) - set(self.classes_seen_so_far))]

        return selected_plastic_units, selected_candidate_units

    

    def drop_plastic_to_others(self) -> List[int]:
        drop_masks = []
        connectivity_masks = [w for w, _ in self.backbone.get_weight_bias_masks_numpy()] # type: ignore
        plastic_units = self.current_plastic_units
        non_plastic_units = [list(set(stable).union(candidate))
                             for stable, candidate in zip(self.current_stable_units, self.current_candidate_units)]
        
        for i, (current_plastic, next_non_plastic) in enumerate(zip(plastic_units[:-1], non_plastic_units[1:])):
            drop_mask = np.zeros(connectivity_masks[i].shape, dtype=np.intc)
            if current_plastic:
                #Conv2Linear
                if len(connectivity_masks[i].shape) == 2 and len(connectivity_masks[i-1].shape) == 4:
                    for u0_index in current_plastic:
                        start = u0_index*self.backbone.conv2lin_size
                        end = (u0_index+1)*self.backbone.conv2lin_size
                        drop_mask[next_non_plastic, start:end] = 1
                else:
                    drop_mask[np.ix_(next_non_plastic, current_plastic)] = 1
            drop_masks.append(drop_mask * connectivity_masks[i])
        
        mask_index, num_drops = 0, []
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                weight_mask, bias_mask = module.get_mask() # type: ignore
                weight_mask[torch.tensor(drop_masks[mask_index], dtype= torch.bool)] = 0
                num_drops.append(int(np.sum(drop_masks[mask_index])))
                module.set_mask(weight_mask, bias_mask) # type: ignore
                mask_index += 1

        return num_drops

    def grow_connections(self, connection_quota_grow: List[int]):
        grower = Grower(self.backbone, self.current_stable_units,self.current_plastic_units, self.current_candidate_units)
        backbone, remaining_conns = grower.grow(self.backbone, connection_quota_grow)
        return backbone, remaining_conns


    def update_freeze_masks(self):
        weights = self.backbone.get_weight_bias_masks_numpy() # type: ignore
        freeze_masks = []
        list_stable_units = self.current_stable_units
        for i, target_stable in enumerate(list_stable_units[1:]):
            target_stable =  np.array(target_stable, dtype=np.int32)
            mask_w = np.zeros(weights[i][0].shape)
            mask_b = np.zeros(weights[i][1].shape)
            if len(target_stable) != 0:
                mask_w[target_stable, :] = 1
                mask_b[target_stable] = 1
            freeze_masks.append((mask_w * weights[i][0], mask_b))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        freeze_masks = [(torch.tensor(w).to(torch.bool).to(device),
                         torch.tensor(b).to(torch.bool).to(device))
                         for w, b in freeze_masks]
        self.freeze_masks = freeze_masks

    
    def set_platic_units(self, plastic_units):
        self.current_plastic_units = plastic_units

    def set_candidate_untis(self, candidate_units):
        self.current_candidate_units = candidate_units

    def set_stable_units(self, stable_units):
        self.current_stable_units = stable_units

class MemoryBuffer():

    def __init__(self, args: Namespace,  task2classes: Dict, representation_size: int):
        self.args = args
        self.memory = {}
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

    def insert_samples(self, all_samples: List[torch.Tensor], labels: List) -> None:
        for label, class_samples in zip(labels, all_samples):
            self.memory[self.class2task[int(label)]][int(label)] = class_samples
        return None
            

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
    
    def get_n_from_classes(self, n: int, current_task_index: int) -> Tuple:
        tasks = list(range(1, current_task_index))
        labels = []
        samples = []
        time = 0
        ages = []
        for task in reversed(tasks):
            for class_ in self.memory[task]:
                class_samples = self.memory[task][class_].cpu()
                try:
                    samples.append(class_samples[np.random.choice(class_samples.shape[0],n, replace=False)])
                except:
                    # If not enough memory samples, sample with replacement.
                    samples.append(class_samples[np.random.choice(class_samples.shape[0],n, replace=True)])
                labels.append(class_)
                ages.append(time)
            time = time + 1
        return samples, labels, ages


    def get_samples_per_class(self, n: int, current_task_index: int) -> List:
        tasks = list(range(1, current_task_index))
        number_of_classes = self.task2numclasses(tasks)
        a =  int(n / number_of_classes)
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
    

def random_prune(network: nn.Module, pruning_perc: float, skip_first_conv = True) -> nn.Module:
    network = copy.deepcopy(network)
    pruning_perc = pruning_perc / 100.0
    weight_masks = []
    bias_masks = []
    first_conv_flag = skip_first_conv
    for module in network.modules():
        if isinstance(module, nn.Linear):
            weight_masks.append(torch.from_numpy(np.random.choice([0, 1], module.weight.shape,
                                                                  p =  [pruning_perc, 1 - pruning_perc])))
            # We do not prune biases
            bias_masks.append(torch.from_numpy(np.random.choice([0, 1], module.bias.shape, p =  [0, 1])))
        #Channel wise pruning Conv Layer
        elif isinstance(module, nn.Conv2d):
           connectivity_mask = torch.from_numpy(np.random.choice([0, 1],
                                                (module.weight.shape[0],  module.weight.shape[1]),
                                                p =  [0, 1] if first_conv_flag else [pruning_perc, 1 - pruning_perc]))
           first_conv_flag = False
           in_range, out_range = range(module.weight.shape[1]), range(module.weight.shape[0])
           kernel_shape = (module.weight.shape[2], module.weight.shape[3])
           filter_masks = [[np.ones(kernel_shape) if connectivity_mask[out_index, in_index] else np.zeros(kernel_shape)
                            for in_index in in_range]
                            for out_index in out_range]
           weight_masks.append(torch.from_numpy(np.array(filter_masks)).to(torch.float32))
           
           #do not prune biases
           bias_mask = torch.from_numpy(np.random.choice([0, 1], module.bias.shape, p =  [0, 1])).to(torch.float32)  # type: ignore
           bias_masks.append(bias_mask)
    network.set_masks(weight_masks, bias_masks) # type: ignore
    return network


def _compute_weight_sparsity(self):
    parameters = 0
    ones = 0
    for module in self.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            shape = module.weight.data.shape
            parameters += torch.prod(torch.tensor(shape))
            w_mask, _ = copy.deepcopy(module.get_mask()) # type: ignore
            ones += torch.count_nonzero(w_mask)
    return float((parameters - ones) / parameters) * 100


class Grower():

    def __init__(self, backbone: nn.Module, stable_units: List,
                 plastic_units: List, candidate_units: List):
        self.stable_units = stable_units
        self.candidate_units = candidate_units
        self.plastic_units = plastic_units
        try:
            self.conv2lin_mapping_size = backbone.conv2lin_mapping_size
        except:
            self.conv2lin_mapping_size = None


    def grow(self, backbone: nn.Module, connection_quota: List[int]) -> Tuple[nn.Module, List[int]]:
        # No quota
        if sum(connection_quota) == 0:
            return backbone, connection_quota
        return self._random_grow(backbone, connection_quota)
        
    def _random_grow(self, backbone: nn.Module, connection_quota: List[int]) -> Tuple[nn.Module, List[int]]:
        connectivity_masks = [w for w, _ in backbone.get_weight_bias_masks_numpy()] # type: ignore

        def _get_possible_conns(source_units: List, target_units: List, conv2lin_mapping_size):
            possible_connections = []

            for layer_index, (sources, targets) in enumerate(zip(source_units[:-1], target_units[1:])):
                if len(connectivity_masks[layer_index].shape) == 4:
                    conn_shape = (connectivity_masks[layer_index].shape[2], connectivity_masks[layer_index].shape[3])
                else:
                    conn_shape = (1, )
                
                # No target that we can grow
                if len(targets) == 0:
                    pos_conn = np.zeros(connectivity_masks[layer_index].shape)
                    possible_connections.append(pos_conn)
                    continue

                conn_type_1 = np.ones(conn_shape)
                conn_type_0 = np.zeros(conn_shape)
                pos_conn = np.zeros(connectivity_masks[layer_index].shape)

                #Conv2Linear
                if len(connectivity_masks[layer_index].shape) == 2 and len(connectivity_masks[layer_index-1].shape) == 4:
                    for conv_index in sources:
                        start = conv_index*conv2lin_mapping_size
                        end = (conv_index+1)*conv2lin_mapping_size
                        pos_conn[targets, start:end] = 1
                else:
                    pos_conn[np.ix_(targets, sources)] = conn_type_1
                # Remove already existing weights from pos_conn
                if len(connectivity_masks[layer_index].shape) == 4:
                    pos_conn[np.all(connectivity_masks[layer_index][:,:] == conn_type_1, axis = (2, 3))]  = conn_type_0
                else:
                    pos_conn[connectivity_masks[layer_index] != 0] = 0

                possible_connections.append(pos_conn)
            return possible_connections
        
        pos_conn_list = []
        # Candidate Stable to Plastic
        sources, targets = self.candidate_units, self.plastic_units
        pos_conn_list.append(_get_possible_conns(sources, targets, self.conv2lin_mapping_size))
        # Stable to Plastic
        sources, targets = self.stable_units, self.plastic_units
        pos_conn_list.append(_get_possible_conns(sources, targets, self.conv2lin_mapping_size))
        # Plastic to Plastic
        sources, targets = self.plastic_units, self.plastic_units
        pos_conn_list.append(_get_possible_conns(sources, targets, self.conv2lin_mapping_size))
        # Candidate Stable to Candidate Plastic
        sources, targets = self.candidate_units, self.candidate_units
        pos_conn_list.append(_get_possible_conns(sources, targets, self.conv2lin_mapping_size))
        # Stable to Candidate Plastic
        sources, targets = self.stable_units, self.candidate_units
        pos_conn_list.append(_get_possible_conns(sources, targets, self.conv2lin_mapping_size))

        possible_connections = []
        for layer_index in range(len(connectivity_masks)):
            pos_conn = np.zeros(connectivity_masks[layer_index].shape)
            for possible in pos_conn_list:
                if layer_index < len(possible):
                    pos_conn = pos_conn + possible[layer_index]

            possible_connections.append(np.array(pos_conn))

        return self._grow_connections(backbone, possible_connections, connection_quota)

    # ------- Helper methods ------- # 
    def _grow_connections(self, backbone: nn.Module, possible_connections: List, connection_quota: List[int]) -> Tuple[nn.Module, List[int]]:
        layer_index = 0
        weight_init = lambda size: torch.zeros(size).to(get_device())
        remainder_connections = []
        for module in backbone.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                if connection_quota[layer_index] == 0:
                    remainder_connections.append(0)
                    layer_index = layer_index + 1
                    continue
                weight_mask, bias_mask = module.get_mask() # type: ignore
                # Conv layer
                if len(possible_connections[layer_index].shape) == 4:
                    grow_indices = np.nonzero(np.sum(possible_connections[layer_index], axis = (2 , 3)))
                # Linear layer
                else:
                    grow_indices = np.nonzero(possible_connections[layer_index])
                
                conn_shape = (possible_connections[layer_index].shape[2], possible_connections[layer_index].shape[3]) if len(possible_connections[layer_index].shape) == 4 else (1,)
                conn_size = (possible_connections[layer_index].shape[2] * possible_connections[layer_index].shape[3]) if len(possible_connections[layer_index].shape) == 4 else 1

                # There are connections that we can grow
                if len(grow_indices[0]) != 0:
                    # We can partial accommodate grow request (we will have remainder connections)
                    if len(grow_indices[0])*conn_size <= connection_quota[layer_index]:
                        weight_mask[grow_indices] = torch.ones(weight_mask[grow_indices].shape,
                                                               dtype=weight_mask[grow_indices].dtype).to(get_device())
                        module.weight.data[grow_indices] = torch.zeros(module.weight.data[grow_indices].shape,
                                                                       dtype=module.weight.data.dtype).to(get_device())
                        remainder_connections.append(connection_quota[layer_index] - len(grow_indices[0])*conn_size)
                    else:
                        selection = np.random.choice(len(grow_indices[0]),
                                    size = int(connection_quota[layer_index]/ conn_size), replace = False)
                        tgt_selection = torch.tensor(grow_indices[0][selection]).to(get_device())
                        src_selection = torch.tensor(grow_indices[1][selection]).to(get_device())
                        weight_mask[tgt_selection, src_selection] = torch.squeeze(torch.ones((len(tgt_selection), *conn_shape), dtype = weight_mask.dtype)).to(get_device())
                        module.weight.data[tgt_selection, src_selection] = torch.squeeze(weight_init((len(tgt_selection), *conn_shape)))
                        remainder_connections.append(0)
                else:
                    remainder_connections.append(connection_quota[layer_index])
                module.set_mask(weight_mask, bias_mask)  # type: ignore
                layer_index += 1
        return backbone, remainder_connections
        
