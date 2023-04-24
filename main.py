import os
from utils import get_argument_parser, set_seeds, log   
from datasets import get_experience_streams
import torch

from Source.naive_continual_learner import NaiveContinualLearner
from Source.nispa_replay_plus import NispaReplayPlus
from Source.memory_aware_synapses import MemoryAwareSynapses
from Source.dark_experience_replay import DarkExperienceReplay
from Source.Backbones.vanilla_mlp import VanillaMLP
from Source.Backbones.vanilla_cnn import VanillaCNN 
from Source.Backbones.vgg11_base import vgg11_wrapper, VGG11
from Source.Backbones.cnn_small import CNN_Small
from Source.remind import Remind
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = get_argument_parser()
    print(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Needed for reproducibility on Windows and some GPUs
    if args.deterministic: 
        set_seeds(args.seed)
    scenario, input_size, output_size, task2classes = get_experience_streams(args)

    args.num_classes = output_size

    print('scenario', scenario)
    print('input_size', input_size)
    print('output_size', output_size)
    print('task2classes', task2classes)
    
    # Pick Backbone
    if args.backbone == "vanilla_mlp":
        backbone = VanillaMLP(input_size, output_size, args).to(get_device())
    elif args.backbone == "vanilla_cnn":
        backbone = VanillaCNN(input_size, output_size, args).to(get_device())
    # elif args.backbone == "resnet18":
    #     backbone = timm.create_model(model_name="resnet18", pretrained=False, num_classes=output_size, in_chans=input_size[0]).to(get_device()) #models.resnet18(pretrained=False) #
    elif args.backbone == "vgg11":
        if args.method == "nispa_replay_plus":
            backbone = VGG11(input_size, output_size, args).to(get_device())
        else:
            backbone = vgg11_wrapper(input_size, output_size, args).to(get_device())
    elif args.backbone == "cnn_small":
        backbone = CNN_Small(input_size, output_size, args).to(get_device())
    else:
        raise Exception("Unknown args.backbone={}".format(args.backbone))

    # Pick Learner
    if args.method == "naive_continual_learner":
        learner = NaiveContinualLearner(args, backbone, scenario, task2classes)
    elif args.method == "nispa_replay_plus":
        learner = NispaReplayPlus(args, backbone, scenario, task2classes)
    elif args.method == "dark_experience_replay":
        learner = DarkExperienceReplay(args, backbone, scenario, task2classes)
    elif args.method == "memory_aware_synapses":
        learner = MemoryAwareSynapses(args, backbone, scenario, task2classes)
    elif args.method == "remind":
        learner = Remind(args, backbone, scenario, task2classes)
    else:
        raise Exception("Unknown args.method={}".format(args.method))


    all_accuracies = []
    for task_index, (train_task, val_task, test_task) in enumerate(zip(scenario.train_stream,
                                                                       scenario.val_stream,  # type: ignore
                                                                       scenario.test_stream), 1):
        print("Starting Task: {}".format(task_index))
        learner.begin_task(train_task, val_task, task_index)
        learner.learn_task(train_task, val_task, task_index)
        learner.end_task(train_task, val_task, task_index)
        all_accuracies.append(learner.accuracies_on_previous_task(task_index, args.scenario == "TIL"))
        print("Ending Task: {}".format(task_index))
        print()

    log(args, all_accuracies)
        

