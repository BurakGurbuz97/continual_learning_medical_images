import os
from utils import get_argument_parser, set_seeds, log   
from datasets import get_experience_streams
import torch

from Source.naive_continual_learner import NaiveContinualLearner
from Source.nispa_replay_plus import NispaReplayPlus
from Source.dark_experience_replay import DarkExperienceReplay
from Source.Backbones.vanilla_mlp import VanillaMLP
from Source.Backbones.vanilla_cnn import VanillaCNN

def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    args = get_argument_parser()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Needed for reproducibility on Windows and some GPUs
    if args.deterministic: 
        set_seeds(args.seed)
    scenario, input_size, output_size, task2classes = get_experience_streams(args)

    # Pick Backbone
    if args.backbone == "vanilla_mlp":
        backbone = VanillaMLP(input_size, output_size, args).to(get_device())
    elif args.backbone == "vanilla_cnn":
        backbone = VanillaCNN(input_size, output_size, args).to(get_device())
    else:
        raise Exception("Unknown args.backbone={}".format(args.backbone))
    
    print(backbone)


    # Pick Learner
    if args.method == "naive_continual_learner":
        learner = NaiveContinualLearner(args, backbone, scenario, task2classes)
    elif args.method == "nispa_replay_plus":
        learner = NispaReplayPlus(args, backbone, scenario, task2classes)
    elif args.method == "dark_experience_replay":
        learner = DarkExperienceReplay(args, backbone, scenario, task2classes)
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
        print("Ending Task: {}".format(task_index))
        all_accuracies.append(learner.accuracies_on_previous_task(task_index, args.scenario == "TIL"))
    log(args, all_accuracies)
        

