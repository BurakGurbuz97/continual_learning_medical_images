@echo off



python main.py --experiment_name "MNIST_TIL_NAIVE_SGD"  --number_of_tasks 5 --scenario "TIL"
python main.py --experiment_name "MNIST_CIL_NAIVE_SGD"  --number_of_tasks 5 --scenario "CIL"
python main.py --experiment_name "MNIST_CIL_JOINT_SGD"  --number_of_tasks 1 --scenario "CIL"
