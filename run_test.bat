@echo off
@REM for %%s in (0, 1, 2) do (
@REM    python main.py --experiment_name "Microscope_CIL_nispa_seed%%s" --dataset "Microscopic" --method "nispa_replay_plus" --phase_epochs 5 --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta" --seed %%s  --batch_size_memory 256 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0
@REM    python main.py --experiment_name "Radiological_CIL_nispa_seed%%s" --dataset "Radiological" --method "nispa_replay_plus" --phase_epochs 5 --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta" --seed %%s  --batch_size_memory 256 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0
@REM)

python main.py --experiment_name "Microscope_CIL_JOINT_LEARNER" --epochs 3 --number_of_tasks 1 --dataset "Microscopic" --method naive_continual_learner --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta" --seed 0
python main.py --experiment_name "Radio_CIL_JOINT_LEARNER" --epochs 3 --number_of_tasks 1 --dataset "Radiological" --method naive_continual_learner --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta" --seed 0


python main.py --experiment_name "Microscope_CIL_NAIVE_LEARNER" --epochs 3 --number_of_tasks 3 --dataset "Microscopic" --method naive_continual_learner --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta" --seed 0
python main.py --experiment_name "Radio_CIL_NAIVE_LEARNER" --epochs 3 --number_of_tasks 6 --dataset "Radiological" --method naive_continual_learner --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta" --seed 0
