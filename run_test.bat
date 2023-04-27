@echo off



python main.py --experiment_name "Microscope_CIL_nispa" --dataset "Microscopic" --method "nispa_replay_plus" --phase_epochs 5 --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta"  --batch_size_memory 256 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0
python main.py --experiment_name "Radiological_CIL_nispa" --dataset "Radiological" --method "nispa_replay_plus" --phase_epochs 5 --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta"  --batch_size_memory 256 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0