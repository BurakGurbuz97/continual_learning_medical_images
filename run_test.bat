@echo off



python main.py --experiment_name "Microscope_CIL_nispa" --method "nispa_replay_plus" --phase_epochs 3 --learning_rate 0.0001 --batch_size 128 --optimizer "Adam"  --batch_size_memory 128 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0
