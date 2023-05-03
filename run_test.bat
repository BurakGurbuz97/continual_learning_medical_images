@echo off

@REM python main.py --experiment_name "MAS_TEST_L4.0_TIL_smallCNN" --scenario "TIL" --method "memory_aware_synapses" --lambda_val 4

python main.py --experiment_name "MAS_TEST_L1_CIL_smallCNN" --scenario "CIL" --method "memory_aware_synapses" --lambda_val 1

python main.py --experiment_name "MAS_TEST_L1_CIL_smallCNN_radio" --scenario "CIL" --method "memory_aware_synapses" --lambda_val 1 --dataset "Radiological"

python main.py --experiment_name "MAS_TEST_L10_CIL_smallCNN_radio" --scenario "CIL" --method "memory_aware_synapses" --lambda_val 10 --dataset "Radiological"

python main.py --experiment_name "MASR_TEST_L1_CIL_smallCNN_radio" --scenario "CIL" --method "memory_aware_synapses_replay" --lambda_val 1 --dataset "Radiological"

python main.py --experiment_name "MASR_TEST_L0.1_CIL_smallCNN_radio" --scenario "CIL" --method "memory_aware_synapses_replay" --lambda_val 0.1 --dataset "Radiological"

python main.py --experiment_name "MAS_TEST_L10_TIL_smallCNN_radio" --scenario "TIL" --method "memory_aware_synapses" --lambda_val 10 --dataset "Radiological"


@REM python main.py --experiment_name "Microscope_CIL_nispa" --dataset "Microscopic" --method "nispa_replay_plus" --phase_epochs 5 --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta"  --batch_size_memory 256 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0
@REM python main.py --experiment_name "Radiological_CIL_nispa" --dataset "Radiological" --method "nispa_replay_plus" --phase_epochs 5 --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta"  --batch_size_memory 256 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0