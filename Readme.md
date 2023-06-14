# Continual learning in Medical imaging : CS7643 DL Spring 2023 


## Instructions to run REMIND
```console
python main.py --method remind --dataset Microscopic  --experiment_name REMIND_Mircoscope_CIL --scenario CIL --batch_size 256
```
```console
python main.py --method remind --dataset Radiological --experiment_name REMIND_Radiological_CIL --scenario CIL --batch_size 256
```


## Instructions to run NISPA
```console
python main.py --experiment_name "Microscope_CIL_nispa" --dataset "Microscopic" --method "nispa_replay_plus" --phase_epochs 5 --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta"  --batch_size_memory 256 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0
```
```console
python main.py --experiment_name "Radiological_CIL_nispa" --dataset "Radiological" --method "nispa_replay_plus" --phase_epochs 5 --learning_rate 1.0 --batch_size 256 --optimizer "Adadelta"  --batch_size_memory 256 --memory_per_class 100 --min_activation_perc 70.0 --num_phases 20 --prune_perc 70.0
```

## Instructions to run MAS
To run MAS:

```console
python main.py --experiment_name "MAS_TEST_L1_CIL_smallCNN" --method "memory_aware_synapses" --lambda_val 1
```

To run MAS with Replay

```console
python main.py --experiment_name "MASR_TEST_L1_CIL_smallCNN" --method "memory_aware_synapses_replay" --lambda_val 1
```

To run MAS (TIL setting)

```console
python main.py --experiment_name "MAS_TEST_L10_TIL_smallCNN" --scenario "TIL" --method "memory_aware_synapses" --lambda_val 10
```
## Instructions to run DER

To run vanilla DER:

```console
python main.py --method "dark_experience_replay" --optimizer "Adam" --experiment_name "Microscope_CIL_DER" --epochs 10 --alpha 0.75
```

To run DER++

```console
python main.py --method "dark_experience_replay" --optimizer "Adam" --experiment_name "Microscope_CIL_DERPP" --pp --epochs 10 --alpha 0.75 --beta 0.5
```

## Team Members
Amritpal Singh, Mustafa Burak Gurbuz, Shiva Souhith Gantha , Prahlad Jasti 


# Batch run

1. In the file job.sh, put this
   ''' #!/bin/sh
    python python_script.py
    '''
2. Execute this command to make the script runnable for you : chmod u+x run_test.sh
3. Run it : ./run_test.sh
