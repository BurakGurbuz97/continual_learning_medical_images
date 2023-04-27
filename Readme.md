# Continual learning in Medical imaging : CS7643 DL Spring 2023 


## Instructions to run REMIND

Note: This doesnt work with Mircoscopic dataset only.
return_idx = True is required to get additional 

Following is the training code
```console
python main.py --method remind --backbone vgg11 --dataset Microscopic  --return_idx True --pretrain_epochs 10 --overfit_batches 100  --epochs 10   --experiment_name REMIND_Mircoscope_CIL --scenario CIL --batch_size 512
```

While experimenting, use --overfit_batches 5 --pretrain_overfit_batches 5  This will limit training/pretraining batches.
```console
python main.py --method remind --backbone vgg11 --dataset Microscopic  --return_idx True --pretrain_epochs 10 --pretrain_overfit_batches 10  --epochs 3 --overfit_batches 10     --experiment_name REMIND_Mircoscope_CIL --scenario CIL --batch_size 64
```

## Instructions to run NISPA

## Instructions to run MAS

## Instructions to run DER

To run vanilla DER:

```console
python main.py --method "dark_experience_replay" --experiment_name "Microscope_CIL_DER" --epochs 10
```

To run DER++

```console
python main.py --method "dark_experience_replay" --experiment_name "Microscope_CIL_DERPP" --pp --epochs 10
```

## Team Members
Amritpal Singh, Mustafa Burak Gurbuz, Shiva Souhith Gantha , Prahlad Jasti 
