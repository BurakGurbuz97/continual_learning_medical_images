# Continual learning in Medical imaging




### Instructions to run REMIND

return_idx = True is required to get additional 
Note: This doesnt work with non

 python main.py --method remind --backbone vgg11 --dataset Microscopic  --return_idx True --pretrain_epochs 10 --overfit_batches 100  --epochs 10   --experiment_name REMIND_Mircoscope_CIL --scenario CIL --batch_size 512
 
While experimenting, use --overfit_batches 5. This will limit training/pretraining batches.
--overfit_batches 5 
example

python main.py --method remind --backbone vgg11 --dataset Microscopic  --return_idx True --pretrain_epochs 10 --pretrain_overfit_batches 10  --epochs 3 --overfit_batches 10     --experiment_name REMIND_Mircoscope_CIL --scenario CIL --batch_size 64
