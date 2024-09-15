


# Training
#### Place the following file run_finetuning_xLSTM_bottom_model.py inside run folder /home/aqayyum/xLSTM-UNet-PyTorch/UxLSTM/nnunetv2/run/

## download SSL trained weights and put inside data folder and then run
https://mega.nz/file/AGVSHToZ#nEeMJ1zU8qY_JuhTdKwXSDyWjM-BFC-w0NewHQjz_4M

> CUDA_VISIBLE_DEVICES=0 python3 /home/aqayyum/xLSTM-UNet-PyTorch/UxLSTM/nnunetv2/run/run_finetuning_xLSTM_bottom_model.py 177 3d_fullres all -pretrained_weights /home/aqayyum/xLSTM-UNet-PyTorch/data/best_student_model_enco_bottom_aorta1.pth -tr nnUNetTrainerUxLSTMBot -lr 0.001 -bs 2
