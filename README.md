# AortSeg2024 Solution

## Training
Please inside the training function and follow the instruction to run the training code
### Installation
Requirements: Ubuntu 20.04, CUDA 11.8

Create a virtual environment: conda create -n uxlstm python=3.10 -y and conda activate uxlstm 
Install Pytorch 2.0.1: pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
Download code: git clone https://github.com/tianrun-chen/xLSTM-UNet-PyTorch.git
cd xLSTM-UNet-PyTorch/UxLSTM and run pip install -e .

## Dataset conversion

> python3 conversion_nnunetv2_aorta_update

## Dataset preprocessing

> nnUNetv2_plan_and_preprocess -d DataID --verify_dataset_integrity

## SSL student teacher
Go inside Training/SSL_student_teacher
Put dataset inside the main_data folder,
Add plans file inside dataset and set the path on your system and run

> python xlstm_bottom_ssl_updatedAorta.py


## Training 

Go inside Training/Downstream_seg
Place the following file run_finetuning_xLSTM_bottom_model.py inside run folder /home/aqayyum/xLSTM-UNet-PyTorch/UxLSTM/nnunetv2/run/
download SSL trained weights and put inside data folder and then run
https://mega.nz/file/AGVSHToZ#nEeMJ1zU8qY_JuhTdKwXSDyWjM-BFC-w0NewHQjz_4M

CUDA_VISIBLE_DEVICES=0 python3 /home/aqayyum/xLSTM-UNet-PyTorch/UxLSTM/nnunetv2/run/run_finetuning_xLSTM_bottom_model.py 177 3d_fullres all -pretrained_weights /home/aqayyum/xLSTM-UNet-PyTorch/data/best_student_model_enco_bottom_aorta1.pth -tr nnUNetTrainerUxLSTMBot -lr 0.001 -bs 2

##  Prediction

> pip install -r requirements.txt

### donwload the trained weight and put in fold_all folder
https://mega.nz/file/0Kt1mKpD#CIVsKsiC-CwxA_1Etz7MRAy_evMae0DVVUwUCz7pSdI

####### please change all dataset, input and output folder##############

> python3 inference.py


### Acknowledgement
https://github.com/tianrun-chen/xLSTM-UNet-PyTorch

https://github.com/MIC-DKFZ/nnUNet
