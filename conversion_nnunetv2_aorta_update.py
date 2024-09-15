# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 09:31:26 2024

@author: aq22
"""

import os
from glob import glob
import nibabel as nib
import SimpleITK as sitk
import nibabel as nib
path='/home/aqayyum/xLSTM-UNet-PyTorch/data/training/'
imges_path=glob(os.path.join(path,'images','*.mha'))
pathmask='C:/Users/aq22/Desktop/kcl2022/MICCAI2024_challeneges/AORTA_seg/training/training/masks'
#labels_path=glob(os.path.join(path,'masks','*.mha'))
pathsaveim='/home/aqayyum/xLSTM-UNet-PyTorch/data/aorta_preprocessed/imagesTr'
pathsavemask='/home/aqayyum/xLSTM-UNet-PyTorch/data/aorta_preprocessed/labelsTr'

for i in range(0,len(imges_path)):
    img=imges_path[i]
    print(img)
    pat_name1=img.split('/')[-1]
    pat_name=img.split('/')[-1].split('.')[0]
    #break
    #pat_name=img.split('\\')[-1][0:8]
    pathmask=os.path.join(pathmask,pat_name1.replace('CTA','label'))
    img_obj=sitk.ReadImage(img)
    img_mask=sitk.ReadImage(pathmask)
    
    #print(img_obj.size())
    #nib.save(img_obj,os.path.join(pathsaveim,pat_name+'.nii.gz'))
    sitk.WriteImage(img_obj,os.path.join(pathsaveim,pat_name+'_0000.nii.gz'))
    sitk.WriteImage(img_mask,os.path.join(pathsavemask,pat_name+'.nii.gz'))
    #break

#%%
import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from glob import glob
data_dir = '/home/aqayyum/xLSTM-UNet-PyTorch/data/aorta_preprocessed/'
case_ids = glob(os.path.join(data_dir,'imagesTr','*.nii.gz'))
case_idslabels = glob(os.path.join(data_dir,'labelsTr','*.nii.gz'))
task_id = 177
task_name = "Aorta_2"

foldername = "Dataset%03.0d_%s" % (task_id, task_name)
nnUNet_raw='/home/aqayyum/xLSTM-UNet-PyTorch/data/nnUNet_raw/'
# setting up nnU-Net folders
out_base = join(nnUNet_raw, foldername)
imagestr = join(out_base, "imagesTr")
labelstr = join(out_base, "labelsTr")
maybe_mkdir_p(imagestr)
maybe_mkdir_p(labelstr)
################### iterate images and shitl in imagestr folder####################
for i in range(0,len(case_ids)):
    pathimgs=case_ids[i]
    pathmasks=case_idslabels[i]
    shutil.copy(pathimgs,imagestr)
    shutil.copy(pathmasks,labelstr)
    #break
    
	
  
generate_dataset_json(
    str(out_base),
    channel_names={
        0: "CT",
    },
    labels={'Background': 0, 
               'Zone 0': 1, 
               'Innominate': 2, 
               'Zone 1': 3, 
               'Left Common Carotid': 4, 
               'Zone 2': 5, 
               'Left Subclavian Artery': 6, 
               'Zone 3': 7, 
               'Zone 4': 8, 
               'Zone 5': 9, 
               'Zone 6': 10, 
               'Celiac Artery': 11, 
               'Zone 7': 12, 
               'SMA': 13, 
               'Zone 8': 14, 
               'Right Renal Artery': 15, 
               'Left Renal Artery': 16, 
               'Zone 9': 17, 
               'Zone 10 R (Right Common Iliac Artery)': 18, 
               'Zone 10 L (Left Common Iliac Artery)': 19, 
               'Right Internal Iliac Artery': 20, 
               'Left Internal Iliac Artery': 21, 
               'Zone 11 R (Right External Iliac Artery)': 22, 
               'Zone 11 L (Left External Iliac Artery)': 23, 
               },
    file_ending=".nii.gz",
    num_training_cases=len(case_ids),
)