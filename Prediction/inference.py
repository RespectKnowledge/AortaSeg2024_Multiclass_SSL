# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:52:35 2024

@author: aq22
"""

import torch
import os
import numpy as np
import SimpleITK
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import time
def run():
    _show_torch_cuda_info()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    start_time = time.time()
    nnUNet_results = "/opt/app/resources/nnUNet_results"
    predictor = nnUNetPredictor(
        tile_step_size=0.65,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset229_Aorta/nnUNetTrainerUxLSTMBot__nnUNetPlans__3d_fullres'),
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )
    
    input_folder = "/input/images/ct-angiography"
    output_folder = "/output/images/aortic-branches"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_files = glob(str(input_folder + "/*.tiff")) + glob(str(input_folder + "/*.mha"))
    input_file_name = input_files[0]
    output_file_name = output_folder + "/output.mha"

    # predict a single numpy array
    img, props = SimpleITKIO().read_images([input_file_name])
    print("img.shape: ", img.shape)
    print("props: ", props)
    pred_array = predictor.predict_single_npy_array(img, props, None, None, False)
    pred_array = pred_array.astype(np.uint8)
    print("pred_array.shape: ", pred_array.shape)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for prediction: {elapsed_time:.4f} seconds")
    torch.cuda.empty_cache()

    image = SimpleITK.GetImageFromArray(pred_array)
    image.SetDirection(props['sitk_stuff']['direction'])
    image.SetOrigin(props['sitk_stuff']['origin'])
    image.SetSpacing(props['sitk_stuff']['spacing'])
    SimpleITK.WriteImage(
        image,
        output_file_name,
        useCompression=True,
    )
                                 
    print('Saved!!!')
    return 0

def _show_torch_cuda_info():

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())