
import os
import torch

import pickle
import imageio
import numpy as np

from PIL import Image
from torchvision import transforms


CROP_LIST = [48 * 4, 48 * 8, 48 * 12, 48 * 16, 48 * 20, 48 * 24]
SAMPLING_LIST = [16 * 16, 12 * 12, 10 * 10, 15, 4, 2]

def create_and_save_crops(img_org, filename, output_folder, crop_size, num_samples):
    min_size = min([img_org.shape[0], img_org.shape[1]])
    if crop_size >= min_size - 1:
        return 0
    for i in range(num_samples):
        center_r = np.random.randint(int(crop_size / 2), img_org.shape[0] - int(crop_size / 2))
        center_c = np.random.randint(int(crop_size / 2), img_org.shape[1] - int(crop_size / 2))
        tl_r = center_r - int(crop_size / 2)
        tl_c = center_c - int(crop_size / 2)
        crop = img_org[tl_r:tl_r + crop_size, tl_c:tl_c + crop_size, :]
        output_filename = os.path.join(output_folder, filename[:-4] + "_crop_" + str(crop_size) + "_" + str(i) + ".png")
        transforms.ToPILImage()(crop.permute(2, 0, 1)).save(output_filename)
    return num_samples

def create_patches(source_folder, filename, output_folder, curr_idx, crop_size, num_samples):
    file_path = os.path.join(source_folder, filename)
    img_org = transforms.ToTensor()(Image.open(file_path).convert('RGB')).permute(1, 2, 0)    
    count = create_and_save_crops(img_org, filename, output_folder, crop_size, num_samples)
    return curr_idx + count

def create_patch_data(source_folder, output_folder):
    file_list = os.listdir(source_folder)
    for filename in file_list:        
        print("processing Image:" + filename)
        curr_idx = 0
        for idx in range(len(CROP_LIST)):
            crop_size = CROP_LIST[idx]
            num_samples = SAMPLING_LIST[idx]
            curr_idx = create_patches(source_folder, filename, output_folder, curr_idx, crop_size, num_samples)

if __name__ == '__main__':
    div2k_train_folder = "../data/DIV2K_train_HR"
    output_folder = "/mnt/SSD/Datasets/DIV2K_train_HR_Patches"
    create_patch_data(div2k_train_folder, output_folder)
