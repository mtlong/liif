import os
import torch

import pickle
import imageio
import numpy as np

from PIL import Image
from torchvision import transforms
import lpips
import argparse

CROP_LIST = [48, 48 * 2, 48 * 3, 48 * 4, 48 * 6, 48 * 7]
SAMPLING_LIST = [50, 30, 16, 9, 4, 2]
FILE_LIST = ["0802.png", "0803.png", 
             "0804.png", "0809.png", 
             "0810.png", "0818.png", 
             "0820.png", "0822.png",
             "0823.png", "0825.png", 
             "0826.png", "0834.png", 
             "0835.png", "0836.png", 
             "0837.png", "0839.png", 
             "0846.png", "0849.png", 
             "0855.png", "0879.png", 
             "0883.png", "0898.png"]

def get_crop_size(img_filename):
    l = img_filename.split("_")
    return int(l[2])

def collect_NN_from_crop(loss_fn_vgg, img, crop_size, num_samples, database_folder, output_folder, dist_list_filename, DEVICE):
    copied_file_list = []
    dist_dict = {}
    ## Crop the image
    min_size = min([img.shape[1], img.shape[2]])

    if crop_size >= min_size - 1:
        return
    for i in range(num_samples):
        print("--- Sample #" + str(i))
        center_r = np.random.randint(int(crop_size / 2), img.shape[1] - int(crop_size / 2))
        center_c = np.random.randint(int(crop_size / 2), img.shape[2] - int(crop_size / 2))
        tl_r = center_r - int(crop_size / 2)
        tl_c = center_c - int(crop_size / 2)
        crop = img[:, tl_r:tl_r + crop_size, tl_c:tl_c + crop_size].clone()
        database_filelist = os.listdir(database_folder)
        min_dist = 10
        curr_NN_filepath = ""
        for database_filename in database_filelist:
            if database_filename in copied_file_list:
                continue
            file_path = os.path.join(database_folder, database_filename)
            if file_path in copied_file_list:
                continue
            database_patch_size = get_crop_size(database_filename)
            if database_patch_size <= crop_size * 4:
                # print("Database patch size" + str(database_patch_size) + " is smaller than query size " + str(crop_size) + " -- skip")
                continue
            database_img = transforms.ToTensor()(transforms.Resize((crop_size, crop_size))(Image.open(file_path).convert('RGB'))).to(DEVICE)
            img_processed = (database_img - 0.5) / 0.5
            dist = loss_fn_vgg(crop.unsqueeze(0), img_processed.unsqueeze(0)).detach().cpu().numpy()[0][0][0][0]
            if dist < min_dist:
                min_dist = dist
                curr_NN_filepath = file_path
                curr_filename = database_filename
        os.system("cp " + curr_NN_filepath + " " + output_folder)
        copied_file_list.append(curr_NN_filepath)
        dist_dict[curr_filename] = min_dist
    pickle.dump(dist_dict, open(dist_list_filename, "wb"))

def collect_LPIPS_NN(loss_fn_vgg, file_path, database_folder, output_folder, dist_list_filename, DEVICE):
    img = transforms.ToTensor()(Image.open(file_path).convert('RGB')).to(DEVICE)
    img = (img - 0.5) / 0.5
    for idx in range(len(CROP_LIST)):
        crop_size = CROP_LIST[idx]
        num_samples = SAMPLING_LIST[idx]
        collect_NN_from_crop(loss_fn_vgg, img, crop_size, num_samples, database_folder, output_folder, dist_list_filename, DEVICE) 

def generate_LPIPS_NN(img_folder, database_folder, output_root, start_idx, end_idx, DEVICE):
    loss_fn_vgg = lpips.LPIPS(net = 'vgg').to(DEVICE)
    file_list = FILE_LIST
    if end_idx > len(file_list):
        end_idx = len(file_list)
    for idx in range(start_idx, end_idx):
        filename = file_list[idx]
        print("processing Image:" + filename)
        file_path = os.path.join(img_folder, filename)
        output_folder = os.path.join(output_root, filename)
        dist_list_filename = os.path.join(output_root, filename + "_distlist.pickle")
        os.system("mkdir -p " + output_folder)
        collect_LPIPS_NN(loss_fn_vgg, file_path, database_folder, output_folder, dist_list_filename, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--startIdx')    
    parser.add_argument('--endIdx')    
    parser.add_argument('--gpu', default='0')    
    args = parser.parse_args()
    DEVICE="cuda:" + args.gpu
    start_idx = int(args.startIdx)
    end_idx = int(args.endIdx)

    database_folder = "/media/malong/PNYSSD/Datasets/DIV2K_train_HR_Patches"
    output_root = "../data/samples/Nearest_Neighbors/LPIPS"
    img_folder = "../data/samples/4X"
    
    generate_LPIPS_NN(img_folder, database_folder, output_root, start_idx, end_idx, DEVICE)

    # python create_sample_LPIPSNN_Leopard.py --startIdx 1 --endIdx 3 --gpu 0
    # python create_sample_LPIPSNN_Leopard.py --startIdx 3 --endIdx 5 --gpu 0
    # python create_sample_LPIPSNN_Leopard.py --startIdx 6 --endIdx 9 --gpu 1
    # python create_sample_LPIPSNN_Leopard.py --startIdx 9 --endIdx 11 --gpu 1
    # python create_sample_LPIPSNN_Leopard.py --startIdx 14 --endIdx 18 --gpu 2
    # python create_sample_LPIPSNN_Leopard.py --startIdx 18 --endIdx 22 --gpu 2
