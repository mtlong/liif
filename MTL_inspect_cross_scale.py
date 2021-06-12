import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

import MTL_utils as helper


def generate_SR_lowest_res(model, input_path, result_path):
    img_file_list = os.listdir(input_path)
    for filename in img_file_list:
        img_path = os.path.join(input_path, filename)
        output_folder = os.path.join(result_path, "lowest_res_SR_" + filename)
        os.system("mkdir -p " + output_folder)
        helper.generate_SR_list(model, img_path, output_folder)

def generate_SR_gt_res(model, input_path, gt_path, result_path):
    img_file_list = os.listdir(input_path)
    for filename in img_file_list:
        img_path = os.path.join(input_path, filename)
        output_folder_root = os.path.join(result_path, "gt_res_SR_" + filename)
        img_gt_path = os.path.join(gt_path, filename)
        for start_res_level in range(4):
            output_folder = os.path.join(output_folder_root, str(start_res_level))
            os.system("mkdir -p " + output_folder)
            helper.generate_SR_GT_list(model, img_path, img_gt_path, output_folder, start_res_level)

def generate_SR_pred_res(model, input_path, result_path):
    img_file_list = os.listdir(input_path)
    for filename in img_file_list:
        img_path = os.path.join(input_path, filename)
        output_folder_root = os.path.join(result_path, "pred_res_SR_" + filename)
        base_pred_folder = os.path.join(result_path, "lowest_res_SR_" + filename)
        for start_res_level in range(4):
            img_pred_path = os.path.join(result_path, "lowest_res_SR_" + filename, str(start_res_level) + ".png")
            output_folder = os.path.join(output_folder_root, str(start_res_level))
            os.system("mkdir -p " + output_folder)
            helper.generate_SR_pred_list(model, base_pred_folder, img_path, img_pred_path, output_folder, start_res_level)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model')
    parser.add_argument('--input_path')
    parser.add_argument('--gt_path')
    parser.add_argument('--result_path')
    parser.add_argument('--mode') ## One of these: 'SR_lowest_res', 'SR_GT_res', 'SR_pred_res'
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = helper.load_model(args.model)

    if args.mode == "SR_lowest_res":
        ## Generate SR results from the lowest resolution
        generate_SR_lowest_res(model, args.input_path, args.result_path)
    elif args.mode == "SR_GT_res":
        ## Generate SR results from ground-truth resolution
        generate_SR_gt_res(model, args.input_path, args.gt_path, args.result_path)
    elif args.mode == "SR_pred_res":
        ## Generate SR results from predicted results
        generate_SR_pred_res(model, args.input_path, args.result_path)
    else:
        print("Error -- Unrecognized mode")

    