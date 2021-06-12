
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

SCALE_LIST = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]

def get_sr_params(resolution):
    h, w = list(map(int, resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    return h, w, coord, cell

def perform_sr_from_file(model, img_filename, resolution):
    img = transforms.ToTensor()(Image.open(img_filename).convert('RGB'))
    h, w, coord, cell = get_sr_params(resolution)
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    return pred

def load_img(img_filename):
    img = transforms.ToTensor()(Image.open(img_filename).convert('RGB'))
    return img

def perform_sr_from_img(model, img, resolution):
    h, w, coord, cell = get_sr_params(resolution)
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    return pred

def load_model(model_path):
    model = models.make(torch.load(model_path)['model'], load_sd=True).cuda()
    return model

def get_img_dimension(img_filename):
    img = transforms.ToTensor()(Image.open(img_filename).convert('RGB'))
    h = img.shape[1]
    w = img.shape[2]
    return h, w

def get_encoded_feature(model, img):
    with torch.no_grad():
        img_tensor = ((img - 0.5) / 0.5).cuda().unsqueeze(0)
        feat = model.gen_feat(img_tensor)
        return feat

def get_resolution_list(img_filename):
    h, w = get_img_dimension(img_filename)
    res_list = [{'h': int(SCALE_LIST[i] * h), 'w': int(SCALE_LIST[i] * w)} for i in range(len(SCALE_LIST))]
    return res_list

def generate_SR_list(model, img_path, output_folder):
    res_list = get_resolution_list(img_path)
    h, w = get_img_dimension(img_path)
    for i in range(len(res_list)):
        resolution = str(res_list[i]['h']) + "," + str(res_list[i]['w'])
        pred = perform_sr_from_file(model, img_path, resolution)
        ## crop the output
        c_r = int(res_list[i]['h'] / 2)
        c_c = int(res_list[i]['w'] / 2)
        tl_r = c_r - int(h / 2)
        tl_c = c_c - int(w / 2)
        br_r = tl_r + h
        br_c = tl_c + w
        # pred_crop = pred[tl_r:br_r, tl_c:br_c, :]
        pred_crop = pred[:, tl_r:br_r, tl_c:br_c]
        output_filename = str(i) + ".png"
        transforms.ToPILImage()(pred_crop).save(os.path.join(output_folder, output_filename))

def generate_SR_GT_list(model, img_path, img_gt_path, output_folder, start_res_level):
    res_list = get_resolution_list(img_path)
    h, w = get_img_dimension(img_path)
    img_gt = load_img(img_gt_path)
    gt_temp = transforms.ToPILImage()(img_gt.clone()).resize((res_list[start_res_level]['w'], res_list[start_res_level]['h']), Image.ANTIALIAS)
    gt = transforms.ToTensor()(gt_temp)
    c_r_1 = int(res_list[start_res_level]['h'] / 2)
    c_c_1 = int(res_list[start_res_level]['w'] / 2)
    tl_r_1 = c_r_1 - int(h / 2)
    tl_c_1 = c_c_1 - int(w / 2)
    br_r_1 = tl_r_1 + h
    br_c_1 = tl_c_1 + w
    gt_crop = gt[:, tl_r_1:br_r_1, tl_c_1:br_c_1]
    for i in range(len(res_list)):
        if i <= start_res_level:
            # pred = transforms.Resize((res_list[i]['h'], res_list[i]['w']))(img_gt)
            pred_temp = transforms.ToPILImage()(img_gt).resize((res_list[i]['w'], res_list[i]['h']), Image.ANTIALIAS)
            pred = transforms.ToTensor()(pred_temp)
            scale = 1
        else:
            # gt = transforms.Resize((res_list[start_res_level]['h'], res_list[start_res_level]['w']))(img_gt)
            scale = h / res_list[start_res_level]['h']
            resolution = str(int(res_list[i]['h'] * scale)) + "," + str(int(res_list[i]['w'] * scale))
            pred = perform_sr_from_img(model, gt_crop, resolution)
        ## crop the output
        c_r = int(res_list[i]['h'] * scale / 2)
        c_c = int(res_list[i]['w'] * scale / 2)
        tl_r = c_r - int(h / 2)
        tl_c = c_c - int(w / 2)
        br_r = tl_r + h
        br_c = tl_c + w
        pred_crop = pred[:, tl_r:br_r, tl_c:br_c]
        output_filename = str(i) + ".png"
        transforms.ToPILImage()(pred_crop).save(os.path.join(output_folder, output_filename))

def generate_SR_pred_list(model, base_pred_folder, img_path, img_pred_path, output_folder, start_res_level):
    res_list = get_resolution_list(img_path)
    h, w = get_img_dimension(img_path)
    img_pred = load_img(img_pred_path)
    for i in range(len(res_list)):
        if i <= start_res_level:
            pred_crop = load_img(os.path.join(base_pred_folder, str(i) + ".png"))
        else:
            scale = h / res_list[start_res_level]['h']
            # resolution = str(res_list[i]['h']) + "," + str(res_list[i]['w'])
            resolution = str(int(res_list[i]['h'] * scale)) + "," + str(int(res_list[i]['w'] * scale))
            pred = perform_sr_from_img(model, img_pred, resolution)
            ## crop the output
            c_r = int(res_list[i]['h'] * scale / 2)
            c_c = int(res_list[i]['w'] * scale / 2)
            tl_r = c_r - int(h / 2)
            tl_c = c_c - int(w / 2)
            br_r = tl_r + h
            br_c = tl_c + w
            pred_crop = pred[:, tl_r:br_r, tl_c:br_c]
        output_filename = str(i) + ".png"
        transforms.ToPILImage()(pred_crop).save(os.path.join(output_folder, output_filename))


if __name__ == '__main__':
    ## scratch script for testing
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = load_model("./download/rdn-liif.pth")
    img_path = "/mnt/HDD1/home/mtlong/workspace/2021/Aggregative_Learning/Sand_Box/liif/data/samples/4X/0802.png"
    img = load_img(img_path)
    # feat = get_encoded_feature(model, img)
    # print(img.shape)
    # print(feat.shape)
    h = img.shape[1]
    w = img.shape[2]
    c_r = int(h / 2)
    c_c = int(w / 2)
    tl_r = c_r - 24
    tl_c = c_c - 24
    br_r = tl_r + 48
    br_c = tl_c + 48
    img_crop = img[:, tl_r:br_r, tl_c:br_c]
    target_resolution = "192,192"
    pred_crop = perform_sr_from_img(model, img_crop, target_resolution)
    pred = perform_sr_from_img(model, img, str(4 * h) + "," + str(4 * w))
    # transforms.ToPILImage()(pred).save("./data/samples/test_result.png")
    # transforms.ToPILImage()(pred_crop).save("./data/samples/test_result_crop.png")    