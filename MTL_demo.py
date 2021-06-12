import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


def get_sr_params(resolution):
    h, w = list(map(int, resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    return h, w, coord, cell

def perform_sr(model, img_filename, resolution):
    img = transforms.ToTensor()(Image.open(img_filename).convert('RGB'))
    h, w, coord, cell = get_sr_params(resolution)
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    pred = perform_sr(model, args.input, args.resolution)
    transforms.ToPILImage()(pred).save(args.output)
