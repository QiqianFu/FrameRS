# -*- coding: utf-8 -*-
import argparse
from tabnanny import check
import numpy as np
import torch
import os
import pdb
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import DataAugmentationForVideoMAE
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
import torch.nn as nn
from masking_generator import TubeMaskingGenerator

from model import Best_Frame_Select, fit
from dataset_build import FrameSelect


def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('--img_path', default='/home/srtp_ghw/fqq/data2/', type=str, help='input video path')
    parser.add_argument('--model_path', default='/home/srtp_ghw/fqq/MyMAE8/output_dir/checkpoint-1600.pth', type=str,
                        help='checkpoint path of model')
    parser.add_argument('--save_path', default="/home/srtp_ghw/fqq/sth_for_selector.npy", type=str,
                        help='where to save npy file and the name')
    parser.add_argument('--fine_tune', default=False, type=bool,
                        help='If True, then finetune. If False, then key frame select')
    parser.add_argument('--fine_tune_groundtruth', default="txt", type=bool,
                        help='the txt file of label')
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=3002)
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )

    return model


def dataset_build(args, model, patch_size, frame_dict):
    device = torch.device(args.device)
    a = FrameSelect(root=args.img_path,
                    args=args,
                    device=device,
                    patch_size=patch_size,
                    model=model,
                    frame_dict=frame_dict,
                    )
    return a


def main(args):
    frame_dict = {}
    j = 0
    for i in range(8):
        for k in range(i + 1, 8):
            frame_dict[(i, k)] = j
            j += 1

    device = torch.device(args.device)

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if args.fine_tune:
        f = open(args.fine_tune_groundtruth)
        lista = f.readlines()
        a = dataset_build(args, model, patch_size, frame_dict, lista)
        data_loader_train = torch.utils.data.DataLoader(a)
        list = []
        for i in data_loader_train:
            img, label = i
            # file = open(file="dataset_sth3.txt", mode="a")
            # file.write(str(label.item()) + "\n")
            # file.close()
            list.append(img.cpu().clone().detach().numpy())
        s_numpy = [x for x in list]  # 步骤1
        np.save(args.save_path, s_numpy)



    else:

        a = dataset_build(args, model, patch_size,frame_dict)
        data_loader_train = torch.utils.data.DataLoader(a)
        list = []
        for i in data_loader_train:
            img, label = i
            file = open(file="dataset_sth.txt", mode="a")
            file.write(str(label.item()) + "\n")
            file.close()

            list.append(img.cpu().clone().detach().numpy())

        s_numpy = [x for x in list]  # 步骤1
        np.save(args.save_path, s_numpy)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
