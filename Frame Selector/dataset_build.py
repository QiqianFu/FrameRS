# -*- coding: utf-8 -*-
import argparse
from tabnanny import check
import numpy as np
import torch
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
from utils import evaluate_fun


class FrameSelect(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 args,
                 device,
                 patch_size,
                 model,
                 frame_dict,
                 ):
        super(FrameSelect, self).__init__()
        self.root = root
        self.args = args
        self.device = device
        self.patch_size = patch_size
        self.model = model
        self.frame_dict = frame_dict

    def __getitem__(self, index):

        directory = self.root + str(int(index)) + ".mp4"

        with open(directory, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        duration = len(vr)
        new_length = 1
        new_step = 1
        skip_length = new_length * new_step
        # frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]

        tmp = np.arange(0, 32, 2)

        frame_id_list = tmp.tolist()
        try:
            video_data = vr.get_batch(frame_id_list).asnumpy()
        except:
            try:
                tmp = np.arange(0, 16, 1)
                frame_id_list = tmp.tolist()

                video_data = vr.get_batch(frame_id_list).asnumpy()

            except:

                print("now is %i", index)
                linshi_root = '/home/srtp_ghw/fqq/hmdb51/video_'
                directory = linshi_root + str(int((index / 2) % 10000)) + ".avi"

            with open(directory, 'rb') as f:
                vr = VideoReader(f, ctx=cpu(0))
            duration = len(vr)
            new_length = 1
            new_step = 1
            skip_length = new_length * new_step
            # frame_id_list = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61]

            tmp = np.arange(0, 16, 1)
            frame_id_list = tmp.tolist()

            video_data = vr.get_batch(frame_id_list).asnumpy()

        list_1 = []
        list_1, img, bool_masked_pos = evaluate_fun(video_data=video_data, args=self.args, frame_id_list=frame_id_list,
                                                    device=self.device, patch_size=self.patch_size, model=self.model,
                                                    list_1=list_1)
        shape = bool_masked_pos.shape
        no_mask = np.zeros(shape=shape)
        no_mask = torch.from_numpy(no_mask)
        no_mask = no_mask.unsqueeze(0)
        no_mask = no_mask.to(self.device, non_blocking=True).flatten(1).to(torch.bool)
        middle_layer = self.model(img, no_mask, want_middle=True)  # 应该是[1,368*8,14,14] 这里是肯定不对的
        middle_layer = rearrange(middle_layer, 'b c (t p0 p1) -> b (c t) (p0 p1)', p0=14, p1=14)
        # middle_layer = middle_layer#.reshape(1,384*8,14,14)
        # middle_layer = middle_layer.reshape(1,384,8*14*14)
        # avg_method = nn.AvgPool2d(2,stride=2)  #avg默认前两个维度是batch和channel，14是square matrix的宽度
        max_method = nn.MaxPool1d(kernel_size=49,stride=49)
        middle_layer = max_method(middle_layer).reshape(384, 4,
                                                        8)  # .reshape(middle_layer.shape[0],384,8,-1) #1,384*8,14*14
        # print(middle_layer.shape) #这里是1，1568，768
        # middle_layer=middle_layer.transpose(1,0)
        list_1.sort(key=lambda x: x[1], reverse=False)
        dict = {}
        for i in range(len(list_1)):
            dict[i] = list_1[i][0]
        label = self.frame_dict[dict[0]]
        return (middle_layer, label)

    def __len__(self):
        return int(43825)  # 先跑前50000吧


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label_list):
        super(MyDataset, self).__init__()
        self.data = data
        self.label_list = label_list


    def __getitem__(self, index):

        img = self.data[index]

        label = int(self.label_list[index])


        return img, label

    def __len__(self):
        return int(50000)

