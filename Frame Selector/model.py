# -*- coding: utf-8 -*-
import argparse
from tabnanny import check
import numpy as np
import torch
import pdb
import torch.backends.cudnn as cudnn

from timm.models.layers import drop_path
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


class Best_Frame_Select(nn.Module):
    def __init__(self):
        super(Best_Frame_Select, self).__init__()

        self.conv1 = nn.Conv2d(384, 192, kernel_size=(3,1),stride = (1,1), padding=(1,0))
        self.conv2 = nn.Conv2d(192, 192, kernel_size=(3,4),stride = (1,1), padding=(1,0))
        self.conv3 = nn.Conv1d(192,192,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm1d(384)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(1 * 8 * 384, 28)

    def epoch_end(self, epoch, total_accuracy, result):
        print("Epoch [{}], total_accurate: {:.4f},  val_loss: {:.4f}".format(epoch, total_accuracy, result))

    def forward(self, input):
        input = self.bn(input)
        input = self.conv1(input)
        input = self.conv2(input) #output should be (b, 192, 8, 1)
        input = input.transpose(2, 1)
        input = rearrange(input, 'b t c a-> b (t c)', a=1)
        identity = input
        input = self.conv3(input)
        input = self.drop(input)
        input = self.relu(input)

        input = self.conv3(input)
        input = self.drop(input)
        input += identity
        input = self.relu(input)

        identity = input
        input = self.conv3(input)
        input = self.drop(input)
        input = self.relu(input)

        input = self.conv3(input)
        input = self.drop(input)
        input += identity
        input = self.relu(input)

        identity = input
        input = self.conv3(input)
        input = self.drop(input)
        input = self.relu(input)

        input = self.conv3(input)
        input = self.drop(input)
        input += identity
        input = self.relu(input)

        input = self.flat(input)
        output = self.linear(input)

        return output


def fit(lr, model, input, opt, loss_fc, devices, current_epoch, statistic_dict, save_path, log_writer):
    optimizer = opt(model.parameters(), lr)  # momentum=0
    model.train()
    accuracy = 0
    for step, batch in enumerate(input):
        image, label = batch
        image = image.to(devices)
        image = image.reshape(16, 384, 8)
        label = label.to(devices)  # label是按重要性排序后排出来的顺序

        out = model(image)  # out是我的linear层的输出，28个数对应着每一个组合的重要性

        for i in range(len(label)):
            accuracy += int((out[i].argmax().item() == label[i]).item())

        loss = loss_fc(out, label)  # Calculate loss
        loss_value = loss.item()

        # print(model.state_dict())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if current_epoch % 4 == 0 and current_epoch != 0:

        torch.save(model.state_dict(), save_path + "mymodel_success_really_10000_" + str(current_epoch) + ".pth")
        model.epoch_end(current_epoch, accuracy / (len(label) * (step + 1)), loss.item())

    file = open(file=statistic_dict, mode="a")
    file.write(
        "epoch = {}    total_acc={:.4f}   loss={:.4f}\n".format(current_epoch, accuracy / (len(label) * (step + 1)),
                                                                loss.item()))
    file.close()
    print("now epoch={},now accurate={:.4f},now loss={:.4f}".format(current_epoch, accuracy / (len(label) * (step + 1)),
                                                                    loss.item()))
    log_writer.update(loss=loss_value, head="loss")
    log_writer.update(accurate=accuracy / (len(label) * (step + 1)), head="accurate")
    log_writer.set_step()
    return accuracy




