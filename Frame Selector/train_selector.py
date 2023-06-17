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
from dataset_build import FrameSelect
from model import Best_Frame_Select, fit
from dataset_build import MyDataset


def get_args():
    parser = argparse.ArgumentParser('VideoMAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input video path')
    parser.add_argument('save_path', type=str, help='save video path')
    parser.add_argument('model_path', default='/home/srtp_ghw/fqq/MyMAE8/output_dir/checkpoint-1600.pth', type=str,
                        help='checkpoint path of model')
    parser.add_argument('--statistic_path', default='/home/srtp_ghw/fqq_srtp/statistic.txt', type=str,
                        help='checkpoint path of model')
    parser.add_argument('--log_dir', default='/home/srtp_ghw/fqq/log_dir/',
                        help='path where to tensorboard log')
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--device', default='cuda:2',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=1201)
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
    device = torch.device("cuda:3")
    a = FrameSelect(root=args.img_path,
                    args=args,
                    device=device,
                    patch_size=patch_size,
                    model=model,
                    frame_dict=frame_dict
                    )
    return a


def main(args):
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.

    frame_dict = {}
    j = 0
    for i in range(8):
        for k in range(i + 1, 8):
            frame_dict[(i, k)] = j
            j += 1
    print(args)
    device = torch.device(args.device)
    device1 = torch.device("cuda:3")
    cudnn.benchmark = True
    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device1)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()


    fuk = np.load("/home/srtp_ghw/fqq/sth_for_selector.npy", allow_pickle=True)
    shet = [torch.tensor(x) for x in fuk]

    da = open("dataset_sth.txt", "r")
    it = da.readlines()

    dataset_train = MyDataset(shet,it)

    Model = Best_Frame_Select()
    Model = Model.to(device)
    loss_fc = nn.CrossEntropyLoss()  # Calculate loss
    loss_fc = loss_fc.to(device)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, drop_last=True)
    save_path = args.save_path

    log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    num_training_steps_per_epoch = len(dataset_train) // 8

    for epoch in range(args.epochs):
        print("we are in epoch", epoch)
        log_writer.set_step(epoch * num_training_steps_per_epoch)
        fit(0.0001, Model, data_loader_train, torch.optim.SGD, loss_fc, devices=device, current_epoch=epoch,
            statistic_dict=args.statistic_path, save_path=save_path, log_writer=log_writer)
        log_writer.flush()


if __name__ == '__main__':
    opts = get_args()
    main(opts)