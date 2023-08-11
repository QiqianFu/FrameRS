
import argparse
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
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
import torch.nn as nn
from utils import TubeMaskingGenerator
from utils import evaluate_fun
from dataset_build import FrameSelect
from model import Best_Frame_Select, fit
from dataset_build import MyDataset

def get_args():
    # parser.add_argument('model_path', type=str, help='checkpoint path of model')
    # parser.add_argument('model_depth', type=str, help='checkpoint path of model')
    parser = argparse.ArgumentParser('Test selector accuracy script', add_help=False)
    parser.add_argument('--selector_path', default= "/home/srtp_ghw/fqq/output_dir/mymodel_success_really_10000_",type=str, help='checkpoint path of model')
    parser.add_argument('--model_path', default="/home/srtp_ghw/fqq/MyMAE8/output_dir/checkpoint-1600.pth" ,type=str, help='checkpoint path of model')
    parser.add_argument('--data_path', default="/home/srtp_ghw/fqq/data2/" ,type=str, help='the path of video')
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


def runit(model_path, model_depth,data_path):
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    args = get_args()
    frame_dict = {}
    j = 0
    for i in range(8):
        for k in range(i + 1, 8):
            frame_dict[(i, k)] = j
            j += 1
    print(args)
    device = torch.device(args.device)
    cudnn.benchmark = True
    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    model.to(device)
    checkpoint = torch.load(model_path, map_location='cpu')  # 这里删掉了agrs
    model.load_state_dict(checkpoint['model'])
    model.eval()

    Model = Best_Frame_Select()
    Model.load_state_dict(torch.load(model_depth, map_location='cpu'))  # 这里删掉了args

    Model = Model.to(device)
    Model.eval()

    accurate = 0
    only_acc = 0
    k=0
    for i in range(100):

        directory = data_path + str(50000+i) + ".mp4"
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
            continue
        list_1 = []
        list_1, img, bool_masked_pos = evaluate_fun(video_data=video_data, args=args, frame_id_list=frame_id_list,
                                                    device=device, patch_size=patch_size, model=model, list_1=list_1)

        shape = bool_masked_pos.shape

        no_mask = np.zeros(shape=shape)
        no_mask = torch.from_numpy(no_mask)
        no_mask = no_mask.unsqueeze(0)
        no_mask = no_mask.to(device, non_blocking=True).flatten(1).to(torch.bool)
        middle_layer = model(img, no_mask, want_middle=True)  # 应该是[1,768*8,14,14] 这里是肯定不对的

        middle_layer = rearrange(middle_layer, 'b c (t p0 p1) -> b (c t) (p0 p1)', p0=14, p1=14)
        # middle_layer = middle_layer#.reshape(1,384*8,14,14)
        # middle_layer = middle_layer.reshape(1,384,8*14*14)
        # avg_method = nn.AvgPool2d(2,stride=2)  #avg默认前两个维度是batch和channel，14是square matrix的宽度
        max_method = nn.MaxPool1d(kernel_size=49, stride=49)
        middle_layer = max_method(middle_layer)
        middle_layer = rearrange(middle_layer, 'b (c t) a -> b c t a', c=384, t=8)

        # print(middle_layer.shape) #这里是1，1568，768
        list_1.sort(key=lambda x: x[1], reverse=False)
        dict = {}
        for i in range(len(list_1)):
            dict[i] = list_1[i][0]
        label = frame_dict[dict[0]]
        label2 = frame_dict[dict[1]]
        middle_layer = middle_layer.clone().detach().reshape(1, 384, 8)

        middle_layer.to(device)

        with torch.no_grad():

            a = Model(middle_layer)

        a = a.cpu().numpy()
        predict_1 = a.argmax().item()
        a[0, a.argmax().item()] = 0
        right2 = a.argmax().item()
        if predict_1 == label:
            accurate += 1
            only_acc += 1
        # elif right2==label2:
        #     accurate+=1
        elif right2 == label:
            accurate += 1
        k+=1
        # elif a.argmax().item()==label2 :
        #     accurate+=1
    # print(accurate/80)
    return accurate / k, only_acc / k


def main(args):
    begin_number = 420
    for i in range(10):
        current = begin_number+20*i
        a,b=runit(model_path=args.model_path,model_depth=args.selector_path+str(current)+".pth",data_path=args.data_path)
        print("now is checkpoint %d, the accurate is %f, first acc is %f"%(current,a,b))



if __name__ == '__main__':
    opts = get_args()
    main(opts)