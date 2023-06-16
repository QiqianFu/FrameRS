import torch
import pdb
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
import numpy as np
import modeling_pretrain
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
import torch.nn as nn
from dataset_build import FrameSelect
from torch.utils.data import DataLoader, Dataset
import argparse
from tabnanny import check
import torch
import pdb
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
import modeling_pretrain
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
import torch.nn as nn
class MyDataset(Dataset):

    def __init__(self, data, label_list, data2, list2, data3, list3, data4, list4):
        super(MyDataset, self).__init__()
        self.data = data
        self.label_list = label_list
        self.data2 = data2
        self.list2 = list2
        self.data3 = data3
        self.list3 = list3
        self.data4 = data4
        self.list4 = list4

    def __getitem__(self, index):
        if index < 26640:
            img = self.data[index]

            label = int(self.label_list[index])
        elif index >= 26640 and index < 36640:
            img = self.data2[index - 26640]
            label = int(self.list2[index - 26640])
        elif index >= 36640 and index < 86640:
            img = self.data3[index - 36640]
            label = int(self.list3[index - 36640])
        else:
            img = self.data4[index - 86640]
            label = int(self.list4[index - 86640])

        return img, label

    def __len__(self):
        return int(136638)


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio, frame_list):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame
        self.frame_list = frame_list

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        # mask_per_frame = np.hstack([
        #     np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
        #     np.ones(self.num_masks_per_frame),
        # ])
        # np.random.shuffle(mask_per_frame)
        # mask = np.tile(mask_per_frame, (self.frames,1)).flatten() #沿着y轴将shuffle后的再复制一个大小并flatten
        # return mask #因此frames数为8，一个帧中的height*width应该为196，也就是14*14，因为crop成了224*224的

        mask_ones = np.ones(self.num_patches_per_frame)
        mask_zeros = np.zeros(self.num_patches_per_frame)

        array = []
        mask = []
        for i in range(self.frames):
            array.append(i)
        for i in range(0, self.frames):
            if i not in self.frame_list:
                mask = np.hstack([
                    mask,
                    mask_ones
                ])
            else:
                mask = np.hstack([
                    mask,
                    mask_zeros
                ])

        return mask.flatten()





def evaluate_fun(video_data,args,frame_id_list,device,patch_size,model,list_1):

    for i in range (8):
        for k in range(i+1,8):
            frame_list=[i,k]
            img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
            transforms = DataAugmentationForVideoMAE(args)
            img, bool_masked_pos = transforms((img, None),frame_list=frame_list) # T*C,H,W
            # print(img.shape)
            img = img.view((args.num_frames , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
            # img = img.view(( -1 , args.num_frames) + img.size()[-2:])
            bool_masked_pos = torch.from_numpy(bool_masked_pos)

            with torch.no_grad():
                # img = img[None, :]
                # bool_masked_pos = bool_masked_pos[None, :]
                img = img.unsqueeze(0) #1,3,16,224,224
                bool_masked_pos = bool_masked_pos.unsqueeze(0)
                img = img.to(device, non_blocking=True)
                bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
                std  = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
                ori_img = img * std + mean  # in [0, 1]
                videos_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2,
                                        p1=patch_size[0], p2=patch_size[0])
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                            ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                B, _, C = videos_patch.shape

                labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

            # with torch.cuda.amp.autocast():
                outputs = model(img, bool_masked_pos, want_middle=0)
                loss_func = nn.MSELoss()
                loss = loss_func(input=outputs, target=labels)
                loss_value = loss.item()
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
            a=[(i,k),loss_value]
            list_1.append(a)

    # frame_list=[0,1,2,3,4,5,6,7]
    # img = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
    # transforms = DataAugmentationForVideoMAE(args)
    # img, bool_masked_pos = transforms((img, None),frame_list=frame_list) # T*C,H,W
    # # print(img.shape)
    # img = img.view((args.num_frames , 3) + img.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
    # # img = img.view(( -1 , args.num_frames) + img.size()[-2:])
    # bool_masked_pos = torch.from_numpy(bool_masked_pos)

    # with torch.no_grad():
    #     # img = img[None, :]
    #     # bool_masked_pos = bool_masked_pos[None, :]
    #     img = img.unsqueeze(0) #1,3,16,224,224
    #     bool_masked_pos = bool_masked_pos.unsqueeze(0)
    #     img = img.to(device, non_blocking=True)
    return list_1  ,  img,  bool_masked_pos


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupCenterCrop(args.input_size)
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        self.window_size = args.window_size
        self.mask_ratio = args.mask_ratio

    def __call__(self, images, frame_list):
        process_data, _ = self.transform(images)
        self.masked_position_generator = TubeMaskingGenerator(
            self.window_size, self.mask_ratio, frame_list
        )
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr