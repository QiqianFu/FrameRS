import numpy as np


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