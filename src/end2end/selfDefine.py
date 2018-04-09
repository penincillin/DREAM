import os, sys, shutil
import random as rd

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

def load_imgs(img_dir, image_list_file, label_file):
    imgs = list()
    max_label = 0
    with open(image_list_file, 'r') as imf:
        with open(label_file, 'r') as laf:
            record = laf.readline().strip().split()
            total_num, label_num = int(record[0]), int(record[1])
            for line in imf:
                img_path = os.path.join(img_dir, line.strip())
                record = laf.readline().strip().split()
                label,yaw = int(record[0]), float(record[1])
                max_label = max(max_label, label)
                imgs.append((img_path, label, yaw))
    assert(total_num == len(imgs))
    assert(label_num == max_label+1)
    return imgs, max_label


class MsCelebDataset(data.Dataset):
    def __init__(self, img_dir, image_list_file, label_file, transform=None):
        self.imgs, self.max_label = load_imgs(img_dir, image_list_file, label_file)
        self.transform = transform

    def __getitem__(self, index):
        path, target, yaw = self.imgs[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target, yaw
    
    def __len__(self):
        return len(self.imgs)




class CaffeCrop(object):
    """
    This class take the same behavior as sensenet
    """
    def __init__(self, phase):
        assert(phase=='train' or phase=='test')
        self.phase = phase

    def __call__(self, img):
        # pre determined parameters
        final_size = 224
        final_width = final_height = final_size
        crop_size = 110
        crop_height = crop_width = crop_size
        crop_center_y_offset = 15
        crop_center_x_offset = 0
        if self.phase == 'train':
            scale_aug = 0.02
            trans_aug = 0.01
        else:
            scale_aug = 0.0
            trans_aug = 0.0
        
        # computed parameters
        randint = rd.randint
        scale_height_diff = (randint(0,1000)/500-1)*scale_aug
        crop_height_aug = crop_height*(1+scale_height_diff)
        scale_width_diff = (randint(0,1000)/500-1)*scale_aug
        crop_width_aug = crop_width*(1+scale_width_diff)


        trans_diff_x = (randint(0,1000)/500-1)*trans_aug
        trans_diff_y = (randint(0,1000)/500-1)*trans_aug


        center = ((img.width/2 + crop_center_x_offset)*(1+trans_diff_x),
                 (img.height/2 + crop_center_y_offset)*(1+trans_diff_y))

        
        if center[0] < crop_width_aug/2:
            crop_width_aug = center[0]*2-0.5
        if center[1] < crop_height_aug/2:
            crop_height_aug = center[1]*2-0.5
        if (center[0]+crop_width_aug/2) >= img.width:
            crop_width_aug = (img.width-center[0])*2-0.5
        if (center[1]+crop_height_aug/2) >= img.height:
            crop_height_aug = (img.height-center[1])*2-0.5

        crop_box = (center[0]-crop_width_aug/2, center[1]-crop_height_aug/2,
                    center[0]+crop_width_aug/2, center[1]+crop_width_aug/2)

        mid_img = img.crop(crop_box)
        res_img = mid_img.resize( (final_width, final_height) )
        return res_img
