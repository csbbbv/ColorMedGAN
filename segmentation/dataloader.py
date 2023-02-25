import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
# from utils import helpers
import helpers
'''
#ID   Label Name                            R   G   B   A

0     Unknown                               0   0   0   0
1     Cortex                                205 62  78  0
2     Subcortical-Gray-Matter               119 159 176 0
3     White-Matter                          245 245 245 0
4     CSF                                   120 18  134 0
'''
palette = [[0], [1], [2],[3],[4]] 
palette2 = [[0], [1], [2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18],[19],[20],[21],[22],[23],[24],[25],[26],[27],[28],[29],[30],[31],[32],[33],[34],[35]]

num_classes = 36

def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    image_dir = 'orig'
    label_dir = 'label35'
    if mode == 'train':
        img_path = os.path.join(root, image_dir)
        mask_path = os.path.join(root, label_dir)

        if 'Augdata' in root:
            data_list = os.listdir(os.path.join(root, image_dir))
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root,image_dir)
        mask_path = os.path.join(root, label_dir)
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    else:
        pass
    return items

class Brain(data.Dataset):
    def __init__(self, root, mode, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode)
        self.palette = palette2
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.transform = transform
        self.target_transform = target_transform
        self.mean_std =  [11.10725,26.627174]

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.center_crop is not None:
            img, mask = self.center_crop(img, mask)
        img = np.array(img)
        mask = np.array(mask)
        # Image.open读取灰度图像时shape=(H, W) 而非(H, W, 1)
        # 因此先扩展出通道维度，以便在通道维度上进行one-hot映射
        img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, self.palette)
        # shape from (H, W, C) to (C, H, W)
        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])
        if self.transform is not None:
            img = self.transform(img)
            img = (img - self.mean_std[0])/self.mean_std[1]
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)