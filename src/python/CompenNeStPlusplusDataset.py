'''
CompenNeSt++ dataset for data loader.
'''

import os
from torch.utils.data import Dataset
import cv2 as cv
from os.path import join as fullfile
import torchvision
import numpy as np

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

# Use Pytorch multi-threaded dataloader and opencv to load image faster
class SimpleDataset(Dataset):
    """Simple dataset."""

    def __init__(self, data_root, index=None, size=None):
        self.data_root = data_root
        self.size = size
        # img list
        img_list = sorted(os.listdir(data_root))
        if index is not None: img_list = [img_list[x] for x in index]

        self.img_names = [fullfile(self.data_root, name) for name in img_list]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        assert os.path.isfile(img_name), img_name + ' does not exist'
        im = cv.imread(self.img_names[idx])

        # resize image if size is specified
        if self.size is not None:
            im = cv.resize(im, self.size[::-1])
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        # im = gamma_trans(im, 2.2)  # gamma变换
        return im
