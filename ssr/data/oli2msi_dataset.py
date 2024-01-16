import glob
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from osgeo import gdal
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY

totensor = torchvision.transforms.ToTensor()


@DATASET_REGISTRY.register()
class OLI2MSIDataset(data.Dataset):
    """
    Dataset object to format the OLI2MSI data in a way that works with satlas-super-resolution.
      - Data downloaded from the google drive linked on the OLI2MSI github repo.
      - We train on 40x40 LR chunks, for more unique training samples and faster training.
      - The L2 models train with an upsample factor of 4 and then resize output to be upsampled 3x 
      due to how some of the components work. 

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
    """

    def __init__(self, opt):
        super(OLI2MSIDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.data_root = opt['data_root']

        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        if self.split == 'train':
            hr_fps = glob.glob(self.data_root + 'train_hr/*.TIF')
            lr_fps = [hr_fp.replace('train_hr', 'train_lr') for hr_fp in hr_fps]
        else:
            hr_fps = hr_fps = glob.glob(self.data_root + 'test_hr/*.TIF')
            lr_fps = [hr_fp.replace('test_hr', 'test_lr') for hr_fp in hr_fps]
        
        self.datapoints = []
        for i,hr_fp in enumerate(hr_fps):
            self.datapoints.append([hr_fp, lr_fps[i]])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs for split ", self.split)

    def __getitem__(self, index):
        hr_path, lr_path = self.datapoints[index]

        # Load the 480x840 high-res image.
        hr_ds = gdal.Open(hr_path)
        hr_arr = np.array(hr_ds.ReadAsArray())
        hr_tensor = torch.tensor(hr_arr).float()

        # Load the 160x160 low-res image.
        lr_ds = gdal.Open(lr_path)
        lr_arr = np.array(lr_ds.ReadAsArray())
        lr_tensor = torch.tensor(lr_arr).float()

        # Find a random 40x40 HR chunk, to create more, smaller training samples.
        hr_start_x = random.randint(0, 480-120)
        hr_start_y = random.randint(0, 480-120)
        lr_start_x = int(hr_start_x // 3)
        lr_start_y = int(hr_start_y // 3)

        hr_tensor = hr_tensor[:, hr_start_x:hr_start_x+120, hr_start_y:hr_start_y+120]
        lr_tensor = lr_tensor[:, lr_start_x:lr_start_x+40, lr_start_y:lr_start_y+40]

        if self.use_3d:
            lr_tensor = lr_tensor.unsqueeze(0)

        img_HR = hr_tensor
        img_LR = lr_tensor

        return {'hr': img_HR, 'lr': img_LR, 'Index': index}

    def __len__(self):
        return self.data_len
