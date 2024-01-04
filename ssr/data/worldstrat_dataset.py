import os
import cv2
import csv
import torch
import random
import torchvision
import skimage.io
import numpy as np
#from osgeo import gdal
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY

totensor = torchvision.transforms.ToTensor()


@DATASET_REGISTRY.register()
class WorldStratDataset(data.Dataset):
    """
    Dataset object to format the WorldStrat data in a way that works with satlas-super-resolution.
      - Note that this dataset assumes that just RGB low-res bands will be utilized to generate an RGB
    high-res image; code will need to be added to make this work with all low-res bands.
      - Also, this dataset assumes the use of L1C low-res imagery; small change needed to work with L2A.
      - As shown in the WorldStrat inference tutorial, we super-resolve images of shape (160,160) to (640,640).
      - Dataset based on format from data downloaded from Zenodo (not Kaggle).

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            lr_path, hr_path, splits_csv, use_3d, n_s2_images
    """

    def __init__(self, opt):
        super(WorldStratDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        if self.split == 'validation':
            self.split = 'val'

        self.lr_path = opt['lr_path']
        self.hr_path = opt['hr_path']
        self.splits_csv = opt['splits_csv']

        # Flags whether the model being used expects [b, n_images, channels, h, w] or [b, n_images*channels, h, w].
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        # Read in the csv file containing splits and filter out non-relevant images for this split.
        # Build a list of [hr_path, [lr_paths]] lists. 
        self.datapoints = []
        with open(self.splits_csv, newline='') as csvfile:
            read = csv.reader(csvfile, delimiter=' ')
            for i,row in enumerate(read):
                # Skip the row with columns.
                if i == 0:
                    continue

                row = row[0].split(',')
                tile = row[1]
                split = row[-1]
                if split != self.split:
                    continue

                # A few paths are missing even though specified in the split csv, so skip them.
                if not os.path.exists((os.path.join(self.lr_path, tile, 'L1C', tile+'-'+str(1)+'-L1C_data.tiff'))):
                    continue

                # HR image for the current datapoint.
                hr_img_path = os.path.join(self.hr_path, tile, tile+'_rgb.png')

                # Each HR image has 16 corresponding LR images.
                lrs = []
                for img in range(1, int(opt['n_s2_images'])+1):
                    lr_img_path = os.path.join(self.lr_path, tile, 'L1C', tile+'-'+str(img)+'-L1C_data.tiff')
                    lrs.append(lr_img_path)

                self.datapoints.append([hr_img_path, lrs])

        self.data_len = len(self.datapoints)
        print("Loaded ", self.data_len, " data pairs.")

    def __getitem__(self, index):
        hr_path, lr_paths = self.datapoints[index]

        hr_im = skimage.io.imread(hr_path)[:, :, 0:3]  # remove alpha channel
        hr_im = cv2.resize(hr_im, (640, 640)) # resize the HR image to match the SR image
        img_HR = totensor(hr_im)

        # Load each of the LR images with gdal, since they're tifs.
        lr_ims = []
        for lr_path in lr_paths:
            raster = gdal.Open(lr_path)
            array = raster.ReadAsArray()
            lr_im = array.transpose(1, 2, 0)[:, :, 1:4]  # only using RGB bands (bands 2,3,4)
            lr_ims.append(lr_im)

        # Resize each Sentinel-2 image to the same spatial dimension. Then stack along first dimension.
        lr_ims = [totensor(cv2.resize(im, (160,160))) for im in lr_ims]
        img_LR = torch.stack(lr_ims, dim=0)

        # Find a random 40x40 HR chunk, to create more, smaller training samples.
        hr_start_x = random.randint(0, 640-160)
        hr_start_y = random.randint(0, 640-160)
        lr_start_x = int(hr_start_x // 4)
        lr_start_y = int(hr_start_y // 4)

        img_HR = img_HR[:, hr_start_x:hr_start_x+160, hr_start_y:hr_start_y+160]
        img_LR = img_LR[:, :, lr_start_x:lr_start_x+40, lr_start_y:lr_start_y+40]

        if not self.use_3d:
            img_LR = torch.reshape(img_LR, (-1, 40, 40))

        return {'hr': img_HR, 'lr': img_LR, 'Index': index}

    def __len__(self):
        return self.data_len
