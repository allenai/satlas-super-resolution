import os
import glob
import torch
import random
import torchvision
import numpy as np
from torch.utils import data as data
from torch.utils.data import WeightedRandomSampler

from basicsr.utils.registry import DATASET_REGISTRY

from ssr.utils.data_utils import has_black_pixels

random.seed(123)

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples to be sampled.
    Source code: https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

@DATASET_REGISTRY.register()
class S2NAIPDataset(data.Dataset):
    """
    Dataset object for the S2NAIP data. Builds a list of Sentinel-2 time series and NAIP image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            sentinel2_path (str): Data path for Sentinel-2 imagery.
            naip_path (str): Data path for NAIP imagery.
            n_sentinel2_images (int): Number of Sentinel-2 images to use as input to model.
            scale (int): Upsample amount, only 4x is supported currently.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(S2NAIPDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        self.n_s2_images = int(opt['n_s2_images'])
        self.scale = int(opt['scale'])

        # Flags whether the model being used expects [b, n_images, channels, h, w] or [b, n_images*channels, h, w].
        # The L2-based models expect the first shape, while the ESRGAN models expect the latter.
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        # Path to high-res images of older timestamps and corresponding locations to training data.
        # In the case of the S2NAIP dataset, that means NAIP images from 2016-2018.
        self.old_naip_path = opt['old_naip_path'] if 'old_naip_path' in opt else None

        # Sentinel-2 bands to be used as input. Default to just using tci.
        self.s2_bands = opt['s2_bands'] if 's2_bands' in opt else ['tci']
        # Move tci to front of list for later logic.
        self.s2_bands.insert(0, self.s2_bands.pop(self.s2_bands.index('tci')))

        # If a path to older NAIP imagery is provided, build dictionary of each chip:path to png.
        if self.old_naip_path is not None:
            old_naip_chips = {}
            for old_naip in glob.glob(self.old_naip_path + '/**/*.png', recursive=True):
                old_chip = old_naip.split('/')[-1][:-4]

                if not old_chip in old_naip_chips:
                    old_naip_chips[old_chip] = []
                old_naip_chips[old_chip].append(old_naip)

        # Paths to Sentinel-2 and NAIP imagery.
        self.s2_path = opt['sentinel2_path']
        self.naip_path = opt['naip_path']
        if not (os.path.exists(self.s2_path) and os.path.exists(self.naip_path)):
            raise Exception("Please make sure the paths to the data directories are correct.")

        self.naip_chips = glob.glob(self.naip_path + '/**/*.png', recursive=True)

        # Reduce the training set down to a specified number of samples. If not specified, whole set is used.
        if 'train_samples' in opt and self.split == 'train':
            self.naip_chips = random.sample(self.naip_chips, opt['train_samples'])

        self.datapoints = []
        for n in self.naip_chips:
            # Extract the X,Y chip from this NAIP image filepath.
            split_path = n.split('/')
            chip = split_path[-2]

            # If old_hr_path is specified, grab an old high-res image (NAIP) for the current datapoint.
            if self.old_naip_path is not None:
                old_chip = old_naip_chips[chip][0]

            # Gather the filepaths to the Sentinel-2 bands specified in the config.
            s2_paths = [os.path.join(self.s2_path, chip, band + '.png') for band in self.s2_bands]

            # Return the low-res, high-res, and [optionally] older high-res image paths. 
            if self.old_naip_path:
                self.datapoints.append([n, s2_paths, old_chip])
            else:
                self.datapoints.append([n, s2_paths])

        self.data_len = len(self.datapoints)
        print("Number of datapoints for split ", self.split, ": ", self.data_len)

    def get_tile_weight_sampler(self, tile_weights):
        weights = []
        for dp in self.datapoints:
            # Extract the NAIP chip from this datapoint's NAIP path.
            # With the chip, we can index into the tile_weights dict (naip_chip : weight)
            # and then weight this datapoint pair in self.datapoints based on that value.
            naip_path = dp[0]
            split = naip_path.split('/')[-1]
            chip = split[:-4]

            # If the chip isn't in the tile weights dict, then there weren't any OSM features
            # in that chip, so we can set the weight to be relatively low (ex. 1).
            if not chip in tile_weights:
                weights.append(1)
            else:
                weights.append(tile_weights[chip])

        print('Using tile_weight_sampler, min={} max={} mean={}'.format(min(weights), max(weights), np.mean(weights)))
        return CustomWeightedRandomSampler(weights, len(self.datapoints))

    def __getitem__(self, index):

        # A while loop and try/excepts to catch a few images that we want to ignore during 
        # training but do not necessarily want to remove from the dataset, such as the
        # ground truth NAIP image being partially invalid (all black). 
        counter = 0
        while True:
            index += counter  # increment the index based on what errors have been caught
            if index >= self.data_len:
                index = 0

            datapoint = self.datapoints[index]

            if self.old_naip_path:
                naip_path, s2_paths, old_naip_path = datapoint[0], datapoint[1], datapoint[2]
            else:
                naip_path, s2_paths = datapoint[0], datapoint[1]

            # Load the 128x128 NAIP chip in as a tensor of shape [channels, height, width].
            naip_chip = torchvision.io.read_image(naip_path)

            # Check for black pixels (almost certainly invalid) and skip if found.
            if has_black_pixels(naip_chip):
                counter += 1
                continue
            img_HR = naip_chip

            # Load the T*32x32xC S2 files for each band in as a tensor.
            # There are a few rare cases where loading the Sentinel-2 image fails, skip if found.
            try:
                s2_tensor = None
                for i,s2_path in enumerate(s2_paths):

                    # There are tiles where certain bands aren't available, use zero tensors in this case.
                    if not os.path.exists(s2_path):
                        img_size = (self.n_s2_images, 3, 32, 32) if 'tci' in s2_path else (self.n_s2_images, 1, 32, 32)
                        s2_img = torch.zeros(img_size, dtype=torch.uint8)
                    else:
                        s2_img = torchvision.io.read_image(s2_path)
                        s2_img = torch.reshape(s2_img, (-1, s2_img.shape[0], 32, 32))

                    if i == 0:
                        s2_tensor = s2_img
                    else:
                        s2_tensor = torch.cat((s2_tensor, s2_img), dim=1)
            except:
                counter += 1
                continue

            # Skip the cases when there are not as many Sentinel-2 images as requested.
            if s2_tensor.shape[0] < self.n_s2_images:
                counter += 1
                continue

            # Iterate through the 32x32 tci chunks at each timestep, separating them into "good" (valid)
            # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
            tci_chunks = s2_tensor[:, :3, :, :]
            goods, bads = [], []
            for i,ts in enumerate(tci_chunks):
                if has_black_pixels(ts):
                    bads.append(i)
                else:
                    goods.append(i)

            # Pick self.n_s2_images random indices of S2 images to use. Skip ones that are partially black.
            if len(goods) >= self.n_s2_images:
                rand_indices = random.sample(goods, self.n_s2_images)
            else:
                need = self.n_s2_images - len(goods)
                rand_indices = goods + random.sample(bads, need)
            rand_indices_tensor = torch.as_tensor(rand_indices)

            # Extract the self.n_s2_images from the first dimension.
            img_S2 = s2_tensor[rand_indices_tensor]

            # If using a model that expects 5 dimensions, we will not reshape to 4 dimensions.
            if not self.use_3d:
                img_S2 = torch.reshape(img_S2, (-1, 32, 32))

            if self.old_naip_path is not None:
                old_naip_chip = torchvision.io.read_image(old_naip_path)
                img_old_HR = old_naip_chip
                return {'hr': img_HR, 'lr': img_S2, 'old_hr': img_old_HR, 'Index': index}

            return {'hr': img_HR, 'lr': img_S2, 'Index': index}

    def __len__(self):
        return self.data_len
