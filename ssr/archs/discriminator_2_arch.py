"""
Taken from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/discriminator_arch.py 
Authors: xinntao
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class SSR_UNetDiscriminatorSN2(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True, diff_mod_layers=None):
        super(SSR_UNetDiscriminatorSN2, self).__init__()
        self.skip_connection = skip_connection
        self.diff_mod_layers = diff_mod_layers
        norm = spectral_norm

        # Not sure the best way to code this up yet but diff_modality_first_layers will be a list
        # of [s2 in channels, naip in channels] which will usually be [n_s2_images * 3, 3].
        if self.diff_mod_layers is not None:
            self.s2_conv0 = nn.Conv2d(self.diff_mod_layers[0], num_feat//2, kernel_size=3, stride=1, padding=1)
            self.naip_conv0 = nn.Conv2d(self.diff_mod_layers[1], num_feat//2, kernel_size=3, stride=4, padding=1)
        else:
            # the first convolution
            self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):

        if self.diff_mod_layers is not None:
            s2, naip = x[1], x[0]
            s20 = F.leaky_relu(self.s2_conv0(s2), negative_slope=0.2, inplace=True)
            naip0 = F.leaky_relu(self.naip_conv0(naip), negative_slope=0.2, inplace=True)
            x0 = torch.cat((naip0, s20), dim=1)
        else:
            x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)

        # downsample
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
