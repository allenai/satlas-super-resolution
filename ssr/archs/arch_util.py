"""
Taken from: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/arch_util.py
Authors: XPixelGroup
"""
import collections.abc
import math
import torch
import torchvision
import warnings
from math import log2
from torch import nn, Tensor
from typing import Tuple, Optional
from distutils.version import LooseVersion
from itertools import repeat
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


class OneHot(nn.Module):
    """ One-hot encoder. """

    def __init__(self, num_classes):
        """ Initialize the OneHot layer.

        Parameters
        ----------
        num_classes : int
            The number of classes to encode.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        """ Forward pass. 

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The encoded tensor.
        """
        # Current shape of x: (..., 1, H, W).
        x = x.to(torch.int64)
        # Remove the empty dimension: (..., H, W).
        x = x.squeeze(-3)
        # One-hot encode: (..., H, W, num_classes)
        x = F.one_hot(x, num_classes=self.num_classes)

        # Permute the dimensions so the number of classes is before the height and width.
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)  # (..., num_classes, H, W)
        elif x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        return x


class DoubleConv2d(nn.Module):
    """ Two-layer 2D convolutional block with a PReLU activation in between. """

    # TODO: verify if we still need reflect padding-mode. If we do, set via constructor.

    def __init__(self, in_channels, out_channels, kernel_size=3, use_batchnorm=False):
        """ Initialize the DoubleConv2d layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        """
        super().__init__()

        self.doubleconv2d = nn.Sequential(
            # ------- First block -------
            # First convolutional layer.
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
            # Dropout regularization, keep probability 0.5.
            nn.Dropout(p=0.5),
            # ------- Second block -------
            # Second convolutional layer.
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
            # Dropout regularization, keep probability 0.5.
            nn.Dropout(p=0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the DoubleConv2d layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.doubleconv2d(x)

class ResidualBlock(nn.Module):
    """ Two-layer 2D convolutional block (DoubleConv2d) 
    with a skip-connection to a sum."""

    def __init__(self, in_channels, kernel_size=3, **kws):
        """ Initialize the ResidualBlock layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        """
        super().__init__()
        self.residualblock = DoubleConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            **kws,
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the ResidualBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, in_channels, height, width).
        """
        x = x + self.residualblock(x)
        return x


class DenseBlock(ResidualBlock):
    """ Two-layer 2D convolutional block (DoubleConv2d) with a skip-connection 
    to a concatenation (instead of a sum used in ResidualBlock)."""

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the DenseBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, in_channels, height, width).
        """
        return torch.cat([x, self.residualblock(x)], dim=1)


class FusionBlock(nn.Module):
    """ A block that fuses two revisits into one. """

    def __init__(self, in_channels, kernel_size=3, use_batchnorm=False):
        """ Initialize the FusionBlock layer.

        Fuse workflow:
        xx ---> xx ---> x
        |       ^
        |-------^

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        """
        super().__init__()

        # TODO: it might be better to fuse the encodings in groups - one per channel.

        number_of_revisits_to_fuse = 2

        self.fuse = nn.Sequential(
            # A two-layer 2D convolutional block with a skip-connection to a sum.
            ResidualBlock(
                number_of_revisits_to_fuse * in_channels,
                kernel_size,
                use_batchnorm=use_batchnorm,
            ),
            # A 2D convolutional layer.
            nn.Conv2d(
                in_channels=number_of_revisits_to_fuse * in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
        )

    @staticmethod
    def split(x):
        """ Split the input tensor (revisits) into two parts/halves.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        tuple of Tensor
            The two output tensors of shape (batch_size, revisits//2, in_channels, height, width).
        """

        number_of_revisits = x.shape[1]
        #assert number_of_revisits % 2 == 0, f"number_of_revisits={number_of_revisits}"

        # (batch_size, revisits//2, in_channels, height, width)
        first_half = x[:, : number_of_revisits // 2].contiguous()
        second_half = x[:, number_of_revisits // 2 :].contiguous()

        # TODO: return a carry-encoding?
        return first_half, second_half

    def forward(self, x):
        """ Forward pass of the FusionBlock layer.


        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).
            Revisits encoding of the input.

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, revisits/2, in_channels, height, width).
            Fused encoding of the input.
        """

        first_half, second_half = self.split(x)
        batch_size, half_revisits, channels, height, width = first_half.shape

        first_half = first_half.view(
            batch_size * half_revisits, channels, height, width
        )

        second_half = second_half.view(
            batch_size * half_revisits, channels, height, width
        )

        # Current shape of x: (batch_size * revisits//2, 2*in_channels, height, width)
        x = torch.cat([first_half, second_half], dim=-3)

        # Fused shape of x: (batch_size * revisits//2, in_channels, height, width)
        fused_x = self.fuse(x)

        # Fused encoding shape of x: (batch_size, revisits/2, channels, width, height)
        fused_x = fused_x.view(batch_size, half_revisits, channels, height, width)

        return fused_x


class RecursiveFusion(nn.Module):
    """ Recursively fuses a set of encodings. """

    def __init__(self, in_channels, kernel_size, revisits):
        """ Initialize the RecursiveFusion layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int
            The kernel size.
        revisits : int
            The number of revisits.
        """
        super().__init__()

        log2_revisits = log2(revisits)
        if log2_revisits % 1 == 0:
            num_fusion_layers = int(log2_revisits)
        else:
            num_fusion_layers = int(log2_revisits) + 1

        pairwise_fusion = FusionBlock(in_channels, kernel_size, use_batchnorm=False)

        self.fusion = nn.Sequential(
            *(pairwise_fusion for _ in range(num_fusion_layers))
        )

    @staticmethod
    def pad(x):
        """ Pad the input tensor with black revisits to a power of 2.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, revisits, in_channels, height, width).
        """

        # TODO: should we pad with copies of revisits instead of zeros?
        # TODO: move to transforms.py
        batch_size, revisits, channels, height, width = x.shape
        log2_revisits = torch.log2(torch.tensor(revisits))

        if log2_revisits % 1 > 0:

            nextpower = torch.ceil(log2_revisits)
            number_of_black_padding_revisits = int(2 ** nextpower - revisits)

            black_revisits = torch.zeros(
                batch_size,
                number_of_black_padding_revisits,
                channels,
                height,
                width,
                dtype=x.dtype,
                device=x.device,
            )

            x = torch.cat([x, black_revisits], dim=1)
        return x

    def forward(self, x):
        """ Forward pass of the RecursiveFusion layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        Tensor
            The fused output tensor of shape (batch_size, in_channels, height, width).
        """
        x = self.pad(x)  # Zero-pad if neccessary to ensure power of 2 revisits
        x = self.fusion(x)  # (batch_size, 1, channels, height, width)
        return x.squeeze(1)  # (batch_size, channels, height, width)


class ConvTransposeBlock(nn.Module):
    """ Upsampler block with ConvTranspose2d. """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        sr_kernel_size,
        zoom_factor,
        use_batchnorm=False,
    ):
        """ Initialize the ConvTransposeBlock layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        sr_kernel_size : int
            The kernel size of the SR convolution.
        zoom_factor : int
            The zoom factor.
        use_batchnorm : bool, optional
            Whether to use batchnorm, by default False.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # TODO: check if sr_kernel_size is the correct name
        self.sr_kernel_size = sr_kernel_size
        self.zoom_factor = zoom_factor
        self.use_batchnorm = use_batchnorm

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.zoom_factor,
                padding=0,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.in_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.out_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
        )

    def forward(self, x):
        """ Forward pass of the ConvTransposeBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.upsample(x)

class RecursiveFusion(nn.Module):
    """ Recursively fuses a set of encodings. """

    def __init__(self, in_channels, kernel_size, revisits):
        """ Initialize the RecursiveFusion layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int
            The kernel size.
        revisits : int
            The number of revisits.
        """
        super().__init__()

        log2_revisits = log2(revisits)
        if log2_revisits % 1 == 0:
            num_fusion_layers = int(log2_revisits)
        else:
            num_fusion_layers = int(log2_revisits) + 1

        pairwise_fusion = FusionBlock(in_channels, kernel_size, use_batchnorm=False)

        self.fusion = nn.Sequential(
            *(pairwise_fusion for _ in range(num_fusion_layers))
        )

    @staticmethod
    def pad(x):
        """ Pad the input tensor with black revisits to a power of 2.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, revisits, in_channels, height, width).
        """

        # TODO: should we pad with copies of revisits instead of zeros?
        # TODO: move to transforms.py
        batch_size, revisits, channels, height, width = x.shape
        log2_revisits = torch.log2(torch.tensor(revisits))

        if log2_revisits % 1 > 0:

            nextpower = torch.ceil(log2_revisits)
            number_of_black_padding_revisits = int(2 ** nextpower - revisits)

            black_revisits = torch.zeros(
                batch_size,
                number_of_black_padding_revisits,
                channels,
                height,
                width,
                dtype=x.dtype,
                device=x.device,
            )

            x = torch.cat([x, black_revisits], dim=1)
        return x

    def forward(self, x):
        """ Forward pass of the RecursiveFusion layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        Tensor
            The fused output tensor of shape (batch_size, in_channels, height, width).
        """
        x = self.pad(x)  # Zero-pad if neccessary to ensure power of 2 revisits
        x = self.fusion(x)  # (batch_size, 1, channels, height, width)
        return x.squeeze(1)  # (batch_size, channels, height, width)

class PixelShuffleBlock(ConvTransposeBlock):

    """PixelShuffle block with ConvTranspose2d for sub-pixel convolutions. """

    # TODO: add a Dropout layer between the convolution layers?

    def __init__(self, **kws):
        super().__init__(**kws)
        #assert self.in_channels % self.zoom_factor ** 2 == 0
        self.in_channels = self.in_channels // self.zoom_factor ** 2
        self.upsample = nn.Sequential(
            nn.PixelShuffle(self.zoom_factor),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not self.use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.in_channels) if self.use_batchnorm else nn.Identity(),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not self.use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.out_channels) if self.use_batchnorm else nn.Identity(),
            nn.PReLU(),
        )

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
