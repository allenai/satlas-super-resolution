"""
Adapted from: https://https://github.com/worldstrat/worldstrat/blob/main/src/modules.py
Authors: Ivan Oršolić, Julien Cornebise, Ulf Mertens, Freddie Kalaitzis
"""
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional 
from basicsr.utils.registry import ARCH_REGISTRY
  
from .arch_util import *

@ARCH_REGISTRY.register()
class SRCNN(nn.Module):
    """ Super-resolution CNN.
    Uses no recursive function, revisits are treated as channels.
    """

    def __init__(
        self,
        in_channels,
        mask_channels,
        revisits,
        hidden_channels,
        out_channels,
        kernel_size,
        residual_layers,
        output_size,
        zoom_factor,
        sr_kernel_size,
        use_reference_frame=False,
        **kws,
    ) -> None:
        """ Initialize the Super-resolution CNN.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        mask_channels : int
            The number of mask channels.
        revisits : int
            The number of revisits.
        hidden_channels : int
            The number of hidden channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        residual_layers : int
            The number of residual layers.
        output_size : tuple of int
            The output size.
        zoom_factor : int
            The zoom factor.
        sr_kernel_size : int
            The kernel size of the SR convolution.
        use_reference_frame : bool, optional
            Whether to use the reference frame, by default False.
        """
        super().__init__()

        self.in_channels = 2 * in_channels if use_reference_frame else in_channels
        self.mask_channels = mask_channels
        self.revisits = revisits
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.residual_layers = residual_layers
        self.output_size = output_size
        self.zoom_factor = zoom_factor
        self.sr_kernel_size = sr_kernel_size
        self.use_batchnorm = False
        self.use_reference_frame = use_reference_frame

        # Image encoder
        self.encoder = DoubleConv2d(
            in_channels=self.in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            use_batchnorm=self.use_batchnorm,
        )

        # Mask encoder
        self.mask_encoder = nn.Sequential(
            OneHot(num_classes=12),
            DoubleConv2d(in_channels=self.mask_channels, out_channels=1, kernel_size=3),
            nn.Sigmoid(),
        )

        # Fusion
        self.doubleconv2d = DoubleConv2d(
            in_channels=hidden_channels * revisits,  # revisits as channels
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            use_batchnorm=self.use_batchnorm,
        )
        self.residualblocks = nn.Sequential(
            *(
                ResidualBlock(
                    in_channels=hidden_channels,
                    kernel_size=kernel_size,
                    use_batchnorm=self.use_batchnorm,
                )
                for _ in range(residual_layers)
            )
        )
        self.fusion = nn.Sequential(self.doubleconv2d, self.residualblocks)

        ## Super-resolver (upsampler + renderer)
        self.sr = PixelShuffleBlock(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            sr_kernel_size=sr_kernel_size,
            zoom_factor=zoom_factor,
            use_batchnorm=self.use_batchnorm,
        )
        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )

    def reference_frame(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """ Compute the reference frame as the median of all low-res revisits.

        Parameters
        ----------
        x : Tensor
            The input tensor (low-res revisits).
        mask : Tensor, optional
            The mask tensor of shape.

        Returns
        -------
        Tensor
            The reference frame.
        """
        return x.median(dim=-4, keepdim=True).values

    def forward(
        self, x: Tensor, y: Optional[Tensor] = None, mask: Optional[Tensor] = None,
    ) -> Tensor:
        """ Forward pass of the Super-resolution CNN.

        Parameters
        ----------
        x : Tensor
            The input tensor (low-res revisits).
            Shape: (batch_size, revisits, channels, height, width).
        y : Tensor, optional
            The target tensor (high-res revisits).
            Shape: (batch_size, 1, channels, height, width).
        mask : Tensor, optional
            The mask tensor.y
            Shape: (batch_size, revisits, mask_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor (super-resolved images).
            Shape: (batch_size, 1, channels, height, width).
        """
        if self.use_reference_frame:
            # Concatenated shape: (batch_size, revisits, 2*channels, height, width)
            x = self.compute_and_concat_reference_frame_to_input(x)

        batch_size, revisits, channels, height, width = x.shape
        hidden_channels = self.hidden_channels

        x = x.view(batch_size * revisits, channels, height, width)

        # Encoded shape: (batch_size * revisits, hidden_channels, height, width)
        x = self.encoder(x)

        # Concatenated shape:
        # (batch_size * revisits, hidden_channels+mask_channels, height, width)
        x, mask_channels = self.encode_and_concatenate_masks_to_input(
            x, mask, batch_size, revisits, height, width
        )

        x = x.view(
            batch_size, revisits * (hidden_channels + mask_channels), height, width
        )
        # Fused shape: (batch_size, hidden_channels, height, width)
        x = self.fusion(x)
        x = self.sr(x)

        # Ensure output size of (batch_size, channels, height, width)
        x = self.resize(x)

        # Pad with empty revisit dimension: (batch_size, 1, channels, height, width)
        x = x[:, None]
        return x

    def compute_and_concat_reference_frame_to_input(self, x):
        # Current shape: (batch_size, revisits, channels, height, width)
        reference_frame = self.reference_frame(x).expand_as(x)
        # Concatenated shape: (batch_size, revisits, 2*channels, height, width)
        x = torch.cat([x, reference_frame], dim=-3)
        return x

    def encode_and_concatenate_masks_to_input(
        self, x, mask, batch_size, revisits, height, width
    ):
        if mask is not None:
            mask, mask_channels = mask, self.mask_channels
            mask = mask.view(batch_size * revisits, mask_channels, height, width)
            # Encoded shape: (batch_size * revisits, mask_channels, height, width)
            mask = self.mask_encoder(mask)
            mask_channels = mask.shape[-3]
            # Concatenated shape:
            # (batch_size * revisits, hidden_channels+mask_channels, height, width)
            x = torch.cat([x, mask], dim=-3)
        else:
            mask_channels = 0
        return x, mask_channels

