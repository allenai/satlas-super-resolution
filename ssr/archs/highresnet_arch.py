"""
Adapted from: https://https://github.com/worldstrat/worldstrat/blob/main/src/modules.py
Authors: Ivan Oršolić, Julien Cornebise, Ulf Mertens, Freddie Kalaitzis
"""
from basicsr.utils.registry import ARCH_REGISTRY
from .srcnn_arch import SRCNN
from .arch_util import *

@ARCH_REGISTRY.register()
class HighResNet(SRCNN):
    """ High-resolution CNN.
    Inherits as many elements from SRCNN as possible for as fair a comparison:
    - DoubleConv2d: the in_channels are doubled by use_reference_frame in SRCNN init
    - Encoder: nn.Sequential(DoubleConv2d, ResidualBlock)
    """

    def __init__(self, skip_paddings=True, **kws) -> None:
        super().__init__(**kws)

        self.skip_paddings = skip_paddings
        self.fusion = RecursiveFusion(
            in_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            revisits=self.revisits,
        )

    def forward(self, x, y=None, mask=None):
        """ Forward pass of the High-resolution CNN.

        Parameters
        ----------
        x : Tensor
            The input tensor (low-res revisits).
            Shape: (batch_size, revisits, channels, height, width).
        y : Tensor, optional
            The target tensor (high-res revisits).
            Shape: (batch_size, 1, channels, height, width).
        mask : Tensor, optional
            The mask tensor.
            Shape: (batch_size, revisits, mask_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor (super-resolved images).
            Shape: (batch_size, 1, channels, height, width).
        """
        hidden_channels = self.hidden_channels

        if self.use_reference_frame:
            x = self.compute_and_concat_reference_frame_to_input(x)

        # Note: we could put all these layers in a Sequential, but we
        # would lose the logging abilities for inspection and lose
        # on readability.
        batch_size, revisits, channels, height, width = x.shape
        x = x.view(batch_size * revisits, channels, height, width)

        # Encoded shape: (batch_size * revisits, hidden_channels, height, width)
        x = self.encoder(x)

        x, mask_channels = self.encode_and_concatenate_masks_to_input(
            x, mask, batch_size, revisits, height, width
        )

        x = x.view(batch_size, revisits, hidden_channels + mask_channels, height, width)

        # Fused shape: (batch_size, hidden_channels, height, width)
        x = self.fusion(x)

        # Super-resolved shape:
        # (batch_size, out_channels, height * zoom_factor, width * zoom_factor)
        x = self.sr(x)

        # Pad with empty revisit dimension: (batch_size, 1, channels, height, width)
        x = x[:, None]
        return x
