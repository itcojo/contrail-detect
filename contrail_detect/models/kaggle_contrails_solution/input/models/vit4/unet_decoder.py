"""
https://github.com/qubvel/segmentation_models.pytorch
"""

from __future__ import annotations

from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class CenterBlock(nn.Sequential):
    def __init__(
        self: Self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ) -> None:
        """Initialize a double-conv center block (bottleneck).

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_batchnorm: Whether to use batch normalization.
        """
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class DecoderBlock(nn.Module):
    def __init__(
        self: Self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: int = 0,
    ) -> None:
        """Initialize a single U-Net decoder block.

        Args:
            in_channels: Number of channels from the previous decoder stage.
            skip_channels: Number of channels from the encoder skip connection.
            out_channels: Number of output channels after convolution.
            use_batchnorm: Whether to use batch normalization in convolutions.
            dropout: Dropout probability applied to the skip connection.
        """
        super().__init__()

        conv_in_channels = in_channels + skip_channels

        # Convole input embedding and upscaled embedding
        self.conv1 = md.Conv2dReLU(
            conv_in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )

        self.dropout_skip = nn.Dropout(p=dropout)

    def forward(
        self: Self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Upsample and optionally fuse with encoder skip connection.

        Args:
            x: Feature tensor from previous decoder stage, shape (B, C, H, W).
            skip: Optional encoder skip tensor to concatenate, shape (B, S, H, W).

        Returns:
            Decoded feature tensor of shape (B, out_channels, 2H, 2W).
        """
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            skip = self.dropout_skip(skip)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(
        self: Self,
        encoder_channels: list[int],
        decoder_channels: list[int],
        use_batchnorm: bool = True,
        dropout: int = 0,
    ) -> None:
        """Initialize the U-Net decoder from encoder and decoder channel specs.

        Args:
            encoder_channels: List of channel widths from the encoder (finest last).
            decoder_channels: List of output channel widths for each decoder stage.
            use_batchnorm: Whether to use batch normalization in decoder blocks.
            dropout: Dropout probability for skip connections.
        """
        super().__init__()

        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = nn.Identity()

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self: Self, features: list[torch.Tensor]) -> torch.Tensor:
        """Decode a list of encoder feature maps into a segmentation feature map.

        Args:
            features: List of encoder feature tensors, finest first.

        Returns:
            Decoded feature tensor of shape (B, decoder_channels[-1], H, W).
        """
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
