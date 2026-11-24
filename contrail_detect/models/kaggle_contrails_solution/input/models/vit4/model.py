from typing import Self

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.initialization import initialize_decoder

from .unet_decoder import UnetDecoder


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
class Model(nn.Module):
    # See also TimmUniversalEncoder
    # x (input) is added to features?
    def __init__(self: Self, cfg: dict, pretrained: bool = True) -> None:
        """Initialize vit4 model with 4-panel TTA encoder-decoder and asymmetric head.

        Args:
            cfg: Configuration dict with keys:
                - 'encoder': timm model name.
                - 'decoder_channels': Tuple of decoder channel widths.
                - 'tta': TTA strategy ('none' or 'd4prob').
                - 'resize': Target resolution (256 or 512).
            pretrained: Unused; kept for interface compatibility.
        """
        super().__init__()
        name = cfg['encoder']
        self.encoder = timm.create_model(
            name, 
            features_only=True, 
            pretrained=False,
        )
        encoder_channels = self.encoder.feature_info.channels()
        decoder_channels = cfg['decoder_channels']  # (256, 128, 64, 32, 16)
        assert len(encoder_channels) == len(decoder_channels)

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=0,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1, activation=None, kernel_size=3,
        )

        initialize_decoder(self.decoder)

        nc = cfg['resize']
        self.asym_conv = self.get_asym_conv(nc)

        self.tta = cfg['tta']
        assert self.tta in ['none', 'd4prob']

    
    def create_4panels_tta(self: Self, x: torch.Tensor, k: int):
        """Create 4-panel test-time augmentation.
        
        Args:
            x: Input tensor of shape (batch_size, T, nch, h, w)
            k: Rotation parameter for torch.rot90
        
        Returns:
            Augmented tensor for 4-panel TTA
        """
        batch_size, T, nch, h, w = x.shape

        x = torch.rot90(x, k, [3, 4])

        x4 = torch.zeros((2 * batch_size, 3, 2 * h, 2 * w), dtype=torch.float32, device=x.device)
        x4[:batch_size, :, :h, :w] = x[:, 3]  # t=4
        x4[:batch_size, :, :h, w:] = x[:, 0]
        x4[:batch_size, :, h:, :w] = x[:, 1]
        x4[:batch_size, :, h:, w:] = x[:, 2]

        x = torch.flip(x, dims=[4, ])
        x4[batch_size:, :, :h, :w] = x[:, 3]  # t=4
        x4[batch_size:, :, :h, w:] = x[:, 0]
        x4[batch_size:, :, h:, :w] = x[:, 1]
        x4[batch_size:, :, h:, w:] = x[:, 2]

        return x4

    def forward(self: Self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, T=4, nch=3, h=512, w=512)
        
        Returns:
            Tuple of (features, predictions)
        """
        batch_size, T, nch, h, w = x.shape
        y_sym_avg = torch.zeros((batch_size, 1, h, w), dtype=torch.float32, device=x.device)
        n = 8  # 8 patterns of D4 TTA
        #@ks = [1, ]
        #n = 2 * len(ks)

        for k in range(4):
            # 4 panel image with flip and rot90
            x4 = self.create_4panels_tta(x, k)  # (2 * batch_size, 3, 2h, 2w) for flip

            # Encode
            features4 = self.encoder(x4)  # => list of 5 features
            features = []
            for xf in features4:
                _, _, hf, wf = xf.shape
                features.append(xf[:, :, :(hf // 2), :(wf // 2)])

            # Decode
            decoder_output = self.decoder(features)
            y_sym = self.segmentation_head(decoder_output)  # (2 * batch_size, 1, h, w)

            # TTA average
            y_sym = y_sym.sigmoid()
            #y_sym = torch.rot90(y_sym, -k, dims=(2, 3))
            y_sym_avg += (1 / n) * torch.rot90(y_sym[:batch_size], -k, dims=(2, 3))
            y_sym_avg += (
                (1 / n) * torch.rot90(y_sym[batch_size:].flip(dims=[3, ]), -k, dims=(2, 3))
            )

        # TTA done
        y_sym_avg = y_sym_avg.clamp(1e-6, 1 - 1e-6)
        y_sym_avg = y_sym_avg.logit()

        y_pred = self.asym_conv(y_sym_avg)

        return y_sym, y_pred
    
    
    def get_asym_conv(self: Self, nc: int) -> nn.Module:
        """Create asymmetric convolution module for contrail label refinement.

        Args:
            nc: Target resolution (256 or 512).

        Returns:
            Sequential conv-relu-conv module.

        Raises:
            NotImplementedError: If nc is not 256 or 512.
        """
        if nc == 256:
            hidden_size = 9
            asym_conv = nn.Sequential(
                nn.Conv2d(1, hidden_size, kernel_size=(3, 3), padding=1, padding_mode='replicate'),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size, 1, kernel_size=1),
            )
        elif nc == 512:
            hidden_size = 25
            asym_conv = nn.Sequential(
                nn.Conv2d(1, hidden_size, kernel_size=(5, 5), padding=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size, 1, kernel_size=1),
            )
        else:
            raise NotImplementedError

        return asym_conv

