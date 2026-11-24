
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.initialization import initialize_decoder

from .unet_decoder import UnetDecoder


def get_asym_conv(nc: int) -> nn.Sequential:
    """Create asymmetric convolution module for contrail label refinement.

    Args:
        nc: Target resolution (256, 512, or 1024).

    Returns:
        Sequential conv-relu-conv module.

    Raises:
        NotImplementedError: If nc is not 256, 512, or 1024.
    """
    if nc == 256:
        hidden_size = 9
        asym_conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=(3, 3), padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 1, kernel_size=1),
        )
    elif nc == 512 or nc == 1024:
        hidden_size = 25
        asym_conv = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=(5, 5), padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 1, kernel_size=1),
        )
    else:
        raise NotImplementedError

    return asym_conv


def create_tta(x: torch.Tensor, k: int) -> torch.Tensor:
    """Create test-time augmentation variants with rotation and flipping.
    
    Applies k*90 degree rotation and horizontal flip to generate 2 augmented
    versions for test-time augmentation (TTA).
    
    Args:
        x: Input tensor of shape (batch_size, 3, h, w).
        k: Number of 90-degree rotations (0-3).
    
    Returns:
        Tensor of shape (2*batch_size, 3, h, w) with original rotated images
        in first half and horizontally flipped versions in second half.
    """
    batch_size, nch, h, w = x.shape

    x = torch.rot90(x, k, [2, 3])

    x_tta = torch.zeros((2 * batch_size, nch, h, w), dtype=torch.float32, device=x.device)
    x_tta[:batch_size] = x

    x = torch.flip(x, dims=[3, ])
    x_tta[batch_size:] = x

    return x_tta


class Model(nn.Module):
    # See also TimmUniversalEncoder
    # x (input) is added to features?
    def __init__(self, cfg: dict, pretrained: bool = True) -> None:
        super().__init__()
        name = cfg['encoder']

        self.encoder = timm.create_model(name, features_only=True, pretrained=False)
        encoder_channels = self.encoder.feature_info.channels()
        decoder_channels = cfg['decoder_channels']  # (256, 128, 64, 32, 16)

        assert len(encoder_channels) == len(decoder_channels)

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=0,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-2],  # adhoc change for 1024. Usually -1
            out_channels=1, activation=None, kernel_size=3,
        )

        initialize_decoder(self.decoder)

        self.asym_conv = get_asym_conv(cfg['resize'])

    # def forward(self, x):
    #     batch_size, nch, h, w = x.shape  # (batch_size, 3, 1024, 1024)
    #     y_sym_avg = torch.zeros((batch_size, 1, 512, 512), dtype=torch.float32, device=x.device)
    #     n = 8  # 8 patterns of D4 TTA
    
    #     for k in range(4):
    #         x_tta = create_tta(x, k) 

    #         features = self.encoder(x_tta)

    #         decoder_output = self.decoder(features)

    #         y_sym = self.segmentation_head(decoder_output)

    #         y_sym = y_sym.sigmoid()
    #         y_sym_avg += (1 / n) * torch.rot90(y_sym[:batch_size], -k, dims=(2, 3))
    #         y_sym_avg += (
    #             (1 / n) * torch.rot90(y_sym[batch_size:].flip(dims=[3, ]), -k, dims=(2, 3))
    #         )

    #     # TTA done
    #     y_sym_avg = y_sym_avg.clamp(1e-6, 1 - 1e-6)
    #     y_sym_avg = y_sym_avg.logit()

    #     y_pred = self.asym_conv(y_sym_avg)

    #     return y_sym, y_pred
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (global_batch, 3, 1024, 1024)
        global_batch, _, _, _ = x.shape
        n = 8  # 8 patterns of D4 TTA
        y_sym_avg = None  # will be initialized on first TTA iteration
    
        for k in range(4):
            x_tta = create_tta(x, k)  # shape: (2*global_batch, 3, 1024, 1024)
            features = self.encoder(x_tta)
            decoder_output = self.decoder(features)
            # expected shape: (2*global_batch, 1, H, W)
            y_sym = self.segmentation_head(decoder_output)
            y_sym = y_sym.sigmoid()
    
            # Determine the local batch size for this replica:
            local_batch = y_sym.shape[0] // 2  # since create_tta doubles the batch
    
            # Initialize y_sym_avg based on the actual output shape
            if y_sym_avg is None:
                _, c, H, W = y_sym[:local_batch].shape
                y_sym_avg = torch.zeros(
                    (local_batch, c, H, W), dtype=torch.float32, device=x.device
                )
    
            # Now split the TTA output into the two halves and add the rotated results.
            y_sym_avg += (1 / n) * torch.rot90(y_sym[:local_batch], -k, dims=(2, 3))
            y_sym_avg += (1 / n) * torch.rot90(y_sym[local_batch:].flip(dims=[3]), -k, dims=(2, 3))
    
        # TTA done: adjust and pass through asym_conv
        y_sym_avg = y_sym_avg.clamp(1e-6, 1 - 1e-6)
        y_sym_avg = y_sym_avg.logit()
        y_pred = self.asym_conv(y_sym_avg)
        
        return y_sym, y_pred

