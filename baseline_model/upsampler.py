import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleResBlock(nn.Module):
    """Refine upsampler features through two convolutions with a residual link."""
    def __init__(self, channels, mid_channels=None, dims=3):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels
        Conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
        self.conv1 = Conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, mid_channels)
        self.conv2 = Conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return self.activation(hidden_states + residual)


class PixelShuffleND(nn.Module):
    """Fold expanded channels into extra spatial or temporal resolution."""
    def __init__(self, dims, upscale_factors=(2, 2, 2)):
        super().__init__()
        self.dims = dims
        self.upscale_factors = upscale_factors
        if dims not in [1, 2, 3]:
            raise ValueError("dims must be 1, 2, or 3")

    def forward(self, x):
        if self.dims == 3:
            return x.unflatten(1, (-1, *self.upscale_factors[:3])).permute(
                0, 1, 5, 2, 6, 3, 7, 4
            ).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        elif self.dims == 2:
            return x.unflatten(1, (-1, *self.upscale_factors[:2])).permute(
                0, 1, 4, 2, 5, 3
            ).flatten(4, 5).flatten(2, 3)
        elif self.dims == 1:
            return x.unflatten(1, (-1, *self.upscale_factors[:1])).permute(
                0, 1, 3, 2, 4, 5
            ).flatten(2, 3)


class BlurDownsample(nn.Module):
    """Downsample with a binomial blur so edges survive the stride.
    The 3D path reuses the 2D blur frame by frame, since the blur itself is spatial.
    """
    def __init__(self, dims, stride, kernel_size=5):
        super().__init__()
        if dims not in (2, 3):
            raise ValueError(f"`dims` must be either 2 or 3 but is {dims}")
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError(f"`kernel_size` must be an odd number >= 3 but is {kernel_size}")
        self.dims = dims
        self.stride = stride
        self.kernel_size = kernel_size
        k = torch.tensor([math.comb(kernel_size - 1, kk) for kk in range(kernel_size)])
        k2d = k[:, None] @ k[None, :]
        k2d = (k2d / k2d.sum()).float()
        self.register_buffer("kernel", k2d[None, None, :, :])

    def forward(self, x):
        if self.stride == 1:
            return x
        if self.dims == 2:
            c = x.shape[1]
            weight = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)
            x = F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.kernel_size // 2, groups=c)
        else:
            b, c, f, _, _ = x.shape
            x = x.transpose(1, 2).flatten(0, 1)
            weight = self.kernel.expand(c, 1, self.kernel_size, self.kernel_size)
            x = F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.kernel_size // 2, groups=c)
            h2, w2 = x.shape[-2:]
            x = x.unflatten(0, (b, f)).reshape(b, -1, f, h2, w2)
        return x


RATIONAL_RESAMPLER_SCALE_MAPPING = {0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)}


class SpatialRationalResampler(nn.Module):
    """Upsample by a fractional ratio through integer upscaling plus blur-downsampling.

    Only ratios with an exact integer decomposition are supported, since anything
    else cannot be split into whole up and down steps.
    """
    def __init__(self, mid_channels=1024, scale=2.0):
        super().__init__()
        self.scale = float(scale)
        num_denom = RATIONAL_RESAMPLER_SCALE_MAPPING.get(scale, None)
        if num_denom is None:
            raise ValueError(
                f"The supplied `scale` {scale} is not supported; "
                f"supported scales are {list(RATIONAL_RESAMPLER_SCALE_MAPPING.keys())}"
            )
        self.num, self.den = num_denom
        self.conv = torch.nn.Conv2d(mid_channels, (self.num ** 2) * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffleND(2, upscale_factors=(self.num, self.num))
        self.blur_down = BlurDownsample(dims=2, stride=self.den)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.blur_down(x)


class LatentUpsampler(nn.Module):
    """Raise latent resolution for the second refinement stage.

    The configuration picks the spatial, temporal, or combined path, so a checkpoint
    that upsamples in both directions folds them out with one 3D shuffle.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        ConvNd = torch.nn.Conv2d if config.dims == 2 else torch.nn.Conv3d
        self.initial_conv = ConvNd(config.in_channels, config.mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, config.mid_channels)
        self.initial_activation = torch.nn.SiLU()
        self.res_blocks = torch.nn.ModuleList([
            UpsampleResBlock(config.mid_channels, dims=config.dims) for _ in range(config.num_blocks_per_stage)
        ])
        if config.spatial_upsample and config.temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(config.mid_channels, 8 * config.mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(3),
            )
        elif config.spatial_upsample:
            if config.use_rational_resampler:
                self.upsampler = SpatialRationalResampler(
                    mid_channels=config.mid_channels, scale=config.rational_spatial_scale
                )
            else:
                self.upsampler = torch.nn.Sequential(
                    torch.nn.Conv2d(config.mid_channels, 4 * config.mid_channels, kernel_size=3, padding=1),
                    PixelShuffleND(2),
                )
        elif config.temporal_upsample:
            self.upsampler = torch.nn.Sequential(
                torch.nn.Conv3d(config.mid_channels, 2 * config.mid_channels, kernel_size=3, padding=1),
                PixelShuffleND(1),
            )
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")
        self.post_upsample_res_blocks = torch.nn.ModuleList([
            UpsampleResBlock(config.mid_channels, dims=config.dims) for _ in range(config.num_blocks_per_stage)
        ])
        self.final_conv = ConvNd(config.mid_channels, config.in_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states):
        config = self.config
        batch_size = hidden_states.shape[0]
        if config.dims == 2:
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
            hidden_states = self.initial_conv(hidden_states)
            hidden_states = self.initial_norm(hidden_states)
            hidden_states = self.initial_activation(hidden_states)
            for block in self.res_blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.upsampler(hidden_states)
            for block in self.post_upsample_res_blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.final_conv(hidden_states)
            hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        else:
            hidden_states = self.initial_conv(hidden_states)
            hidden_states = self.initial_norm(hidden_states)
            hidden_states = self.initial_activation(hidden_states)
            for block in self.res_blocks:
                hidden_states = block(hidden_states)
            if config.temporal_upsample:
                hidden_states = self.upsampler(hidden_states)
                # Temporal upsampling produces one extra leading frame, which is dropped
                # to keep the frame count consistent with the refinement schedule.
                hidden_states = hidden_states[:, :, 1:, :, :]
            else:
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
                hidden_states = self.upsampler(hidden_states)
                hidden_states = hidden_states.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
            for block in self.post_upsample_res_blocks:
                hidden_states = block(hidden_states)
            hidden_states = self.final_conv(hidden_states)
        return hidden_states
