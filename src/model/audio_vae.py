import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.common import DiagonalGaussian


class AudioCausalConv2d(nn.Module):
    """2D convolution that never looks beyond the current position along one axis.
    Padding is computed by hand so the causal axis only repeats past values,
    while the other axis uses symmetric padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, causality_axis="height"):
        super().__init__()
        self.causality_axis = causality_axis
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]
        if self.causality_axis == "none":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis in {"width", "width-compatibility"}:
            padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis == "height":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")
        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class AudioPixelNorm(nn.Module):
    """Normalize each position to unit root-mean-square across the channel axis."""
    def __init__(self, dim=1, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        return x * torch.rsqrt(mean_sq + self.eps)


class AudioAttnBlock(nn.Module):
    """Refine audio features with self-attention over the flattened time-frequency grid."""
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.in_channels = in_channels
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm_type == "pixel":
            self.norm = AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1).contiguous()
        k = k.reshape(batch, channels, height * width).contiguous()
        attn = torch.bmm(q, k) * (int(channels) ** (-0.5))
        attn = torch.nn.functional.softmax(attn, dim=2)
        v = v.reshape(batch, channels, height * width)
        attn = attn.permute(0, 2, 1).contiguous()
        h_ = torch.bmm(v, attn).reshape(batch, channels, height, width)
        h_ = self.proj_out(h_)
        return x + h_


class AudioResnetBlock(nn.Module):
    """Refine audio features through two convolutions with a residual link.
    Group normalization cannot track running stats causally, so configurations
    pairing a causal axis with group normalization are rejected up front.
    """
    def __init__(
        self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_channels=512, 
        norm_type="group", causality_axis="height",
    ):
        super().__init__()
        self.causality_axis = causality_axis
        if self.causality_axis is not None and self.causality_axis != "none" and norm_type == "group":
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        if norm_type == "group":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm_type == "pixel":
            self.norm1 = AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.non_linearity = nn.SiLU()
        if causality_axis is not None:
            self.conv1 = AudioCausalConv2d(
                in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        if norm_type == "group":
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        elif norm_type == "pixel":
            self.norm2 = AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.dropout = nn.Dropout(dropout)
        if causality_axis is not None:
            self.conv2 = AudioCausalConv2d(
                out_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
            )
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                if causality_axis is not None:
                    self.conv_shortcut = AudioCausalConv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                    )
                else:
                    self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                if causality_axis is not None:
                    self.nin_shortcut = AudioCausalConv2d(
                        in_channels, out_channels, kernel_size=1, stride=1, causality_axis=causality_axis
                    )
                else:
                    self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = self.non_linearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class AudioDownsample(nn.Module):
    """Halve audio resolution with a causal convolution or plain averaging."""
    def __init__(self, in_channels, with_conv, causality_axis="height"):
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            if self.causality_axis == "none":
                pad = (0, 1, 0, 1)
            elif self.causality_axis == "width":
                pad = (2, 0, 0, 1)
            elif self.causality_axis == "height":
                pad = (0, 1, 2, 0)
            elif self.causality_axis == "width-compatibility":
                pad = (1, 0, 0, 1)
            else:
                raise ValueError(f"Invalid causality_axis {self.causality_axis}")
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class AudioUpsample(nn.Module):
    """Double audio resolution, then trim the extra border row the convolution adds.
    The trim keeps the output causally aligned instead of leaking one step ahead.
    """
    def __init__(self, in_channels, with_conv, causality_axis="height"):
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            if causality_axis is not None:
                self.conv = AudioCausalConv2d(
                    in_channels, in_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                )
            else:
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if not self.with_conv:
            return x
        x = self.conv(x)
        if self.causality_axis == "height":
            return x[:, :, 1:, :]
        if self.causality_axis == "width":
            return x[:, :, :, 1:]
        if self.causality_axis in (None, "none", "width-compatibility"):
            return x
        raise ValueError(f"Invalid causality_axis: {self.causality_axis}")


class AudioEncoder(nn.Module):
    """Compress audio spectrograms into latent features through downsampling."""
    def __init__(
        self, base_channels, output_channels, num_res_blocks, attn_resolutions, in_channels,
        resolution, latent_channels, ch_mult, norm_type, causality_axis,
        dropout, mid_block_add_attention, sample_rate, mel_hop_length, is_causal,
        mel_bins, double_z,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.base_channels = base_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = output_channels
        self.norm_type = norm_type
        self.latent_channels = latent_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis
        base_block_channels = base_channels
        base_resolution = resolution
        if self.causality_axis is not None:
            self.conv_in = AudioCausalConv2d(
                in_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
            )
        else:
            self.conv_in = nn.Conv2d(in_channels, base_block_channels, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution
        for level in range(self.num_resolutions):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]
            for _ in range(self.num_res_blocks):
                stage.block.append(
                    AudioResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout,
                        norm_type=self.norm_type, causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions:
                    if curr_res in self.attn_resolutions:
                        stage.attn.append(AudioAttnBlock(block_in, norm_type=self.norm_type))
            if level != self.num_resolutions - 1:
                stage.downsample = AudioDownsample(block_in, True, causality_axis=self.causality_axis)
                curr_res = curr_res // 2
            self.down.append(stage)
        self.mid = nn.Module()
        self.mid.block_1 = AudioResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout,
            norm_type=self.norm_type, causality_axis=self.causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = AudioAttnBlock(block_in, norm_type=self.norm_type)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = AudioResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout,
            norm_type=self.norm_type, causality_axis=self.causality_axis,
        )
        final_block_channels = block_in
        z_channels = 2 * latent_channels if double_z else latent_channels
        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(num_groups=32, num_channels=final_block_channels, eps=1e-6, affine=True)
        elif self.norm_type == "pixel":
            self.norm_out = AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")
        self.non_linearity = nn.SiLU()
        if self.causality_axis is not None:
            self.conv_out = AudioCausalConv2d(
                final_block_channels, z_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
            )
        else:
            self.conv_out = nn.Conv2d(final_block_channels, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx, block in enumerate(stage.block):
                hidden_states = block(hidden_states, temb=None)
                if stage.attn:
                    hidden_states = stage.attn[block_idx](hidden_states)
            if level != self.num_resolutions - 1 and hasattr(stage, "downsample"):
                hidden_states = stage.downsample(hidden_states)
        hidden_states = self.mid.block_1(hidden_states, temb=None)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states, temb=None)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        return self.conv_out(hidden_states)


class AudioDecoder(nn.Module):
    """Rebuild audio spectrograms from latent features through upsampling.

    The output is cropped or padded to the exact expected time and frequency
    sizes, since causal convolutions shift the feature length.
    """
    def __init__(
        self, base_channels, output_channels, num_res_blocks, attn_resolutions, in_channels,
        resolution, latent_channels, ch_mult, norm_type, causality_axis,
        dropout, mid_block_add_attention, sample_rate, mel_hop_length, is_causal,
        mel_bins,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.base_channels = base_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = output_channels
        self.norm_type = norm_type
        self.latent_channels = latent_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis
        base_block_channels = base_channels * self.channel_multipliers[-1]
        if self.causality_axis is not None:
            self.conv_in = AudioCausalConv2d(
                latent_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
            )
        else:
            self.conv_in = nn.Conv2d(latent_channels, base_block_channels, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = AudioResnetBlock(
            in_channels=base_block_channels, out_channels=base_block_channels, temb_channels=self.temb_ch,
            dropout=dropout, norm_type=self.norm_type, causality_axis=self.causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = AudioAttnBlock(base_block_channels, norm_type=self.norm_type)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = AudioResnetBlock(
            in_channels=base_block_channels, out_channels=base_block_channels, temb_channels=self.temb_ch,
            dropout=dropout, norm_type=self.norm_type, causality_axis=self.causality_axis,
        )
        self.up = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))
        for level in reversed(range(self.num_resolutions)):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]
            for _ in range(self.num_res_blocks + 1):
                stage.block.append(
                    AudioResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout,
                        norm_type=self.norm_type, causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions:
                    if curr_res in self.attn_resolutions:
                        stage.attn.append(AudioAttnBlock(block_in, norm_type=self.norm_type))
            if level != 0:
                stage.upsample = AudioUpsample(block_in, True, causality_axis=self.causality_axis)
                curr_res *= 2
            self.up.insert(0, stage)
        final_block_channels = block_in
        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(num_groups=32, num_channels=final_block_channels, eps=1e-6, affine=True)
        elif self.norm_type == "pixel":
            self.norm_out = AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")
        self.non_linearity = nn.SiLU()
        if self.causality_axis is not None:
            self.conv_out = AudioCausalConv2d(
                final_block_channels, output_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
            )
        else:
            self.conv_out = nn.Conv2d(final_block_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, sample):
        _, _, frames, mel_bins = sample.shape
        target_frames = frames * 4
        if self.causality_axis is not None:
            target_frames = max(target_frames - 3, 1)
        target_channels = self.out_ch
        target_mel_bins = self.mel_bins if self.mel_bins is not None else mel_bins
        decoded = self.conv_in(sample)
        decoded = self.mid.block_1(decoded, temb=None)
        decoded = self.mid.attn_1(decoded)
        decoded = self.mid.block_2(decoded, temb=None)
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                decoded = block(decoded, temb=None)
                if stage.attn:
                    decoded = stage.attn[block_idx](decoded)
            if level != 0 and hasattr(stage, "upsample"):
                decoded = stage.upsample(decoded)
        decoded = self.norm_out(decoded)
        decoded = self.non_linearity(decoded)
        decoded = self.conv_out(decoded)
        _, _, current_frames, current_mel_bins = decoded.shape
        decoded = decoded[
            :, :target_channels, : min(current_frames, target_frames), : min(current_mel_bins, target_mel_bins)
        ]
        time_padding_needed = target_frames - decoded.shape[2]
        freq_padding_needed = target_mel_bins - decoded.shape[3]
        if time_padding_needed > 0 or freq_padding_needed > 0:
            padding = (0, max(freq_padding_needed, 0), 0, max(time_padding_needed, 0))
            decoded = F.pad(decoded, padding)
        return decoded[:, :target_channels, :target_frames, :target_mel_bins]


class AudioVAE(nn.Module):
    """Compress audio spectrograms into latents and rebuild them.
    The encoder is optional because inference only decodes, and leaving it out
    avoids loading its weights.
    """
    def __init__(self, config, with_encoder=True):
        super().__init__()
        self.config = config
        self.encoder = None
        if with_encoder:
            self.encoder = AudioEncoder(
                base_channels=config.base_channels,
                output_channels=config.output_channels,
                num_res_blocks=config.num_res_blocks,
                attn_resolutions=config.attn_resolutions,
                in_channels=config.in_channels,
                resolution=config.resolution,
                latent_channels=config.latent_channels,
                ch_mult=config.ch_mult,
                norm_type=config.norm_type,
                causality_axis=config.causality_axis,
                dropout=config.dropout,
                mid_block_add_attention=config.mid_block_add_attention,
                sample_rate=config.sample_rate,
                mel_hop_length=config.mel_hop_length,
                is_causal=config.is_causal,
                mel_bins=config.mel_bins,
                double_z=config.double_z,
            )
        self.decoder = AudioDecoder(
            base_channels=config.base_channels,
            output_channels=config.output_channels,
            num_res_blocks=config.num_res_blocks,
            attn_resolutions=config.attn_resolutions,
            in_channels=config.in_channels,
            resolution=config.resolution,
            latent_channels=config.latent_channels,
            ch_mult=config.ch_mult,
            norm_type=config.norm_type,
            causality_axis=config.causality_axis,
            dropout=config.dropout,
            mid_block_add_attention=config.mid_block_add_attention,
            sample_rate=config.sample_rate,
            mel_hop_length=config.mel_hop_length,
            is_causal=config.is_causal,
            mel_bins=config.mel_bins,
        )
        latents_std = torch.ones((config.base_channels,))
        latents_mean = torch.zeros((config.base_channels,))
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)
        self.temporal_compression_ratio = config.temporal_compression_ratio
        self.mel_compression_ratio = config.mel_compression_ratio
        self.sample_rate = config.sample_rate
        self.mel_hop_length = config.mel_hop_length

    def encode(self, x):
        if self.encoder is None:
            raise ValueError("AudioVAE was built without an encoder, audio input encoding is not available.")
        h = self.encoder(x)
        posterior = DiagonalGaussian(h)
        return posterior.sample()

    def decode(self, z):
        return self.decoder(z)
