import torch
import torch.nn as nn

from src.model.common import PixArtAlphaCombinedTimestepSizeEmbeddings


class PerChannelRMSNorm(nn.Module):
    """Normalize each channel to unit root-mean-square, preserving sign."""
    def __init__(self, channel_dim=1, eps=1e-8):
        super().__init__()
        self.channel_dim = channel_dim
        self.eps = eps

    def forward(self, x):
        mean_sq = torch.mean(x**2, dim=self.channel_dim, keepdim=True)
        return x * torch.rsqrt(mean_sq + self.eps)


class VideoCausalConv3d(nn.Module):
    """3D convolution that never looks at future frames"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states, causal=True):
        time_kernel_size = self.kernel_size[0]
        if causal:
            # Repeat the first frame backward so the convolution window never reaches
            # past the current frame, which keeps streaming generation causal.
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, time_kernel_size - 1, 1, 1))
            hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)
        else:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            hidden_states = torch.concatenate([pad_left, hidden_states, pad_right], dim=2)
        return self.conv(hidden_states)


class VideoResnetBlock3d(nn.Module):
    """Refine video features through two causal convolutions with a residual link.
    Optionally adds per-channel noise and timestep modulation and bridges
    channel-count changes with a shortcut convolution.
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        dropout=0.0,
        eps=1e-6,
        non_linearity="swish",
        inject_noise=False,
        timestep_conditioning=False,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        if non_linearity != "swish":
            raise ValueError(f"Unsupported non_linearity {non_linearity!r}, only 'swish' is implemented.")
        self.nonlinearity = nn.SiLU()
        self.norm1 = PerChannelRMSNorm()
        self.conv1 = VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = PerChannelRMSNorm()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = VideoCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(in_channels, eps=eps, elementwise_affine=True, bias=True)
            self.conv_shortcut = nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1
            )
        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))
        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(torch.randn(4, in_channels) / in_channels**0.5)

    def forward(self, inputs, temb=None, generator=None, causal=True):
        hidden_states = inputs
        hidden_states = self.norm1(hidden_states)
        if self.scale_shift_table is not None:
            temb = temb.unflatten(1, (4, -1)) + self.scale_shift_table[None, ..., None, None, None]
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal)
        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]
        hidden_states = self.norm2(hidden_states)
        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal)
        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape, generator=generator, device=hidden_states.device, dtype=hidden_states.dtype
            )[None]
            hidden_states = hidden_states + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]
        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)
        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)
        return hidden_states + inputs


class VideoDownsampler3d(nn.Module):
    """Downsample video features while preserving information through averaging"""
    def __init__(self, in_channels, out_channels, stride=1, spatial_padding_mode="zeros"):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.group_size = (in_channels * stride[0] * stride[1] * stride[2]) // out_channels
        out_channels = out_channels // (self.stride[0] * self.stride[1] * self.stride[2])
        self.conv = VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states, causal=True):
        hidden_states = torch.cat([hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2)
        # The shortcut is the input averaged over each output group, so it carries the
        # same spatial layout as the convolution result and can be added to it.
        averaged_input = hidden_states.unflatten(4, (-1, self.stride[2])).unflatten(3, (-1, self.stride[1]))
        averaged_input = averaged_input.unflatten(2, (-1, self.stride[0]))
        averaged_input = averaged_input.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        averaged_input = averaged_input.unflatten(1, (-1, self.group_size))
        averaged_input = averaged_input.mean(dim=2)
        hidden_states = self.conv(hidden_states, causal=causal)
        hidden_states = hidden_states.unflatten(4, (-1, self.stride[2])).unflatten(3, (-1, self.stride[1]))
        hidden_states = hidden_states.unflatten(2, (-1, self.stride[0]))
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        return hidden_states + averaged_input


class VideoUpsampler3d(nn.Module):
    """Upsample video features by rearranging channels into space and time.
    A convolution expands the channels, which are then folded out into extra
    frames, height, and width, with an optional residual shortcut added back.
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        stride=1,
        residual=False,
        upscale_factor=1,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.residual = residual
        self.upscale_factor = upscale_factor
        out_channels = out_channels or in_channels
        out_channels = (out_channels * stride[0] * stride[1] * stride[2]) // upscale_factor
        self.conv = VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states, causal=True):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        if self.residual:
            residual = hidden_states.reshape(
                batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
            )
            residual = residual.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
            repeats = (self.stride[0] * self.stride[1] * self.stride[2]) // self.upscale_factor
            residual = residual.repeat(1, repeats, 1, 1, 1)
            residual = residual[:, :, self.stride[0] - 1:]
        hidden_states = self.conv(hidden_states, causal=causal)
        hidden_states = hidden_states.reshape(
            batch_size, -1, self.stride[0], self.stride[1], self.stride[2], num_frames, height, width
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        hidden_states = hidden_states[:, :, self.stride[0] - 1:]
        if self.residual:
            hidden_states = hidden_states + residual
        return hidden_states


class VideoDownBlock3d(nn.Module):
    """Refine video features, then halve resolution along the configured axes."""
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_layers=1,
        dropout=0.0,
        resnet_eps=1e-6,
        resnet_act_fn="swish",
        spatio_temporal_scale=True,
        downsample_type="conv",
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        resnets = []
        for _ in range(num_layers):
            resnets.append(
                VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList()
            if downsample_type == "conv":
                self.downsamplers.append(
                    VideoCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatial":
                self.downsamplers.append(
                    VideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(1, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "temporal":
                self.downsamplers.append(
                    VideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 1, 1),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatiotemporal":
                self.downsamplers.append(
                    VideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )

    def forward(self, hidden_states, temb=None, generator=None, causal=True):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator, causal=causal)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, causal=causal)
        return hidden_states


class VideoMidBlock3d(nn.Module):
    """Refine video features at the bottleneck without changing resolution."""
    def __init__(
        self,
        in_channels,
        num_layers=1,
        dropout=0.0,
        resnet_eps=1e-6,
        resnet_act_fn="swish",
        inject_noise=False,
        timestep_conditioning=False,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4)
        resnets = []
        for _ in range(num_layers):
            resnets.append(
                VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, generator=None, causal=True):
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator, causal=causal)
        return hidden_states


class VideoUpBlock3d(nn.Module):
    """Refine video features, then double resolution along the configured axes."""
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_layers=1,
        dropout=0.0,
        resnet_eps=1e-6,
        resnet_act_fn="swish",
        spatio_temporal_scale=True,
        upsample_type="spatiotemporal",
        inject_noise=False,
        timestep_conditioning=False,
        upsample_residual=False,
        upscale_factor=1,
        spatial_padding_mode="zeros",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels * 4)
        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = VideoResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            )
        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList()
            if upsample_type == "spatial":
                upsample_stride = (1, 2, 2)
            elif upsample_type == "temporal":
                upsample_stride = (2, 1, 1)
            elif upsample_type == "spatiotemporal":
                upsample_stride = (2, 2, 2)
            self.upsamplers.append(
                VideoUpsampler3d(
                    in_channels=out_channels * upscale_factor,
                    stride=upsample_stride,
                    residual=upsample_residual,
                    upscale_factor=upscale_factor,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        resnets = []
        for _ in range(num_layers):
            resnets.append(
                VideoResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, generator=None, causal=True):
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator, causal=causal)
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, causal=causal)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, generator, causal=causal)
        return hidden_states


class VideoEncoder3d(nn.Module):
    """Compress video into latent features through patching and downsampling.
    Input frames are folded into patches, refined through downsampling blocks,
    and projected to distribution parameters with one extra channel.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        block_out_channels,
        down_block_types,
        spatio_temporal_scaling,
        layers_per_block,
        downsample_type,
        patch_size,
        patch_size_t,
        resnet_norm_eps,
        is_causal,
        spatial_padding_mode,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels * patch_size**2
        self.is_causal = is_causal
        output_channel = out_channels
        self.conv_in = VideoCausalConv3d(
            in_channels=self.in_channels, out_channels=output_channel, kernel_size=3, stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )
        num_block_out_channels = len(block_out_channels)
        self.down_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            if down_block_types[i] == "LTX2VideoDownBlock3D":
                down_block = VideoDownBlock3d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    downsample_type=downsample_type[i],
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"Unknown down block type: {down_block_types[i]}")
            self.down_blocks.append(down_block)
        self.mid_block = VideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[-1],
            resnet_eps=resnet_norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm_out = PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = VideoCausalConv3d(
            in_channels=output_channel,
            out_channels=out_channels + 1,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states, causal=None):
        p = self.patch_size
        p_t = self.patch_size_t
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        causal = causal or self.is_causal
        hidden_states = hidden_states.reshape(
            batch_size, num_channels, post_patch_num_frames, p_t, post_patch_height, p, post_patch_width, p
        )
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        hidden_states = self.conv_in(hidden_states, causal=causal)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states, causal=causal)
        hidden_states = self.mid_block(hidden_states, causal=causal)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)
        # conv_out returns out_channels + 1 channels. Repeating the final channel
        # fills the output to 2 * out_channels, giving the mean and log-variance
        # halves equal length.
        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        return torch.cat([hidden_states, last_channel], dim=1)


class VideoDecoder3d(nn.Module):
    """Rebuild video from latent features through upsampling and unpatching."""
    def __init__(
        self,
        in_channels,
        out_channels,
        block_out_channels,
        spatio_temporal_scaling,
        layers_per_block,
        upsample_type,
        patch_size,
        patch_size_t,
        resnet_norm_eps,
        is_causal,
        inject_noise,
        timestep_conditioning,
        upsample_residual,
        upsample_factor,
        spatial_padding_mode,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels * patch_size**2
        self.is_causal = is_causal
        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))
        output_channel = block_out_channels[0]
        self.conv_in = VideoCausalConv3d(
            in_channels=in_channels, out_channels=output_channel, kernel_size=3, stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.mid_block = VideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            inject_noise=inject_noise[0],
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]
            up_block = VideoUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                upsample_type=upsample_type[i],
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
                spatial_padding_mode=spatial_padding_mode,
            )
            self.up_blocks.append(up_block)
        self.norm_out = PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = VideoCausalConv3d(
            in_channels=output_channel, out_channels=self.out_channels, kernel_size=3, stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.time_embedder = None
        self.scale_shift_table = None
        self.timestep_scale_multiplier = None
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0, dtype=torch.float32))
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(output_channel * 2)
            self.scale_shift_table = nn.Parameter(torch.randn(2, output_channel) / output_channel**0.5)

    def forward(self, hidden_states, temb=None, causal=None):
        causal = causal or self.is_causal
        hidden_states = self.conv_in(hidden_states, causal=causal)
        if self.timestep_scale_multiplier is not None:
            temb = temb * self.timestep_scale_multiplier
        hidden_states = self.mid_block(hidden_states, temb, causal=causal)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states, temb, causal=causal)
        hidden_states = self.norm_out(hidden_states)
        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift, scale = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)
        p = self.patch_size
        p_t = self.patch_size_t
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, -1, p_t, p, p, num_frames, height, width)
        return hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3).flatten(6, 7).flatten(4, 5).flatten(2, 3)


class VideoVAE(nn.Module):
    """Compress video into latents and rebuild it, tracking normalization stats.
    Compression ratios are read from the config.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = VideoEncoder3d(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            spatio_temporal_scaling=config.spatio_temporal_scaling,
            layers_per_block=config.layers_per_block,
            downsample_type=config.downsample_type,
            patch_size=config.patch_size,
            patch_size_t=config.patch_size_t,
            resnet_norm_eps=config.resnet_norm_eps,
            is_causal=config.encoder_causal,
            spatial_padding_mode=config.encoder_spatial_padding_mode,
        )
        self.decoder = VideoDecoder3d(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            block_out_channels=config.decoder_block_out_channels,
            spatio_temporal_scaling=config.decoder_spatio_temporal_scaling,
            layers_per_block=config.decoder_layers_per_block,
            upsample_type=config.upsample_type,
            patch_size=config.patch_size,
            patch_size_t=config.patch_size_t,
            resnet_norm_eps=config.resnet_norm_eps,
            is_causal=config.decoder_causal,
            timestep_conditioning=config.timestep_conditioning,
            inject_noise=config.decoder_inject_noise,
            upsample_residual=config.upsample_residual,
            upsample_factor=config.upsample_factor,
            spatial_padding_mode=config.decoder_spatial_padding_mode,
        )
        latents_mean = torch.zeros((config.latent_channels,), requires_grad=False)
        latents_std = torch.ones((config.latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)
        self.spatial_compression_ratio = config.spatial_compression_ratio
        self.temporal_compression_ratio = config.temporal_compression_ratio

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, temb=None):
        """Decode latents into video frames, optionally modulated by a timestep embedding."""
        return self.decoder(z, temb)
