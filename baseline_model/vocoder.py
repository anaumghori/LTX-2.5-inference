import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Build a normalized low-pass filter using a windowed sinc function.

    Args:
        cutoff: Cutoff frequency as a fraction of the sampling rate.
        half_width: Half-width of the transition band.
        kernel_size: Number of filter taps.

    Returns:
        The normalized filter taps.
    """
    transition_width = 4 * half_width
    half_size = kernel_size // 2
    attenuation_db = 2.285 * (half_size - 1) * math.pi * transition_width + 7.95
    if attenuation_db > 50.0:
        beta = 0.1102 * (attenuation_db - 8.7)
    elif attenuation_db >= 21.0:
        beta = 0.5842 * (attenuation_db - 21) ** 0.4 + 0.07886 * (attenuation_db - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    is_even_length = kernel_size % 2 == 0
    sample_positions = torch.arange(-half_size, half_size) + 0.5 if is_even_length else torch.arange(kernel_size) - half_size
    if cutoff == 0.0:
        filter_taps = torch.zeros_like(sample_positions)
    else:
        sample_positions = 2 * cutoff * sample_positions
        sinc = torch.where(sample_positions == 0, torch.ones_like(sample_positions), torch.sin(math.pi * sample_positions) / math.pi / sample_positions)
        filter_taps = 2 * cutoff * window * sinc
        filter_taps = filter_taps / filter_taps.sum()
    return filter_taps


class DownSample1d(nn.Module):
    """Low-pass filter a signal before reducing its sampling rate."""
    def __init__(self, ratio=2, kernel_size=None, use_padding=True, padding_mode="replicate", persistent=True):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size or int(6 * ratio // 2) * 2
        self.pad_left = self.kernel_size // 2 + (self.kernel_size % 2) - 1
        self.pad_right = self.kernel_size // 2
        self.use_padding = use_padding
        self.padding_mode = padding_mode
        # The cutoff sits at the Nyquist frequency of the decimated signal, so
        # frequencies that would alias are filtered out before the stride drops samples.
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        low_pass_filter = kaiser_sinc_filter1d(cutoff, half_width, self.kernel_size)
        self.register_buffer("filter", low_pass_filter.view(1, 1, self.kernel_size), persistent=persistent)

    def forward(self, x):
        num_channels = x.shape[1]
        if self.use_padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        # Apply the shared filter to each channel independently with grouped
        # convolution, which leaves the channel count unchanged.
        return F.conv1d(x, self.filter.expand(num_channels, -1, -1), stride=self.ratio, groups=num_channels)


class UpSample1d(nn.Module):
    """Increase a signal's sampling rate using a low-pass filter."""
    def __init__(self, ratio=2, kernel_size=None, window_type="kaiser", padding_mode="replicate", persistent=True):
        super().__init__()
        self.ratio = ratio
        self.padding_mode = padding_mode
        # The Hann branch builds its filter inline with rolloff compensation and is
        # used by the sample-rate converter; the Kaiser branch reuses the
        # windowed-sinc design from the downsampler.
        if window_type == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            time_axis = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            sinc_filter = (torch.sinc(time_axis) * window * rolloff / ratio).view(1, 1, -1)
        else:
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.ratio + (self.kernel_size - self.ratio) // 2
            self.pad_right = self.pad * self.ratio + (self.kernel_size - self.ratio + 1) // 2
            sinc_filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", sinc_filter.view(1, 1, self.kernel_size), persistent=persistent)

    def forward(self, x):
        num_channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode=self.padding_mode)
        low_pass_filter = self.filter.to(dtype=x.dtype, device=x.device).expand(num_channels, -1, -1)
        # The transpose inserts zeros between samples, which lowers the amplitude by the
        # ratio, so the output is scaled back up. The slice then trims the filter delay
        # at both ends so the upsampled signal stays time-aligned with the input.
        x = self.ratio * F.conv_transpose1d(x, low_pass_filter, stride=self.ratio, groups=num_channels)
        return x[..., self.pad_left:-self.pad_right]


class SnakeBeta(nn.Module):
    """Apply a periodic activation with learned sine-based adjustments."""
    def __init__(self, channels, alpha=1.0, eps=1e-9, trainable_params=True, logscale=True, use_beta=True):
        super().__init__()
        self.eps = eps
        self.logscale = logscale
        self.use_beta = use_beta
        self.alpha = nn.Parameter(torch.zeros(channels) if self.logscale else torch.ones(channels) * alpha)
        self.alpha.requires_grad = trainable_params
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(channels) if self.logscale else torch.ones(channels) * alpha)
            self.beta.requires_grad = trainable_params

    def forward(self, hidden_states, channel_dim=1):
        broadcast_shape = [1] * hidden_states.ndim
        broadcast_shape[channel_dim] = -1
        alpha = self.alpha.view(broadcast_shape)
        if self.use_beta:
            beta = self.beta.view(broadcast_shape)
        # Alpha and beta are stored in log scale so they stay positive after training.
        # Exponentiating here recovers the actual frequencies and amplitudes.
        if self.logscale:
            alpha = torch.exp(alpha)
            if self.use_beta:
                beta = torch.exp(beta)
        amplitude = beta if self.use_beta else alpha
        # The sine-squared term adds a bounded periodic offset that shrinks as the
        # learned amplitude grows, so each channel controls how much oscillation it adds.
        hidden_states = hidden_states + (1.0 / (amplitude + self.eps)) * torch.sin(hidden_states * alpha).pow(2)
        return hidden_states


class AntiAliasedActivation1d(nn.Module):
    """Apply an activation at a higher sampling rate to reduce aliasing.
    The signal is upsampled before the activation and downsampled afterward.
    """
    def __init__(self, act, ratio=2, kernel_size=12):
        super().__init__()
        self.upsample = UpSample1d(ratio=ratio, kernel_size=kernel_size)
        self.act = act
        self.downsample = DownSample1d(ratio=ratio, kernel_size=kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


class VocoderResBlock(nn.Module):
    """Refine a signal using dilated convolutions and residual connections.
    Each convolution pair is applied sequentially and its output is added to the running signal.
    """
    def __init__(
        self,
        channels,
        kernel_size=3,
        stride=1,
        dilations=(1, 3, 5),
        act_fn="leaky_relu",
        leaky_relu_negative_slope=0.1,
        antialias=False,
        antialias_ratio=2,
        antialias_kernel_size=12,
        padding_mode="same",
    ):
        super().__init__()
        self.dilations = dilations
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_mode,
            )
            for dilation in dilations
        ])
        self.acts1 = nn.ModuleList()
        for _ in range(len(self.convs1)):
            if act_fn == "snakebeta":
                act = SnakeBeta(channels, use_beta=True)
            elif act_fn == "snake":
                act = SnakeBeta(channels, use_beta=False)
            else:
                act = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
            if antialias:
                act = AntiAliasedActivation1d(act, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
            self.acts1.append(act)
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                stride=stride,
                dilation=1,
                padding=padding_mode,
            )
            for _ in range(len(dilations))
        ])
        self.acts2 = nn.ModuleList()
        for _ in range(len(self.convs2)):
            if act_fn == "snakebeta":
                act = SnakeBeta(channels, use_beta=True)
            elif act_fn == "snake":
                act = SnakeBeta(channels, use_beta=False)
            else:
                act = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
            if antialias:
                act = AntiAliasedActivation1d(act, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
            self.acts2.append(act)

    def forward(self, x):
        for dilated_activation, dilated_conv, refinement_activation, refinement_conv in zip(self.acts1, self.convs1, self.acts2, self.convs2):
            residual_update = dilated_activation(x)
            residual_update = dilated_conv(residual_update)
            residual_update = refinement_activation(residual_update)
            residual_update = refinement_conv(residual_update)
            x = x + residual_update
        return x


class VocoderGenerator(nn.Module):
    """Generate a waveform from input features using upsampling and residual blocks."""
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        upsample_kernel_sizes,
        upsample_factors,
        resnet_kernel_sizes,
        resnet_dilations,
        act_fn,
        leaky_relu_negative_slope,
        antialias,
        antialias_ratio,
        antialias_kernel_size,
        final_act_fn,
        final_bias,
    ):
        super().__init__()
        self.num_upsample_layers = len(upsample_kernel_sizes)
        self.resnets_per_upsample = len(resnet_kernel_sizes)
        self.out_channels = out_channels
        self.total_upsample_factor = math.prod(upsample_factors)
        self.act_fn = act_fn
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.final_act_fn = final_act_fn
        if self.num_upsample_layers != len(upsample_factors):
            raise ValueError(
                "`upsample_kernel_sizes` and `upsample_factors` should be lists of the same length but are "
                f"length {self.num_upsample_layers} and {len(upsample_factors)}, respectively."
            )
        if self.resnets_per_upsample != len(resnet_dilations):
            raise ValueError("`resnet_kernel_sizes` and `resnet_dilations` should be lists of the same length.")
        if self.act_fn not in ["snakebeta", "snake", "leaky_relu"]:
            raise ValueError(f"Unsupported activation function: {self.act_fn}.")
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=1, padding=3)
        self.upsamplers = nn.ModuleList()
        self.resnets = nn.ModuleList()
        input_channels = hidden_channels
        for i, (stride, kernel_size) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            output_channels = input_channels // 2
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            )
            for kernel_size, dilations in zip(resnet_kernel_sizes, resnet_dilations):
                self.resnets.append(
                    VocoderResBlock(
                        channels=output_channels,
                        kernel_size=kernel_size,
                        dilations=dilations,
                        act_fn=act_fn,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                        antialias=antialias,
                        antialias_ratio=antialias_ratio,
                        antialias_kernel_size=antialias_kernel_size,
                    )
                )
            input_channels = output_channels
        if act_fn == "snakebeta" or act_fn == "snake":
            act_out = SnakeBeta(channels=output_channels, use_beta=True)
            self.act_out = AntiAliasedActivation1d(act_out, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
        elif act_fn == "leaky_relu":
            self.act_out = nn.LeakyReLU()
        self.conv_out = nn.Conv1d(output_channels, out_channels, 7, stride=1, padding=3, bias=final_bias)

    def forward(self, hidden_states, mel_bins_last=True):
        """Generate a waveform from features laid out as [batch, channels, time, mel bins].

        Pass `mel_bins_last=False` for input that already has time on the last axis.
        """
        # The generator expects [batch, feature, time], so the mel bins are folded into
        # the feature axis before the first convolution.
        if mel_bins_last:
            hidden_states = hidden_states.transpose(2, 3)
        hidden_states = hidden_states.flatten(1, 2)
        hidden_states = self.conv_in(hidden_states)
        for i in range(self.num_upsample_layers):
            if self.act_fn == "leaky_relu":
                hidden_states = F.leaky_relu(hidden_states, negative_slope=self.leaky_relu_negative_slope)
            hidden_states = self.upsamplers[i](hidden_states)
            start = i * self.resnets_per_upsample
            end = (i + 1) * self.resnets_per_upsample
            # Each upsampling stage runs several residual blocks with different kernel
            # sizes, then averages their outputs so no single kernel size dominates.
            resnet_outputs = torch.stack([self.resnets[j](hidden_states) for j in range(start, end)], dim=0)
            hidden_states = torch.mean(resnet_outputs, dim=0)
        hidden_states = self.act_out(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        if self.final_act_fn == "tanh":
            hidden_states = torch.tanh(hidden_states)
        elif self.final_act_fn == "clamp":
            hidden_states = torch.clamp(hidden_states, -1, 1)
        return hidden_states


class CausalSTFT(nn.Module):
    """Compute magnitude and phase spectra using causal waveform frames."""
    def __init__(self, filter_length=512, hop_length=80, window_length=512):
        super().__init__()
        self.hop_length = hop_length
        self.window_length = window_length
        n_freqs = filter_length // 2 + 1
        # These start as zeros and are filled with the real Fourier bases when weights
        # are loaded, so the module carries no meaningful transform until then.
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length), persistent=True)
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length), persistent=True)

    def forward(self, waveform):
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        # Only the left side is padded so each output frame sees past samples and the
        # current window, never future audio. This keeps streaming generation causal.
        left_pad = max(0, self.window_length - self.hop_length)
        waveform = F.pad(waveform, (left_pad, 0))
        spectrum = F.conv1d(waveform, self.forward_basis, stride=self.hop_length, padding=0)
        n_freqs = spectrum.shape[1] // 2
        real, imag = spectrum[:, :n_freqs], spectrum[:, n_freqs:]
        magnitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag.float(), real.float()).to(dtype=real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    """Convert a waveform into a log-mel spectrogram and related outputs."""
    def __init__(self, filter_length=512, hop_length=80, window_length=512, num_mel_channels=64):
        super().__init__()
        self.stft_fn = CausalSTFT(filter_length, hop_length, window_length)
        num_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(num_mel_channels, num_freqs), persistent=True)

    def forward(self, waveform):
        magnitude, phase = self.stft_fn(waveform)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        # Silent frames produce zero mel energy, so clamp before the log to avoid
        # taking the log of zero and producing infinities in the spectrogram.
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class Vocoder(nn.Module):
    """Generate a full-rate waveform from a mel spectrogram.

    The model generates a base waveform, estimates a high-frequency residual
    from its spectrum, adds that residual to an upsampled version of the base
    waveform and trims the result to the expected output length.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hop_length = config.hop_length
        self.vocoder = VocoderGenerator(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=config.out_channels,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            resnet_kernel_sizes=config.resnet_kernel_sizes,
            resnet_dilations=config.resnet_dilations,
            act_fn=config.act_fn,
            leaky_relu_negative_slope=config.leaky_relu_negative_slope,
            antialias=config.antialias,
            antialias_ratio=config.antialias_ratio,
            antialias_kernel_size=config.antialias_kernel_size,
            final_act_fn=config.final_act_fn,
            final_bias=config.final_bias,
        )
        self.bwe_generator = VocoderGenerator(
            in_channels=config.bwe_in_channels,
            hidden_channels=config.bwe_hidden_channels,
            out_channels=config.bwe_out_channels,
            upsample_kernel_sizes=config.bwe_upsample_kernel_sizes,
            upsample_factors=config.bwe_upsample_factors,
            resnet_kernel_sizes=config.bwe_resnet_kernel_sizes,
            resnet_dilations=config.bwe_resnet_dilations,
            act_fn=config.bwe_act_fn,
            leaky_relu_negative_slope=config.bwe_leaky_relu_negative_slope,
            antialias=config.bwe_antialias,
            antialias_ratio=config.bwe_antialias_ratio,
            antialias_kernel_size=config.bwe_antialias_kernel_size,
            final_act_fn=config.bwe_final_act_fn,
            final_bias=config.bwe_final_bias,
        )
        self.mel_stft = MelSTFT(
            filter_length=config.filter_length,
            hop_length=config.hop_length,
            window_length=config.window_length,
            num_mel_channels=config.num_mel_channels,
        )
        self.resampler = UpSample1d(
            ratio=config.output_sampling_rate // config.input_sampling_rate,
            window_type="hann",
            persistent=False,
        )

    def forward(self, mel_spec):
        x = self.vocoder(mel_spec)
        batch_size, num_channels, num_samples = x.shape
        # The spectrum analysis below steps by whole hops, so pad the base waveform up
        # to a hop multiple. The padding is trimmed from the final output below.
        remainder = num_samples % self.hop_length
        if remainder != 0:
            x = F.pad(x, (0, self.hop_length - remainder))
        # The spectrum helper expects one channel per batch entry, so channels are
        # folded into the batch axis for analysis and unfolded right after.
        mel, _, _, _ = self.mel_stft(x.flatten(0, 1))
        mel = mel.unflatten(0, (-1, num_channels))
        mel_for_bandwidth_extension = mel.transpose(2, 3)
        high_frequency_residual = self.bwe_generator(mel_for_bandwidth_extension)
        upsampled_base = self.resampler(x)
        # The bandwidth extension learns only the missing high frequencies, which are
        # added onto the upsampled base waveform. Clamping keeps the sum a valid
        # waveform even if both branches overshoot slightly.
        waveform = torch.clamp(high_frequency_residual + upsampled_base, -1, 1)
        output_samples = num_samples * self.config.output_sampling_rate // self.config.input_sampling_rate
        return waveform[..., :output_samples]
