import math
import torch
import torch.nn as nn

def resolve_device(device=None):
    """Return a concrete torch device, pinning CUDA to an explicit ordinal"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def timestep_frequencies(embedding_dim, downscale_freq_shift=1, max_period=10000, device=None):
    """Build the inverse-frequency ladder used by the timestep embedding.

    The ladder depends only on the embedding width and the frequency settings, so
    it is the same on every call. Callers that encode many timesteps of one shape
    build it once and pass it back through `freqs` instead of rebuilding it.

    Args:
        embedding_dim: Width of the embedding the ladder is used for.
        downscale_freq_shift: Frequency denominator adjustment.
        max_period: Longest oscillation period.
        device: Where to place the result. Left on the CPU when omitted.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    freqs = torch.exp(exponent)
    return freqs if device is None else freqs.to(device=device)


def get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False, downscale_freq_shift=1, scale=1, max_period=10000, freqs=None):
    """Build sinusoidal embeddings for a batch of timesteps.

    Args:
        timesteps: One timestep value per batch entry.
        embedding_dim: Width of the returned embeddings.
        flip_sin_to_cos: Swap the sine and cosine halves.
        downscale_freq_shift: Frequency denominator adjustment.
        scale: Multiplier applied to the timesteps before encoding.
        max_period: Longest oscillation period.
        freqs: Frequency ladder from `timestep_frequencies`, already on the
            timestep device. Built on demand when omitted.
    """
    half_dim = embedding_dim // 2
    if freqs is None:
        freqs = timestep_frequencies(embedding_dim, downscale_freq_shift, max_period, device=timesteps.device)
    emb = scale * timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Project raw timesteps into sinusoidal embeddings."""
    def __init__(self, num_channels, flip_sin_to_cos, downscale_freq_shift, scale=1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self._freqs = {}

    def forward(self, timesteps):
        device = timesteps.device
        freqs = self._freqs.get(device)
        if freqs is None:
            freqs = timestep_frequencies(self.num_channels, self.downscale_freq_shift, device=device)
            self._freqs[device] = freqs
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
            freqs=freqs,
        )


class TimestepEmbedding(nn.Module):
    """Map sinusoidal timestep embeddings through a small feed-forward network."""
    def __init__(self, in_channels, time_embed_dim, out_dim=None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim if out_dim is not None else time_embed_dim, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """Produce the shared timestep embedding used for adaptive modulation."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        return timesteps_emb


def duration_frame_bounds(frame_rate, min_seconds, max_seconds, temporal_compression_ratio):
    """Return the frame counts the configured duration range allows.

    Rejects a range that contains no whole frame or no point on the temporal grid.
    The window is rounded inward, so every accepted frame count stays within the
    configured seconds range.

    Returns:
        The inclusive minimum and maximum frame counts before grid alignment.
    """
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"frame_rate must be a positive finite number, got {frame_rate}.")
    if not math.isfinite(min_seconds) or not math.isfinite(max_seconds):
        raise ValueError(
            f"duration bounds must be finite, got [{min_seconds}, {max_seconds}]."
        )
    if min_seconds <= 0 or max_seconds <= min_seconds:
        raise ValueError(
            f"duration bounds must satisfy 0 < min_seconds < max_seconds, got [{min_seconds}, {max_seconds}]."
        )
    if temporal_compression_ratio < 1:
        raise ValueError(
            f"temporal_compression_ratio must be positive, got {temporal_compression_ratio}."
        )
    min_frames = max(1, math.ceil(min_seconds * frame_rate))
    max_frames = math.floor(max_seconds * frame_rate)
    if max_frames < 1 or min_frames > max_frames:
        raise ValueError(
            f"duration bounds [{min_seconds}s, {max_seconds}s] at {frame_rate} fps do not contain a whole frame."
        )
    first_grid_frame = min_frames + ((1 - min_frames) % temporal_compression_ratio)
    last_grid_frame = max_frames - ((max_frames - 1) % temporal_compression_ratio)
    if first_grid_frame > last_grid_frame:
        raise ValueError(
            f"duration bounds [{min_seconds}s, {max_seconds}s] at {frame_rate} fps admit no frame count on the "
            f"{temporal_compression_ratio}-frame temporal grid; widen the range."
        )
    return min_frames, max_frames


def nearest_valid_frame_count(frame_count, temporal_compression_ratio, min_frames, max_frames):
    """Return the closest valid frame count inside the allowed window.

    Valid counts are one plus multiples of the temporal compression ratio. Ties
    use the shorter video. The bounds must contain at least one grid point.
    """
    if not math.isfinite(frame_count):
        raise ValueError(f"frame_count must be finite, got {frame_count}.")
    if temporal_compression_ratio < 1:
        raise ValueError(
            f"temporal_compression_ratio must be positive, got {temporal_compression_ratio}."
        )
    if min_frames < 1 or max_frames < min_frames:
        raise ValueError(f"invalid frame window [{min_frames}, {max_frames}].")
    first_grid_frame = min_frames + ((1 - min_frames) % temporal_compression_ratio)
    last_grid_frame = max_frames - ((max_frames - 1) % temporal_compression_ratio)
    if first_grid_frame > last_grid_frame:
        raise ValueError(f"frame window [{min_frames}, {max_frames}] contains no temporal grid point.")
    lower = math.floor((frame_count - 1) / temporal_compression_ratio) * temporal_compression_ratio + 1
    upper = lower + temporal_compression_ratio
    nearest = lower if frame_count - lower <= upper - frame_count else upper
    return max(first_grid_frame, min(nearest, last_grid_frame))


def resolve_num_frames(seconds, frame_rate=24.0, temporal_compression_ratio=8, min_seconds=1.0, max_seconds=20.0):
    """Resolve a duration to the nearest valid frame count.
    Durations outside the supported range are clamped rather than rejected.
    NaN is the exception, since it cannot be clamped.
    """
    min_frames, max_frames = duration_frame_bounds(
        frame_rate, min_seconds, max_seconds, temporal_compression_ratio
    )
    try:
        seconds = float(seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"duration must be a real number of seconds, got {seconds!r}.") from exc
    if math.isnan(seconds):
        raise ValueError("duration must be a real number of seconds, got NaN.")
    clamped_seconds = min(max(seconds, min_seconds), max_seconds)
    return nearest_valid_frame_count(
        clamped_seconds * frame_rate, temporal_compression_ratio, min_frames, max_frames
    )


class DiagonalGaussian:
    """Wrap encoder output as a diagonal Gaussian posterior.
    The channel axis holds the mean followed by the log-variance. Sampling draws
    fresh noise, while the mode returns the mean directly for deterministic output.
    """
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator=None):
        return self.mean + self.std * torch.randn(
            self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype
        )

    def mode(self):
        return self.mean
