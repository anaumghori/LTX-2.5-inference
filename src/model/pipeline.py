import copy
import dataclasses
import numpy as np

from src.model.common import resolve_num_frames
from src.model.scheduler import FlowMatchEulerScheduler, calculate_shift, retrieve_timesteps


def select_positive_branch(tensor, text_guidance):
    """Picks the positive half of a [negative, positive] prompt batch"""
    if tensor is None:
        return None
    return tensor.chunk(2, dim=0)[1] if text_guidance else tensor


def pack_video_latents(latents, patch_size=1, patch_size_t=1):
    """Packs spatiotemporal patches of video latents into transformer tokens.
    Args:
        latents: Video latents with shape [batch, channels, frames, height, width].
        patch_size: Spatial patch size along the height and width dimensions.
        patch_size_t: Number of frames grouped into each temporal patch.
    """
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(batch_size, -1, post_patch_num_frames, patch_size_t, post_patch_height, patch_size, post_patch_width, patch_size)
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def unpack_video_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
    """Unpacks transformer tokens into video latents with shape [batch, channels, frames, height, width]."""
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def normalize_video_latents(latents, latents_mean, latents_std, scaling_factor=1.0):
    """Normalizes video latents using the VAE channel statistics and scaling factor."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - latents_mean) * scaling_factor / latents_std


def denormalize_video_latents(latents, latents_mean, latents_std, scaling_factor=1.0):
    """Restores video latents to the VAE's native scale using the channel statistics and scaling factor."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * latents_std / scaling_factor + latents_mean


def pack_audio_latents(latents):
    """Packs audio latents into a sequence of transformer tokens."""
    return latents.transpose(1, 2).flatten(2, 3)


def unpack_audio_latents(latents, num_mel_bins):
    """Unpacks audio tokens into latents with separate channel and mel-bin dimensions."""
    return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)


def denormalize_audio_latents(latents, latents_mean, latents_std):
    """Restores audio latents to the audio VAE's native scale using the channel statistics."""
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


@dataclasses.dataclass(frozen=True)
class VideoGrid:
    """The packed video token grid one denoising stage runs at."""
    num_frames: int
    latent_frames: int
    latent_height: int
    latent_width: int
    tokens: int
    tokens_per_frame: int

    def refined(self):
        """The same clip at twice the spatial resolution."""
        return dataclasses.replace(
            self,
            latent_height=self.latent_height * 2,
            latent_width=self.latent_width * 2,
            tokens=self.tokens * 4,
            tokens_per_frame=self.tokens_per_frame * 4,
        )


@dataclasses.dataclass(frozen=True)
class AudioGrid:
    """The packed audio token grid, which is the same in both denoising stages."""
    frames: int
    mel_bins: int
    tokens: int
    width: int


@dataclasses.dataclass(frozen=True)
class LatentGeometry:
    """The dimensions both GPU sides need to size a request, without loading the other's weights."""
    vae_spatial_compression: int
    vae_temporal_compression: int
    latent_channels: int
    transformer_patch_size: int
    transformer_patch_size_t: int
    audio_sampling_rate: int
    audio_hop_length: int
    audio_temporal_compression: int
    audio_mel_compression: int
    audio_mel_bins: int
    audio_latent_channels: int
    frame_rate: float
    min_seconds: float
    max_seconds: float

    @classmethod
    def from_configs(cls, model_config, engine_config):
        return cls(
            vae_spatial_compression=model_config.vae.spatial_compression_ratio,
            vae_temporal_compression=model_config.vae.temporal_compression_ratio,
            latent_channels=model_config.transformer.in_channels,
            transformer_patch_size=model_config.transformer.patch_size,
            transformer_patch_size_t=model_config.transformer.patch_size_t,
            audio_sampling_rate=model_config.audio_vae.sample_rate,
            audio_hop_length=model_config.audio_vae.mel_hop_length,
            audio_temporal_compression=model_config.audio_vae.temporal_compression_ratio,
            audio_mel_compression=model_config.audio_vae.mel_compression_ratio,
            audio_mel_bins=model_config.audio_vae.mel_bins or 64,
            audio_latent_channels=model_config.audio_vae.latent_channels or 8,
            frame_rate=engine_config.frame_rate,
            min_seconds=engine_config.min_seconds,
            max_seconds=engine_config.max_seconds,
        )

    def resolve_frames(self, duration_seconds):
        """The clip length a requested duration maps to, clamped to the engine's limits."""
        return resolve_num_frames(
            duration_seconds,
            frame_rate=self.frame_rate,
            temporal_compression_ratio=self.vae_temporal_compression,
            min_seconds=self.min_seconds,
            max_seconds=self.max_seconds,
        )

    def video_grid_for(self, num_frames, height, width):
        """The packed token grid a clip length and pixel size produce."""
        latent_frames = (num_frames - 1) // self.vae_temporal_compression + 1
        latent_height = height // self.vae_spatial_compression
        latent_width = width // self.vae_spatial_compression
        if latent_frames < 1 or latent_height < 1 or latent_width < 1:
            raise ValueError(
                f"Latent grid dimensions must be positive, got ({latent_frames}, {latent_height}, {latent_width})."
            )
        tokens = (
            (latent_frames // self.transformer_patch_size_t)
            * (latent_height // self.transformer_patch_size)
            * (latent_width // self.transformer_patch_size)
        )
        return VideoGrid(
            num_frames=num_frames,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            tokens=tokens,
            tokens_per_frame=tokens // latent_frames,
        )

    def video_grid(self, settings):
        """Resolve the requested duration and pixel size into the coarse stage's packed token grid."""
        return self.video_grid_for(
            self.resolve_frames(settings.duration_seconds), settings.height, settings.width
        )

    def audio_grid(self, num_frames):
        """The audio latent grid that matches a clip's length."""
        duration_seconds = num_frames / self.frame_rate
        frames_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_temporal_compression)
        )
        frames = round(duration_seconds * frames_per_second)
        if frames < 1:
            raise ValueError(f"Audio latent frame count must be positive, got {frames}.")
        mel_bins = self.audio_mel_bins // self.audio_mel_compression
        return AudioGrid(
            frames=frames,
            mel_bins=mel_bins,
            tokens=frames,
            width=self.audio_latent_channels * mel_bins,
        )


def _stage_one_sigmas(scheduler_config, settings):
    # A single-step schedule collapses to full noise, which makes the terminal stretch
    # divide by zero and leaves every sigma NaN, so one step is treated as two.
    num_inference_steps = max(int(settings.num_inference_steps), 2)
    sigmas = list(np.linspace(1.0, 1 / num_inference_steps, num_inference_steps))
    # The shift parameter mu is taken at the maximum sequence anchor and shared by the
    # video and audio schedules.
    mu = calculate_shift(
        scheduler_config.max_image_seq_len,
        scheduler_config.base_image_seq_len,
        scheduler_config.max_image_seq_len,
        scheduler_config.base_shift,
        scheduler_config.max_shift,
    )
    return num_inference_steps, sigmas, mu


def stage_one_schedulers(scheduler_config, settings, device):
    """Build the coarse stage's video and audio sampler instances and its timesteps."""
    num_inference_steps, sigmas, mu = _stage_one_sigmas(scheduler_config, settings)
    video = FlowMatchEulerScheduler(scheduler_config)
    audio = FlowMatchEulerScheduler(scheduler_config)
    retrieve_timesteps(audio, num_inference_steps, device, sigmas=sigmas, mu=mu)
    timesteps, _ = retrieve_timesteps(video, num_inference_steps, device, sigmas=sigmas, mu=mu)
    video.set_begin_index(0)
    audio.set_begin_index(0)
    return video, audio, timesteps


def stage_one_timesteps(scheduler_config, settings, device):
    """The coarse stage's timesteps, for the first GPU, which builds the model input."""
    num_inference_steps, sigmas, mu = _stage_one_sigmas(scheduler_config, settings)
    scheduler = FlowMatchEulerScheduler(scheduler_config)
    timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    return timesteps


def _stage_two_config(scheduler_config):
    stage2_config = copy.copy(scheduler_config)
    stage2_config.use_dynamic_shifting = False
    stage2_config.shift_terminal = None
    return stage2_config


def stage_two_schedule(scheduler_config, sigmas, device):
    """Precompute the refinement stage's config, sigmas, and timesteps once on device."""
    stage2_config = _stage_two_config(scheduler_config)
    scheduler = FlowMatchEulerScheduler(stage2_config)
    timesteps, _ = retrieve_timesteps(scheduler, len(sigmas), device, sigmas=sigmas)
    return stage2_config, scheduler.sigmas, timesteps


def renoise_packed_latents(latents, noise_scale, noise):
    """Mix the request's own noise into packed latents to seed the refinement stage."""
    if not 0.0 < float(noise_scale) <= 1.0:
        raise ValueError(f"noise_scale must be in (0, 1], got {noise_scale}.")
    if noise.shape != latents.shape:
        raise ValueError(
            f"Refinement noise has shape {tuple(noise.shape)} but the latents have {tuple(latents.shape)}."
        )
    return noise_scale * noise + (1 - noise_scale) * latents
