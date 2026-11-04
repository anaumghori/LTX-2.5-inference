import copy
import dataclasses
import numpy as np
import torch
from pathlib import Path
import imageio.v2 as imageio

from src.cache.residual_cache import ResidualCache, cache_for_pass
from src.model.common import (
    DiagonalGaussian,
    duration_frame_bounds,
    resolve_device,
    resolve_num_frames,
)
from src.model.config import EngineConfig
from src.model.logging_utils import get_logger
from src.model.lora import apply_lora, remove_lora
from src.model.scheduler import FlowMatchEulerScheduler, calculate_shift, rescale_noise_cfg, retrieve_timesteps

logger = get_logger(__name__)

# Fallback stage-2 sigma schedule for refinement.
# The scheduler appends the terminal zero internally, so no trailing 0.0 here.
STAGE_2_SIGMA_FALLBACK = [0.909375, 0.725, 0.421875]


def uses_text_guidance(settings):
    """Whether text guidance needs a paired negative and positive prompt batch"""
    return settings.guidance_scale != 1.0 or settings.audio_guidance_scale != 1.0


def select_positive_branch(tensor, text_guidance):
    """Picks the positive half of a [negative, positive] prompt batch"""
    return tensor.chunk(2, dim=0)[1] if text_guidance else tensor


def stage2_generation_settings(settings):
    """Strip guidance from the settings of the distilled stage-2 pass.

    The three-step schedule is distilled to run one pass per step, so every guidance
    scale is returned to its neutral value. Running stage-1 guidance there applies
    the perturbations the distilled model was never trained against.
    """
    return dataclasses.replace(
        settings,
        guidance_scale=1.0,
        stg_scale=0.0,
        modality_scale=1.0,
        guidance_rescale=0.0,
        audio_guidance_scale=1.0,
        audio_stg_scale=0.0,
        audio_modality_scale=1.0,
        audio_guidance_rescale=0.0,
    )


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


def normalize_audio_latents(latents, latents_mean, latents_std):
    """Normalizes audio latents using the audio VAE statistics."""
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents - latents_mean) / latents_std


def denormalize_audio_latents(latents, latents_mean, latents_std):
    """Restores audio latents to the audio VAE's native scale using the channel statistics."""
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


class LTX25Pipeline:
    """Generates synchronized video and audio from text by denoising packed latents with
    the transformer and decoding them through the VAEs.
    """
    def __init__(
        self,
        model_config,
        engine_config=None,
        transformer=None,
        tokenizer=None,
        text_encoder=None,
        connectors=None,
        vae=None,
        audio_vae=None,
        vocoder=None,
        duration_head=None,
        upsampler=None,
        device=None,
        dtype=None,
        lora_entries=None,
        connector_lora_entries=None,
    ):
        if engine_config is None:
            engine_config = EngineConfig()
        self.config = engine_config
        self.model_config = model_config
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.connectors = connectors
        self.vae = vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.duration_head = duration_head
        self.upsampler = upsampler
        self.device = resolve_device(device)
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.lora_entries = lora_entries
        self.connector_lora_entries = connector_lora_entries
        self.scheduler = FlowMatchEulerScheduler(model_config.scheduler)
        self.vae_spatial_compression_ratio = vae.spatial_compression_ratio
        self.vae_temporal_compression_ratio = vae.temporal_compression_ratio
        self.audio_vae_mel_compression_ratio = audio_vae.mel_compression_ratio
        self.audio_vae_temporal_compression_ratio = audio_vae.temporal_compression_ratio
        self.transformer_spatial_patch_size = transformer.config.patch_size
        self.transformer_temporal_patch_size = transformer.config.patch_size_t
        self.audio_sampling_rate = audio_vae.sample_rate
        self.audio_hop_length = audio_vae.mel_hop_length
        self.frame_rate = engine_config.frame_rate
        self.min_seconds = engine_config.min_seconds
        self.max_seconds = engine_config.max_seconds
        duration_frame_bounds(
            self.frame_rate, self.min_seconds, self.max_seconds, self.vae_temporal_compression_ratio
        )

    def encode_text(self, prompt, max_sequence_length):
        """Tokenizes a single prompt and returns text-encoder embeddings with their attention mask."""
        if not isinstance(prompt, str):
            raise ValueError("`prompt` must be a single string, not a list.")
        text_inputs = self.tokenizer(
            prompt.strip(),
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        logger.debug(f"Text encoder input shape: {tuple(input_ids.shape)}")
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(encoder_outputs.hidden_states, dim=-1)
        prompt_embeds = hidden_states.flatten(2, 3).to(dtype=self.dtype)
        logger.debug(f"Prompt embeds shape: {tuple(prompt_embeds.shape)}")
        return prompt_embeds, attention_mask

    def encode_reference_image(self, image_path, height, width, latent_num_frames, device):
        """Cover-crops a reference image to the target size and encodes it to normalized latents repeated across frames."""
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Reference image not found: {image_path}")
        image = imageio.imread(image_path)
        logger.debug(f"Loaded reference image {image_path} with shape {image.shape}")
        if image.ndim == 2:
            image = image[:, :, None].repeat(3, axis=2)
        image = image[:, :, :3]
        image = (
            torch.from_numpy(image)
            .to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        image = image / 255.0 * 2.0 - 1.0
        cover_scale = max(height / image.shape[2], width / image.shape[3])
        resized_h = max(1, round(image.shape[2] * cover_scale))
        resized_w = max(1, round(image.shape[3] * cover_scale))
        image = torch.nn.functional.interpolate(
            image,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )
        top = (resized_h - height) // 2
        left = (resized_w - width) // 2
        image = image[:, :, top:top + height, left:left + width]
        vae_dtype = next(self.vae.parameters()).dtype
        latents = self.vae.encode(image.to(vae_dtype).unsqueeze(2))
        latents = DiagonalGaussian(latents.float()).mode()
        latents = normalize_video_latents(latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor)
        return latents.repeat(1, 1, latent_num_frames, 1, 1)

    def convert_velocity_to_x0(self, sample, velocity_prediction, step_idx, scheduler=None):
        """Converts a velocity prediction to a clean-sample (x0) estimate at the given scheduler step."""
        if scheduler is None:
            scheduler = self.scheduler
        return sample - velocity_prediction * scheduler.sigmas[step_idx]

    def convert_x0_to_velocity(self, sample, x0_prediction, step_idx, scheduler=None):
        """Converts a clean-sample (x0) estimate back to a velocity prediction at the given scheduler step."""
        if scheduler is None:
            scheduler = self.scheduler
        return (sample - x0_prediction) / scheduler.sigmas[step_idx]

    def create_stage2_scheduler(self):
        """Builds the stage-2 scheduler copy without dynamic shifting or terminal stretch"""
        stage2_config = copy.copy(self.scheduler.config)
        stage2_config.use_dynamic_shifting = False
        stage2_config.shift_terminal = None
        return FlowMatchEulerScheduler(stage2_config)

    def renoise_packed_latents(self, latents, noise_scale, generator):
        """Re-noises packed latents with fresh Gaussian noise to seed stage-2 refinement"""
        if not 0.0 < float(noise_scale) <= 1.0:
            raise ValueError(f"noise_scale must be in (0, 1], got {noise_scale}.")
        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=torch.float32)
        return noise_scale * noise + (1 - noise_scale) * latents

    def _denoise_stage(
        self,
        settings,
        latents,
        audio_latents,
        connector_prompt_embeds,
        connector_audio_prompt_embeds,
        connector_attention_mask,
        video_coords,
        audio_coords,
        latent_num_frames,
        latent_height,
        latent_width,
        audio_num_frames,
        scheduler,
        audio_scheduler,
        timesteps,
        ref_strength_mask_packed=None,
        ref_first_frame_tokens=None,
        rotary_emb=None,
        cache=None,
    ):
        """Run the transformer denoising loop over packed latents.

        The prompt embeddings carry the negative branch only when text guidance is
        enabled, so the batch they describe is the batch this stage runs.

        When a cache is passed, it decides once per step whether the middle of the block
        stack is computed or replayed from the residual it holds. The STG pass starts from
        the positive sample's state at the first perturbed block and runs without the cache,
        because it never opens the cache's leading window. Stage 2 also runs without one:
        its three steps leave nothing to reuse and its token grid is four times larger.
        """
        stg_blocks = settings.stg_blocks if settings.stg_blocks is not None else [28]
        text_guidance = connector_prompt_embeds.shape[0] == 2 * latents.shape[0]
        # A guidance pass whose scale is neutral contributes exactly zero, so it is skipped
        # rather than executed and discarded. Stage 2 disables all of them.
        stg_guidance = settings.stg_scale != 0.0 or settings.audio_stg_scale != 0.0
        modality_guidance = settings.modality_scale != 1.0 or settings.audio_modality_scale != 1.0
        # The perturbed blocks are the only thing the STG pass does differently, so the pass can
        # continue from the first of them on the state the positive sample already holds instead of
        # running the blocks before it again. A schedule that perturbs from block 0 leaves no prefix
        # to skip, and the pass then runs the whole stack as it always did.
        stg_start_block = min(stg_blocks)
        if not stg_guidance or stg_start_block == 0:
            stg_start_block = None
        batch_repeat = 2 if text_guidance else 1
        video_prompt_embeds = select_positive_branch(connector_prompt_embeds, text_guidance)
        audio_prompt_embeds = select_positive_branch(connector_audio_prompt_embeds, text_guidance)
        prompt_attn_mask = select_positive_branch(connector_attention_mask, text_guidance)
        # The rotary tables cover a single sample and are broadcast over the prompt batch, so only
        # the reference mask has to be expanded to it.
        # The checkpoint enables use_keyframes_abs_pos_embedding, so the first latent
        # frame is always marked as a reference. The marker is added per packed token,
        # so the mask spans the token sequence rather than the latent frame count.
        tokens_per_frame = latents.shape[1] // latent_num_frames
        video_keyframes_mask = torch.zeros(
            (1, latents.shape[1], 1),
            device=latents.device,
            dtype=torch.float32,
        )
        video_keyframes_mask[:, :tokens_per_frame] = 1.0
        for i, t in enumerate(timesteps):
            if cache is not None:
                cache.begin_step(i)
            captured_states = {}
            latent_model_input = torch.cat([latents] * batch_repeat).to(self.dtype)
            audio_latent_model_input = torch.cat([audio_latents] * batch_repeat).to(self.dtype)
            timestep = t.expand(batch_repeat)
            if ref_strength_mask_packed is not None:
                step_ref_mask = ref_strength_mask_packed.repeat(
                    (batch_repeat,) + (1,) * (ref_strength_mask_packed.ndim - 1)
                )
                video_timestep = timestep.unsqueeze(-1) * (1 - step_ref_mask)
            else:
                video_timestep = timestep
            video_velocity_pred, audio_velocity_pred = self.transformer(
                hidden_states=latent_model_input,
                audio_hidden_states=audio_latent_model_input,
                encoder_hidden_states=connector_prompt_embeds,
                audio_encoder_hidden_states=connector_audio_prompt_embeds,
                timestep=video_timestep,
                audio_timestep=timestep,
                sigma=timestep,
                encoder_attention_mask=connector_attention_mask,
                audio_encoder_attention_mask=connector_attention_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                fps=self.frame_rate,
                audio_num_frames=audio_num_frames,
                video_coords=video_coords,
                audio_coords=audio_coords,
                rotary_emb=rotary_emb,
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=True,
                video_keyframes_mask=video_keyframes_mask,
                capture_block=stg_start_block,
                captured_states=captured_states,
                cache=cache_for_pass(cache, "cfg"),
            )
            video_velocity_pred = video_velocity_pred.float()
            audio_velocity_pred = audio_velocity_pred.float()
            if text_guidance:
                video_velocity_neg_prompt, video_velocity_pred = video_velocity_pred.chunk(2)
                audio_velocity_neg_prompt, audio_velocity_pred = audio_velocity_pred.chunk(2)
            video_x0_pred = self.convert_velocity_to_x0(latents, video_velocity_pred, i, scheduler)
            audio_x0_pred = self.convert_velocity_to_x0(audio_latents, audio_velocity_pred, i, audio_scheduler)
            if text_guidance:
                video_x0_neg_prompt = self.convert_velocity_to_x0(
                    latents, video_velocity_neg_prompt, i, scheduler
                )
                audio_x0_neg_prompt = self.convert_velocity_to_x0(
                    audio_latents, audio_velocity_neg_prompt, i, audio_scheduler
                )
                video_cfg_x0_delta = (settings.guidance_scale - 1) * (video_x0_pred - video_x0_neg_prompt)
                audio_cfg_x0_delta = (settings.audio_guidance_scale - 1) * (
                    audio_x0_pred - audio_x0_neg_prompt
                )
            else:
                video_cfg_x0_delta = audio_cfg_x0_delta = 0
            # The remaining passes carry no negative branch, so they run one sample.
            if stg_guidance or modality_guidance:
                timestep = timestep[:1]
                video_timestep = video_timestep[:1]
            if stg_guidance:
                # The state the positive pass held entering the first perturbed block. A step whose
                # middle the cache replaced never reaches that block, so the pass runs in full there.
                stg_start_states = captured_states.pop("state", None)
                if stg_start_block is not None and stg_start_states is None:
                    logger.debug(
                        f"Step {i + 1}/{len(timesteps)}: the CFG pass stopped short of block {stg_start_block}, "
                        "so the STG pass runs in full"
                    )
                video_velocity_stg_perturbed, audio_velocity_stg_perturbed = self.transformer(
                    hidden_states=latents.to(dtype=self.dtype),
                    audio_hidden_states=audio_latents.to(dtype=self.dtype),
                    encoder_hidden_states=video_prompt_embeds,
                    audio_encoder_hidden_states=audio_prompt_embeds,
                    timestep=video_timestep,
                    audio_timestep=timestep,
                    sigma=timestep,
                    encoder_attention_mask=prompt_attn_mask,
                    audio_encoder_attention_mask=prompt_attn_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=self.frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    rotary_emb=rotary_emb,
                    isolate_modalities=False,
                    spatio_temporal_guidance_blocks=stg_blocks,
                    perturbation_mask=None,
                    use_cross_timestep=True,
                    video_keyframes_mask=video_keyframes_mask,
                    start_block=stg_start_block if stg_start_states is not None else None,
                    start_states=stg_start_states,
                    # Once the pass skips the stack prefix it can neither refresh the cache window that
                    # opens at the cache boundary nor replay it, so it runs without one either way.
                    cache=None if stg_start_block is not None else cache_for_pass(cache, "stg"),
                )
                video_x0_stg_perturbed = self.convert_velocity_to_x0(
                    latents, video_velocity_stg_perturbed.float(), i, scheduler
                )
                audio_x0_stg_perturbed = self.convert_velocity_to_x0(
                    audio_latents, audio_velocity_stg_perturbed.float(), i, audio_scheduler
                )
                video_stg_x0_delta = settings.stg_scale * (video_x0_pred - video_x0_stg_perturbed)
                audio_stg_x0_delta = settings.audio_stg_scale * (audio_x0_pred - audio_x0_stg_perturbed)
            else:
                video_stg_x0_delta = audio_stg_x0_delta = 0
            if modality_guidance:
                video_velocity_modality_isolated, audio_velocity_modality_isolated = self.transformer(
                    hidden_states=latents.to(dtype=self.dtype),
                    audio_hidden_states=audio_latents.to(dtype=self.dtype),
                    encoder_hidden_states=video_prompt_embeds,
                    audio_encoder_hidden_states=audio_prompt_embeds,
                    timestep=video_timestep,
                    audio_timestep=timestep,
                    sigma=timestep,
                    encoder_attention_mask=prompt_attn_mask,
                    audio_encoder_attention_mask=prompt_attn_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=self.frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    rotary_emb=rotary_emb,
                    isolate_modalities=True,
                    spatio_temporal_guidance_blocks=None,
                    perturbation_mask=None,
                    use_cross_timestep=True,
                    video_keyframes_mask=video_keyframes_mask,
                    cache=cache_for_pass(cache, "modality"),
                )
                video_x0_modality_isolated = self.convert_velocity_to_x0(
                    latents, video_velocity_modality_isolated.float(), i, scheduler
                )
                audio_x0_modality_isolated = self.convert_velocity_to_x0(
                    audio_latents, audio_velocity_modality_isolated.float(), i, audio_scheduler
                )
                video_modality_x0_delta = (settings.modality_scale - 1) * (
                    video_x0_pred - video_x0_modality_isolated
                )
                audio_modality_x0_delta = (settings.audio_modality_scale - 1) * (
                    audio_x0_pred - audio_x0_modality_isolated
                )
            else:
                video_modality_x0_delta = audio_modality_x0_delta = 0
            video_x0_guided = video_x0_pred + video_cfg_x0_delta + video_stg_x0_delta + video_modality_x0_delta
            audio_x0_guided = audio_x0_pred + audio_cfg_x0_delta + audio_stg_x0_delta + audio_modality_x0_delta
            video_x0_pred = rescale_noise_cfg(
                video_x0_guided, video_x0_pred, guidance_rescale=settings.guidance_rescale
            )
            audio_x0_pred = rescale_noise_cfg(
                audio_x0_guided, audio_x0_pred, guidance_rescale=settings.audio_guidance_rescale
            )
            video_guided_velocity = self.convert_x0_to_velocity(latents, video_x0_pred, i, scheduler)
            audio_guided_velocity = self.convert_x0_to_velocity(audio_latents, audio_x0_pred, i, audio_scheduler)
            latents = scheduler.step(video_guided_velocity, t, latents)
            audio_latents = audio_scheduler.step(audio_guided_velocity, t, audio_latents)
            if ref_strength_mask_packed is not None and ref_first_frame_tokens is not None:
                # For image-conditioned generation, frame 0 must stay close to the reference image
                # while the other frames are freely denoised. Frame 0 is the leading token block, so
                # the blend runs in packed space: image_strength 1.0 pins it exactly, lower values
                # let it drift with the generation.
                latents[:, :tokens_per_frame] = (
                    settings.image_strength * ref_first_frame_tokens
                    + (1 - settings.image_strength) * latents[:, :tokens_per_frame]
                )
        return latents, audio_latents

    @torch.no_grad()
    def generate(self, settings=None):
        """Generates synchronized video and audio from settings through denoising, optional upsampling, and VAE decoding."""
        if settings is None:
            settings = self.config.generation

        
        if settings.height % 32 != 0 or settings.width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {settings.height} and {settings.width}.")
        if not 0.0 <= settings.image_strength <= 1.0:
            raise ValueError(f"`image_strength` must be in [0, 1], got {settings.image_strength}.")
        device = self.device
        generator = torch.Generator(device=device).manual_seed(settings.seed)
        if not isinstance(settings.prompt, str) or not settings.prompt.strip():
            raise ValueError("`prompt` must be a single non-empty string.")
        if not isinstance(settings.negative_prompt, str):
            raise ValueError("`negative_prompt` must be a single string, not a list.")
        prompt_embeds, prompt_attention_mask = self.encode_text(settings.prompt, settings.max_sequence_length)
        raw_prompt_embeds, raw_prompt_attention_mask = prompt_embeds, prompt_attention_mask
        negative_prompt_embeds, negative_prompt_attention_mask = self.encode_text(
            settings.negative_prompt, settings.max_sequence_length
        )
        # The negative branch only exists when a guidance scale needs it.
        if uses_text_guidance(settings):
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, prompt_attention_mask, padding_side="left"
        )
        logger.debug(f"Connector video embeds: {tuple(connector_prompt_embeds.shape)}, audio embeds: {tuple(connector_audio_prompt_embeds.shape)}")

        # A request may provide a duration; otherwise the duration head predicts one.
        # Both paths share the same clamping and nearest-grid resolution, so an
        # explicit request and an automatic prediction cannot drift apart.

        if settings.duration_seconds is None:
            if self.duration_head is None:
                raise ValueError(
                    "Automatic duration requires a duration_head; load the duration head or provide duration_seconds."
                )
            video_tokens = select_positive_branch(connector_prompt_embeds, prompt_embeds.shape[0] == 2)
            audio_tokens = select_positive_branch(connector_audio_prompt_embeds, prompt_embeds.shape[0] == 2)
            requested_seconds = self.duration_head(video_tokens[:1], audio_tokens[:1]).item()
            logger.debug(f"Predicted duration {requested_seconds:.2f}s")
        else:
            requested_seconds = settings.duration_seconds
        if requested_seconds < self.min_seconds or requested_seconds > self.max_seconds:
            logger.warning(
                f"Duration {float(requested_seconds):.2f}s outside [{self.min_seconds:.2f}s, {self.max_seconds:.2f}s], "
                "clamping"
            )
        num_frames = resolve_num_frames(
            requested_seconds,
            frame_rate=self.frame_rate,
            temporal_compression_ratio=self.vae_temporal_compression_ratio,
            min_seconds=self.min_seconds,
            max_seconds=self.max_seconds,
        )
        logger.debug(f"Duration {float(requested_seconds):.2f}s -> {num_frames} frames ({num_frames / self.frame_rate:.2f}s)")
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = settings.height // self.vae_spatial_compression_ratio
        latent_width = settings.width // self.vae_spatial_compression_ratio
        if latent_num_frames < 1 or latent_height < 1 or latent_width < 1:
            raise ValueError(f"Latent grid dimensions must be positive, got ({latent_num_frames}, {latent_height}, {latent_width}).")
        logger.debug(f"Video grid: {num_frames} frames -> latent ({latent_num_frames}, {latent_height}, {latent_width})")
        shape = (1, self.transformer.config.in_channels, latent_num_frames, latent_height, latent_width)
        ref_strength_mask_packed = None
        ref_first_frame_tokens = None
        if settings.image_path:
            if self.transformer_spatial_patch_size != 1 or self.transformer_temporal_patch_size != 1:
                raise ValueError(
                    "Image conditioning pins frame 0 in packed token space, which needs unpatched latents, "
                    f"but the transformer patch sizes are ({self.transformer_spatial_patch_size}, "
                    f"{self.transformer_temporal_patch_size})."
                )
            clean_latents = self.encode_reference_image(
                settings.image_path, settings.height, settings.width, latent_num_frames, device
            )
            ref_strength_mask = torch.zeros(
                (1, 1, latent_num_frames, latent_height, latent_width),
                device=device,
                dtype=torch.float32,
            )
            ref_strength_mask[:, :, 0] = settings.image_strength
            noise_latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            latents = clean_latents * ref_strength_mask + noise_latents * (1 - ref_strength_mask)
            ref_first_frame_tokens = pack_video_latents(
                clean_latents[:, :, :1],
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            ).squeeze(0)
            ref_strength_mask_packed = pack_video_latents(
                ref_strength_mask,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            ).squeeze(-1)
            latents = pack_video_latents(
                latents,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
        else:
            latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            latents = pack_video_latents(
                latents,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
        duration_s = num_frames / self.frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        if audio_num_frames < 1:
            raise ValueError(f"Audio latent frame count must be positive, got {audio_num_frames}.")
        num_mel_bins = self.audio_vae.config.mel_bins or 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels_latents_audio = self.audio_vae.config.latent_channels or 8
        audio_shape = (1, num_channels_latents_audio, audio_num_frames, latent_mel_bins)
        audio_latents = torch.randn(audio_shape, generator=generator, device=device, dtype=torch.float32)
        audio_latents = pack_audio_latents(audio_latents)
        logger.debug(f"Packed latents: {tuple(latents.shape)}, packed audio latents: {tuple(audio_latents.shape)}")
        # A single-step schedule collapses to full noise, which makes the terminal
        # stretch divide by zero and leaves every sigma NaN. Match the serving stack
        # by treating one step as the two-step minimum.
        num_inference_steps = max(int(settings.num_inference_steps), 2)
        sigmas = list(np.linspace(1.0, 1 / num_inference_steps, num_inference_steps))
        # The shift parameter mu is taken at the maximum sequence anchor and shared
        # by the video and audio schedulers.
        mu = calculate_shift(
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        audio_scheduler = self.scheduler.clone()
        retrieve_timesteps(audio_scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        self.scheduler.set_begin_index(0)
        audio_scheduler.set_begin_index(0)
        logger.debug(f"Denoising steps: {num_inference_steps}, timesteps: {timesteps[:4].tolist()}...")
        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0],
            latent_num_frames,
            latent_height,
            latent_width,
            latents.device,
            fps=self.frame_rate,
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device
        )
        # The rotary tables depend only on the positions, so they are built once here
        # and reused by every pass of this stage.
        rotary_emb = self.transformer.prepare_rotary_embeddings(
            video_coords, audio_coords, latents.device, audio_latents.device
        )
        # The cache covers one denoising stage: stage 2 runs a larger token grid on a
        # three-step schedule, so it stays uncached.
        cache = ResidualCache(
            total_blocks=len(self.transformer.transformer_blocks),
            decision_pass="cfg",
        )
        latents, audio_latents = self._denoise_stage(
            settings,
            latents,
            audio_latents,
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            connector_attention_mask,
            video_coords,
            audio_coords,
            latent_num_frames,
            latent_height,
            latent_width,
            audio_num_frames,
            self.scheduler,
            audio_scheduler,
            timesteps,
            ref_strength_mask_packed,
            ref_first_frame_tokens,
            rotary_emb=rotary_emb,
            cache=cache,
        )
        # Clearing here releases the cached tensors before the refine pass allocates its
        # larger token grid.
        cache.reset("end of stage 1")
        if settings.higher_resolution:
            if self.upsampler is None:
                raise ValueError("higher_resolution is enabled but no upsampler is loaded.")
            if self.lora_entries is None:
                raise ValueError("higher_resolution is enabled but no stage-2 LoRA weights are loaded.")
            if self.connector_lora_entries is None:
                raise ValueError("higher_resolution is enabled but no stage-2 connector LoRA weights are loaded.")
            # Stage 2 refines from a short three-value sigma schedule, and the distilled
            # model runs it without any guidance stage.
            stage2_settings = stage2_generation_settings(settings)
            stage2_sigmas = STAGE_2_SIGMA_FALLBACK
            logger.debug(f"Stage 2 sigmas: {stage2_sigmas}")
            stage1_video = denormalize_video_latents(
                unpack_video_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                ),
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            )
            logger.debug(f"Upsampling stage-1 latents {tuple(stage1_video.shape)}")
            upsampler_dtype = next(self.upsampler.parameters()).dtype
            upsampled_video = self.upsampler(stage1_video.to(upsampler_dtype))

            # The stage-2 upsampler only doubles spatial resolution, not time
            # (temporal_upsample=False in its config). So height and width double here
            # while the frame count stays exactly as stage 1 produced it.

            latent_height = latent_height * 2
            latent_width = latent_width * 2
            stage2_video = normalize_video_latents(
                upsampled_video,
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            )
            latents = pack_video_latents(
                stage2_video,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            )
            stage2_ref_strength_mask_packed = None
            stage2_ref_first_frame_tokens = None
            # Image conditioning pins frame 0 in both stages. Stage 2 runs at twice
            # the pixel resolution, so the source image is encoded again at that
            # resolution instead of reusing the coarser stage-1 frame.
            if ref_strength_mask_packed is not None:
                stage2_ref_strength_mask = torch.zeros(
                    (1, 1, latent_num_frames, latent_height, latent_width),
                    device=device,
                    dtype=torch.float32,
                )
                stage2_ref_strength_mask[:, :, 0] = settings.image_strength
                stage2_ref_strength_mask_packed = pack_video_latents(
                    stage2_ref_strength_mask,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                ).squeeze(-1)
                stage2_reference_image_latents = self.encode_reference_image(
                    settings.image_path,
                    settings.height * 2,
                    settings.width * 2,
                    latent_num_frames,
                    device,
                )
                stage2_ref_first_frame_tokens = pack_video_latents(
                    stage2_reference_image_latents[:, :, :1],
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                ).squeeze(0)
                logger.debug(
                    f"Stage-2 image conditioning: mask {tuple(stage2_ref_strength_mask_packed.shape)}, "
                    f"reference tokens {tuple(stage2_ref_first_frame_tokens.shape)} encoded at "
                    f"{settings.height * 2}x{settings.width * 2}"
                )
            noise_scale = stage2_sigmas[0]
            latents = self.renoise_packed_latents(latents, noise_scale, generator)
            # Renoise would otherwise erase the pinned first frame before stage 2 starts.
            if stage2_ref_strength_mask_packed is not None:
                tokens_per_frame = latents.shape[1] // latent_num_frames
                latents[:, :tokens_per_frame] = (
                    settings.image_strength * stage2_ref_first_frame_tokens
                    + (1 - settings.image_strength) * latents[:, :tokens_per_frame]
                )
            # Refinement is a super-resolution pass over the picture and produces no audio
            # of its own, so `audio_latents` keeps the stage-1 result. The audio stream
            # still has to run, because the video branch cross-attends to it: the stage-2
            # stream lives in its own variable and is thrown away with the latents the
            # refine pass returns.
            logger.debug(f"Stage 2 keeps the stage-1 audio latents {tuple(audio_latents.shape)}")
            stage2_audio_latents = self.renoise_packed_latents(audio_latents, noise_scale, generator)
            stage2_scheduler = self.create_stage2_scheduler()
            stage2_audio_scheduler = self.create_stage2_scheduler()
            stage2_timesteps, _ = retrieve_timesteps(stage2_scheduler, len(stage2_sigmas), device, sigmas=stage2_sigmas)
            retrieve_timesteps(stage2_audio_scheduler, len(stage2_sigmas), device, sigmas=stage2_sigmas)
            stage2_scheduler.set_begin_index(0)
            stage2_audio_scheduler.set_begin_index(0)
            logger.debug(f"Stage 2 steps: {len(stage2_timesteps)}")
            video_coords = self.transformer.rope.prepare_video_coords(
                latents.shape[0],
                latent_num_frames,
                latent_height,
                latent_width,
                latents.device,
                fps=self.frame_rate,
            )
            audio_coords = self.transformer.audio_rope.prepare_audio_coords(
                stage2_audio_latents.shape[0], audio_num_frames, stage2_audio_latents.device
            )
            # Stage 2 runs at twice the spatial resolution, so it needs its own tables;
            # rebuilding here releases the stage-1 ones.
            rotary_emb = self.transformer.prepare_rotary_embeddings(
                video_coords, audio_coords, latents.device, stage2_audio_latents.device
            )
            connector_lora_originals = apply_lora(
                self.connectors, self.connector_lora_entries, scale=settings.stage2_lora_strength
            )
            try:
                # The refine pass has no negative branch, so only the positive prompt
                # goes through the LoRA-adapted connectors.
                (
                    stage2_connector_prompt_embeds,
                    stage2_connector_audio_prompt_embeds,
                    stage2_connector_attention_mask,
                ) = self.connectors(raw_prompt_embeds, raw_prompt_attention_mask, padding_side="left")
                lora_originals = apply_lora(
                    self.transformer, self.lora_entries, scale=settings.stage2_lora_strength
                )
                try:
                    latents, _ = self._denoise_stage(
                        stage2_settings,
                        latents,
                        stage2_audio_latents,
                        stage2_connector_prompt_embeds,
                        stage2_connector_audio_prompt_embeds,
                        stage2_connector_attention_mask,
                        video_coords,
                        audio_coords,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                        audio_num_frames,
                        stage2_scheduler,
                        stage2_audio_scheduler,
                        stage2_timesteps,
                        stage2_ref_strength_mask_packed,
                        stage2_ref_first_frame_tokens,
                        rotary_emb=rotary_emb,
                        cache=None,
                    )
                finally:
                    remove_lora(lora_originals)
            finally:
                remove_lora(connector_lora_originals)
        latents = unpack_video_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        audio_latents = denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = unpack_audio_latents(audio_latents, latent_mel_bins)
        latents = denormalize_video_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )

        # VAE decoding needs a large contiguous block of VRAM, so park the
        # transformer, text encoder, and connectors on CPU first. They are moved
        # back to the original device at the end, so the pipeline stays reusable
        # across calls.

        offloaded = []
        for module in [self.text_encoder, self.transformer, self.connectors, self.upsampler, self.duration_head]:
            if module is not None:
                module.to("cpu")
                offloaded.append(module)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug(f"Decoding video latents {tuple(latents.shape)}")
            vae_dtype = next(self.vae.parameters()).dtype
            video = self.vae.decode(latents.to(vae_dtype))
            audio_latents = audio_latents.to(next(self.audio_vae.parameters()).dtype)
            generated_mel = self.audio_vae.decode(audio_latents)
            logger.debug(f"Decoding audio mel {tuple(generated_mel.shape)}")
            # The vocoder's transposed convolutions can select a nondeterministic
            # cuDNN algorithm, so the same mel input would decode to slightly
            # different waveforms. Scope determinism to this call.
            previous_cudnn_deterministic = torch.backends.cudnn.deterministic
            if self.device.type == "cuda":
                torch.backends.cudnn.deterministic = True
            try:
                with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                    audio = self.vocoder(generated_mel.float())
            finally:
                torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        finally:
            for module in offloaded:
                module.to(device)
        return video, audio
