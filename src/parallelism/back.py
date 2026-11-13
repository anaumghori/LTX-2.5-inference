import dataclasses
from pathlib import Path
import threading
from typing import Optional
import imageio.v2 as imageio
import torch
import torch.nn.functional as F

from src.cache.residual_cache import ResidualCache
from src.model.attention_backend import stage_backends
from src.model.common import DiagonalGaussian
from src.model.logging_utils import get_logger
from src.model.lora import lora_strength
from src.model.pipeline import (
    STAGE_2_SIGMA_FALLBACK,
    denormalize_audio_latents,
    denormalize_video_latents,
    normalize_video_latents,
    pack_video_latents,
    renoise_packed_latents,
    stage_one_schedulers,
    stage_two_schedulers,
    unpack_audio_latents,
    unpack_video_latents,
)
from src.model.scheduler import rescale_noise_cfg
from src.model.transformer import (
    STAGE_CONDITIONING_TENSORS,
    STEP_CONDITIONING_TENSORS,
    PassConditioning,
)
from src.parallelism.half import Half, LaneState
from src.parallelism.partition import (
    stage2_settings,
    stg_pass_after_decision,
    uses_text_guidance,
)

logger = get_logger(__name__)

_conv_lock = threading.Lock()

@dataclasses.dataclass
class BackLane(LaneState):
    """The second GPU's own state, on top of what every lane carries."""
    coarse_audio_latents: Optional[torch.Tensor]
    scheduler: object
    audio_scheduler: object
    captured: dict
    predictions: dict
    cfg_conditioning: Optional[PassConditioning]
    ref_tokens: Optional[torch.Tensor]
    # The stage-constant conditioning of each (stage, batch) pair already sent to this GPU.
    conditioning: dict
    text_guidance: bool = False


class BackHalf(Half):
    """The pipeline components that run on the second GPU: the trailing blocks, the samplers, the decoders and the file output."""
    def __init__(
        self,
        model_config,
        engine_config,
        transformer,
        vae,
        audio_vae,
        vocoder,
        upsampler,
        device,
        dtype,
        split,
        lora_installed,
    ):
        super().__init__(
            model_config, engine_config, transformer, device, dtype, split, lora_installed
        )
        self.vae = vae
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        self.upsampler = upsampler

    def _validate(self, settings):
        self._check_crop_size(settings)
        if settings.higher_resolution and self.upsampler is None:
            raise ValueError("higher_resolution is enabled but no upsampler is loaded.")
        self._check_refinement_support(settings)
        self._check_patch_size(settings)

    @torch.no_grad()
    def encode_reference_image(self, image_path, height, width):
        """Cover-crop a reference image to the target size and encode its single frame."""
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Reference image not found: {image_path}")
        image = imageio.imread(image_path)
        if image.ndim == 2:
            image = image[:, :, None].repeat(3, axis=2)
        image = image[:, :, :3]
        image = (
            torch.from_numpy(image)
            .to(device=self.device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        image = image / 255.0 * 2.0 - 1.0
        cover_scale = max(height / image.shape[2], width / image.shape[3])
        resized_h = max(1, round(image.shape[2] * cover_scale))
        resized_w = max(1, round(image.shape[3] * cover_scale))
        image = F.interpolate(image, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        top = (resized_h - height) // 2
        left = (resized_w - width) // 2
        image = image[:, :, top:top + height, left:left + width]
        vae_dtype = next(self.vae.parameters()).dtype
        with _conv_lock:
            latents = self.vae.encode(image.to(vae_dtype).unsqueeze(2))
        latents = DiagonalGaussian(latents.float()).mode()
        return normalize_video_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )

    @torch.no_grad()
    def prepare(self, request_id, lane_index, settings, num_frames):
        """Build a request's coarse-stage state and encode its reference frame, if any."""
        self._validate(settings)
        video_grid = self.geometry.video_grid_for(num_frames, settings.height, settings.width)
        audio_grid = self.geometry.audio_grid(num_frames)
        patch_size = self.geometry.transformer_patch_size
        patch_size_t = self.geometry.transformer_patch_size_t
        ref_tokens = None
        ref_mask = None
        if settings.image_path:
            clean = self.encode_reference_image(settings.image_path, settings.height, settings.width)
            ref_tokens = pack_video_latents(clean[:, :, :1], patch_size, patch_size_t).squeeze(0)
            ref_mask = torch.zeros((1, video_grid.tokens), device=self.device, dtype=torch.float32)
            ref_mask[:, : video_grid.tokens_per_frame] = settings.image_strength
        lane = BackLane(
            request_id=request_id,
            lane=lane_index,
            settings=settings,
            stage=0,
            stage_settings=settings,
            passes=(),
            queue=[],
            step_reuse=False,
            latents=None,
            audio_latents=None,
            coarse_audio_latents=None,
            timesteps=torch.empty(0),
            step_index=0,
            scheduler=None,
            audio_scheduler=None,
            cache=None,
            captured={},
            predictions={},
            cfg_conditioning=None,
            rotary_emb=None,
            ref_mask=ref_mask,
            conditioning={},
            ref_tokens=ref_tokens,
            video_grid=video_grid,
            audio_grid=audio_grid,
        )
        self.start_stage(lane, 0)
        return lane, ref_tokens

    def start_stage(self, lane, stage):
        if stage == 0:
            lane.stage_settings = lane.settings
            lane.scheduler, lane.audio_scheduler, lane.timesteps = stage_one_schedulers(
                self.model_config.scheduler, lane.settings, self.device
            )
            lane.cache = None
            if self.split.split_index < self.split.cache_end:
                # The first GPU decides whether the cache is reused; this one only replays
                # its own part of the recorded residual, so it holds no decision logic.
                lane.cache = ResidualCache(
                    total_blocks=self.split.block_count,
                    replay_start=self.split.split_index,
                    replay_end=self.split.cache_end,
                    decision_pass=None,
                )
        else:
            lane.stage_settings = stage2_settings(lane.settings)
            lane.scheduler, lane.audio_scheduler, lane.timesteps = stage_two_schedulers(
                self.model_config.scheduler, self.device
            )
            lane.cache = None
            lane.video_grid = lane.video_grid.refined()
        lane.stage = stage
        lane.text_guidance = uses_text_guidance(lane.stage_settings)
        lane.passes = self._stage_passes(lane.stage_settings, lane.cache)
        lane.queue = []
        lane.step_index = 0
        lane.step_reuse = False
        lane.predictions = {}
        lane.captured = {}
        lane.cfg_conditioning = None
        lane.conditioning = {}
        lane.rotary_emb = self._rotary(lane.video_grid, lane.audio_grid)
        logger.debug(
            f"Request {lane.request_id} stage {stage + 1}: {len(lane.timesteps)} steps, "
            f"{lane.video_grid.tokens} video tokens, {lane.audio_grid.tokens} audio tokens"
        )

    def adopt_state(self, lane, video, audio):
        """Take the latents a stage starts from, as sent by the first GPU."""
        self._adopt_latents(lane, video, audio, f"stage {lane.stage + 1}")
        self.begin_step(lane)

    def begin_step(self, lane):
        self._open_step(lane)
        lane.queue = [plan for plan in lane.passes if plan.name != "stg"]
        lane.predictions = {}
        lane.captured = {}

    def _run_blocks(self, lane, plan, hidden_states, audio_hidden_states, conditioning, reuse):
        with lora_strength(self._adapter_strength(lane)), stage_backends(self._backend_stage(lane)):
            hidden_states, audio_hidden_states, used = self._blocks(
                lane,
                plan,
                hidden_states,
                audio_hidden_states,
                conditioning,
                reuse,
                lo=max(plan.start_block, self.split.split_index),
                hi=self.split.block_count,
                capture_block=self.split.stg_start if plan.captures else None,
                captured_states=lane.captured,
            )
            video, audio = self.transformer.finalize(hidden_states, audio_hidden_states, conditioning)
        return used, video, audio

    def accept_pass(self, lane, plan, header, tensors):
        """Run this GPU's blocks for a pass whose input arrived from the first GPU."""
        if header.stage != lane.stage:
            raise RuntimeError(
                f"Request {lane.request_id} received a stage {header.stage + 1} pass while in stage "
                f"{lane.stage + 1}."
            )
        if header.step != lane.step_index:
            raise RuntimeError(
                f"Request {lane.request_id} received a pass for step {header.step} while in step "
                f"{lane.step_index}."
            )
        if header.pass_code != plan.code or header.batch != plan.batch:
            raise RuntimeError(
                f"Request {lane.request_id} expected pass {plan.name} with batch {plan.batch} but received "
                f"pass code {header.pass_code} with batch {header.batch}."
            )
        if header.start_block != 0:
            raise RuntimeError(
                f"Request {lane.request_id} received a {plan.name} pass that starts at block "
                f"{header.start_block}; only passes built from the top of the stack cross the boundary."
            )
        hidden_states, audio_hidden_states = tensors[0], tensors[1]
        if (
            header.video_tokens != hidden_states.shape[1]
            or header.video_width != hidden_states.shape[2]
            or header.audio_tokens != audio_hidden_states.shape[1]
            or header.audio_width != audio_hidden_states.shape[2]
        ):
            raise RuntimeError(
                f"Request {lane.request_id} announced {header.video_tokens} video tokens of width "
                f"{header.video_width} and {header.audio_tokens} audio tokens of width {header.audio_width}, "
                f"but sent {tuple(hidden_states.shape)} and {tuple(audio_hidden_states.shape)}."
            )
        if not plan.cached and header.reuse:
            raise RuntimeError(
                f"Request {lane.request_id} received a {plan.name} pass that claims to replay the cache, "
                f"but that pass runs the whole stack from block {plan.start_block} and reuses nothing."
            )
        if plan.cached and header.reuse:
            recorded = lane.cache.residual_step(plan.name)
            if recorded != header.cache_epoch:
                raise RuntimeError(
                    f"Request {lane.request_id} pass {plan.name} was told to replay the residual recorded at "
                    f"step {header.cache_epoch}, but this half recorded it at step {recorded}; the two halves "
                    "disagree about which step they are in."
                )
        conditioning = self._conditioning(lane, plan, header, tensors[2:])
        reuse, video, audio = self._run_blocks(
            lane, plan, hidden_states, audio_hidden_states, conditioning, header.reuse
        )
        lane.predictions[plan.name] = (video, audio)
        if plan.cached:
            lane.step_reuse = reuse
        if plan.name == "cfg":
            lane.cfg_conditioning = conditioning
        return reuse

    def _conditioning(self, lane, plan, header, sent):
        """Rebuild a pass's conditioning, storing the stage-constant part once per (stage, batch)."""
        key = (header.stage, plan.batch)
        expected = STEP_CONDITIONING_TENSORS + (
            STAGE_CONDITIONING_TENSORS if header.carries_conditioning else 0
        )
        if len(sent) != expected:
            raise RuntimeError(
                f"Request {lane.request_id} pass {plan.name} carried {len(sent)} conditioning tensors, "
                f"expected {expected} with carries_conditioning={header.carries_conditioning}."
            )
        if header.carries_conditioning:
            lane.conditioning[key] = sent[STEP_CONDITIONING_TENSORS:]
        constant = lane.conditioning.get(key)
        if constant is None:
            raise RuntimeError(
                f"Request {lane.request_id} pass {plan.name} arrived without the stage conditioning for "
                f"stage {header.stage + 1} at batch {plan.batch}, and this half was never sent one."
            )
        if constant[0].shape[0] != plan.batch:
            raise RuntimeError(
                f"Request {lane.request_id} pass {plan.name} runs at batch {plan.batch} but the stage "
                f"conditioning this half holds describes {constant[0].shape[0]} sample(s)."
            )
        return PassConditioning.from_tensors(sent[:STEP_CONDITIONING_TENSORS], constant)

    @torch.no_grad()
    def run_local(self, lane, plan):
        """Run a mid-stack (STG) pass on this GPU, starting from the state it captured earlier in the step."""
        captured = lane.captured.get("state")
        if captured is None:
            raise RuntimeError(
                f"Request {lane.request_id} reached its {plan.name} pass without a captured state at block "
                f"{self.split.stg_start}; the guidance pass never reached that block."
            )
        conditioning = lane.cfg_conditioning.positive_branch()
        hidden_states, audio_hidden_states = captured
        _, video, audio = self._run_blocks(
            lane, plan, hidden_states, audio_hidden_states, conditioning, False
        )
        lane.predictions[plan.name] = (video, audio)

    def finish_pass(self, lane, plan):
        self._pop_plan(lane, plan)
        if plan.name == "cfg":
            follow = stg_pass_after_decision(lane.passes, self.split, lane.step_reuse)
            if follow is not None:
                lane.queue.insert(0, follow)

    def _x0(self, sample, velocity, step_index, scheduler):
        return sample - velocity * scheduler.sigmas[step_index]

    def finish_step(self, lane):
        """Combine a step's pass predictions into one sampler step and return the new latents."""
        settings = lane.stage_settings
        index = lane.step_index
        timestep = lane.timesteps[index]
        scheduler = lane.scheduler
        audio_scheduler = lane.audio_scheduler
        video_velocity, audio_velocity = lane.predictions["cfg"]
        video_velocity = video_velocity.float()
        audio_velocity = audio_velocity.float()
        text_guidance = lane.text_guidance
        if text_guidance:
            video_velocity_negative, video_velocity_positive = video_velocity.chunk(2)
            audio_velocity_negative, audio_velocity_positive = audio_velocity.chunk(2)
        else:
            video_velocity_positive = video_velocity
            audio_velocity_positive = audio_velocity
        video_x0 = self._x0(lane.latents, video_velocity_positive, index, scheduler)
        audio_x0 = self._x0(lane.audio_latents, audio_velocity_positive, index, audio_scheduler)
        if text_guidance:
            video_x0_negative = self._x0(lane.latents, video_velocity_negative, index, scheduler)
            audio_x0_negative = self._x0(lane.audio_latents, audio_velocity_negative, index, audio_scheduler)
            video_cfg_delta = (settings.guidance_scale - 1) * (video_x0 - video_x0_negative)
            audio_cfg_delta = (settings.audio_guidance_scale - 1) * (audio_x0 - audio_x0_negative)
        else:
            video_cfg_delta = audio_cfg_delta = 0
        if "stg" in lane.predictions:
            video_velocity_stg, audio_velocity_stg = lane.predictions["stg"]
            video_x0_stg = self._x0(lane.latents, video_velocity_stg.float(), index, scheduler)
            audio_x0_stg = self._x0(lane.audio_latents, audio_velocity_stg.float(), index, audio_scheduler)
            video_stg_delta = settings.stg_scale * (video_x0 - video_x0_stg)
            audio_stg_delta = settings.audio_stg_scale * (audio_x0 - audio_x0_stg)
        else:
            video_stg_delta = audio_stg_delta = 0
        if "modality" in lane.predictions:
            video_velocity_modality, audio_velocity_modality = lane.predictions["modality"]
            video_x0_modality = self._x0(lane.latents, video_velocity_modality.float(), index, scheduler)
            audio_x0_modality = self._x0(
                lane.audio_latents, audio_velocity_modality.float(), index, audio_scheduler
            )
            video_modality_delta = (settings.modality_scale - 1) * (video_x0 - video_x0_modality)
            audio_modality_delta = (settings.audio_modality_scale - 1) * (audio_x0 - audio_x0_modality)
        else:
            video_modality_delta = audio_modality_delta = 0
        video_guided_x0 = video_x0 + video_cfg_delta + video_stg_delta + video_modality_delta
        audio_guided_x0 = audio_x0 + audio_cfg_delta + audio_stg_delta + audio_modality_delta
        video_guided_x0 = rescale_noise_cfg(
            video_guided_x0, video_x0, guidance_rescale=settings.guidance_rescale
        )
        audio_guided_x0 = rescale_noise_cfg(
            audio_guided_x0, audio_x0, guidance_rescale=settings.audio_guidance_rescale
        )
        video_guided_velocity = (lane.latents - video_guided_x0) / scheduler.sigmas[index]
        audio_guided_velocity = (lane.audio_latents - audio_guided_x0) / audio_scheduler.sigmas[index]
        lane.latents = scheduler.step(video_guided_velocity, timestep, lane.latents)
        lane.audio_latents = audio_scheduler.step(audio_guided_velocity, timestep, lane.audio_latents)
        if lane.ref_mask is not None and lane.ref_tokens is not None:
            # The reference frame is the leading token block, so the blend runs in packed
            # space: strength 1.0 pins it exactly and lower values let it drift.
            tokens_per_frame = lane.video_grid.tokens_per_frame
            lane.latents[:, :tokens_per_frame] = (
                settings.image_strength * lane.ref_tokens
                + (1 - settings.image_strength) * lane.latents[:, :tokens_per_frame]
            )
        lane.step_index += 1
        lane.queue = []
        lane.predictions = {}
        lane.captured = {}
        return lane.latents, lane.audio_latents

    def refine_noise(self, lane):
        """The refinement stage's video and audio noise, derived here instead of sent over"""
        generator = torch.Generator(device=self.device).manual_seed(lane.settings.seed)
        coarse = lane.video_grid
        torch.randn(
            (
                1,
                self.geometry.latent_channels,
                coarse.latent_frames,
                coarse.latent_height,
                coarse.latent_width,
            ),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        torch.randn(
            (1, self.geometry.audio_latent_channels, lane.audio_grid.frames, lane.audio_grid.mel_bins),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        refined = coarse.refined()
        return (
            torch.randn(
                (1, refined.tokens, self.geometry.latent_channels),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            ),
            torch.randn(
                (1, lane.audio_grid.tokens, lane.audio_grid.width),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            ),
        )

    @torch.no_grad()
    def begin_refinement(self, lane):
        """Reshape a finished coarse stage into the refinement stage's larger grid.

        The upsampler lives on this GPU, so the refined latents are built here and sent to
        the first GPU, which uses them to build the refinement pass's model input. The
        coarse audio latents are kept because refinement produces no audio of its own.
        """
        video_noise, audio_noise = self.refine_noise(lane)
        patch_size = self.geometry.transformer_patch_size
        patch_size_t = self.geometry.transformer_patch_size_t
        coarse_grid = lane.video_grid
        stage1_video = denormalize_video_latents(
            unpack_video_latents(
                lane.latents,
                coarse_grid.latent_frames,
                coarse_grid.latent_height,
                coarse_grid.latent_width,
                patch_size,
                patch_size_t,
            ),
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )
        upsampler_dtype = next(self.upsampler.parameters()).dtype
        with _conv_lock:
            upsampled = self.upsampler(stage1_video.to(upsampler_dtype))
        refined = coarse_grid.refined()
        stage2_video = normalize_video_latents(
            upsampled,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )
        latents = pack_video_latents(stage2_video, patch_size, patch_size_t)
        del stage1_video, upsampled, stage2_video
        lane.coarse_audio_latents = lane.audio_latents
        if (1, refined.tokens, self.geometry.latent_channels) != tuple(latents.shape):
            raise RuntimeError(
                f"Request {lane.request_id} upsampled to {tuple(latents.shape)} but its refined grid needs "
                f"{(1, refined.tokens, self.geometry.latent_channels)}."
            )
        ref_mask = None
        ref_tokens = None
        if lane.settings.image_path:
            # The refined grid runs at twice the pixel resolution, so the source image is
            # encoded again at that resolution rather than reusing the coarse frame.
            clean = self.encode_reference_image(
                lane.settings.image_path, lane.settings.height * 2, lane.settings.width * 2
            )
            ref_tokens = pack_video_latents(clean[:, :, :1], patch_size, patch_size_t).squeeze(0)
            ref_mask = torch.zeros((1, refined.tokens), device=self.device, dtype=torch.float32)
            ref_mask[:, : refined.tokens_per_frame] = lane.settings.image_strength
        noise_scale = STAGE_2_SIGMA_FALLBACK[0]
        latents = renoise_packed_latents(latents, noise_scale, video_noise)
        if ref_mask is not None and ref_tokens is not None:
            # Renoising would erase the pinned frame, so the pin is applied after it.
            strength = lane.settings.image_strength
            latents[:, : refined.tokens_per_frame] = (
                strength * ref_tokens + (1 - strength) * latents[:, : refined.tokens_per_frame]
            )
        audio_latents = renoise_packed_latents(lane.audio_latents, noise_scale, audio_noise)
        lane.latents = latents
        lane.audio_latents = audio_latents
        lane.ref_mask = ref_mask
        lane.ref_tokens = ref_tokens
        self.start_stage(lane, 1)
        self.begin_step(lane)
        logger.info(
            f"Request {lane.request_id} refinement stage prepared: {refined.tokens} video tokens at "
            f"{refined.latent_height}x{refined.latent_width}, {len(lane.timesteps)} steps"
        )
        return lane.latents, lane.audio_latents, (video_noise, audio_noise)

    @torch.no_grad()
    def decode(self, lane):
        """Decode the finished latents into a video and a waveform."""
        final = lane.video_grid
        latents = unpack_video_latents(
            lane.latents,
            final.latent_frames,
            final.latent_height,
            final.latent_width,
            self.geometry.transformer_patch_size,
            self.geometry.transformer_patch_size_t,
        )
        # Refinement produces no audio of its own, so a refined request decodes the audio
        # the coarse stage finished with.
        final_audio = lane.coarse_audio_latents if lane.stage == 1 else lane.audio_latents
        audio_latents = denormalize_audio_latents(
            final_audio, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = unpack_audio_latents(audio_latents, lane.audio_grid.mel_bins)
        latents = denormalize_video_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )
        generator = torch.Generator(device=self.device).manual_seed(lane.settings.seed)
        vae_dtype = next(self.vae.parameters()).dtype
        with _conv_lock:
            video = self.vae.decode(latents.to(vae_dtype), generator=generator)
        with _conv_lock:
            generated_mel = self.audio_vae.decode(
                audio_latents.to(next(self.audio_vae.parameters()).dtype)
            )
        with _conv_lock:
            # The vocoder's transposed convolutions can select a nondeterministic cuDNN
            # algorithm, so determinism is scoped to this call.
            previous_cudnn_deterministic = torch.backends.cudnn.deterministic
            if self.device.type == "cuda":
                torch.backends.cudnn.deterministic = True
            try:
                audio = self.vocoder(generated_mel.to(next(self.vocoder.parameters()).dtype))
            finally:
                torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        return video, audio
