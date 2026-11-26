import dataclasses
from typing import Optional
import torch

from src.cache.residual_cache import ResidualCache, cache_for_pass
from src.config import GenerationSettings
from src.model.pipeline import LatentGeometry, stage_two_schedule
from src.model.scheduler import FlowMatchEulerScheduler
from src.parallelism.partition import step_passes
from src.profiling import begin_step


@dataclasses.dataclass
class LaneState:
    """One request's state on one GPU: its settings, its latents and the passes it still owes."""
    request_id: int
    lane: int
    settings: GenerationSettings
    stage: int
    stage_settings: GenerationSettings
    passes: tuple
    queue: list
    step_reuse: bool
    latents: Optional[torch.Tensor]
    audio_latents: Optional[torch.Tensor]
    timesteps: torch.Tensor
    step_index: int
    cache: Optional[ResidualCache]
    rotary_emb: object
    ref_mask: Optional[torch.Tensor]
    video_grid: object
    audio_grid: object


class Half:
    """One GPU's side of the pipeline: its share of the model, plus the lane helpers both GPUs use"""
    def __init__(
        self,
        model_config,
        engine_config,
        transformer,
        device,
        dtype,
        split,
        lora_installed,
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.transformer = transformer
        self.device = device
        self.dtype = dtype
        self.split = split
        self.lora_installed = lora_installed
        self.geometry = LatentGeometry.from_configs(model_config, engine_config)
        self.frame_rate = engine_config.frame_rate
        self._stage2_scheduler_config, self._stage2_sigmas, self._stage2_timesteps = (
            stage_two_schedule(model_config.scheduler, engine_config.stage2_sigmas, device)
        )

    def stage_two_schedulers(self):
        return (
            FlowMatchEulerScheduler.from_schedule(
                self._stage2_scheduler_config, self._stage2_sigmas, self._stage2_timesteps
            ),
            FlowMatchEulerScheduler.from_schedule(
                self._stage2_scheduler_config, self._stage2_sigmas, self._stage2_timesteps
            ),
            self._stage2_timesteps,
        )

    def _check_crop_size(self, settings):
        if settings.height % 32 != 0 or settings.width % 32 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 32 but are {settings.height} and {settings.width}."
            )

    def _check_refinement_support(self, settings):
        if settings.higher_resolution and not self.lora_installed:
            raise ValueError("higher_resolution is enabled but no stage-2 LoRA weights are installed.")

    def _check_patch_size(self, settings):
        if settings.image_path and (
            self.geometry.transformer_patch_size != 1 or self.geometry.transformer_patch_size_t != 1
        ):
            raise ValueError(
                "Image conditioning pins frame 0 in packed token space, which needs unpatched latents, "
                f"but the transformer patch sizes are ({self.geometry.transformer_patch_size}, "
                f"{self.geometry.transformer_patch_size_t})."
            )

    def _adapter_strength(self, lane):
        return 0.0 if lane.stage == 0 else lane.stage_settings.stage2_lora_strength

    def _backend_stage(self, lane):
        return "stage1" if lane.stage == 0 else "stage2"

    def _rotary(self, video_grid, audio_grid):
        """Build the rotary tables a token grid needs; they depend on position only, so once per stage is enough."""
        video_coords = self.transformer.rope.prepare_video_coords(
            1,
            video_grid.latent_frames,
            video_grid.latent_height,
            video_grid.latent_width,
            self.device,
            fps=self.frame_rate,
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            1, audio_grid.frames, self.device
        )
        return self.transformer.prepare_rotary_embeddings(
            video_coords, audio_coords, self.device, self.device
        )

    def next_plan(self, lane):
        """The next pass this GPU has to run, or None when it has no more work in the step."""
        return lane.queue[0] if lane.queue else None

    def _stage_passes(self, settings, cache):
        """The passes one step runs, with caching turned off for a stage that has no cache.

        A pass plan describes the block stack; whether a stage carries a cache is a fact
        about the lane. Resolving the two here lets every pass trust its own `cached` flag
        instead of checking the lane's cache again. The refinement stage reuses the coarse
        stage's plans but holds no cache, so its plans have to say so.
        """
        return tuple(
            dataclasses.replace(plan, cached=plan.cached and cache is not None)
            for plan in step_passes(settings, self.split)
        )

    def _open_step(self, lane):
        """Start a step: tell the cache which step this is, clear the reuse decision, mark the step."""
        if lane.cache is not None:
            lane.cache.begin_step(lane.step_index)
        lane.step_reuse = False
        begin_step(lane.lane, lane.request_id, lane.stage, lane.step_index)

    def _pop_plan(self, lane, plan):
        """Remove the pass the queue is waiting for, refusing any other pass."""
        if not lane.queue or lane.queue[0] is not plan:
            raise RuntimeError(f"Request {lane.request_id} finished pass {plan.name!r} out of order.")
        lane.queue.pop(0)

    def _adopt_latents(self, lane, video, audio, what):
        """Take the latents a stage starts from, checking they match that stage's grid."""
        expected_audio = (1, lane.audio_grid.tokens, lane.audio_grid.width)
        expected_video = (1, lane.video_grid.tokens, self.geometry.latent_channels)
        if tuple(video.shape) != expected_video:
            raise RuntimeError(
                f"Request {lane.request_id} received {what} latents of shape "
                f"{tuple(video.shape)} but its grid needs {expected_video}."
            )
        if tuple(audio.shape) != expected_audio:
            raise RuntimeError(
                f"Request {lane.request_id} received {what} audio latents of shape "
                f"{tuple(audio.shape)} but its grid needs {expected_audio}."
            )
        lane.latents = video.to(device=self.device, dtype=torch.float32)
        lane.audio_latents = audio.to(device=self.device, dtype=torch.float32)

    def _blocks(self, lane, plan, hidden_states, audio_hidden_states, conditioning, reuse, *,
                lo, hi, capture_block=None, captured_states=None):
        """Run the blocks this GPU owns for one pass; callers differ only in range and capture point."""
        return self.transformer.run_blocks(
            hidden_states,
            audio_hidden_states,
            lo=lo,
            hi=hi,
            conditioning=conditioning,
            rotary_emb=lane.rotary_emb,
            pass_name=plan.name,
            step_index=lane.step_index,
            stage=lane.stage,
            cache=cache_for_pass(lane.cache, plan.name) if plan.cached else None,
            reuse=reuse,
            capture_block=capture_block,
            captured_states=captured_states,
            perturb_blocks=lane.stage_settings.stg_blocks if plan.name == "stg" else (),
            isolate_modalities=plan.isolates_modalities,
        )
