import dataclasses
from typing import Optional
import torch

from src.cache.residual_cache import ResidualCache
from src.model.attention_backend import stage_backends
from src.model.logging_utils import get_logger
from src.model.lora import lora_strength
from src.model.pipeline import (
    pack_audio_latents,
    pack_video_latents,
    select_positive_branch,
    stage_one_timesteps,
    stage_two_timesteps,
)
from src.parallelism.half import Half, LaneState
from src.parallelism.partition import (
    stage2_settings,
    stg_pass_after_decision,
    uses_text_guidance,
)

logger = get_logger(__name__)


@dataclasses.dataclass
class FrontLane(LaneState):
    prompt_embeds: torch.Tensor
    audio_prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor
    raw_prompt_embeds: Optional[torch.Tensor]
    raw_prompt_attention_mask: Optional[torch.Tensor]
    keyframes_mask: torch.Tensor
    conditioning_sent: set


class FrontHalf(Half):
    """The pipeline components that run on the first GPU: text encoding, prompt conditioning and the leading blocks."""
    def __init__(
        self,
        model_config,
        engine_config,
        transformer,
        tokenizer,
        text_encoder,
        connectors,
        duration_head,
        device,
        dtype,
        split,
        lora_installed,
    ):
        super().__init__(
            model_config, engine_config, transformer, device, dtype, split, lora_installed
        )
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.connectors = connectors
        self.duration_head = duration_head

    def _validate(self, settings):
        self._check_crop_size(settings)
        if not 0.0 <= settings.image_strength <= 1.0:
            raise ValueError(f"`image_strength` must be in [0, 1], got {settings.image_strength}.")
        if not isinstance(settings.prompt, str) or not settings.prompt.strip():
            raise ValueError("`prompt` must be a single non-empty string.")
        if not isinstance(settings.negative_prompt, str):
            raise ValueError("`negative_prompt` must be a single string, not a list.")
        self._check_refinement_support(settings)
        self._check_patch_size(settings)

    def encode_text(self, prompt, max_sequence_length):
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
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(encoder_outputs.hidden_states, dim=-1)
        prompt_embeds = hidden_states.flatten(2, 3).to(dtype=self.dtype)
        logger.debug(f"Prompt embeds shape: {tuple(prompt_embeds.shape)}")
        return prompt_embeds, attention_mask

    def _predict_duration(self, video_tokens, audio_tokens, text_guidance):
        if self.duration_head is None:
            raise ValueError(
                "Automatic duration requires a duration_head; load the duration head or provide duration_seconds."
            )
        video_tokens = select_positive_branch(video_tokens, text_guidance)
        audio_tokens = select_positive_branch(audio_tokens, text_guidance)
        return self.duration_head(video_tokens[:1], audio_tokens[:1]).item()

    def _keyframes_mask(self, video_grid):
        # The checkpoint marks the first latent frame as a reference frame and the marker
        # is added per packed token.
        mask = torch.zeros((1, video_grid.tokens, 1), device=self.device, dtype=torch.float32)
        mask[:, : video_grid.tokens_per_frame] = 1.0
        return mask

    @torch.no_grad()
    def prepare(self, request_id, lane_index, settings):
        """Validate a request, build its coarse-stage state and draw the noise it starts from."""
        self._validate(settings)
        device = self.device
        generator = torch.Generator(device=device).manual_seed(settings.seed)
        prompt_embeds, attention_mask = self.encode_text(settings.prompt, settings.max_sequence_length)
        raw_prompt_embeds = prompt_embeds
        raw_prompt_attention_mask = attention_mask
        negative_embeds, negative_mask = self.encode_text(
            settings.negative_prompt, settings.max_sequence_length
        )
        text_guidance = uses_text_guidance(settings)
        if text_guidance:
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
            attention_mask = torch.cat([negative_mask, attention_mask], dim=0)
        video_prompt, audio_prompt, connector_mask = self.connectors(
            prompt_embeds, attention_mask, padding_side="left"
        )
        requested = settings.duration_seconds
        if requested is None:
            requested = self._predict_duration(video_prompt, audio_prompt, text_guidance)
            logger.debug(f"Predicted duration {requested:.2f}s")
        try:
            requested = float(requested)
        except (TypeError, ValueError) as error:
            raise ValueError(f"duration must be a real number of seconds, got {requested!r}.") from error
        if requested < self.geometry.min_seconds or requested > self.geometry.max_seconds:
            clamped = min(max(requested, self.geometry.min_seconds), self.geometry.max_seconds)
            logger.warning(
                f"Duration {requested:.2f}s outside "
                f"[{self.geometry.min_seconds:.2f}s, {self.geometry.max_seconds:.2f}s], "
                f"using {clamped:.2f}s"
            )
            requested = clamped
        settings = dataclasses.replace(settings, duration_seconds=requested)
        video_grid = self.geometry.video_grid(settings)
        audio_grid = self.geometry.audio_grid(video_grid.num_frames)
        logger.debug(
            f"Request {request_id}: {video_grid.num_frames} frames -> "
            f"latent ({video_grid.latent_frames}, {video_grid.latent_height}, {video_grid.latent_width}), "
            f"{video_grid.tokens} video tokens, {audio_grid.tokens} audio tokens"
        )
        patch_size = self.geometry.transformer_patch_size
        patch_size_t = self.geometry.transformer_patch_size_t
        shape = (
            1,
            self.geometry.latent_channels,
            video_grid.latent_frames,
            video_grid.latent_height,
            video_grid.latent_width,
        )
        ref_mask = None
        noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        latents = pack_video_latents(noise, patch_size, patch_size_t)
        if settings.image_path:
            ref_mask = torch.zeros((1, video_grid.tokens), device=device, dtype=torch.float32)
            ref_mask[:, : video_grid.tokens_per_frame] = settings.image_strength
        audio_shape = (1, self.geometry.audio_latent_channels, audio_grid.frames, audio_grid.mel_bins)
        audio_latents = pack_audio_latents(
            torch.randn(audio_shape, generator=generator, device=device, dtype=torch.float32)
        )
        cache = None
        if self.split.cache_start > 0:
            cache = ResidualCache(
                total_blocks=self.split.block_count,
                replay_start=self.split.cache_start,
                replay_end=self.split.split_index,
                decision_pass="cfg",
            )
        lane = FrontLane(
            request_id=request_id,
            lane=lane_index,
            settings=settings,
            stage=0,
            stage_settings=settings,
            passes=self._stage_passes(settings, cache),
            queue=[],
            step_reuse=False,
            prompt_embeds=video_prompt,
            audio_prompt_embeds=audio_prompt,
            attention_mask=connector_mask,
            raw_prompt_embeds=raw_prompt_embeds,
            raw_prompt_attention_mask=raw_prompt_attention_mask,
            latents=latents,
            audio_latents=audio_latents,
            timesteps=stage_one_timesteps(self.model_config.scheduler, settings, device),
            step_index=0,
            keyframes_mask=self._keyframes_mask(video_grid),
            rotary_emb=self._rotary(video_grid, audio_grid),
            cache=cache,
            ref_mask=ref_mask,
            conditioning_sent=set(),
            video_grid=video_grid,
            audio_grid=audio_grid,
        )
        return lane

    def adopt_reference_tokens(self, lane, tokens):
        """Blend the reference frame's tokens into the request's leading frame."""
        expected = (lane.video_grid.tokens_per_frame, self.geometry.latent_channels)
        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"Request {lane.request_id} received reference tokens of shape {tuple(tokens.shape)} "
                f"but its grid needs {expected}."
            )
        tokens = tokens.to(device=self.device, dtype=torch.float32)
        strength = lane.settings.image_strength
        lane.latents[:, : expected[0]] = (
            strength * tokens.unsqueeze(0) + (1 - strength) * lane.latents[:, : expected[0]]
        )

    def state_payload(self, lane):
        """The latents the second GPU needs to start or resume a stage."""
        return (lane.latents, lane.audio_latents)

    def begin_step(self, lane):
        self._open_step(lane)
        lane.queue = [plan for plan in lane.passes if plan.needs_front]

    @torch.no_grad()
    def execute_pass(self, lane, plan):
        """Build one pass's model input, run this GPU's blocks and return what to send.

        The return value holds the cache decision, the step whose residual that decision
        replays (-1 when nothing is replayed), the step-varying payload, and the
        stage-constant conditioning the second GPU keeps after the first pass of a pair.
        """
        timestep = lane.timesteps[lane.step_index]
        batch = plan.batch
        # The CFG pass carries both the negative and the positive prompt, so its embeddings
        # describe two samples. Every other pass runs the positive sample alone and takes
        # the positive branch of those same embeddings.
        text_guidance = uses_text_guidance(lane.stage_settings)
        if plan.name == "cfg":
            prompt_embeds = lane.prompt_embeds
            audio_prompt_embeds = lane.audio_prompt_embeds
            attention_mask = lane.attention_mask
        else:
            prompt_embeds = select_positive_branch(lane.prompt_embeds, text_guidance)
            audio_prompt_embeds = select_positive_branch(lane.audio_prompt_embeds, text_guidance)
            attention_mask = select_positive_branch(lane.attention_mask, text_guidance)
        latent_model_input = torch.cat([lane.latents] * batch).to(self.dtype)
        audio_model_input = torch.cat([lane.audio_latents] * batch).to(self.dtype)
        timestep = timestep.expand(batch)
        if lane.ref_mask is not None:
            step_ref_mask = lane.ref_mask.repeat((batch,) + (1,) * (lane.ref_mask.ndim - 1))
            video_timestep = timestep.unsqueeze(-1) * (1 - step_ref_mask)
        else:
            video_timestep = timestep
        with lora_strength(self._adapter_strength(lane)), stage_backends(self._backend_stage(lane)):
            hidden_states, audio_hidden_states, conditioning = self.transformer.run_input(
                latent_model_input,
                audio_model_input,
                timestep=video_timestep,
                audio_timestep=timestep,
                sigma=timestep,
                encoder_hidden_states=prompt_embeds,
                audio_encoder_hidden_states=audio_prompt_embeds,
                encoder_attention_mask=attention_mask,
                audio_encoder_attention_mask=attention_mask,
                video_keyframes_mask=lane.keyframes_mask,
                use_cross_timestep=True,
            )
            reuse = None if plan.name == "cfg" else (lane.step_reuse if plan.cached else False)
            hidden_states, audio_hidden_states, reuse = self._blocks(
                lane,
                plan,
                hidden_states,
                audio_hidden_states,
                conditioning,
                reuse,
                lo=0,
                hi=self.split.split_index,
            )
        payload = (hidden_states, audio_hidden_states) + conditioning.step_tensors(self.device, self.dtype)
        constant = conditioning.stage_tensors(self.device, self.dtype)
        if not plan.cached:
            return False, -1, payload, constant
        reuse = bool(reuse)
        epoch = lane.cache.residual_step(plan.name) if reuse else -1
        return reuse, epoch, payload, constant

    def finish_pass(self, lane, plan, reuse):
        self._pop_plan(lane, plan)
        if plan.cached:
            lane.step_reuse = reuse
        if plan.name == "cfg":
            follow = stg_pass_after_decision(lane.passes, self.split, reuse)
            if follow is not None and follow.needs_front:
                lane.queue.insert(0, follow)

    @torch.no_grad()
    def adopt_step_latents(self, lane, video, audio):
        """Take a step's latents from the second GPU, or report that the stage just ended."""
        self._adopt_latents(lane, video, audio, "step")
        lane.step_index += 1
        if lane.step_index >= len(lane.timesteps):
            if lane.cache is not None:
                lane.cache.reset("end of stage 1")
            return True
        self.begin_step(lane)
        return False

    @torch.no_grad()
    def transition(self, lane):
        """Turn a finished coarse stage into the refinement stage."""
        settings = lane.settings
        stage_settings = stage2_settings(settings)
        refined = lane.video_grid.refined()
        with lora_strength(stage_settings.stage2_lora_strength):
            video_prompt, audio_prompt, connector_mask = self.connectors(
                lane.raw_prompt_embeds,
                lane.raw_prompt_attention_mask,
                padding_side="left",
            )
        lane.raw_prompt_embeds = None
        lane.raw_prompt_attention_mask = None
        lane.stage = 1
        lane.stage_settings = stage_settings
        lane.prompt_embeds = video_prompt
        lane.audio_prompt_embeds = audio_prompt
        lane.attention_mask = connector_mask
        # The refinement stage runs the same plans with no cache at all.
        lane.passes = self._stage_passes(stage_settings, None)
        lane.queue = []
        lane.step_reuse = False
        lane.timesteps = stage_two_timesteps(self.model_config.scheduler, self.device)
        lane.step_index = 0
        lane.cache = None
        lane.conditioning_sent = set()
        lane.latents = None
        lane.audio_latents = None
        lane.video_grid = refined
        lane.keyframes_mask = self._keyframes_mask(refined)
        lane.rotary_emb = self._rotary(refined, lane.audio_grid)
        if settings.image_path:
            lane.ref_mask = torch.zeros((1, refined.tokens), device=self.device, dtype=torch.float32)
            lane.ref_mask[:, : refined.tokens_per_frame] = settings.image_strength
        else:
            lane.ref_mask = None

    @torch.no_grad()
    def adopt_refined_state(self, lane, video, audio):
        """Take the refinement stage's starting latents from the second GPU."""
        self._adopt_latents(lane, video, audio, "refinement")
        self.begin_step(lane)
