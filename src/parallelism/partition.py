import dataclasses
from typing import Optional

# Passes travel as short codes, so the receiving GPU can name the pass it was told to run
# without sending the name as text.
PASS_CFG = 1
PASS_STG = 2
PASS_MODALITY = 3


def uses_text_guidance(settings):
    """Whether text guidance runs a negative and a positive prompt side by side."""
    return settings.guidance_scale != 1.0 or settings.audio_guidance_scale != 1.0


def stage2_settings(settings):
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


@dataclasses.dataclass(frozen=True)
class PipelineSplit:
    """How the transformer block stack is divided between the two GPUs, decided once at startup."""
    block_count: int
    front_blocks: int
    cache_start: int
    cache_end: int
    stg_start: int
    has_stg: bool

    @property
    def split_index(self):
        """Index of the first block the second GPU owns."""
        return self.front_blocks

    @property
    def front_range(self):
        return range(0, self.front_blocks)

    @property
    def back_range(self):
        return range(self.front_blocks, self.block_count)

    def describe(self):
        return (
            f"{self.block_count} blocks split after block {self.front_blocks - 1}: "
            f"front {self.front_range.start}-{self.front_range.stop - 1}, "
            f"back {self.back_range.start}-{self.back_range.stop - 1}, "
            f"cache {self.cache_start}-{self.cache_end - 1}, "
            f"mid-stack start {self.stg_start if self.has_stg else 'unused'}"
        )


def plan_split(block_count, settings, cache_settings):
    has_stg, has_modality = has_guidance_passes(settings)
    stg_start = min(settings.stg_blocks) if has_stg else block_count
    full_stack_passes = (2 if uses_text_guidance(settings) else 1) + (1 if has_modality else 0)
    mid_stack_blocks = block_count - stg_start if has_stg else 0
    split = round((block_count * full_stack_passes + mid_stack_blocks) / (2 * full_stack_passes))
    lower_bound = cache_settings.leading_blocks + 1
    upper_bound = min(block_count - cache_settings.tail_blocks, stg_start)
    if lower_bound > upper_bound:
        raise ValueError(
            f"The cache keeps blocks {cache_settings.leading_blocks} through "
            f"{block_count - cache_settings.tail_blocks - 1} and the mid-stack pass starts at {stg_start}, "
            "so there is no block to split after."
        )
    split = min(max(split, lower_bound), upper_bound)
    return PipelineSplit(
        block_count=block_count,
        front_blocks=split,
        cache_start=cache_settings.leading_blocks,
        cache_end=block_count - cache_settings.tail_blocks,
        stg_start=stg_start,
        has_stg=has_stg,
    )


@dataclasses.dataclass(frozen=True)
class PassPlan:
    """One transformer pass over the block stack within a step"""
    name: str
    code: int
    batch: int
    start_block: int
    captures: bool
    isolates_modalities: bool
    cached: bool

    @property
    def needs_front(self):
        """Whether the first GPU has any blocks to run for this pass."""
        return self.start_block == 0

    def describe(self):
        return (
            f"{self.name} batch {self.batch} from block {self.start_block}"
            f"{' capturing' if self.captures else ''}"
            f"{' isolated' if self.isolates_modalities else ''}"
        )


def has_guidance_passes(settings):
    """Which guidance passes the settings enable: mid-stack (STG) and modality isolation."""
    stg = settings.stg_scale != 0.0 or settings.audio_stg_scale != 0.0
    modality = settings.modality_scale != 1.0 or settings.audio_modality_scale != 1.0
    return stg, modality


def step_passes(settings, split):
    """The passes one step runs, in the order both GPUs execute and send them.

    The order is fixed because the CFG pass takes the step-wide cache decision and every
    later pass follows it.
    """
    stg, modality = has_guidance_passes(settings)
    text_guidance = uses_text_guidance(settings)
    passes = [
        PassPlan(
            name="cfg",
            code=PASS_CFG,
            batch=2 if text_guidance else 1,
            start_block=0,
            captures=stg and split.stg_start > 0,
            isolates_modalities=False,
            cached=split.cache_start > 0,
        )
    ]
    if stg:
        passes.append(
            PassPlan(
                name="stg",
                code=PASS_STG,
                batch=1,
                start_block=split.stg_start,
                captures=False,
                isolates_modalities=False,
                cached=False,
            )
        )
    if modality:
        passes.append(
            PassPlan(
                name="modality",
                code=PASS_MODALITY,
                batch=1,
                start_block=0,
                captures=False,
                isolates_modalities=True,
                cached=split.cache_start > 0,
            )
        )
    return tuple(passes)


def stg_pass_after_decision(passes, split, reuse) -> Optional[PassPlan]:
    """The mid-stack (STG) pass to run, now that the CFG pass has made the cache decision."""
    template = next((plan for plan in passes if plan.name == "stg"), None)
    if template is None:
        return None
    return dataclasses.replace(template, start_block=0 if reuse else split.stg_start)
