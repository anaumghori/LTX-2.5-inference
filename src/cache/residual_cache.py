import logging

logger = logging.getLogger(__name__)

LEADING_BLOCKS = 8
TAIL_BLOCKS = 8
DRIFT_BUDGET = 0.30
WARMUP_STEPS = 10
MAX_CONSECUTIVE_REUSED_STEPS = 1


def cache_for_pass(cache, pass_name):
    """Return the cache switched to one transformer pass or None for a stage that runs without one."""
    return None if cache is None else cache.select_pass(pass_name)


class _PassState:
    """What one transformer pass remembers from the last step it computed in full."""
    def __init__(self):
        self.leading_video = None
        self.leading_audio = None
        self.residual_video = None
        self.residual_audio = None
        self.residual_step_index = -1


class ResidualCache:
    """Reuses the middle of the transformer block stack across denoising steps.

    The first and last blocks always run. The blocks in between are skipped when their
    input barely changed since the last full step, and the residual that step produced
    for the same range is added back instead.

    One pass per step makes the decision and the other passes follow it, so a step is
    computed or reused as a whole. Guidance combines the passes by subtraction, so
    mixing a stale result with a fresh one would corrupt that difference even when each
    pass looks acceptable on its own.
    """

    def __init__(self, total_blocks, decision_pass):
        if LEADING_BLOCKS + TAIL_BLOCKS >= total_blocks:
            raise ValueError(
                f"The cache keeps {LEADING_BLOCKS} leading and {TAIL_BLOCKS} trailing blocks but the "
                f"transformer has {total_blocks} blocks, so there would be nothing left to replace."
            )
        if not decision_pass:
            raise ValueError("decision_pass must name the pass that takes the per-step decision.")
        self.total_blocks = total_blocks
        self.decision_pass = decision_pass
        self._passes = {}
        self._pass_name = None
        self._step_index = -1
        self._step_eligible = False
        self._step_reuse = False
        self._consecutive_reused_steps = 0
        self._computed_passes = 0
        self._reused_passes = 0
        self._drift_spent = 0.0

    @property
    def boundary_index(self):
        """Index of the first block the cache may replace."""
        return LEADING_BLOCKS

    @property
    def tail_index(self):
        """Index of the first trailing block, which always runs."""
        return self.total_blocks - TAIL_BLOCKS

    def begin_step(self, step_index):
        """Open a denoising step and work out whether its passes may be reused."""
        self._step_index = step_index
        self._pass_name = None
        self._step_reuse = False
        self._step_eligible = (
            step_index >= WARMUP_STEPS
            and self._consecutive_reused_steps < MAX_CONSECUTIVE_REUSED_STEPS
            and bool(self._passes)
        )
        if not self._step_eligible:
            logger.debug(
                f"Cache step {step_index}: no reuse possible "
                f"(warmup {WARMUP_STEPS}, {self._consecutive_reused_steps} consecutive reused, "
                f"passes seen {sorted(self._passes)})"
            )

    def select_pass(self, pass_name):
        """Make one transformer pass the current pass and return the cache."""
        self._pass_name = pass_name
        return self

    def should_reuse_middle_blocks(self, leading_video, leading_audio):
        """Return whether the blocks between the boundary and the tail can be skipped this step.

        Only the deciding pass compares residuals. The other passes return the decision
        already taken for the step, so a step is computed or reused as a whole. Every pass
        records its leading residual either way, because the next step compares against it.

        The video residual drives the comparison because it holds nearly all the tokens.
        The audio difference is only logged, to keep the smaller stream visible.
        """
        pass_name = self._require_pass()
        state = self._passes.setdefault(pass_name, _PassState())
        positive_video = leading_video[-1:]
        positive_audio = leading_audio[-1:]
        if pass_name != self.decision_pass:
            reuse = self._step_reuse
            logger.debug(f"Cache step {self._step_index}: pass {pass_name!r} follows the decision, reuse={reuse}")
        elif state.leading_video is None:
            # The first step of the stage has nothing to compare against, so it spends none of
            # the budget either.
            reuse = self._step_reuse = False
        else:
            self._check_shapes(positive_video, state.leading_video, pass_name)
            self._check_shapes(positive_audio, state.leading_audio, pass_name)
            difference = self._relative_difference(positive_video, state.leading_video)
            audio_difference = self._relative_difference(positive_audio, state.leading_audio)
            reuse = self._step_eligible and self._drift_spent + difference <= DRIFT_BUDGET
            if reuse:
                self._drift_spent += difference
            logger.debug(
                f"Cache step {self._step_index}: leading residual difference {difference:.5f} "
                f"(audio {audio_difference:.5f}), drift {self._drift_spent:.5f} of budget "
                f"{DRIFT_BUDGET:.5f}, eligible {self._step_eligible}, reuse={reuse} for blocks "
                f"{self.boundary_index} to {self.tail_index}"
            )
        if pass_name == self.decision_pass:
            self._step_reuse = reuse
            if reuse:
                self._consecutive_reused_steps += 1
            else:
                self._consecutive_reused_steps = 0
        state.leading_video = positive_video.detach()
        state.leading_audio = positive_audio.detach()
        return reuse

    def cached_residual(self):
        """Return the stored residual for the blocks between the boundary and the tail."""
        pass_name = self._require_pass()
        state = self._passes.get(pass_name)
        if state is None or state.residual_video is None:
            raise RuntimeError(f"Pass {pass_name!r} was asked to reuse its blocks but has no computed residual yet.")
        self._reused_passes += 1
        logger.debug(
            f"Cache step {self._step_index}: pass {pass_name!r} replays the residual computed at step "
            f"{state.residual_step_index}, {self._step_index - state.residual_step_index} step(s) old, "
            f"{tuple(state.residual_video.shape)} video and {tuple(state.residual_audio.shape)} audio"
        )
        return state.residual_video, state.residual_audio

    def store_residual(self, residual_video, residual_audio):
        """Record this pass's residual for the blocks between the boundary and the tail."""
        pass_name = self._require_pass()
        state = self._passes.setdefault(pass_name, _PassState())
        state.residual_video = residual_video.detach()
        state.residual_audio = residual_audio.detach()
        state.residual_step_index = self._step_index
        self._computed_passes += 1
        logger.debug(
            f"Cache step {self._step_index}: pass {pass_name!r} stored a residual of "
            f"{tuple(residual_video.shape)} video, {tuple(residual_audio.shape)} audio"
        )

    def reset(self, reason):
        """Drop every cached residual and log what the finished phase did."""
        logger.debug(
            f"Cache reset ({reason}): {self._computed_passes} computed passes, "
            f"{self._reused_passes} reused passes, drift {self._drift_spent:.5f} of budget {DRIFT_BUDGET:.5f}"
        )
        self._passes.clear()
        self._pass_name = None
        self._step_index = -1
        self._step_eligible = False
        self._step_reuse = False
        self._consecutive_reused_steps = 0
        self._computed_passes = 0
        self._reused_passes = 0
        self._drift_spent = 0.0

    def _require_pass(self):
        if self._pass_name is None:
            raise RuntimeError("The residual cache was used outside a transformer pass; call select_pass() first.")
        return self._pass_name

    @staticmethod
    def _check_shapes(current, cached, pass_name):
        if current.shape != cached.shape:
            raise RuntimeError(
                f"The residual cache for pass {pass_name!r} holds shape {tuple(cached.shape)} but received "
                f"{tuple(current.shape)}. A cache cannot span token grids; reset it at the stage boundary."
            )

    @staticmethod
    def _relative_difference(current, cached):
        return (current - cached).abs().mean().item() / (cached.abs().mean().item() + 1e-8)
