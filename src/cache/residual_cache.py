import logging

logger = logging.getLogger(__name__)


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

    The stack is cut in three: blocks before `replay_start` always run, blocks in
    `[replay_start, replay_end)` are the region this cache owns, and blocks from
    `replay_end` on always run. When a step is reused, the cache adds the residual the
    last computed step produced for its own region and jumps to the end of it.

    The region is split between the two GPUs at the pipeline boundary, so each half
    owns one part of the same residual and replays only that part.
    One pass per step makes the decision and the other passes follow it, so a step is
    computed or reused as a whole."""

    def __init__(self, total_blocks, replay_start, replay_end, settings, decision_pass=None):
        if not 0 <= replay_start <= replay_end <= total_blocks:
            raise ValueError(
                f"The cache region [{replay_start}, {replay_end}) does not fit inside {total_blocks} blocks."
            )
        self.replay_start = replay_start
        self.replay_end = replay_end
        self.decision_pass = decision_pass
        self.decides = decision_pass is not None
        self.drift_budget = settings.drift_budget
        self.max_consecutive_reused_steps = settings.max_consecutive_reused_steps
        self.warmup_steps = settings.warmup_steps
        self._passes = {}
        self._pass_name = None
        self._step_index = -1
        self._step_eligible = False
        self._step_reuse = False
        self._consecutive_reused_steps = 0
        self._computed_passes = 0
        self._reused_passes = 0
        self._drift_spent = 0.0

    def begin_step(self, step_index):
        """Open a denoising step and work out whether its passes may be reused."""
        self._step_index = step_index
        self._pass_name = None
        self._step_reuse = False
        self._step_eligible = (
            self.decides
            and step_index >= self.warmup_steps
            and self._consecutive_reused_steps < self.max_consecutive_reused_steps
            and bool(self._passes)
        )
        if self.decides and not self._step_eligible:
            logger.debug(
                f"Cache step {step_index}: no reuse possible "
                f"(warmup {self.warmup_steps}, {self._consecutive_reused_steps} consecutive reused, "
                f"cap {self.max_consecutive_reused_steps}, passes seen {sorted(self._passes)})"
            )

    def select_pass(self, pass_name):
        """Make one transformer pass the current pass and return the cache."""
        self._pass_name = pass_name
        return self

    def decide(self, pass_name, leading_video, leading_audio):
        """Return whether this half's region can be replayed instead of computed"""
        self._require_pass(pass_name)
        if not self.decides:
            raise RuntimeError("A following cache has no decision logic; it replays what it is told to.")
        state = self._passes.setdefault(pass_name, _PassState())
        positive_video = leading_video[-1:]
        positive_audio = leading_audio[-1:]
        if pass_name != self.decision_pass:
            reuse = self._step_reuse
            logger.debug(f"Cache step {self._step_index}: pass {pass_name!r} follows the decision, reuse={reuse}")
        elif state.leading_video is None:
            # The first step of the stage has nothing to compare against, so it spends none of
            # the budget either.
            reuse = False
        else:
            self._check_shapes(positive_video, state.leading_video, pass_name)
            self._check_shapes(positive_audio, state.leading_audio, pass_name)
            difference = self._relative_difference(positive_video, state.leading_video)
            within_budget = self._drift_spent + difference <= self.drift_budget
            reuse = self._step_eligible and within_budget
            if reuse:
                self._drift_spent += difference
            logger.debug(
                f"Cache step {self._step_index}: leading residual difference {difference:.5f}, "
                f"drift {self._drift_spent:.5f} of {self.drift_budget:.5f}, "
                f"eligible {self._step_eligible}, reuse={reuse} for blocks "
                f"{self.replay_start} to {self.replay_end - 1}"
            )
        if pass_name == self.decision_pass:
            self._step_reuse = reuse
            if reuse:
                self._consecutive_reused_steps += 1
            else:
                self._consecutive_reused_steps = 0
        state.leading_video = positive_video.detach().clone()
        state.leading_audio = positive_audio.detach().clone()
        return reuse

    def replay(self, pass_name):
        """Return this half's stored residual part for one pass."""
        self._require_pass(pass_name)
        state = self._passes.get(pass_name)
        if state is None or state.residual_video is None:
            raise RuntimeError(f"Pass {pass_name!r} was asked to reuse its blocks but has no computed residual yet.")
        self._reused_passes += 1
        logger.debug(
            f"Cache step {self._step_index}: pass {pass_name!r} replays its half of the residual computed at "
            f"step {state.residual_step_index}, {self._step_index - state.residual_step_index} step(s) old"
        )
        return state.residual_video, state.residual_audio

    def record(self, pass_name, step_index, residual_video, residual_audio):
        """Record this half's residual part for the blocks between the region boundaries."""
        self._require_pass(pass_name)
        state = self._passes.setdefault(pass_name, _PassState())
        state.residual_video = residual_video.detach()
        state.residual_audio = residual_audio.detach()
        state.residual_step_index = step_index
        self._computed_passes += 1

    def residual_step(self, pass_name):
        """The step this pass's stored residual was recorded at or -1 when it holds none."""
        state = self._passes.get(pass_name)
        if state is None or state.residual_video is None:
            return -1
        return state.residual_step_index

    def reset(self, reason):
        """Drop every cached residual and log what the finished phase did."""
        logger.debug(
            f"Cache reset ({reason}): {self._computed_passes} computed passes, "
            f"{self._reused_passes} reused passes, drift {self._drift_spent:.5f} of {self.drift_budget:.5f}"
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

    def _require_pass(self, pass_name):
        if self._pass_name != pass_name:
            raise RuntimeError(
                f"The residual cache for pass {pass_name!r} was used while {self._pass_name!r} was selected."
            )

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
