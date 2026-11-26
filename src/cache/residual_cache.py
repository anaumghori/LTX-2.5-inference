import logging

from src.profiling import mark

logger = logging.getLogger(__name__)


def cache_for_pass(cache, pass_name):
    """Return the cache switched to one transformer pass or None for a stage that runs without one."""
    return None if cache is None else cache.select_pass(pass_name)


class _PassState:
    """What one transformer pass remembers from the last step it computed in full."""
    def __init__(self):
        self.leading_video = None
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

    def __init__(self, total_blocks, replay_start, replay_end, settings, lane_index, decision_pass=None):
        if not 0 <= replay_start <= replay_end <= total_blocks:
            raise ValueError(
                f"The cache region [{replay_start}, {replay_end}) does not fit inside {total_blocks} blocks."
            )
        self.lane_index = lane_index
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
        self._exhausted = False
        self._min_difference = None

    @property
    def exhausted(self):
        return self._exhausted

    def needs_probe(self, pass_name):
        return (
            self.decides
            and pass_name == self.decision_pass
            and not self._exhausted
            and self._step_index >= self.warmup_steps - 1
        )

    def should_record(self):
        """Whether computed residuals from this step could still be replayed on a later step."""
        return not self._exhausted and self._step_index >= self.warmup_steps - 1

    def begin_step(self, step_index):
        """Open a denoising step and work out whether its passes may be reused."""
        self._step_index = step_index
        self._pass_name = None
        self._step_reuse = False
        if self._exhausted:
            self._passes.clear()
            self._step_eligible = False
            return
        decision_state = self._passes.get(self.decision_pass)
        has_baseline = (
            decision_state is not None
            and decision_state.leading_video is not None
            and decision_state.residual_video is not None
        )
        self._step_eligible = (
            self.decides
            and step_index >= self.warmup_steps
            and self._consecutive_reused_steps < self.max_consecutive_reused_steps
            and has_baseline
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

    def decide(self, pass_name, leading_video=None):
        """Return whether this half's region can be replayed instead of computed"""
        self._require_pass(pass_name)
        if not self.decides:
            raise RuntimeError("A following cache has no decision logic; it replays what it is told to.")
        if pass_name != self.decision_pass:
            reuse = self._step_reuse
            logger.debug(f"Cache step {self._step_index}: pass {pass_name!r} follows the decision, reuse={reuse}")
            return reuse
        state = self._passes.setdefault(pass_name, _PassState())
        difference = 0.0
        if self._exhausted or leading_video is None:
            reuse = False
        else:
            positive_video = leading_video[-1:]
            if state.leading_video is None:
                # The first probed step has nothing to compare against, so it spends none of
                # the budget either.
                reuse = False
            else:
                self._check_shapes(positive_video, state.leading_video, pass_name)
                if self._step_eligible:
                    difference = self._relative_difference(positive_video, state.leading_video)
                    self._min_difference = (
                        difference if self._min_difference is None else min(self._min_difference, difference)
                    )
                    reuse = self._drift_spent + difference <= self.drift_budget
                    if reuse:
                        self._drift_spent += difference
                    remaining = self.drift_budget - self._drift_spent
                    if (
                        remaining <= 0.0
                        or (self._drift_spent > 0.0 and remaining < self._min_difference * 0.25)
                        or (not reuse and self._drift_spent > 0.0 and remaining < self._min_difference * 0.5)
                    ):
                        self._exhausted = True
                        logger.debug(
                            f"Cache step {self._step_index}: drift budget exhausted "
                            f"(remaining {remaining:.5f}, min step delta {self._min_difference:.5f})"
                        )
                else:
                    reuse = False
                logger.debug(
                    f"Cache step {self._step_index}: leading residual difference {difference:.5f}, "
                    f"drift {self._drift_spent:.5f} of {self.drift_budget:.5f}, "
                    f"eligible {self._step_eligible}, reuse={reuse} for blocks "
                    f"{self.replay_start} to {self.replay_end - 1}"
                )
            state.leading_video = None if self._exhausted else positive_video.detach()
        self._step_reuse = reuse
        self._consecutive_reused_steps = self._consecutive_reused_steps + 1 if reuse else 0
        age = self._step_index - state.residual_step_index if state.residual_video is not None else -1
        mark(
            f"cache lane={self.lane_index} step={self._step_index} reuse={int(reuse)} "
            f"eligible={int(self._step_eligible)} "
            f"difference={difference:.5f} spent={self._drift_spent:.5f} "
            f"budget={self.drift_budget:.5f} age={age}"
        )
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
        self._exhausted = False
        self._min_difference = None

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
        return ((current - cached).abs().mean() / (cached.abs().mean() + 1e-8)).item()
