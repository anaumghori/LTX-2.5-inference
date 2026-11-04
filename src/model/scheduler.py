import copy
import math
import numpy as np
import torch

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    """Calculate the shift value based on the image sequence length.

    Args:
        image_seq_len: Number of tokens in the image or video sequence.
        base_seq_len: Sequence length at which to use `base_shift`.
        max_seq_len: Sequence length at which to use `max_shift`.
        base_shift: Shift value for the base sequence length.
        max_shift: Shift value for the maximum sequence length.

    Returns:
        The calculated shift value.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    # Longer packed sequences need a stronger shift to keep the noise schedule in a
    # useful range, so interpolate linearly between the base and maximum settings.
    return image_seq_len * m + b


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """Adjust the guided prediction to reduce problems caused by strong guidance.

    Args:
        noise_cfg: Prediction after classifier-free guidance has been applied.
        noise_pred_text: Prediction made using the text prompt, used as a reference for rescaling.
        guidance_rescale: Controls how much rescaling to apply.

    Returns:
        The guided prediction after rescaling.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # A degenerate prediction has no spread to match, so leave it unscaled instead
    # of dividing by zero.
    scale = torch.where(std_cfg > torch.finfo(noise_cfg.dtype).eps, std_text / std_cfg, torch.ones_like(std_cfg))
    noise_pred_rescaled = noise_cfg * scale
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg


def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, sigmas=None, mu=None):
    """Set up the scheduler and return its timesteps and step count.
    Uses a custom timestep list, a custom sigma list, or the scheduler's
    default schedule. Custom timesteps and sigmas cannot both be supplied.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, timesteps=timesteps)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, sigmas=sigmas, mu=mu)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FlowMatchEulerScheduler:
    """Euler sampler for flow-matching models.

    The schedule is a descending list of noise levels. Each step moves the sample
    along the predicted velocity by the gap to the next level, so the model produces
    a velocity and the noise level is reduced in order.
    """
    def __init__(self, config):
        if config.invert_sigmas or config.stochastic_sampling:
            raise NotImplementedError(
                "The checkpoint scheduler requests invert_sigmas="
                f"{config.invert_sigmas}, stochastic_sampling={config.stochastic_sampling}, "
                "which this engine does not implement."
            )
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.shift = config.shift
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.num_inference_steps = 0
        self.timesteps = torch.tensor([1.0])
        self.sigmas = torch.tensor([1.0, 0.0])
        self._step_index = None
        self._begin_index = None
        self.order = 1

    def time_shift(self, mu, exponent, sigmas):
        """Remap sigmas so sampling spends more steps at the noise levels `mu` selects.

        Args:
            mu: Controls how strongly the schedule is shifted.
            exponent: Shapes the shift curve. The engine always passes 1.0.
            sigmas: Sigma values to transform.

        Returns:
            The shifted sigma values.
        """
        # Linear applies the shift directly, while exponential applies it in log space,
        # which pushes the schedule more strongly toward low noise.
        if self.config.time_shift_type == "linear":
            return mu / (mu + (1 / sigmas - 1) ** exponent)
        return math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1) ** exponent)

    def stretch_shift_to_terminal(self, sigmas):
        """Adjust the sigma values so the last value reaches the target terminal value."""
        # Shifting moves the endpoint away from its configured target, so rescale the
        # distance from fully denoised to put the final sigma back on the target.
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        return 1 - (one_minus_z / scale_factor)

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        """Create the inference schedule and store its timesteps and sigma values.

        Args:
            num_inference_steps: Number of inference steps. If omitted, it is taken from the supplied schedule.
            device: Device on which to store the schedule.
            sigmas: Optional custom sigma values.
            mu: Value required for dynamic shifting when that option is enabled.
            timesteps: Optional custom timestep values.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("`mu` must be passed when `use_dynamic_shifting` is True")
        if sigmas is not None and timesteps is not None:
            if len(sigmas) != len(timesteps):
                raise ValueError("`sigmas` and `timesteps` should have the same length")
        if num_inference_steps is not None:
            if (sigmas is not None and len(sigmas) != num_inference_steps) or (timesteps is not None and len(timesteps) != num_inference_steps):
                raise ValueError("`sigmas` and `timesteps` should have the same length as num_inference_steps, if `num_inference_steps` is provided")
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)
        self.num_inference_steps = num_inference_steps
        if timesteps is not None:
            timesteps = np.array(timesteps).astype(np.float32)
        if sigmas is None:
            if timesteps is None:
                # sigma_max and sigma_min are noise levels, not training-scale timesteps,
                # so the default schedule is spread across sigma directly.
                sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps).astype(np.float32)
            else:
                sigmas = timesteps / self.num_train_timesteps
        else:
            sigmas = np.array(sigmas).astype(np.float32)
            num_inference_steps = len(sigmas)
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)
        if self.config.use_karras_sigmas or self.config.use_exponential_sigmas or self.config.use_beta_sigmas:
            raise ValueError("Karras, exponential, and beta sigma schedules are not supported.")
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        # Timesteps are the training-scale version of the sigmas, while the extra trailing
        # zero gives every denoising step a current and next sigma pair, including the last one.
        timesteps = sigmas * self.num_train_timesteps
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self.timesteps = timesteps
        self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        """Find the index of a timestep in the schedule."""
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        # The first schedule entry can duplicate the requested value when sampling resumes
        # or shares a schedule, so take the second match in that case to land on the
        # actual current step instead of the start of the schedule.
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        """Set the schedule index the next `step` call advances from."""
        if self._begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def set_begin_index(self, begin_index):
        """Set the index at which inference should begin."""
        self._begin_index = begin_index

    def step(self, model_output, timestep, sample):
        """Update the latent sample by one Euler step.

        Uses the model's prediction and the difference between the current and
        next sigma values to calculate the next sample. Advances the step index
        so the following call uses the next pair of sigma values."""
        if self._step_index is None:
            self._init_step_index(timestep)
        sample = sample.to(torch.float32)
        sigma_idx = self._step_index
        sigma = self.sigmas[sigma_idx]
        sigma_next = self.sigmas[sigma_idx + 1]
        # The schedule runs from high to low noise, so this difference is negative and
        # moves the sample toward the cleaner next sigma along the predicted direction.
        dt = sigma_next - sigma
        prev_sample = sample + dt * model_output
        self._step_index += 1
        return prev_sample.to(model_output.dtype)

    def clone(self):
        """Return an independent copy of the scheduler, including its current state."""
        return copy.deepcopy(self)
