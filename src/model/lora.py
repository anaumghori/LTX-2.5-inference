import torch.nn as nn
import torch.nn.functional as F

from src.model.logging_utils import get_logger

logger = get_logger(__name__)

# Checkpoint files name the same layers differently than this codebase, so each entry
# maps a checkpoint prefix to the matching local path. Longer, more specific prefixes
# come first because only the first matching entry is applied to each weight name.
CHECKPOINT_TO_LOCAL_PREFIXES = [
    ("audio_embeddings_connector.transformer_1d_blocks.", "connectors.audio_connector.transformer_blocks."),
    ("video_embeddings_connector.transformer_1d_blocks.", "connectors.video_connector.transformer_blocks."),
    ("audio_embeddings_connector.", "connectors.audio_connector."),
    ("video_embeddings_connector.", "connectors.video_connector."),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
    ("audio_adaln_single", "audio_time_embed"),
    ("audio_patchify_proj", "audio_proj_in"),
    ("adaln_single", "time_embed"),
    ("patchify_proj", "proj_in"),
]


def map_lora_target(key):
    """Translate a LoRA weight name from checkpoint naming to this codebase's naming"""
    # Strip training-wrapper prefixes added by different saving paths, since they do
    # not correspond to real modules in this codebase.
    for prefix in ["base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer."]:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    # Some checkpoints store every name under an extra grouping level that carries no
    # meaning here, so collapse it before matching.
    key = key.replace(".default.", ".")
    # Apply only the first matching rename so a general prefix cannot overwrite a more
    # specific one above. Matching requires a full path segment so similar names that
    # merely share leading characters are not renamed by accident.
    for checkpoint_prefix, local_prefix in CHECKPOINT_TO_LOCAL_PREFIXES:
        checkpoint_prefix = checkpoint_prefix.rstrip(".")
        if key == checkpoint_prefix or key.startswith(checkpoint_prefix + "."):
            key = local_prefix.rstrip(".") + key[len(checkpoint_prefix):]
            break
    return key


def parse_lora_state_dict(state):
    """Collect LoRA matrices from a checkpoint into per-layer entries.
    Groups each layer's A and B matrices with its scaling factor, then splits
    the entries into transformer targets and connector targets.

    Args:
        state: Raw LoRA weights loaded from the checkpoint file.
    """
    flat = {}
    scales = {}
    for key, value in state.items():
        key = map_lora_target(key)
        if key.endswith(".lora_A.weight"):
            target = key[: -len(".lora_A.weight")]
            flat.setdefault(target, {})["A"] = value
        elif key.endswith(".lora_B.weight"):
            target = key[: -len(".lora_B.weight")]
            flat.setdefault(target, {})["B"] = value
        elif key.endswith(".alpha"):
            target = key[: -len(".alpha")]
            scales[target] = float(value.reshape(-1)[0])
        else:
            raise ValueError(f"Unrecognized LoRA key {key}")
    if not flat:
        raise ValueError("LoRA state dict contains no lora_A/lora_B weight pairs")
    transformer_entries = {}
    connector_entries = {}
    for target, pair in flat.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"LoRA target {target} is missing its A or B matrix")
        rank = pair["A"].shape[0]
        if pair["B"].shape[1] != rank:
            raise ValueError(f"LoRA matrices for {target} have incompatible shapes {tuple(pair['A'].shape)} and {tuple(pair['B'].shape)}")
        # The effective strength is the stored alpha divided by rank. When no alpha was saved,
        # defaulting to rank gives a strength of one, which leaves the low-rank update unscaled.
        alpha = scales.get(target, float(rank))
        pair["scale"] = alpha / float(rank)
        # Connector and transformer weights live in different root modules at apply time,
        # so split them now and strip the shared leading path from connector entries.
        if target.startswith("connectors."):
            connector_entries[target[len("connectors."):]] = pair
        else:
            transformer_entries[target] = pair
    logger.info(f"Parsed LoRA with {len(transformer_entries)} transformer targets and {len(connector_entries)} connector targets")
    return transformer_entries, connector_entries


def resolve_parent(root, dotted_path):
    """Find the parent module and attribute name for a dotted LoRA target path."""
    parts = dotted_path.split(".")
    module = root
    # Numeric segments address entries inside a list-like container rather than named
    # attributes, which is how per-layer blocks are stored.
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module, parts[-1]


class LoRAAugmentedLinear(nn.Module):
    """A linear layer with a low-rank adapter added to its output."""
    def __init__(self, base, lora_a, lora_b, scale):
        super().__init__()
        self.base = base
        # Keep the adapter on the base layer's device and precision.
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(lora_a.detach().clone().to(device=device, dtype=dtype))
        self.lora_B = nn.Parameter(lora_b.detach().clone().to(device=device, dtype=dtype))
        self.lora_scale = scale

    def forward(self, x):
        out = self.base(x)
        # Run the low-rank path in its stored precision for stability, then convert back
        # so the adjustment matches the base output before the two are combined.
        lora_out = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + self.lora_scale * lora_out.to(out.dtype)


def apply_lora(root, entries, scale=1.0):
    """Attach LoRA adapters to the matching linear layers of a module.

    Layers with no match are skipped and if too many targets fail to match,
    every change is rolled back before raising an error, so the model is left
    exactly as it was found.

    Args:
        root: Module whose linear layers should receive adapters.
        entries: Parsed LoRA entries mapping layer paths to weight pairs.
        scale: Extra multiplier applied on top of each entry's own scale.

    Returns:
        Records of the replaced layers, for use with `remove_lora`.
    """
    originals = []
    original_devices = []
    original_dtypes = []
    skipped = []
    try:
        for target, pair in entries.items():
            try:
                parent, name = resolve_parent(root, target)
                base = getattr(parent, name, None)
            except (AttributeError, IndexError, TypeError):
                base = None
            if not isinstance(base, nn.Linear):
                skipped.append(target)
                continue
            if pair["A"].shape[1] != base.in_features or pair["B"].shape[0] != base.out_features:
                raise ValueError(f"LoRA matrices for {target} do not match Linear({base.in_features}, {base.out_features})")
            base_device = base.weight.device
            base_dtype = base.weight.dtype
            augmented = LoRAAugmentedLinear(base, pair["A"], pair["B"], scale * pair["scale"])
            setattr(parent, name, augmented)
            originals.append((parent, name, base))
            original_devices.append(base_device)
            original_dtypes.append(base_dtype)
    except Exception:
        # If anything fails partway, put every already-swapped layer back so the model
        # is never left with only half of the adapter applied.
        for (parent, name, base), base_device, base_dtype in zip(originals, original_devices, original_dtypes):
            setattr(parent, name, base)
            base.to(device=base_device, dtype=base_dtype)
        raise
    if skipped:
        logger.warning(f"LoRA skipped {len(skipped)} targets with no matching module:")
        for target in skipped[:30]:
            logger.debug("  skipped LoRA target: " + target)
    # A small number of misses is expected across checkpoint variants, but matching fewer
    # than half the targets usually means the wrong adapter is being applied, so roll back
    # everything instead of leaving a partially adapted model.
    if len(originals) * 2 < len(entries):
        for (parent, name, base), base_device, base_dtype in zip(originals, original_devices, original_dtypes):
            setattr(parent, name, base)
            base.to(device=base_device, dtype=base_dtype)
        raise ValueError(f"LoRA applied to only {len(originals)} of {len(entries)} targets, refusing partial application")
    logger.debug(f"Applied LoRA to {len(originals)} linear layers with scale {scale}")
    return originals


def remove_lora(originals):
    """Restore the original layers saved by `apply_lora`."""
    for parent, name, base in originals:
        setattr(parent, name, base)
    logger.debug(f"Removed LoRA from {len(originals)} linear layers")
