import dataclasses
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file

from src.model.config import (
    AudioVAEConfig,
    ConnectorConfig,
    DurationHeadConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
    TransformerConfig,
    UpsamplerConfig,
    VideoVAEConfig,
    VocoderConfig,
)
from src.model.logging_utils import get_logger
from src.model.lora import parse_lora_state_dict

logger = get_logger(__name__)

# Curated download scope so the engine never fetches components it cannot reach:
# transformer/ holds the distilled DiT (this engine loads transformer_full), the
# prompt enhancer and its processor are not referenced, the temporal upsampler is
# never run, and the connectors weights ship both merged and split, so only the
# merged file is requested.
DOWNLOAD_PATTERNS = [
    "model_index.json",
    "transformer_full/*",
    "text_encoder/*",
    "tokenizer/*",
    "connectors/config.json",
    "connectors/diffusion_pytorch_model.safetensors",
    "vae/*",
    "audio_vae/*",
    "vocoder/*",
    "duration_head/*",
    "latent_upsampler/*",
    "scheduler/scheduler_config.json",
    "*lora*.safetensors",  # stage-2 LoRA sits at the repo root
]


def read_hf_token(env_path):
    """Read the Hugging Face token from a .env file at the given path."""
    token = None
    for line in Path(env_path).read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN="):
            token = line[len("HF_TOKEN="):].strip().strip("\"").strip("'")
    if not token:
        raise ValueError(f"HF_TOKEN not found in {env_path}")
    return token


def to_tuple(value):
    return tuple(to_tuple(item) for item in value) if isinstance(value, list) else value


def build_component_config(cls, folder, name, filename="config.json", key_map=None):
    """Build a component config from the checkpoint's own config file.

    Every architecture value comes from the checkpoint, so a setting its config stops
    naming fails here rather than silently running on a value the engine invented.
    """
    path = Path(folder) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} config at {path}")
    data = json.loads(path.read_text())
    if key_map is not None:
        data = {key_map.get(key, key): value for key, value in data.items()}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    values = {}
    for key, value in data.items():
        if key not in fields:
            continue
        if fields[key].type is tuple and isinstance(value, list):
            value = to_tuple(value)
        values[key] = value
    missing = sorted(
        f.name for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.name not in values
    )
    if missing:
        raise ValueError(f"{name} config at {path} does not name required settings: {missing}")
    return cls(**values)


def safetensors_state_dict(folder):
    """Load every safetensors shard in a folder into one weight map"""
    folder = Path(folder)
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No .safetensors files found in {folder}")
    state = {}
    for file in files:
        logger.info(f"Loading weights file {file} ({file.stat().st_size / 1e9:.2f} GB)")
        state.update(load_file(str(file), device="cpu"))
    logger.info(f"Loaded {len(state)} tensors from {folder}")
    return state


def safetensors_subset(folder, keys, name):
    folder = Path(folder)
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No .safetensors files found in {folder}")
    state = {}
    stored = 0
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                stored += 1
                if key in keys:
                    state[key] = handle.get_tensor(key)
    logger.info(
        f"{name}: read {len(state)} of {stored} stored tensors from {folder}, "
        f"the rest belongs to the other half"
    )
    return state


def strict_load(module, state, name):
    """Load weights into a module and fail loudly on any mismatch.

    Every missing or unexpected key is an error, because a checkpoint that does not
    line up with the architecture would otherwise run with untrained layers.
    """
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.error(f"Weight load mismatch for {name}: missing={len(missing)} unexpected={len(unexpected)}")
        for key in list(missing)[:20]:
            logger.error(f"  missing: {key}")
        for key in list(unexpected)[:20]:
            logger.error(f"  unexpected: {key}")
        raise ValueError(f"Weight mapping failed for {name}: missing={len(missing)} unexpected={len(unexpected)}")
    logger.info(f"Weights loaded for {name}: {len(state)} tensors")


def resolve_subfolder(checkpoint_dir, candidates):
    """Find the first existing checkpoint subfolder from a list of candidates.

    Checkpoints name some components differently across versions, so callers
    pass every known name and the first one found on disk wins.
    """
    for candidate in candidates:
        folder = Path(checkpoint_dir) / candidate
        if folder.is_dir():
            return folder
    raise ValueError(f"None of the candidate subfolders {candidates} exist in {checkpoint_dir}")


def download_checkpoint(repo_id, revision, token):
    logger.info(f"Downloading checkpoint {repo_id} at revision {revision}")
    logger.info(f"Requesting {len(DOWNLOAD_PATTERNS)} component patterns, other repo components are not fetched")
    local_dir = snapshot_download(
        repo_id=repo_id, revision=revision, token=token, allow_patterns=DOWNLOAD_PATTERNS
    )
    logger.info(f"Checkpoint resolved at {local_dir}")
    return Path(local_dir)


def resolve_checkpoint(engine_config, checkpoint_dir=None):
    """The local checkpoint directory, downloading it when none was given."""
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)
    project_root = Path(__file__).resolve().parent.parent.parent
    token = read_hf_token(project_root / ".env")
    return download_checkpoint(engine_config.repo_id, engine_config.repo_revision, token)


def load_model_configs(engine_config=None, checkpoint_dir=None):
    """Read every component's architecture from the checkpoint without loading weights.

    The engine reads these to decide where to cut the block stack, which it has to know
    before either worker starts, so nothing here allocates a model.
    """
    if engine_config is None:
        engine_config = EngineConfig()
    checkpoint_dir = resolve_checkpoint(engine_config, checkpoint_dir)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    transformer_dir = resolve_subfolder(checkpoint_dir, ["transformer_full"])
    transformer_config = build_component_config(TransformerConfig, transformer_dir, "transformer")
    connectors_dir = resolve_subfolder(checkpoint_dir, ["connectors"])
    connectors_config = build_component_config(ConnectorConfig, connectors_dir, "connectors")
    vae_dir = resolve_subfolder(checkpoint_dir, ["vae"])
    vae_config = build_component_config(VideoVAEConfig, vae_dir, "vae")
    audio_vae_dir = resolve_subfolder(checkpoint_dir, ["audio_vae"])
    audio_vae_config = build_component_config(AudioVAEConfig, audio_vae_dir, "audio_vae")
    vocoder_dir = resolve_subfolder(checkpoint_dir, ["vocoder"])
    vocoder_config = build_component_config(VocoderConfig, vocoder_dir, "vocoder")
    duration_head_dir = checkpoint_dir / "duration_head"
    duration_head_config = build_component_config(DurationHeadConfig, duration_head_dir, "duration_head")
    upsampler_dir = checkpoint_dir / "latent_upsampler"
    upsampler_config = build_component_config(UpsamplerConfig, upsampler_dir, "upsampler")
    scheduler_dir = resolve_subfolder(checkpoint_dir, ["scheduler"])
    scheduler_config = build_component_config(
        SchedulerConfig, scheduler_dir, "scheduler", filename="scheduler_config.json"
    )
    # The checkpoint ships sampling values tuned for the distilled few-step path, so
    # re-enable length-adaptive shifting for full-model denoising.
    scheduler_config = dataclasses.replace(scheduler_config, use_dynamic_shifting=True, shift_terminal=0.1)
    model_config = ModelConfig(
        transformer=transformer_config,
        connectors=connectors_config,
        vae=vae_config,
        audio_vae=audio_vae_config,
        vocoder=vocoder_config,
        scheduler=scheduler_config,
        duration_head=duration_head_config,
        upsampler=upsampler_config,
    )
    return model_config, checkpoint_dir


def load_stage2_adapters(checkpoint_dir, engine_config):
    """Parse the stage-2 LoRA file into its transformer and connector halves."""
    if not engine_config.generation.higher_resolution:
        logger.info("higher_resolution is disabled, skipping stage-2 LoRA weights")
        return None, None
    lora_files = sorted(Path(checkpoint_dir).glob("*lora*.safetensors"))
    if not lora_files:
        raise ValueError("higher_resolution is enabled but no LoRA weight file exists at the checkpoint root.")
    if len(lora_files) > 1:
        raise ValueError(f"higher_resolution needs one stage-2 LoRA file but found several: {lora_files}")
    logger.info(f"Loading stage-2 LoRA file {lora_files[0]}")
    return parse_lora_state_dict(load_file(str(lora_files[0]), device="cpu"))


