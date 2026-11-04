import dataclasses
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from baseline_model.common import resolve_device
from baseline_model.conditioning import DurationHead, TextConnectors
from baseline_model.config import (
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
from baseline_model.logging_utils import get_logger
from baseline_model.lora import parse_lora_state_dict
from baseline_model.transformer import VideoTransformer
from baseline_model.audio_vae import AudioVAE
from baseline_model.upsampler import LatentUpsampler
from baseline_model.video_vae import VideoVAE
from baseline_model.vocoder import Vocoder


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


def strict_load(module, state, name):
    """Load weights into a module and fail loudly on any mismatch."""
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


def load_pipeline_from_checkpoint(engine_config=None, checkpoint_dir=None, device=None, dtype=None):
    """Build the full pipeline by loading every component from a checkpoint.

    Components that the current settings do not need are skipped, so loading
    stays fast and memory stays low: the upsampler and stage-2 LoRA weights
    are only loaded when those features are enabled. The duration head is
    always loaded, so automatic duration stays available for any request, not
    just the first one.

    Args:
        engine_config: Engine configuration: repository, revision, and generation settings.
        checkpoint_dir: Local checkpoint path. Downloaded when omitted.
        device: Device on which to place the models.
        dtype: Precision used for the model weights.

    Returns:
        The ready-to-run pipeline.
    """
    from baseline_model.pipeline import LTX25Pipeline
    if engine_config is None:
        engine_config = EngineConfig()
    device = resolve_device(device)
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    logger.info(f"Using device {device} with dtype {dtype}")
    if checkpoint_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        token = read_hf_token(project_root / ".env")
        checkpoint_dir = download_checkpoint(engine_config.repo_id, engine_config.repo_revision, token)
    checkpoint_dir = Path(checkpoint_dir)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    transformer_dir = resolve_subfolder(checkpoint_dir, ["transformer_full"])
    transformer_config = build_component_config(TransformerConfig, transformer_dir, "transformer")
    transformer = VideoTransformer(transformer_config).to(device=device, dtype=dtype)
    strict_load(transformer, safetensors_state_dict(transformer_dir), "transformer")
    transformer.eval()
    tokenizer_dir = resolve_subfolder(checkpoint_dir, ["tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    text_encoder_dir = resolve_subfolder(checkpoint_dir, ["text_encoder"])
    text_encoder = AutoModel.from_pretrained(str(text_encoder_dir), dtype=dtype)
    text_encoder.to(device=device)
    text_encoder.eval()

    connectors_dir = resolve_subfolder(checkpoint_dir, ["connectors"])
    connectors_config = build_component_config(ConnectorConfig, connectors_dir, "connectors")
    connectors = TextConnectors(connectors_config).to(device=device, dtype=dtype)
    strict_load(connectors, safetensors_state_dict(connectors_dir), "connectors")
    connectors.eval()

    vae_dir = resolve_subfolder(checkpoint_dir, ["vae"])
    vae_config = build_component_config(VideoVAEConfig, vae_dir, "vae")
    vae = VideoVAE(vae_config).to(device=device, dtype=dtype)
    strict_load(vae, safetensors_state_dict(vae_dir), "vae")
    vae.eval()

    audio_vae_dir = resolve_subfolder(checkpoint_dir, ["audio_vae"])
    audio_vae_config = build_component_config(AudioVAEConfig, audio_vae_dir, "audio_vae")
    audio_vae = AudioVAE(audio_vae_config, with_encoder=False).to(device=device, dtype=dtype)
    audio_vae_state = safetensors_state_dict(audio_vae_dir)
    # Only the decoder side is ever run, so encoder weights are dropped before loading rather than kept resident
    skipped_encoder_keys = [key for key in audio_vae_state if key.startswith("encoder.")]
    decoder_state = {key: value for key, value in audio_vae_state.items() if not key.startswith("encoder.")}
    logger.info(f"Skipping {len(skipped_encoder_keys)} audio encoder weights (audio input deferred)")
    strict_load(audio_vae, decoder_state, "audio_vae")
    audio_vae.eval()

    vocoder_dir = resolve_subfolder(checkpoint_dir, ["vocoder"])
    vocoder_config = build_component_config(VocoderConfig, vocoder_dir, "vocoder")
    vocoder = Vocoder(vocoder_config).to(device=device, dtype=dtype)
    strict_load(vocoder, safetensors_state_dict(vocoder_dir), "vocoder")
    vocoder.eval()

    duration_head_dir = checkpoint_dir / "duration_head"
    duration_head_config = build_component_config(DurationHeadConfig, duration_head_dir, "duration_head")
    duration_head = DurationHead(duration_head_config).to(device=device, dtype=dtype)
    strict_load(duration_head, safetensors_state_dict(duration_head_dir), "duration_head")
    duration_head.eval()

    upsampler_dir = checkpoint_dir / "latent_upsampler"
    upsampler_config = build_component_config(UpsamplerConfig, upsampler_dir, "upsampler")
    upsampler = None

    if engine_config.generation.higher_resolution:
        upsampler = LatentUpsampler(upsampler_config).to(device=device, dtype=dtype)
        strict_load(upsampler, safetensors_state_dict(upsampler_dir), "upsampler")
        upsampler.eval()
    else:
        logger.info("higher_resolution is disabled, skipping upsampler weights")
    lora_entries = None
    connector_lora_entries = None

    if engine_config.generation.higher_resolution:
        lora_files = sorted(checkpoint_dir.glob("*lora*.safetensors"))
        if not lora_files:
            raise ValueError("higher_resolution is enabled but no LoRA weight file exists at the checkpoint root.")
        if len(lora_files) > 1:
            raise ValueError(f"higher_resolution needs one stage-2 LoRA file but found several: {lora_files}")
        logger.info(f"Loading stage-2 LoRA file {lora_files[0]}")
        lora_entries, connector_lora_entries = parse_lora_state_dict(load_file(str(lora_files[0]), device="cpu"))
    scheduler_dir = resolve_subfolder(checkpoint_dir, ["scheduler"])
    scheduler_config = build_component_config(
        SchedulerConfig, scheduler_dir, "scheduler", filename="scheduler_config.json"
    )
    # The checkpoint ships sampling values tuned for the distilled few-step path, so
    # re-enable length-adaptive shifting for full-model denoising.
    scheduler_config = dataclasses.replace(scheduler_config, use_dynamic_shifting=True, shift_terminal=0.1)
    logger.debug(f"Scheduler settings in effect: {dataclasses.asdict(scheduler_config)}")

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
    pipeline = LTX25Pipeline(
        model_config=model_config,
        engine_config=engine_config,
        transformer=transformer,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        connectors=connectors,
        vae=vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        duration_head=duration_head,
        upsampler=upsampler,
        device=device,
        dtype=dtype,
        lora_entries=lora_entries,
        connector_lora_entries=connector_lora_entries,
    )
    return pipeline
