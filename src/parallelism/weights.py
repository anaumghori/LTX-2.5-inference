from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

from src.model.audio_vae import AudioVAE
from src.model.common import resolve_device
from src.model.conditioning import DurationHead, TextConnectors
from src.model.lora import apply_lora
from src.model.logging_utils import get_logger
from src.model.transformer import VideoTransformer
from src.model.upsampler import LatentUpsampler
from src.model.video_vae import VideoVAE
from src.model.vocoder import Vocoder
from src.model.weights import (
    load_stage2_adapters,
    resolve_subfolder,
    safetensors_state_dict,
    safetensors_subset,
    strict_load,
)
from src.parallelism.back import BackHalf
from src.parallelism.front import FrontHalf

logger = get_logger(__name__)


def load_front_half(model_config, checkpoint_dir, engine_config, split, device=None, dtype=None):
    """Build and load the first GPU's share of the model"""
    checkpoint_dir = Path(checkpoint_dir)
    device = resolve_device(device)
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    logger.info(f"Loading the front half on {device} with dtype {dtype} ({split.describe()})")
    transformer_dir = resolve_subfolder(checkpoint_dir, ["transformer_full"])
    transformer = VideoTransformer(
        model_config.transformer,
        block_range=range(0, split.split_index),
        with_input=True,
        with_output=False,
    ).to(device=device, dtype=dtype)
    strict_load(
        transformer,
        safetensors_subset(transformer_dir, set(transformer.state_dict().keys()), "front transformer"),
        "front transformer",
    )
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
    connectors = TextConnectors(model_config.connectors).to(device=device, dtype=dtype)
    strict_load(connectors, safetensors_state_dict(connectors_dir), "connectors")
    connectors.eval()
    duration_head_dir = checkpoint_dir / "duration_head"
    duration_head = DurationHead(model_config.duration_head).to(device=device, dtype=dtype)
    strict_load(duration_head, safetensors_state_dict(duration_head_dir), "duration_head")
    duration_head.eval()
    lora_entries, connector_lora_entries = load_stage2_adapters(checkpoint_dir, engine_config)
    if lora_entries is not None:
        apply_lora(transformer, lora_entries)
    if connector_lora_entries is not None:
        apply_lora(connectors, connector_lora_entries)
    return FrontHalf(
        model_config=model_config,
        engine_config=engine_config,
        transformer=transformer,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        connectors=connectors,
        duration_head=duration_head,
        device=device,
        dtype=dtype,
        split=split,
        lora_installed=lora_entries is not None and connector_lora_entries is not None,
    )


def load_back_half(model_config, checkpoint_dir, engine_config, split, device=None, dtype=None):
    """Build and load the second GPU's share of the model"""
    checkpoint_dir = Path(checkpoint_dir)
    device = resolve_device(device)
    dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    logger.info(f"Loading the back half on {device} with dtype {dtype} ({split.describe()})")
    transformer_dir = resolve_subfolder(checkpoint_dir, ["transformer_full"])
    transformer = VideoTransformer(
        model_config.transformer,
        block_range=range(split.split_index, split.block_count),
        with_input=False,
        with_output=True,
    ).to(device=device, dtype=dtype)
    strict_load(
        transformer,
        safetensors_subset(transformer_dir, set(transformer.state_dict().keys()), "back transformer"),
        "back transformer",
    )
    transformer.eval()
    vae_dir = resolve_subfolder(checkpoint_dir, ["vae"])
    vae = VideoVAE(model_config.vae).to(device=device, dtype=dtype)
    strict_load(vae, safetensors_state_dict(vae_dir), "vae")
    vae.eval()
    audio_vae_dir = resolve_subfolder(checkpoint_dir, ["audio_vae"])
    audio_vae = AudioVAE(model_config.audio_vae, with_encoder=False).to(device=device, dtype=dtype)
    audio_vae_state = safetensors_state_dict(audio_vae_dir)
    # Only the decoder side is ever run, so encoder weights are dropped before loading rather than kept resident
    skipped_encoder_keys = [key for key in audio_vae_state if key.startswith("encoder.")]
    decoder_state = {key: value for key, value in audio_vae_state.items() if not key.startswith("encoder.")}
    logger.info(f"Skipping {len(skipped_encoder_keys)} audio encoder weights (audio input deferred)")
    strict_load(audio_vae, decoder_state, "audio_vae")
    audio_vae.eval()
    vocoder_dir = resolve_subfolder(checkpoint_dir, ["vocoder"])
    vocoder = Vocoder(model_config.vocoder).to(device=device, dtype=dtype)
    strict_load(vocoder, safetensors_state_dict(vocoder_dir), "vocoder")
    vocoder.eval()
    upsampler = None
    if engine_config.generation.higher_resolution:
        upsampler_dir = checkpoint_dir / "latent_upsampler"
        upsampler = LatentUpsampler(model_config.upsampler).to(device=device, dtype=dtype)
        strict_load(upsampler, safetensors_state_dict(upsampler_dir), "upsampler")
        upsampler.eval()
    else:
        logger.info("higher_resolution is disabled, skipping upsampler weights")
    lora_entries, connector_lora_entries = load_stage2_adapters(checkpoint_dir, engine_config)
    if lora_entries is not None:
        apply_lora(transformer, lora_entries)
    return BackHalf(
        model_config=model_config,
        engine_config=engine_config,
        transformer=transformer,
        vae=vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        upsampler=upsampler,
        device=device,
        dtype=dtype,
        split=split,
        lora_installed=lora_entries is not None and connector_lora_entries is not None,
    )
