from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

from src.device_memory import BYTES_PER_GB, device_memory, place_on_device, release_device_memory
from src.model.audio_vae import AudioVAE
from src.model.conditioning import DurationHead, TextConnectors
from src.model.fp8 import apply_fp8
from src.model.lora import apply_lora
from src.logging_utils import get_logger
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


def _convert_to_fp8(transformer, device, layers):
    """Swap this half's selected bf16 linears for fp8 ones, in place, before anything runs"""
    if not layers:
        logger.info("Transformer linears run in bf16 (fp8 is off)")
        return
    if device.type != "cuda":
        raise RuntimeError(
            "the transformer runs in fp8, which needs a CUDA device: the converted layers "
            "multiply with torch._scaled_mm and the load probes that kernel at every shape it "
            "converts. This load resolved to "
            f"{device}."
        )
    loaded = device_memory(device)
    torch.cuda.reset_peak_memory_stats(device)
    report = apply_fp8(transformer, layers)
    conversion_peak = torch.cuda.max_memory_allocated(device) / BYTES_PER_GB
    retained = release_device_memory(device)
    logger.info(
        f"FP8 is on for blocks {report['blocks']}: converted {report['converted']} layers "
        f"({report['weight_gb_before']} GB -> {report['weight_gb_after']} GB of weights), "
        f"left {report['kept']} in bf16. This half loaded at {loaded['peak_allocated_gb']:.2f} GB "
        f"and the conversion peaked at {conversion_peak:.2f} GB, leaving "
        f"{retained['allocated_gb']:.2f} GB in use and {retained['free_gb']:.2f} GB free on "
        f"{device}"
    )


def load_front_half(model_config, checkpoint_dir, engine_config, split, device, dtype):
    """Build and load the first GPU's share of the model"""
    checkpoint_dir = Path(checkpoint_dir)
    logger.info(f"Loading the front half on {device} with dtype {dtype} ({split.describe()})")
    transformer_dir = resolve_subfolder(checkpoint_dir, ["transformer_full"])
    transformer = place_on_device(
        VideoTransformer(
            model_config.transformer,
            block_range=range(0, split.split_index),
            with_input=True,
            with_output=False,
            compile_blocks=engine_config.compile_blocks,
        ),
        device,
        dtype,
        "the front half's transformer blocks",
    )
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
    text_encoder = place_on_device(
        AutoModel.from_pretrained(str(text_encoder_dir), dtype=dtype),
        device,
        dtype,
        "the text encoder",
    )
    text_encoder.eval()
    connectors_dir = resolve_subfolder(checkpoint_dir, ["connectors"])
    connectors = place_on_device(
        TextConnectors(model_config.connectors), device, dtype, "the connectors"
    )
    strict_load(connectors, safetensors_state_dict(connectors_dir), "connectors")
    connectors.eval()
    duration_head_dir = checkpoint_dir / "duration_head"
    duration_head = place_on_device(
        DurationHead(model_config.duration_head), device, dtype, "the duration head"
    )
    strict_load(duration_head, safetensors_state_dict(duration_head_dir), "duration_head")
    duration_head.eval()
    lora_entries, connector_lora_entries = load_stage2_adapters(checkpoint_dir, engine_config)
    if lora_entries is not None:
        apply_lora(transformer, lora_entries)
    if connector_lora_entries is not None:
        apply_lora(connectors, connector_lora_entries)
    _convert_to_fp8(transformer, device, engine_config.fp8_layers)
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


def load_back_half(model_config, checkpoint_dir, engine_config, split, device, dtype):
    """Build and load the second GPU's share of the model"""
    checkpoint_dir = Path(checkpoint_dir)
    logger.info(f"Loading the back half on {device} with dtype {dtype} ({split.describe()})")
    transformer_dir = resolve_subfolder(checkpoint_dir, ["transformer_full"])
    transformer = place_on_device(
        VideoTransformer(
            model_config.transformer,
            block_range=range(split.split_index, split.block_count),
            with_input=False,
            with_output=True,
            compile_blocks=engine_config.compile_blocks,
        ),
        device,
        dtype,
        "the back half's transformer blocks",
    )
    strict_load(
        transformer,
        safetensors_subset(transformer_dir, set(transformer.state_dict().keys()), "back transformer"),
        "back transformer",
    )
    transformer.eval()
    vae_dir = resolve_subfolder(checkpoint_dir, ["vae"])
    vae = place_on_device(VideoVAE(model_config.vae), device, dtype, "the video VAE")
    strict_load(vae, safetensors_state_dict(vae_dir), "vae")
    vae.eval()
    audio_vae_dir = resolve_subfolder(checkpoint_dir, ["audio_vae"])
    audio_vae = place_on_device(AudioVAE(model_config.audio_vae), device, dtype, "the audio VAE")
    audio_vae_state = safetensors_state_dict(audio_vae_dir)
    # Only the decoder side is ever run, so encoder weights are dropped before loading rather than kept resident
    skipped_encoder_keys = [key for key in audio_vae_state if key.startswith("encoder.")]
    decoder_state = {key: value for key, value in audio_vae_state.items() if not key.startswith("encoder.")}
    logger.debug(f"Skipping {len(skipped_encoder_keys)} audio encoder weights (audio input deferred)")
    strict_load(audio_vae, decoder_state, "audio_vae")
    audio_vae.eval()
    vocoder_dir = resolve_subfolder(checkpoint_dir, ["vocoder"])
    vocoder = place_on_device(Vocoder(model_config.vocoder), device, dtype, "the vocoder")
    strict_load(vocoder, safetensors_state_dict(vocoder_dir), "vocoder")
    vocoder.eval()
    upsampler = None
    if engine_config.generation.higher_resolution:
        upsampler_dir = checkpoint_dir / "latent_upsampler"
        upsampler = place_on_device(
            LatentUpsampler(model_config.upsampler), device, dtype, "the latent upsampler"
        )
        strict_load(upsampler, safetensors_state_dict(upsampler_dir), "upsampler")
        upsampler.eval()
    else:
        logger.info("higher_resolution is disabled, skipping upsampler weights")
    lora_entries, connector_lora_entries = load_stage2_adapters(checkpoint_dir, engine_config)
    if lora_entries is not None:
        apply_lora(transformer, lora_entries)
    _convert_to_fp8(transformer, device, engine_config.fp8_layers)
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
