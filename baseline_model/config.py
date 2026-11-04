from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TransformerConfig:
    """Architecture of the transformer that denoises the packed latents."""
    in_channels: int
    out_channels: int
    patch_size: int
    patch_size_t: int
    num_attention_heads: int
    attention_head_dim: int
    cross_attention_dim: int
    vae_scale_factors: tuple
    pos_embed_max_pos: int
    base_height: int
    base_width: int
    gated_attn: bool
    cross_attn_mod: bool
    audio_in_channels: int
    audio_out_channels: int
    audio_patch_size: int
    audio_patch_size_t: int
    audio_num_attention_heads: int
    audio_attention_head_dim: int
    audio_cross_attention_dim: int
    audio_scale_factor: int
    audio_pos_embed_max_pos: int
    audio_sampling_rate: int
    audio_hop_length: int
    audio_gated_attn: bool
    audio_cross_attn_mod: bool
    num_layers: int
    activation_fn: str
    qk_norm: str
    norm_elementwise_affine: bool
    norm_eps: float
    caption_channels: int
    attention_bias: bool
    attention_out_bias: bool
    rope_theta: float
    rope_double_precision: bool
    causal_offset: int
    timestep_scale_multiplier: int
    cross_attn_timestep_scale_multiplier: int
    rope_type: str
    use_prompt_embeddings: bool
    perturbed_attn: bool
    ff_bias: bool
    audio_ff_bias: bool
    use_prompt_adaln_single: bool
    use_keyframes_abs_pos_embedding: bool


@dataclass
class ConnectorConfig:
    """Architecture of the connectors that compress text into conditioning tokens."""
    caption_channels: int
    text_proj_in_factor: int
    video_connector_num_attention_heads: int
    video_connector_attention_head_dim: int
    video_connector_num_layers: int
    video_connector_num_learnable_registers: int
    video_gated_attn: bool
    audio_connector_num_attention_heads: int
    audio_connector_attention_head_dim: int
    audio_connector_num_layers: int
    audio_connector_num_learnable_registers: int
    audio_gated_attn: bool
    connector_rope_base_seq_len: int
    rope_theta: float
    rope_double_precision: bool
    rope_type: str
    per_modality_projections: bool
    video_hidden_dim: int
    audio_hidden_dim: int
    proj_bias: bool


@dataclass
class DurationHeadConfig:
    """Architecture of the head that predicts a duration from the prompt."""
    video_cross_attention_dim: int
    audio_cross_attention_dim: int
    pooler_hidden_dim: int
    num_queries: int
    num_pooler_heads: int
    mlp_hidden_dim: int


@dataclass
class VideoVAEConfig:
    """Architecture of the video VAE that converts between pixels and latents."""
    in_channels: int
    out_channels: int
    latent_channels: int
    block_out_channels: tuple
    down_block_types: tuple
    decoder_block_out_channels: tuple
    layers_per_block: tuple
    decoder_layers_per_block: tuple
    spatio_temporal_scaling: tuple
    decoder_spatio_temporal_scaling: tuple
    decoder_inject_noise: tuple
    downsample_type: tuple
    upsample_type: tuple
    upsample_residual: tuple
    upsample_factor: tuple
    timestep_conditioning: bool
    patch_size: int
    patch_size_t: int
    resnet_norm_eps: float
    scaling_factor: float
    encoder_causal: bool
    decoder_causal: bool
    encoder_spatial_padding_mode: str
    decoder_spatial_padding_mode: str
    spatial_compression_ratio: int
    temporal_compression_ratio: int


@dataclass
class AudioVAEConfig:
    """Architecture of the audio VAE that converts between mel spectrograms and latents."""
    base_channels: int
    output_channels: int
    ch_mult: tuple
    num_res_blocks: int
    attn_resolutions: Optional[list]
    in_channels: int
    resolution: int
    latent_channels: int
    norm_type: str
    causality_axis: str
    dropout: float
    mid_block_add_attention: bool
    sample_rate: int
    mel_hop_length: int
    is_causal: bool
    mel_bins: int
    double_z: bool
    # Not named by the audio VAE's own config, so the engine supplies them.
    temporal_compression_ratio: int = 4
    mel_compression_ratio: int = 4


@dataclass
class VocoderConfig:
    """Architecture of the vocoder that turns mel spectrograms into a waveform."""
    in_channels: int
    hidden_channels: int
    out_channels: int
    upsample_kernel_sizes: tuple
    upsample_factors: tuple
    resnet_kernel_sizes: tuple
    resnet_dilations: tuple
    act_fn: str
    leaky_relu_negative_slope: float
    antialias: bool
    antialias_ratio: int
    antialias_kernel_size: int
    final_act_fn: Optional[str]
    final_bias: bool
    bwe_in_channels: int
    bwe_hidden_channels: int
    bwe_out_channels: int
    bwe_upsample_kernel_sizes: tuple
    bwe_upsample_factors: tuple
    bwe_resnet_kernel_sizes: tuple
    bwe_resnet_dilations: tuple
    bwe_act_fn: str
    bwe_leaky_relu_negative_slope: float
    bwe_antialias: bool
    bwe_antialias_ratio: int
    bwe_antialias_kernel_size: int
    bwe_final_act_fn: Optional[str]
    bwe_final_bias: bool
    filter_length: int
    hop_length: int
    window_length: int
    num_mel_channels: int
    input_sampling_rate: int
    output_sampling_rate: int


@dataclass
class UpsamplerConfig:
    """Architecture of the latent upsampler used by stage-2 refinement."""
    in_channels: int
    mid_channels: int
    num_blocks_per_stage: int
    dims: int
    spatial_upsample: bool
    temporal_upsample: bool
    rational_spatial_scale: float
    use_rational_resampler: bool


@dataclass
class SchedulerConfig:
    """Sampling settings for the flow-matching scheduler."""
    num_train_timesteps: int
    shift: float
    use_dynamic_shifting: bool
    base_shift: float
    max_shift: float
    base_image_seq_len: int
    max_image_seq_len: int
    shift_terminal: Optional[float]
    use_karras_sigmas: bool
    use_exponential_sigmas: bool
    use_beta_sigmas: bool
    invert_sigmas: bool
    stochastic_sampling: bool
    time_shift_type: str
    # Not named by the scheduler config, so the engine supplies them.
    sigma_min: float = 0.0
    sigma_max: float = 1.0


@dataclass
class ModelConfig:
    """Settings read from the checkpoint's component config files."""
    transformer: TransformerConfig
    connectors: ConnectorConfig
    vae: VideoVAEConfig
    audio_vae: AudioVAEConfig
    vocoder: VocoderConfig
    scheduler: SchedulerConfig
    duration_head: DurationHeadConfig
    upsampler: UpsamplerConfig


@dataclass
class GenerationSettings:
    prompt: str = ""
    negative_prompt: str = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )
    height: int = 512
    width: int = 768
    duration_seconds: Optional[float] = None
    num_inference_steps: int = 30
    guidance_scale: float = 3.0
    stg_scale: float = 1.0
    modality_scale: float = 3.0
    guidance_rescale: float = 0.7
    audio_guidance_scale: float = 7.0
    audio_stg_scale: float = 1.0
    audio_modality_scale: float = 3.0
    audio_guidance_rescale: float = 0.7
    stg_blocks: Optional[list] = None
    seed: int = 0
    higher_resolution: bool = False
    stage2_lora_strength: float = 0.5
    image_path: str = ""
    image_strength: float = 0.8
    max_sequence_length: int = 1024
    output_name: str = "ltx25_output.mp4"


@dataclass
class EngineConfig:
    """Settings the engine owns: which checkpoint to load and how to run it."""
    repo_id: str = "Lightricks/LTX-2.5-Diffusers"
    repo_revision: str = "4791a1100e0cf97f7ed5c55c12dd71dce13f9dd3"
    # Deployment limits, not per-request inputs.
    frame_rate: float = 24.0
    min_seconds: float = 1.0
    max_seconds: float = 20.0
    generation: GenerationSettings = field(default_factory=GenerationSettings)
