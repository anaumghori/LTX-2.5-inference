import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_model.transformer import FeedForward, LTX2Attention


def per_layer_masked_mean_norm(text_hidden_states, sequence_lengths, device, padding_side="left", scale_factor=8, eps=1e-6):
    """Normalize text hidden states while ignoring padding tokens.

    Args:
        text_hidden_states: Encoder states stacked as [batch, tokens, width, layers].
        sequence_lengths: Number of real tokens per batch entry.
        device: Device used for the position indices.
        padding_side: Which end of the sequence holds the real tokens.
        scale_factor: Multiplier applied to the normalized values.

    Returns:
        The normalized states with padding zeroed out.
    """
    batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype
    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    mask = mask[:, :, None, None]
    masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)
    # Fill padding with infinities so the min/max reductions only ever see real
    # tokens, no matter how extreme the actual values are.
    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized_hidden_states = normalized_hidden_states * scale_factor
    normalized_hidden_states = normalized_hidden_states.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
    normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
    return normalized_hidden_states.to(dtype=original_dtype)


def per_token_rms_norm(text_encoder_hidden_states, eps=1e-6):
    """Rescale every token to unit root-mean-square, keeping its direction unchanged."""
    variance = torch.mean(text_encoder_hidden_states**2, dim=2, keepdim=True)
    return text_encoder_hidden_states * torch.rsqrt(variance + eps)


class ConnectorRotaryPosEmbed1d(nn.Module):
    """Build cosine and sine rotary tables for each position in a 1D token sequence."""
    def __init__(
        self,
        dim,
        base_seq_len=4096,
        theta=10000.0,
        double_precision=True,
        rope_type="interleaved",
        num_attention_heads=32,
    ):
        super().__init__()
        if rope_type not in ["interleaved", "split"]:
            raise ValueError(f"{rope_type=} not supported. Choose between 'interleaved' and 'split'.")
        self.dim = dim
        self.base_seq_len = base_seq_len
        self.theta = theta
        self.double_precision = double_precision
        self.rope_type = rope_type
        self.num_attention_heads = num_attention_heads

    def forward(self, batch_size, seq_len, device):
        grid_1d = torch.arange(seq_len, dtype=torch.float32, device=device)
        grid_1d = grid_1d / self.base_seq_len
        grid = grid_1d.unsqueeze(0).repeat(batch_size, 1)
        num_rope_elems = 2
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                start=0.0,
                end=1.0,
                steps=self.dim // num_rope_elems,
                dtype=freqs_dtype,
                device=device,
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)
        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)
            # Leftover dimensions that do not fill a rotary pair carry no position
            # signal, so pad them with a neutral rotation of zero.
            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % num_rope_elems])
                sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % num_rope_elems])
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)
        elif self.rope_type == "split":
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()
            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
                cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
                sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)
            b = cos_freq.shape[0]
            t = cos_freq.shape[1]
            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)
            cos_freqs = torch.swapaxes(cos_freq, 1, 2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)
        return cos_freqs, sin_freqs


class ConnectorBlock1d(nn.Module):
    """Refine text tokens using self-attention and a feed-forward layer, with residual connections."""
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        eps=1e-6,
        rope_type="interleaved",
        apply_gated_attention=False,
    ):
        super().__init__()
        self.norm1 = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )
        self.norm2 = torch.nn.RMSNorm(dim, eps=eps, elementwise_affine=False)
        self.ff = FeedForward(dim, bias=True, activation_fn="gelu-approximate")

    def forward(self, hidden_states, attention_mask=None, rotary_emb=None):
        norm_hidden_states = self.norm1(hidden_states)
        attn_hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, query_rotary_emb=rotary_emb)
        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        ff_hidden_states = self.ff(norm_hidden_states)
        return hidden_states + ff_hidden_states


class ConnectorTransformer1d(nn.Module):
    """Compress a text sequence into a fixed set of conditioning registers.

    Real tokens are gathered to the front and padding slots are filled with
    learned registers, so the transformer always returns the same number of
    conditioning tokens regardless of prompt length.
    """
    def __init__(
        self,
        num_attention_heads,
        attention_head_dim,
        num_layers,
        num_learnable_registers,
        rope_base_seq_len,
        rope_theta,
        rope_double_precision,
        rope_type,
        gated_attention,
        eps=1e-6,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_learnable_registers = num_learnable_registers
        self.learnable_registers = None
        if num_learnable_registers is not None:
            init_registers = torch.rand(num_learnable_registers, self.inner_dim) * 2.0 - 1.0
            self.learnable_registers = torch.nn.Parameter(init_registers)
        self.rope = ConnectorRotaryPosEmbed1d(
            self.inner_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=num_attention_heads,
        )
        self.transformer_blocks = torch.nn.ModuleList([
            ConnectorBlock1d(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                rope_type=rope_type,
                apply_gated_attention=gated_attention,
            )
            for _ in range(num_layers)
        ])
        self.norm_out = torch.nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=False)

    def forward(self, hidden_states, attention_mask=None, attn_mask_binarize_threshold=-9000.0):
        """Compress the text sequence and replace its padding slots with learned registers.

        Args:
            hidden_states: Input text tokens.
            attention_mask: Additive mask marking real tokens.
            attn_mask_binarize_threshold: Cutoff separating real tokens from padding.

        Returns:
            The compressed conditioning tokens and the updated mask.
        """
        batch_size, seq_len, _ = hidden_states.shape
        if self.learnable_registers is not None:
            if seq_len % self.num_learnable_registers != 0:
                raise ValueError(
                    f"The `hidden_states` sequence length {hidden_states.shape[1]} should be divisible by "
                    f"the number of learnable registers {self.num_learnable_registers}"
                )
            num_register_repeats = seq_len // self.num_learnable_registers
            registers = self.learnable_registers.unsqueeze(0).expand(num_register_repeats, -1, -1).reshape(seq_len, -1)
            binary_attn_mask = (attention_mask >= attn_mask_binarize_threshold).int()
            if binary_attn_mask.ndim == 4:
                binary_attn_mask = binary_attn_mask.squeeze(1).squeeze(1)
            # Sort real tokens to the front so padding always lands in the trailing
            # slots that the learned registers replace. The stable sort keeps the
            # original token order among the real tokens.
            order = torch.argsort(1 - binary_attn_mask, dim=1, stable=True)
            front_aligned = torch.gather(hidden_states, 1, order.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]))
            num_valid = binary_attn_mask.sum(dim=1, keepdim=True)
            positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            front_mask = (positions < num_valid).unsqueeze(-1)
            registers_expanded = registers.unsqueeze(0).expand(batch_size, -1, -1)
            hidden_states = torch.where(front_mask, front_aligned, registers_expanded.to(hidden_states.dtype))
            # Every slot now holds either a real token or a register, so the mask
            # is cleared and the blocks attend to the full compressed sequence.
            attention_mask = torch.zeros_like(attention_mask)
        rotary_emb = self.rope(batch_size, seq_len, device=hidden_states.device)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states, attention_mask


class TextConnectors(nn.Module):
    """Turn text-encoder output into the video and audio conditioning tokens"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        text_encoder_dim = config.caption_channels * config.text_proj_in_factor
        if config.per_modality_projections:
            self.video_text_proj_in = nn.Linear(text_encoder_dim, config.video_hidden_dim, bias=config.proj_bias)
            self.audio_text_proj_in = nn.Linear(text_encoder_dim, config.audio_hidden_dim, bias=config.proj_bias)
        else:
            self.text_proj_in = nn.Linear(text_encoder_dim, config.caption_channels, bias=config.proj_bias)
        self.video_connector = ConnectorTransformer1d(
            num_attention_heads=config.video_connector_num_attention_heads,
            attention_head_dim=config.video_connector_attention_head_dim,
            num_layers=config.video_connector_num_layers,
            num_learnable_registers=config.video_connector_num_learnable_registers,
            rope_base_seq_len=config.connector_rope_base_seq_len,
            rope_theta=config.rope_theta,
            rope_double_precision=config.rope_double_precision,
            rope_type=config.rope_type,
            gated_attention=config.video_gated_attn,
        )
        self.audio_connector = ConnectorTransformer1d(
            num_attention_heads=config.audio_connector_num_attention_heads,
            attention_head_dim=config.audio_connector_attention_head_dim,
            num_layers=config.audio_connector_num_layers,
            num_learnable_registers=config.audio_connector_num_learnable_registers,
            rope_base_seq_len=config.connector_rope_base_seq_len,
            rope_theta=config.rope_theta,
            rope_double_precision=config.rope_double_precision,
            rope_type=config.rope_type,
            gated_attention=config.audio_gated_attn,
        )

    def forward(self, text_encoder_hidden_states, attention_mask, padding_side="left", scale_factor=8):
        """Turn encoder states into the video and audio conditioning tokens.

        Args:
            text_encoder_hidden_states: Raw encoder states, optionally still stacked by layer.
            attention_mask: Binary mask marking real prompt tokens.
            padding_side: Which end of the sequence holds the real tokens.
            scale_factor: Output range used by the fallback normalization path.

        Returns:
            The video tokens, audio tokens, and the binary token mask.
        """
        if text_encoder_hidden_states.ndim == 3:
            text_encoder_hidden_states = text_encoder_hidden_states.unflatten(2, (self.config.caption_channels, -1))
        if self.config.per_modality_projections:
            norm_text_encoder_hidden_states = per_token_rms_norm(text_encoder_hidden_states)
            norm_text_encoder_hidden_states = norm_text_encoder_hidden_states.flatten(2, 3)
            bool_mask = attention_mask.bool().unsqueeze(-1)
            norm_text_encoder_hidden_states = torch.where(
                bool_mask, norm_text_encoder_hidden_states, torch.zeros_like(norm_text_encoder_hidden_states)
            )
            video_scale_factor = math.sqrt(self.config.video_hidden_dim / self.config.caption_channels)
            video_norm_text_emb = norm_text_encoder_hidden_states * video_scale_factor
            audio_scale_factor = math.sqrt(self.config.audio_hidden_dim / self.config.caption_channels)
            audio_norm_text_emb = norm_text_encoder_hidden_states * audio_scale_factor
            video_text_emb_proj = self.video_text_proj_in(video_norm_text_emb)
            audio_text_emb_proj = self.audio_text_proj_in(audio_norm_text_emb)
        else:
            sequence_lengths = attention_mask.sum(dim=-1)
            norm_text_encoder_hidden_states = per_layer_masked_mean_norm(
                text_hidden_states=text_encoder_hidden_states,
                sequence_lengths=sequence_lengths,
                device=text_encoder_hidden_states.device,
                padding_side=padding_side,
                scale_factor=scale_factor,
            )
            text_emb_proj = self.text_proj_in(norm_text_encoder_hidden_states)
            video_text_emb_proj = text_emb_proj
            audio_text_emb_proj = text_emb_proj
        text_dtype = video_text_emb_proj.dtype
        # The transformer blocks expect an additive mask, so convert the binary mask:
        # real tokens contribute zero and padding contributes the most negative value.
        attention_mask = (attention_mask.to(torch.int64) - 1).to(text_dtype)
        attention_mask = attention_mask.reshape(attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        add_attn_mask = attention_mask * torch.finfo(text_dtype).max
        video_text_embedding, video_attn_mask = self.video_connector(video_text_emb_proj, add_attn_mask)
        binary_attn_mask = (video_attn_mask < 1e-6).to(torch.int64)
        binary_attn_mask = binary_attn_mask.reshape(video_text_embedding.shape[0], video_text_embedding.shape[1], 1)
        video_text_embedding = video_text_embedding * binary_attn_mask
        audio_text_embedding, _ = self.audio_connector(audio_text_emb_proj, add_attn_mask)
        return video_text_embedding, audio_text_embedding, binary_attn_mask.squeeze(-1)


class DurationAttentionPooler(nn.Module):
    """Compress a token sequence into a fixed summary using learned query tokens."""
    def __init__(self, hidden_dim=256, num_queries=1, num_heads=4):
        super().__init__()
        self.heads = num_heads
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens):
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        query = self.to_q(queries).unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = self.to_k(tokens).unflatten(2, (self.heads, -1)).transpose(1, 2)
        value = self.to_v(tokens).unflatten(2, (self.heads, -1)).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        return self.to_out(hidden_states)


class DurationHead(nn.Module):
    """Predict the video duration in seconds from the conditioning tokens.

    Video and audio tokens are projected into one shared space, marked so the
    pooler can tell the modalities apart, then pooled and mapped to a duration.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_input_proj = nn.Linear(config.video_cross_attention_dim, config.pooler_hidden_dim)
        self.video_modality_emb = nn.Parameter(torch.randn(config.pooler_hidden_dim) * 0.02)
        self.audio_input_proj = nn.Linear(config.audio_cross_attention_dim, config.pooler_hidden_dim)
        self.audio_modality_emb = nn.Parameter(torch.randn(config.pooler_hidden_dim) * 0.02)
        self.attention_pooler = DurationAttentionPooler(
            hidden_dim=config.pooler_hidden_dim,
            num_queries=config.num_queries,
            num_heads=config.num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(config.pooler_hidden_dim * config.num_queries, config.mlp_hidden_dim)
        self.mlp_out = nn.Linear(config.mlp_hidden_dim, 1)

    def forward(self, video_tokens=None, audio_tokens=None):
        if video_tokens is None and audio_tokens is None:
            raise ValueError("DurationHead requires at least one of video_tokens / audio_tokens.")
        head_dtype = next(self.parameters()).dtype
        token_groups = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens.to(head_dtype)) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens.to(head_dtype)) + self.audio_modality_emb)
        tokens = torch.cat(token_groups, dim=1)
        pooled = self.attention_pooler(tokens).flatten(1)
        hidden_states = F.gelu(self.mlp_hidden(pooled), approximate="tanh")
        log_duration = self.mlp_out(hidden_states).squeeze(-1)
        # The network predicts in log space so its output stays positive and covers
        # short and long durations evenly; exponentiating recovers plain seconds.
        return log_duration.exp()
