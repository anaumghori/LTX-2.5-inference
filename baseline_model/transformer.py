import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_model.common import PixArtAlphaCombinedTimestepSizeEmbeddings

def apply_interleaved_rotary_emb(x, freqs):
    """Rotate paired features by the given rotary encodings.
    The result is cast back to the input precision afterwards."""
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def apply_split_rotary_emb(x, freqs):
    """Apply rotary encodings by rotating two halves of the feature axis, reshaping
    per-head inputs as needed and restoring them to their original layout."""
    cos, sin = freqs
    x_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        b, h, t, _ = cos.shape
        x = x.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True
    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"Expected x.shape[-1] to be even for split rotary, got {last}.")
    r = last // 2
    split_x = x.reshape(*x.shape[:-1], 2, r).float()
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)
    out = split_x * cos_u
    first_out = out[..., :1, :]
    second_out = out[..., 1:, :]
    first_out.addcmul_(-sin_u, second_x)
    second_out.addcmul_(sin_u, first_x)
    out = out.reshape(*out.shape[:-2], last)
    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(b, t, -1)
    return out.to(dtype=x_dtype)


class PixArtAlphaTextProjection(nn.Module):
    """Project text embeddings into the transformer's working width."""
    def __init__(self, in_features, hidden_size, out_features=None):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        return self.linear_2(hidden_states)


class TanhGELU(nn.Module):
    """Apply a linear projection followed by an approximate GELU activation."""
    def __init__(self, dim_in, dim_out, bias, activation_fn):
        super().__init__()
        if activation_fn != "gelu-approximate":
            raise ValueError(f"Unsupported activation_fn {activation_fn!r}, only 'gelu-approximate' is implemented.")
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate="tanh")


class FeedForward(nn.Module):
    """Expand, activate, and contract features with an approximate GELU in between."""
    def __init__(self, dim, bias, activation_fn):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(TanhGELU(dim, inner_dim, bias=bias, activation_fn=activation_fn))
        self.net.append(nn.Dropout(0.0))
        self.net.append(nn.Linear(inner_dim, dim, bias=bias))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AdaLayerNormSingle(nn.Module):
    """Turn one timestep into the shift, scale, and gate values used for modulation."""
    def __init__(self, embedding_dim, num_mod_params=6):
        super().__init__()
        self.num_mod_params = num_mod_params
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, self.num_mod_params * embedding_dim, bias=True)

    def forward(self, timestep, hidden_dtype=None):
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class LTX2Attention(nn.Module):
    """Multi-head attention with query/key normalization, rotary positions, and guidance hooks.

    The same module serves self-attention, text cross-attention, and cross-modal
    attention. Perturbation inputs let guidance passes weaken or bypass attention
    without changing the module itself.
    """
    def __init__(
        self,
        query_dim,
        heads=8,
        kv_heads=8,
        dim_head=64,
        dropout=0.0,
        bias=True,
        cross_attention_dim=None,
        out_bias=True,
        qk_norm="rms_norm_across_heads",
        norm_eps=1e-6,
        norm_elementwise_affine=True,
        rope_type="interleaved",
        apply_gated_attention=False,
    ):
        super().__init__()
        if qk_norm != "rms_norm_across_heads":
            raise ValueError(f"Only 'rms_norm_across_heads' is supported, got {qk_norm}.")
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.rope_type = rope_type
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head * kv_heads, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_v = torch.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
        perturbation_mask=None,
        all_perturbed=None,
    ):
        """Run attention over the inputs, optionally perturbed for guidance.

        Args:
            hidden_states: Query-side tokens.
            encoder_hidden_states: Key/value tokens. Defaults to self-attention.
            attention_mask: Additive mask broadcast across heads and queries.
            query_rotary_emb: Rotary encodings for the queries.
            key_rotary_emb: Rotary encodings for the keys. Falls back to the query ones.
            perturbation_mask: Per-token blend between normal output and values.
            all_perturbed: Skip attention entirely and return projected values.

        Returns:
            The attention output after projection and dropout.
        """
        query_len = hidden_states.shape[1]
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            if attention_mask.ndim == 2:
                attention_mask = ((1 - attention_mask) * -10000.0).unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.expand(-1, self.heads, query_len, -1)
            elif attention_mask.ndim == 3 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask.unsqueeze(2).expand(-1, self.heads, query_len, -1)
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1).expand(-1, self.heads, -1, -1)
            elif attention_mask.ndim == 4 and attention_mask.shape[1] == 1:
                attention_mask = attention_mask.expand(-1, self.heads, -1, -1)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(hidden_states)
        value = self.to_v(encoder_hidden_states)
        if perturbation_mask is not None:
            if all_perturbed is None:
                all_perturbed = bool(torch.all(perturbation_mask == 0))
            if all_perturbed:
                # A fully perturbed pass skips attention and returns the values, which
                # is what spatiotemporal guidance compares against the normal output.
                hidden_states = value
                if self.to_gate_logits is not None:
                    hidden_states = hidden_states.unflatten(2, (self.heads, -1))
                    gates = 2.0 * torch.sigmoid(gate_logits)
                    hidden_states = (hidden_states * gates.unsqueeze(-1)).flatten(2, 3)
                hidden_states = self.to_out[0](hidden_states)
                return self.to_out[1](hidden_states)
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        query = self.norm_q(query)
        key = self.norm_k(key)
        if query_rotary_emb is not None:
            if self.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
            elif self.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
        query = query.unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.heads, -1)).transpose(1, 2)
        value_reshaped = value.unflatten(2, (self.heads, -1)).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value_reshaped, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        if perturbation_mask is not None:
            # Partial perturbation blends each token between its values and the full
            # attention result, so guidance can weaken attention instead of removing it.
            value_flat = value_reshaped.transpose(1, 2).flatten(2, 3)
            hidden_states = torch.lerp(value_flat, hidden_states, perturbation_mask)
        if self.to_gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (self.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = (hidden_states * gates.unsqueeze(-1)).flatten(2, 3)
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class LTX2VideoTransformerBlock(nn.Module):
    """One joint video-audio transformer block.

    Each block runs self-attention, text cross-attention, audio-video cross-attention,
    and feed-forward refinement for both modalities, all steered by timestep-derived
    modulation values. The cross-modal attentions can be switched off to isolate
    the modalities for guidance.
    """
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        cross_attention_dim,
        audio_dim,
        audio_num_attention_heads,
        audio_attention_head_dim,
        audio_cross_attention_dim,
        video_gated_attn,
        video_cross_attn_adaln,
        audio_gated_attn,
        audio_cross_attn_adaln,
        qk_norm,
        attention_bias,
        attention_out_bias,
        eps,
        elementwise_affine,
        rope_type,
        perturbed_attn,
        ff_bias,
        audio_ff_bias,
        activation_fn,
    ):
        super().__init__()
        self.perturbed_attn = perturbed_attn
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = LTX2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=video_gated_attn,
        )
        self.audio_norm1 = nn.RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn1 = LTX2Attention(
            query_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=audio_gated_attn,
        )
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=video_gated_attn,
        )
        self.audio_norm2 = nn.RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_attn2 = LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=audio_cross_attention_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=audio_gated_attn,
        )
        self.audio_to_video_norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_to_video_attn = LTX2Attention(
            query_dim=dim,
            cross_attention_dim=audio_dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=video_gated_attn,
        )
        self.video_to_audio_norm = nn.RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.video_to_audio_attn = LTX2Attention(
            query_dim=audio_dim,
            cross_attention_dim=dim,
            heads=audio_num_attention_heads,
            kv_heads=audio_num_attention_heads,
            dim_head=audio_attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            rope_type=rope_type,
            apply_gated_attention=audio_gated_attn,
        )
        self.norm3 = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.ff = FeedForward(dim, bias=ff_bias, activation_fn=activation_fn)
        self.audio_norm3 = nn.RMSNorm(audio_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.audio_ff = FeedForward(audio_dim, bias=audio_ff_bias, activation_fn=activation_fn)
        self.video_cross_attn_adaln = video_cross_attn_adaln
        self.audio_cross_attn_adaln = audio_cross_attn_adaln
        video_mod_param_num = 9 if self.video_cross_attn_adaln else 6
        audio_mod_param_num = 9 if self.audio_cross_attn_adaln else 6
        self.scale_shift_table = nn.Parameter(torch.randn(video_mod_param_num, dim) / dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(audio_mod_param_num, audio_dim) / audio_dim**0.5)
        self.cross_attn_adaln = video_cross_attn_adaln or audio_cross_attn_adaln
        if self.cross_attn_adaln:
            self.prompt_scale_shift_table = nn.Parameter(torch.randn(2, dim))
            self.audio_prompt_scale_shift_table = nn.Parameter(torch.randn(2, audio_dim))
        self.video_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, dim))
        self.audio_a2v_cross_attn_scale_shift_table = nn.Parameter(torch.randn(5, audio_dim))

    @staticmethod
    def get_mod_params(scale_shift_table, temb, batch_size):
        """Combine a learned table with timestep embeddings into modulation values"""
        num_ada_params = scale_shift_table.shape[0]
        ada_values = scale_shift_table[None, None].to(temb.device) + temb.reshape(
            batch_size, temb.shape[1], num_ada_params, -1
        )
        return ada_values.unbind(dim=2)

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        temb,
        temb_audio,
        temb_ca_scale_shift,
        temb_ca_audio_scale_shift,
        temb_ca_gate,
        temb_ca_audio_gate,
        temb_prompt=None,
        temb_prompt_audio=None,
        video_rotary_emb=None,
        audio_rotary_emb=None,
        ca_video_rotary_emb=None,
        ca_audio_rotary_emb=None,
        encoder_attention_mask=None,
        audio_encoder_attention_mask=None,
        self_attention_mask=None,
        audio_self_attention_mask=None,
        a2v_cross_attention_mask=None,
        v2a_cross_attention_mask=None,
        use_a2v_cross_attention=True,
        use_v2a_cross_attention=True,
        perturbation_mask=None,
        all_perturbed=None,
    ):
        """Run one joint video-audio refinement step.

        Video and audio streams pass through self-attention, text cross-attention,
        cross-modal attention and feed-forward layers in that order, each steered
        by its own timestep modulation. Perturbation inputs only take effect in
        blocks built with perturbed attention, which is how guidance compares
        normal and weakened attention.

        Args:
            hidden_states: Video tokens.
            audio_hidden_states: Audio tokens.
            encoder_hidden_states: Video text conditioning.
            audio_encoder_hidden_states: Audio text conditioning.
            temb: Video timestep modulation values.
            temb_audio: Audio timestep modulation values.
            temb_ca_scale_shift: Cross-modal scale/shift modulation for video.
            temb_ca_audio_scale_shift: Cross-modal scale/shift modulation for audio.
            temb_ca_gate: Cross-modal gate modulation for video.
            temb_ca_audio_gate: Cross-modal gate modulation for audio.
            temb_prompt: Optional prompt modulation for video.
            temb_prompt_audio: Optional prompt modulation for audio.
            video_rotary_emb: Rotary encodings for video self-attention.
            audio_rotary_emb: Rotary encodings for audio self-attention.
            ca_video_rotary_emb: Rotary encodings for video cross-modal attention.
            ca_audio_rotary_emb: Rotary encodings for audio cross-modal attention.
            encoder_attention_mask: Mask for video text attention.
            audio_encoder_attention_mask: Mask for audio text attention.
            self_attention_mask: Mask for video self-attention.
            audio_self_attention_mask: Mask for audio self-attention.
            a2v_cross_attention_mask: Mask for audio-to-video attention.
            v2a_cross_attention_mask: Mask for video-to-audio attention.
            use_a2v_cross_attention: Disable to isolate video from audio.
            use_v2a_cross_attention: Disable to isolate audio from video.
            perturbation_mask: Per-token attention weakening for guidance.
            all_perturbed: Bypass attention entirely for guidance.

        Returns:
            The refined video and audio tokens.
        """
        batch_size = hidden_states.size(0)
        video_ada_params = self.get_mod_params(self.scale_shift_table, temb, batch_size)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = video_ada_params[:6]
        if self.video_cross_attn_adaln:
            shift_text_q, scale_text_q, gate_text_q = video_ada_params[6:9]
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        video_self_attn_args = {
            "hidden_states": norm_hidden_states,
            "encoder_hidden_states": None,
            "query_rotary_emb": video_rotary_emb,
            "attention_mask": self_attention_mask,
        }
        if self.perturbed_attn:
            video_self_attn_args["perturbation_mask"] = perturbation_mask
            video_self_attn_args["all_perturbed"] = all_perturbed
        attn_hidden_states = self.attn1(**video_self_attn_args)
        hidden_states = hidden_states + attn_hidden_states * gate_msa
        audio_ada_params = self.get_mod_params(self.audio_scale_shift_table, temb_audio, batch_size)
        (
            audio_shift_msa,
            audio_scale_msa,
            audio_gate_msa,
            audio_shift_mlp,
            audio_scale_mlp,
            audio_gate_mlp,
        ) = audio_ada_params[:6]
        if self.audio_cross_attn_adaln:
            audio_shift_text_q, audio_scale_text_q, audio_gate_text_q = audio_ada_params[6:9]
        norm_audio_hidden_states = self.audio_norm1(audio_hidden_states)
        norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_msa) + audio_shift_msa
        audio_self_attn_args = {
            "hidden_states": norm_audio_hidden_states,
            "encoder_hidden_states": None,
            "query_rotary_emb": audio_rotary_emb,
            "attention_mask": audio_self_attention_mask,
        }
        if self.perturbed_attn:
            audio_self_attn_args["perturbation_mask"] = perturbation_mask
            audio_self_attn_args["all_perturbed"] = all_perturbed
        attn_audio_hidden_states = self.audio_attn1(**audio_self_attn_args)
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states * audio_gate_msa
        if self.cross_attn_adaln:
            if temb_prompt is not None:
                shift_text_kv, scale_text_kv = self.get_mod_params(
                    self.prompt_scale_shift_table, temb_prompt, batch_size
                )
            else:
                # Without prompt modulation the table values are used as-is, giving a
                # fixed conditioning shift learned in training.
                shift_text_kv, scale_text_kv = self.prompt_scale_shift_table[None, None].to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                ).unbind(dim=2)
            if temb_prompt_audio is not None:
                audio_shift_text_kv, audio_scale_text_kv = self.get_mod_params(
                    self.audio_prompt_scale_shift_table, temb_prompt_audio, batch_size
                )
            else:
                audio_shift_text_kv, audio_scale_text_kv = self.audio_prompt_scale_shift_table[None, None].to(
                    device=audio_hidden_states.device, dtype=audio_hidden_states.dtype
                ).unbind(dim=2)
        norm_hidden_states = self.norm2(hidden_states)
        if self.video_cross_attn_adaln:
            norm_hidden_states = norm_hidden_states * (1 + scale_text_q) + shift_text_q
        if self.cross_attn_adaln:
            encoder_hidden_states = encoder_hidden_states * (1 + scale_text_kv) + shift_text_kv
        attn_hidden_states = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=encoder_attention_mask,
        )
        if self.video_cross_attn_adaln:
            attn_hidden_states = attn_hidden_states * gate_text_q
        hidden_states = hidden_states + attn_hidden_states
        norm_audio_hidden_states = self.audio_norm2(audio_hidden_states)
        if self.audio_cross_attn_adaln:
            norm_audio_hidden_states = norm_audio_hidden_states * (1 + audio_scale_text_q) + audio_shift_text_q
        if self.cross_attn_adaln:
            audio_encoder_hidden_states = audio_encoder_hidden_states * (1 + audio_scale_text_kv) + audio_shift_text_kv
        attn_audio_hidden_states = self.audio_attn2(
            norm_audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            query_rotary_emb=None,
            attention_mask=audio_encoder_attention_mask,
        )
        if self.audio_cross_attn_adaln:
            attn_audio_hidden_states = attn_audio_hidden_states * audio_gate_text_q
        audio_hidden_states = audio_hidden_states + attn_audio_hidden_states
        if use_a2v_cross_attention or use_v2a_cross_attention:
            norm_hidden_states = self.audio_to_video_norm(hidden_states)
            norm_audio_hidden_states = self.video_to_audio_norm(audio_hidden_states)
            video_per_layer_ca_scale_shift = self.video_a2v_cross_attn_scale_shift_table[:4, :]
            video_per_layer_ca_gate = self.video_a2v_cross_attn_scale_shift_table[4:, :]
            video_ca_ada_params = self.get_mod_params(video_per_layer_ca_scale_shift, temb_ca_scale_shift, batch_size)
            video_ca_gate_param = self.get_mod_params(video_per_layer_ca_gate, temb_ca_gate, batch_size)
            video_a2v_ca_scale, video_a2v_ca_shift, video_v2a_ca_scale, video_v2a_ca_shift = video_ca_ada_params
            a2v_gate = video_ca_gate_param[0].squeeze(2)
            audio_per_layer_ca_scale_shift = self.audio_a2v_cross_attn_scale_shift_table[:4, :]
            audio_per_layer_ca_gate = self.audio_a2v_cross_attn_scale_shift_table[4:, :]
            audio_ca_ada_params = self.get_mod_params(
                audio_per_layer_ca_scale_shift, temb_ca_audio_scale_shift, batch_size
            )
            audio_ca_gate_param = self.get_mod_params(audio_per_layer_ca_gate, temb_ca_audio_gate, batch_size)
            audio_a2v_ca_scale, audio_a2v_ca_shift, audio_v2a_ca_scale, audio_v2a_ca_shift = audio_ca_ada_params
            v2a_gate = audio_ca_gate_param[0].squeeze(2)
            if use_a2v_cross_attention:
                mod_norm_hidden_states = (
                    norm_hidden_states * (1 + video_a2v_ca_scale.squeeze(2)) + video_a2v_ca_shift.squeeze(2)
                )
                mod_norm_audio_hidden_states = (
                    norm_audio_hidden_states * (1 + audio_a2v_ca_scale.squeeze(2))
                    + audio_a2v_ca_shift.squeeze(2)
                )
                a2v_attn_hidden_states = self.audio_to_video_attn(
                    mod_norm_hidden_states,
                    encoder_hidden_states=mod_norm_audio_hidden_states,
                    query_rotary_emb=ca_video_rotary_emb,
                    key_rotary_emb=ca_audio_rotary_emb,
                    attention_mask=a2v_cross_attention_mask,
                )
                hidden_states = hidden_states + a2v_gate * a2v_attn_hidden_states
            if use_v2a_cross_attention:
                mod_norm_hidden_states = (
                    norm_hidden_states * (1 + video_v2a_ca_scale.squeeze(2)) + video_v2a_ca_shift.squeeze(2)
                )
                mod_norm_audio_hidden_states = (
                    norm_audio_hidden_states * (1 + audio_v2a_ca_scale.squeeze(2))
                    + audio_v2a_ca_shift.squeeze(2)
                )
                v2a_attn_hidden_states = self.video_to_audio_attn(
                    mod_norm_audio_hidden_states,
                    encoder_hidden_states=mod_norm_hidden_states,
                    query_rotary_emb=ca_audio_rotary_emb,
                    key_rotary_emb=ca_video_rotary_emb,
                    attention_mask=v2a_cross_attention_mask,
                )
                audio_hidden_states = audio_hidden_states + v2a_gate * v2a_attn_hidden_states
        norm_hidden_states = self.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp
        norm_audio_hidden_states = self.audio_norm3(audio_hidden_states) * (1 + audio_scale_mlp) + audio_shift_mlp
        audio_ff_output = self.audio_ff(norm_audio_hidden_states)
        audio_hidden_states = audio_hidden_states + audio_ff_output * audio_gate_mlp
        return hidden_states, audio_hidden_states


class RotaryPosEmbed(nn.Module):
    """Build rotary position encodings from spatiotemporal coordinates.

    Latent grid positions are converted to physical units (seconds and pixels),
    normalized against reference dimensions and turned into rotation frequencies
    that attention uses to sense relative position.
    """
    def __init__(
        self,
        dim,
        patch_size=1,
        patch_size_t=1,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        sampling_rate=16000,
        hop_length=160,
        scale_factors=(8, 32, 32),
        theta=10000.0,
        causal_offset=1,
        modality="video",
        double_precision=True,
        rope_type="interleaved",
        num_attention_heads=32,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        if rope_type not in ["interleaved", "split"]:
            raise ValueError(f"{rope_type=} not supported. Choose between 'interleaved' and 'split'.")
        self.rope_type = rope_type
        self.base_num_frames = base_num_frames
        self.num_attention_heads = num_attention_heads
        self.base_height = base_height
        self.base_width = base_width
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.audio_latents_per_second = float(sampling_rate) / float(hop_length) / float(scale_factors[0])
        self.scale_factors = scale_factors
        self.theta = theta
        self.causal_offset = causal_offset
        if modality not in ["video", "audio"]:
            raise ValueError(f"Modality {modality} is not supported.")
        self.modality = modality
        self.double_precision = double_precision

    def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps=24.0):
        """Build start/end coordinate boxes for every packed video token"""
        grid_f = torch.arange(start=0, end=num_frames, step=self.patch_size_t, dtype=torch.float32, device=device)
        grid_h = torch.arange(start=0, end=height, step=self.patch_size, dtype=torch.float32, device=device)
        grid_w = torch.arange(start=0, end=width, step=self.patch_size, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(patch_size, dtype=grid.dtype, device=grid.device)
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)
        latent_coords = torch.stack([grid, patch_ends], dim=-1)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        scale_tensor = torch.tensor(self.scale_factors, device=latent_coords.device)
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)
        # The first frame starts before zero because of the causal shift, so clamp it
        # and express the time axis in seconds for the rotary tables below.
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]).clamp(min=0)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
        return pixel_coords

    def prepare_audio_coords(self, batch_size, num_frames, device, shift=0):
        """Build start/end time boxes for every packed audio token"""
        grid_f = torch.arange(
            start=shift, end=num_frames + shift, step=self.patch_size_t, dtype=torch.float32, device=device
        )
        audio_scale_factor = self.scale_factors[0]
        grid_start_mel = grid_f * audio_scale_factor
        grid_start_mel = (grid_start_mel + self.causal_offset - audio_scale_factor).clip(min=0)
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate
        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (grid_end_mel + self.causal_offset - audio_scale_factor).clip(min=0)
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate
        audio_coords = torch.stack([grid_start_s, grid_end_s], dim=-1)
        audio_coords = audio_coords.unsqueeze(0).expand(batch_size, -1, -1)
        return audio_coords.unsqueeze(1)

    def forward(self, coords, device=None):
        """Turn coordinate boxes into cosine/sine rotary tables"""
        device = device or coords.device
        num_pos_dims = coords.shape[1]
        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)
        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        elif self.modality == "audio":
            max_positions = (self.base_num_frames,)
        grid = torch.stack([coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1).to(device)
        num_rope_elems = num_pos_dims * 2
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                start=0.0, end=1.0, steps=self.dim // num_rope_elems, dtype=freqs_dtype, device=device
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)
        freqs = (grid.unsqueeze(-1) * 2 - 1) * freqs
        freqs = freqs.transpose(-1, -2).flatten(2)
        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)
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


class VideoTransformer(nn.Module):
    """Joint video-audio diffusion transformer.

    Packed video and audio latents are projected inward, conditioned on text and
    timestep, refined through the joint blocks, then projected back to velocity
    predictions for both modalities.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        inner_dim = config.num_attention_heads * config.attention_head_dim
        audio_inner_dim = config.audio_num_attention_heads * config.audio_attention_head_dim
        out_channels = config.out_channels or config.in_channels
        audio_out_channels = config.audio_out_channels or config.audio_in_channels
        self.proj_in = nn.Linear(config.in_channels, inner_dim)
        self.audio_proj_in = nn.Linear(config.audio_in_channels, audio_inner_dim)
        if config.use_keyframes_abs_pos_embedding:
            self.keyframes_abs_pos_embedding = nn.Parameter(torch.zeros(1, inner_dim))
        if config.use_prompt_embeddings:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=config.caption_channels, hidden_size=inner_dim
            )
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=config.caption_channels, hidden_size=audio_inner_dim
            )
        self.prompt_modulation = config.cross_attn_mod or config.audio_cross_attn_mod
        video_time_emb_mod_params = 9 if config.cross_attn_mod else 6
        audio_time_emb_mod_params = 9 if config.audio_cross_attn_mod else 6
        self.time_embed = AdaLayerNormSingle(inner_dim, num_mod_params=video_time_emb_mod_params)
        self.audio_time_embed = AdaLayerNormSingle(audio_inner_dim, num_mod_params=audio_time_emb_mod_params)
        self.av_cross_attn_video_scale_shift = AdaLayerNormSingle(inner_dim, num_mod_params=4)
        self.av_cross_attn_audio_scale_shift = AdaLayerNormSingle(audio_inner_dim, num_mod_params=4)
        self.av_cross_attn_video_a2v_gate = AdaLayerNormSingle(inner_dim, num_mod_params=1)
        self.av_cross_attn_audio_v2a_gate = AdaLayerNormSingle(audio_inner_dim, num_mod_params=1)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.audio_scale_shift_table = nn.Parameter(torch.randn(2, audio_inner_dim) / audio_inner_dim**0.5)
        if self.prompt_modulation and config.use_prompt_adaln_single:
            self.prompt_adaln = AdaLayerNormSingle(inner_dim, num_mod_params=2)
            self.audio_prompt_adaln = AdaLayerNormSingle(audio_inner_dim, num_mod_params=2)
        self.rope = RotaryPosEmbed(
            dim=inner_dim,
            patch_size=config.patch_size,
            patch_size_t=config.patch_size_t,
            base_num_frames=config.pos_embed_max_pos,
            base_height=config.base_height,
            base_width=config.base_width,
            scale_factors=config.vae_scale_factors,
            theta=config.rope_theta,
            causal_offset=config.causal_offset,
            modality="video",
            double_precision=config.rope_double_precision,
            rope_type=config.rope_type,
            num_attention_heads=config.num_attention_heads,
        )
        self.audio_rope = RotaryPosEmbed(
            dim=audio_inner_dim,
            patch_size=config.audio_patch_size,
            patch_size_t=config.audio_patch_size_t,
            base_num_frames=config.audio_pos_embed_max_pos,
            sampling_rate=config.audio_sampling_rate,
            hop_length=config.audio_hop_length,
            scale_factors=[config.audio_scale_factor],
            theta=config.rope_theta,
            causal_offset=config.causal_offset,
            modality="audio",
            double_precision=config.rope_double_precision,
            rope_type=config.rope_type,
            num_attention_heads=config.audio_num_attention_heads,
        )
        cross_attn_pos_embed_max_pos = max(config.pos_embed_max_pos, config.audio_pos_embed_max_pos)
        self.cross_attn_rope = RotaryPosEmbed(
            dim=config.audio_cross_attention_dim,
            patch_size=config.patch_size,
            patch_size_t=config.patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            base_height=config.base_height,
            base_width=config.base_width,
            theta=config.rope_theta,
            causal_offset=config.causal_offset,
            modality="video",
            double_precision=config.rope_double_precision,
            rope_type=config.rope_type,
            num_attention_heads=config.num_attention_heads,
        )
        self.cross_attn_audio_rope = RotaryPosEmbed(
            dim=config.audio_cross_attention_dim,
            patch_size=config.audio_patch_size,
            patch_size_t=config.audio_patch_size_t,
            base_num_frames=cross_attn_pos_embed_max_pos,
            sampling_rate=config.audio_sampling_rate,
            hop_length=config.audio_hop_length,
            theta=config.rope_theta,
            causal_offset=config.causal_offset,
            modality="audio",
            double_precision=config.rope_double_precision,
            rope_type=config.rope_type,
            num_attention_heads=config.audio_num_attention_heads,
        )
        self.transformer_blocks = nn.ModuleList([
            LTX2VideoTransformerBlock(
                dim=inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                cross_attention_dim=config.cross_attention_dim,
                audio_dim=audio_inner_dim,
                audio_num_attention_heads=config.audio_num_attention_heads,
                audio_attention_head_dim=config.audio_attention_head_dim,
                audio_cross_attention_dim=config.audio_cross_attention_dim,
                video_gated_attn=config.gated_attn,
                video_cross_attn_adaln=config.cross_attn_mod,
                audio_gated_attn=config.audio_gated_attn,
                audio_cross_attn_adaln=config.audio_cross_attn_mod,
                qk_norm=config.qk_norm,
                attention_bias=config.attention_bias,
                attention_out_bias=config.attention_out_bias,
                eps=config.norm_eps,
                elementwise_affine=config.norm_elementwise_affine,
                rope_type=config.rope_type,
                perturbed_attn=config.perturbed_attn,
                ff_bias=config.ff_bias,
                audio_ff_bias=config.audio_ff_bias,
                activation_fn=config.activation_fn,
            )
            for _ in range(config.num_layers)
        ])
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)
        self.audio_norm_out = nn.LayerNorm(audio_inner_dim, eps=1e-6, elementwise_affine=False)
        self.audio_proj_out = nn.Linear(audio_inner_dim, audio_out_channels)

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        audio_timestep=None,
        sigma=None,
        audio_sigma=None,
        encoder_attention_mask=None,
        audio_encoder_attention_mask=None,
        num_frames=None,
        height=None,
        width=None,
        fps=24.0,
        audio_num_frames=None,
        video_coords=None,
        audio_coords=None,
        isolate_modalities=False,
        spatio_temporal_guidance_blocks=None,
        perturbation_mask=None,
        use_cross_timestep=False,
        video_self_attention_mask=None,
        video_keyframes_mask=None,
    ):
        """Denoise packed video and audio latents by one transformer pass.

        Args:
            hidden_states: Packed video latents.
            audio_hidden_states: Packed audio latents.
            encoder_hidden_states: Video text conditioning.
            audio_encoder_hidden_states: Audio text conditioning.
            timestep: Video timestep per batch entry.
            audio_timestep: Audio timestep per batch entry. Falls back to the video one.
            sigma: Video sigma per batch entry, used for prompt modulation.
            audio_sigma: Audio sigma per batch entry. Falls back to the video one.
            encoder_attention_mask: Mask for video text attention.
            audio_encoder_attention_mask: Mask for audio text attention.
            num_frames: Latent video frame count, used when building coordinates.
            height: Latent video height, used when building coordinates.
            width: Latent video width, used when building coordinates.
            fps: Frame rate used for the time coordinates.
            audio_num_frames: Latent audio frame count, used when building coordinates.
            video_coords: Prebuilt video coordinates. Built when omitted.
            audio_coords: Prebuilt audio coordinates. Built when omitted.
            isolate_modalities: Disable cross-modal attention for guidance.
            spatio_temporal_guidance_blocks: Block indices running perturbed attention.
            perturbation_mask: Per-token attention weakening for guidance.
            use_cross_timestep: Modulate each modality with the other stream's time.
            video_self_attention_mask: Mask for video self-attention.
            video_keyframes_mask: Marks keyframe tokens for extra position signal.

        Returns:
            The video and audio velocity predictions.
        """
        config = self.config
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = audio_sigma if audio_sigma is not None else sigma
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)
        if video_self_attention_mask is not None:
            video_self_attention_mask = (1 - video_self_attention_mask.to(hidden_states.dtype)) * -10000.0
        batch_size = hidden_states.size(0)
        if video_coords is None:
            video_coords = self.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
        if audio_coords is None:
            audio_coords = self.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, audio_hidden_states.device
            )
        video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)
        video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device)
        audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )
        hidden_states = self.proj_in(hidden_states)
        audio_hidden_states = self.audio_proj_in(audio_hidden_states)
        if config.use_keyframes_abs_pos_embedding and video_keyframes_mask is not None:
            # Keyframe tokens get an extra learned marker so the model can tell
            # reference frames apart from freely generated ones.
            marker = (video_keyframes_mask > 0).to(dtype=hidden_states.dtype)
            hidden_states = hidden_states + marker * self.keyframes_abs_pos_embedding.to(dtype=hidden_states.dtype)
        timestep_cross_attn_gate_scale_factor = (
            config.cross_attn_timestep_scale_multiplier / config.timestep_scale_multiplier
        )
        temb, embedded_timestep = self.time_embed(timestep.flatten(), hidden_dtype=hidden_states.dtype)
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))
        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(), hidden_dtype=audio_hidden_states.dtype
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1, audio_embedded_timestep.size(-1))
        if self.prompt_modulation and config.use_prompt_adaln_single:
            temb_prompt, _ = self.prompt_adaln(sigma.flatten(), hidden_dtype=hidden_states.dtype)
            temb_prompt_audio, _ = self.audio_prompt_adaln(
                audio_sigma.flatten(), hidden_dtype=audio_hidden_states.dtype
            )
            temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
            temb_prompt_audio = temb_prompt_audio.view(batch_size, -1, temb_prompt_audio.size(-1))
        else:
            temb_prompt = temb_prompt_audio = None
        video_ca_timestep = audio_sigma.flatten() if use_cross_timestep else timestep.flatten()
        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            video_ca_timestep, hidden_dtype=hidden_states.dtype
        )
        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            video_ca_timestep * timestep_cross_attn_gate_scale_factor,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(batch_size, -1, video_cross_attn_scale_shift.shape[-1])
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(batch_size, -1, video_cross_attn_a2v_gate.shape[-1])
        audio_ca_timestep = sigma.flatten() if use_cross_timestep else audio_timestep.flatten()
        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            audio_ca_timestep, hidden_dtype=audio_hidden_states.dtype
        )
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(batch_size, -1, audio_cross_attn_scale_shift.shape[-1])
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(batch_size, -1, audio_cross_attn_v2a_gate.shape[-1])
        if config.use_prompt_embeddings:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))
            audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(batch_size, -1, audio_hidden_states.size(-1))
        spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks or []
        if len(spatio_temporal_guidance_blocks) > 0 and perturbation_mask is None:
            # Requesting guidance blocks without an explicit mask means perturbing
            # every batch entry, so start from an all-perturbed default mask.
            perturbation_mask = torch.zeros((batch_size,))
        if perturbation_mask is not None and perturbation_mask.ndim == 1:
            perturbation_mask = perturbation_mask[:, None, None]
        all_perturbed = bool(torch.all(perturbation_mask == 0)) if perturbation_mask is not None else False
        stg_blocks = set(spatio_temporal_guidance_blocks)
        for block_idx, block in enumerate(self.transformer_blocks):
            block_perturbation_mask = perturbation_mask if block_idx in stg_blocks else None
            block_all_perturbed = all_perturbed if block_idx in stg_blocks else False
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=video_cross_attn_scale_shift,
                temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                temb_ca_gate=video_cross_attn_a2v_gate,
                temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                temb_prompt=temb_prompt,
                temb_prompt_audio=temb_prompt_audio,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=video_cross_attn_rotary_emb,
                ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
                self_attention_mask=video_self_attention_mask,
                audio_self_attention_mask=None,
                a2v_cross_attention_mask=None,
                v2a_cross_attention_mask=None,
                use_a2v_cross_attention=not isolate_modalities,
                use_v2a_cross_attention=not isolate_modalities,
                perturbation_mask=block_perturbation_mask,
                all_perturbed=block_all_perturbed,
            )
        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)
        audio_scale_shift_values = self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
        audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]
        audio_hidden_states = self.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = self.audio_proj_out(audio_hidden_states)
        return output, audio_output
