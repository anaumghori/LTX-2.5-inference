import dataclasses
import torch
import torch.nn as nn

from src.device_memory import require_device_capacity
from src.model.lora import LoRAAugmentedLinear, resolve_parent


FP8_DTYPE = torch.float8_e4m3fn
FP8_LIMIT = float(torch.finfo(FP8_DTYPE).max)
ROW_QUANTUM = 16
SCALE_FLOOR = 1.0 / (FP8_LIMIT * 512.0)
WEIGHT_CHUNK_ELEMENTS = 1 << 22

def _row_chunks(rows, columns):
    step = max(1, WEIGHT_CHUNK_ELEMENTS // max(1, columns))
    for start in range(0, rows, step):
        yield start, min(start + step, rows)


def quantize_weight(weight):
    values = weight.detach()
    rows, columns = values.shape
    quantized = torch.empty(values.shape, dtype=FP8_DTYPE, device=values.device)
    row_max = torch.empty(rows, dtype=torch.float32, device=values.device)
    for start, stop in _row_chunks(rows, columns):
        row_max[start:stop] = values[start:stop].float().abs().amax(dim=1)
    scale = (row_max / FP8_LIMIT).clamp(min=SCALE_FLOOR)
    for start, stop in _row_chunks(rows, columns):
        block = values[start:stop].float()
        block.div_(scale[start:stop, None])
        quantized[start:stop] = block.clamp_(-FP8_LIMIT, FP8_LIMIT).to(FP8_DTYPE)
    return quantized, scale.view(1, -1).contiguous()


# (rows, in, out, device) -> (works, reason)
_CAPABILITY = {}


def scaled_mm_supports(rows, in_features, out_features, device):
    """Whether the library accepts one fp8 multiply at this shape, and why not if it does not"""
    key = (int(rows), int(in_features), int(out_features), str(device))
    if key in _CAPABILITY:
        return _CAPABILITY[key]
    weight = torch.zeros((out_features, in_features), dtype=FP8_DTYPE, device=device)
    left = torch.zeros((rows, in_features), dtype=FP8_DTYPE, device=device)
    scale_left = torch.ones((rows, 1), dtype=torch.float32, device=device)
    scale_right = torch.ones((1, out_features), dtype=torch.float32, device=device)
    try:
        torch._scaled_mm(
            left,
            weight.t(),
            scale_a=scale_left,
            scale_b=scale_right,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
    except RuntimeError as error:
        result = (False, f"{type(error).__name__}: {error}")
    else:
        result = (True, None)
    del left, weight
    _CAPABILITY[key] = result
    return result


def padded_rows(rows):
    """The row count a call is padded up to, because the kernel requires a multiple of 16."""
    return rows if rows % ROW_QUANTUM == 0 else (rows + ROW_QUANTUM - 1) // ROW_QUANTUM * ROW_QUANTUM


class Fp8Linear(nn.Module):
    """A bf16 linear replaced by one fp8 multiply, with its scales fixed at load."""
    def __init__(self, base):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        weight, scale = quantize_weight(base.weight)
        self.register_buffer("weight_fp8", weight, persistent=False)
        self.register_buffer("weight_scale", scale, persistent=False)
        if base.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(base.bias.detach().to(torch.bfloat16).clone())

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"

    def forward(self, values):
        """Quantise the input, multiply on the fp8 tensor cores, add the bias, return bf16"""
        if values.dtype is not torch.bfloat16:
            raise RuntimeError(f"Fp8Linear expects a bfloat16 input, got {values.dtype}.")
        shape = values.shape
        if shape[-1] != self.in_features:
            raise RuntimeError(
                f"Fp8Linear expects a last dimension of {self.in_features}, got {tuple(shape)}."
            )
        rows = values.numel() // self.in_features
        flat = values.reshape(rows, self.in_features)
        # One scale per token: the e4m3 cast saturates, so a quotient above the limit becomes the
        # largest representable magnitude without a separate clamp pass over the activation.
        scale = (flat.abs().amax(dim=-1, keepdim=True).float() / FP8_LIMIT).clamp(min=SCALE_FLOOR)
        quantized = torch.div(
            flat, scale.to(flat.dtype), out=torch.empty_like(flat)
        ).to(FP8_DTYPE)
        target = padded_rows(rows)
        if target != rows:
            quantized = torch.cat(
                (quantized, quantized.new_zeros((target - rows, self.in_features)))
            )
            scale = torch.cat(
                (scale, torch.ones((target - rows, 1), dtype=scale.dtype, device=scale.device))
            )
        output = torch._scaled_mm(
            quantized,
            self.weight_fp8.t(),
            scale_a=scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        output = output[:rows]
        if self.bias is not None:
            output = output + self.bias
        return output.view(*shape[:-1], self.out_features)


def _classify(path):
    """The block index and block-relative name of a module path, or (None, path) outside a block."""
    parts = path.split(".")
    if len(parts) > 2 and parts[0] == "transformer_blocks" and parts[1].isdigit():
        return int(parts[1]), ".".join(parts[2:])
    return None, path


@dataclasses.dataclass(frozen=True)
class _PlannedLayer:
    path: str
    wrapped: bool
    weight_bytes: int
    in_features: int
    out_features: int


def _conversion_headroom(planned):
    """Bytes the conversion needs on top of the bf16 layers that are still resident.
    One layer is converted at a time, so the requirement never grows past a single layer"""
    scratch = WEIGHT_CHUNK_ELEMENTS * 4 * 3
    headroom = 0
    for entry in planned:
        probe = (
            ROW_QUANTUM * entry.in_features
            + entry.out_features * entry.in_features
            + ROW_QUANTUM * entry.out_features * 2
        )
        headroom = max(headroom, entry.weight_bytes // 2 + probe + scratch)
    return headroom


def linear_plan(root):
    """Every linear a conversion could touch, as (path, held by a LoRA wrapper)"""
    wrappers = {
        path for path, module in root.named_modules() if isinstance(module, LoRAAugmentedLinear)
    }
    plan = []
    for path, module in root.named_modules():
        if not path:
            continue
        if isinstance(module, LoRAAugmentedLinear):
            plan.append((path, True))
            continue
        if not isinstance(module, nn.Linear):
            continue
        # A LoRA wrapper's own base was taken above; walking into it again would convert the
        # same weights twice.
        if path.endswith(".base") and path[: -len(".base")] in wrappers:
            continue
        plan.append((path, False))
    return plan


def apply_fp8(transformer, layers):
    """Convert this half's selected layers in place and report what the conversion cost"""
    selected_blocks = list(transformer.block_range)
    if not selected_blocks:
        raise RuntimeError("this half owns no transformer block, so there is nothing to convert.")
    planned = []
    kept = 0
    for path, wrapped in linear_plan(transformer):
        module = transformer.get_submodule(path)
        base = module.base if wrapped else module
        if isinstance(base, Fp8Linear):
            raise RuntimeError(f"{path} is already fp8; a model may only be converted once.")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{path} holds {type(base).__name__}, not a Linear.")
        block_index, relative = _classify(path)
        if block_index not in selected_blocks or relative not in layers:
            kept += 1
            continue
        if not base.weight.is_floating_point() or base.weight.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            raise TypeError(
                f"{path} holds a {base.weight.dtype} weight, which an fp8 layer cannot be built from."
            )
        if base.in_features % ROW_QUANTUM or base.out_features % ROW_QUANTUM:
            raise RuntimeError(
                f"{path} is {base.in_features}x{base.out_features}, which is not a multiple of "
                f"{ROW_QUANTUM}; torch._scaled_mm cannot multiply it, and leaving it in bf16 "
                f"would silently mix two precisions in one block."
            )
        planned.append(
            _PlannedLayer(
                path,
                wrapped,
                base.weight.numel() * base.weight.element_size(),
                base.in_features,
                base.out_features,
            )
        )

    if not planned:
        raise RuntimeError(
            f"the fp8 conversion matched no linear in blocks "
            f"[{selected_blocks[0]}, {selected_blocks[-1] + 1}); the layers the engine's fp8 "
            f"setting selects do not match this model."
        )
    device = next(transformer.parameters()).device
    require_device_capacity(
        device,
        _conversion_headroom(planned),
        f"converting {len(planned)} layers to fp8",
        hint=(
            "The fp8 weights have to fit beside the bf16 layer they replace until that original "
            "is dropped, so this device has no room to convert in place."
        ),
    )

    bytes_before = 0
    bytes_after = 0
    for entry in planned:
        works, reason = scaled_mm_supports(
            ROW_QUANTUM, entry.in_features, entry.out_features, device
        )
        if not works:
            raise RuntimeError(
                f"torch._scaled_mm cannot multiply {entry.in_features}x{entry.out_features} on "
                f"{device}, which {entry.path} needs: {reason}"
            )
        module = transformer.get_submodule(entry.path)
        base = module.base if entry.wrapped else module
        layer = Fp8Linear(base)
        if entry.wrapped:
            module.base = layer
        else:
            holder, attribute = resolve_parent(transformer, entry.path)
            setattr(holder, attribute, layer)
        bytes_before += entry.weight_bytes
        bytes_after += (
            layer.weight_fp8.numel() * layer.weight_fp8.element_size()
            + layer.weight_scale.numel() * 4
        )
        del base, module, layer

    return {
        "blocks": [selected_blocks[0], selected_blocks[-1] + 1],
        "converted": len(planned),
        "kept": kept,
        "weight_gb_before": round(bytes_before / 1024**3, 4),
        "weight_gb_after": round(bytes_after / 1024**3, 4),
    }
