import contextlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.logging_utils import get_logger

logger = get_logger(__name__)

# Roles describe what an attention call is for. Each one carries its own kernel policy below,
# because the paths differ in sequence length, key width and whether a mask is present.
VIDEO_SELF = "video_self"
VIDEO_TEXT = "video_text"
AUDIO_TEXT = "audio_text"
SHARED = "shared"
SDPA = "sdpa"
CUDNN = "cudnn"
SAGE = "sage"
SAGE_MODULE = "sageattention"
SAGE_KERNEL = "sageattn_qk_int8_pv_fp8_cuda"
# Settings passed to the kernel on every call. "HND" means the tensors are laid out as
# heads-first, which is the layout the model already uses, so no reshuffling is needed.
SAGE_KERNEL_OPTIONS = {
    "tensor_layout": "HND",
    "qk_quant_gran": "per_warp",
    "pv_accum_dtype": "fp32+fp16",
}
# The GPU capability this wheel is compiled for, written as (major, minor).
SAGE_COMPUTE_CAPABILITY = (12, 0)
# How far the kernel's output may drift from PyTorch's own attention and still be accepted.
# SageAttention trades a little precision for speed on purpose, so the two outputs are never
# bit-for-bit identical. The error is the average absolute difference divided by the average
# size of the reference output. The pinned INT8/FP8 kernel lands around 0.036-0.039 relative error,
# so this limit accepts the kernel we chose and rejects a build whose kernel is noticeably less accurate.
SAGE_ACCEPTANCE_RELATIVE_ERROR = 0.05
# The tensor shape used for the GPU check: (batch, heads, tokens, head size).
SAGE_VERIFICATION_SHAPE = (1, 2, 1024, 128)
# The video-to-text cuDNN path is checked at its engine width: 32 heads of 128 against 1024
# prompt tokens, with a one-row bias expanded across heads and queries like the real call.
CUDNN_VERIFICATION_SHAPE = (2, 32, 1024, 128)
CUDNN_VERIFICATION_KEYS = 1024
# cuDNN attention and PyTorch's own choice for a masked call run the same formula in different
# kernels; anything above this is a different implementation rather than a rounding difference.
CUDNN_ACCEPTANCE_RELATIVE_ERROR = 0.01
# The command that installs and checks the kernel, quoted in error messages.
PREPARE_COMMAND = "uv run python -m scripts.prepare_attention"

WHEEL_DIR = Path(__file__).resolve().parents[2] / "vendor" / "wheels"
MANIFEST_NAME = "manifest.json"

_stage = "stage1"
_cached_sage_kernel = None

def environment_identity():
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_abi": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.machine(),
    }


def read_manifest():
    path = WHEEL_DIR / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


@contextlib.contextmanager
def stage_backends(stage):
    """Run the enclosed denoising stage with the backends that stage is configured for."""
    global _stage
    if stage not in ("stage1", "stage2"):
        raise ValueError(f"Expected 'stage1' or 'stage2', got {stage!r}")
    previous = _stage
    _stage = stage
    try:
        yield
    finally:
        _stage = previous


def resolve_backend(role):
    """The backend a role runs on, given the stage the denoising loop is in."""
    if role == VIDEO_SELF:
        return SAGE if _stage == "stage2" else SDPA
    if role == VIDEO_TEXT:
        return CUDNN
    return SDPA


def attention(query, key, value, attn_mask=None, *, role=SHARED):
    """Run one attention call on the backend the policy assigns to its role."""
    backend = resolve_backend(role)
    if backend == SAGE:
        return sage_attention(query, key, value, attn_mask)
    if backend == CUDNN:
        return cudnn_attention(query, key, value, attn_mask)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )


@torch.library.custom_op("ltx_engine::cudnn_attention", mutates_args=())
def cudnn_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )


@cudnn_attention.register_fake
def _cudnn_attention_meta(query, key, value, attn_mask):
    return torch.empty_like(query)


@torch.library.custom_op("ltx_engine::sage_attention", mutates_args=())
def sage_attention_op(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return sage_kernel()(query, key, value, **SAGE_KERNEL_OPTIONS).contiguous()


@sage_attention_op.register_fake
def _sage_attention_meta(query, key, value):
    return torch.empty(query.shape, dtype=query.dtype, device=query.device)


def sage_attention(query, key, value, attn_mask=None):
    """Run the INT8-query/FP8-value SageAttention kernel on head-first tensors"""
    if attn_mask is not None:
        raise ValueError(
            f"{SAGE_KERNEL} ignores attn_mask, so it cannot run a masked attention call; "
            f"mask of shape {tuple(attn_mask.shape)} reached it."
        )
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"{SAGE_KERNEL} needs bf16 or fp16 input, got {query.dtype}")
    return sage_attention_op(query, key, value)


def sage_kernel():
    """The installed kernel, refusing a package that does not export it"""
    global _cached_sage_kernel
    if _cached_sage_kernel is not None:
        return _cached_sage_kernel
    try:
        module = importlib.import_module(SAGE_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            f"{SAGE_MODULE} is not importable and stage-2 video self-attention has no "
            f"fallback. Run `{PREPARE_COMMAND}` from the project root.\n{exc}"
        ) from exc
    kernel = getattr(module, SAGE_KERNEL, None)
    if kernel is None:
        raise RuntimeError(
            f"{SAGE_MODULE} {getattr(module, '__version__', 'unknown')} does not export "
            f"{SAGE_KERNEL}; rebuild it with `{PREPARE_COMMAND} --force`."
        )
    _cached_sage_kernel = kernel
    return kernel


def verify_attention_backends(device):
    """Prove the pinned kernels run on this device, before any weights are loaded.
    Neither stage-2 video self-attention nor video-to-text attention has a fallback.
    """
    device = torch.device(device)
    if device.type != "cuda":
        logger.warning(
            f"Attention verification skipped on {device}: the SageAttention kernel is CUDA-only "
            "and every path runs on PyTorch's attention without it"
        )
        return {"device": str(device), "verified": False, "reason": "device is not CUDA"}

    capability = torch.cuda.get_device_capability(device)
    if capability != SAGE_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"SageAttention is built for compute capability "
            f"{SAGE_COMPUTE_CAPABILITY[0]}.{SAGE_COMPUTE_CAPABILITY[1]}, but {device} reports "
            f"{capability[0]}.{capability[1]}. Rebuild the wheel with `{PREPARE_COMMAND} --force`."
        )

    manifest = _verified_manifest()
    sage_kernel()
    module = importlib.import_module(SAGE_MODULE)
    batch, heads, tokens, head_dim = SAGE_VERIFICATION_SHAPE
    query = torch.randn(
        (batch, tokens, heads * head_dim), device=device, dtype=torch.bfloat16
    ).unflatten(2, (heads, head_dim)).transpose(1, 2)
    key = torch.randn(
        (batch, tokens, heads * head_dim), device=device, dtype=torch.bfloat16
    ).unflatten(2, (heads, head_dim)).transpose(1, 2)
    value = torch.randn(
        (batch, tokens, heads * head_dim), device=device, dtype=torch.bfloat16
    ).unflatten(2, (heads, head_dim)).transpose(1, 2)
    output = sage_attention_op(query, key, value)
    reference = F.scaled_dot_product_attention(query, key, value)
    if not torch.isfinite(output).all():
        raise RuntimeError(f"{SAGE_KERNEL} produced non-finite output on {device}")
    if tuple(output.shape) != SAGE_VERIFICATION_SHAPE:
        raise RuntimeError(
            f"{SAGE_KERNEL} returned {tuple(output.shape)}, expected "
            f"{SAGE_VERIFICATION_SHAPE}"
        )
    if not output.is_contiguous():
        raise RuntimeError(
            f"{SAGE_KERNEL} returned non-contiguous output strides {tuple(output.stride())}; "
            "its fake kernel requires contiguous output."
        )
    difference = (output.float() - reference.float()).abs()
    relative_error = float(
        difference.mean() / reference.float().abs().mean().clamp_min(1e-8)
    )
    if relative_error > SAGE_ACCEPTANCE_RELATIVE_ERROR:
        raise RuntimeError(
            f"{SAGE_KERNEL} differs from PyTorch's attention by {relative_error:.4f} "
            f"relative, above the {SAGE_ACCEPTANCE_RELATIVE_ERROR} acceptance bound; the "
            "installed kernel is not the one the engine was validated against."
        )

    masked = _verify_masked_attention(device)
    logger.info(
        f"Verified {SAGE_MODULE} {getattr(module, '__version__', 'unknown')} on {device}: "
        f"relative error {relative_error:.6f} against PyTorch attention"
    )
    logger.info(
        f"Verified cuDNN attention on {device}: relative error {masked:.6f} against PyTorch attention"
    )
    return {
        "device": str(device),
        "verified": True,
        "kernel": SAGE_KERNEL,
        "version": getattr(module, "__version__", "unknown"),
        "manifest_built_at_utc": manifest["built_at_utc"],
        "relative_error": round(relative_error, 8),
    }


def _verify_masked_attention(device):
    """Prove the pinned mask path on this device and that it agrees with PyTorch's own choice"""
    batch, heads, tokens, head_dim = CUDNN_VERIFICATION_SHAPE
    query = torch.randn(
        (batch, tokens, heads * head_dim), device=device, dtype=torch.bfloat16
    ).unflatten(2, (heads, head_dim)).transpose(1, 2)
    key = torch.randn((batch, heads, CUDNN_VERIFICATION_KEYS, head_dim), device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.zeros((1, 1, 1, CUDNN_VERIFICATION_KEYS), device=device, dtype=torch.bfloat16).expand(
        batch, heads, tokens, CUDNN_VERIFICATION_KEYS
    )
    max_error = 0.0
    for attn_mask in (mask, None):
        output = cudnn_attention(query, key, value, attn_mask)
        reference = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        if not torch.isfinite(output).all():
            raise RuntimeError(f"cuDNN attention produced non-finite output on {device}")
        if tuple(output.shape) != CUDNN_VERIFICATION_SHAPE:
            raise RuntimeError(
                f"cuDNN attention returned {tuple(output.shape)}, expected {CUDNN_VERIFICATION_SHAPE}"
            )
        if output.stride() != query.stride():
            raise RuntimeError(
                f"cuDNN attention returned strides {tuple(output.stride())} for a query strided "
                f"{tuple(query.stride())}; the fake kernel in this module must declare the layout "
                "the kernel actually writes, or a compiled block stops on its layout assertion."
            )
        relative_error = float(
            (output.float() - reference.float()).abs().mean()
            / reference.float().abs().mean().clamp_min(1e-8)
        )
        if relative_error > CUDNN_ACCEPTANCE_RELATIVE_ERROR:
            raise RuntimeError(
                f"cuDNN attention differs from PyTorch's attention by {relative_error:.4f} "
                f"relative, above the {CUDNN_ACCEPTANCE_RELATIVE_ERROR} acceptance bound; the "
                "pinned video-to-text kernel is not the one the engine was validated against."
            )
        max_error = max(max_error, relative_error)
    return max_error


def _verified_manifest():
    """The manifest, checked against this environment"""
    manifest = read_manifest()
    if manifest is None:
        raise RuntimeError(
            f"No SageAttention manifest at {WHEEL_DIR / MANIFEST_NAME}; rebuild it with "
            f"`{PREPARE_COMMAND} --force`."
        )
    built = manifest.get("environment", {})
    differences = {
        key: {"built_against": built.get(key), "running": value}
        for key, value in environment_identity().items()
        if built.get(key) != value
    }
    if differences:
        raise RuntimeError(
            f"{manifest.get('wheel')} was built for a different environment "
            f"({json.dumps(differences)}); rebuild it with `{PREPARE_COMMAND} --force`."
        )
    return manifest
