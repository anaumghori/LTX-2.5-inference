import contextlib
import contextvars
import importlib
import json
import platform
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

from src.model.logging_utils import get_logger

logger = get_logger(__name__)

# Roles describe what an attention call is for
VIDEO_SELF = "video_self"
SHARED = "shared"
SAGE_MODULE = "sageattention"
SAGE_KERNEL = "sageattn_qk_int8_pv_fp16_cuda"
# Settings passed to the kernel on every call. "HND" means the tensors are laid out as
# heads-first, which is the layout the model already uses, so no reshuffling is needed.
SAGE_KERNEL_OPTIONS = {"tensor_layout": "HND", "pv_accum_dtype": "fp32"}
# The GPU capability this wheel is compiled for, written as (major, minor).
SAGE_COMPUTE_CAPABILITY = (12, 0)
# How far the kernel's output may drift from PyTorch's own attention and still be accepted.
# SageAttention trades a little precision for speed on purpose, so the two outputs are never
# bit-for-bit identical. The error is the average absolute difference divided by the average
# size of the reference output. The pinned INT8/FP16 kernel lands around 0.007, while the FP8
# variant lands around 0.02, so this limit accepts the kernel we chose and rejects a build
# whose kernel is noticeably less accurate.
SAGE_ACCEPTANCE_RELATIVE_ERROR = 0.02
# The tensor shape used for the GPU check: (batch, heads, tokens, head size).
SAGE_VERIFICATION_SHAPE = (1, 2, 1024, 128)
# The command that installs and checks the kernel, quoted in error messages.
PREPARE_COMMAND = "uv run python -m scripts.prepare_attention"

WHEEL_DIR = Path(__file__).resolve().parents[2] / "vendor" / "wheels"
MANIFEST_NAME = "manifest.json"

_stage = contextvars.ContextVar("attention_backend_stage", default="stage1")


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
    if stage not in ("stage1", "stage2"):
        raise ValueError(f"Expected 'stage1' or 'stage2', got {stage!r}")
    token = _stage.set(stage)
    try:
        yield
    finally:
        _stage.reset(token)


def resolve_backend(role):
    """The backend a role runs on, given the stage the denoising loop is in."""
    if role == VIDEO_SELF and _stage.get() == "stage2":
        return SAGE_KERNEL
    return "sdpa"


def attention(query, key, value, attn_mask=None, *, role=SHARED):
    """Run one attention call on the backend the policy assigns to its role."""
    if resolve_backend(role) == "sdpa":
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
    return sage_attention(query, key, value, attn_mask)


def sage_attention(query, key, value, attn_mask=None):
    """Run the INT8-query/FP16-value SageAttention kernel on head-first tensors"""
    if attn_mask is not None:
        raise ValueError(
            f"{SAGE_KERNEL} ignores attn_mask, so it cannot run a masked attention call; "
            f"mask of shape {tuple(attn_mask.shape)} reached it."
        )
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"{SAGE_KERNEL} needs bf16 or fp16 input, got {query.dtype}")
    return sage_kernel()(query, key, value, **SAGE_KERNEL_OPTIONS)


def sage_kernel():
    """The installed kernel, refusing a package that does not export it"""
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
    return kernel


def verify_attention_backends(device):
    """Prove the stage-2 kernel runs on this device, before any weights are loaded.
    There is no fallback for stage-2 video self-attention.
    """
    device = torch.device(device)
    if device.type != "cuda":
        logger.warning(
            "Attention verification skipped on %s: the SageAttention kernel is CUDA-only and "
            "every path runs on PyTorch's attention without it",
            device,
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
    kernel = sage_kernel()
    module = importlib.import_module(SAGE_MODULE)
    query = torch.randn(SAGE_VERIFICATION_SHAPE, device=device, dtype=torch.bfloat16)
    key = torch.randn(SAGE_VERIFICATION_SHAPE, device=device, dtype=torch.bfloat16)
    value = torch.randn(SAGE_VERIFICATION_SHAPE, device=device, dtype=torch.bfloat16)
    output = kernel(query, key, value, **SAGE_KERNEL_OPTIONS)
    reference = F.scaled_dot_product_attention(query, key, value)
    if not torch.isfinite(output).all():
        raise RuntimeError(f"{SAGE_KERNEL} produced non-finite output on {device}")
    if tuple(output.shape) != SAGE_VERIFICATION_SHAPE:
        raise RuntimeError(
            f"{SAGE_KERNEL} returned {tuple(output.shape)}, expected "
            f"{SAGE_VERIFICATION_SHAPE}"
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

    logger.info(
        "Verified %s %s on %s: relative error %.6f against PyTorch attention",
        SAGE_MODULE,
        getattr(module, "__version__", "unknown"),
        device,
        relative_error,
    )
    return {
        "device": str(device),
        "verified": True,
        "kernel": SAGE_KERNEL,
        "version": getattr(module, "__version__", "unknown"),
        "manifest_built_at_utc": manifest["built_at_utc"],
        "relative_error": round(relative_error, 8),
    }


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
