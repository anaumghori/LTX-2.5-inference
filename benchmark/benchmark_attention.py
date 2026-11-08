"""This answers one question: which attention backend should each attention call in the model run on.

Configurations
--------------
Every configuration below runs on every launch; no option selects or skips one. "PyTorch automatic"
means PyTorch's own dispatcher picks the kernel (whatever the shape, dtype and mask allow), while a
forced entry pins a single PyTorch kernel. A wiring's name is its stage-1 backend followed by its
stage-2 backend.

    wiring                 stage 1                          stage 2
    ---------------------  -------------------------------  -------------------------------
    py__py                 PyTorch automatic (reference)    PyTorch automatic
    py__sage_fp16          PyTorch automatic                SageAttention INT8-q / FP16-v
    py__sage_fp8           PyTorch automatic                SageAttention INT8-q / FP8-v
    sage_fp16__py          SageAttention INT8-q / FP16-v     PyTorch automatic
    sage_fp16__sage_fp16   SageAttention INT8-q / FP16-v     SageAttention INT8-q / FP16-v
    sage_fp16__sage_fp8    SageAttention INT8-q / FP16-v     SageAttention INT8-q / FP8-v
    sage_fp8__py           SageAttention INT8-q / FP8-v      PyTorch automatic
    sage_fp8__sage_fp16    SageAttention INT8-q / FP8-v      SageAttention INT8-q / FP16-v
    sage_fp8__sage_fp8     SageAttention INT8-q / FP8-v      SageAttention INT8-q / FP8-v
    cudnn__cudnn           PyTorch cuDNN attention           PyTorch cuDNN attention

Every wiring runs on the same optimised model body; only the two attention backends differ. The
reference wiring is what the others are compared against for speed and output quality. Nothing here
checks the model against the frozen baseline: those are two different bodies. 

Run the code:

    uv run -m benchmark.benchmark_attention

Options:

    --checkpoint-dir     Folder holding the checkpoint. If omitted, the checkpoint is
                         downloaded from Hugging Face, which needs the token in .env.

    --output-json        Where the JSON report is written. Defaults to
                         benchmark/results/attention/benchmark_attention_results.json.

    --save-media         Write each wiring's first generated video and audio to
                         benchmark/results/attention/. Off by default, so nothing is written unless
                         this is passed. LTX_BENCHMARK_MEDIA_DIR overrides the directory, which the
                         Modal runner uses to write somewhere the mounted sources allow.

    --runs               Timed generations per wiring. Default 3.

    --microbench-reps    Repetitions per kernel-table case after warm-up. Default 20.


What We Measure
---------------
- The warm-up. One generation on the reference wiring, discarded.

- The kernel table. For every captured shape (six paths, three stage-1 passes and the stage-2 pass)
each available backend runs on tensors with the real layout. Recorded per case: median, minimum,
maximum and the full series of device milliseconds, effective teraflops, peak allocated memory,
whether repeated calls agree bit for bit, the launched kernel names with their times, and the
absolute, relative and cosine distance from PyTorch's own answer for the same inputs. The layout
each shape is measured on (shapes, strides, dtype, contiguity) is recorded once per shape, because
a kernel's speed depends on how the tensors are laid out and not only on their dimensions. Only
the unmasked case is measured: nothing in this model ever passes an attention mask.

- The wirings. Every wiring runs on every launch.

- The GPU's clock, power draw and temperature, sampled before and after every generation. A card
that is power-capped or thermally slowed then shows up in the report.

- One extra generation per wiring is profiled with PyTorch's own profiler. It is excluded from the
latency statistics because the profiler inflates wall time

What We Do Not Measure
----------------------
- Model loading: excluded from generation latency but timed and reported separately.
- The warm-up generation: discarded.
- The connector attention and the duration head's pooler.
- The reference-image path: image encoding and frame-0 pinning never run.
- Batching. Every generation is one request.

Results
-------
The JSON report is written atomically to `--output-json` and contains:

- Status, errors, warnings, the run configuration and the environment.
- Environment identity: GPU, driver, torch, CUDA, cuDNN, SageAttention version and which SDPA
  backends are enabled.
- Workload: the settings the generation ran with.
- Kernel table: every case with its series, kernels and error metrics, plus the layout measured for
  each shape and skipped cases with the reason they could not run.
- Wirings: latency summaries, stage split, attention attribution, quality, the GPU samples taken
  during that wiring's generations, verdicts, and the paths of the video and audio saved for it.
- Media: only present with `--save-media`.
- Progress: every generation completed so far, written after each one, so a run that fails or is
  interrupted still keeps the generations it paid for. The section is dropped once the wirings are
  summarised.
"""

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
import traceback
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import imageio.v2 as imageio
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, record_function

from src.model import attention_backend, pipeline as pipeline_module, transformer as transformer_module
from src.model.config import EngineConfig
from src.model.weights import load_pipeline_from_checkpoint

DEVICE = torch.device("cuda")
FIXED_DURATION_SECONDS = 12.0
DEFAULT_RUNS = 3
DEFAULT_MICROBENCH_REPS = 20
MICROBENCH_WARMUP_REPS = 5
MATH_BACKEND_MEMORY_LIMIT_GB = 8.0
BYTES_PER_GB = 1024 ** 3
FRAME_COMPARE_CHUNK = 16
SAGE_PREPARE_COMMAND = "uv run python -m scripts.prepare_attention"
ATTENTION_RANGE_PREFIX = "attn::"

PATH_MODULES = (
    ("video_self", "attn1"),
    ("audio_self", "audio_attn1"),
    ("video_text", "attn2"),
    ("audio_text", "audio_attn2"),
    ("audio_to_video", "audio_to_video_attn"),
    ("video_to_audio", "video_to_audio_attn"),
)
STAGE_NAMES = ("stage1", "stage2")
CONNECTOR_PATH = "connector"

FORCED_SDPA = {
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
    "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_math": SDPBackend.MATH,
}
SAGE_KERNELS = {
    "sage_fp16": ("sageattn_qk_int8_pv_fp16_cuda", {"tensor_layout": "HND", "pv_accum_dtype": "fp32"}),
    "sage_fp8": (
        "sageattn_qk_int8_pv_fp8_cuda",
        {"tensor_layout": "HND", "qk_quant_gran": "per_warp", "pv_accum_dtype": "fp32+fp16"},
    ),
}
BACKENDS = ("sdpa_auto",) + tuple(FORCED_SDPA) + tuple(SAGE_KERNELS)

# The thresholds a wiring has to clear on a full generation before it is shippable.
GATES = {
    "video_psnr_db": 32.0,
    "stage2_video_latent_cosine": 0.9999,
    "audio_envelope_correlation": 0.999,
    "audio_waveform_correlation": 0.99,
}
# How far a kernel may drift from PyTorch's answer before it stops being a candidate, however
# fast it is.
ACCURACY_TOLERANCE = 0.02

# The workload the recorded frozen-baseline report used, so the two sets of numbers describe the
# same request. The outputs are not comparable between the two: the baseline and this model are
# different bodies, so only their latency figures are meant to be read side by side.
TEXT_ONLY_PROMPT = (
    "A white and black humanoid robot with sleek futuristic armor stands outdoors in front of "
    "a massive dark building under a clear blue sky. The robot slowly turns its head, looks around, "
    "and takes a few heavy mechanical steps forward while moving its arms naturally. Sunlight reflects "
    "off its glossy armor as the camera captures it from a slightly low angle. Add realistic servo sounds, "
    "mechanical whirring, heavy footsteps, and subtle outdoor ambience in the background."
)

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_ATTENTION_DIR = BENCHMARK_DIR / "results" / "attention"
DEFAULT_OUTPUT_JSON = str(RESULTS_ATTENTION_DIR / "benchmark_attention_results.json")
MEDIA_DIR_ENV = "LTX_BENCHMARK_MEDIA_DIR"


@dataclass(frozen=True)
class Wiring:
    name: str
    stage1: str
    stage2: str
    reference: bool = False


def neutral_wirings():
    return (
        Wiring("py__py", "sdpa_auto", "sdpa_auto", reference=True),
        Wiring("py__sage_fp16", "sdpa_auto", "sage_fp16"),
        Wiring("py__sage_fp8", "sdpa_auto", "sage_fp8"),
        Wiring("sage_fp16__py", "sage_fp16", "sdpa_auto"),
        Wiring("sage_fp16__sage_fp16", "sage_fp16", "sage_fp16"),
        Wiring("sage_fp16__sage_fp8", "sage_fp16", "sage_fp8"),
        Wiring("sage_fp8__py", "sage_fp8", "sdpa_auto"),
        Wiring("sage_fp8__sage_fp16", "sage_fp8", "sage_fp16"),
        Wiring("sage_fp8__sage_fp8", "sage_fp8", "sage_fp8"),
        Wiring("cudnn__cudnn", "sdpa_cudnn", "sdpa_cudnn"),
    )


def summarize(values):
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None, "spread_percent": None}
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "n": len(ordered),
        "median": round(median, 4),
        "mean": round(statistics.mean(ordered), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "spread_percent": round(100.0 * (ordered[-1] - ordered[0]) / median, 2) if median > 0 else None,
    }


def dtype_from_name(name):
    return getattr(torch, name.split(".")[-1])


def score_matrix_gb(batch, heads, queries, keys):
    return batch * heads * queries * keys * 4 / BYTES_PER_GB


def sage_module():
    try:
        return importlib.import_module("sageattention")
    except ImportError as exc:
        raise RuntimeError(
            f"sageattention is not importable and the selected backends need it. "
            f"Run `{SAGE_PREPARE_COMMAND}` from the project root.\n{exc}"
        ) from exc


def backend_callables():
    """One callable per backend candidate, all taking (query, key, value, mask)."""
    calls = {}

    def sdpa(query, key, value, mask):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
        )

    calls["sdpa_auto"] = sdpa

    for name, forced in FORCED_SDPA.items():
        def forced_call(query, key, value, mask, backends=(forced,)):
            with sdpa_kernel(list(backends)):
                return sdpa(query, key, value, mask)

        calls[name] = forced_call

    package = sage_module()
    for name, (kernel_name, options) in SAGE_KERNELS.items():
        kernel = getattr(package, kernel_name, None)
        if kernel is None:
            raise RuntimeError(
                f"sageattention {getattr(package, '__version__', 'unknown')} does not export "
                f"{kernel_name}; rebuild it with `{SAGE_PREPARE_COMMAND} --force`."
            )

        def sage_call(query, key, value, mask, kernel=kernel, options=options):
            if mask is not None:
                raise ValueError(
                    "the SageAttention kernel ignores attn_mask, so it cannot serve a masked call"
                )
            return kernel(query, key, value, **options)

        calls[name] = sage_call
    return calls


def sage_version():
    package = sage_module()
    return {
        "version": getattr(package, "__version__", "unknown"),
        "module": package.__file__,
    }


def environment_info():
    properties = torch.cuda.get_device_properties(DEVICE)
    capability = torch.cuda.get_device_capability(DEVICE)
    return {
        "device": properties.name,
        "capability": f"{capability[0]}.{capability[1]}",
        "sm_count": properties.multi_processor_count,
        "total_memory_gb": round(properties.total_memory / BYTES_PER_GB, 3),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sageattention": sage_version(),
        "sdpa_backends_enabled": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "driver_version": gpu_driver_version(),
    }


def parse_optional_float(value):
    value = value.strip()
    if value in ("", "N/A", "[N/A]", "[Not Supported]", "[Unknown Error]"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_flag(value):
    value = value.strip().lower()
    if value == "active":
        return True
    if value == "not active":
        return False
    return None


def nvidia_smi_query(fields):
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.strip().splitlines()]


def gpu_driver_version():
    rows = nvidia_smi_query("driver_version")
    return rows[0].strip() if rows else None


def gpu_state():
    """Clock, power, temperature and throttle flags at one moment, sampled around every generation."""
    fields = (
        "clocks.sm,power.draw,temperature.gpu,"
        "clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.sw_thermal_slowdown"
    )
    rows = nvidia_smi_query(fields)
    if not rows:
        return {"available": False, "reason": "nvidia-smi unavailable"}
    values = [value.strip() for value in rows[0].split(",")]
    return {
        "available": True,
        "sm_clock_mhz": parse_optional_float(values[0]),
        "power_watts": parse_optional_float(values[1]),
        "temperature_celsius": parse_optional_float(values[2]),
        "power_cap_active": parse_optional_flag(values[3]),
        "thermal_slowdown_active": parse_optional_flag(values[4]),
    }


def describe_call(query, key, value):
    return {
        "q": {"shape": list(query.shape), "strides": list(query.stride())},
        "k": {"shape": list(key.shape), "strides": list(key.stride())},
        "v": {"shape": list(value.shape), "strides": list(value.stride())},
        "dtype": str(query.dtype),
        "contiguous": [query.is_contiguous(), key.is_contiguous(), value.is_contiguous()],
    }


def shape_key(stage, path, batch):
    return f"{stage}/{path}/b{batch}"


class StageTimer:
    """Times each denoising stage and keeps the latents it produced"""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.restorers = []
        self.records = []
        self.current_stage = None
        self.transformer_calls = {}

    def install(self):
        original = self.pipeline._denoise_stage
        original_transformer = self.pipeline.transformer.forward

        def transformer_probe(*args, **kwargs):
            if self.current_stage is not None:
                self.transformer_calls[self.current_stage] = self.transformer_calls.get(self.current_stage, 0) + 1
            return original_transformer(*args, **kwargs)

        def denoise_probe(*args, **kwargs):
            index = len(self.records)
            stage = STAGE_NAMES[index] if index < len(STAGE_NAMES) else f"stage{index + 1}"
            record = {
                "stage": stage,
                "start": torch.cuda.Event(enable_timing=True),
                "end": torch.cuda.Event(enable_timing=True),
                "host_start": time.perf_counter(),
                "host_end": None,
                "latents": None,
            }
            self.records.append(record)
            self.current_stage = stage
            record["start"].record()
            try:
                latents, audio_latents = original(*args, **kwargs)
            finally:
                record["end"].record()
                record["host_end"] = time.perf_counter()
                self.current_stage = None
            record["latents"] = (latents.detach().float().cpu(), audio_latents.detach().float().cpu())
            return latents, audio_latents

        self.pipeline._denoise_stage = denoise_probe
        self.pipeline.transformer.forward = transformer_probe
        self.restorers.append(lambda: vars(self.pipeline).pop("_denoise_stage", None))
        self.restorers.append(lambda: vars(self.pipeline.transformer).pop("forward", None))

    def start_generation(self):
        self.records = []
        self.transformer_calls = {}

    def finish(self):
        torch.cuda.synchronize()
        stages = {}
        for record in self.records:
            stages[record["stage"]] = {
                "device_seconds": round(record["start"].elapsed_time(record["end"]) / 1000.0, 3),
                "host_seconds": round(record["host_end"] - record["host_start"], 3),
                "transformer_passes": self.transformer_calls.get(record["stage"]),
                "video_latents": record["latents"][0],
                "audio_latents": record["latents"][1],
            }
        return stages

    def restore(self):
        for restore in reversed(self.restorers):
            restore()
        self.restorers = []


class AttentionHarness:
    """Serves every attention call from a wiring for the duration of one generation"""
    def __init__(self, pipeline, backends):
        self.pipeline = pipeline
        self.backends = backends
        self.wiring = None
        self.profile = False
        self.capture_shapes = False
        self.current_path = None
        self.calls = {}
        self.masked = {}
        self.shapes = {}
        self.restorers = []

    def begin(self, wiring, profile, capture_shapes):
        self.wiring = wiring
        self.profile = profile
        self.capture_shapes = capture_shapes
        self.calls = {}
        self.masked = {}
        self.current_path = None

    def _labelled(self, forward, path):
        def probe(*args, **kwargs):
            previous = self.current_path
            self.current_path = path
            try:
                return forward(*args, **kwargs)
            finally:
                self.current_path = previous

        return probe

    def dispatch(self, query, key, value, attn_mask=None, *, role=None):
        # `role` is accepted because the model passes it and deliberately ignored: a wiring names one
        # backend per stage, so every path in a stage runs on the same one.
        stage = attention_backend._stage
        path = self.current_path
        if path == CONNECTOR_PATH:
            # The connectors reuse the attention module but sit outside the six measured paths,
            # so they stay on PyTorch in every wiring and are not recorded.
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
        if path is None:
            raise RuntimeError(
                "attention ran outside a labelled module, so its path is unknown; "
                "the path labels were not installed"
            )
        batch = int(query.shape[0])
        key_ = shape_key(stage, path, batch)
        self.calls[key_] = self.calls.get(key_, 0) + 1
        if attn_mask is not None:
            self.masked[key_] = self.masked.get(key_, 0) + 1
        if self.capture_shapes and key_ not in self.shapes:
            self.shapes[key_] = describe_call(query, key, value)
        backend = {"stage1": self.wiring.stage1, "stage2": self.wiring.stage2}[stage]
        if not backend:
            raise RuntimeError(f"wiring {self.wiring.name} has no backend for {stage}")
        label = f"{ATTENTION_RANGE_PREFIX}{stage}::{path}::b{batch}"
        context = record_function(label) if self.profile else contextlib.nullcontext()
        with context:
            return self.backends[backend](query, key, value, attn_mask)

    def install(self):
        def patch_instance(module, attribute, value):
            setattr(module, attribute, value)
            self.restorers.append(lambda: vars(module).pop(attribute, None))

        def patch_module_attr(module, attribute, value):
            original = getattr(module, attribute)
            setattr(module, attribute, value)
            self.restorers.append(lambda: setattr(module, attribute, original))

        patch_module_attr(transformer_module, "attention", self.dispatch)
        for block in self.pipeline.transformer.transformer_blocks:
            for path, attribute in PATH_MODULES:
                module = getattr(block, attribute)
                patch_instance(module, "forward", self._labelled(module.forward, path))
        for module in self.pipeline.connectors.modules():
            if isinstance(module, transformer_module.LTX2Attention):
                patch_instance(module, "forward", self._labelled(module.forward, CONNECTOR_PATH))
        # Every pass receives its cache through this one function, so returning nothing here turns
        # block reuse off without touching the model and keeps the workload identical across
        # wirings. 
        patch_module_attr(pipeline_module, "cache_for_pass", lambda cache, pass_name: None)

    def uninstall(self):
        for restore in reversed(self.restorers):
            restore()
        self.restorers = []

    def summary(self):
        return {
            "calls": dict(sorted(self.calls.items())),
            "masked_calls": dict(sorted(self.masked.items())),
        }


def attribute_attention(profiler):
    attribution = {}
    for event in profiler.key_averages():
        if not event.key.startswith(ATTENTION_RANGE_PREFIX):
            continue
        attribution[event.key] = {
            "calls": event.count,
            "device_seconds": round(event.self_device_time_total / 1e6, 4),
        }
    return dict(sorted(attribution.items()))


def attention_breakdown(attribution, stages):
    rows = {}
    for label, info in attribution.items():
        _, stage, path, batch = label.split("::")
        share = None
        if stage in stages and stages[stage]["device_seconds"] > 0:
            share = round(100.0 * info["device_seconds"] / stages[stage]["device_seconds"], 2)
        rows[f"{stage}/{path}/{batch}"] = {
            "calls": info["calls"],
            "device_seconds": info["device_seconds"],
            "median_ms": round(1000.0 * info["device_seconds"] / info["calls"], 3) if info["calls"] else None,
            "share_of_stage_percent": share,
        }
    return dict(sorted(rows.items()))


def build_tensors(spec, device):
    dtype = dtype_from_name(spec["dtype"])

    def one(part):
        tensor = torch.empty_strided(
            tuple(part["shape"]), tuple(part["strides"]), dtype=dtype, device=device
        )
        tensor.normal_(0.0, 1.0)
        return tensor

    return one(spec["q"]), one(spec["k"]), one(spec["v"])


def error_against(reference, candidate):
    reference = reference.float()
    candidate = candidate.float()
    difference = (candidate - reference).abs()
    return {
        "max_abs": round(float(difference.max()), 6),
        "mean_abs": round(float(difference.mean()), 8),
        "relative_mean_abs": round(
            float(difference.mean() / reference.abs().mean().clamp_min(1e-8)), 8
        ),
        "cosine": round(
            float(torch.nn.functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0)),
            8,
        ),
        "finite": bool(candidate.isfinite().all()),
    }


def time_call(call, reps):
    for _ in range(MICROBENCH_WARMUP_REPS):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def launched_kernels(call):
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as profiler:
        call()
        torch.cuda.synchronize()
    rows = [
        {"kernel": event.key, "ms": round(event.device_time_total / 1000.0, 4)}
        for event in profiler.key_averages()
        if event.device_time_total > 0
    ]
    rows.sort(key=lambda row: -row["ms"])
    return rows[:6]


def failure_reason(exc):
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def microbenchmark_case(stage, path, batch, spec, backend_names, calls, reps):
    label = shape_key(stage, path, batch)
    query, key, value = build_tensors(spec, DEVICE)
    heads = int(query.shape[1])
    queries = int(query.shape[2])
    keys = int(key.shape[2])
    head_dim = int(query.shape[3])
    score_gb = score_matrix_gb(batch, heads, queries, keys)
    flop = 4 * batch * heads * queries * keys * head_dim
    cases = []
    unavailable = []
    try:
        reference = calls["sdpa_auto"](query, key, value, None).clone()
    except Exception as exc:
        # Without a reference answer there is nothing to measure a backend against, so the whole
        # shape is reported as unavailable instead of aborting the table.
        unavailable.append({"shape": label, "backend": "sdpa_auto", "reason": failure_reason(exc)})
        del query, key, value
        return cases, unavailable
    for name in backend_names:
        if name == "sdpa_math" and score_gb > MATH_BACKEND_MEMORY_LIMIT_GB:
            unavailable.append({
                "shape": label,
                "backend": name,
                "reason": f"float32 scores need {round(score_gb, 2)} GiB, above the "
                          f"{MATH_BACKEND_MEMORY_LIMIT_GB} GiB this case is allowed",
            })
            continue
        call = lambda name=name: calls[name](query, key, value, None)
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = torch.cuda.memory_allocated()
            output = call()
            peak_allocated = torch.cuda.max_memory_allocated()
            first = output.clone()
            output = call()
            deterministic = bool(torch.equal(first, output))
            samples = time_call(call, reps)
            kernels = launched_kernels(call)
        except Exception as exc:
            # One backend refusing one shape must not cost the whole table. The case is recorded as
            # unavailable with the reason; nothing is substituted for it.
            unavailable.append({"shape": label, "backend": name, "reason": failure_reason(exc)})
            torch.cuda.empty_cache()
            continue
        median_ms = statistics.median(samples)
        cases.append({
            "shape": label,
            "stage": stage,
            "path": path,
            "batch": batch,
            "backend": name,
            "median_ms": round(median_ms, 4),
            "min_ms": round(min(samples), 4),
            "max_ms": round(max(samples), 4),
            "samples_ms": [round(value, 4) for value in samples],
            "tflops": round(flop / (median_ms / 1000.0) / 1e12, 2),
            "peak_allocated_gb": round((peak_allocated - baseline_allocated) / BYTES_PER_GB, 4),
            "deterministic": deterministic,
            "kernels": kernels,
            "error_vs_sdpa": error_against(reference, first),
        })
        del output, first
    del query, key, value, reference
    return cases, unavailable


def run_microbenchmark(shapes, backend_names, calls, reps):
    """The kernel table, plus the layout each shape was measured on so the rows stay readable."""
    cases = []
    unavailable = []
    for label, spec in sorted(shapes.items()):
        parts = label.split("/")
        stage, path, batch = parts[0], parts[1], int(parts[2].removeprefix("b"))
        case_rows, skipped = microbenchmark_case(stage, path, batch, spec, backend_names, calls, reps)
        cases.extend(case_rows)
        unavailable.extend(skipped)
        print(f"  {label}: " + " ".join(
            f"{row['backend']}={row['median_ms']}ms" for row in case_rows
        ))
    return cases, unavailable, dict(sorted(shapes.items()))


def best_per_shape(cases):
    """The fastest backend per shape and the fastest one that stays close to PyTorch's answer."""
    grouped = {}
    for case in cases:
        grouped.setdefault(case["shape"], []).append(case)
    summary = {}
    for label, rows in sorted(grouped.items()):
        default = next((row for row in rows if row["backend"] == "sdpa_auto"), None)
        fastest = min(rows, key=lambda row: row["median_ms"])
        accurate = [
            row for row in rows if row["error_vs_sdpa"]["relative_mean_abs"] <= ACCURACY_TOLERANCE
        ]
        fastest_accurate = min(accurate, key=lambda row: row["median_ms"]) if accurate else None
        summary[label] = {
            "sdpa_auto_ms": default["median_ms"] if default else None,
            "fastest": {
                "backend": fastest["backend"],
                "median_ms": fastest["median_ms"],
                "relative_error": fastest["error_vs_sdpa"]["relative_mean_abs"],
            },
            "fastest_within_accuracy": None if fastest_accurate is None else {
                "backend": fastest_accurate["backend"],
                "median_ms": fastest_accurate["median_ms"],
                "relative_error": fastest_accurate["error_vs_sdpa"]["relative_mean_abs"],
            },
            "kernels_by_backend": {row["backend"]: [kernel["kernel"] for kernel in row["kernels"]]
                                   for row in rows},
        }
    return summary


def output_fingerprint(video, audio):
    """Subsample hashes of one generation, so runs at a fixed seed can be shown to agree."""
    subsamples = {
        "video": video.detach()[0, :, ::7, ::29, ::37].float().contiguous(),
        "audio": audio.detach()[0, :, ::53].float().contiguous(),
    }
    fingerprint = {}
    for name, subsample in subsamples.items():
        fingerprint[f"{name}_sha256_16"] = hashlib.sha256(
            subsample.cpu().numpy().tobytes()
        ).hexdigest()[:16]
    return fingerprint


def media_directory():
    return Path(os.environ.get(MEDIA_DIR_ENV, RESULTS_ATTENTION_DIR))


def save_generated_media(video, audio, directory, stem, frame_rate, sample_rate):
    destination = Path(directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    video_path = destination / f"{stem}.mp4"
    audio_path = destination / f"{stem}.wav"
    frames = video[0].detach().float().cpu()
    frames = ((frames.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()
    imageio.mimsave(str(video_path), list(frames), fps=frame_rate)
    samples = (audio[0].detach().float().cpu().clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(samples.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.T.tobytes())
    return {"video": str(video_path), "audio": str(audio_path)}


def video_comparison(reference, candidate):
    """Full-frame comparison in frame chunks, so two decoded videos never sit in float32 at once."""
    total_squared = 0.0
    total_absolute = 0.0
    total_reference_absolute = 0.0
    peak = 0.0
    count = 0
    for start in range(0, reference.shape[2], FRAME_COMPARE_CHUNK):
        reference_chunk = reference[:, :, start:start + FRAME_COMPARE_CHUNK].float()
        candidate_chunk = candidate[:, :, start:start + FRAME_COMPARE_CHUNK].float()
        difference = candidate_chunk - reference_chunk
        total_squared += float(difference.pow(2).sum())
        total_absolute += float(difference.abs().sum())
        total_reference_absolute += float(reference_chunk.abs().sum())
        peak = max(peak, float(difference.abs().max()))
        count += difference.numel()
        del reference_chunk, candidate_chunk, difference
    mse = total_squared / count
    return {
        "max_abs": round(peak, 6),
        "mean_abs": round(total_absolute / count, 8),
        "relative_mean_abs": round(total_absolute / max(total_reference_absolute, 1e-8), 8),
        # The decoded video lives in [-1, 1], so the peak power is 4.
        "psnr_db": round(10.0 * math.log10(4.0 / mse), 3) if mse > 0 else None,
    }


def envelope(waveform, window=2048):
    flat = waveform.float().flatten()
    usable = flat.numel() - flat.numel() % window
    return flat[:usable].view(-1, window).pow(2).mean(dim=1).sqrt()


def correlation(reference, candidate):
    first = reference.float().flatten()
    second = candidate.float().flatten()
    if first.numel() != second.numel():
        raise ValueError(f"cannot correlate {tuple(reference.shape)} with {tuple(candidate.shape)}")
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if float(denominator) == 0.0:
        return None
    return round(float((first @ second) / denominator), 6)


def audio_comparison(reference, candidate):
    difference = candidate.float() - reference.float()
    return {
        "max_abs": round(float(difference.abs().max()), 6),
        "mean_abs": round(float(difference.abs().mean()), 8),
        "relative_mean_abs": round(
            float(difference.abs().mean() / reference.float().abs().mean().clamp_min(1e-8)), 8
        ),
        "envelope_correlation": correlation(envelope(reference), envelope(candidate)),
        "waveform_correlation": correlation(reference, candidate),
        "finite": bool(candidate.isfinite().all()),
    }


def latent_comparison(reference, candidate):
    return error_against(reference, candidate)


def gate_verdict(measured):
    failed = {
        name: {"measured": measured.get(name), "required": threshold}
        for name, threshold in GATES.items()
        if measured.get(name) is None or measured[name] < threshold
    }
    return {"required": dict(GATES), "measured": measured, "passes": not failed, "failed": failed}


def quality_comparison(reference, candidate):
    """Restates one wiring's output against the reference wiring's output."""
    latents = {}
    for stage, reference_latents in reference["stages"].items():
        candidate_latents = candidate["stages"].get(stage)
        if candidate_latents is None:
            latents[stage] = None
            continue
        latents[stage] = {
            "video": latent_comparison(reference_latents["video_latents"], candidate_latents["video_latents"]),
            "audio": latent_comparison(reference_latents["audio_latents"], candidate_latents["audio_latents"]),
        }
    video = video_comparison(reference["video"], candidate["video"])
    audio = audio_comparison(reference["audio"], candidate["audio"])
    measured = {
        "video_psnr_db": video["psnr_db"],
        "stage2_video_latent_cosine": latents.get("stage2", {}).get("video", {}).get("cosine")
        if latents.get("stage2") else None,
        "audio_envelope_correlation": audio["envelope_correlation"],
        "audio_waveform_correlation": audio["waveform_correlation"],
    }
    return {"video": video, "audio": audio, "stages": latents, "gates": gate_verdict(measured)}


def run_generation(pipeline, settings, stage_timer, harness, wiring, profiled, capture_shapes=False):
    stage_timer.start_generation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    retries_before = int(torch.cuda.memory_stats().get("num_alloc_retries", 0))
    state_before = gpu_state()
    profiler = (
        torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        )
        if profiled
        else None
    )
    with contextlib.ExitStack() as stack:
        harness.begin(wiring, profile=profiled, capture_shapes=capture_shapes)
        harness.install()
        stack.callback(harness.uninstall)
        if profiler is not None:
            stack.enter_context(profiler)
        torch.cuda.synchronize()
        started = time.perf_counter()
        video, audio = pipeline.generate(settings)
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
    stages = stage_timer.finish()
    state_after = gpu_state()
    record = {
        "wall_seconds": round(wall_seconds, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / BYTES_PER_GB, 4),
        "alloc_retries": int(torch.cuda.memory_stats().get("num_alloc_retries", 0)) - retries_before,
        "gpu_before": state_before,
        "gpu_after": state_after,
        "output": output_fingerprint(video, audio),
        "stages": {
            name: {
                "device_seconds": info["device_seconds"],
                "host_seconds": info["host_seconds"],
                "transformer_passes": info["transformer_passes"],
            }
            for name, info in stages.items()
        },
    }
    record["attention"] = harness.summary()
    if profiler is not None:
        record["attention_attribution"] = attention_breakdown(attribute_attention(profiler), stages)
    return record, stages, video, audio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attention backend benchmark: kernel table plus end-to-end stage wirings."
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--save-media", action="store_true",
                        help="write each wiring's first generated video and audio to "
                             "benchmark/results/attention/ (off by default)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"timed generations per wiring (default {DEFAULT_RUNS})")
    parser.add_argument("--microbench-reps", type=int, default=DEFAULT_MICROBENCH_REPS,
                        help=f"repetitions per kernel-table case (default {DEFAULT_MICROBENCH_REPS})")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.microbench_reps < 3:
        parser.error("--microbench-reps must be >= 3")
    return args


def write_report_atomic(output_path, report):
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp"
    try:
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def progress_snapshot(report, output_path, measurements):
    """Save what has been measured so far, so an interrupted run keeps its completed generations."""
    report["progress"] = {
        name: {
            "records": measurement["records"],
            "profiled": measurement.get("profiled"),
            "quality": measurement.get("quality"),
        }
        for name, measurement in measurements.items()
    }
    write_report_atomic(output_path, report)


def build_settings(engine_config):
    settings = engine_config.generation
    settings.prompt = TEXT_ONLY_PROMPT
    settings.duration_seconds = FIXED_DURATION_SECONDS
    settings.higher_resolution = True
    settings.image_path = ""
    return settings


def print_summary(report):
    print()
    print("kernel table, median device time in ms")
    header = f"{'shape':28s} " + " ".join(f"{name[:11]:>11s}" for name in BACKENDS)
    print(header)
    by_shape = {}
    for case in report["microbenchmarks"]["cases"]:
        by_shape.setdefault(case["shape"], {})[case["backend"]] = case["median_ms"]
    for label in sorted(by_shape):
        row = by_shape[label]
        print(f"{label:28s} " + " ".join(
            f"{row[name]:>11.3f}" if name in row else f"{'-':>11s}" for name in BACKENDS
        ))
    for skipped in report["microbenchmarks"]["unavailable"]:
        print(f"  unavailable {skipped['shape']} {skipped['backend']}: {skipped['reason']}")

    print()
    print("wirings")
    print(f"{'wiring':24s} {'wall s':>9s} {'stage1':>9s} {'stage2':>9s} {'vs ref':>8s} "
          f"{'attn %':>7s} quality")
    for wiring in report["wirings"]:
        quality = wiring.get("quality", {}).get("gates", {})
        outcome = "reference" if wiring["reference"] else ("pass" if quality.get("passes") else "FAIL")
        print(
            f"{wiring['name']:24s} {wiring['wall_seconds']['median']:>9.2f} "
            f"{wiring['stage_seconds'].get('stage1', {}).get('median') or 0.0:>9.2f} "
            f"{wiring['stage_seconds'].get('stage2', {}).get('median') or 0.0:>9.2f} "
            f"{wiring.get('speedup_vs_reference') or 0.0:>8.4f} "
            f"{wiring.get('attention_share_of_request_percent') or 0.0:>7.2f} {outcome}"
        )
        for failure, detail in quality.get("failed", {}).items():
            print(f"    {failure}: measured {detail['measured']} required {detail['required']}")
    for warning in report["warnings"]:
        print(f"warning: {warning}")


def summarize_gpu_states(records):
    """Clock, power and temperature across one wiring's generations, so throttling stays visible."""
    states = [state for record in records for state in (record["gpu_before"], record["gpu_after"])]
    measured = [state for state in states if state["available"]]
    return {
        "available": bool(measured),
        "samples": len(measured),
        "sm_clock_mhz": summarize([
            state["sm_clock_mhz"] for state in measured if state["sm_clock_mhz"] is not None
        ]),
        "power_watts": summarize([
            state["power_watts"] for state in measured if state["power_watts"] is not None
        ]),
        "temperature_celsius": summarize([
            state["temperature_celsius"] for state in measured
            if state["temperature_celsius"] is not None
        ]),
        "power_cap_active_samples": sum(1 for state in measured if state["power_cap_active"]),
        "thermal_slowdown_active_samples": sum(
            1 for state in measured if state["thermal_slowdown_active"]
        ),
    }


def summarize_wiring(wiring, measurement, reference_wall):
    wall = summarize([record["wall_seconds"] for record in measurement["records"]])
    stage_seconds = {}
    for stage in STAGE_NAMES:
        stage_seconds[stage] = summarize([
            record["stages"][stage]["device_seconds"]
            for record in measurement["records"] if stage in record["stages"]
        ])
    fingerprints = {
        (record["output"]["video_sha256_16"], record["output"]["audio_sha256_16"])
        for record in measurement["records"]
    }
    calls = measurement["records"][0].get("attention", {}).get("calls")
    counts_agree = all(
        record.get("attention", {}).get("calls") == calls for record in measurement["records"]
    ) if calls is not None else None
    summary = {
        "name": wiring.name,
        "stage1": wiring.stage1,
        "stage2": wiring.stage2,
        "reference": wiring.reference,
        "wall_seconds": {**wall, "series": [record["wall_seconds"] for record in measurement["records"]]},
        "stage_seconds": stage_seconds,
        "runs": len(measurement["records"]),
        "identical_outputs_across_runs": len(fingerprints) == 1,
        "attention_call_counts_identical_across_runs": counts_agree,
        "peak_allocated_gb": summarize([record["peak_allocated_gb"] for record in measurement["records"]]),
        "alloc_retries": sum(record["alloc_retries"] for record in measurement["records"]),
        "gpu": summarize_gpu_states(measurement["records"]),
    }
    media = [record["media"] for record in measurement["records"] if record.get("media")]
    if media:
        summary["media"] = media[0]
    if reference_wall:
        summary["speedup_vs_reference"] = round(reference_wall / wall["median"], 4) if wall["median"] else None
    if wiring.reference:
        summary["speedup_vs_reference"] = 1.0
    denoise_seconds = sum((stage_seconds[stage]["median"] or 0.0) for stage in STAGE_NAMES)
    if wall["median"]:
        # How much of the whole request the two denoising stages account for, and therefore the
        # ceiling on anything that only makes attention faster: encoding, upsampling and decoding
        # are outside it.
        summary["denoise_stage_share_of_request_percent"] = round(
            100.0 * denoise_seconds / wall["median"], 2
        )
    profiled = measurement["profiled"]
    if "attention_attribution" in profiled:
        summary["attention"] = profiled["attention_attribution"]
        summary["attention_calls"] = profiled.get("attention", {}).get("calls")
        summary["masked_calls"] = profiled.get("attention", {}).get("masked_calls")
        summary["profiled_call_counts_match"] = summary["attention_calls"] == calls
        attention_seconds = sum(row["device_seconds"] for row in summary["attention"].values())
        profiled_denoise_seconds = sum(info["device_seconds"] for info in profiled["stages"].values())
        if profiled_denoise_seconds > 0:
            share = 100.0 * attention_seconds / profiled_denoise_seconds
            summary["attention_share_of_denoise_stages_percent"] = round(share, 2)
            if "denoise_stage_share_of_request_percent" in summary:
                summary["attention_share_of_request_percent"] = round(
                    share * summary["denoise_stage_share_of_request_percent"] / 100.0, 2
                )
    return summary


def decide(summary, reference_spread):
    if summary["reference"]:
        return "reference"
    gates = summary.get("quality", {}).get("gates", {})
    if not gates.get("passes"):
        return "reject_quality"
    noise = max((reference_spread or 0.0) / 100.0, 0.01)
    if (summary.get("speedup_vs_reference") or 1.0) - 1.0 <= noise:
        return "keep_current"
    return "consider"


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this benchmark, none found")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"benchmark expects exactly one visible CUDA device, found {torch.cuda.device_count()}")

    started_at = datetime.now(timezone.utc)
    report = {
        "status": "running",
        "generated_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "phase": "startup",
        "errors": [],
        "warnings": [],
        "run_config": {
            "runs": args.runs,
            "microbench_reps": args.microbench_reps,
        },
        "environment": environment_info(),
        "microbenchmarks": {"cases": [], "unavailable": [], "layouts": {}, "best_per_shape": {}},
        "wirings": [],
        "gates": dict(GATES),
        "timings_seconds": {},
    }
    print(f"gpu={report['environment']['device']!r} capability={report['environment']['capability']} "
          f"torch={report['environment']['torch']} cuda={report['environment']['cuda']} "
          f"cudnn={report['environment']['cudnn']} "
          f"sage={report['environment']['sageattention']['version']}")

    engine_config = EngineConfig()
    settings = build_settings(engine_config)
    report["workload"] = {
        "prompt": settings.prompt,
        "height": int(settings.height),
        "width": int(settings.width),
        "duration_seconds": settings.duration_seconds,
        "num_inference_steps": int(settings.num_inference_steps),
        "seed": int(settings.seed),
        "higher_resolution": bool(settings.higher_resolution),
        "stage2_lora_strength": float(settings.stage2_lora_strength),
        "max_sequence_length": int(settings.max_sequence_length),
    }
    print(f"workload: {settings.height}x{settings.width} for {settings.duration_seconds}s, "
          f"{settings.num_inference_steps} stage-1 steps, seed {settings.seed}")

    pipeline = None
    stage_timer = None
    harness = None
    try:
        report["phase"] = "backends"
        calls = backend_callables()
        report["phase"] = "load"
        torch.cuda.synchronize()
        load_started = time.perf_counter()
        pipeline = load_pipeline_from_checkpoint(
            engine_config=engine_config,
            checkpoint_dir=args.checkpoint_dir,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        report["timings_seconds"]["load"] = round(time.perf_counter() - load_started, 3)
        for module in (pipeline.transformer, pipeline.text_encoder, pipeline.connectors, pipeline.vae,
                       pipeline.audio_vae, pipeline.vocoder, pipeline.upsampler):
            if module is not None:
                module.eval()
        print(f"pipeline loaded in {report['timings_seconds']['load']}s")
        if args.save_media:
            media_dir = media_directory()
            report["media"] = {
                "directory": str(media_dir.resolve()),
                "frame_rate": engine_config.frame_rate,
                "sample_rate": pipeline.vocoder.config.output_sampling_rate,
            }
            print(f"generated video and audio are written to {report['media']['directory']}")

        stage_timer = StageTimer(pipeline)
        stage_timer.install()
        harness = AttentionHarness(pipeline, calls)
        reference_wiring = next(wiring for wiring in neutral_wirings() if wiring.reference)

        report["phase"] = "warmup"
        print("warm-up generation on the reference wiring: discarded, activates kernels and captures shapes")
        warmup_started = time.perf_counter()
        warmup_record, _, video, audio = run_generation(
            pipeline, settings, stage_timer, harness, reference_wiring,
            profiled=False, capture_shapes=True,
        )
        del video, audio
        report["warmup"] = {
            "wall_seconds": warmup_record["wall_seconds"],
            "stages": warmup_record["stages"],
            "distinct_shapes": len(harness.shapes),
        }
        report["timings_seconds"]["warmup"] = round(time.perf_counter() - warmup_started, 3)
        print(f"warm-up {warmup_record['wall_seconds']}s, captured {len(harness.shapes)} attention shapes")

        report["phase"] = "microbenchmark"
        print(f"kernel table: {len(harness.shapes)} shapes, {args.microbench_reps} repetitions each")
        microbench_started = time.perf_counter()
        cases, unavailable, layouts = run_microbenchmark(
            harness.shapes, BACKENDS, calls, args.microbench_reps,
        )
        report["microbenchmarks"]["cases"] = cases
        report["microbenchmarks"]["unavailable"] = unavailable
        report["microbenchmarks"]["layouts"] = layouts
        report["microbenchmarks"]["best_per_shape"] = best_per_shape(cases)
        report["timings_seconds"]["microbenchmark"] = round(time.perf_counter() - microbench_started, 3)

        selected = neutral_wirings()
        reference_name = next(wiring.name for wiring in selected if wiring.reference)
        measurements = {}
        report["phase"] = "wirings"
        for round_index in range(args.runs):
            # Alternating the direction keeps a slow drift from landing on the same wiring twice.
            order = selected if round_index % 2 == 0 else list(reversed(selected))
            print(f"timed round {round_index + 1}/{args.runs}")
            for wiring in order:
                measurement = measurements.setdefault(
                    wiring.name,
                    {"records": [], "stages": None, "video": None, "audio": None, "quality": None},
                )
                record, stages, video, audio = run_generation(
                    pipeline, settings, stage_timer, harness, wiring, profiled=False,
                )
                record["run_index"] = len(measurement["records"]) + 1
                measurement["records"].append(record)
                if record["run_index"] == 1:
                    if args.save_media:
                        # Saved before the host copy is taken, so the float32 decode and the
                        # float16 copy never sit in RAM together. The runs are bit-identical at a
                        # fixed seed, so one artifact per wiring describes all of them.
                        record["media"] = save_generated_media(
                            video, audio, media_dir, f"{wiring.name}_run1",
                            report["media"]["frame_rate"], report["media"]["sample_rate"],
                        )
                    measurement["stages"] = stages
                    measurement["video"] = video.to(torch.float16).cpu()
                    measurement["audio"] = audio.to(torch.float16).cpu()
                del video, audio
                reference_measurement = measurements.get(reference_name)
                if (measurement["quality"] is None and not wiring.reference
                        and reference_measurement is not None
                        and reference_measurement["video"] is not None):
                    measurement["quality"] = quality_comparison(reference_measurement, measurement)
                    measurement["video"] = None
                    measurement["audio"] = None
                    measurement["stages"] = None
                print(f"  {wiring.name} run {record['run_index']}: {record['wall_seconds']}s "
                      f"(stage1 {record['stages'].get('stage1', {}).get('device_seconds')}s "
                      f"stage2 {record['stages'].get('stage2', {}).get('device_seconds')}s)")
                progress_snapshot(report, args.output_json, measurements)
            if round_index == 0:
                # Every wiring has now been compared against the reference, so its copy can go too.
                reference_measurement = measurements[reference_name]
                reference_measurement["video"] = None
                reference_measurement["audio"] = None
                reference_measurement["stages"] = None

        print("profiled generations, excluded from the latency statistics")
        for wiring in selected:
            measurement = measurements[wiring.name]
            profiled_record, _, video, audio = run_generation(
                pipeline, settings, stage_timer, harness, wiring, profiled=True,
            )
            del video, audio
            measurement["profiled"] = profiled_record
            print(f"  {wiring.name}: {profiled_record['wall_seconds']}s with profiling active")
            progress_snapshot(report, args.output_json, measurements)

        reference = measurements["py__py"]
        reference_wall = summarize([record["wall_seconds"] for record in reference["records"]])["median"]
        reference_spread = summarize([record["wall_seconds"] for record in reference["records"]])["spread_percent"]

        summaries = []
        for wiring in selected:
            measurement = measurements[wiring.name]
            summary = summarize_wiring(wiring, measurement, reference_wall)
            if measurement["quality"] is not None:
                summary["quality"] = measurement["quality"]
            elif not wiring.reference:
                report["warnings"].append(
                    f"{wiring.name}: never compared against the reference output, so it has no "
                    "quality verdict"
                )
            summaries.append(summary)
        for summary in summaries:
            summary["decision"] = decide(summary, reference_spread)

        reference_calls = reference["records"][0]["attention"]["calls"]
        for summary in summaries:
            if summary.get("attention_calls") is None:
                continue
            summary["workload_matches_reference"] = summary["attention_calls"] == reference_calls
            if not summary["workload_matches_reference"]:
                report["warnings"].append(
                    f"{summary['name']}: attention call counts differ from the reference wiring, so the "
                    "workload changed and its latency is not comparable"
                )

        report["wirings"] = summaries
        report["reference_wall_seconds"] = reference_wall

        report["phase"] = "verification"
        for summary in summaries:
            if summary.get("speedup_vs_reference") is None:
                continue
            if summary["attention_call_counts_identical_across_runs"] is False:
                report["warnings"].append(
                    f"{summary['name']}: attention call counts changed between its own runs, so the "
                    "workload is not stable and its latency cannot be compared"
                )
            if summary["identical_outputs_across_runs"] is False:
                report["warnings"].append(
                    f"{summary['name']}: output fingerprints differ between runs at a fixed seed, so a "
                    "latency change cannot be separated from a change in what was generated"
                )
            if summary.get("profiled_call_counts_match") is False:
                report["warnings"].append(
                    f"{summary['name']}: the profiled run made a different number of attention calls "
                    "than the timed runs, so its per-path breakdown describes a different workload"
                )
            gpu = summary["gpu"]
            if gpu["power_cap_active_samples"] + gpu["thermal_slowdown_active_samples"]:
                report["warnings"].append(
                    f"{summary['name']}: the card reported the power cap active in "
                    f"{gpu['power_cap_active_samples']} and thermal slowdown in "
                    f"{gpu['thermal_slowdown_active_samples']} of {gpu['samples']} samples, so its "
                    "latency was measured on a limited card"
                )
        for summary in summaries:
            if summary["reference"]:
                continue
            if not summary.get("quality", {}).get("gates", {}).get("passes", True):
                report["warnings"].append(
                    f"{summary['name']}: fails {sorted(summary['quality']['gates']['failed'])}"
                )

        stage_timer.restore()
        report.pop("progress", None)
        report["status"] = "ok"
        report["phase"] = "done"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report_atomic(args.output_json, report)
        print(f"report written to {args.output_json}")
        print_summary(report)
    except BaseException as exc:
        report["status"] = "failed"
        report["errors"].append({
            "phase": report["phase"],
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report_atomic(args.output_json, report)
        print(f"benchmark failed during {report['phase']}, partial report written to {args.output_json}")
        raise


if __name__ == "__main__":
    main()
