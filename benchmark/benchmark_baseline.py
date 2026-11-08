"""Baseline end-to-end benchmark for the LTX-2.5 engine.

This benchmarks the `baseline_model` implementation using a fixed text-to-video-and-audio request. Each invocation runs 
multiple generations in three stages: warm-up generations, instrumented measured generations and clean bare generations. 

Exactly one visible GPU is required. If a generation fails, the report is marked as partial and records the error while 
preserving results from completed runs. Headline latency is calculated as the median wall time across measured generations, 
with the minimum, maximum, and spread also reported."

Run the code:

    uv run -m benchmark.benchmark_baseline

Options:

    --checkpoint-dir    Folder holding the checkpoint. If omitted, the checkpoint is
                        downloaded from Hugging Face, which needs the token in .env.

    --output-json       Where the JSON report is written. Defaults to
                        benchmark/results/baseline/benchmark_baseline_results.json.

    --warmup-runs       Generations run and discarded before measuring, so kernels,
                        the memory pool and GPU clocks settle. Default 1.

    --measured-runs     Instrumented generations that are measured, with per-phase
                        timing and GPU telemetry active. Default 6.

    --bare-runs         Clean generations after the measured ones, with no extra
                        timing and no GPU telemetry. Default 6.

    --emit-raw-samples  Write every GPU telemetry sample as CSV beside the report.
                        Default off; the report keeps the summaries either way.

What We Measure
---------------
Each measured generation is one full `pipeline.generate` call. With the default settings it performs:

- Text encoding: the positive and negative prompts are encoded separately (2 calls).
- Upsampler: one 2x spatial latent upsampler between stage 1 and stage 2.
- Vocoding: one vocoder call that turns the mel into a stereo waveform.
- Connectors: stage-1 conditioning for both prompt branches, then stage-2 conditioning for the positive prompt only (2 calls).
- Stage 1 denoising: 30 steps with 3 transformer passes per step (CFG, STG and modality guidance), so 90 transformer calls. 
- LoRA: the stage-2 connector and transformer adapters are applied and removed (2 applies, 2 removes).
- Stage 2 refining: 3 fixed steps with guidance neutralized, so 1 transformer pass per step (3 calls).
- Video and audio decoding: one full video VAE decode, one audio VAE decode to a mel spectrogram
  and one vocoder call, timed as a single decoding phase because each of them hands its work to
  the GPU and returns before that work has finished.
- Weight movement: five modules (text encoder, transformer, connectors, upsampler and duration head) 
  are moved to CPU before decoding and back afterwards, with one allocator cache clear.

Per measured run the report records:

- End-to-end wall time around `pipeline.generate`, bracketed by CUDA synchronize.
- Device time (CUDA events) and host time (`perf_counter`) for every timed phase.
- Denoise detail: per-step and per-pass times, transformer call counts, token counts and video tokens per second.
- Peak allocator memory, incremental peak, allocation retries and OOM counts.
- A compact output fingerprint (video/audio hashes, finiteness and value range); the full output
  description is reported once.

What We Do Not Measure
----------------------
- Model loading: It is excluded from generation latency but timed and reported separately.
- Warm-up generations: They run but are discarded from the latency statistics.
- The reference-image path: image encoding, frame-0 pinning and the stage-2 image re-encode never run.
- The duration head. It is loaded but never invoked because `duration_seconds` is fixed at 12.
- The audio encoder. Only the audio VAE decoder is loaded.
- Clock, power, temperature and throttle flags per individual phase. Half-second polling is too
  coarse for phases that last milliseconds, so those describe each measured generation as a whole.

Results
-------
The JSON report is written atomically to `--output-json` and contains:

- Status, errors and warnings.
- Load time excluded from latency.
- Workload: requested and actual sizes, steps, frames and duration.
- Throughput: seconds per frame, frames per second and realtime factor.
- Measurements: wall-time summaries and every measured run, plus the bare runs.
- Output: one full description of the generated shapes, value ranges and subsample statistics.
- GPU telemetry: session summary and per-measured-generation samples and verdicts.
- Output consistency: distinct fingerprint counts and an identical verdict across all measured and bare generations.
- Memory: pre/post-load snapshots, per-component weight footprints and per-run allocator statistics.
- Attribution: per-phase caller and gpu times, which of the two is longer, caller and gpu shares
  of end-to-end time, the unattributed remainder and the denoise breakdown.

With `--emit-raw-samples`, a CSV is written beside the JSON. It has one row per nvidia-smi 
poll with elapsed time, pipeline stage, utilisation, memory utilisation, temperature, power, 
SM clock and throttle flags. Without the flag, only the summaries above are saved.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import torch

from baseline_model import pipeline as pipeline_module
from baseline_model.config import EngineConfig
from baseline_model.weights import load_pipeline_from_checkpoint

FIXED_DURATION_SECONDS = 12.0
DEFAULT_WARMUP_RUNS = 1
DEFAULT_MEASURED_RUNS = 6
DEFAULT_BARE_RUNS = 6
SAMPLER_INTERVAL_MS = 500
SAMPLER_STOP_TIMEOUT_S = 10
BYTES_PER_GB = 1024 ** 3
SPREAD_WARNING_PERCENT = 5.0
CLOCK_WARNING_FRACTION = 0.9
MIN_CLOCK_SAMPLES = 5
POWER_LIMIT_FRACTION = 0.98
DEVICE = torch.device("cuda")
PASS_ROLES = ("cfg", "stg", "modality")
REFINE_PASS_ROLES = ("refine",)

TEXT_ENCODE = "text_encode"
CONNECTORS = "connectors"
UPSAMPLER = "upsampler"
STAGE1_DENOISE = "stage1_denoise"
STAGE2_DENOISE = "stage2_denoise"
DECODE = "decode"
WEIGHT_OFFLOAD = "weight_offload"
WEIGHT_RELOAD = "weight_reload"
LORA_APPLY = "lora_apply"
LORA_REMOVE = "lora_remove"
CACHE_CLEAR = "cache_clear"
OFFLOADED_MODULES = ("text_encoder", "transformer", "connectors", "upsampler", "duration_head")

TEXT_ONLY_PROMPT = (
    "A white and black humanoid robot with sleek futuristic armor stands outdoors in front of "
    "a massive dark building under a clear blue sky. The robot slowly turns its head, looks around, "
    "and takes a few heavy mechanical steps forward while moving its arms naturally. Sunlight reflects "
    "off its glossy armor as the camera captures it from a slightly low angle. Add realistic servo sounds, "
    "mechanical whirring, heavy footsteps, and subtle outdoor ambience in the background."
)

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_BASELINE_DIR = BENCHMARK_DIR / "results" / "baseline"
DEFAULT_OUTPUT_JSON = str(RESULTS_BASELINE_DIR / "benchmark_baseline_results.json")


@dataclass(frozen=True)
class SamplerMetric:
    field: str
    sample_key: str
    average_key: str
    peak_key: str
    percentile_key: str
    count_key: str
    scale: float
    digits: int
    unit: str


UTIL_METRIC = SamplerMetric("utilization.gpu", "util_pct", "gpu_utilization_average_percent",
                            "gpu_utilization_sampled_peak_percent", "gpu_utilization_sampled_p95_percent",
                            "gpu_utilization_valid_sample_count", 1.0, 2, "%")
POWER_METRIC = SamplerMetric("power.draw", "power_w", "gpu_power_average_watts",
                             "gpu_power_sampled_peak_watts", "gpu_power_sampled_p95_watts",
                             "gpu_power_valid_sample_count", 1.0, 2, "W")
SM_CLOCK_METRIC = SamplerMetric("clocks.sm", "clock_sm_mhz", "graphics_clock_average_mhz",
                                "graphics_clock_sampled_peak_mhz", "graphics_clock_sampled_p95_mhz",
                                "graphics_clock_valid_sample_count", 1.0, 2, "MHz")
SAMPLER_METRICS = (
    UTIL_METRIC,
    SamplerMetric("temperature.gpu", "temp_c", "gpu_temperature_average_celsius",
                  "gpu_temperature_sampled_peak_celsius", "gpu_temperature_sampled_p95_celsius",
                  "gpu_temperature_valid_sample_count", 1.0, 2, "C"),
    POWER_METRIC,
    SM_CLOCK_METRIC,
)


@dataclass(frozen=True)
class ThrottleFlag:
    field: str
    sample_key: str


THROTTLE_FLAGS = (
    ThrottleFlag("clocks_throttle_reasons.sw_power_cap", "power_cap"),
    ThrottleFlag("clocks_throttle_reasons.sw_thermal_slowdown", "thermal_slowdown_sw"),
    ThrottleFlag("clocks_throttle_reasons.hw_thermal_slowdown", "thermal_slowdown_hw"),
)


def have_nvidia_smi():
    return shutil.which("nvidia-smi") is not None


def query_nvidia_smi(fields, timeout=30):
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    out = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi query failed: {out.stderr.strip()}")
    return [line.strip() for line in out.stdout.strip().splitlines()]


def parse_optional_float(value):
    value = value.strip()
    if value in ("", "N/A", "[N/A]", "[Not Supported]", "[Unknown Error]", "[Insufficient Permissions]"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_flag(value):
    """Throttle reasons report Active/Not Active, or an unsupported placeholder that must stay
    unknown rather than read as 'not throttled'."""
    value = value.strip().lower()
    if value == "active":
        return True
    if value == "not active":
        return False
    return None


def percentile_nearest_rank(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    rank = min(len(ordered), max(1, math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def summarize_values(values):
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None, "spread_percent": None}
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "n": len(ordered),
        "median": round(median, 3),
        "mean": round(statistics.mean(ordered), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "spread_percent": round(100.0 * (ordered[-1] - ordered[0]) / median, 2) if median > 0 else None,
    }


def environment_info():
    props = torch.cuda.get_device_properties(DEVICE)
    info = {
        "torch_device_name": props.name,
        "total_memory_gb": round(props.total_memory / BYTES_PER_GB, 3),
        "multi_processor_count": props.multi_processor_count,
        "cuda_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "backend_flags": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "environment_variables": {
            name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE",
                "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM",
            )
        },
        "driver_version": None,
        "max_sm_clock_mhz": None,
        "power_limit_watts": None,
        "pcie_link": None,
    }
    if not have_nvidia_smi():
        print("nvidia-smi not found; driver-level device limits and board memory will be unavailable")
        return info
    try:
        info["driver_version"] = query_nvidia_smi("driver_version")[0].strip()
    except Exception as exc:
        print(f"nvidia-smi driver query failed: {exc}")
        return info
    try:
        row = query_nvidia_smi("clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current")[0]
        max_sm, power_limit, link_generation, link_width = [value.strip() for value in row.split(",")]
        info["max_sm_clock_mhz"] = parse_optional_float(max_sm)
        info["power_limit_watts"] = parse_optional_float(power_limit)
        info["pcie_link"] = f"gen{link_generation} x{link_width}"
    except Exception as exc:
        print(f"nvidia-smi clock/link query failed: {exc}")
    return info


def torch_memory_snapshot():
    torch.cuda.synchronize()
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / BYTES_PER_GB, 4),
        "reserved_gb": round(torch.cuda.memory_reserved() / BYTES_PER_GB, 4),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / BYTES_PER_GB, 4),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / BYTES_PER_GB, 4),
    }


def board_memory_snapshot():
    """Device-wide GPU memory. nvidia-smi counts every process on the board, so this
    is not attributable to the benchmark process."""
    null_result = {
        "available": False,
        "device_memory_used_gb": None,
        "device_memory_total_gb": None,
        "device_memory_free_gb": None,
        "reason": None,
    }
    if not have_nvidia_smi():
        return {**null_result, "reason": "nvidia-smi not found"}
    try:
        used_mib, total_mib = query_nvidia_smi("memory.used,memory.total")[0].split(",")
        used_mib = float(used_mib.strip())
        total_mib = float(total_mib.strip())
    except Exception as exc:
        return {**null_result, "reason": f"nvidia-smi memory query failed: {exc}"}
    return {
        "available": True,
        "device_memory_used_gb": round(used_mib / 1024, 4),
        "device_memory_total_gb": round(total_mib / 1024, 4),
        "device_memory_free_gb": round((total_mib - used_mib) / 1024, 4),
        "reason": None,
    }


def module_dtype(module):
    for tensor in module.parameters():
        return str(tensor.dtype)
    for tensor in module.buffers():
        return str(tensor.dtype)
    return None


def unique_footprint_bytes(modules):
    """Sum parameter and buffer bytes across modules, counting each underlying tensor
    once so tied or otherwise shared storage is not double-counted."""
    parameter_bytes = 0
    buffer_bytes = 0
    seen_parameters = set()
    seen_buffers = set()
    for module in modules:
        if module is None:
            continue
        for tensor in module.parameters():
            key = tensor.data_ptr()
            if key in seen_parameters:
                continue
            seen_parameters.add(key)
            parameter_bytes += tensor.numel() * tensor.element_size()
        for tensor in module.buffers():
            key = tensor.data_ptr()
            if key in seen_buffers:
                continue
            seen_buffers.add(key)
            buffer_bytes += tensor.numel() * tensor.element_size()
    return parameter_bytes, buffer_bytes


def component_weight_footprint(pipeline):
    """Per-component parameter and buffer footprint, computed from tensor shapes and
    dtypes and kept separate from runtime allocator memory."""
    named = {
        "transformer": pipeline.transformer,
        "text_encoder": pipeline.text_encoder,
        "connectors": pipeline.connectors,
        "vae": pipeline.vae,
        "audio_vae": pipeline.audio_vae,
        "vocoder": pipeline.vocoder,
        "upsampler": pipeline.upsampler,
        "duration_head": pipeline.duration_head,
    }
    footprint = {}
    for name, module in named.items():
        if module is None:
            footprint[name] = {
                "loaded": False,
                "parameter_footprint_gb": 0.0,
                "buffer_footprint_gb": 0.0,
                "dtype": None,
            }
            continue
        parameter_bytes, buffer_bytes = unique_footprint_bytes([module])
        footprint[name] = {
            "loaded": True,
            "parameter_footprint_gb": round(parameter_bytes / BYTES_PER_GB, 4),
            "buffer_footprint_gb": round(buffer_bytes / BYTES_PER_GB, 4),
            "dtype": module_dtype(module),
        }
    total_parameter_bytes, total_buffer_bytes = unique_footprint_bytes(named.values())
    return (
        footprint,
        round(total_parameter_bytes / BYTES_PER_GB, 4),
        round(total_buffer_bytes / BYTES_PER_GB, 4),
    )


def output_report(video, audio):
    """Shape, value range and a subsample fingerprint of the generated tensors."""
    report = {
        "video_shape": list(video.shape),
        "video_dtype": str(video.dtype),
        "audio_shape": list(audio.shape),
        "audio_dtype": str(audio.dtype),
    }
    bounds = torch.stack([video.detach().min(), video.detach().max(),
                          audio.detach().min(), audio.detach().max()])
    report["all_finite"] = bool(torch.isfinite(bounds).all().item())
    report["value_range"] = {
        "video": [round(float(bounds[0]), 6), round(float(bounds[1]), 6)],
        "audio": [round(float(bounds[2]), 6), round(float(bounds[3]), 6)],
    }
    subsamples = {
        "video": video.detach()[0, :, ::7, ::29, ::37].float().contiguous(),
        "audio": audio.detach()[0, :, ::53].float().contiguous(),
    }
    for name, subsample in subsamples.items():
        variance, mean = torch.var_mean(subsample)
        report[f"{name}_subsample"] = {
            "shape": list(subsample.shape),
            "mean": round(float(mean), 6),
            "std": round(float(variance) ** 0.5, 6),
            "sha256_16": hashlib.sha256(subsample.cpu().numpy().tobytes()).hexdigest()[:16],
        }
    return report


def output_fingerprint(output):
    """Compact per-generation output identity; the full description is reported once."""
    return {
        "video_sha256_16": output["video_subsample"]["sha256_16"],
        "audio_sha256_16": output["audio_subsample"]["sha256_16"],
        "all_finite": output["all_finite"],
        "value_range": output["value_range"],
    }


def output_consistency(measured_runs, bare_runs):
    """Compare every generation's fingerprint, instrumented and clean alike"""
    generations = [
        {
            "group": group,
            "run_index": run["run_index"],
            "video_sha256_16": run["output"]["video_sha256_16"],
            "audio_sha256_16": run["output"]["audio_sha256_16"],
            "all_finite": run["output"]["all_finite"],
        }
        for group, runs in (("measured", measured_runs), ("bare", bare_runs))
        for run in runs
    ]
    video = sorted({generation["video_sha256_16"] for generation in generations})
    audio = sorted({generation["audio_sha256_16"] for generation in generations})
    return {
        "generations_compared": len(generations),
        "distinct_video_fingerprints": len(video),
        "distinct_audio_fingerprints": len(audio),
        "all_finite": all(generation["all_finite"] for generation in generations) if generations else None,
        "identical": (len(video) == 1 and len(audio) == 1) if generations else None,
    }


def stage_pass_roles(stage, settings):
    if stage.startswith(STAGE2_DENOISE):
        return REFINE_PASS_ROLES
    guidance = (
        settings.stg_scale != 0.0 or settings.audio_stg_scale != 0.0,
        settings.modality_scale != 1.0 or settings.audio_modality_scale != 1.0,
    )
    return ("cfg",) + tuple(role for role, enabled in zip(PASS_ROLES[1:], guidance) if enabled)


def denoise_detail(pass_seconds, host_pass_seconds, tokens, roles):
    """Per-step and per-pass breakdown of one denoise stage"""
    passes_per_step = len(roles)
    detail = {
        "transformer_calls": len(pass_seconds),
        "passes_per_step": passes_per_step,
        "steps": None,
        "step_seconds": None,
        "step_seconds_summary": None,
        "host_step_seconds": None,
        "host_step_seconds_summary": None,
        "pass_seconds": {role: [round(value, 4) for value in pass_seconds[index::passes_per_step]]
                         for index, role in enumerate(roles)},
        "host_pass_seconds": {role: [round(value, 4) for value in host_pass_seconds[index::passes_per_step]]
                              for index, role in enumerate(roles)},
        "tokens": {role: tokens.get(role) for role in roles},
        "video_tokens_per_second": {},
        "token_batches_per_step": None,
    }
    if len(pass_seconds) % passes_per_step == 0:
        steps = len(pass_seconds) // passes_per_step
        step_seconds = [sum(pass_seconds[step * passes_per_step:(step + 1) * passes_per_step])
                        for step in range(steps)]
        host_step_seconds = [
            sum(host_pass_seconds[step * passes_per_step:(step + 1) * passes_per_step])
            for step in range(steps)
        ]
        detail["steps"] = steps
        detail["step_seconds"] = [round(value, 4) for value in step_seconds]
        detail["step_seconds_summary"] = summarize_values(step_seconds)
        detail["host_step_seconds"] = [round(value, 4) for value in host_step_seconds]
        detail["host_step_seconds_summary"] = summarize_values(host_step_seconds)
    else:
        print(f"denoise stage made {len(pass_seconds)} transformer calls, not a multiple of "
              f"{passes_per_step}; per-step timing skipped")
    token_batches = 0
    complete = True
    for role in roles:
        info = tokens.get(role)
        seconds = detail["pass_seconds"][role]
        if not info or not info.get("video_tokens") or not seconds:
            complete = False
            continue
        token_batch = info["video_batch"] * info["video_tokens"]
        token_batches += token_batch
        median_seconds = statistics.median(seconds)
        if median_seconds > 0:
            detail["video_tokens_per_second"][role] = round(token_batch / median_seconds, 1)
    if complete:
        detail["token_batches_per_step"] = token_batches
    if passes_per_step == 1:
        detail.pop("transformer_calls")
        detail.pop("pass_seconds")
        detail.pop("host_pass_seconds")
    return detail


def phase_time_relation(device_seconds, caller_seconds):
    """Which of a phase's two durations is the longer, and what it leaves unaccounted"""
    if caller_seconds <= 0.0:
        return {"relation": "gpu_work_only", "gpu_over_caller_ratio": None,
                "caller_seconds_beyond_gpu_work": 0.0}
    return {
        "relation": ("gpu_work_covers_caller_time" if device_seconds >= caller_seconds
                     else "caller_time_exceeds_gpu_work"),
        "gpu_over_caller_ratio": round(device_seconds / caller_seconds, 3),
        "caller_seconds_beyond_gpu_work": max(caller_seconds - device_seconds, 0.0),
    }


def measured_generation_telemetry(samples, runs, boost_clock_mhz, power_limit_watts):
    """GPU state inside each measured generation's wall window

    The session-wide samples also cover the warm-up, the idle gaps between runs and the output
    checks, so only these per-window samples describe the generations whose latency is reported.
    """
    threshold_mhz = round(boost_clock_mhz * CLOCK_WARNING_FRACTION, 2) if boost_clock_mhz else None
    per_generation = []
    values_by_run = []
    for run in runs:
        window_start_s, window_end_s = run["window_monotonic_s"]
        window = [sample for sample in samples
                  if window_start_s <= sample["timestamp_monotonic_s"] <= window_end_s]
        values = {
            metric.sample_key: [sample[metric.sample_key] * metric.scale for sample in window
                                if sample.get(metric.sample_key) is not None]
            for metric in SAMPLER_METRICS
        }
        values_by_run.append(values)
        clock_values = values[SM_CLOCK_METRIC.sample_key]
        clock_median = round(statistics.median(clock_values), SM_CLOCK_METRIC.digits) if clock_values else None
        if threshold_mhz is None:
            clock_verdict = "unknown"
        elif len(clock_values) < MIN_CLOCK_SAMPLES:
            clock_verdict = "insufficient_samples"
        elif clock_median < threshold_mhz:
            clock_verdict = "below_threshold"
        else:
            clock_verdict = "sustained"
        power_values = values[POWER_METRIC.sample_key]
        per_generation.append({
            "run_index": run["run_index"],
            "sample_count": len(window),
            "sm_clock_median_mhz": clock_median,
            "sm_clock_min_mhz": round(min(clock_values), SM_CLOCK_METRIC.digits) if clock_values else None,
            "sm_clock_valid_sample_count": len(clock_values),
            "sm_clock_verdict": clock_verdict,
            "power_median_watts": round(statistics.median(power_values), POWER_METRIC.digits) if power_values else None,
            "power_max_watts": round(max(power_values), POWER_METRIC.digits) if power_values else None,
            "power_valid_sample_count": len(power_values),
            "throttle_reasons": {
                flag.sample_key: {
                    "active_sample_count": sum(1 for sample in window if sample.get(flag.sample_key) is True),
                    "inactive_sample_count": sum(1 for sample in window if sample.get(flag.sample_key) is False),
                    "unknown_sample_count": sum(1 for sample in window if sample.get(flag.sample_key) is None),
                }
                for flag in THROTTLE_FLAGS
            },
        })

    metrics = {}
    for metric in SAMPLER_METRICS:
        pooled = [value for values in values_by_run for value in values[metric.sample_key]]
        metrics[metric.sample_key] = {
            "unit": metric.unit,
            "valid_sample_count": len(pooled),
            "median": round(statistics.median(pooled), metric.digits) if pooled else None,
            "min": round(min(pooled), metric.digits) if pooled else None,
            "max": round(max(pooled), metric.digits) if pooled else None,
            "p95": round(percentile_nearest_rank(pooled, 0.95), metric.digits) if pooled else None,
        }

    verdicts = [detail["sm_clock_verdict"] for detail in per_generation]
    if not verdicts or "unknown" in verdicts:
        clock_verdict = "unknown"
    elif "below_threshold" in verdicts:
        clock_verdict = "below_threshold"
    elif "insufficient_samples" in verdicts:
        clock_verdict = "insufficient_samples"
    else:
        clock_verdict = "sustained"
    run_medians = [detail["sm_clock_median_mhz"] for detail in per_generation
                   if detail["sm_clock_median_mhz"] is not None]

    throttle_reasons = {}
    for flag in THROTTLE_FLAGS:
        counts = {
            key: sum(detail["throttle_reasons"][flag.sample_key][key] for detail in per_generation)
            for key in ("active_sample_count", "inactive_sample_count", "unknown_sample_count")
        }
        counts["verdict"] = (
            "active" if counts["active_sample_count"]
            else "inactive" if counts["inactive_sample_count"]
            else "unknown"
        )
        throttle_reasons[flag.sample_key] = counts

    pooled_power = [value for values in values_by_run for value in values[POWER_METRIC.sample_key]]
    if power_limit_watts is None or not pooled_power:
        power_verdict = "unknown"
    elif statistics.median(pooled_power) >= power_limit_watts * POWER_LIMIT_FRACTION:
        power_verdict = "at_limit"
    else:
        power_verdict = "below_limit"

    return {
        "scope": "each measured generation's wall window; warm-up, idle gaps and output checks excluded",
        "sample_count": sum(detail["sample_count"] for detail in per_generation),
        "sm_clock": {
            "warning_threshold_mhz": threshold_mhz,
            "warning_fraction": CLOCK_WARNING_FRACTION,
            "minimum_valid_samples": MIN_CLOCK_SAMPLES,
            "median_of_run_medians_mhz": (
                round(statistics.median(run_medians), SM_CLOCK_METRIC.digits) if run_medians else None
            ),
            "runs_below_threshold": [detail["run_index"] for detail in per_generation
                                     if detail["sm_clock_verdict"] == "below_threshold"],
            "verdict": clock_verdict,
        },
        "power_limit": {
            "verdict": power_verdict,
        },
        "throttle_reasons": throttle_reasons,
        "per_generation": per_generation,
        "metrics": metrics,
    }


def effective_inference_steps(settings):
    return max(int(settings.num_inference_steps), 2)


def build_settings(engine_config, prompt):
    settings = copy.copy(engine_config.generation)
    settings.prompt = prompt
    settings.duration_seconds = FIXED_DURATION_SECONDS
    settings.higher_resolution = True
    settings.image_path = ""
    print(f"benchmark settings: duration={settings.duration_seconds}s "
          f"stage1_steps={effective_inference_steps(settings)} "
          f"size={settings.height}x{settings.width} fps={engine_config.frame_rate} seed={settings.seed} "
          f"higher_resolution={settings.higher_resolution}")
    return settings


def expected_phase_calls(settings):
    calls = {
        TEXT_ENCODE: 2,
        CONNECTORS: 2 if settings.higher_resolution else 1,
        STAGE1_DENOISE: 1,
        DECODE: 3,
        WEIGHT_OFFLOAD: len(OFFLOADED_MODULES),
        WEIGHT_RELOAD: len(OFFLOADED_MODULES),
        CACHE_CLEAR: 1,
    }
    if settings.higher_resolution:
        calls[UPSAMPLER] = 1
        calls[STAGE2_DENOISE] = 1
        calls[LORA_APPLY] = 2
        calls[LORA_REMOVE] = 2
    return calls


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline end-to-end inference benchmark: Stage 1 + Stage 2 + VAE decode.")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS,
                        help=f"discarded warm-up generations before measurement (default {DEFAULT_WARMUP_RUNS})")
    parser.add_argument("--measured-runs", type=int, default=DEFAULT_MEASURED_RUNS,
                        help=f"measured generations (default {DEFAULT_MEASURED_RUNS})")
    parser.add_argument("--bare-runs", type=int, default=DEFAULT_BARE_RUNS,
                        help=f"clean reference generations after the measured runs, with no component "
                             f"timing and no gpu telemetry (default {DEFAULT_BARE_RUNS})")
    parser.add_argument("--emit-raw-samples", action="store_true",
                        help="write every gpu telemetry sample as CSV beside the report; the report "
                             "itself keeps only the summaries (default: off)")
    args = parser.parse_args()
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.measured_runs < 1:
        parser.error("--measured-runs must be >= 1")
    if args.bare_runs < 0:
        parser.error("--bare-runs must be >= 0")
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


def write_raw_samples_csv(output_path, samples, started_at_s):
    """Write every telemetry sample as CSV beside the report. The report keeps the summaries
    computed from these; this is the series itself, one row per nvidia-smi poll."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    csv_path = destination.with_name(f"{destination.stem}_raw_samples.csv")
    temporary = destination.parent / f".{csv_path.name}.tmp"
    columns = [metric.sample_key for metric in SAMPLER_METRICS] + [flag.sample_key for flag in THROTTLE_FLAGS]
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_s", "stage", *columns])
            for sample in samples:
                row = [
                    round(sample["timestamp_monotonic_s"] - started_at_s, 3),
                    sample.get("stage"),
                ]
                for metric in SAMPLER_METRICS:
                    value = sample.get(metric.sample_key)
                    row.append("" if value is None else round(value * metric.scale, metric.digits))
                for flag in THROTTLE_FLAGS:
                    value = sample.get(flag.sample_key)
                    row.append("" if value is None else int(value))
                writer.writerow(row)
        temporary.replace(csv_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return csv_path


class GpuSampler:

    def __init__(self, interval_ms=SAMPLER_INTERVAL_MS):
        self.interval_ms = interval_ms
        self.query_fields = ",".join(
            [metric.field for metric in SAMPLER_METRICS] + [flag.field for flag in THROTTLE_FLAGS]
        )
        self.expected_field_count = len(SAMPLER_METRICS) + len(THROTTLE_FLAGS)
        self.samples = []
        self.stage = "startup"
        self.failed_line_count = 0
        self.unavailable_reason = None
        self.started_at_s = None
        self.ended_at_s = None
        self.process = None
        self.reader = None
        self.exit_code = None

    def set_stage(self, stage):
        self.stage = stage

    def start(self):
        """Start one long-lived nvidia-smi stream so telemetry can run during the measured generations"""
        self.started_at_s = time.monotonic()
        if not have_nvidia_smi():
            self.unavailable_reason = "nvidia-smi not found"
            print("gpu telemetry disabled: nvidia-smi not found")
            return False
        command = [
            "nvidia-smi", f"--query-gpu={self.query_fields}", "--format=csv,noheader,nounits",
            "-lms", str(self.interval_ms),
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_stream, daemon=True)
        self.reader.start()
        print(f"gpu telemetry streaming every {self.interval_ms}ms")
        return True

    def _read_stream(self):
        for line in self.process.stdout:
            row = line.strip().split(",")
            if len(row) != self.expected_field_count:
                self.failed_line_count += 1
                if self.failed_line_count <= 3:
                    print(f"gpu telemetry dropped unexpected line: {line.strip()!r}")
                continue
            sample = {"timestamp_monotonic_s": time.monotonic(), "stage": self.stage}
            for metric, raw_value in zip(SAMPLER_METRICS, row):
                sample[metric.sample_key] = parse_optional_float(raw_value)
            for flag, raw_value in zip(THROTTLE_FLAGS, row[len(SAMPLER_METRICS):]):
                sample[flag.sample_key] = parse_optional_flag(raw_value)
            self.samples.append(sample)

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=SAMPLER_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print(f"nvidia-smi stream did not exit within {SAMPLER_STOP_TIMEOUT_S}s, killing it")
                self.process.kill()
                self.process.wait(timeout=SAMPLER_STOP_TIMEOUT_S)
            self.exit_code = self.process.returncode
            self.reader.join(timeout=SAMPLER_STOP_TIMEOUT_S)
            if self.reader.is_alive():
                print("gpu telemetry reader thread did not exit after the stream closed")
        self.ended_at_s = time.monotonic()
        if self.unavailable_reason is None and not self.samples:
            self.unavailable_reason = (
                f"nvidia-smi stream produced no usable samples "
                f"(exit code {self.exit_code}, dropped lines {self.failed_line_count})"
            )
            print(f"gpu telemetry unavailable: {self.unavailable_reason}")
        return self.samples

    def summary(self):
        summary = {
            "reason": self.unavailable_reason,
            "scope": "whole session: warm-up, measured generations, idle gaps and output checks",
            "polling_interval_ms": self.interval_ms,
            "successful_sample_count": len(self.samples),
            "failed_line_count": self.failed_line_count,
            "process_exit_code": self.exit_code,
            "sampling_duration_s": (
                round(self.ended_at_s - self.started_at_s, 3)
                if self.started_at_s is not None and self.ended_at_s is not None else None
            ),
        }
        missing_metrics = []
        for metric in SAMPLER_METRICS:
            values = [sample[metric.sample_key] * metric.scale for sample in self.samples
                      if sample.get(metric.sample_key) is not None]
            summary[metric.count_key] = len(values)
            summary[metric.average_key] = round(statistics.mean(values), metric.digits) if values else None
            summary[metric.peak_key] = round(max(values), metric.digits) if values else None
            summary[metric.percentile_key] = (
                round(percentile_nearest_rank(values, 0.95), metric.digits) if values else None
            )
            if not values:
                missing_metrics.append(metric.field)
        summary["missing_metrics"] = missing_metrics
        summary["throttle_reasons"] = {
            flag.sample_key: {
                "active_sample_count": sum(1 for sample in self.samples if sample.get(flag.sample_key) is True),
                "inactive_sample_count": sum(1 for sample in self.samples if sample.get(flag.sample_key) is False),
                "unknown_sample_count": sum(1 for sample in self.samples if sample.get(flag.sample_key) is None),
            }
            for flag in THROTTLE_FLAGS
        }
        return summary


class Instrumenter:
    """Times pipeline phases with a CUDA event pair and a host stopwatch, so both the device
    time a phase spends and the time its caller spends in it are recorded. Installed once,
    reset before every measured run."""

    def __init__(self, pipeline, settings, stage_sink=None):
        self.pipeline = pipeline
        self.settings = settings
        self.stage_sink = stage_sink
        self.restorers = []
        self.original = {}
        self.current_stage = None
        self.stage_depth = 0
        self.records = {}
        self.transformer_inputs = {}
        self.transformer_calls = {}
        self.denoise_call_count = 0

    def _set_stage(self, stage):
        """Label the phase the top-level call is in, so a raw telemetry sample can say what
        the process was running when it was taken."""
        self.current_stage = stage
        if self.stage_sink is not None:
            self.stage_sink(stage or "between_phases")

    def _begin(self, label):
        record = {
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
            "host_start": time.perf_counter(),
            "host_end": None,
        }
        self.records.setdefault(label, []).append(record)
        record["start"].record()
        return record

    @staticmethod
    def _end(record):
        record["end"].record()
        record["host_end"] = time.perf_counter()

    def _leaf(self, label, call, args, kwargs):
        top_level = self.stage_depth == 0
        if top_level:
            self._set_stage(label)
        record = self._begin(label)
        try:
            return call(*args, **kwargs)
        finally:
            self._end(record)
            if top_level:
                self._set_stage(None)

    def _stage(self, label, call, args, kwargs):
        self._set_stage(label)
        self.stage_depth += 1
        record = self._begin(label)
        try:
            return call(*args, **kwargs)
        finally:
            self._end(record)
            self.stage_depth -= 1
            self._set_stage(None)

    def _patch_instance(self, module, attribute, replacement):
        setattr(module, attribute, replacement)
        self.restorers.append(lambda: vars(module).pop(attribute, None))

    def _patch_module_function(self, owner, attribute, label):
        original = getattr(owner, attribute)

        def probe(*args, **kwargs):
            return self._leaf(label, original, args, kwargs)

        setattr(owner, attribute, probe)
        self.restorers.append(lambda: setattr(owner, attribute, original))

    def _to_probe(self, module):
        original_to = module.to

        def probe(*args, **kwargs):
            target = args[0] if args else kwargs.get("device")
            if not isinstance(target, (str, torch.device)):
                return original_to(*args, **kwargs)
            label = WEIGHT_OFFLOAD if str(target).startswith("cpu") else WEIGHT_RELOAD
            return self._leaf(label, original_to, args, kwargs)

        return probe

    def start_run(self):
        self.records = {}
        self.transformer_inputs = {}
        self.transformer_calls = {}
        self.denoise_call_count = 0
        self.stage_depth = 0
        self._set_stage(None)

    def finish_run(self):
        torch.cuda.synchronize()
        self._set_stage(None)
        device_seconds = {}
        host_seconds = {}
        for label, records in self.records.items():
            device_seconds[label] = [
                record["start"].elapsed_time(record["end"]) / 1000.0 for record in records
            ]
            host_seconds[label] = [
                record["host_end"] - record["host_start"] if record["host_end"] is not None else 0.0
                for record in records
            ]
        return {
            "components_seconds": device_seconds,
            "components_host_seconds": host_seconds,
            "transformer_inputs": self.transformer_inputs,
        }

    def install(self):
        self.original = {
            "encode_text": self.pipeline.encode_text,
            "denoise_stage": self.pipeline._denoise_stage,
            "connectors": self.pipeline.connectors.forward,
            "transformer": self.pipeline.transformer.forward,
            "video_decode": self.pipeline.vae.decode,
            "audio_decode": self.pipeline.audio_vae.decode,
            "vocoder": self.pipeline.vocoder.forward,
            "upsampler": self.pipeline.upsampler.forward if self.pipeline.upsampler is not None else None,
        }

        def text_encode_probe(prompt, max_sequence_length):
            return self._leaf(TEXT_ENCODE, self.original["encode_text"], (prompt, max_sequence_length), {})

        def connectors_probe(*args, **kwargs):
            return self._leaf(CONNECTORS, self.original["connectors"], args, kwargs)

        def transformer_probe(*args, **kwargs):
            stage = self.current_stage or "denoise_unattributed"
            call_index = self.transformer_calls.get(stage, 0)
            self.transformer_calls[stage] = call_index + 1
            roles = stage_pass_roles(stage, self.settings)
            role = roles[call_index % len(roles)]
            key = f"{stage}:{role}"
            if key not in self.transformer_inputs:
                video = kwargs.get("hidden_states")
                audio = kwargs.get("audio_hidden_states")
                self.transformer_inputs[key] = {
                    "video_batch": int(video.shape[0]) if video is not None else None,
                    "video_tokens": int(video.shape[1]) if video is not None and video.dim() > 1 else None,
                    "audio_batch": int(audio.shape[0]) if audio is not None else None,
                    "audio_tokens": int(audio.shape[1]) if audio is not None and audio.dim() > 1 else None,
                }
            return self._leaf(f"{stage}:transformer", self.original["transformer"], args, kwargs)

        def denoise_stage_probe(*args, **kwargs):
            self.denoise_call_count += 1
            if self.denoise_call_count == 1:
                label = STAGE1_DENOISE
            elif self.denoise_call_count == 2:
                label = STAGE2_DENOISE
            else:
                label = f"{STAGE2_DENOISE}_call{self.denoise_call_count - 1}"
            return self._stage(label, self.original["denoise_stage"], args, kwargs)

        def video_decode_probe(*args, **kwargs):
            return self._leaf(DECODE, self.original["video_decode"], args, kwargs)

        def audio_decode_probe(*args, **kwargs):
            return self._leaf(DECODE, self.original["audio_decode"], args, kwargs)

        def vocoder_probe(*args, **kwargs):
            return self._leaf(DECODE, self.original["vocoder"], args, kwargs)

        def upsampler_probe(*args, **kwargs):
            return self._leaf(UPSAMPLER, self.original["upsampler"], args, kwargs)

        self._patch_instance(self.pipeline, "encode_text", text_encode_probe)
        self._patch_instance(self.pipeline, "_denoise_stage", denoise_stage_probe)
        self._patch_instance(self.pipeline.connectors, "forward", connectors_probe)
        self._patch_instance(self.pipeline.transformer, "forward", transformer_probe)
        self._patch_instance(self.pipeline.vae, "decode", video_decode_probe)
        self._patch_instance(self.pipeline.audio_vae, "decode", audio_decode_probe)
        self._patch_instance(self.pipeline.vocoder, "forward", vocoder_probe)
        if self.original["upsampler"] is not None:
            self._patch_instance(self.pipeline.upsampler, "forward", upsampler_probe)
        for name in OFFLOADED_MODULES:
            module = getattr(self.pipeline, name)
            if module is not None:
                self._patch_instance(module, "to", self._to_probe(module))
        self._patch_module_function(pipeline_module, "apply_lora", LORA_APPLY)
        self._patch_module_function(pipeline_module, "remove_lora", LORA_REMOVE)
        self._patch_module_function(torch.cuda, "empty_cache", CACHE_CLEAR)

    def restore(self):
        for restore in reversed(self.restorers):
            restore()
        self.restorers = []


def run_measurements(pipeline, settings, instrumenter, sampler, warmup_runs, measured_runs, errors):
    """Warm up, then measure with instrumentation and telemetry active, keeping partial
    results if a generation fails."""
    measurements = {"warmup_runs_completed": 0, "runs": []}
    phase = "warmup"
    current_index = 0
    try:
        print(f"warmup: {warmup_runs} generation(s), discarded; covers kernel activation and allocator growth")
        for index in range(warmup_runs):
            current_index = index + 1
            sampler.set_stage("warmup")
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            video, audio = pipeline.generate(settings)
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - wall_start
            print(f"warmup {current_index}/{warmup_runs}: wall={wall_seconds:.2f}s "
                  f"allocated={torch.cuda.memory_allocated() / BYTES_PER_GB:.3f}GB")
            del video, audio
            measurements["warmup_runs_completed"] = current_index

        instrumenter.install()
        print(f"instrumentation installed on {len(instrumenter.restorers)} call sites")

        phase = "measured"
        print(f"measured: {measured_runs} generation(s) with component timing and gpu telemetry active")
        for run_index in range(1, measured_runs + 1):
            current_index = run_index
            instrumenter.start_run()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_allocated_gb = torch.cuda.memory_allocated() / BYTES_PER_GB
            start_reserved_gb = torch.cuda.memory_reserved() / BYTES_PER_GB
            start_allocator_stats = torch.cuda.memory_stats()
            window_start_s = time.monotonic()
            wall_start = time.perf_counter()
            video, audio = pipeline.generate(settings)
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - wall_start
            window_end_s = time.monotonic()
            peak_allocated_gb = torch.cuda.max_memory_allocated() / BYTES_PER_GB
            peak_reserved_gb = torch.cuda.max_memory_reserved() / BYTES_PER_GB
            allocator_stats = torch.cuda.memory_stats()
            instrumented = instrumenter.finish_run()
            sampler.set_stage("output_check")
            output = output_report(video, audio)
            del video, audio

            components_seconds = instrumented["components_seconds"]
            components_host_seconds = instrumented["components_host_seconds"]
            components = {
                label: {
                    "num_calls": len(seconds),
                    "total_seconds": round(sum(seconds), 3),
                    "max_seconds": round(max(seconds), 3),
                    "host_total_seconds": round(sum(components_host_seconds.get(label, [])), 3),
                    "host_max_seconds": round(max(components_host_seconds.get(label, [0.0])), 3),
                }
                for label, seconds in components_seconds.items()
                if not label.endswith(":transformer")
            }
            denoise = {}
            for label, seconds in components_seconds.items():
                if not label.endswith(":transformer"):
                    continue
                stage = label[: -len(":transformer")]
                roles = stage_pass_roles(stage, settings)
                denoise[stage] = denoise_detail(
                    seconds,
                    components_host_seconds.get(label, []),
                    {role: instrumented["transformer_inputs"].get(f"{stage}:{role}")
                     for role in roles},
                    roles,
                )
            run = {
                "run_index": run_index,
                "wall_seconds": round(wall_seconds, 3),
                "unattributed_seconds": round(
                    wall_seconds - sum(info["host_total_seconds"] for info in components.values()), 3
                ),
                "start_allocated_gb": round(start_allocated_gb, 4),
                "start_reserved_gb": round(start_reserved_gb, 4),
                "peak_allocated_gb": round(peak_allocated_gb, 4),
                "peak_reserved_gb": round(peak_reserved_gb, 4),
                "incremental_peak_allocated_gb": round(peak_allocated_gb - start_allocated_gb, 4),
                "alloc_retries": int(allocator_stats.get("num_alloc_retries", 0)) - int(
                    start_allocator_stats.get("num_alloc_retries", 0)
                ),
                "ooms": int(allocator_stats.get("num_ooms", 0)) - int(
                    start_allocator_stats.get("num_ooms", 0)
                ),
                "components": components,
                "denoise": denoise,
                "output": output,
                "window_monotonic_s": [round(window_start_s, 6), round(window_end_s, 6)],
            }
            measurements["runs"].append(run)
            print(f"run {run_index}/{measured_runs}: wall={wall_seconds:.2f}s "
                  f"unattributed={run['unattributed_seconds']:.2f}s "
                  f"peak_allocated={peak_allocated_gb:.3f}GB peak_reserved={peak_reserved_gb:.3f}GB "
                  f"frames={output['video_shape'][2]}")
            for label, info in sorted(components.items(), key=lambda item: -item[1]["total_seconds"]):
                print(f"  {label}: gpu={info['total_seconds']:.2f}s host={info['host_total_seconds']:.2f}s "
                      f"over {info['num_calls']} call(s)")
            for stage, info in denoise.items():
                print(f"  {stage}: steps={info['steps']} "
                      f"median_step={info['step_seconds_summary']['median'] if info['step_seconds_summary'] else None}s "
                      f"median_host_step="
                      f"{info['host_step_seconds_summary']['median'] if info['host_step_seconds_summary'] else None}s")
    except Exception as exc:
        errors.append({
            "stage": phase,
            "run_index": current_index,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        print(f"benchmark stopped during {phase} (run {current_index}): {exc}")
        traceback.print_exc()
    finally:
        if instrumenter.restorers:
            instrumenter.restore()
            print("instrumentation restored")
    return measurements


def run_bare_measurements(pipeline, settings, bare_runs, errors):
    """Repeat the measured workload with no component timing and no gpu telemetry"""
    measurements = {"runs": []}
    if bare_runs < 1:
        print("bare: no clean reference generations requested")
        return measurements
    print(f"bare: {bare_runs} generation(s) after the instrumented runs, no component timing and no gpu "
          "telemetry")
    phase = "bare_measured"
    current_index = 0
    try:
        for run_index in range(1, bare_runs + 1):
            current_index = run_index
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_allocated_gb = torch.cuda.memory_allocated() / BYTES_PER_GB
            start_reserved_gb = torch.cuda.memory_reserved() / BYTES_PER_GB
            start_allocator_stats = torch.cuda.memory_stats()
            wall_start = time.perf_counter()
            video, audio = pipeline.generate(settings)
            torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - wall_start
            peak_allocated_gb = torch.cuda.max_memory_allocated() / BYTES_PER_GB
            peak_reserved_gb = torch.cuda.max_memory_reserved() / BYTES_PER_GB
            allocator_stats = torch.cuda.memory_stats()
            output = output_report(video, audio)
            del video, audio
            run = {
                "run_index": run_index,
                "wall_seconds": round(wall_seconds, 3),
                "start_allocated_gb": round(start_allocated_gb, 4),
                "start_reserved_gb": round(start_reserved_gb, 4),
                "peak_allocated_gb": round(peak_allocated_gb, 4),
                "peak_reserved_gb": round(peak_reserved_gb, 4),
                "incremental_peak_allocated_gb": round(peak_allocated_gb - start_allocated_gb, 4),
                "alloc_retries": int(allocator_stats.get("num_alloc_retries", 0)) - int(
                    start_allocator_stats.get("num_alloc_retries", 0)
                ),
                "ooms": int(allocator_stats.get("num_ooms", 0)) - int(
                    start_allocator_stats.get("num_ooms", 0)
                ),
                "output": output,
            }
            measurements["runs"].append(run)
            print(f"bare run {run_index}/{bare_runs}: wall={wall_seconds:.2f}s "
                  f"peak_allocated={peak_allocated_gb:.3f}GB peak_reserved={peak_reserved_gb:.3f}GB "
                  f"video_fingerprint={output['video_subsample']['sha256_16']}")
    except Exception as exc:
        errors.append({
            "stage": phase,
            "run_index": current_index,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        print(f"benchmark stopped during {phase} (run {current_index}): {exc}")
        traceback.print_exc()
    return measurements


def aggregate_attribution(runs, settings):
    labels = sorted({label for run in runs for label in run["components"]})
    components = {}
    for label in labels:
        present = [run for run in runs if label in run["components"]]
        totals = [run["components"][label]["total_seconds"] for run in present]
        host_totals = [run["components"][label]["host_total_seconds"] for run in present]
        relations = [phase_time_relation(device, caller)
                     for device, caller in zip(totals, host_totals)]
        shares = [100.0 * caller / run["wall_seconds"] for caller, run in zip(host_totals, present)]
        gpu_shares = [100.0 * device / run["wall_seconds"] for device, run in zip(totals, present)]
        relation_names = sorted({relation["relation"] for relation in relations})
        components[label] = {
            "num_calls": present[0]["components"][label]["num_calls"],
            "total_seconds": summarize_values(totals),
            "host_total_seconds": summarize_values(host_totals),
            "time_relation": relation_names[0] if len(relation_names) == 1 else "mixed",
            "gpu_over_caller_ratio": summarize_values(
                [relation["gpu_over_caller_ratio"] for relation in relations
                 if relation["gpu_over_caller_ratio"] is not None]
            ),
            "caller_seconds_beyond_gpu_work": summarize_values(
                [relation["caller_seconds_beyond_gpu_work"] for relation in relations]
            ),
            "share_of_end_to_end_percent": summarize_values(shares),
            "gpu_share_of_end_to_end_percent": summarize_values(gpu_shares),
        }
    unattributed = [run["unattributed_seconds"] for run in runs]
    unattributed_shares = [100.0 * run["unattributed_seconds"] / run["wall_seconds"] for run in runs]
    denoise = {}
    for stage in sorted({stage for run in runs for stage in run["denoise"]}):
        present = [run["denoise"][stage] for run in runs if stage in run["denoise"]]
        roles = stage_pass_roles(stage, settings)
        entry = {
            "steps": present[0]["steps"],
            "step_seconds": summarize_values(
                [value for detail in present for value in (detail["step_seconds"] or [])]
            ),
            "host_step_seconds": summarize_values(
                [value for detail in present for value in (detail["host_step_seconds"] or [])]
            ),
            "video_tokens_per_second": {
                role: summarize_values([detail["video_tokens_per_second"][role] for detail in present
                                        if role in detail["video_tokens_per_second"]])
                for role in roles
            },
            "tokens": present[0]["tokens"],
            "token_batches_per_step": present[0]["token_batches_per_step"],
        }
        if len(roles) > 1:
            entry["transformer_calls"] = present[0]["transformer_calls"]
            entry["pass_seconds"] = {
                role: summarize_values([value for detail in present for value in detail["pass_seconds"][role]])
                for role in roles
            }
            entry["host_pass_seconds"] = {
                role: summarize_values(
                    [value for detail in present for value in detail["host_pass_seconds"][role]]
                )
                for role in roles
            }
        denoise[stage] = entry
    return {
        "unattributed_seconds": summarize_values(unattributed),
        "unattributed_share_percent": summarize_values(unattributed_shares),
        "components": components,
        "denoise": denoise,
    }


def workload_report(settings, run, frame_rate):
    output = run["output"] if run is not None else None
    denoise = run["denoise"] if run is not None else {}
    shape = output["video_shape"] if output is not None else None
    frames = output["video_shape"][2] if output is not None else None
    stage1 = denoise.get(STAGE1_DENOISE) or {}
    stage2 = denoise.get(STAGE2_DENOISE) or {}
    steps = [stage1.get("steps"), stage2.get("steps")]
    passes_per_step = [stage1.get("passes_per_step"), stage2.get("passes_per_step")]
    passes = [
        step * per_step if step is not None and per_step is not None else None
        for step, per_step in zip(steps, passes_per_step)
    ]
    return {
        "stage1_inference_steps": steps[0],
        "stage2_inference_steps": steps[1],
        "total_denoise_steps": sum(steps) if None not in steps else None,
        "total_transformer_passes": sum(passes) if None not in passes else None,
        "frame_rate": frame_rate,
        "stage1_generation_size": f"{settings.height}x{settings.width}",
        "output_frames": frames,
        "output_duration_seconds": round(frames / frame_rate, 3) if frames else None,
        "output_size": f"{shape[3]}x{shape[4]}" if shape else None,
    }


def throughput_report(wall_stats, frames, frame_rate):
    """Generation rate for one clip. The realtime factor is how much compute one second of
    finished video costs."""
    seconds = wall_stats["median"]
    if seconds is None or not frames:
        return {
            "seconds_per_frame": None,
            "frames_per_second": None,
            "realtime_factor": None,
        }
    output_seconds = frames / frame_rate
    return {
        "seconds_per_frame": round(seconds / frames, 4),
        "frames_per_second": round(frames / seconds, 4),
        "realtime_factor": round(seconds / output_seconds, 3),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this benchmark, none found")
    visible_devices = torch.cuda.device_count()
    if visible_devices != 1:
        raise SystemExit(f"benchmark expects exactly one visible CUDA device, found {visible_devices}")
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} python={platform.python_version()} "
          f"date={datetime.now(timezone.utc).isoformat()}")

    environment = environment_info()
    print(f"gpu={environment['torch_device_name']!r} driver={environment['driver_version']} "
          f"memory={environment['total_memory_gb']}GB max_sm_clock={environment['max_sm_clock_mhz']}MHz "
          f"pcie={environment['pcie_link']}")

    pre_load_torch = torch_memory_snapshot()
    pre_load_device_memory = board_memory_snapshot()
    print(f"pre-load: torch allocated={pre_load_torch['allocated_gb']}GB "
          f"reserved={pre_load_torch['reserved_gb']}GB "
          f"device_used={pre_load_device_memory['device_memory_used_gb']}GB")

    engine_config = EngineConfig()
    engine_config.generation.duration_seconds = FIXED_DURATION_SECONDS
    engine_config.generation.higher_resolution = True
    torch.cuda.synchronize()
    load_start = time.perf_counter()
    pipeline = load_pipeline_from_checkpoint(
        engine_config=engine_config, checkpoint_dir=args.checkpoint_dir, device=DEVICE, dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    print(f"model loading took {load_seconds:.2f}s, excluded from generation latency")

    for module in (pipeline.transformer, pipeline.text_encoder, pipeline.connectors, pipeline.vae,
                   pipeline.audio_vae, pipeline.vocoder, pipeline.upsampler):
        if module is not None:
            module.eval()
    if pipeline.duration_head is not None:
        print("duration_head resident but unused: the fixed duration skips the prediction it exists for")
    else:
        print("duration_head not loaded, although every checkpoint the engine loads carries one")

    footprint, parameter_footprint_gb, buffer_footprint_gb = component_weight_footprint(pipeline)
    post_load_torch = torch_memory_snapshot()
    post_load_device_memory = board_memory_snapshot()
    print(f"post-load: torch allocated={post_load_torch['allocated_gb']}GB "
          f"reserved={post_load_torch['reserved_gb']}GB "
          f"device_used={post_load_device_memory['device_memory_used_gb']}GB "
          f"parameters={parameter_footprint_gb}GB buffers={buffer_footprint_gb}GB")
    for name, info in footprint.items():
        print(f"component footprint {name}: {info}")

    settings = build_settings(engine_config, TEXT_ONLY_PROMPT)
    errors = []
    sampler = GpuSampler()
    sampler.start()
    instrumenter = Instrumenter(pipeline, settings, stage_sink=sampler.set_stage)
    measurements = run_measurements(
        pipeline, settings, instrumenter, sampler, args.warmup_runs, args.measured_runs, errors,
    )
    sampler.stop()
    telemetry = sampler.summary()
    if args.emit_raw_samples:
        csv_path = write_raw_samples_csv(args.output_json, sampler.samples, sampler.started_at_s)
        print(f"raw telemetry samples written to {csv_path}")

    bare_measurements = run_bare_measurements(pipeline, settings, args.bare_runs, errors)

    runs = measurements["runs"]
    wall_series = [run["wall_seconds"] for run in runs]
    wall_stats = summarize_values(wall_series)
    bare_wall_series = [run["wall_seconds"] for run in bare_measurements["runs"]]
    bare_wall_stats = summarize_values(bare_wall_series)
    attribution = aggregate_attribution(runs, settings) if runs else {
        "unattributed_seconds": summarize_values([]),
        "unattributed_share_percent": summarize_values([]),
        "components": {},
        "denoise": {},
    }
    generation_settings = asdict(settings)
    for field_name in ("image_path", "image_strength", "output_name"):
        generation_settings.pop(field_name)
    last_run = runs[-1] if runs else None
    last_output = last_run["output"] if last_run is not None else None
    frames = last_output["video_shape"][2] if last_output is not None else None
    workload = workload_report(settings, last_run, engine_config.frame_rate)
    throughput = throughput_report(wall_stats, frames, engine_config.frame_rate)
    for run in runs + bare_measurements["runs"]:
        run["output"] = output_fingerprint(run["output"])
    consistency = output_consistency(runs, bare_measurements["runs"])
    telemetry["measured_generations"] = measured_generation_telemetry(
        sampler.samples, runs, environment["max_sm_clock_mhz"], environment["power_limit_watts"],
    )
    measured = telemetry["measured_generations"]
    clock_check = measured["sm_clock"]

    warnings = []
    if wall_stats["spread_percent"] is not None and wall_stats["spread_percent"] > SPREAD_WARNING_PERCENT:
        warnings.append(
            f"wall time spread is {wall_stats['spread_percent']}% across {wall_stats['n']} runs, above "
            f"{SPREAD_WARNING_PERCENT}%; the median is approximate"
        )
    if telemetry["reason"] is not None:
        warnings.append(f"gpu telemetry unavailable: {telemetry['reason']}")
    if runs:
        expected_calls = expected_phase_calls(settings)
        structure = []
        for run in runs:
            actual_calls = {label: info["num_calls"] for label, info in run["components"].items()}
            missing = sorted(set(expected_calls) - set(actual_calls))
            unexpected = sorted(set(actual_calls) - set(expected_calls))
            mismatched = sorted(
                f"{label} ran {actual_calls[label]}x, expected {expected_calls[label]}x"
                for label in set(expected_calls) & set(actual_calls)
                if actual_calls[label] != expected_calls[label]
            )
            if missing or unexpected or mismatched:
                structure.append(
                    f"run {run['run_index']}: missing={missing} unexpected={unexpected} counts={mismatched}"
                )
        if structure:
            warnings.append(
                "instrumentation no longer matches the pipeline call structure, so the component split is "
                f"untrustworthy: {'; '.join(structure)}"
            )
        if attribution["unattributed_seconds"]["min"] is not None and attribution["unattributed_seconds"]["min"] < -0.05:
            warnings.append("phase windows cover more than the wall time in at least one run, so a phase is "
                            "being timed twice")
        if clock_check["verdict"] == "below_threshold":
            warnings.append(
                f"median sm clock during runs {clock_check['runs_below_threshold']} is below "
                f"{clock_check['warning_threshold_mhz']}MHz ({CLOCK_WARNING_FRACTION:.0%} of the "
                f"{environment['max_sm_clock_mhz']}MHz boost clock), so those runs may be power or thermally limited"
            )
        elif clock_check["verdict"] != "sustained":
            warnings.append(
                f"sm clock during the measured generations could not be judged ({clock_check['verdict']}): "
                f"{measured['metrics']['clock_sm_mhz']['valid_sample_count']} valid samples over "
                f"{len(measured['per_generation'])} runs, "
                f"needs {clock_check['minimum_valid_samples']} per run, "
                f"boost clock {environment['max_sm_clock_mhz']}MHz"
            )
        for reason, counts in measured["throttle_reasons"].items():
            if counts["verdict"] == "active":
                warnings.append(
                    f"{reason} active on {counts['active_sample_count']} of "
                    f"{counts['active_sample_count'] + counts['inactive_sample_count']} measured generation samples"
                )
            elif counts["verdict"] == "unknown":
                warnings.append(f"{reason} unavailable from nvidia-smi, so limiting by it is unverified")
        if measured["power_limit"]["verdict"] == "at_limit":
            warnings.append(
                f"median power {measured['metrics']['power_w']['median']}W is at the "
                f"{environment['power_limit_watts']}W limit during the measured generations"
            )
        elif measured["power_limit"]["verdict"] == "unknown":
            warnings.append("power limit comparison unavailable: no power samples or no reported limit")
        if consistency["identical"] is False:
            warnings.append(
                f"generated outputs are not reproducible: {consistency['distinct_video_fingerprints']} distinct "
                f"video and {consistency['distinct_audio_fingerprints']} distinct audio fingerprints across "
                f"{consistency['generations_compared']} generations, so a latency change cannot be separated "
                "from a change in what was generated"
            )

    report = {
        "status": "ok" if not errors else "partial",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "warnings": warnings,
        "run_config": {
            "warmup_runs": args.warmup_runs,
            "warmup_runs_completed": measurements["warmup_runs_completed"],
            "measured_runs": args.measured_runs,
            "bare_runs": args.bare_runs,
        },
        "environment": environment,
        "identity": {
            "checkpoint_dir": args.checkpoint_dir,
            "repo_id": engine_config.repo_id,
            "generation_settings": generation_settings,
        },
        "load_seconds_excluded": round(load_seconds, 3),
        "workload": workload,
        "throughput": throughput,
        "memory": {
            "pre_load_torch_gb": pre_load_torch,
            "pre_load_device_memory": pre_load_device_memory,
            "post_load_torch_gb": post_load_torch,
            "post_load_device_memory": post_load_device_memory,
            "per_component_footprint_gb": footprint,
            "parameter_footprint_gb": parameter_footprint_gb,
            "buffer_footprint_gb": buffer_footprint_gb,
        },
        "measurements": {
            "wall_seconds": {**wall_stats, "series": wall_series},
            "runs": runs,
            "bare_runs": {
                "wall_seconds": {**bare_wall_stats, "series": bare_wall_series},
                "runs": bare_measurements["runs"],
            },
        },
        "attribution": attribution,
        "gpu_telemetry": telemetry,
        "output": last_output,
        "output_consistency": consistency,
    }
    write_report_atomic(args.output_json, report)
    print(f"report written to {args.output_json}")
    print(f"latency (instrumented): median={wall_stats['median']}s min={wall_stats['min']}s "
          f"max={wall_stats['max']}s spread={wall_stats['spread_percent']}% n={wall_stats['n']}")
    print(f"latency (bare, no component timing and no telemetry): median={bare_wall_stats['median']}s "
          f"min={bare_wall_stats['min']}s max={bare_wall_stats['max']}s "
          f"spread={bare_wall_stats['spread_percent']}% n={bare_wall_stats['n']}")
    print(f"unattributed: median={attribution['unattributed_seconds']['median']}s "
          f"({attribution['unattributed_share_percent']['median']}% of wall)")
    for label, info in sorted(attribution["components"].items(),
                              key=lambda item: -(item[1]["host_total_seconds"]["median"] or 0.0)):
        print(f"  {label}: share={info['share_of_end_to_end_percent']['median']}% "
              f"gpu_share={info['gpu_share_of_end_to_end_percent']['median']}% "
              f"caller={info['host_total_seconds']['median']}s gpu={info['total_seconds']['median']}s "
              f"beyond_gpu={info['caller_seconds_beyond_gpu_work']['median']}s "
              f"calls={info['num_calls']}")
    for stage, info in attribution["denoise"].items():
        pass_summary = ""
        if "pass_seconds" in info:
            pass_summary = " passes_median " + " ".join(
                f"{role}={values['median']}s" for role, values in info["pass_seconds"].items()
            )
        print(f"  {stage}: steps={info['steps']} median_step={info['step_seconds']['median']}s"
              f"{pass_summary} token_batches_per_step={info['token_batches_per_step']}")
    print(f"gpu during measured generations: samples={measured['sample_count']} "
          f"sm_clock_verdict={clock_check['verdict']} boost={environment['max_sm_clock_mhz']}MHz "
          f"warning_threshold={clock_check['warning_threshold_mhz']}MHz")
    for detail in measured["per_generation"]:
        print(f"  run {detail['run_index']}: samples={detail['sample_count']} "
              f"sm_clock_median={detail['sm_clock_median_mhz']}MHz sm_clock_min={detail['sm_clock_min_mhz']}MHz "
              f"valid={detail['sm_clock_valid_sample_count']} verdict={detail['sm_clock_verdict']} "
              f"power_median={detail['power_median_watts']}W power_max={detail['power_max_watts']}W")
    for reason, counts in measured["throttle_reasons"].items():
        print(f"  {reason}: active={counts['active_sample_count']} inactive={counts['inactive_sample_count']} "
              f"unknown={counts['unknown_sample_count']}")
    print(f"  power_limit={environment['power_limit_watts']}W "
          f"power_median={measured['metrics']['power_w']['median']}W "
          f"verdict={measured['power_limit']['verdict']}")
    print(f"gpu telemetry whole session, warm-up and idle gaps included: "
          f"samples={telemetry['successful_sample_count']} "
          f"failed_lines={telemetry['failed_line_count']} missing={telemetry['missing_metrics']}")
    for metric in SAMPLER_METRICS:
        print(f"  {metric.average_key}={telemetry[metric.average_key]}{metric.unit} "
              f"peak={telemetry[metric.peak_key]} p95={telemetry[metric.percentile_key]} "
              f"valid={telemetry[metric.count_key]}")
    if throughput["realtime_factor"] is not None:
        print(f"throughput: {frames} frames ({round(frames / engine_config.frame_rate, 3)}s of video) in "
              f"{wall_stats['median']}s -> {throughput['seconds_per_frame']}s/frame "
              f"({throughput['frames_per_second']} frames/s, {throughput['realtime_factor']}x realtime)")
    if consistency["identical"] is not None:
        print(f"output consistency: {wall_stats['n']} measured + "
              f"{bare_wall_stats['n']} bare generations, "
              f"video={consistency['distinct_video_fingerprints']} distinct, "
              f"audio={consistency['distinct_audio_fingerprints']} distinct, "
              f"identical={consistency['identical']}, all_finite={consistency['all_finite']}")
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        raise SystemExit(f"benchmark finished with {len(errors)} error(s), see the report for details")
    print("benchmark finished")


if __name__ == "__main__":
    main()
