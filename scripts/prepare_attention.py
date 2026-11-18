"""
Prepare SageAttention on the machine that will actually run the engine.

SageAttention is the fast attention kernel the engine uses for stage-2 video
self-attention. It is a compiled CUDA extension and the binary has to match 
the torch version, the CUDA toolkit and the GPU it will run on. This script 
creates and installs that binary, but only when it is needed.

What it does when you run it:

- If SageAttention is already installed, it leaves it alone and skips the build.
- If it is missing, it installs a matching pre-built package from vendor/wheels, or, when
  there is none, builds SageAttention v2.2.0 from the pinned source and caches it there for
  next time. The build applies two small fixes that torch 2.14 needs to compile the source.
- Finally it runs the kernel on the GPU with small random tensors and compares the output
  with PyTorch's own attention. It rejects NaNs, the wrong output shape, or numbers that are
  too far from the reference. This is the step that proves the binary really works on this
  GPU.

Usage:

    uv run python -m scripts.prepare_attention
    uv run python -m scripts.prepare_attention --force
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import torch

from src.model.attention_backend import (
    MANIFEST_NAME,
    SAGE_COMPUTE_CAPABILITY,
    SAGE_KERNEL,
    SAGE_MODULE,
    WHEEL_DIR,
    environment_identity,
    read_manifest,
    verify_attention_backends,
)

SAGE_REPO = "https://github.com/thu-ml/SageAttention.git"
SAGE_REF = "v2.2.0"
SAGE_COMMIT = "eb615cf6cf4d221338033340ee2de1c37fbdba4a"

_COMMANDS = []


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Install the SageAttention kernel if it is missing, then verify it on this GPU."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild and reinstall even when SageAttention is already installed",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "SageAttention must be prepared on the machine that will run it, and this one has "
            "no CUDA device."
        )

    identity = environment_identity()
    print(
        f"environment: torch {identity['torch']} (CUDA {identity['torch_cuda']}), "
        f"Python {identity['python_abi']}, {identity['platform']}"
    )

    installed = _installed_version()
    if installed is not None and not args.force:
        print(f"{SAGE_MODULE} {installed} is already installed; skipping the build")
        _import_check()
    else:
        wheel, _ = _cached_wheel(identity)
        if wheel is not None and not args.force:
            print(f"installing the cached {wheel.name} from {WHEEL_DIR}")
            _install(wheel)
            _import_check()
        else:
            _build(identity)

    summary = verify_attention_backends(torch.device("cuda", torch.cuda.current_device()))
    print(
        f"verified {summary['kernel']} {summary['version']} on {summary['device']}: relative "
        f"error {summary['relative_error']} against PyTorch attention"
    )
    return 0


def _build(identity):
    """Compile the pinned source into a wheel, install it and cache it with its manifest."""
    cuda_home = _probe_build_environment()
    arch = f"{SAGE_COMPUTE_CAPABILITY[0]}.{SAGE_COMPUTE_CAPABILITY[1]}"
    print(f"building {SAGE_MODULE} {SAGE_REF} for {arch}, this takes a few minutes")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sage-attention-build-") as directory:
        workdir = Path(directory)
        source = workdir / "SageAttention"
        _prepare_source(source)
        _install_build_tools()
        dist = workdir / "dist"
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(dist),
                str(source),
            ],
            env=_build_environment(cuda_home),
        )
        wheels = sorted(dist.glob("*.whl"))
        if not wheels:
            raise RuntimeError(f"the build wrote no wheel into {dist}")
        if len(wheels) > 1:
            raise RuntimeError(f"the build wrote several wheels: {[wheel.name for wheel in wheels]}")
        built = wheels[0]
        package = _inspect_wheel(built)
        _install(built)
        import_check = _import_check()
        manifest = {
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": identity,
            "source": {
                "repository": SAGE_REPO,
                "ref": SAGE_REF,
                "commit": SAGE_COMMIT,
                "patches": [
                    "setup.py: -std=c++17 -> -std=c++20",
                    "csrc/utils.cuh: IntArrayRef == -> .equals()",
                ],
            },
            "torch_cuda_arch_list": arch,
            "package": package,
            "wheel": built.name,
            "bytes": built.stat().st_size,
            "import_check": import_check,
            "build_seconds": round(time.perf_counter() - started, 1),
            "build_commands": list(_COMMANDS),
        }
        destination = _publish(built, manifest)
    return destination, manifest


def _probe_build_environment():
    """Everything the compile needs, checked at once so one run reports every missing piece"""
    missing = []
    if platform.system() != "Linux":
        missing.append(f"a Linux build host (this is {platform.system()})")
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        located = shutil.which("nvcc")
        if located is None:
            missing.append("nvcc from a CUDA toolkit (set CUDA_HOME or put it on PATH)")
        else:
            cuda_home = Path(located).resolve().parent.parent
    if shutil.which("g++") is None:
        missing.append("g++ as nvcc's host compiler (install build-essential)")
    if shutil.which("git") is None:
        missing.append("git, to clone the pinned SageAttention source")
    if missing:
        raise RuntimeError(
            "SageAttention cannot be built on this machine. Missing:\n  - "
            + "\n  - ".join(missing)
        )
    return cuda_home


def _build_environment(cuda_home):
    """The environment nvcc and torch's extension builder read while compiling."""
    return {
        **os.environ,
        # Without this, a host with no GPU falls back to torch's default architecture list and
        # produces a binary that fails at its first launch instead of at build time.
        "TORCH_CUDA_ARCH_LIST": f"{SAGE_COMPUTE_CAPABILITY[0]}.{SAGE_COMPUTE_CAPABILITY[1]}",
        "CUDA_HOME": str(cuda_home),
        "CC": "gcc",
        "CXX": "g++",
        "CUDAHOSTCXX": shutil.which("g++"),
    }


def _prepare_source(source):
    """Clone the pinned tree and apply the two fixes torch 2.14 requires to build it."""
    _run(["git", "clone", "--depth", "1", "--branch", SAGE_REF, SAGE_REPO, str(source)])
    checked_out = _run(["git", "-C", str(source), "rev-parse", "HEAD"]).stdout.strip()
    if checked_out != SAGE_COMMIT:
        raise RuntimeError(f"{SAGE_REF} resolved to {checked_out}, but the patches target {SAGE_COMMIT}")

    # SageAttention still passes -std=c++17 to both compilers, while torch 2.14's ATen headers
    # require C++20.
    setup_path = source / "setup.py"
    setup = setup_path.read_text()
    if "-std=c++17" not in setup:
        raise RuntimeError(f"{setup_path} no longer contains -std=c++17; the C++20 patch does not apply")
    setup_path.write_text(setup.replace("-std=c++17", "-std=c++20"))

    # Its CHECK_SHAPE macro compares two IntArrayRef values with ==, which torch 2.14 makes
    # ambiguous.
    utils_path = source / "csrc" / "utils.cuh"
    utils = utils_path.read_text()
    comparison = "x.sizes() == torch::IntArrayRef({__VA_ARGS__})"
    if comparison not in utils:
        raise RuntimeError(
            f"{utils_path} no longer contains the CHECK_SHAPE comparison; the patch does not apply"
        )
    utils_path.write_text(utils.replace(comparison, "x.sizes().equals(torch::IntArrayRef({__VA_ARGS__}))"))


def _install_build_tools():
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
        _run(_installer() + ["pip"])
    _run(_installer() + ["--upgrade", "setuptools", "wheel", "packaging", "ninja"])


def _installer():
    """The command that installs into this interpreter: uv when present, pip otherwise."""
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable]
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
        raise RuntimeError(
            f"Neither uv nor pip is available to install into {sys.executable}; put uv or pip "
            "on PATH and try again."
        )
    return [sys.executable, "-m", "pip", "install"]


def _install(wheel):
    """Install the wheel into this interpreter, replacing any copy already there."""
    _run(_installer() + ["--no-deps", "--force-reinstall", str(wheel)])


def _import_check():
    """Import the installed package in a fresh interpreter and look for the pinned kernel."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {SAGE_MODULE} as sage\n"
            "print(sage.__file__)\n"
            "print(getattr(sage, '__version__', 'unknown'))\n"
            f"print(hasattr(sage, '{SAGE_KERNEL}'))",
        ],
        capture_output=True,
        text=True,
    )
    record = {
        "returncode": probe.returncode,
        "stdout": probe.stdout.strip()[-2000:],
        "stderr": probe.stderr.strip()[-3000:],
    }
    if probe.returncode != 0:
        raise RuntimeError(
            "the wheel installed but does not import, so the compiled extension will not load:\n"
            f"{record['stderr']}"
        )
    if record["stdout"].splitlines()[-1] != "True":
        raise RuntimeError(
            f"the installed package does not export {SAGE_KERNEL}:\n{record['stdout']}"
        )
    return record


def _inspect_wheel(wheel):
    """Name, version and compiled libraries, read from the wheel"""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = next((name for name in names if name.endswith(".dist-info/METADATA")), None)
        if metadata is None:
            raise RuntimeError(f"{wheel.name} has no METADATA; the build produced no package")
        fields = {}
        for line in archive.read(metadata).decode().splitlines():
            key, separator, value = line.partition(": ")
            if separator and key in ("Name", "Version"):
                fields[key.lower()] = value
        libraries = sorted(name for name in names if name.endswith(".so"))
    if not libraries:
        raise RuntimeError(f"{wheel.name} contains no .so; the CUDA extension was not compiled")
    return {"name": fields.get("name"), "version": fields.get("version"), "libraries": libraries}


def _publish(wheel, manifest):
    """Copy the wheel into the cache and write its manifest only after the copy succeeds."""
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)
    for stale in WHEEL_DIR.glob("*.whl"):
        stale.unlink()
    destination = WHEEL_DIR / wheel.name
    shutil.copy2(wheel, destination)
    # Written after the copy so the manifest never names a wheel that failed to arrive.
    (WHEEL_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"cached {destination.name} ({manifest['bytes']} bytes) and {MANIFEST_NAME}")
    return destination


def _cached_wheel(identity):
    """The cached wheel and manifest, when both still describe this environment."""
    manifest = read_manifest()
    if manifest is None:
        return None, None
    built = manifest.get("environment", {})
    differences = {
        key: {"built_against": built.get(key), "running": value}
        for key, value in identity.items()
        if built.get(key) != value
    }
    if differences:
        print(f"the cached wheel does not match this environment: {json.dumps(differences)}")
        return None, None
    wheel = WHEEL_DIR / manifest.get("wheel", "")
    if not wheel.is_file():
        print(f"the manifest names {wheel.name}, which is not in {WHEEL_DIR}")
        return None, None
    return wheel, manifest


def _installed_version():
    """The version of the package installed in this interpreter, or None when absent."""
    try:
        return importlib.metadata.version(SAGE_MODULE)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(command, env=None):
    """Run one step, failing with the part of its output that explains the failure."""
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    _COMMANDS.append(
        {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "seconds": round(time.perf_counter() - started, 2),
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(command)}` failed with code {completed.returncode}:\n"
            f"{_error_excerpt(completed.stderr)}"
        )
    return completed


def _error_excerpt(stderr):
    lines = stderr.splitlines()
    for index, line in enumerate(lines):
        if "error:" in line or "fatal error" in line or line.startswith("FAILED:"):
            return "\n".join(lines[max(0, index - 5) : index + 25])
    return stderr[-4000:]


if __name__ == "__main__":
    sys.exit(main())
