from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GenerationRequest:
    """One caller-supplied generation request"""
    prompt: str
    duration_seconds: Optional[float] = None
    refine: bool = False
    image_path: str = ""


class RequestStatus(str, Enum):
    """Where a request currently sits in the engine's lifecycle"""
    ADMITTED = "admitted"
    PREPARING = "preparing"
    RUNNING = "running"
    TRANSITIONING = "transitioning"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class GenerationResult:
    """The files one completed request produced, named by its request identifier."""
    request_id: int
    request: GenerationRequest
    video_path: Path
    audio_path: Path
    frames: int
    duration_seconds: float
