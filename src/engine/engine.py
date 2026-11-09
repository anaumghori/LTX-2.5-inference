from concurrent.futures import CancelledError, Future, InvalidStateError
import copy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
import queue
import threading
import time
import uuid
import wave
import torch
import imageio.v2 as imageio

from src.engine.request import GenerationRequest, GenerationResult, RequestStatus
from src.model.attention_backend import verify_attention_backends
from src.model.common import resolve_device
from src.model.config import EngineConfig, GenerationSettings
from src.model.logging_utils import get_logger
from src.model.weights import load_pipeline_from_checkpoint

logger = get_logger(__name__)

# How many requests may wait before submission blocks the caller.
SUBMISSION_CAPACITY = 8
# How many requests may hold a lane at once. One device holds the whole model, so a second
# lane's tables, cache, latents and activations would not fit beside the first;
LANE_COUNT = 1


class _Unit(str, Enum):
    """One dispatchable step of a request's program."""
    PREPARE = "prepare"
    STAGE_ONE = "stage one"
    TRANSITION = "transition"
    STAGE_TWO = "stage two"
    DECODE = "decode"


_UNIT_STATUS = {
    _Unit.PREPARE: RequestStatus.PREPARING,
    _Unit.STAGE_ONE: RequestStatus.RUNNING,
    _Unit.TRANSITION: RequestStatus.TRANSITIONING,
    _Unit.STAGE_TWO: RequestStatus.RUNNING,
    _Unit.DECODE: RequestStatus.DECODING,
}


@dataclass
class _Submission:
    """One request as it arrives from a caller, with the handle its result resolves."""
    request: GenerationRequest
    future: Future


@dataclass
class _RequestRecord:
    """What the engine knows about a request"""
    request_id: int
    request: GenerationRequest
    settings: GenerationSettings
    future: Future
    status: RequestStatus


class _Lane:
    """One request's worth of resident state on the device"""
    def __init__(self, index):
        self.index = index
        self.record = None
        self.context = None
        self.program = ()
        self.cursor = 0
        self.result = None

    def occupied(self):
        return self.record is not None

    def next_unit(self):
        return self.program[self.cursor]


class Engine:
    """Loads the pipeline once and runs requests through a table of lanes.

    Callers submit onto a bounded queue and wait on their own result handle; a dedicated
    control thread owns the pipeline and walks each admitted request through its program
    one unit at a time. Submission therefore stays decoupled from execution, and the
    submission order of one batch has no bearing on the files another batch produces.
    """

    def __init__(self, engine_config=None, results_dir=None, lane_count=LANE_COUNT):
        if lane_count < 1:
            raise ValueError(f"lane_count must be positive, got {lane_count}.")
        self.engine_config = copy.deepcopy(engine_config) if engine_config is not None else EngineConfig()
        # Refinement is selected per request, so its components must be resident even
        # when the engine's default request does not use them.
        self.engine_config.generation.higher_resolution = True
        if results_dir is None:
            results_dir = Path(__file__).resolve().parents[2] / "results"
        self.results_dir = Path(results_dir)
        self.device = resolve_device()
        verification = verify_attention_backends(self.device)
        if verification["verified"]:
            logger.info(
                f"Attention: {verification['kernel']} {verification['version']} for stage-2 video "
                f"self-attention on {verification['device']}, SDPA on every other path"
            )
        else:
            logger.info(f"Attention: {verification['reason']}, SDPA on every path")
        self.pipeline = load_pipeline_from_checkpoint(engine_config=self.engine_config, device=self.device)
        self.lane_count = lane_count
        self._lanes = [_Lane(index) for index in range(lane_count)]
        self._submissions = queue.Queue(maxsize=SUBMISSION_CAPACITY)
        self._submission_lock = threading.Lock()
        self._next_request_id = 1
        self._output_namespace = uuid.uuid4().hex
        self._last_lane_index = -1
        self._completed = 0
        self._closed = threading.Event()
        self._control = threading.Thread(target=self._control_loop, name="engine-control", daemon=True)
        self._control.start()
        logger.info(f"Engine ready with {lane_count} lane(s) and room for {SUBMISSION_CAPACITY} waiting requests")

    def submit(self, request):
        """Queues one request and returns the handle its result arrives on."""
        if not isinstance(request, GenerationRequest):
            raise TypeError(f"request must be a GenerationRequest, got {type(request).__name__}.")
        future = Future()
        submission = _Submission(request, future)
        while True:
            with self._submission_lock:
                if self._closed.is_set():
                    raise RuntimeError("The engine is closed and cannot accept requests.")
                try:
                    self._submissions.put_nowait(submission)
                except queue.Full:
                    pass
                else:
                    return future
            time.sleep(0.1)

    def generate(self, requests):
        """Accepts one request or several, and waits for those results and no others."""
        if isinstance(requests, GenerationRequest):
            requests = [requests]
        else:
            requests = list(requests)
        for request in requests:
            if not isinstance(request, GenerationRequest):
                raise TypeError(f"request must be a GenerationRequest, got {type(request).__name__}.")
        futures = []
        try:
            for request in requests:
                futures.append(self.submit(request))
        except Exception as submission_error:
            for future in futures:
                try:
                    future.result()
                except Exception as request_error:
                    logger.error(f"A submitted request failed while the batch was being drained: {request_error}")
            raise submission_error
        results = []
        first_error = None
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        return results

    def close(self):
        """Lets the requests already running finish, then stops the control thread."""
        with self._submission_lock:
            first_close = not self._closed.is_set()
            self._closed.set()
        self._control.join()
        self._abandon_waiting()
        if first_close:
            logger.info(f"Engine closed after {self._completed} completed request(s)")

    def _control_loop(self):
        try:
            self._run_control_loop()
        except Exception as error:
            logger.exception("Engine control loop failed; failing all outstanding requests")
            self._closed.set()
            for lane in self._lanes:
                if lane.occupied():
                    lane.record.status = RequestStatus.FAILED
                    self._fail_future(lane.record.future, error)
                    self._release(lane)
            self._abandon_waiting(error)

    def _run_control_loop(self):
        while True:
            self._admit()
            lane = self._next_occupied_lane()
            if lane is None:
                if self._closed.is_set():
                    self._abandon_waiting()
                    return
                continue
            if lane.record.future.cancelled():
                lane.record.status = RequestStatus.CANCELLED
                logger.info(f"Request {lane.record.request_id} cancelled before {lane.next_unit().value}")
                self._release(lane)
                continue
            self._run_one_unit(lane)

    def _admit(self):
        """Moves one waiting submission into a free lane, waiting when a lane is free."""
        if self._closed.is_set():
            return
        if self._free_lane() is None:
            return
        try:
            submission = self._submissions.get(timeout=0.1)
        except queue.Empty:
            return
        if submission.future.cancelled():
            return
        if self._closed.is_set():
            self._fail_future(submission.future, RuntimeError("The engine was closed before this request was admitted."))
            return
        lane = self._free_lane()
        request_id = self._next_request_id
        self._next_request_id += 1
        try:
            settings = replace(
                copy.deepcopy(self.engine_config.generation),
                prompt=submission.request.prompt,
                duration_seconds=submission.request.duration_seconds,
                higher_resolution=submission.request.refine,
                image_path=submission.request.image_path,
                output_name=f"result_{self._output_namespace}_{request_id}.mp4",
            )
            program = self._program(settings)
        except Exception as error:
            logger.error(f"Request {request_id} could not be admitted: {type(error).__name__}: {error}")
            self._fail_future(submission.future, error)
            return
        lane.record = _RequestRecord(
            request_id=request_id,
            request=submission.request,
            settings=settings,
            future=submission.future,
            status=RequestStatus.ADMITTED,
        )
        lane.context = None
        lane.result = None
        lane.cursor = 0
        lane.program = program
        logger.info(
            f"Request {request_id} admitted into lane {lane.index}: {settings.duration_seconds}s, "
            f"refine {settings.higher_resolution}, reference image {settings.image_path or 'none'}, "
            f"{len(settings.prompt)} char prompt, program {[unit.value for unit in lane.program]}"
        )

    def _program(self, settings):
        """The unit sequence one request runs, which depends only on its settings."""
        if settings.higher_resolution:
            return (_Unit.PREPARE, _Unit.STAGE_ONE, _Unit.TRANSITION, _Unit.STAGE_TWO, _Unit.DECODE)
        return (_Unit.PREPARE, _Unit.STAGE_ONE, _Unit.DECODE)

    def _free_lane(self):
        for lane in self._lanes:
            if not lane.occupied():
                return lane
        return None

    def _next_occupied_lane(self):
        occupied = [lane for lane in self._lanes if lane.occupied()]
        if not occupied:
            return None
        occupied.sort(key=lambda lane: lane.index)
        lane = next((lane for lane in occupied if lane.index > self._last_lane_index), occupied[0])
        self._last_lane_index = lane.index
        return lane

    def _run_one_unit(self, lane):
        record = lane.record
        unit = lane.next_unit()
        record.status = _UNIT_STATUS[unit]
        started = time.perf_counter()
        try:
            unit_complete = self._execute(lane, unit)
        except Exception as error:
            # A request's failure becomes its caller's exception; the engine itself keeps
            # running so one bad request cannot take the process down with it.
            logger.error(
                f"Request {record.request_id} failed in {unit.value}: {type(error).__name__}: {error}"
            )
            if record.future.cancelled():
                record.status = RequestStatus.CANCELLED
            else:
                record.status = RequestStatus.FAILED
                self._fail_future(record.future, error)
            self._release(lane)
            return
        logger.debug(
            f"Request {record.request_id}: {unit.value} {'finished' if unit_complete else 'step finished'} "
            f"in {time.perf_counter() - started:.2f}s "
            f"on {self.device}"
        )
        if not unit_complete:
            return
        lane.cursor += 1
        if lane.cursor == len(lane.program):
            self._complete(lane)

    def _execute(self, lane, unit):
        if unit is _Unit.PREPARE:
            lane.context = self.pipeline.prepare(lane.record.settings)
        elif unit is _Unit.STAGE_ONE:
            return self.pipeline.run_stage_one_step(lane.context)
        elif unit is _Unit.TRANSITION:
            self.pipeline.transition_to_stage_two(lane.context)
        elif unit is _Unit.STAGE_TWO:
            return self.pipeline.run_stage_two_step(lane.context)
        else:
            video, audio = self.pipeline.decode(lane.context)
            self._write(lane, video, audio)
        return True

    def _write(self, lane, video, audio):
        record = lane.record
        frames = video.shape[2]
        duration_seconds = frames / self.engine_config.frame_rate
        video_path = self.results_dir / record.settings.output_name
        audio_path = video_path.with_suffix(".wav")
        temporary_video_path = video_path.with_name(f".{video_path.stem}.partial{video_path.suffix}")
        temporary_audio_path = audio_path.with_name(f".{audio_path.stem}.partial{audio_path.suffix}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if video_path.exists() or audio_path.exists():
            raise FileExistsError(f"Request {record.request_id} output already exists in {self.results_dir}.")
        video_published = False
        audio_published = False
        try:
            self._save_video(video, temporary_video_path, self.engine_config.frame_rate)
            self._save_audio(audio, temporary_audio_path, self.pipeline.vocoder.config.output_sampling_rate)
            if record.future.cancelled():
                raise CancelledError(f"Request {record.request_id} was cancelled before output publication.")
            temporary_video_path.replace(video_path)
            video_published = True
            temporary_audio_path.replace(audio_path)
            audio_published = True
        except Exception:
            temporary_video_path.unlink(missing_ok=True)
            temporary_audio_path.unlink(missing_ok=True)
            if video_published:
                video_path.unlink(missing_ok=True)
            if audio_published:
                audio_path.unlink(missing_ok=True)
            raise
        lane.result = GenerationResult(
            request_id=record.request_id,
            request=record.request,
            video_path=video_path,
            audio_path=audio_path,
            frames=frames,
            duration_seconds=duration_seconds,
        )
        logger.info(
            f"Request {record.request_id} wrote {video_path.name} with {frames} frames "
            f"({duration_seconds:.2f}s) and {audio_path.name}"
        )

    def _complete(self, lane):
        record = lane.record
        try:
            record.future.set_result(lane.result)
        except InvalidStateError:
            if not record.future.cancelled():
                raise
            record.status = RequestStatus.CANCELLED
            if lane.result is not None:
                lane.result.video_path.unlink(missing_ok=True)
                lane.result.audio_path.unlink(missing_ok=True)
            logger.info(f"Request {record.request_id} cancelled as its result became available")
        else:
            record.status = RequestStatus.COMPLETED
            self._completed += 1
        if record.status is RequestStatus.COMPLETED:
            logger.info(f"Request {record.request_id} completed")
        self._release(lane)

    def _release(self, lane):
        """Drops everything the lane held, which is what returns its memory."""
        logger.debug(f"Releasing lane {lane.index}")
        lane.record = None
        lane.context = None
        lane.program = ()
        lane.cursor = 0
        lane.result = None

    def _abandon_waiting(self, error=None):
        """Fails every submission that was never admitted."""
        if error is None:
            error = RuntimeError("The engine was closed before this request was admitted.")
        while True:
            try:
                submission = self._submissions.get_nowait()
            except queue.Empty:
                return
            self._fail_future(submission.future, error)

    @staticmethod
    def _fail_future(future, error):
        if future.done():
            return
        try:
            future.set_exception(error)
        except InvalidStateError:
            if not future.cancelled():
                raise

    def _save_video(self, video, path, fps):
        frames = video[0].detach().float().cpu()
        frames = (frames.clamp(-1, 1) + 1) / 2 * 255
        frames = frames.to(torch.uint8).permute(1, 2, 3, 0).numpy()
        logger.debug(f"Writing video {frames.shape} at {fps} fps to {path}")
        imageio.mimsave(str(path), list(frames), fps=fps)

    def _save_audio(self, audio, path, sample_rate):
        waveform = audio[0].detach().float().cpu()
        waveform = waveform.clamp(-1, 1)
        samples = (waveform * 32767).to(torch.int16).numpy()
        logger.debug(f"Writing audio {samples.shape} at {sample_rate} Hz to {path}")
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(samples.shape[0])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.T.tobytes())
