import collections
from concurrent.futures import Future, InvalidStateError
import copy
from dataclasses import dataclass, replace
import dataclasses
import multiprocessing
import multiprocessing.connection
import os
from pathlib import Path
import socket
import threading
import time
import uuid
import torch

from src.config import EngineConfig, GenerationSettings
from src.device_memory import BYTES_PER_GB
from src.engine.protocol import WakeSignal
from src.engine.request import GenerationRequest, GenerationResult, RequestStatus
from src.engine.worker import WorkerConfig, worker_main
from src.logging_utils import get_logger
from src.model.pipeline import LatentGeometry
from src.model.weights import load_model_configs
from src.parallelism.partition import plan_split

logger = get_logger(__name__)

ROLES = ("front", "back") # The two worker roles, one per GPU.
TERMINATE_JOIN_SECONDS = 10.0
KILL_JOIN_SECONDS = 5.0


@dataclass
class _RequestRecord:
    """Everything the engine tracks for one admitted request."""
    request_id: int
    request: GenerationRequest
    settings: GenerationSettings
    future: Future
    status: RequestStatus
    started: float


class _Lane:
    """One request's slot in the pipeline, from the engine's side."""
    def __init__(self, index):
        self.index = index
        self.record = None
        self.request_id = 0
        self.front_released = True
        self.back_released = True
        self.result = None
        self.error = None
        self.cancel_requested = False

    def free(self):
        return self.record is None and self.front_released and self.back_released


class Engine:
    """Owns the two GPU workers, the request queue and every request's lifetime.

    Callers submit onto a bounded queue and wait on their own result handle. A control
    thread owns the workers: it admits a waiting submission into a free lane, forwards the
    workers' events, cancels requests whose handle its caller cancelled, and frees the
    lane once both workers have let the request go.
    """

    def __init__(self, engine_config=None):
        self.engine_config = copy.deepcopy(engine_config) if engine_config is not None else EngineConfig()
        # A lane is one request's slot on both GPUs, so this is also how many requests can
        # occupy different stages of the pipeline at once.
        self.lane_count = self.engine_config.lane_count
        if self.lane_count < 1:
            raise ValueError(f"lane_count must be positive, got {self.lane_count}.")
        self.results_dir = Path(
            self.engine_config.results_dir or Path(__file__).resolve().parents[2] / "results"
        )
        self.model_config, self.checkpoint_dir = load_model_configs(
            engine_config=self.engine_config,
            checkpoint_dir=self.engine_config.checkpoint_dir or None,
        )
        logger.info(f"Checkpoint: {self.checkpoint_dir}")
        self.split = plan_split(
            self.model_config.transformer.num_layers,
            self.engine_config.generation,
            self.engine_config.cache,
        )
        logger.info(f"Pipeline split: {self.split.describe()}")
        self._check_devices()
        self._lanes = [_Lane(index) for index in range(self.lane_count)]
        self._decoding = {}
        self._submissions = collections.deque()
        self._submission_cond = threading.Condition()
        self._wake = WakeSignal()
        self._next_request_id = 1
        self._output_namespace = uuid.uuid4().hex
        self._completed = 0
        self._closed = threading.Event()
        self._connections = {}
        self._processes = {}
        self._memory_reports = {}
        self._start_workers()
        try:
            self._handshake()
        except BaseException:
            # A half that fails to start already holds whatever it managed to load, and its
            # partner is parked waiting for a start message that will never come. Stopping both
            # here is what keeps a failed start from leaving a device occupied until the other
            # half happens to notice its pipe is gone.
            self._stop_workers(self.engine_config.failure_shutdown_grace_seconds)
            self._wake.close()
            raise
        self._control = threading.Thread(target=self._control_loop, name="engine-control", daemon=True)
        self._control.start()
        logger.info(
            f"Engine ready with {self.lane_count} lane(s), two GPUs and room for "
            f"{self.engine_config.queue_capacity} waiting requests"
        )

    def _check_devices(self):
        front = self.engine_config.front_device
        back = self.engine_config.back_device
        if front == back:
            raise ValueError(
                f"The two halves need different devices but both were set to {front}; "
                "set front_device and back_device in the engine configuration."
            )
        available = torch.cuda.device_count()
        if available < 2:
            raise RuntimeError(
                f"Pipeline parallelism needs two CUDA devices but this machine reports {available}."
            )
        for index in (front, back):
            if index >= available:
                raise RuntimeError(f"Device {index} was requested but only {available} CUDA devices are visible.")

    def _lane_allowance_bytes(self):
        """What one resident request needs on each half, at the largest clip this engine accepts"""
        geometry = LatentGeometry.from_configs(self.model_config, self.engine_config)
        grid = geometry.video_grid_for(
            geometry.resolve_frames(self.engine_config.max_seconds),
            self.engine_config.generation.height,
            self.engine_config.generation.width,
        ).refined()
        width = (
            self.model_config.transformer.num_attention_heads
            * self.model_config.transformer.attention_head_dim
        )
        return int(grid.tokens * width * 4 * 2 * self.engine_config.lane_memory_multiple)

    def _check_lane_budget(self):
        """Refuse to start when a device cannot hold as many requests as this engine accepts"""
        required = self._lane_allowance_bytes() * self.lane_count
        shortfall = []
        for role in ROLES:
            report = self._memory_reports.get(role) or {}
            if report.get("free_gb") is None:
                continue
            entry = {
                "role": role,
                "device": report.get("device"),
                "free_gb": report["free_gb"],
                "total_gb": report.get("total_gb"),
                "required_gb": round(required / BYTES_PER_GB, 3),
            }
            logger.info(
                f"The {role} half on {entry['device']} left {entry['free_gb']} GB of "
                f"{entry['total_gb']} GB free for {self.lane_count} lane(s), against a "
                f"{entry['required_gb']} GB allowance"
            )
            if entry["free_gb"] * BYTES_PER_GB < required:
                shortfall.append(entry)
        if shortfall:
            raise RuntimeError(
                f"Not enough device memory for {self.lane_count} request lane(s): "
                + "; ".join(
                    f"the {entry['role']} half on {entry['device']} has {entry['free_gb']} GB "
                    f"free and needs {entry['required_gb']} GB"
                    for entry in shortfall
                )
                + ". Lower lane_count, free the device, or lower lane_memory_multiple when the "
                "requests this engine serves are smaller than its configured maximum."
            )

    def _start_workers(self):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        worker_threads = str(max(1, (os.cpu_count() or 4) // 4))
        os.environ.setdefault("OMP_NUM_THREADS", worker_threads)
        os.environ.setdefault("MKL_NUM_THREADS", worker_threads)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        rendezvous = f"tcp://127.0.0.1:{port}"
        context = multiprocessing.get_context("spawn")
        front_peer, back_peer = context.Pipe()
        for role, rank, device_index, peer_connection in (
            ("front", 0, self.engine_config.front_device, front_peer),
            ("back", 1, self.engine_config.back_device, back_peer),
        ):
            parent_connection, child_connection = context.Pipe()
            config = WorkerConfig(
                role=role,
                rank=rank,
                device_index=device_index,
                checkpoint_dir=str(self.checkpoint_dir),
                results_dir=str(self.results_dir),
                rendezvous=rendezvous,
                engine_config=self.engine_config,
                split=self.split,
            )
            process = context.Process(
                target=worker_main,
                args=(child_connection, peer_connection, config),
                name=f"ltx-{role}",
                daemon=True,
            )
            process.start()
            child_connection.close()
            peer_connection.close()
            self._connections[role] = parent_connection
            self._processes[role] = process
            logger.info(f"Started the {role} worker as pid {process.pid} on device {device_index}")

    def _handshake(self):
        """Wait for both workers, check they were built from the same split, then let them start."""
        startup_timeout = self.engine_config.startup_timeout_seconds
        split = dataclasses.asdict(self.split)
        for role in ROLES:
            message = self._receive(role, startup_timeout)
            if message.get("kind") == "crashed":
                raise RuntimeError(f"The {role} half failed during startup: {message.get('error')}")
            if message.get("kind") != "ready":
                raise RuntimeError(
                    f"The {role} half sent {message.get('kind')!r} where its readiness report was expected."
                )
            if message.get("split") != split:
                raise RuntimeError(
                    "The two halves were built from different partitions: "
                    f"{self.split.describe()} was configured but the {role} half reports {message.get('split')}."
                )
            if message.get("memory") is not None:
                self._memory_reports[role] = message["memory"]
        self._check_lane_budget()
        for role in ROLES:
            self._connections[role].send({"kind": "start", "split": split})

    def _receive(self, role, timeout):
        connection = self._connections[role]
        process = self._processes[role]
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"The {role} worker did not report ready within {timeout:.0f}s; "
                    "loading the checkpoint is the slowest step of startup."
                )
            multiprocessing.connection.wait([connection, process.sentinel], timeout=min(0.5, remaining))
            if connection.poll():
                try:
                    return connection.recv()
                except EOFError as error:
                    # The half died without reporting, which is what a fault inside a kernel
                    # looks like: the exit code and its last log line are all there is.
                    process.join(timeout=2.0)
                    raise RuntimeError(
                        f"The {role} worker exited with code {process.exitcode} before it "
                        "reported ready; see its traceback in the process log."
                    ) from error
            if not process.is_alive():
                process.join(timeout=2.0)
                raise RuntimeError(
                    f"The {role} worker exited with code {process.exitcode} while the engine was "
                    "waiting for it; see its traceback in the process log."
                )

    def submit(self, request):
        """Queue one request and return the handle its result arrives on."""
        if not isinstance(request, GenerationRequest):
            raise TypeError(f"request must be a GenerationRequest, got {type(request).__name__}.")
        future = Future()
        with self._submission_cond:
            while len(self._submissions) >= self.engine_config.queue_capacity and not self._closed.is_set():
                self._submission_cond.wait()
            if self._closed.is_set():
                raise RuntimeError("The engine is closed and cannot accept requests.")
            self._submissions.append((request, future))
        future.add_done_callback(lambda _: self._wake.set())
        self._wake.set()
        return future

    def generate(self, requests):
        """Run one request, or several, and wait only for those results."""
        requests = [requests] if isinstance(requests, GenerationRequest) else list(requests)
        futures = [self.submit(request) for request in requests]
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
        """Let the requests already running finish, then stop both workers."""
        with self._submission_cond:
            first_close = not self._closed.is_set()
            self._closed.set()
            self._submission_cond.notify_all()
        self._wake.set()
        self._control.join()
        self._abandon_waiting()
        self._wake.close()
        if first_close:
            logger.info(f"Engine closed after {self._completed} completed request(s)")

    def _control_loop(self):
        try:
            self._run_control_loop()
        except Exception as error:
            logger.exception("Engine control loop failed; stopping both workers and failing outstanding requests")
            with self._submission_cond:
                self._closed.set()
                self._submission_cond.notify_all()
            for lane in self._lanes:
                if lane.record is not None:
                    lane.record.status = RequestStatus.FAILED
                    self._fail_future(lane.record.future, error)
                    self._release(lane)
            for record in self._decoding.values():
                record.status = RequestStatus.FAILED
                self._fail_future(record.future, error)
            self._decoding.clear()
            self._abandon_waiting(error)
            self._stop_workers(self.engine_config.failure_shutdown_grace_seconds)

    def _run_control_loop(self):
        wait_handles = [
            self._wake.read_fd,
            *(self._connections[role] for role in ROLES),
            *(self._processes[role].sentinel for role in ROLES),
        ]
        while True:
            self._wake.clear()
            self._poll_events()
            self._admit()
            for lane in self._lanes:
                if (
                    lane.record is not None
                    and lane.record.future.cancelled()
                    and not lane.cancel_requested
                ):
                    self._request_cancel(lane)
            active = any(not lane.free() for lane in self._lanes) or bool(self._decoding)
            if self._closed.is_set() and not active:
                self._stop_workers(self.engine_config.shutdown_grace_seconds)
                return
            has_admittable = (
                not self._closed.is_set()
                and any(lane.free() for lane in self._lanes)
                and bool(self._submissions)
            )
            if not has_admittable and not any(self._connections[role].poll() for role in ROLES):
                multiprocessing.connection.wait(wait_handles, timeout=1.0)

    def _poll_events(self):
        for role in ROLES:
            connection = self._connections[role]
            while connection.poll():
                try:
                    event = connection.recv()
                except EOFError as error:
                    process = self._processes[role]
                    process.join(timeout=2.0)
                    raise RuntimeError(
                        f"The {role} worker exited with code {process.exitcode} without reporting "
                        "what happened; see its traceback in the process log."
                    ) from error
                self._handle_event(role, event)
        for role in ROLES:
            process = self._processes[role]
            if not process.is_alive():
                process.join(timeout=2.0)
                raise RuntimeError(
                    f"The {role} worker exited unexpectedly with code {process.exitcode}; "
                    "see its traceback in the process log."
                )

    def _handle_event(self, role, event):
        kind = event.get("kind")
        if kind == "crashed":
            raise RuntimeError(f"The {role} worker failed: {event.get('error')}")
        if kind == "completed":
            self._resolve_decoded(event)
            return
        if kind == "decode_failed":
            self._fail_decoded(event)
            return
        lane = self._lanes[event["lane"]]
        if event.get("request_id") != lane.request_id:
            logger.debug(
                f"Ignoring stale {kind!r} event from {role} for request {event.get('request_id')} "
                f"while lane {lane.index} holds request {lane.request_id}"
            )
            return
        if kind == "prepared":
            if role == "front":
                # The second GPU cannot size its grids until it knows how long the clip is,
                # and the first GPU is the side that resolves a predicted duration.
                self._send("back", {
                    "kind": "frames",
                    "request_id": event["request_id"],
                    "lane": lane.index,
                    "num_frames": event["num_frames"],
                })
            else:
                # The first GPU holds its first state until the second GPU's lane exists,
                # because a state that arrives before that has no lane to be adopted into.
                self._send("front", {
                    "kind": "begin",
                    "request_id": event["request_id"],
                    "lane": lane.index,
                })
            return
        if kind == "denoised":
            record = lane.record
            if record is not None:
                self._decoding[record.request_id] = record
                lane.record = None
            lane.back_released = True
            self._finish_front(lane, decoding=True)
            if lane.front_released and lane.back_released:
                self._release(lane)
            return
        if kind == "released":
            if role == "front":
                lane.front_released = True
            else:
                lane.back_released = True
            if lane.record is None:
                if lane.front_released and lane.back_released:
                    self._release(lane)
            else:
                self._maybe_resolve(lane)
            return
        if kind == "failed":
            if lane.error is None:
                # Both halves can report a failure for the same request. The first report is
                # the cause; the one that follows is usually what the abort did to the other
                # side, so keeping the first is what makes the root error visible.
                lane.error = RuntimeError(event["error"])
            other = "back" if role == "front" else "front"
            self._cancel_half(other, lane)
            return
        logger.warning(f"The {role} worker sent an event the engine does not handle: {kind!r}")

    def _admit(self):
        if self._closed.is_set():
            return
        while True:
            lane = next((candidate for candidate in self._lanes if candidate.free()), None)
            if lane is None:
                return
            with self._submission_cond:
                if not self._submissions:
                    return
                request, future = self._submissions.popleft()
                self._submission_cond.notify()
            if future.cancelled():
                continue
            request_id = self._next_request_id
            self._next_request_id += 1
            settings = replace(
                copy.deepcopy(self.engine_config.generation),
                prompt=request.prompt,
                duration_seconds=request.duration_seconds,
                higher_resolution=request.refine,
                image_path=request.image_path,
                output_name=f"result_{self._output_namespace}_{request_id}.mp4",
            )
            lane.record = _RequestRecord(
                request_id=request_id,
                request=request,
                settings=settings,
                future=future,
                status=RequestStatus.ADMITTED,
                started=time.monotonic(),
            )
            lane.request_id = request_id
            lane.front_released = False
            lane.back_released = False
            lane.result = None
            lane.error = None
            lane.cancel_requested = False
            message = {
                "kind": "admit",
                "request_id": request_id,
                "lane": lane.index,
                "settings": settings,
            }
            for role in ROLES:
                self._send(role, message)
            logger.debug(
                f"Request {request_id} admitted into lane {lane.index}: {settings.duration_seconds}s, "
                f"refine {settings.higher_resolution}, reference image {settings.image_path or 'none'}, "
                f"{len(settings.prompt)} char prompt"
            )

    def _request_cancel(self, lane):
        lane.cancel_requested = True
        if lane.record is not None:
            lane.record.status = RequestStatus.CANCELLED
        logger.info(f"Request {lane.request_id} cancelled by its caller")
        self._cancel_half("front", lane)
        self._cancel_half("back", lane)
        self._maybe_resolve(lane)

    def _cancel_half(self, role, lane):
        released = lane.front_released if role == "front" else lane.back_released
        if released:
            return
        self._send(role, {
            "kind": "cancel",
            "request_id": lane.request_id,
            "lane": lane.index,
            "reason": "cancelled",
        })

    def _finish_front(self, lane, *, decoding=False):
        if lane.front_released:
            return
        self._send("front", {
            "kind": "finish",
            "request_id": lane.request_id,
            "lane": lane.index,
            "decoding": decoding,
        })

    def _resolve_decoded(self, event):
        request_id = event["request_id"]
        record = self._decoding.pop(request_id, None)
        self._send("front", {"kind": "request_finished", "request_id": request_id})
        if record is None:
            return
        self._publish_result(record, event)

    def _fail_decoded(self, event):
        request_id = event["request_id"]
        record = self._decoding.pop(request_id, None)
        self._send("front", {"kind": "request_finished", "request_id": request_id})
        if record is None or record.future.cancelled():
            return
        record.status = RequestStatus.FAILED
        self._fail_future(record.future, RuntimeError(event["error"]))

    def _publish_result(self, record, event):
        result = self._result(record, event)
        if record.future.cancelled():
            # The files were published before the caller gave up; the caller asked for
            # no result, so nothing is left behind for it.
            result.video_path.unlink(missing_ok=True)
            result.audio_path.unlink(missing_ok=True)
            record.status = RequestStatus.CANCELLED
            logger.info(f"Request {record.request_id} cancelled as its result became available")
            return
        try:
            record.future.set_result(result)
        except InvalidStateError:
            if not record.future.cancelled():
                raise
            result.video_path.unlink(missing_ok=True)
            result.audio_path.unlink(missing_ok=True)
            record.status = RequestStatus.CANCELLED
            logger.info(f"Request {record.request_id} cancelled as its result became available")
        else:
            record.status = RequestStatus.COMPLETED
            self._completed += 1
            logger.info(
                f"Request {record.request_id} finished in "
                f"{time.monotonic() - record.started:.1f}s: {result.video_path.name} "
                f"({result.frames} frames, {result.duration_seconds:.2f}s)"
            )

    def _maybe_resolve(self, lane):
        """Resolve a request once both workers have let its lane go."""
        if not (lane.front_released and lane.back_released):
            return
        record = lane.record
        if record is None:
            return
        if lane.error is not None and not record.future.cancelled():
            record.status = RequestStatus.FAILED
            self._fail_future(record.future, lane.error)
        elif lane.result is not None:
            self._publish_result(record, lane.result)
        else:
            record.status = RequestStatus.CANCELLED
            if not record.future.done():
                record.future.cancel()
            logger.info(f"Request {record.request_id} cancelled before it produced a result")
        self._release(lane)

    def _result(self, record, event):
        return GenerationResult(
            request_id=record.request_id,
            request=record.request,
            video_path=Path(event["video"]),
            audio_path=Path(event["audio"]),
            frames=int(event["frames"]),
            duration_seconds=float(event["duration_seconds"]),
        )

    def _release(self, lane):
        logger.debug(f"Releasing lane {lane.index}")
        lane.record = None
        lane.request_id = 0
        lane.front_released = True
        lane.back_released = True
        lane.result = None
        lane.error = None
        lane.cancel_requested = False

    def _stop_workers(self, grace):
        """Ask both workers to stop, escalate to terminate then kill if they outlive grace."""
        for role in ROLES:
            try:
                self._connections[role].send({"kind": "shutdown"})
            except Exception as error:
                logger.warning(f"The {role} worker could not be told to stop: {error}")
        for role, process in self._processes.items():
            process.join(timeout=grace)
            if process.is_alive():
                logger.warning(f"The {role} worker did not stop within {grace:.0f}s; terminating it")
                process.terminate()
                process.join(timeout=TERMINATE_JOIN_SECONDS)
            if process.is_alive():
                logger.error(
                    f"The {role} worker ignored SIGTERM for {TERMINATE_JOIN_SECONDS:.0f}s; killing it"
                )
                process.kill()
                process.join(timeout=KILL_JOIN_SECONDS)
            self._connections[role].close()

    def _abandon_waiting(self, error=None):
        """Fail every submission that was never admitted into a lane."""
        if error is None:
            error = RuntimeError("The engine was closed before this request was admitted.")
        with self._submission_cond:
            pending = list(self._submissions)
            self._submissions.clear()
            self._submission_cond.notify_all()
        for _, future in pending:
            self._fail_future(future, error)

    @staticmethod
    def _fail_future(future, error):
        if future.done():
            return
        try:
            future.set_exception(error)
        except InvalidStateError:
            if not future.cancelled():
                raise

    def _send(self, role, message):
        self._connections[role].send(message)
