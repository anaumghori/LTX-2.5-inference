import contextlib
import dataclasses
from datetime import timedelta
import faulthandler
import os
from pathlib import Path
import queue
import threading
import time
import wave
import imageio.v2 as imageio
import torch
import torch.distributed as dist

from src.config import EngineConfig
from src.device_memory import device_memory
from src.engine.protocol import (
    FIRST_SEQ,
    KIND_ABORT,
    KIND_LATENTS,
    KIND_PASS,
    KIND_REF_TOKENS,
    KIND_REFINE_NOISE,
    KIND_STATE,
    Inbound,
    MessageHeader,
    post_send,
)
from src.model.attention_backend import verify_attention_backends
from src.model.common import default_float_dtype
from src.logging_utils import get_logger
from src.model.weights import load_model_configs
from src.parallelism.partition import PipelineSplit
from src.parallelism.weights import load_back_half, load_front_half
from src.profiling import release_lane, span

logger = get_logger(__name__)

# How long a cancelled worker waits for the other worker's abort before releasing anyway
CANCEL_GRACE_SECONDS = 2.0
# How long the loop sleeps when no lane had anything to do.
IDLE_SLEEP_SECONDS = 0.002
# How often the stall watchdog looks for movement, and the status a stalled worker exits with
STALL_CHECK_SECONDS = 5.0
STALL_EXIT_CODE = 4
RETIRED_REQUESTS = 4


@dataclasses.dataclass
class WorkerConfig:
    """Everything a worker needs, handed over at spawn time."""
    role: str
    rank: int
    device_index: int
    checkpoint_dir: str
    results_dir: str
    rendezvous: str
    engine_config: EngineConfig
    split: PipelineSplit


def worker_main(connection, config):
    """Entry point of a worker process"""
    try:
        worker = FrontWorker(connection, config) if config.role == "front" else BackWorker(connection, config)
        worker.run()
    except BaseException as error:
        logger.exception(f"The {config.role} half failed")
        try:
            connection.send({"kind": "crashed", "role": config.role, "error": f"{type(error).__name__}: {error}"})
        except Exception as report_error:
            # The engine reads this pipe, so a report that cannot be sent leaves the process
            # log as the only record of what went wrong.
            logger.error(f"The {config.role} half could not report its failure to the engine: {report_error}")
        raise
    finally:
        connection.close()


@dataclasses.dataclass
class _DecodeJob:
    """One lane's decode and file writing, running off the event loop."""
    lane: int
    request: int
    half: object
    output_name: str
    started: float
    done: threading.Event = dataclasses.field(default_factory=threading.Event)
    seconds: float = 0.0
    files: dict = dataclasses.field(default_factory=dict)
    error: str = ""


class _Lane:
    """One request's slot on one GPU worker."""
    def __init__(self, index):
        self.index = index
        self.request_id = 0
        self.state = "idle"
        self.settings = None
        self.half = None
        self.num_frames = 0
        self.abort_sent = False
        self.abort_deadline = 0.0
        # When this lane started waiting for the other worker's next message, or 0 while it
        # is not waiting. This is what the message timeout is measured from.
        self.wait_since = 0.0
        # Posted messages whose transfers may still be reading their buffers.
        self.sends = []

    def reset(self):
        """Forget the request while keeping any send that is still in flight alive."""
        self.request_id = 0
        self.state = "idle"
        self.settings = None
        self.half = None
        self.num_frames = 0
        self.abort_sent = False
        self.wait_since = 0.0


class _Worker:
    role = ""

    def __init__(self, connection, config):
        self.connection = connection
        self.config = config
        self.device = torch.device("cuda", config.device_index)
        self.dtype = torch.bfloat16
        self.peer = 1 - config.rank
        self.lanes = {index: _Lane(index) for index in range(config.engine_config.lane_count)}
        self.inbound = None
        self.outbound_seq = FIRST_SEQ
        self.inbound_seq = FIRST_SEQ - 1
        self.retired = []
        self.running = True
        self.last_progress = time.monotonic()

    def run(self):
        faulthandler.enable()
        try:
            self._start_group()
            self._load()
            self._handshake()
            self._start_background()
            threading.Thread(
                target=self._watch_stall, name=f"{self.role}-stall", daemon=True
            ).start()
            self._loop()
        finally:
            self._stop_background()
            self._stop_group()

    def _start_background(self):
        """Start whatever this worker runs beside its event loop."""

    def _stop_background(self):
        """Stop it before the device is torn down."""

    def _parked(self, lane):
        """Whether this lane's work currently runs on another thread."""
        return False

    def _watch_stall(self):
        """Fail the process when no lane or message has moved for too long"""
        while self.running:
            time.sleep(STALL_CHECK_SECONDS)
            if not self.running:
                return
            stalled = time.monotonic() - self.last_progress
            stall_timeout = max(
                self.config.engine_config.stall_timeout_seconds,
                self.config.engine_config.message_timeout_seconds,
            )
            if stalled < stall_timeout:
                continue
            lanes = {lane.index: lane.state for lane in self.lanes.values()}
            logger.error(
                f"The {self.role} half made no progress for {stalled:.0f}s "
                f"(timeout {stall_timeout:.0f}s) with lanes {lanes}; "
                f"dumping every thread's stack and exiting with code {STALL_EXIT_CODE}"
            )
            faulthandler.dump_traceback()
            os._exit(STALL_EXIT_CODE)

    def _start_group(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        timeout = timedelta(seconds=self.config.engine_config.startup_timeout_seconds)
        dist.init_process_group(
            backend="nccl" if self.device.type == "cuda" else "gloo",
            init_method=self.config.rendezvous,
            rank=self.config.rank,
            world_size=2,
            timeout=timeout,
        )
        self.downstream = dist.new_group([0, 1], timeout=timeout)
        self.upstream = dist.new_group([0, 1], timeout=timeout)
        self.send_group, self.recv_group = (
            (self.downstream, self.upstream) if self.role == "front" else (self.upstream, self.downstream)
        )
        self._warm_up_channels()
        logger.debug(f"The {self.role} half joined the data channel as rank {self.config.rank} on {self.device}")

    def _warm_up_channels(self):
        """Exchange one tensor each way so both communicators exist before the first request"""
        for group in (self.downstream, self.upstream):
            outgoing = torch.zeros(1, device=self.device)
            incoming = torch.empty(1, device=self.device)
            works = dist.batch_isend_irecv([
                dist.P2POp(op=dist.isend, tensor=outgoing, peer=self.peer, group=group),
                dist.P2POp(op=dist.irecv, tensor=incoming, peer=self.peer, group=group),
            ])
            for work in works:
                work.wait()

    def _stop_group(self):
        if not dist.is_initialized():
            return
        for group in (getattr(self, "downstream", None), getattr(self, "upstream", None)):
            abort = getattr(group, "abort", None)
            if abort is not None:
                abort()
        dist.destroy_process_group()

    def _handshake(self):
        """Announce this worker and wait for the engine to confirm the split"""
        self._send({
            "kind": "ready",
            "role": self.role,
            "device": str(self.device),
            "split": dataclasses.asdict(self.config.split),
            "memory": device_memory(self.device),
        })
        message = self._receive_control(self.config.engine_config.startup_timeout_seconds)
        kind = message.get("kind")
        if kind == "shutdown":
            # The engine stops both halves before starting them when the other one failed or
            # when the split does not fit its device. This half has nothing left to do, and
            # exiting cleanly keeps its own log from reading as the failure.
            logger.info(f"The engine stopped the {self.role} half before it was started")
            self.running = False
            return
        if kind != "start":
            raise RuntimeError(f"The {self.role} half expected the engine's start message, got {kind!r}.")
        if message.get("split") != dataclasses.asdict(self.config.split):
            raise RuntimeError(
                "The two halves were told different splits; the engine and the workers disagree "
                "about the block partition."
            )
        logger.info(f"The {self.role} half is ready on {self.device}.")

    def _load(self):
        raise NotImplementedError

    def _load_half(self, loader, components):
        """Verify the attention kernels, read the model configs and load this GPU's share."""
        verification = verify_attention_backends(self.device)
        if verification["verified"]:
            logger.info(
                f"Attention: {verification['kernel']} {verification['version']} for stage-2 video "
                f"self-attention on {verification['device']}, SDPA on every other path"
            )
        else:
            logger.info(f"Attention: {verification['reason']}, SDPA on every path")
        model_config, _ = load_model_configs(self.config.engine_config, self.config.checkpoint_dir)
        with default_float_dtype(self.dtype):
            half = loader(
                model_config,
                self.config.checkpoint_dir,
                self.config.engine_config,
                self.config.split,
                self.device,
                self.dtype,
            )
        snapshot = device_memory(self.device)
        logger.debug(
            f"The {self.role} half holds its share of the model on {self.device}: "
            f"{snapshot.get('allocated_gb')} GB in use, {snapshot.get('free_gb')} GB free"
        )
        return half

    def _loop(self):
        while self.running:
            self._read_control()
            self._advance_receive()
            progressed = False
            for lane in self.lanes.values():
                if not self.running:
                    break
                if self._advance(lane):
                    progressed = True
                if self._parked(lane):
                    progressed = True
                self._reap(lane)
            if progressed:
                self.last_progress = time.monotonic()
            else:
                time.sleep(IDLE_SLEEP_SECONDS)

    def _read_control(self):
        while self.connection.poll():
            self._on_control(self.connection.recv())

    def _on_control(self, message):
        kind = message.get("kind")
        if kind == "shutdown":
            self.running = False
            return
        if kind == "cancel":
            self._cancel_lane(message["request_id"], message["lane"], message.get("reason", "cancelled"))
            return
        self._on_lane_control(message)

    def _cancel_lane(self, request_id, lane_index, reason):
        """Stop a request on this worker and unblock the other one"""
        lane = self.lanes.get(lane_index)
        if lane is None or lane.request_id != request_id:
            return
        if lane.state in ("idle", "decoding"):
            return
        if lane.state == "done":
            self._release(lane)
            return
        if lane.state == "aborting":
            return
        logger.debug(f"Request {request_id} on the {self.role} half is aborting ({reason})")
        lane.state = "aborting"
        lane.abort_deadline = time.monotonic() + CANCEL_GRACE_SECONDS
        if not lane.abort_sent:
            self._send_message(lane, KIND_ABORT, [], stage=0, step=0)
            lane.abort_sent = True

    def _advance_abort(self, lane):
        """Wait briefly for the other worker's abort so both streams stay paired."""
        armed = self._wait_for_message(lane)
        if time.monotonic() >= lane.abort_deadline:
            self._release(lane)
        return armed

    def _wait_for_message(self, lane):
        """Arm this worker's receive for the message the lane waits on, and note the wait."""
        if not lane.wait_since:
            lane.wait_since = time.monotonic()
        if self.inbound is None:
            self.inbound = Inbound(self.recv_group, self.peer, self.device)
            return True
        return False

    def _advance_receive(self):
        if self.inbound is None:
            return
        message = self.inbound.poll()
        if message is None:
            waiting = [lane for lane in self.lanes.values() if lane.wait_since]
            if waiting:
                oldest = min(waiting, key=lambda lane: lane.wait_since)
                waited = time.monotonic() - oldest.wait_since
                if waited > self.config.engine_config.message_timeout_seconds:
                    lanes = {lane.index: lane.state for lane in waiting}
                    raise RuntimeError(
                        f"The {self.role} half waited {waited:.0f}s for a message from the other half and "
                        f"none arrived; lane(s) {lanes} were still waiting, so the pipeline is stuck."
                    )
            return
        self.inbound = None
        self.last_progress = time.monotonic()
        header, tensors = message
        if header.seq != self.inbound_seq + 1:
            raise RuntimeError(
                f"The {self.role} half expected message {self.inbound_seq + 1} on the data channel but "
                f"received {header.describe()}; the two halves have lost each other's place."
            )
        self.inbound_seq = header.seq
        lane = self.lanes.get(header.lane)
        if lane is None:
            raise RuntimeError(
                f"The {self.role} half received {header.describe()} for a lane it does not have."
            )
        lane.wait_since = 0.0
        if lane.state == "idle" or lane.request_id != header.request_id:
            if header.request_id not in self.retired:
                raise RuntimeError(
                    f"The {self.role} half dropped {header.describe()} while lane {header.lane} held "
                    f"request {lane.request_id} in state {lane.state!r}; request {header.request_id} was "
                    "never retired here, so the two halves disagree about which request owns the lane."
                )
            logger.debug(f"The {self.role} half dropped a stale {header.describe()} from a retired request.")
            return
        if lane.state == "aborting" and header.kind != KIND_ABORT:
            logger.debug(
                f"The {self.role} half drained a stale {header.describe()} while aborting "
                f"request {header.request_id}."
            )
            return
        self._accept(lane, header, tensors)

    def _send_message(self, lane, kind, tensors, *, stage, step, **fields):
        seq = self.outbound_seq
        self.outbound_seq += 1
        header = MessageHeader(
            request_id=lane.request_id,
            lane=lane.index,
            stage=stage,
            step=step,
            kind=kind,
            payload_count=len(tensors),
            seq=seq,
            **fields,
        )
        send = post_send(self.send_group, self.peer, header, list(tensors), self.device)
        lane.sends.append(send)
        return header

    def _reap(self, lane):
        """Drop the sends the other worker has taken delivery of, releasing their buffers."""
        if lane.sends:
            lane.sends = [send for send in lane.sends if not send.complete()]

    def _send(self, message):
        self.connection.send(message)

    def _receive_control(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if self.connection.poll(0.05):
                return self.connection.recv()
            if time.monotonic() > deadline:
                raise RuntimeError(f"The engine did not answer the {self.role} half within {timeout:.0f}s.")

    def _release(self, lane):
        if lane.request_id:
            self._send({
                "kind": "released",
                "role": self.role,
                "request_id": lane.request_id,
                "lane": lane.index,
            })
            self.retired = (self.retired + [lane.request_id])[-RETIRED_REQUESTS:]
            logger.debug(f"Request {lane.request_id} released the {self.role} lane {lane.index}")
            release_lane(lane.index, lane.request_id)
        lane.reset()

    def _fail(self, lane, error):
        """Report a request-level failure and stop talking to the other worker."""
        logger.exception(f"Request {lane.request_id} failed on the {self.role} half: {type(error).__name__}: {error}")
        if lane.state != "aborting":
            lane.state = "aborting"
            lane.abort_deadline = time.monotonic() + CANCEL_GRACE_SECONDS
            if not lane.abort_sent:
                self._send_message(lane, KIND_ABORT, [], stage=0, step=0)
                lane.abort_sent = True
        self._send({
            "kind": "failed",
            "role": self.role,
            "request_id": lane.request_id,
            "lane": lane.index,
            "error": f"{type(error).__name__}: {error}",
        })
        self._release(lane)

    def _advance(self, lane):
        raise NotImplementedError

    def _accept(self, lane, header, tensors):
        raise NotImplementedError

    def _on_lane_control(self, message):
        raise NotImplementedError

    def _admit_lane(self, lane, message, state):
        """Arm a lane for a request, refusing to overwrite one that is still running."""
        if lane.state != "idle":
            raise RuntimeError(f"Lane {lane.index} was admitted while running request {lane.request_id}.")
        lane.request_id = message["request_id"]
        lane.settings = message["settings"]
        lane.state = state


class FrontWorker(_Worker):
    """Runs the first GPU's work: text encoding, the input projection and the leading blocks."""
    role = "front"

    def __init__(self, connection, config):
        super().__init__(connection, config)
        self.front = None

    def _load(self):
        self.front = self._load_half(
            load_front_half,
            ("transformer", "text_encoder", "connectors", "duration_head"),
        )

    def _on_lane_control(self, message):
        kind = message.get("kind")
        lane = self.lanes[message["lane"]]
        if kind == "admit":
            self._admit_lane(lane, message, "preparing")
        elif kind == "begin":
            if lane.request_id != message["request_id"] or lane.state != "waiting_start":
                return
            self._send_state(lane)
            lane.state = "running"
        elif kind == "finish":
            if lane.request_id == message["request_id"] and lane.state != "idle":
                self._release(lane)

    def _advance(self, lane):
        if lane.state in ("idle", "done", "waiting_start"):
            return False
        if lane.state == "aborting":
            return self._advance_abort(lane)
        try:
            if lane.state == "preparing":
                return self._prepare(lane)
            if lane.state == "running":
                return self._run_pass(lane)
            if lane.state in ("waiting_ref", "waiting_latents", "waiting_state"):
                return self._wait_for_message(lane)
            if lane.state == "transitioning":
                return self._transition(lane)
        except Exception as error:
            self._fail(lane, error)
            return True
        raise RuntimeError(f"The front lane {lane.index} entered an unknown state {lane.state!r}.")

    def _prepare(self, lane):
        lane.half = self.front.prepare(lane.request_id, lane.index, lane.settings)
        self._send({
            "kind": "prepared",
            "role": self.role,
            "request_id": lane.request_id,
            "lane": lane.index,
            "num_frames": lane.half.video_grid.num_frames,
        })
        logger.debug(
            f"Request {lane.request_id}: {lane.half.video_grid.num_frames} frames, "
            f"{lane.half.video_grid.tokens} video tokens, {lane.half.audio_grid.tokens} audio tokens, "
            f"{len(lane.half.timesteps)} coarse steps"
        )
        if lane.settings.image_path:
            lane.state = "waiting_ref"
            return True
        # The first state can only cross once the second GPU has a lane to adopt it into,
        # and that GPU cannot build its lane until it has the frame count reported above.
        # The engine sends a start message once the second GPU is ready.
        lane.state = "waiting_start"
        return True

    def _send_state(self, lane):
        video, audio = self.front.state_payload(lane.half)
        self.front.begin_step(lane.half)
        self._send_message(
            lane,
            KIND_STATE,
            [video, audio],
            stage=lane.half.stage,
            step=lane.half.step_index,
            steps_total=len(lane.half.timesteps),
        )
        logger.debug(f"Request {lane.request_id}: sent stage {lane.half.stage + 1} state")

    def _run_pass(self, lane):
        plan = self.front.next_plan(lane.half)
        if plan is None:
            # second GPU now owes the step's latents.
            lane.state = "waiting_latents"
            return self._wait_for_message(lane)
        with span(f"pass {plan.name} b{plan.batch}"):
            reuse, epoch, payload, constant = self.front.execute_pass(lane.half, plan)
        key = (lane.half.stage, plan.batch)
        carried = key not in lane.half.conditioning_sent
        if carried:
            lane.half.conditioning_sent.add(key)
            payload = payload + constant
        self._send_message(
            lane,
            KIND_PASS,
            payload,
            stage=lane.half.stage,
            step=lane.half.step_index,
            pass_code=plan.code,
            batch=plan.batch,
            reuse=reuse,
            cache_epoch=epoch,
            carries_conditioning=carried,
            start_block=0,
            video_tokens=payload[0].shape[1],
            video_width=payload[0].shape[2],
            audio_tokens=payload[1].shape[1],
            audio_width=payload[1].shape[2],
            steps_total=len(lane.half.timesteps),
        )
        logger.debug(
            f"Request {lane.request_id}: sent {plan.describe()} at step "
            f"{lane.half.step_index + 1}/{len(lane.half.timesteps)}, reuse {reuse}, "
            f"conditioning {'included' if carried else 'reused'}"
        )
        self.front.finish_pass(lane.half, plan, reuse)
        return True

    def _transition(self, lane):
        with span("transition"):
            self.front.transition(lane.half)
        # The barrier is a marker rather than a payload: both GPUs derive the refinement noise
        # from the request's seed, and this message is what tells the second GPU to go first.
        self._send_message(
            lane,
            KIND_REFINE_NOISE,
            [],
            stage=1,
            step=0,
            steps_total=len(lane.half.timesteps),
        )
        lane.state = "waiting_state"
        return True

    def _accept(self, lane, header, tensors):
        try:
            if header.kind == KIND_ABORT:
                logger.debug(f"Request {lane.request_id} on the front half aborted by the other half")
                self._release(lane)
                return
            if lane.state == "waiting_ref" and header.kind == KIND_REF_TOKENS:
                self.front.adopt_reference_tokens(lane.half, tensors[0])
                self._send_state(lane)
                lane.state = "running"
                return
            if lane.state == "waiting_latents" and header.kind == KIND_LATENTS:
                if header.stage != lane.half.stage or header.step != lane.half.step_index:
                    raise RuntimeError(
                        f"Request {lane.request_id} expected the latents of stage {lane.half.stage + 1} step "
                        f"{lane.half.step_index}, received stage {header.stage + 1} step {header.step}."
                    )
                finished = self.front.adopt_step_latents(lane.half, tensors[0], tensors[1])
                if not finished:
                    lane.state = "running"
                    return
                if lane.half.stage == 0 and lane.half.settings.higher_resolution:
                    lane.state = "transitioning"
                    return
                self._finish(lane)
                return
            if lane.state == "waiting_state" and header.kind == KIND_STATE:
                if header.stage != 1:
                    raise RuntimeError(
                        f"Request {lane.request_id} expected refinement state, received stage "
                        f"{header.stage + 1} state."
                    )
                self.front.adopt_refined_state(lane.half, tensors[0], tensors[1])
                lane.state = "running"
                return
            raise RuntimeError(
                f"Request {lane.request_id} on the front half received an unexpected {header.describe()} "
                f"while {lane.state}."
            )
        except Exception as error:
            self._fail(lane, error)

    def _finish(self, lane):
        lane.state = "done"
        logger.debug(f"Request {lane.request_id} finished its denoising on the front half")


class BackWorker(_Worker):
    """Runs the second GPU's work: the trailing blocks, the samplers, the decoders and the files."""
    role = "back"

    def __init__(self, connection, config):
        super().__init__(connection, config)
        self.back = None
        self.decode_stream = None
        self.decode_queue = queue.SimpleQueue()
        self.decode_jobs = {}
        self.decode_thread = None

    def _load(self):
        self.back = self._load_half(
            load_back_half,
            ("transformer", "vae", "audio_vae", "vocoder", "upsampler"),
        )

    def _start_background(self):
        if self.device.type == "cuda":
            self.decode_stream = torch.cuda.Stream(device=self.device)
        self.decode_thread = threading.Thread(
            target=self._decode_loop, name=f"{self.role}-decode", daemon=True
        )
        self.decode_thread.start()

    def _stop_background(self):
        self.decode_queue.put(None)
        if self.decode_thread is not None:
            self.decode_thread.join(timeout=self.config.engine_config.decode_shutdown_seconds)

    def _parked(self, lane):
        return lane.index in self.decode_jobs

    def _decode_loop(self):
        """Decode and write one lane's output at a time, off the event loop."""
        while True:
            job = self.decode_queue.get()
            if job is None:
                return
            stream = self.decode_stream
            try:
                if stream is not None:
                    # The latents were produced on the event loop's stream, so this waits for
                    # them before reading them on another one.
                    stream.wait_stream(torch.cuda.current_stream())
                scope = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
                with scope:
                    with span("decode"):
                        video, audio = self.back.decode(job.half)
                    with span("write"):
                        files = self._write(job.request, job.output_name, video, audio)
            except BaseException as error:
                logger.exception(f"Request {job.request} failed while decoding")
                job.error = f"{type(error).__name__}: {error}"
            else:
                job.files = files
            job.seconds = time.monotonic() - job.started
            job.done.set()

    def _on_lane_control(self, message):
        kind = message.get("kind")
        lane = self.lanes[message["lane"]]
        if kind == "admit":
            # The frame count comes from the first GPU, which is the side that predicts a
            # duration when the request does not name one.
            self._admit_lane(lane, message, "waiting_frames")
        elif kind == "frames":
            if lane.request_id != message["request_id"]:
                return
            lane.num_frames = message["num_frames"]
            lane.state = "preparing"

    def _advance(self, lane):
        if lane.state in ("idle", "done", "waiting_frames"):
            return False
        if lane.state == "aborting":
            return self._advance_abort(lane)
        try:
            if lane.state == "preparing":
                return self._prepare(lane)
            if lane.state == "running":
                return self._run(lane)
            if lane.state in ("starting", "waiting_refine"):
                return self._wait_for_message(lane)
            if lane.state == "decoding":
                return self._collect_decode(lane)
        except Exception as error:
            self._fail(lane, error)
            return True
        raise RuntimeError(f"The back lane {lane.index} entered an unknown state {lane.state!r}.")

    def _prepare(self, lane):
        lane.half, ref_tokens = self.back.prepare(
            lane.request_id, lane.index, lane.settings, lane.num_frames
        )
        self._send({
            "kind": "prepared",
            "role": self.role,
            "request_id": lane.request_id,
            "lane": lane.index,
        })
        if ref_tokens is not None:
            self._send_message(lane, KIND_REF_TOKENS, [ref_tokens], stage=0, step=0)
            logger.debug(f"Request {lane.request_id}: sent reference tokens {tuple(ref_tokens.shape)}")
        lane.state = "starting"
        return True

    def _run(self, lane):
        plan = self.back.next_plan(lane.half)
        if plan is None:
            return self._finish_step(lane)
        if plan.needs_front:
            return self._wait_for_message(lane)
        with span(f"pass {plan.name} local b{plan.batch}"):
            self.back.run_local(lane.half, plan)
        self.back.finish_pass(lane.half, plan)
        logger.debug(f"Request {lane.request_id}: ran {plan.describe()} locally at step {lane.half.step_index + 1}")
        return True

    def _finish_step(self, lane):
        if not lane.half.predictions:
            raise RuntimeError(
                f"Request {lane.request_id} finished a step with no predictions to combine."
            )
        finished_step = lane.half.step_index
        with span("combine"):
            video, audio = self.back.finish_step(lane.half)
        done = lane.half.step_index >= len(lane.half.timesteps)
        self._send_message(
            lane,
            KIND_LATENTS,
            [video, audio],
            stage=lane.half.stage,
            step=finished_step,
            steps_total=len(lane.half.timesteps),
        )
        logger.debug(
            f"Request {lane.request_id}: finished stage {lane.half.stage + 1} step "
            f"{finished_step + 1}/{len(lane.half.timesteps)}"
        )
        if not done:
            self.back.begin_step(lane.half)
            return True
        if lane.half.stage == 0 and lane.settings.higher_resolution:
            # The other GPU only learns that refinement was asked for when this noise arrives,
            # so this half waits for the first GPU's barrier before it builds the next stage.
            lane.state = "waiting_refine"
            return True
        lane.state = "decoding"
        self._start_decode(lane)
        return True

    def _start_decode(self, lane):
        """Hand a finished lane's latents to the decode thread"""
        job = _DecodeJob(
            lane=lane.index,
            request=lane.request_id,
            half=lane.half,
            output_name=lane.settings.output_name,
            started=time.monotonic(),
        )
        self.decode_jobs[lane.index] = job
        self.decode_queue.put(job)

    def _collect_decode(self, lane):
        """Take a decode's result, publish its files and let the lane go."""
        job = self.decode_jobs.get(lane.index)
        if job is None:
            raise RuntimeError(f"The back lane {lane.index} is decoding with no decode in flight.")
        if not job.done.is_set():
            return False
        del self.decode_jobs[lane.index]
        if job.error:
            raise RuntimeError(f"Request {job.request} failed while decoding: {job.error}")
        logger.debug(f"Request {job.request} decoded and wrote in {job.seconds:.2f}s")
        self._send({
            "kind": "completed",
            "role": self.role,
            "request_id": job.request,
            "lane": lane.index,
            **job.files,
        })
        self._release(lane)
        return True

    def _accept(self, lane, header, tensors):
        try:
            if header.kind == KIND_ABORT:
                logger.debug(f"Request {lane.request_id} on the back half aborted by the other half")
                self._release(lane)
                return
            if lane.state == "starting" and header.kind == KIND_STATE:
                self.back.adopt_state(lane.half, tensors[0], tensors[1])
                lane.state = "running"
                return
            if lane.state == "running" and header.kind == KIND_PASS:
                plan = self.back.next_plan(lane.half)
                if plan is None:
                    raise RuntimeError(f"Request {lane.request_id} received a pass it was not expecting.")
                if plan.needs_front is False:
                    raise RuntimeError(
                        f"Request {lane.request_id} received a {plan.name} pass the back half runs locally."
                    )
                with span(f"pass {plan.name} b{plan.batch}"):
                    self.back.accept_pass(lane.half, plan, header, tensors)
                self.back.finish_pass(lane.half, plan)
                return
            if lane.state == "waiting_refine" and header.kind == KIND_REFINE_NOISE:
                if tensors:
                    raise RuntimeError(
                        f"Request {lane.request_id} received {len(tensors)} refinement tensor(s); the "
                        "barrier carries a marker, because each half derives the noise from the seed."
                    )
                with span("refine/upsample"):
                    video, audio, _ = self.back.begin_refinement(lane.half)
                self._send_message(lane, KIND_STATE, [video, audio], stage=1, step=0,
                                   steps_total=len(lane.half.timesteps))
                lane.state = "running"
                return
            raise RuntimeError(
                f"Request {lane.request_id} on the back half received an unexpected {header.describe()} "
                f"while {lane.state}."
            )
        except Exception as error:
            self._fail(lane, error)

    def _write(self, request_id, output_name, video, audio):
        results_dir = Path(self.config.results_dir)
        video_path = results_dir / output_name
        audio_path = video_path.with_suffix(".wav")
        temporary_video_path = video_path.with_name(f".{video_path.stem}.partial{video_path.suffix}")
        temporary_audio_path = audio_path.with_name(f".{audio_path.stem}.partial{audio_path.suffix}")
        results_dir.mkdir(parents=True, exist_ok=True)
        if video_path.exists() or audio_path.exists():
            raise FileExistsError(f"Request {request_id} output already exists in {results_dir}.")
        frames = video.shape[2]
        duration_seconds = frames / self.config.engine_config.frame_rate
        video_published = False
        audio_published = False
        try:
            self._save_video(video, temporary_video_path)
            self._save_audio(audio, temporary_audio_path)
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
        logger.debug(
            f"Request {request_id} wrote {video_path.name} with {frames} frames "
            f"({duration_seconds:.2f}s) and {audio_path.name}"
        )
        return {
            "video": str(video_path),
            "audio": str(audio_path),
            "frames": frames,
            "duration_seconds": duration_seconds,
        }

    def _save_video(self, video, path):
        frames = video[0].detach().float().cpu()
        frames = (frames.clamp(-1, 1) + 1) / 2 * 255
        frames = frames.to(torch.uint8).permute(1, 2, 3, 0).numpy()
        imageio.mimsave(str(path), list(frames), fps=self.config.engine_config.frame_rate)

    def _save_audio(self, audio, path):
        sample_rate = self.back.vocoder.config.output_sampling_rate
        waveform = audio[0].detach().float().cpu().clamp(-1, 1)
        samples = (waveform * 32767).to(torch.int16).numpy()
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(samples.shape[0])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.T.tobytes())
