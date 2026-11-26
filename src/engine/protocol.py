import contextlib
import dataclasses
import os
import threading
import torch
import torch.distributed as dist

MAGIC = 0x4C545850
VERSION = 6
FIRST_SEQ = 1
KIND_PASS = 1
KIND_STATE = 2
KIND_REF_TOKENS = 3
KIND_LATENTS = 4
KIND_ABORT = 6
KIND_NAMES = {
    KIND_PASS: "pass",
    KIND_STATE: "stage state",
    KIND_REF_TOKENS: "reference tokens",
    KIND_LATENTS: "step latents",
    KIND_ABORT: "abort",
}

DTYPE_IDS = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 3, torch.int64: 4, torch.bool: 5}
ID_DTYPES = {identifier: dtype for dtype, identifier in DTYPE_IDS.items()}
MAX_TENSOR_DIMS = 3
# A message carries a fixed, small set of tensors, so a header naming more than this is corrupt
# rather than merely large. The bound is checked before any buffer is sized from that header.
MAX_PAYLOAD_TENSORS = 64


@dataclasses.dataclass(frozen=True)
class MessageHeader:
    """The fields that travel with every message."""
    request_id: int
    lane: int
    stage: int
    step: int
    kind: int
    pass_code: int = 0
    batch: int = 0
    reuse: bool = False
    cache_epoch: int = -1
    start_block: int = 0
    video_tokens: int = 0
    video_width: int = 0
    audio_tokens: int = 0
    audio_width: int = 0
    steps_total: int = 0
    payload_count: int = 0
    # The sender's count of the messages it has posted in this direction. Two workers that
    # fall out of step then disagree at the first message, rather than at whatever a semantic
    # mismatch happens to look like later on.
    seq: int = 0
    # Whether this message carries the stage-constant prompt conditioning, which only the
    # first pass of a (stage, batch) pair does; every later pass rebuilds it from the copy
    # its receiver kept. Only a pass can carry it.
    carries_conditioning: bool = False

    def describe(self):
        return (
            f"{KIND_NAMES.get(self.kind, self.kind)} request {self.request_id} lane {self.lane} "
            f"stage {self.stage + 1} step {self.step} pass {self.pass_code} batch {self.batch} "
            f"seq {self.seq}"
        )


def _describe_payload(tensors):
    if len(tensors) > MAX_PAYLOAD_TENSORS:
        raise ValueError(
            f"This message carries {len(tensors)} payload tensors, above {MAX_PAYLOAD_TENSORS}."
        )
    descriptor = []
    for tensor in tensors:
        if tensor.ndim > MAX_TENSOR_DIMS:
            raise ValueError(
                f"Payload tensors may have at most {MAX_TENSOR_DIMS} dimensions, got {tuple(tensor.shape)}."
            )
        dtype_id = DTYPE_IDS.get(tensor.dtype)
        if dtype_id is None:
            raise ValueError(f"Payload tensors cannot be sent as {tensor.dtype}.")
        descriptor.append((dtype_id, tuple(tensor.shape)))
    return tuple(descriptor)


def _parse_descriptor(descriptor, payload_count):
    """Validate a CPU pipe payload descriptor and resolve its dtypes and shapes."""
    if not 0 <= payload_count <= MAX_PAYLOAD_TENSORS:
        raise RuntimeError(
            f"The peer channel announced {payload_count} payload tensors, above {MAX_PAYLOAD_TENSORS}; "
            "the halves are out of step."
        )
    if len(descriptor) != payload_count:
        raise RuntimeError(
            f"Header announced {payload_count} payload tensors, but descriptor holds {len(descriptor)}."
        )
    described = []
    for index, (dtype_id, shape) in enumerate(descriptor):
        dtype = ID_DTYPES.get(dtype_id)
        if dtype is None:
            raise RuntimeError(
                f"Payload tensor {index} names dtype code {dtype_id}, which this engine does not know."
            )
        if not 0 <= len(shape) <= MAX_TENSOR_DIMS:
            raise RuntimeError(
                f"Payload tensor {index} names {len(shape)} dimensions, above {MAX_TENSOR_DIMS}."
            )
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(f"Payload tensor {index} names negative dimensions {tuple(shape)}.")
        described.append((dtype, tuple(shape)))
    return described


class WakeSignal:
    def __init__(self):
        self.read_fd, self._write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        os.set_blocking(self._write_fd, False)
        self._lock = threading.Lock()
        self._signaled = False
        self._closed = False

    def set(self):
        with self._lock:
            if self._closed or self._signaled:
                return
            self._signaled = True
            os.write(self._write_fd, b"\x00")

    def clear(self):
        with self._lock:
            if self._closed or not self._signaled:
                return
            os.read(self.read_fd, 4096)
            self._signaled = False

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            os.close(self.read_fd)
            os.close(self._write_fd)


class Outbound:
    def __init__(self, works, tensors):
        self.works = works
        # A send reads its buffer asynchronously, so every tensor handed to the library
        # has to outlive the work handle, including the contiguous copies made here.
        self.tensors = tensors

    def complete(self):
        return not self.works or (
            self.works[-1].is_completed() and all(work.is_completed() for work in self.works[:-1])
        )


def _post(group, peer, tensors, operation):
    """Post one payload batch over the directional NCCL communicator."""
    if not tensors:
        return []
    return dist.batch_isend_irecv([dist.P2POp(operation, tensor, peer, group) for tensor in tensors])


def post_send(group, peer, peer_connection, header, tensors):
    """Send message metadata on the CPU pipe and post one async NCCL send batch for payload tensors."""
    if header.payload_count != len(tensors):
        raise ValueError(
            f"Header payload_count {header.payload_count} does not match {len(tensors)} tensor(s)."
        )
    descriptor = _describe_payload(tensors)
    payload = [tensor if tensor.numel() == 0 else tensor.contiguous() for tensor in tensors]
    peer_connection.send((MAGIC, VERSION, header, descriptor))
    # Empty tensors carry no bytes, so no NCCL transfer is posted for them.
    works = _post(group, peer, [tensor for tensor in payload if tensor.numel() > 0], dist.isend)
    return Outbound(works, payload)


class InboundTransfer:
    """One asynchronously queued message whose metadata arrived over the CPU pipe."""
    def __init__(self, header, buffers, works, device, recv_stream=None):
        self.header = header
        self._buffers = buffers
        self._works = works
        self._device = device
        self._recv_stream = recv_stream
        self._finished = False

    def finish(self):
        if self._finished:
            raise RuntimeError("A completed inbound transfer was finished twice.")
        for work in self._works:
            work.wait()
        if self._recv_stream is not None:
            compute_stream = torch.cuda.current_stream(self._device)
            if compute_stream != self._recv_stream:
                for buffer in self._buffers:
                    if buffer.numel() > 0:
                        buffer.record_stream(compute_stream)
        self._works = []
        self._finished = True
        return self.header, self._buffers


def post_recv(group, peer, envelope, device, recv_stream=None):
    """Allocate receive buffers from peer pipe metadata and immediately post NCCL irecv."""
    magic, version, header, descriptor = envelope
    if magic != MAGIC:
        raise RuntimeError(
            f"The peer channel produced {magic:#x} where a message header should start; "
            "the two halves are out of step with each other."
        )
    if version != VERSION:
        raise RuntimeError(f"Message protocol version {version} arrived, expected {VERSION}.")
    described = _parse_descriptor(descriptor, header.payload_count)
    scope = torch.cuda.stream(recv_stream) if recv_stream is not None else contextlib.nullcontext()
    with scope:
        buffers = [torch.empty(shape, dtype=dtype, device=device) for dtype, shape in described]
        moving = [buffer for buffer in buffers if buffer.numel() > 0]
        works = _post(group, peer, moving, dist.irecv)
    return InboundTransfer(header, buffers, works, device, recv_stream=recv_stream)
