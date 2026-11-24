import dataclasses
import torch
import torch.distributed as dist

MAGIC = 0x4C545850
VERSION = 4
HEADER_SLOTS = 24
FIRST_SEQ = 1
KIND_PASS = 1
KIND_STATE = 2
KIND_REF_TOKENS = 3
KIND_LATENTS = 4
KIND_REFINE_NOISE = 5
KIND_ABORT = 6
KIND_NAMES = {
    KIND_PASS: "pass",
    KIND_STATE: "stage state",
    KIND_REF_TOKENS: "reference tokens",
    KIND_LATENTS: "step latents",
    KIND_REFINE_NOISE: "refinement noise",
    KIND_ABORT: "abort",
}

DTYPE_IDS = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 3, torch.int64: 4, torch.bool: 5}
ID_DTYPES = {identifier: dtype for dtype, identifier in DTYPE_IDS.items()}
SLOTS_PER_TENSOR = 5
MAX_TENSOR_DIMS = 3
# A message carries a fixed, small set of tensors, so a header naming more than this is corrupt
# rather than merely large. The bound is checked before any buffer is sized from that header.
MAX_PAYLOAD_TENSORS = 64
# The header and the payload's descriptor travel as one fixed-size control block. Every message
# therefore starts with exactly one transfer of a size both workers know in advance, so the
# receiver never has to ask what is coming and never reads a descriptor as a header.
CONTROL_SLOTS = HEADER_SLOTS + SLOTS_PER_TENSOR * MAX_PAYLOAD_TENSORS


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

    def to_values(self):
        """The header as control-block values, padded to the slots the receiver reads."""
        values = [
            MAGIC,
            VERSION,
            self.request_id,
            self.lane,
            self.stage,
            self.step,
            self.kind,
            self.pass_code,
            self.batch,
            int(self.reuse),
            self.cache_epoch,
            self.start_block,
            self.video_tokens,
            self.video_width,
            self.audio_tokens,
            self.audio_width,
            self.steps_total,
            self.payload_count,
            self.seq,
            int(self.carries_conditioning),
        ]
        if len(values) > HEADER_SLOTS:
            raise ValueError(f"A header carries {len(values)} facts but only {HEADER_SLOTS} slots exist.")
        values.extend([0] * (HEADER_SLOTS - len(values)))
        return values

    @classmethod
    def from_values(cls, values):
        if values[0] != MAGIC:
            raise RuntimeError(
                f"The data channel produced {values[0]:#x} where a message header should start; "
                "the two halves are out of step with each other."
            )
        if values[1] != VERSION:
            raise RuntimeError(f"Message protocol version {values[1]} arrived, expected {VERSION}.")
        return cls(
            request_id=values[2],
            lane=values[3],
            stage=values[4],
            step=values[5],
            kind=values[6],
            pass_code=values[7],
            batch=values[8],
            reuse=bool(values[9]),
            cache_epoch=values[10],
            start_block=values[11],
            video_tokens=values[12],
            video_width=values[13],
            audio_tokens=values[14],
            audio_width=values[15],
            steps_total=values[16],
            payload_count=values[17],
            seq=values[18],
            carries_conditioning=bool(values[19]),
        )

    def describe(self):
        return (
            f"{KIND_NAMES.get(self.kind, self.kind)} request {self.request_id} lane {self.lane} "
            f"stage {self.stage + 1} step {self.step} pass {self.pass_code} batch {self.batch} "
            f"seq {self.seq}"
        )


def _describe_payload(tensors):
    """The descriptor of a payload: rank, dtype and padded shape of every tensor."""
    values = []
    for tensor in tensors:
        if tensor.ndim > MAX_TENSOR_DIMS:
            raise ValueError(f"Payload tensors may have at most {MAX_TENSOR_DIMS} dimensions, got {tuple(tensor.shape)}.")
        dtype_id = DTYPE_IDS.get(tensor.dtype)
        if dtype_id is None:
            raise ValueError(f"Payload tensors cannot be sent as {tensor.dtype}.")
        shape = list(tensor.shape) + [0] * (MAX_TENSOR_DIMS - tensor.ndim)
        values.extend([tensor.ndim, dtype_id, shape[0], shape[1], shape[2]])
    return values


def _read_payload(values, count):
    """Read a control block's descriptor back into a dtype and a shape per tensor."""
    described = []
    for index in range(count):
        base = HEADER_SLOTS + index * SLOTS_PER_TENSOR
        ndim, dtype_id, *dims = values[base:base + SLOTS_PER_TENSOR]
        dtype = ID_DTYPES.get(dtype_id)
        if dtype is None:
            raise RuntimeError(f"Payload tensor {index} names dtype code {dtype_id}, which this engine does not know.")
        if not 0 <= ndim <= MAX_TENSOR_DIMS:
            raise RuntimeError(
                f"Payload tensor {index} names {ndim} dimensions but a descriptor holds at most "
                f"{MAX_TENSOR_DIMS}; the halves are out of step with each other."
            )
        shape = dims[:ndim]
        if any(dimension < 0 for dimension in shape):
            raise RuntimeError(f"Payload tensor {index} names negative dimensions {tuple(shape)}.")
        described.append((dtype, tuple(shape)))
    return described


class Outbound:
    def __init__(self, works, tensors):
        self.works = works
        # A send reads its buffer asynchronously, so every tensor handed to the library
        # has to outlive the work handle, including the contiguous copies made here.
        self.tensors = tensors

    def complete(self):
        return all(work.is_completed() for work in self.works)


def _post(group, peer, tensors, operation):
    """Post one phase of a message as a single batched point-to-point group.

    Every transfer a worker makes goes through here, on the communicator of the direction
    it belongs to. Batching a phase keeps its tensors in one group, so another message
    arriving in between can never split a header from its payload.
    """
    if not tensors:
        return []
    return dist.batch_isend_irecv([dist.P2POp(operation, tensor, peer, group) for tensor in tensors])


def post_send(group, peer, header, tensors, device):
    """Post one message: a fixed-size control block, then the payload, each as one batch."""
    values = header.to_values()
    if tensors:
        values.extend(_describe_payload(tensors))
    if len(values) > CONTROL_SLOTS:
        raise ValueError(
            f"This message describes {len(tensors)} tensors, which needs {len(values)} control slots "
            f"but a control block holds {CONTROL_SLOTS}."
        )
    control = torch.tensor(values + [0] * (CONTROL_SLOTS - len(values)), dtype=torch.int64, device=device)
    works = _post(group, peer, [control], dist.isend)
    retained = [control]
    payload = [tensor if tensor.numel() == 0 else tensor.contiguous() for tensor in tensors]
    # An empty tensor carries no bytes, so no transfer is posted for it.
    works.extend(_post(group, peer, [tensor for tensor in payload if tensor.numel() > 0], dist.isend))
    return Outbound(works, retained + payload)


class Inbound:
    """One message being received, advanced one step per event-loop iteration."""
    def __init__(self, group, peer, device):
        self.group = group
        self.peer = peer
        self.device = device
        self.header = None
        self.phase = "control"
        self._payload_buffers = []
        self._payload_works = []
        self._control_buffer = torch.zeros(CONTROL_SLOTS, dtype=torch.int64, device=device)
        self._control_work = _post(group, peer, [self._control_buffer], dist.irecv)[0]

    def poll(self):
        if self.phase == "control":
            if not self._control_work.is_completed():
                return None
            # The transport fills the buffer on its own stream. Waiting after the completion
            # check returns at once and makes this stream see the bytes before they are read.
            self._control_work.wait()
            values = self._control_buffer.tolist()
            self._control_buffer = None
            self.header = MessageHeader.from_values(values)
            if not 0 <= self.header.payload_count <= MAX_PAYLOAD_TENSORS:
                raise RuntimeError(
                    f"The data channel announced {self.header.payload_count} payload tensors, but a "
                    f"message carries at most {MAX_PAYLOAD_TENSORS}; the halves are out of step."
                )
            described = _read_payload(values, self.header.payload_count)
            self._payload_buffers = [
                torch.empty(shape, dtype=dtype, device=self.device) for dtype, shape in described
            ]
            moving = [buffer for buffer in self._payload_buffers if buffer.numel() > 0]
            self._payload_works = _post(self.group, self.peer, moving, dist.irecv)
            self.phase = "payload"
            if not self._payload_works:
                self.phase = "done"
                return self.header, self._payload_buffers
            return None
        if self.phase == "payload":
            if not all(work.is_completed() for work in self._payload_works):
                return None
            for work in self._payload_works:
                work.wait()
            self._payload_works = []
            self.phase = "done"
            return self.header, self._payload_buffers
        raise RuntimeError("A completed message was polled again; the receive belongs to one message only.")
