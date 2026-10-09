"""Make Mimi's streaming state CUDA-graph-safe WITHOUT editing moshi.

moshi's streaming modules do `state.x = torch.cat(...)[slice]` -- a NEW tensor
every call, rebound onto the dataclass. A captured CUDA graph freezes the
addresses it saw at capture time, so on replay it reads the stale buffer. That
is why graphing whole-Mimi corrupted 27% of the codes.

Fix: hold each piece of state in a tensor allocated once and mutated in place,
so the dataclass always points at the SAME storage. The maths is unchanged --
only the allocation pattern differs. Applied by monkey-patching at import time
so no moshi file is modified.
"""
import math
import torch
from moshi.modules.streaming import (
    RawStreamingConv1d, RawStreamingConvTranspose1d, StreamingAdd,
)

_ORIG = {}
CAPTURED = False   # set by mimi_graph once an encode/decode graph is captured


def _persist(state, name, src):
    """Copy src into a per-state buffer allocated once; return that buffer."""
    buf = getattr(state, name, None)
    if buf is None or buf.shape != src.shape or buf.dtype != src.dtype:
        if CAPTURED and buf is not None:
            print(f"[graph_safe] WARNING {name} reallocated AFTER graph capture "
                  f"{tuple(buf.shape)} -> {tuple(src.shape)}: the captured graph now points at freed memory",
                  flush=True)
        buf = src.detach().clone()
        setattr(state, name, buf)
    else:
        buf.copy_(src)
    return buf


def _conv_forward(self, input: torch.Tensor) -> torch.Tensor:
    stride = self.stride[0]
    kernel = (self.kernel_size[0] - 1) * self.dilation[0] + 1
    st = self._streaming_state
    if st is None:
        return torch.nn.Conv1d.forward(self, input)
    prev = st.previous
    if prev is not None:
        input = torch.cat([prev, input], dim=-1)
    B, C, T = input.shape
    num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
    offset = num_frames * stride
    if num_frames > 0:
        input_length = (num_frames - 1) * stride + kernel
        out = torch.nn.Conv1d.forward(self, input[..., :input_length])
    else:
        out = torch.empty(B, self.out_channels, 0, device=input.device, dtype=input.dtype)
    st.previous = _persist(st, "_gs_prev", input[..., offset:])
    return out


def _convtr_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, C, T = x.shape
    stride = self.stride[0]
    kernel = self.kernel_size[0]
    st = self._streaming_state
    if st is None:
        return torch.nn.ConvTranspose1d.forward(self, x)
    if T == 0:
        return torch.empty(B, self.out_channels, 0, device=x.device, dtype=x.dtype)
    out = torch.nn.ConvTranspose1d.forward(self, x)
    OT = out.shape[-1]
    partial = st.partial
    if partial is not None:
        PT = partial.shape[-1]
        if self.bias is not None:
            out[..., :PT] += partial - self.bias[:, None]
        else:
            out[..., :PT] += partial
    invalid_steps = kernel - stride
    st.partial = _persist(st, "_gs_partial", out[..., OT - invalid_steps:])
    return out[..., : OT - invalid_steps]


def _add_forward(self, x: torch.Tensor, y: torch.Tensor):
    st = self._streaming_state
    if st is None:
        return x + y
    px, py = st.previous_x, st.previous_y
    if px is not None:
        x = torch.cat([px, x], dim=-1)
    if py is not None:
        y = torch.cat([py, y], dim=-1)
    m_l = min(x.shape[-1], y.shape[-1])
    st.previous_x = _persist(st, "_gs_px", x[..., m_l:])
    st.previous_y = _persist(st, "_gs_py", y[..., m_l:])
    return x[..., :m_l] + y[..., :m_l]


def enable():
    if _ORIG:
        return
    _ORIG["conv"] = RawStreamingConv1d.forward
    _ORIG["convtr"] = RawStreamingConvTranspose1d.forward
    _ORIG["add"] = StreamingAdd.forward
    RawStreamingConv1d.forward = _conv_forward
    RawStreamingConvTranspose1d.forward = _convtr_forward
    StreamingAdd.forward = _add_forward


def disable():
    if not _ORIG:
        return
    RawStreamingConv1d.forward = _ORIG.pop("conv")
    RawStreamingConvTranspose1d.forward = _ORIG.pop("convtr")
    StreamingAdd.forward = _ORIG.pop("add")
