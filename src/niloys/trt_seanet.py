"""Stateless sliding-window SEANet decoder on TensorRT, a drop-in for mimi.decoder in streaming.

SEANet's receptive field is < 10 latent steps (25 Hz): decoding the last WIN steps
non-streaming and keeping the newest samples reproduces streaming decode to rounding
(max|d| 1.6e-4 at WIN>=10, same as streaming-vs-full-sequence itself). No streaming state
has to be exported: the only state is this window of latents.
Persistent input/output buffers are allocated once and never freed (no allocator hazards).
"""
import torch

_HOP = 960                                   # samples per 25 Hz latent step at 24 kHz


class TRTSeanet:
    def __init__(self, engine_path, win=12, device="cuda", fallback=None):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.win = win
        self.inp = torch.zeros(1, 512, win, device=device, dtype=torch.float32)
        self.out = torch.zeros(1, 1, win * _HOP, device=device, dtype=torch.float32)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        for n in names:
            mode = self.engine.get_tensor_mode(n)
            shape = tuple(self.engine.get_tensor_shape(n))
            buf = self.inp if mode == trt.TensorIOMode.INPUT else self.out
            if shape != tuple(buf.shape):
                raise RuntimeError(f"engine tensor {n} shape {shape} != buffer {tuple(buf.shape)}")
            self.ctx.set_tensor_address(n, buf.data_ptr())
        self.calls = 0
        self.fallback = fallback                 # original PyTorch SEANet for calls longer than the window

    @torch.no_grad()
    def __call__(self, x):
        """x: [1, 512, T] new latent steps (T=2 per 80 ms frame) -> [1, 1, T*960] pcm."""
        T = x.shape[-1]
        if T == 0:
            return x.new_zeros(1, 1, 0)
        if T > self.win and self.fallback is not None:
            return self.fallback(x)
        if T >= self.win:
            self.inp.copy_(x[..., -self.win:].float())
        else:
            self.inp[..., :-T] = self.inp[..., T:].clone()
            self.inp[..., -T:] = x.float()
        ok = self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self.calls += 1
        return self.out[..., -T * _HOP:].clone()

    def reset(self):
        self.inp.zero_()


def install(mimi, engine_path, win=12, verbose=True):
    """Replace mimi.decoder (SEANet) with the TensorRT window decoder, and clear its window
    whenever mimi's streaming state is reset."""
    t = TRTSeanet(engine_path, win, next(mimi.parameters()).device, fallback=mimi.decoder.forward)
    mimi.decoder.forward = t
    orig_reset = mimi.reset_streaming

    def _reset(*a, **k):
        r = orig_reset(*a, **k)
        t.reset()
        return r
    mimi.reset_streaming = _reset
    if verbose:
        print(f"[trt_seanet] mimi.decoder -> TensorRT {engine_path} (window {win} steps = {win * 40} ms)",
              flush=True)
    return t
