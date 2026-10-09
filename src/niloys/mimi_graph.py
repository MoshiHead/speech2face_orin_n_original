"""Wrap a live Mimi instance's encode/decode in CUDA graphs.

Requires graph_safe_mimi.enable() first -- without it the streaming state is
rebound each call and a replay reads stale buffers (measured: 27% of codes
corrupted). With it, graphed output is bit-exact:
    encode 13.25 -> 4.42 ms, decode 8.47 -> 3.47 ms, 0/320 code mismatches,
    decoded audio max|diff| 0.0.

Captures lazily after `warmup` real calls so shapes are settled, and falls back
to eager for any shape it did not capture.
"""
import os as _os
import torch
# TRY19_MIMI_SAMESTREAM=1 -> replay on the caller's stream.
_SAME_STREAM = _os.environ.get('TRY19_MIMI_SAMESTREAM', '0') == '1'
# TRY19_MIMI_GRAPH_SAFE=1 -> never replay a graph after state was touched outside it:
# drop and re-capture after an eager fallback (odd shape) or a streaming reset.
_GRAPH_SAFE = _os.environ.get('TRY19_MIMI_GRAPH_SAFE', '0') == '1'
from moshi.utils.compile import no_cuda_graph


class _Graphed:
    def __init__(self, fn, warmup=8, name=""):
        self.fn, self.warmup, self.name = fn, warmup, name
        self.n = 0
        self.g = None
        self.inp = None
        self.out = None
        self.shape = None
        self.failed = False
        self.rs = torch.cuda.Stream()      # normal-priority replay stream

    def invalidate(self):
        self.g = None; self.inp = None; self.out = None; self.shape = None; self.n = 0
        import graph_safe_mimi as _gsm
        _gsm.CAPTURED = False

    def __call__(self, x):
        if self.failed:
            return self.fn(x)
        if self.g is not None and tuple(x.shape) == self.shape and _SAME_STREAM:
            # SAME-STREAM replay: no rs.wait_stream/cur.wait_stream ping-pong.
            # The cross-stream round trip below adds two syncs per call; this
            # variant was never tested among the six ruled-out theories.
            self.inp.copy_(x)
            self.g.replay()
            return self.out.clone()
        if self.g is not None and tuple(x.shape) == self.shape:
            # Replay at NORMAL priority. The caller (persona worker) runs on a
            # priority=-1 stream; replaying there made Mimi so cheap that the
            # persona thread monopolised the GPU and starved the avatar thread
            # (avatar publish_wall 19 -> 169 ms, chunks 143 -> 63). Doing the
            # work on a normal-priority stream keeps the two threads fair.
            cur = torch.cuda.current_stream()
            self.rs.wait_stream(cur)
            with torch.cuda.stream(self.rs):
                self.inp.copy_(x)
                self.g.replay()
                out = self.out.clone()
            cur.wait_stream(self.rs)
            return out
        if self.g is not None:
            # unexpected shape -> eager. The eager call may REALLOCATE graph-safe state
            # buffers (graph_safe_mimi._persist on a shape change), freeing the ones this
            # graph was captured on -> later replays would write into freed memory.
            print(f"[mimi_graph] {self.name}: eager fallback for shape {tuple(x.shape)} "
                  f"(graph shape {self.shape}){' -> graph dropped, will re-capture' if _GRAPH_SAFE else ''}",
                  flush=True)
            if _GRAPH_SAFE:
                self.invalidate()
            return self.fn(x)
        self.n += 1
        if self.n <= self.warmup:
            return self.fn(x)
        try:
            with no_cuda_graph():
                self.inp = x.clone()
                # Capture on the stream we are actually CALLED on. The server runs
                # persona work inside `with torch.cuda.stream(priority_stream)`, and
                # replay launches on the current stream; capturing on an unrelated
                # side stream made the replay serialise against the priority stream
                # and halved live throughput (video frames 183 -> 85).
                cur = torch.cuda.current_stream()
                cap = cur if cur != torch.cuda.default_stream() else torch.cuda.Stream()
                warm = torch.cuda.Stream(); warm.wait_stream(cur)
                with torch.cuda.stream(warm), torch.no_grad():
                    for _ in range(2):
                        self.fn(self.inp)
                cur.wait_stream(warm)
                g = torch.cuda.CUDAGraph()
                # capture_error_mode='thread_local' is required when other
                # threads touch CUDA during capture; the default 'global'
                # mode treats their activity as invalidating the capture.
                with torch.no_grad(), torch.cuda.graph(
                        g, stream=cap, capture_error_mode='thread_local'):
                    out = self.fn(self.inp)
                self.g, self.out, self.shape = g, out, tuple(x.shape)
                self.cap_stream = cap
            print(f"[try19] mimi.{self.name} CUDA-graphed at shape {self.shape} "
                  f"on stream {cap}", flush=True)
            import graph_safe_mimi as _gsm
            _gsm.CAPTURED = True
            self.inp.copy_(x); self.g.replay(); return self.out.clone()
        except Exception as e:
            self.failed = True
            print(f"[try19] WARNING mimi.{self.name} graph capture failed "
                  f"({type(e).__name__}: {str(e)[:80]}); staying eager", flush=True)
            return self.fn(x)


def wrap(mimi, warmup=8):
    """Patch this instance's encode/decode. Returns the two wrappers.

    TRY19_MIMI_PARTS selects which to graph: both (default) | encode | decode.
    Graphing decode replays ~100 SEANet conv kernels as one indivisible batch;
    if that is what monopolises Orin's work queues and starves the avatar
    renderer (12 -> 149 ms), encode-only should keep most of the win
    (encode 20.6 -> 3.4 ms) without the cost.
    """
    parts = _os.environ.get("TRY19_MIMI_PARTS", "both").lower()
    ge = gd = None
    if parts in ("both", "encode"):
        ge = _Graphed(mimi.encode, warmup, "encode"); mimi.encode = ge
    if parts in ("seanet", "seanet_quant"):
        # Graph ONLY the SEANet conv encoder (the launch-bound part). The encoder
        # transformer stays on moshi's own CUDAGraphed path, which every normal run
        # already uses safely; capturing its streaming KV state inside our graph
        # corrupted other threads' memory (crash ~358 steps, 4/5 runs).
        ge = _Graphed(mimi.encoder.forward, warmup, "encoder_seanet"); mimi.encoder.forward = ge
    if parts == "seanet_quant":
        # The quantizer (8 codebooks: project, cdist, argmin, residual) is the biggest
        # launch cost of encode (6.85 of ~11 ms CPU) and is STATELESS -> safe to graph.
        gq = _Graphed(mimi.quantizer.encode, warmup, "quantizer_encode"); mimi.quantizer.encode = gq
    if parts in ("both", "decode"):
        gd = _Graphed(mimi.decode, warmup, "decode"); mimi.decode = gd
    print(f"[try19] mimi graph parts={parts}", flush=True)
    _orig_reset = mimi.reset_streaming

    def _reset(*a, **k):
        r = _orig_reset(*a, **k)
        print(f"[mimi_graph] reset_streaming{' -> graphs dropped, will re-capture' if _GRAPH_SAFE else ' (graphs kept)'}",
              flush=True)
        if _GRAPH_SAFE:
            for g in (ge, gd):
                if g is not None:
                    g.invalidate()
        return r
    mimi.reset_streaming = _reset
    return ge, gd
