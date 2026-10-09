#!/usr/bin/env python3
"""Synchronized low-latency speech-to-speech visual conversation pipeline.

Jetson AGX Orin, PersonaPlex-7B (bnb NF4) + IMTalker fixed-identity student.

Why this exists
---------------
On the audited AGX Orin the modules run at very different real-time factors:

    PersonaPlex (Mimi + LM + Mimi)   ~7.0 s per 2.0 s chunk    RTF ~3.50
    UniTalk + Wav2Vec + FM             ~0.07 s per 2.0 s chunk  RTF ~0.03
    Student B-M renderer + output      ~1.04 s per 2.0 s chunk  RTF ~0.52

A single fused live loop is therefore pinned at PersonaPlex's ~3.5x real-time
factor and can never deliver 25 fps.  This pipeline splits the problem in two:

  Phase 1  CAPTURE   Run PersonaPlex as fast as the hardware allows, with no
                     playback deadline.  Cache the input PCM, the reply PCM and
                     -- critically -- the per-step layer[-2] hidden embeddings.

  Phase 2  REPLAY    The audio already exists, so play it at exactly 1.0x while
                     streaming the cached embeddings through
                     UniTalk -> FM -> student renderer on a wall-clock schedule.
                     Only the ~0.55 RTF tail is on the critical path, so the
                     avatar keeps up with the audio.

That is the "bypass the PersonaPlex output FPS bottleneck" requirement: the
bottleneck is not removed, it is moved off the playback critical path.

Launch parameters mirror `start_original_pod_8998_repro.sh` with three
deliberate changes, all noted in DEFAULT_LAUNCH below.

Usage
-----
    conda activate speech-trt
    cd /workspace/speech2avatar/IMTalker

    # Interactive: capture from the browser mic, then choose (A) or (B).
    python s2s_visual_pipeline.py

    # Offline: process a file, then auto-stream.
    python s2s_visual_pipeline.py --input-file /path/to/clip.wav --auto-replay

    # Replay a cached session without re-running PersonaPlex.
    python s2s_visual_pipeline.py --load-session <dir>/session_*.npz --auto-replay

Open http://JETSON_IP:8998/
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Environment must be fixed before torch or the server module is imported.
# ---------------------------------------------------------------------------
import os

# `expandable_segments:True` (the launcher default) caused an allocation
# failure on this Orin; `backend:native` is the validated setting.  Clear the
# newer torch 2.8 spelling so the two cannot disagree.
os.environ.pop("PYTORCH_ALLOC_CONF", None)

# These four are FORCED, not defaulted, because
# `start_original_pod_8998_repro.sh` exports them unconditionally and they are
# correctness-critical rather than preferences.  In particular
# IMTALKER_CACHED_ENGINE selects between `liveTry_cached.MoshiOnlyEngine` (has
# a prompt streaming-state cache) and `liveTry.MoshiOnlyEngine` (does not).
# With the uncached engine every `reset_session()` replays the full system
# prompt one LM step at a time -- ~1,240 tokens x ~290 ms is over five minutes,
# on startup AND on every new conversation.  An inherited
# `IMTALKER_CACHED_ENGINE=0` from the shell must not be able to cause that, so
# `setdefault` is wrong here.
_FORCED_ENV = {
    "IMTALKER_CACHED_ENGINE": "1",
    "IMTALKER_PROMPT_STATE_CACHE": "1",
    "IMTALKER_TRANSITION_BLEND_FRAMES": "0",
    "TOKENIZERS_PARALLELISM": "false",
}
ENV_OVERRIDES: dict[str, tuple[str, str]] = {}
for _key, _want in _FORCED_ENV.items():
    _prev = os.environ.get(_key)
    if _prev is not None and _prev != _want:
        ENV_OVERRIDES[_key] = (_prev, _want)
    os.environ[_key] = _want

# These two honour a deliberate shell choice.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

for _key, (_prev, _want) in ENV_OVERRIDES.items():
    print(f"[s2s] overriding inherited {_key}={_prev!r} -> {_want!r}", flush=True)
if os.environ["PYTORCH_CUDA_ALLOC_CONF"] != "backend:native":
    print(
        "[s2s] WARNING PYTORCH_CUDA_ALLOC_CONF="
        f"{os.environ['PYTORCH_CUDA_ALLOC_CONF']!r} inherited from the shell. "
        "`backend:native` is the validated Jetson setting; `expandable_segments` "
        "caused an allocation failure on this Orin.",
        flush=True,
    )

import argparse
import asyncio
import contextlib
import json
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(ROOT.parent)))
DISTILLATION_ROOT = Path(os.environ.get("DISTILLATION_ROOT", "/workspace/distillation"))
PERSONAPLEX_DIR = Path(
    os.environ.get("PERSONAPLEX_DIR", str(PROJECT_ROOT / "checkpoints" / "personaplex_bnb4"))
)

# The bundled PersonaPlex/Moshi source is imported off the tree, not from
# site-packages.  Make that work even when PYTHONPATH was not exported.
for _p in (ROOT, PERSONAPLEX_DIR / "moshi", PERSONAPLEX_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

# ---------------------------------------------------------------------------
# Constants shared with the reference server.
# ---------------------------------------------------------------------------
TARGET_SR = 24_000          # Mimi sample rate
VIDEO_FPS = 25              # IMTalker frame rate
MIMI_FRAME = 1_920          # samples per Mimi/LM step (80 ms @ 24 kHz)
SAMPLES_PER_VIDEO_FRAME = TARGET_SR // VIDEO_FPS      # 960
STEPS_PER_CHUNK = 25        # --reply_hidden_steps_per_chunk
FRAMES_PER_CHUNK = 50       # --fm_chunk_frames
SAMPLES_PER_CHUNK = STEPS_PER_CHUNK * MIMI_FRAME      # 48000 == 2.0 s
HIDDEN_DIM = 4096

assert FRAMES_PER_CHUNK * SAMPLES_PER_VIDEO_FRAME == SAMPLES_PER_CHUNK


def _ms(t0: float) -> float:
    return 1000.0 * (time.perf_counter() - t0)


def _rms(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))


# ---------------------------------------------------------------------------
# Launch configuration
# ---------------------------------------------------------------------------

def default_launch_argv(port: int, jpeg_quality: int) -> list[str]:
    """Mirror `start_original_pod_8998_repro.sh` for the student backend.

    Three deliberate deltas from the reference launcher:

      1. `--output_audio_codec pcm` instead of opus.  Replay binds each video
         frame to its own 960-sample PCM slice inside one AV01 packet, which
         makes audio/video drift structurally impossible to introduce in
         transport.  Opus is a separate persistent stream and would reintroduce
         a second clock.
      2. `--compile_renderer` omitted.  It only wraps the teacher renderer
         (`_load_renderer`); under `student_torch` the teacher is never loaded
         (`self.renderer = None`) so the flag is dead weight.
      3. `--enable_eye_blink_composite` omitted.  The server disables it for
         student backends anyway and prints a notice; the student consumes the
         exact cursor sin/cos phase inside its 34-dim condition instead.
    """
    prompt_path = ROOT / "prompts" / "RB_Robert_System_Prompt_full.txt"
    prompt = prompt_path.read_text(encoding="utf-8").replace("\n", " ")
    student_cfg = ROOT / "fixed_identity" / "configs" / "experiments" / \
        "M03_B2_BM_oral_crop_haar_seed17.json"
    student_ckpt = DISTILLATION_ROOT / "student_runs" / \
        "M03_B2_BM_oral_crop_haar_seed17" / "checkpoints" / "step_0005000.pt"

    return [
        "--host", "0.0.0.0",
        "--port", str(port),
        "--html_path", str(ROOT / "static" / "index_v3_binary_fullscreen_aj_nodrop.html"),
        "--generator_path", str(PROJECT_ROOT / "checkpoints" / "fullgen_static_2s_6400_resume" / "last.ckpt"),
        "--renderer_path", str(ROOT / "checkpoints" / "renderer.ckpt"),
        # --- fixed-identity student -------------------------------------
        "--render_backend", "student_torch",
        "--student_config", str(student_cfg),
        "--student_checkpoint", str(student_ckpt),
        "--student_weights", "model",          # raw; EMA suppresses teeth
        "--student_precision", "bf16",         # FP16 gave non-finite RGB on Orin
        "--student_max_batch", "10",
        "--render_sub_batch", "10",
        # --- UniTalk adapter ---------------------------------------------
        "--adapter_path", str(PROJECT_ROOT / "checkpoints" / "personaplex_unitalk_strict2s_2gpu_15k" / "last.pt"),
        "--adapter_type", "unitalk_last_layer",
        "--adapter_num_layers", "12",
        "--adapter_dropout", "0.0",
        "--adapter_window_mode", "lookahead",
        "--adapter_future_steps", "0",
        "--wav2vec_model_path", str(ROOT / "checkpoints" / "wav2vec2-base-960h"),
        "--ref_path", str(ROOT / "assets" / "3robert.jpeg"),
        # --- PersonaPlex ---------------------------------------------------
        "--moshi_root", str(PERSONAPLEX_DIR),
        "--mimi_hf_repo", "nvidia/personaplex-7b-v1",
        "--moshi_weight", str(PERSONAPLEX_DIR / "model_bnb_4bit.pt"),
        "--mimi_weight", str(PERSONAPLEX_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors"),
        "--tokenizer", str(PERSONAPLEX_DIR / "tokenizer_spm_32k_3.model"),
        "--quantize_4bit",
        "--text_prompt", prompt,
        "--text_prompt_path", str(prompt_path),
        "--voice_prompt", "VARM3.pt",
        "--voice_prompt_dir", str(PERSONAPLEX_DIR / "voices"),
        "--enable_moshi_reply",
        "--direct_reply_hidden",
        "--reply_hidden_steps_per_chunk", str(STEPS_PER_CHUNK),
        "--silence_helium_path", str(PROJECT_ROOT / "checkpoints" / "personaplex_lookahead_rms_adapter" / "stats" / "silence_helium_mean.pt"),
        # --- chunking / FM --------------------------------------------------
        "--audio_chunk_sec", "2.0",
        "--wav2vec_sec", "2.0",
        "--fm_chunk_frames", str(FRAMES_PER_CHUNK),
        "--helium_deque_size", str(STEPS_PER_CHUNK),
        "--prebuffer_chunks", "1",
        "--skip_fm_audio_encoder",
        "--a_cfg_scale", "1.24",
        "--nfe", "3",
        "--motion_ref_blend", "0.0",
        "--motion_prior_noise_blend", "0.0",
        "--shared_noise",
        "--seed", "42",
        "--noise_seed", "42",
        # --- output ----------------------------------------------------------
        "--renderer_precision", "fp32",
        "--frame_q_backpressure", "32",
        "--buffer_ms", "160",
        "--jpeg_quality", str(jpeg_quality),
        "--output_audio_codec", "pcm",
        "--assistant_speech_rms_threshold", "0.006",
        "--assistant_speech_hold_chunks", "1",
        "--device", "cuda",
        "--fp32",
        "--tf32",
        "--dump_dir", str(ROOT / "live_dumps_s2s_visual"),
    ]


def build_engine_args(port: int, jpeg_quality: int) -> argparse.Namespace:
    """Parse the reference option set.

    `BaseOptions.parse()` reads `sys.argv` directly, so swap it temporarily
    rather than duplicating ~120 argument definitions.
    """
    import OriginalPod8998TransitionBlend as srv

    saved = sys.argv
    try:
        sys.argv = ["s2s_visual_pipeline.py"] + default_launch_argv(port, jpeg_quality)
        args = srv.LiveHeliumFMOptions().parse()
    finally:
        sys.argv = saved
    args.rank = args.device
    return args


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One 80 ms PersonaPlex step."""
    index: int
    lm_ms: float
    encode_ms: float
    decode_ms: float
    total_ms: float
    reply_rms: float
    input_rms: float
    text: str


@dataclass
class SessionCache:
    """Everything Phase 2 needs, and nothing that requires PersonaPlex again."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    input_pcm: list[np.ndarray] = field(default_factory=list)    # 24 kHz f32
    reply_pcm: list[np.ndarray] = field(default_factory=list)    # 1920 each
    hidden: list[np.ndarray] = field(default_factory=list)       # [4096] f32
    steps: list[StepRecord] = field(default_factory=list)
    transcript: str = ""
    created_at: float = field(default_factory=time.time)

    # Timing / benchmark
    first_pcm_wall: Optional[float] = None
    first_text_wall: Optional[float] = None
    first_audio_wall: Optional[float] = None
    capture_wall_s: float = 0.0

    # ---- derived ---------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return len(self.hidden)

    @property
    def input_samples(self) -> int:
        return int(sum(p.size for p in self.input_pcm))

    @property
    def reply_samples(self) -> int:
        return int(sum(p.size for p in self.reply_pcm))

    def input_audio(self) -> np.ndarray:
        if not self.input_pcm:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.input_pcm).astype(np.float32)

    def reply_audio(self) -> np.ndarray:
        if not self.reply_pcm:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.reply_pcm).astype(np.float32)

    def hidden_matrix(self) -> np.ndarray:
        if not self.hidden:
            return np.zeros((0, HIDDEN_DIM), dtype=np.float32)
        return np.stack(self.hidden).astype(np.float32)

    def ttft_ms(self) -> Optional[float]:
        """Time to first token: first PCM in -> first sampled text piece."""
        if self.first_pcm_wall is None or self.first_text_wall is None:
            return None
        return 1000.0 * (self.first_text_wall - self.first_pcm_wall)

    def ttfa_ms(self) -> Optional[float]:
        """Time to first audio: first PCM in -> first above-threshold reply."""
        if self.first_pcm_wall is None or self.first_audio_wall is None:
            return None
        return 1000.0 * (self.first_audio_wall - self.first_pcm_wall)

    def step_throughput(self) -> dict[str, float]:
        if not self.steps:
            return {}
        lm = np.array([s.lm_ms for s in self.steps], dtype=np.float64)
        tot = np.array([s.total_ms for s in self.steps], dtype=np.float64)
        media_s = self.n_steps * MIMI_FRAME / TARGET_SR
        return {
            "steps": float(len(self.steps)),
            "lm_p50_ms": float(np.percentile(lm, 50)),
            "lm_p95_ms": float(np.percentile(lm, 95)),
            "step_p50_ms": float(np.percentile(tot, 50)),
            "step_p95_ms": float(np.percentile(tot, 95)),
            "steps_per_s": float(len(self.steps) / max(1e-6, self.capture_wall_s)),
            "media_s": float(media_s),
            "capture_wall_s": float(self.capture_wall_s),
            # >1.0 means PersonaPlex is slower than real time.
            "capture_rtf": float(self.capture_wall_s / max(1e-6, media_s)),
        }

    # ---- assistant activity span ----------------------------------------
    def active_span(self, threshold: float = 0.006, pad_steps: int = 2) -> tuple[int, int]:
        """[start, end) step range where the assistant is actually speaking.

        The model is full duplex, so the hidden/reply streams also cover the
        stretches where only the user is talking.  Trimming to the active span
        keeps replay from opening with a long silent stare.
        """
        if not self.reply_pcm:
            return (0, 0)
        active = [i for i, p in enumerate(self.reply_pcm) if _rms(p) >= threshold]
        if not active:
            return (0, self.n_steps)
        lo = max(0, active[0] - pad_steps)
        hi = min(self.n_steps, active[-1] + 1 + pad_steps)
        return (lo, hi)

    # ---- persistence -----------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"session_{self.session_id[:12]}.npz"
        np.savez_compressed(
            path,
            input_pcm=self.input_audio(),
            reply_pcm=self.reply_audio(),
            hidden=self.hidden_matrix(),
            meta=np.frombuffer(
                json.dumps({
                    "session_id": self.session_id,
                    "created_at": self.created_at,
                    "transcript": self.transcript,
                    "capture_wall_s": self.capture_wall_s,
                    "ttft_ms": self.ttft_ms(),
                    "ttfa_ms": self.ttfa_ms(),
                    "throughput": self.step_throughput(),
                    "steps": [vars(s) for s in self.steps],
                }).encode("utf-8"),
                dtype=np.uint8,
            ),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "SessionCache":
        blob = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(blob["meta"]).decode("utf-8"))
        cache = cls(session_id=str(meta.get("session_id", uuid.uuid4().hex)))
        cache.created_at = float(meta.get("created_at", time.time()))
        cache.transcript = str(meta.get("transcript", ""))
        cache.capture_wall_s = float(meta.get("capture_wall_s", 0.0))
        inp = np.asarray(blob["input_pcm"], dtype=np.float32)
        if inp.size:
            cache.input_pcm = [inp]
        rep = np.asarray(blob["reply_pcm"], dtype=np.float32)
        n_full = rep.size // MIMI_FRAME
        cache.reply_pcm = [rep[i * MIMI_FRAME:(i + 1) * MIMI_FRAME].copy() for i in range(n_full)]
        hid = np.asarray(blob["hidden"], dtype=np.float32)
        cache.hidden = [hid[i].copy() for i in range(hid.shape[0])]
        for row in meta.get("steps", []):
            cache.steps.append(StepRecord(**row))
        return cache


# ---------------------------------------------------------------------------
# Live metrics
# ---------------------------------------------------------------------------

class Metrics:
    """Thread-safe counters surfaced to /metrics and the control socket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "idle"
        self.reset_stream()
        self.personaplex: dict[str, Any] = {}

    def reset_stream(self) -> None:
        self.frames_rendered = 0
        self.frames_sent = 0
        self.chunks_rendered = 0
        self.render_ms: list[float] = []
        self.fm_ms: list[float] = []
        self.renderer_ms: list[float] = []
        self.av_delta_ms: list[float] = []
        self.starvation_events = 0
        self.stream_started: Optional[float] = None
        self.media_epoch: Optional[float] = None

    # ---- writers ---------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def note_chunk(self, *, frames: int, render_ms: float, fm_ms: float, renderer_ms: float) -> None:
        with self._lock:
            self.chunks_rendered += 1
            self.frames_rendered += frames
            self.render_ms.append(render_ms)
            self.fm_ms.append(fm_ms)
            self.renderer_ms.append(renderer_ms)

    def note_sent(self, delta_ms: float) -> None:
        with self._lock:
            self.frames_sent += 1
            self.av_delta_ms.append(delta_ms)
            if len(self.av_delta_ms) > 2000:
                del self.av_delta_ms[:1000]

    def note_starvation(self) -> None:
        with self._lock:
            self.starvation_events += 1

    # ---- reader ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            def pct(v: list[float], q: float) -> Optional[float]:
                return float(np.percentile(v, q)) if v else None

            elapsed = (
                time.perf_counter() - self.stream_started
                if self.stream_started is not None else 0.0
            )
            gpu: dict[str, Any] = {}
            if torch.cuda.is_available():
                gpu = {
                    "allocated_mib": round(torch.cuda.memory_allocated() / 2 ** 20, 1),
                    "reserved_mib": round(torch.cuda.memory_reserved() / 2 ** 20, 1),
                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                }
            return {
                "phase": self.phase,
                "personaplex": dict(self.personaplex),
                "imtalker": {
                    "chunks_rendered": self.chunks_rendered,
                    "frames_rendered": self.frames_rendered,
                    "frames_sent": self.frames_sent,
                    "render_chunk_p50_ms": pct(self.render_ms, 50),
                    "render_chunk_p95_ms": pct(self.render_ms, 95),
                    "fm_p50_ms": pct(self.fm_ms, 50),
                    "renderer_p50_ms": pct(self.renderer_ms, 50),
                    # Render throughput headroom: <1.0 means the renderer is
                    # producing faster than playback consumes.
                    "render_rtf": (
                        round(float(np.median(self.render_ms)) / 2000.0, 3)
                        if self.render_ms else None
                    ),
                    "delivered_fps": round(self.frames_sent / elapsed, 2) if elapsed > 0.5 else None,
                    "av_delta_p50_ms": pct(self.av_delta_ms, 50),
                    "av_delta_p95_ms": pct(self.av_delta_ms, 95),
                    "av_delta_max_ms": max(self.av_delta_ms) if self.av_delta_ms else None,
                    "starvation_events": self.starvation_events,
                },
                "gpu": gpu,
            }


# ---------------------------------------------------------------------------
# Phase 1: PersonaPlex capture
# ---------------------------------------------------------------------------

class PersonaPlexCapture:
    """Drive PersonaPlex step-by-step and cache everything Phase 2 needs.

    Deliberately has no playback deadline.  Steps are consumed as fast as the
    hardware manages; on the audited Orin that is roughly 0.24 s of wall time
    per 0.08 s of media.
    """

    def __init__(self, reply_engine: Any, metrics: Metrics, *, rms_threshold: float = 0.006) -> None:
        self.engine = reply_engine
        self.metrics = metrics
        self.rms_threshold = float(rms_threshold)
        self._pending = np.zeros(0, dtype=np.float32)
        self.cache = SessionCache()
        self._t_start: Optional[float] = None

    def reset(self) -> None:
        self.engine.reset_session()
        self._pending = np.zeros(0, dtype=np.float32)
        self.cache = SessionCache()
        self._t_start = None

    def feed(self, pcm_f32: np.ndarray) -> int:
        """Buffer 24 kHz mono float32 and consume every complete 80 ms frame."""
        pcm = np.asarray(pcm_f32, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return 0
        if self._t_start is None:
            self._t_start = time.perf_counter()
            self.cache.first_pcm_wall = self._t_start
        self._pending = np.concatenate([self._pending, pcm])

        consumed = 0
        while self._pending.size >= MIMI_FRAME:
            frame = self._pending[:MIMI_FRAME].copy()
            self._pending = self._pending[MIMI_FRAME:]
            self._consume_frame(frame)
            consumed += 1
        return consumed

    def flush(self) -> int:
        """Zero-pad and consume the trailing partial frame."""
        if self._pending.size == 0:
            return 0
        pad = np.zeros(MIMI_FRAME - self._pending.size, dtype=np.float32)
        frame = np.concatenate([self._pending, pad])
        self._pending = np.zeros(0, dtype=np.float32)
        self._consume_frame(frame)
        return 1

    def drain_tail(self, seconds: float = 4.0) -> int:
        """Feed silence so the assistant can finish its reply.

        PersonaPlex is full duplex; the reply continues past the end of user
        speech, so the tail has to be pumped with silence or it is truncated.
        """
        n = int(round(seconds * TARGET_SR / MIMI_FRAME))
        silence = np.zeros(MIMI_FRAME, dtype=np.float32)
        for _ in range(n):
            self._consume_frame(silence, record_input=False)
        return n

    # ---- internals -------------------------------------------------------
    def _consume_frame(self, frame: np.ndarray, *, record_input: bool = True) -> None:
        import base64

        out = self.engine._step(frame)

        reply_i16 = np.frombuffer(base64.b64decode(out["reply_i16_b64"]), dtype=np.int16)
        reply = (reply_i16.astype(np.float32) / 32768.0)
        if reply.size < MIMI_FRAME:
            reply = np.pad(reply, (0, MIMI_FRAME - reply.size))
        else:
            reply = reply[:MIMI_FRAME]

        hidden = out.get("helium_hidden")
        if hidden is None:
            # Warmup/delay steps before the LM emits; hold the last state so the
            # hidden stream stays 1:1 with the reply PCM stream.
            vec = (
                self.cache.hidden[-1].copy()
                if self.cache.hidden
                else np.zeros(HIDDEN_DIM, dtype=np.float32)
            )
        else:
            vec = np.asarray(hidden, dtype=np.float32).reshape(-1)[:HIDDEN_DIM].copy()

        idx = self.cache.n_steps
        if record_input:
            self.cache.input_pcm.append(frame.copy())
        else:
            self.cache.input_pcm.append(np.zeros(MIMI_FRAME, dtype=np.float32))
        self.cache.reply_pcm.append(reply)
        self.cache.hidden.append(vec)

        piece = str(out.get("piece", "") or "")
        if piece:
            self.cache.transcript += piece
            if self.cache.first_text_wall is None:
                self.cache.first_text_wall = time.perf_counter()
        r = float(out.get("reply_rms", _rms(reply)))
        if r >= self.rms_threshold and self.cache.first_audio_wall is None:
            self.cache.first_audio_wall = time.perf_counter()

        self.cache.steps.append(StepRecord(
            index=idx,
            lm_ms=float(out.get("lm_ms", 0.0)),
            encode_ms=float(out.get("encode_ms", 0.0)),
            decode_ms=float(out.get("decode_ms", 0.0)),
            total_ms=float(out.get("total_ms", 0.0)),
            reply_rms=r,
            input_rms=float(out.get("input_rms", 0.0)),
            text=piece,
        ))
        if self._t_start is not None:
            self.cache.capture_wall_s = time.perf_counter() - self._t_start

        # Refresh the live PersonaPlex panel every ~1 s of media.
        if idx % 12 == 0:
            snap = self.cache.step_throughput()
            snap["ttft_ms"] = self.cache.ttft_ms()
            snap["ttfa_ms"] = self.cache.ttfa_ms()
            snap["transcript"] = self.cache.transcript[-400:]
            self.metrics.personaplex = snap


# ---------------------------------------------------------------------------
# Phase 2: embedding -> motion -> frames
# ---------------------------------------------------------------------------

@dataclass
class ReplaySegment:
    """One contiguous stretch of the replay timeline."""
    kind: str                  # "input" | "reply"
    pcm: np.ndarray            # 24 kHz f32
    hidden: Optional[np.ndarray]   # [S, 4096] or None for idle


class MotionRenderPipeline:
    """UniTalk -> FM -> student renderer, driven by cached embeddings.

    Wraps the reference `LiveHeliumFMEngine` rather than reimplementing the
    motion path, so replay uses byte-identical adapter/FM/renderer code.
    """

    def __init__(self, engine: Any, metrics: Metrics) -> None:
        self.engine = engine
        self.metrics = metrics
        self.device = engine.device

    def reset(self) -> None:
        self.engine.reset_session()

    def _idle_hidden(self, steps: int) -> torch.Tensor:
        seed = getattr(self.engine, "silence_helium_seed", None)
        if seed is None:
            return torch.zeros(steps, HIDDEN_DIM, device=self.device, dtype=torch.float32)
        return seed.reshape(1, -1).to(self.device, torch.float32).expand(steps, -1).contiguous()

    def build_timeline(
        self,
        cache: SessionCache,
        *,
        include_input: bool,
        rms_threshold: float,
        pad_steps: int,
    ) -> list[ReplaySegment]:
        """Input audio first, then the assistant reply -- played sequentially."""
        segments: list[ReplaySegment] = []
        if include_input:
            inp = cache.input_audio()
            # Trim the silence tail that was pumped in to finish the reply.
            nz = np.nonzero(np.abs(inp) > 1e-4)[0]
            if nz.size:
                inp = inp[: int(nz[-1]) + 1]
            if inp.size:
                segments.append(ReplaySegment("input", inp, None))

        lo, hi = cache.active_span(rms_threshold, pad_steps)
        if hi > lo:
            hidden = cache.hidden_matrix()[lo:hi]
            pcm = cache.reply_audio()[lo * MIMI_FRAME: hi * MIMI_FRAME]
            segments.append(ReplaySegment("reply", pcm, hidden))
        return segments

    def iter_chunks(self, segments: list[ReplaySegment]) -> Iterator[dict[str, Any]]:
        """Yield one 2 s / 50-frame render job at a time.

        Chunks never straddle a segment boundary, so an "input" chunk is always
        fully idle and a "reply" chunk is always fully embedding-driven.
        """
        chunk_id = 0
        for seg in segments:
            n_chunks = max(1, int(np.ceil(seg.pcm.size / SAMPLES_PER_CHUNK)))
            for c in range(n_chunks):
                a0 = c * SAMPLES_PER_CHUNK
                pcm = seg.pcm[a0:a0 + SAMPLES_PER_CHUNK]
                if pcm.size == 0:
                    continue
                # Always hand FM a full 2 s / 25-step window -- that is the
                # window it was trained on -- but only emit the frames that
                # carry real audio, so a short segment tail does not inject
                # padded silence into the playback timeline.
                emit_frames = min(
                    FRAMES_PER_CHUNK,
                    int(np.ceil(pcm.size / SAMPLES_PER_VIDEO_FRAME)),
                )
                if pcm.size < SAMPLES_PER_CHUNK:
                    pcm = np.pad(pcm, (0, SAMPLES_PER_CHUNK - pcm.size))
                if seg.hidden is None:
                    hidden = None
                else:
                    s0 = c * STEPS_PER_CHUNK
                    h = seg.hidden[s0:s0 + STEPS_PER_CHUNK]
                    if h.shape[0] == 0:
                        continue
                    if h.shape[0] < STEPS_PER_CHUNK:
                        h = np.concatenate(
                            [h, np.repeat(h[-1:], STEPS_PER_CHUNK - h.shape[0], axis=0)]
                        )
                    hidden = h
                yield {
                    "chunk_id": chunk_id,
                    "kind": seg.kind,
                    "pcm": pcm,
                    "hidden": hidden,
                    "emit_frames": emit_frames,
                }
                chunk_id += 1

    @torch.no_grad()
    def render_chunk(self, job: dict[str, Any], frame_offset: int) -> list[dict[str, Any]]:
        """One chunk -> up to 50 AV01 packets, each carrying its own 960 PCM."""
        t_chunk = time.perf_counter()
        eng = self.engine

        if job["hidden"] is None:
            helium = self._idle_hidden(STEPS_PER_CHUNK)
        else:
            helium = torch.from_numpy(np.ascontiguousarray(job["hidden"])).to(
                self.device, torch.float32, non_blocking=True
            )

        t_fm = time.perf_counter()
        motion, info = eng._sample_motion_from_helium(helium, FRAMES_PER_CHUNK)
        fm_ms = _ms(t_fm)

        pcm = job["pcm"]
        slices = [
            pcm[i * SAMPLES_PER_VIDEO_FRAME:(i + 1) * SAMPLES_PER_VIDEO_FRAME]
            for i in range(FRAMES_PER_CHUNK)
        ]

        packets: list[dict[str, Any]] = []
        renderer_ms = 0.0
        sub = max(1, int(eng.render_sub_batch))
        emit = min(int(job.get("emit_frames", FRAMES_PER_CHUNK)), int(motion.shape[0]))
        for s0 in range(0, emit, sub):
            s1 = min(s0 + sub, emit)
            t_r = time.perf_counter()
            pkts, _bench = eng.render_and_encode_subbatch(
                motion[s0:s1],
                slices[s0:s1],
                frame_offset + s0,
                "",
                int(job["chunk_id"]) + 1,
                float(info.get("fm_ms", 0.0)),
            )
            renderer_ms += _ms(t_r)
            packets.extend(pkts)

        self.metrics.note_chunk(
            frames=len(packets),
            render_ms=_ms(t_chunk),
            fm_ms=fm_ms,
            renderer_ms=renderer_ms,
        )
        return packets


# ---------------------------------------------------------------------------
# Application state machine
# ---------------------------------------------------------------------------

STATE_IDLE = "idle"
STATE_CAPTURING = "capturing"
STATE_AWAITING_CHOICE = "awaiting_choice"
STATE_STREAMING = "streaming"


@dataclass
class WarmupReport:
    """Result of the post-load end-to-end warmup."""
    personaplex_steps: int = 0
    personaplex_first_ms: float = 0.0
    personaplex_steady_p50_ms: float = 0.0
    personaplex_hidden_ok: bool = False
    imtalker_chunks: int = 0
    imtalker_first_ms: float = 0.0
    imtalker_steady_ms: float = 0.0
    fm_ms: float = 0.0
    renderer_ms: float = 0.0
    frames: int = 0
    used_live_hidden: bool = False
    total_s: float = 0.0
    peak_alloc_mib: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items()}


class PipelineApp:
    """Owns the engines, the state machine and the replay scheduler.

    Startup is strictly ordered so a failure is attributable:

        1. resolve the reference launch profile
        2. load IMTalker (student renderer + UniTalk adapter + FM generator)
        3. load PersonaPlex-7B NF4 and apply the initialization prompt
        4. verify the persona prompt actually took effect
        5. warm both models through their exact production call paths
    """

    def __init__(self, opts: argparse.Namespace) -> None:
        self.opts = opts
        self.metrics = Metrics()
        self.state = STATE_IDLE
        self.cache: Optional[SessionCache] = None
        self.session_dir = Path(opts.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.ready = False
        self.warmup_report: Optional[WarmupReport] = None
        self.prompt_info: dict[str, Any] = {}
        self.load_times: dict[str, float] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.upload_status: dict[str, Any] = {
            "active": False, "name": None, "duration_s": None, "error": None,
        }

        # Replay plumbing
        self.frame_q: Optional[asyncio.Queue] = None
        self.video_session_id: Optional[str] = None
        self.media_epoch: Optional[float] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_stream = threading.Event()
        self._capture_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._capture_thread: Optional[threading.Thread] = None
        self.last_session_path: Optional[Path] = None

        print("[s2s][load 1/5] resolving the reference launch profile", flush=True)
        self.args = build_engine_args(opts.port, opts.jpeg_quality)
        import OriginalPod8998TransitionBlend as srv
        self._srv = srv

        # The engine class is chosen at server-module import time from
        # IMTALKER_CACHED_ENGINE.  Verify the cached one won: the uncached
        # engine has no prompt streaming-state cache, so every session reset
        # would replay the whole system prompt for minutes.
        engine_module = srv.MoshiOnlyEngine.__module__
        has_cache = hasattr(srv.MoshiOnlyEngine, "_capture_prompt_state_cache")
        print(
            f"[s2s][load 1/5] PersonaPlex engine={engine_module}."
            f"{srv.MoshiOnlyEngine.__name__} prompt_state_cache={has_cache}",
            flush=True,
        )
        if not has_cache:
            raise RuntimeError(
                f"The uncached PersonaPlex engine ({engine_module}) was bound. Every "
                "reset_session() would replay the full system prompt one LM step at a "
                "time (minutes each, on startup and on every new conversation). This "
                "is selected by IMTALKER_CACHED_ENGINE at server-module import time -- "
                "check that nothing re-exports it to 0 after this script's preamble."
            )

        t_all = time.perf_counter()
        self._load_imtalker()
        self._load_personaplex()

        self.capture = PersonaPlexCapture(
            self.reply_engine, self.metrics,
            rms_threshold=float(self.args.assistant_speech_rms_threshold),
        )
        self.renderer = MotionRenderPipeline(self.engine, self.metrics)

        self._verify_init_prompt()

        if opts.warmup:
            self.warmup(
                pp_steps=int(opts.warmup_steps),
                im_chunks=int(opts.warmup_chunks),
            )
        else:
            print("[s2s][load 5/5] warmup skipped (--no-warmup)", flush=True)

        self.ready = True
        print(
            f"[s2s] READY -- both models loaded and warm in {_ms(t_all)/1000:.1f}s "
            f"(imtalker={self.load_times.get('imtalker_s', 0):.1f}s "
            f"personaplex={self.load_times.get('personaplex_s', 0):.1f}s "
            f"warmup={self.load_times.get('warmup_s', 0):.1f}s)",
            flush=True,
        )

    # ---- startup ---------------------------------------------------------
    def _load_imtalker(self) -> None:
        """Student renderer + UniTalk adapter + FM generator.

        `LiveHeliumFMEngine.__init__` runs its own internal warmup (one FM
        sample and one renderer sub-batch).  That is kept, and `warmup()` then
        exercises the full chunk path this pipeline actually calls.
        """
        print(
            "[s2s][load 2/5] IMTalker: student renderer + UniTalk adapter + FM "
            f"(backend={self.args.render_backend}, precision={self.args.student_precision})",
            flush=True,
        )
        t0 = time.perf_counter()
        self.engine = self._srv.LiveHeliumFMEngine(self.args)
        self.load_times["imtalker_s"] = _ms(t0) / 1000.0
        backend = getattr(self.engine, "student_render_backend", None)
        detail = ""
        if backend is not None:
            detail = (
                f" arch={backend.architecture} step={backend.checkpoint_step} "
                f"weights={backend.weights} blink_cycle={backend.blink_cycle_length}"
            )
        print(f"[s2s][load 2/5] IMTalker ready in {self.load_times['imtalker_s']:.1f}s{detail}", flush=True)

    def _load_personaplex(self) -> None:
        """PersonaPlex-7B NF4 + Mimi + the initialization prompt.

        `MoshiOnlyEngineWithHidden.__init__` installs the layer[-2] hidden
        capture as a native `LMGen` output (so CUDA graph replay survives),
        runs its runtime warmup, captures the post-warmup streaming-state
        cache, and calls `reset_session()` -> `_apply_system_prompts()`.  That
        is the same initialization order the reference server uses.
        """
        print(
            "[s2s][load 3/5] PersonaPlex-7B: "
            f"quantize_4bit={bool(self.args.quantize_4bit)} "
            f"codebooks={self.args.num_codebooks} "
            f"voice={self.args.voice_prompt} "
            f"cfg={self.args.moshi_cfg_coef}",
            flush=True,
        )
        t0 = time.perf_counter()
        self.reply_engine = self._srv.MoshiOnlyEngineWithHidden(
            moshi_root=self.args.moshi_root,
            mimi_hf_repo=self.args.mimi_hf_repo,
            device=self.args.device,
            cfg_coef=float(self.args.moshi_cfg_coef),
            placeholder_jpeg_b64="",
            moshi_weight=self.args.moshi_weight,
            mimi_weight=self.args.mimi_weight,
            tokenizer=self.args.tokenizer,
            quantize_4bit=bool(self.args.quantize_4bit),
            num_codebooks=int(self.args.num_codebooks),
            context=(int(self.args.moshi_context) if int(self.args.moshi_context) > 0 else None),
            voice_prompt=self.args.voice_prompt,
            voice_prompt_dir=self.args.voice_prompt_dir,
            text_prompt=self.args.text_prompt,
        )
        self.load_times["personaplex_s"] = _ms(t0) / 1000.0
        print(f"[s2s][load 3/5] PersonaPlex ready in {self.load_times['personaplex_s']:.1f}s", flush=True)

    def _verify_init_prompt(self) -> None:
        """Fail loudly if the Robert persona did not actually load.

        `_apply_system_prompts` wraps its tokenizer call in
        `contextlib.suppress(Exception)`, so a tokenizer problem leaves
        `text_prompt_tokens` empty and the model answers as a generic
        assistant with no visible error.  Check it explicitly instead.
        """
        import hashlib

        eng = self.reply_engine
        lm_gen = eng.lm_gen
        tokens = getattr(lm_gen, "text_prompt_tokens", None)
        n_tokens = len(tokens) if tokens else 0
        prompt_text = str(getattr(eng, "text_prompt", "") or "")
        voice_path = ""
        with contextlib.suppress(Exception):
            voice_path = eng._resolve_voice_prompt_path()

        prompt_path = Path(str(getattr(self.args, "text_prompt_path", "") or ""))
        self.prompt_info = {
            "text_prompt_path": str(prompt_path),
            "text_prompt_sha256": (
                hashlib.sha256(prompt_path.read_bytes()).hexdigest()
                if prompt_path.is_file() else None
            ),
            "text_prompt_chars": len(prompt_text),
            "text_prompt_tokens": n_tokens,
            "newline_free": "\n" not in prompt_text,
            "voice_prompt": str(getattr(eng, "voice_prompt", "")),
            "voice_prompt_path": voice_path,
            "prompt_state_cache_warm": getattr(eng, "_prompt_state_cache", None) is not None,
            "prompt_state_cache_hits": int(getattr(eng, "_prompt_state_cache_hits", 0)),
        }

        if not prompt_text:
            raise RuntimeError(
                "PersonaPlex initialization prompt is empty; expected the contents of "
                f"{prompt_path}"
            )
        if n_tokens == 0:
            raise RuntimeError(
                "PersonaPlex accepted the prompt text but produced 0 prompt tokens. "
                "`_apply_system_prompts` suppresses tokenizer errors, so the persona "
                "would be silently absent. Check --tokenizer and the sentencepiece model."
            )
        if not voice_path:
            print(
                "[s2s][load 4/5] WARNING no voice prompt resolved; "
                "the reply voice will not be VARM3",
                flush=True,
            )
        if not self.prompt_info["prompt_state_cache_warm"]:
            raise RuntimeError(
                "PersonaPlex finished loading with a cold prompt streaming-state cache. "
                "Every reset_session() -- startup warmup, and every new conversation -- "
                f"would replay all {n_tokens} prompt tokens one LM step at a time. "
                "Expected `_capture_prompt_state_cache()` to have run during engine "
                "construction; check IMTALKER_PROMPT_STATE_CACHE and that the cached "
                "engine (liveTry_cached) is the one bound."
            )

        print(
            "[s2s][load 4/5] init prompt OK: "
            f"{self.prompt_info['text_prompt_chars']} chars -> {n_tokens} tokens, "
            f"sha256={str(self.prompt_info['text_prompt_sha256'])[:12]}, "
            f"voice={Path(voice_path).name if voice_path else 'none'}, "
            f"state_cache=warm (hits={self.prompt_info['prompt_state_cache_hits']})",
            flush=True,
        )

    # ---- warmup ----------------------------------------------------------
    @torch.no_grad()
    def warmup(self, *, pp_steps: int = 12, im_chunks: int = 2) -> WarmupReport:
        """Run both models once through their exact production call paths.

        This is not the engines' internal warmup, which only covers
        `LMGen.step` and one bare renderer sub-batch.  Here we drive:

          * `MoshiOnlyEngineWithHidden._step` -- the patched layer[-2] capture,
            Mimi encode/decode, the host-side base64 and D2H copies;
          * `_sample_motion_from_helium` -- UniTalk adapter + FM projection +
            flow matching, including the streaming state carry;
          * `render_and_encode_subbatch` -- student renderer, uint8 conversion,
            frame D2H, the JPEG thread pool and AV01 packing.

        The hidden states produced by the PersonaPlex warmup feed the IMTalker
        warmup, so the motion path is warmed with genuinely in-domain input
        rather than a synthetic latent.  Both engines are reset afterwards, so
        the prompt state and frame cursors are exactly as a fresh session finds
        them.
        """
        report = WarmupReport()
        t_all = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print(f"[s2s][load 5/5] warmup: PersonaPlex x{pp_steps} steps", flush=True)
        silence = np.zeros(MIMI_FRAME, dtype=np.float32)
        step_ms: list[float] = []
        hidden: list[np.ndarray] = []
        for i in range(max(1, int(pp_steps))):
            t0 = time.perf_counter()
            out = self.reply_engine._step(silence)
            step_ms.append(_ms(t0))
            h = out.get("helium_hidden")
            if h is not None:
                hidden.append(np.asarray(h, dtype=np.float32).reshape(-1)[:HIDDEN_DIM].copy())

        report.personaplex_steps = len(step_ms)
        report.personaplex_first_ms = step_ms[0]
        steady = step_ms[1:] or step_ms
        report.personaplex_steady_p50_ms = float(np.percentile(steady, 50))
        report.personaplex_hidden_ok = len(hidden) > 0
        if not report.personaplex_hidden_ok:
            raise RuntimeError(
                "PersonaPlex warmup produced no layer[-2] hidden states. The graphed "
                "hidden-capture install did not take effect, so IMTalker would have "
                "nothing to render from."
            )
        print(
            f"[s2s][load 5/5] PersonaPlex warm: first={report.personaplex_first_ms:.0f}ms "
            f"steady_p50={report.personaplex_steady_p50_ms:.0f}ms "
            f"hidden={len(hidden)}/{len(step_ms)} steps captured "
            f"(step RTF {report.personaplex_steady_p50_ms / 80.0:.2f}x)",
            flush=True,
        )

        # Restore the prompt state that the warmup steps advanced past.
        self.reply_engine.reset_session()

        # ---- IMTalker: full chunk path, fed by the hidden we just captured --
        print(f"[s2s][load 5/5] warmup: IMTalker x{im_chunks} chunks", flush=True)
        if len(hidden) >= STEPS_PER_CHUNK:
            window = np.stack(hidden[-STEPS_PER_CHUNK:])
            report.used_live_hidden = True
        elif hidden:
            pad = np.repeat(hidden[-1][None, :], STEPS_PER_CHUNK - len(hidden), axis=0)
            window = np.concatenate([np.stack(hidden), pad])
            report.used_live_hidden = True
        else:  # unreachable -- guarded above
            window = np.zeros((STEPS_PER_CHUNK, HIDDEN_DIM), dtype=np.float32)

        pcm = np.zeros(SAMPLES_PER_CHUNK, dtype=np.float32)
        self.renderer.reset()
        chunk_ms: list[float] = []
        # Cover the full sub-batch and a short tail: the student slices by
        # max_batch, so the tail shape allocates its own cuDNN workspace.
        emit_plan = [FRAMES_PER_CHUNK] * max(1, int(im_chunks)) + [3]
        frames = 0
        for n, emit in enumerate(emit_plan):
            job = {
                "chunk_id": n,
                "kind": "warmup",
                "pcm": pcm,
                "hidden": window,
                "emit_frames": emit,
            }
            t0 = time.perf_counter()
            packets = self.renderer.render_chunk(job, frames)
            chunk_ms.append(_ms(t0))
            frames += len(packets)

        report.imtalker_chunks = len(chunk_ms)
        report.imtalker_first_ms = chunk_ms[0]
        report.imtalker_steady_ms = float(np.median(chunk_ms[1:-1] or chunk_ms))
        report.frames = frames
        snap = self.metrics.snapshot()["imtalker"]
        report.fm_ms = float(snap.get("fm_p50_ms") or 0.0)
        report.renderer_ms = float(snap.get("renderer_p50_ms") or 0.0)
        print(
            f"[s2s][load 5/5] IMTalker warm: first={report.imtalker_first_ms:.0f}ms "
            f"steady={report.imtalker_steady_ms:.0f}ms "
            f"fm_p50={report.fm_ms:.1f}ms renderer_p50={report.renderer_ms:.0f}ms "
            f"frames={frames} "
            f"(chunk RTF {report.imtalker_steady_ms / 2000.0:.2f}x)",
            flush=True,
        )

        # Drop the warmup samples so the first replay is measured clean.
        self.renderer.reset()
        self.metrics.reset_stream()
        self.metrics.set_phase(STATE_IDLE)

        if torch.cuda.is_available():
            report.peak_alloc_mib = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        report.total_s = _ms(t_all) / 1000.0
        self.warmup_report = report
        self.load_times["warmup_s"] = report.total_s

        pp_rtf = report.personaplex_steady_p50_ms / 80.0
        im_rtf = report.imtalker_steady_ms / 2000.0
        print(
            f"[s2s][load 5/5] warmup done in {report.total_s:.1f}s  "
            f"peak_alloc={report.peak_alloc_mib:.0f}MiB\n"
            f"[s2s]           capture RTF ~{pp_rtf:.2f}x (PersonaPlex, off the playback path)\n"
            f"[s2s]           replay  RTF ~{im_rtf:.2f}x (IMTalker, on the playback path)"
            + ("" if im_rtf < 1.0 else "   <-- WARNING: renderer cannot keep up with playback"),
            flush=True,
        )
        return report

    # ---- phase 1 ---------------------------------------------------------
    def start_capture(self) -> None:
        if self.state == STATE_CAPTURING:
            return
        self.reset()
        self.state = STATE_CAPTURING
        self.metrics.set_phase(STATE_CAPTURING)
        self.capture.reset()
        self._capture_q = queue.Queue()

        def _worker() -> None:
            while True:
                item = self._capture_q.get()
                if item is None:
                    break
                try:
                    self.capture.feed(item)
                except Exception as exc:   # keep the socket alive on model error
                    print(f"[s2s][capture] step failed: {exc!r}", flush=True)

        self._capture_thread = threading.Thread(target=_worker, name="pp-capture", daemon=True)
        self._capture_thread.start()
        print("[s2s] capture started", flush=True)

    def feed_capture(self, pcm: np.ndarray) -> None:
        if self.state == STATE_CAPTURING:
            self._capture_q.put(pcm)

    def end_capture(self) -> SessionCache:
        if self._capture_thread is not None:
            self._capture_q.put(None)
            self._capture_thread.join()
            self._capture_thread = None
        self.capture.flush()
        if self.opts.tail_silence_s > 0:
            print(f"[s2s] draining {self.opts.tail_silence_s:.1f}s reply tail", flush=True)
            self.capture.drain_tail(self.opts.tail_silence_s)

        cache = self.capture.cache
        snap = cache.step_throughput()
        snap["ttft_ms"] = cache.ttft_ms()
        snap["ttfa_ms"] = cache.ttfa_ms()
        snap["transcript"] = cache.transcript[-400:]
        self.metrics.personaplex = snap
        self.cache = cache
        self.state = STATE_AWAITING_CHOICE
        self.metrics.set_phase(STATE_AWAITING_CHOICE)

        if self.opts.save_sessions:
            self.last_session_path = cache.save(self.session_dir)
            print(f"[s2s] session cached -> {self.last_session_path}", flush=True)

        print(
            "[s2s][BENCH] PersonaPlex "
            f"steps={snap.get('steps', 0):.0f} "
            f"media={snap.get('media_s', 0):.1f}s "
            f"wall={snap.get('capture_wall_s', 0):.1f}s "
            f"RTF={snap.get('capture_rtf', 0):.2f} "
            f"lm_p50={snap.get('lm_p50_ms', 0):.1f}ms "
            f"step_p50={snap.get('step_p50_ms', 0):.1f}ms "
            f"TTFT={snap.get('ttft_ms') if snap.get('ttft_ms') is None else round(snap['ttft_ms'])}ms "
            f"TTFA={snap.get('ttfa_ms') if snap.get('ttfa_ms') is None else round(snap['ttfa_ms'])}ms",
            flush=True,
        )
        return cache

    def capture_from_file(self, path: Path) -> SessionCache:
        """Run Phase 1 from an audio file instead of the mic.

        Shared by `--input-file` and the interactive upload endpoint so both
        behave identically -- same decode, resample, normalization and tail
        drain.
        """
        pcm = load_audio_file(path)
        duration = pcm.size / TARGET_SR
        print(
            f"[s2s] file capture {path.name}: {duration:.2f}s @ {TARGET_SR} Hz mono "
            f"(expect ~{duration * 3.6:.0f}s of PersonaPlex wall time)",
            flush=True,
        )
        self.upload_status = {
            "active": True, "name": path.name,
            "duration_s": round(duration, 2), "error": None,
        }
        try:
            self.start_capture()
            t0 = time.perf_counter()
            # 1 s bites so the metrics panel advances while a long file runs.
            for i in range(0, pcm.size, TARGET_SR):
                self.feed_capture(pcm[i:i + TARGET_SR])
            cache = self.end_capture()
            print(f"[s2s] file capture finished in {_ms(t0)/1000:.1f}s", flush=True)
            return cache
        except Exception as exc:
            self.upload_status["error"] = repr(exc)
            raise
        finally:
            self.upload_status["active"] = False

    # ---- phase 2 ---------------------------------------------------------
    def discard(self) -> None:
        print("[s2s] choice A: discard and reset", flush=True)
        self.reset()

    def start_stream(self, loop: asyncio.AbstractEventLoop) -> str:
        if self.cache is None or self.cache.n_steps == 0:
            raise RuntimeError("no cached session to replay")
        self.state = STATE_STREAMING
        self.metrics.reset_stream()
        self.metrics.set_phase(STATE_STREAMING)

        segments = self.renderer.build_timeline(
            self.cache,
            include_input=not self.opts.reply_only,
            rms_threshold=float(self.args.assistant_speech_rms_threshold),
            pad_steps=self.opts.active_pad_steps,
        )
        total_s = sum(s.pcm.size for s in segments) / TARGET_SR
        print(
            "[s2s] replay timeline: "
            + " + ".join(f"{s.kind}={s.pcm.size / TARGET_SR:.1f}s" for s in segments)
            + f"  total={total_s:.1f}s",
            flush=True,
        )

        self.video_session_id = uuid.uuid4().hex
        self.frame_q = asyncio.Queue(maxsize=int(self.args.frame_q_backpressure))
        self.media_epoch = None
        self._stop_stream.clear()

        self._stream_thread = threading.Thread(
            target=self._render_worker,
            args=(loop, segments),
            name="replay-render",
            daemon=True,
        )
        self._stream_thread.start()
        return self.video_session_id

    def _render_worker(self, loop: asyncio.AbstractEventLoop, segments: list[ReplaySegment]) -> None:
        """Render ahead of the playback clock by a bounded lead."""
        assert self.frame_q is not None
        self.renderer.reset()
        frame_index = 0
        lead = float(self.opts.render_lead_s)

        try:
            for job in self.renderer.iter_chunks(segments):
                if self._stop_stream.is_set():
                    break

                # Throttle to `lead` seconds ahead of playback so a long
                # session does not render entirely into the queue up front.
                if self.media_epoch is not None:
                    due = self.media_epoch + (frame_index / VIDEO_FPS) - lead
                    while not self._stop_stream.is_set():
                        wait = due - time.perf_counter()
                        if wait <= 0:
                            break
                        time.sleep(min(wait, 0.05))

                packets = self.renderer.render_chunk(job, frame_index)
                frame_index += len(packets)

                for pkt in packets:
                    if self._stop_stream.is_set():
                        break
                    fut = asyncio.run_coroutine_threadsafe(self.frame_q.put(pkt), loop)
                    try:
                        fut.result(timeout=float(self.opts.client_wait_s))
                    except Exception:
                        self._stop_stream.set()
                        break

                if job["chunk_id"] % 5 == 0:
                    snap = self.metrics.snapshot()["imtalker"]
                    print(
                        f"[s2s][BENCH] chunk={job['chunk_id']} kind={job['kind']} "
                        f"render_p50={snap['render_chunk_p50_ms']:.0f}ms "
                        f"render_rtf={snap['render_rtf']} "
                        f"frames={frame_index}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[s2s][render] worker failed: {exc!r}", flush=True)
        finally:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self.frame_q.put(None), loop).result(timeout=10.0)
            print(f"[s2s] render worker done, frames={frame_index}", flush=True)

    def stop_stream(self) -> None:
        self._stop_stream.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=10.0)
            self._stream_thread = None

    def reset(self) -> None:
        self.stop_stream()
        if self._capture_thread is not None:
            self._capture_q.put(None)
            self._capture_thread.join(timeout=10.0)
            self._capture_thread = None
        self.cache = None
        self.frame_q = None
        self.video_session_id = None
        self.media_epoch = None
        self.state = STATE_IDLE
        self.metrics.set_phase(STATE_IDLE)
        self.metrics.personaplex = {}
        self.metrics.reset_stream()

    def status(self) -> dict[str, Any]:
        cache = self.cache
        return {
            "state": self.state,
            "ready": self.ready,
            "warm": self.warmup_report is not None,
            "warmup": None if self.warmup_report is None else self.warmup_report.as_dict(),
            "prompt": self.prompt_info,
            "load_times": self.load_times,
            "upload": self.upload_status,
            "video_session_id": self.video_session_id,
            "has_session": cache is not None and cache.n_steps > 0,
            "session": None if cache is None else {
                "session_id": cache.session_id,
                "steps": cache.n_steps,
                "input_s": round(cache.input_samples / TARGET_SR, 2),
                "reply_s": round(cache.reply_samples / TARGET_SR, 2),
                "transcript": cache.transcript[-400:],
                "ttft_ms": cache.ttft_ms(),
                "ttfa_ms": cache.ttfa_ms(),
            },
        }


# ---------------------------------------------------------------------------
# HTTP / WebSocket surface
# ---------------------------------------------------------------------------

def build_app(pipeline: PipelineApp) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        # The replay render worker lives on a thread and needs a handle to the
        # serving loop to hand packets back through the asyncio queue.
        pipeline.loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title="IMTalker S2S visual pipeline", lifespan=lifespan)
    ui_path = ROOT / "static" / "s2s_visual_ui.html"
    started = time.perf_counter()

    @app.get("/")
    async def index():
        if ui_path.is_file():
            return HTMLResponse(
                ui_path.read_text(encoding="utf-8"),
                headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
            )
        return HTMLResponse(f"<h1>Missing UI</h1><p>{ui_path}</p>", status_code=500)

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "s2s_visual_pipeline",
            "uptime_sec": round(time.perf_counter() - started, 3),
            "render_backend": pipeline.args.render_backend,
            "student_precision": pipeline.args.student_precision,
            **pipeline.status(),
        })

    @app.get("/metrics")
    async def metrics():
        return JSONResponse({**pipeline.metrics.snapshot(), **pipeline.status()})

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        """Interactive-mode audio upload: same path as `--input-file`.

        Returns as soon as the file is on disk; capture runs on a worker thread
        because it is ~3.6x real time.  The control socket's metrics stream
        reports progress and flips to `awaiting_choice` when it finishes.
        """
        if pipeline.state != STATE_IDLE:
            return JSONResponse(
                {"ok": False, "error": f"pipeline is {pipeline.state}; reset first"},
                status_code=409,
            )
        name = Path(str(file.filename or "upload.wav")).name
        suffix = Path(name).suffix.lower() or ".wav"
        target = pipeline.session_dir / f"upload_{uuid.uuid4().hex[:8]}{suffix}"
        target.write_bytes(await file.read())

        def _worker() -> None:
            try:
                pipeline.capture_from_file(target)
            except Exception as exc:
                print(f"[s2s][upload] capture failed: {exc!r}", flush=True)

        threading.Thread(target=_worker, name="upload-capture", daemon=True).start()
        return JSONResponse({
            "ok": True, "name": name,
            "stored": str(target), "bytes": target.stat().st_size,
        }, status_code=202)

    # -- control channel: capture, A/B prompt, live metrics ------------------
    @app.websocket("/ws/control")
    async def control(ws: WebSocket):
        await ws.accept()
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        async def push_metrics() -> None:
            while not stop.is_set():
                with contextlib.suppress(Exception):
                    await ws.send_json({
                        "type": "metrics",
                        **pipeline.metrics.snapshot(),
                        **pipeline.status(),
                    })
                await asyncio.sleep(0.25)

        pump = asyncio.create_task(push_metrics())
        try:
            await ws.send_json({"type": "state", **pipeline.status()})
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                if msg.get("bytes") is not None:
                    # Raw mic audio: float32 LE, mono, 24 kHz.
                    pcm = np.frombuffer(msg["bytes"], dtype="<f4")
                    pipeline.feed_capture(pcm)
                    continue

                if msg.get("text") is None:
                    continue
                payload = json.loads(msg["text"])
                kind = str(payload.get("type", ""))

                if kind == "start_capture":
                    pipeline.start_capture()
                    await ws.send_json({"type": "state", **pipeline.status()})

                elif kind == "end_capture":
                    await ws.send_json({"type": "state", "state": "finalizing"})
                    cache = await loop.run_in_executor(None, pipeline.end_capture)
                    await ws.send_json({
                        "type": "prompt",
                        "message": (
                            f"Captured {cache.n_steps} PersonaPlex steps "
                            f"({cache.n_steps * MIMI_FRAME / TARGET_SR:.1f}s of media). "
                            "Choose (A) discard and reset, or (B) start the realtime video stream."
                        ),
                        **pipeline.status(),
                        "personaplex": pipeline.metrics.personaplex,
                    })

                elif kind == "choice":
                    value = str(payload.get("value", "")).upper()
                    if value == "A":
                        pipeline.discard()
                        await ws.send_json({"type": "state", **pipeline.status()})
                    elif value == "B":
                        sid = pipeline.start_stream(loop)
                        await ws.send_json({
                            "type": "stream_ready",
                            "video_session_id": sid,
                            **pipeline.status(),
                        })
                    else:
                        await ws.send_json({"type": "error", "message": "choice must be A or B"})

                elif kind == "reset":
                    pipeline.reset()
                    await ws.send_json({"type": "state", **pipeline.status()})

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"[s2s][control] {exc!r}", flush=True)
        finally:
            stop.set()
            pump.cancel()
            with contextlib.suppress(Exception):
                await pump

    # -- video channel: AV01 packets paced against the media epoch -----------
    @app.websocket("/ws/video")
    async def video(ws: WebSocket):
        await ws.accept()
        sid = str(ws.query_params.get("session_id", ""))
        if sid != pipeline.video_session_id or pipeline.frame_q is None:
            await ws.send_json({"type": "error", "message": "unknown or expired video session"})
            await ws.close()
            return

        frame_q = pipeline.frame_q
        # Anchor the media clock only once the first frame exists, plus a small
        # lead so the client can prime its audio graph.
        first = await frame_q.get()
        if first is None:
            await ws.close()
            return
        epoch = time.perf_counter() + float(pipeline.args.buffer_ms) / 1000.0
        pipeline.media_epoch = epoch
        pipeline.metrics.media_epoch = epoch
        pipeline.metrics.stream_started = time.perf_counter()

        sent = 0
        packet: Optional[dict[str, Any]] = first
        print(f"[s2s][VIDEO] connected session={sid[:8]} epoch=+{pipeline.args.buffer_ms}ms", flush=True)
        try:
            while packet is not None:
                idx = int(packet["frame_number"])
                target = epoch + idx / float(VIDEO_FPS)
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)

                await ws.send_bytes(packet["data"])
                sent += 1
                # Positive = frame left late relative to its audio position.
                pipeline.metrics.note_sent(1000.0 * (time.perf_counter() - target))

                try:
                    packet = frame_q.get_nowait()
                except asyncio.QueueEmpty:
                    pipeline.metrics.note_starvation()
                    packet = await frame_q.get()
        except (WebSocketDisconnect, RuntimeError) as exc:
            print(f"[s2s][VIDEO] closed {exc!r}", flush=True)
        except Exception as exc:
            print(f"[s2s][VIDEO] error {exc!r}", flush=True)
        finally:
            snap = pipeline.metrics.snapshot()["imtalker"]
            print(
                f"[s2s][VIDEO] done sent={sent} "
                f"fps={snap['delivered_fps']} "
                f"av_delta_p50={snap['av_delta_p50_ms']}ms "
                f"p95={snap['av_delta_p95_ms']}ms "
                f"starve={snap['starvation_events']}",
                flush=True,
            )
            if pipeline.state == STATE_STREAMING:
                pipeline.state = STATE_AWAITING_CHOICE if pipeline.cache else STATE_IDLE
                pipeline.metrics.set_phase(pipeline.state)

    return app


# ---------------------------------------------------------------------------
# Offline file mode
# ---------------------------------------------------------------------------

def load_audio_file(path: Path) -> np.ndarray:
    """Decode .wav/.mp3/anything ffmpeg reads to 24 kHz mono float32."""
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    pcm = wav.squeeze(0).contiguous().to(torch.float32).numpy()
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak > 1.0:
        pcm = pcm / peak
    return pcm.astype(np.float32)


def run_offline_capture(pipeline: PipelineApp, path: Path) -> SessionCache:
    """`--input-file` entry point; the work lives on PipelineApp so the
    interactive upload endpoint takes the identical path."""
    return pipeline.capture_from_file(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="s2s_visual_pipeline.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8998)))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--jpeg-quality", type=int, default=58)

    p.add_argument("--input-file", type=Path, default=None,
                   help="Offline mode: decode this .wav/.mp3, run PersonaPlex, then replay.")
    p.add_argument("--load-session", type=Path, default=None,
                   help="Skip PersonaPlex entirely and replay a cached session_*.npz.")
    p.add_argument("--auto-replay", action="store_true",
                   help="Answer the (A)/(B) prompt with B automatically.")

    p.add_argument("--session-dir", default=str(ROOT / "s2s_sessions"))
    p.add_argument("--no-save-sessions", dest="save_sessions", action="store_false",
                   help="Do not write session_*.npz caches to disk.")
    p.set_defaults(save_sessions=True)

    p.add_argument("--tail-silence-s", type=float, default=4.0,
                   help="Silence pumped after user speech so the full-duplex reply can finish.")
    p.add_argument("--reply-only", action="store_true",
                   help="Replay only the assistant audio, skipping the input-audio segment.")
    p.add_argument("--active-pad-steps", type=int, default=2,
                   help="80 ms steps kept either side of the assistant-active span.")
    p.add_argument("--render-lead-s", type=float, default=2.0,
                   help="How far ahead of the playback clock the renderer may run.")
    p.add_argument("--client-wait-s", type=float, default=180.0,
                   help="How long the render worker waits for a browser before giving up.")

    p.add_argument("--no-warmup", dest="warmup", action="store_false",
                   help="Skip the post-load end-to-end warmup inference.")
    p.set_defaults(warmup=True)
    p.add_argument("--warmup-steps", type=int, default=12,
                   help="PersonaPlex 80 ms steps driven during warmup.")
    p.add_argument("--warmup-chunks", type=int, default=2,
                   help="Full 2 s IMTalker chunks rendered during warmup.")
    return p.parse_args()


def _cli_prompt_choice(cache: SessionCache, auto: bool) -> str:
    banner = (
        "\n"
        "==================================================================\n"
        f"  Conversation captured: {cache.n_steps} steps "
        f"({cache.n_steps * MIMI_FRAME / TARGET_SR:.1f}s of media)\n"
        f"  Transcript: {cache.transcript[-200:] or '(none)'}\n"
        "------------------------------------------------------------------\n"
        "  (A) Discard and reset\n"
        "  (B) Start realtime video stream\n"
        "==================================================================\n"
    )
    print(banner, flush=True)
    if auto:
        print("[s2s] --auto-replay: choosing (B)", flush=True)
        return "B"
    if not sys.stdin.isatty():
        print("[s2s] stdin is not a TTY; defaulting to (B)", flush=True)
        return "B"
    while True:
        try:
            choice = input("Choice [A/B]: ").strip().upper()
        except EOFError:
            return "B"
        if choice in {"A", "B"}:
            return choice


def main() -> None:
    opts = parse_cli()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this pipeline requires the Orin GPU.")
    print(
        f"[s2s] device={torch.cuda.get_device_name(0)} "
        f"sms={torch.cuda.get_device_properties(0).multi_processor_count} "
        f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}",
        flush=True,
    )

    # Both models are loaded and warmed here, before uvicorn binds the port, so
    # a client that reaches /health has a fully warm pipeline behind it.
    print("[s2s] loading both models before the server accepts connections", flush=True)
    pipeline = PipelineApp(opts)
    app = build_app(pipeline)

    config = uvicorn.Config(app, host=opts.host, port=opts.port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    thread.start()

    deadline = time.perf_counter() + 60.0
    while pipeline.loop is None and time.perf_counter() < deadline:
        time.sleep(0.05)
    if pipeline.loop is None:
        raise SystemExit("server failed to start")
    loop = pipeline.loop

    w = pipeline.warmup_report
    if w is not None:
        print(
            f"[s2s] warm baselines: PersonaPlex step p50 {w.personaplex_steady_p50_ms:.0f}ms "
            f"| IMTalker chunk {w.imtalker_steady_ms:.0f}ms "
            f"(fm {w.fm_ms:.0f}ms + renderer {w.renderer_ms:.0f}ms)",
            flush=True,
        )
    print(f"\n[s2s] UI ready:  http://{opts.host}:{opts.port}/\n", flush=True)

    # ---- headless paths --------------------------------------------------
    cache: Optional[SessionCache] = None
    if opts.load_session is not None:
        cache = SessionCache.load(opts.load_session)
        pipeline.cache = cache
        pipeline.state = STATE_AWAITING_CHOICE
        pipeline.metrics.set_phase(STATE_AWAITING_CHOICE)
        print(
            f"[s2s] loaded cached session {opts.load_session.name}: "
            f"{cache.n_steps} steps, {cache.reply_samples / TARGET_SR:.1f}s reply audio",
            flush=True,
        )
    elif opts.input_file is not None:
        cache = run_offline_capture(pipeline, opts.input_file)

    if cache is not None:
        if _cli_prompt_choice(cache, opts.auto_replay) == "A":
            pipeline.discard()
            print("[s2s] discarded. Server still running for interactive use.", flush=True)
        else:
            sid = pipeline.start_stream(loop)
            print(
                f"\n[s2s] realtime stream armed.\n"
                f"      Open http://<jetson-ip>:{opts.port}/ and it will attach to\n"
                f"      session {sid[:12]} automatically.\n",
                flush=True,
            )

    try:
        while thread.is_alive():
            thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[s2s] shutting down", flush=True)
        pipeline.stop_stream()
        server.should_exit = True
        thread.join(timeout=10.0)


if __name__ == "__main__":
    main()
