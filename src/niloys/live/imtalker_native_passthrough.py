"""PersonaPlex + IMTalker AJ server with split audio/video WebSockets.

Raw PersonaPlex audio is never stored in or cleared with the video queue.
Assistant Opus audio stays on /ws/conversation, while JPEG video frames are
sent through /ws/video?session_id=... to avoid websocket head-of-line blocking.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# FORK of imtalker_personaplex_try_vad14_precomputed_closure_8998.py (design_a)
#
# Restores the ORIGINAL PersonaPlex server behaviour. The upstream server
# (speech2avatar/personaplex/moshi/moshi/server.py) has NO barge-in machinery
# at all: strictly one-user-frame-in / one-reply-frame-out at 12.5 Hz, never
# buffers ahead, never flushes, never gates output by RMS. Interruption is
# handled entirely by the model emitting PAD/silence tokens.
#
# That machinery exists in this file only because IMTalker flow matching needed
# 25 Helium steps (2 s) before it could emit anything, forcing the pipeline to
# run ahead of realtime and therefore to discard on interruption. The Design A
# causal student emits every 80 ms, so the constraint is gone.
#
# 2026-10-07: the whole interruption system (mic-overlap candidates, native commit,
# media reset, speculative closure / motion bridges, barge hints) was REMOVED --
# in passthrough it never changed the output. Every PersonaPlex step is published.
#
# NATIVE_PASSTHROUGH disables:
#   1. the queue flush + media-generation bump (_trigger_media_reset)
#   2. the RMS output gate (assistant_active)
#   3. the idle bridge (TRY13_BRIDGE_FRAMES)
#   4. media_reset messages, so the client 2.5 s idle-video fallback never arms
# The production file is NOT modified; this is a copy.
# ---------------------------------------------------------------------------
import os as _os, sys as _sys
_IMT = _os.environ.get("IMTALKER_DIR",
    "/home/ubuntu/project/speech2avatar_imtalker_personaplex_try_vad6_8998/IMTalker")
if _IMT not in _sys.path:
    _sys.path.insert(0, _IMT)
NATIVE_PASSTHROUGH = _os.environ.get("NATIVE_PASSTHROUGH", "1") == "1"
_STAGE_ACC = {k: 0.0 for k in ('encode','lm','decode','hidden','item','codes','pcm','b64','total')}
_STAGE_ACC['n'] = 0

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import json
import os
_CODE_GUARD = os.environ.get("TRY19_CODE_GUARD", "0") == "1"
_CODE_CARD = 2048
import queue

import ws_av_binary_codec as _wsbin
import sys
import threading
import time
import types
import uuid
import wave
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sphn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.original_pod_8998.FM import FMGenerator
from generator.train_lora import apply_lora_to_model
from generator.helium_w2v_frontend_adapter import HeliumToWav2VecFrontendAdapter
from generator.unitalk_wav2vec_adapter import UniTalkLastLayerLiveAdapter
from generator.options.base_options import BaseOptions
from generator.wav2vec2 import Wav2VecModel
if os.environ.get("IMTALKER_CACHED_ENGINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from liveTry_cached import MoshiOnlyEngine
else:
    from liveTry import MoshiOnlyEngine
from renderer.models import IMTRenderer
try:  # optional: only needed with --enable_seedvc, absent on Jetson
    from seedvc_runtime import SeedVCStreamingConverter
except ImportError:
    SeedVCStreamingConverter = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 24_000          # Mimi sample rate (24kHz)
# REPLY_GAIN: linear gain on the reply audio just before opus. --reply_audio_gain
# is inert (parsed, never used), so this env var is the working knob.
# 1.5 ~= +3.5 dB, which is what the 2-layer decoder student costs.
_REPLY_GAIN = float(_os.environ.get("REPLY_GAIN", "1.0") or 1.0)
VIDEO_FPS = 25              # IMTalker frame rate
MIMI_FRAME_SIZE = 1_920     # samples per Mimi frame (80ms @ 24kHz)
MAIN_CODEBOOKS = 8          # codebooks used for Helium input embeddings
PREBUFFER_CHUNKS = 0        # produce this many chunks before sender starts pacing
WAV2VEC_SR = 16_000
TRANSITION_BLEND_FRAMES = 0
TRY13_BRIDGE_FRAMES = 0 if NATIVE_PASSTHROUGH else 50


class PlasticityProjectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_ln = nn.LayerNorm(4096)
        self.net = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_ln(x))


class PlasticityUpsampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(768, 768, kernel_size=4, stride=4)

    def forward(self, low: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.up(low.transpose(1, 2).contiguous())
        if y.shape[-1] != int(target_len):
            y = F.interpolate(y, size=int(target_len), mode="linear", align_corners=False)
        return y.transpose(1, 2).contiguous()


class PlasticityCausalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.15,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(768)
        self.ff = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 768),
        )
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop1(attn)
        h = self.norm2(x)
        x = x + self.drop2(self.ff(h))
        return x


class PlasticityCausalTransformer(nn.Module):
    def __init__(self, max_len: int = 2048) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([PlasticityCausalBlock() for _ in range(8)])
        self.norm = nn.LayerNorm(768)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.causal_mask[: x.shape[1], : x.shape[1]].to(device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class StudioNativeLiveAdapter(nn.Module):
    """Frontend fp32 adapter live wrapper.

    Training contract:
      raw 12.5Hz Helium -> Wav2Vec2 projected frontend [T50, 768]
      live contract:
      projected frontend -> frozen Wav2Vec2 encoder -> final hidden -> IMTalker audio_projection.
    """

    def __init__(self, wav2vec_model_path: str, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = HeliumToWav2VecFrontendAdapter(num_layers=int(num_layers), dropout=float(dropout))
        self.wav2vec = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).eval().float()
        for param in self.wav2vec.parameters():
            param.requires_grad_(False)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.model.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def forward_single(self, source: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = source.unsqueeze(0).contiguous()
        target_len = int(target_len)
        frontend_len = max(1, target_len * 2)
        frontend50 = self.model(src.float(), target_len=frontend_len).float()
        final50 = self.wav2vec.encode_from_projected_frontend(frontend50).last_hidden_state.float()
        final25 = F.interpolate(
            final50.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)[0].float().contiguous()
        return frontend50[0].float().contiguous(), final50[0].float().contiguous(), final25


class MoshiOnlyEngineWithHidden(MoshiOnlyEngine):
    """Moshi reply engine that also returns the main LM hidden for each generated step.

    Layer[-2] is exposed as a native LMGen output so Moshi can keep CUDA graph
    replay enabled. We do not use Python forward hooks in this path.
    """

    def __init__(self, *args, capture_layer: int = -2, **kwargs) -> None:
        self.tf_capture_layer = int(capture_layer)
        super().__init__(*args, **kwargs)
        self._install_graph_hidden_capture()

    def _install_graph_hidden_capture(self) -> None:
        lm_model = self.lm
        lm_gen = self.lm_gen
        if hasattr(lm_gen, "prepare_step_input") and hasattr(lm_gen, "process_transformer_output"):
            @torch.no_grad()
            def personaplex_step_with_hidden(
                self_gen,
                input_tokens: torch.Tensor = None,
                moshi_tokens: torch.Tensor = None,
                text_token: torch.Tensor = None,
                depformer_replace_tokens: torch.Tensor | None = None,
            ):
                prepared = self_gen.prepare_step_input(input_tokens, moshi_tokens, text_token)
                if prepared is None:
                    return None
                input_, provided_, target_, model_input_position, target_position = prepared
                state = self_gen._streaming_state
                transformer_out, text_logits = state.graphed_main(input_)
                output = self_gen.process_transformer_output(
                    transformer_out,
                    text_logits,
                    provided_,
                    target_,
                    model_input_position,
                    target_position,
                )
                return output, transformer_out, transformer_out

            lm_gen._step = types.MethodType(personaplex_step_with_hidden, lm_gen)
            lm_gen.streaming_forever(1)
            self._warmup_runtime()
            print("[liveTryPlasticity] installed PersonaPlex graphed hidden capture", flush=True)
            return

        from moshi.models.lm import scatter_with_mask_
        from moshi.modules.transformer import create_sin_embedding
        from moshi.utils.sampling import sample_token

        capture_layer = int(self.tf_capture_layer) % len(lm_model.transformer.layers)

        old_state = getattr(lm_gen, "_streaming_state", None)
        if old_state is not None:
            with contextlib.suppress(Exception):
                old_state.__exit__(None, None, None)
            with contextlib.suppress(Exception):
                lm_gen._stop_streaming()

        def forward_text_with_layer(self_lm, sequence, sum_condition=None, cross_attention_src=None):
            B, K, S = sequence.shape
            assert K == self_lm.num_codebooks, (K, self_lm.num_codebooks)
            input_sequence = sequence
            input_ = None
            for cb_index in range(self_lm.num_audio_codebooks):
                audio_emb = self_lm.emb[cb_index](input_sequence[:, cb_index + self_lm.audio_offset])
                input_ = audio_emb if input_ is None else input_ + audio_emb
            text_emb = self_lm.text_emb(input_sequence[:, 0])
            input_ = text_emb if input_ is None else input_ + text_emb
            if sum_condition is not None:
                input_ = input_ + sum_condition.to(input_)
            if cross_attention_src is not None:
                cross_attention_src = cross_attention_src.to(input_)

            transformer = self_lm.transformer
            _, T, C = input_.shape
            dtype_input = input_.dtype
            state = transformer._streaming_state
            if state is None:
                offsets = torch.zeros(1, dtype=torch.long, device=input_.device)
            else:
                offsets = state.offsets

            x = input_
            if transformer.positional_embedding in {"sin", "sin_rope"}:
                positions = torch.arange(T, device=x.device).view(1, -1, 1)
                positions = positions + offsets.view(-1, 1, 1)
                pos_emb = create_sin_embedding(positions, C, max_period=transformer.max_period, dtype=x.dtype)
                x = x + transformer.positional_scale * pos_emb

            captured = x
            for idx, layer in enumerate(transformer.layers):
                x = layer(x, cross_attention_src=cross_attention_src)
                if idx == capture_layer:
                    captured = x

            if state is not None:
                state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)

            transformer_out = x.to(dtype_input)
            layer_hidden = captured.to(dtype_input)
            if self_lm.out_norm:
                transformer_out = self_lm.out_norm(transformer_out)
            text_logits = self_lm.text_linear(transformer_out)
            text_logits = text_logits[:, None]
            return transformer_out, text_logits, layer_hidden

        @torch.no_grad()
        def step_with_layer(self_gen, input_tokens: torch.Tensor, depformer_replace_tokens: torch.Tensor | None = None):
            state = self_gen._streaming_state
            if state is None:
                raise RuntimeError("You should wrap those calls with a `with lm_gen.streaming(): ...`.")
            lm_model_local = self_gen.lm_model

            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert B == state.batch_size, f"Got a batch size {B}, expected {state.batch_size}"
            assert S == 1, "Only support being given steps one by one."
            needed_tokens = lm_model_local.num_codebooks - lm_model_local.dep_q - 1
            assert Ki >= needed_tokens, f"We expect {needed_tokens} tokens from the user stream, got {Ki}."
            if Ki > needed_tokens:
                input_tokens = input_tokens[:, :needed_tokens, :]

            CT = state.cache.shape[2]
            delays = self_gen.delays_cuda[lm_model_local.dep_q + 1:]
            write_positions = (state.offsets[:, None, None] + delays[:, None]) % CT
            scatter_with_mask_(state.cache[:, lm_model_local.dep_q + 1:], -1, write_positions, input_tokens, state.exec_mask[:, None, None])

            is_init = state.offsets[:, None, None] <= self_gen.delays_cuda[:, None]
            is_init |= ~state.exec_mask[:, None, None]
            positions = (state.offsets % CT)[:, None, None].expand_as(is_init)
            input_ = state.cache.gather(dim=2, index=positions)
            input_ = torch.where(is_init, state.initial, input_)

            if self_gen.check:
                assert not (input_ == lm_model_local.ungenerated_token_id).any(), (state.offsets, input_)
                assert (input_[:, lm_model_local.audio_offset:] <= lm_model_local.card).all(), input_
                assert (input_[:, :1] <= lm_model_local.text_card).all()

            zero = torch.full((1,), lm_model_local.zero_token_id, dtype=torch.long, device=input_.device)
            if self_gen.cfg_coef != 1.:
                if state.cfg_is_masked_until is not None:
                    limit = self_gen.delays_cuda[:, None] + state.cfg_is_masked_until.view(-1, 1, 1)
                    is_zeroed = state.offsets[:, None, None] <= limit
                    masked = torch.where(is_zeroed & ~is_init, zero, input_)
                    input_ = torch.cat([input_, masked], dim=0)
                else:
                    input_ = input_.repeat(2, 1, 1)
                if self_gen.cfg_is_no_text:
                    input_[B:, :1] = torch.where(~is_init[:, :1], zero, input_[B:, :1])

            transformer_out, text_logits, layer_hidden = state.graphed_main(input_, state.condition_sum, state.condition_cross)
            if self_gen.cfg_coef != 1.:
                logits, logits_null = text_logits.chunk(2)
                if self_gen.cfg_is_no_text:
                    text_logits = logits
                    layer_hidden = layer_hidden[:B]
                else:
                    text_logits = logits_null + (logits - logits_null) * self_gen.cfg_coef
                    layer_hidden = layer_hidden[:B]

            if self_gen.on_text_logits_hook:
                self_gen.on_text_logits_hook(text_logits)
            text_token = sample_token(text_logits.float(), self_gen.use_sampling, self_gen.temp_text, self_gen.top_k_text)
            assert text_token.dim() == 3, text_token.shape
            assert text_token.shape[2] == 1
            assert text_token.shape[1] == 1, "Only one text stream supported."
            text_token = text_token[:, 0, 0]
            if self_gen.on_text_hook is not None:
                self_gen.on_text_hook(text_token)

            if state.graphed_depth is None:
                audio_tokens = None
            else:
                if depformer_replace_tokens is None:
                    audio_tokens = state.graphed_depth(text_token, transformer_out)
                else:
                    assert depformer_replace_tokens.dim() == 3
                    audio_tokens = depformer_replace_tokens.squeeze(-1)
                if self_gen.on_audio_hook is not None:
                    self_gen.on_audio_hook(audio_tokens)

            state.offsets = torch.where(state.exec_mask, state.offsets + 1, state.offsets)
            state.offset_cpu += 1
            positions = (state.offsets % CT)[:, None, None]
            scatter_with_mask_(state.cache[:, :1], -1, positions, text_token[:, None, None], state.exec_mask[:, None, None])
            if audio_tokens is not None:
                audio_tokens = audio_tokens[:, :, None]
                scatter_with_mask_(state.cache[:, 1: lm_model_local.dep_q + 1, :], -1, positions.expand_as(audio_tokens), audio_tokens, state.exec_mask[:, None, None])

            if not self_gen.support_out_of_sync and state.offset_cpu <= self_gen.max_delay:
                return None
            gen_delays_cuda = self_gen.delays_cuda[: lm_model_local.dep_q + 1]
            index = (state.offsets[:, None, None] - self_gen.max_delay + gen_delays_cuda[:, None]) % CT
            out = state.cache.gather(dim=2, index=index)
            mask = (state.offsets <= self_gen.max_delay) | ~state.exec_mask
            out[mask, :, :] = lm_model_local.ungenerated_token_id
            return out, transformer_out, layer_hidden

        lm_model.forward_text = types.MethodType(forward_text_with_layer, lm_model)
        lm_gen._step = types.MethodType(step_with_layer, lm_gen)
        lm_gen.streaming_forever(1)
        self._warmup_runtime()
        print(f"[liveTryPlasticity] installed graphed layer capture layer={self.tf_capture_layer}", flush=True)

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        t_encode0 = time.perf_counter()
        codes = self.mimi.encode(chunk)
        _g_bad_e = None
        if _CODE_GUARD:
            # Out-of-range codes kill the CUDA context (device-side assert in an embedding
            # lookup). Flag on the GPU, clamp, and read the flag after this step's own sync.
            _g_bad_e = ((codes < 0) | (codes >= _CODE_CARD)).any()
            _g_e_min, _g_e_max = codes.min(), codes.max()
            codes = codes.clamp(0, _CODE_CARD - 1)
        t_encode1 = time.perf_counter()
        if self.skip_first:
            self.mimi.reset_streaming()
            self.skip_first = False

        t_lm0 = time.perf_counter()
        lm_out = self.lm_gen._step(codes[:, :, :1])
        t_lm1 = time.perf_counter()

        tokens = None
        helium_hidden = None
        if lm_out is not None:
            if not (isinstance(lm_out, tuple) and len(lm_out) == 3):
                raise RuntimeError(f"Moshi graph layer[-2] contract failure: got {type(lm_out)} len={len(lm_out) if isinstance(lm_out, tuple) else 'n/a'}")
            tokens, _transformer_out, layer_hidden = lm_out
            _tp0 = time.perf_counter()
            helium_hidden = layer_hidden[:1, -1:].detach().float().cpu()
            _t_hidden = 1000.0 * (time.perf_counter() - _tp0)

        token = -1
        token_piece = ""
        decode_ms = 0.0
        reply_codes = None
        if tokens is None:
            reply_pcm = np.zeros(MIMI_FRAME_SIZE, dtype=np.float32)
        else:
            _tp1 = time.perf_counter()
            token = int(tokens[0, 0, 0].detach().item())
            _t_item = 1000.0 * (time.perf_counter() - _tp1)
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
            _tp2 = time.perf_counter()
            reply_codes = tokens[:, 1:].detach().to(device="cpu", dtype=torch.int16)
            _t_codes = 1000.0 * (time.perf_counter() - _tp2)
            t_decode0 = time.perf_counter()
            _g_tk = tokens[:, 1:9]  # AGENT codebooks only; [:, 1:] also decoded the 8 user-stream codebooks into the voice (3.58 -> 2.58 UTMOS)
            _g_bad_d = None
            if _CODE_GUARD:
                _g_bad_d = ((_g_tk < 0) | (_g_tk >= _CODE_CARD)).any()
                _g_d_min, _g_d_max = _g_tk.min(), _g_tk.max()
                _g_tk = _g_tk.clamp(0, _CODE_CARD - 1)
            reply = self.mimi.decode(_g_tk)
            if _CODE_GUARD and (bool(_g_bad_e) or bool(_g_bad_d)):
                print(f"[CODE-GUARD] step={self.step} encoder_codes_bad={bool(_g_bad_e)} "
                      f"(min {int(_g_e_min)} max {int(_g_e_max)}) decode_tokens_bad={bool(_g_bad_d)} "
                      f"(min {int(_g_d_min)} max {int(_g_d_max)}) -> clamped", flush=True)
            _tp3 = time.perf_counter()
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            _t_pcm = 1000.0 * (time.perf_counter() - _tp3)
            decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
            if reply_pcm.shape[0] < MIMI_FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, MIMI_FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > MIMI_FRAME_SIZE:
                reply_pcm = reply_pcm[:MIMI_FRAME_SIZE]

        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        reply_peak = float(np.max(np.abs(reply_pcm))) if reply_pcm.size else 0.0
        input_rms = float(np.sqrt(np.mean(np.square(pcm24, dtype=np.float32))))
        encode_ms = 1000.0 * (t_encode1 - t_encode0)
        lm_ms = 1000.0 * (t_lm1 - t_lm0)
        total_ms = 1000.0 * (time.perf_counter() - t0)

        _tp4 = time.perf_counter()
        reply_i16 = np.clip(reply_pcm, -1.0, 1.0)
        reply_i16 = (reply_i16 * 32767.0).astype(np.int16)
        audio_b64 = base64.b64encode(reply_i16.tobytes()).decode("ascii")
        _t_b64 = 1000.0 * (time.perf_counter() - _tp4)
        try:
            _A = _STAGE_ACC
            _A["n"] += 1
            _A["encode"] += encode_ms; _A["lm"] += lm_ms; _A["decode"] += decode_ms
            _A["hidden"] += locals().get("_t_hidden", 0.0)
            _A["item"] += locals().get("_t_item", 0.0)
            _A["codes"] += locals().get("_t_codes", 0.0)
            _A["pcm"] += locals().get("_t_pcm", 0.0)
            _A["b64"] += _t_b64
            _A["total"] += total_ms
            if _A["n"] % 40 == 0:
                n = _A["n"]
                acc = (_A["encode"]+_A["lm"]+_A["decode"]+_A["hidden"]+_A["item"]
                       +_A["codes"]+_A["pcm"]+_A["b64"]) / n
                print("[STAGE] " + " ".join(
                    f"{k}={_A[k]/n:.1f}" for k in
                    ("encode","lm","decode","hidden","item","codes","pcm","b64","total"))
                    + f" accounted={acc:.1f} unaccounted={_A['total']/n-acc:.1f}",
                    flush=True)
        except Exception as _e:
            print(f"[STAGE] err {_e!r}", flush=True)

        print(
            "[liveTryStudio] moshi "
            f"step={self.step} token={token} piece={token_piece!r} "
            f"in_rms={input_rms:.5f} reply_rms={reply_rms:.5f} peak={reply_peak:.3f} "
            f"hidden={helium_hidden is not None} "
            f"encode={encode_ms:.1f}ms lm={lm_ms:.1f}ms decode={decode_ms:.1f}ms total={total_ms:.1f}ms",
            flush=True,
        )

        return {
            "step": int(self.step),
            "sample_rate": TARGET_SR,
            "reply_i16_b64": audio_b64,
            "reply_rms": reply_rms,
            "reply_peak": reply_peak,
            "input_rms": input_rms,
            "token": token,
            "piece": token_piece,
            "sampled_text": self.sampled_text,
            "audio_text": self.audio_text,
            "encode_ms": encode_ms,
            "lm_ms": lm_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "helium_hidden": helium_hidden,
            "reply_codes": reply_codes,
        }

    @torch.no_grad()
    def process_ready_steps_limited(self, max_steps: int) -> list[dict]:
        """Process a bounded number of Mimi frames.

        The base MoshiOnlyEngine drains the entire input buffer before returning.
        In live mode that is dangerous: mic audio can accumulate while Moshi is
        loading, then avatar frames do not reach the sender until the backlog is
        fully processed. Bounded draining keeps the producer/sender interleaved.
        """
        events: list[dict] = []
        for _ in range(max(1, int(max_steps))):
            if self.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                break
            pcm = self.input_buffer[:MIMI_FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[MIMI_FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(t0: float) -> float:
    return 1000.0 * (time.perf_counter() - t0)


def encode_jpeg_b64(frame_rgb: np.ndarray, quality: int) -> str:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def encode_jpeg_bytes(frame_rgb: np.ndarray, quality: int) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def _pcm_f32_to_i16_b64(pcm: np.ndarray) -> str:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return base64.b64encode((arr * 32767.0).astype(np.int16).tobytes()).decode("ascii")


def _pcm_f32_to_i16_bytes(pcm: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def split_audio_into_frame_slices(pcm: np.ndarray, fps: float) -> list[np.ndarray]:
    frame_samples = int(round(TARGET_SR / float(fps)))
    arr = np.asarray(pcm, dtype=np.float32)
    n_frames = max(0, int(round(arr.shape[0] / frame_samples)))
    if n_frames == 0:
        return []
    total = n_frames * frame_samples
    if arr.shape[0] < total:
        arr = np.pad(arr, (0, total - arr.shape[0]))
    elif arr.shape[0] > total:
        arr = arr[:total]
    return [arr[i * frame_samples:(i + 1) * frame_samples].copy() for i in range(n_frames)]


def load_audio_24k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).float().numpy()


def load_ref_image(path: str | Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# FM + Renderer weight loading (identical to liveTryFM.py)
# ---------------------------------------------------------------------------

def _clean_generator_state(ckpt: dict) -> dict:
    raw = ckpt.get("ema_state_dict") or ckpt.get("state_dict", ckpt.get("model", ckpt))
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    return {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in raw.items()}


def _load_fm(args: argparse.Namespace, device: torch.device) -> FMGenerator:
    t_total = time.perf_counter()
    fm = FMGenerator(args).to(device).eval()
    ckpt = torch.load(args.generator_path, map_location="cpu")
    cleaned = _clean_generator_state(ckpt)
    missing, unexpected = fm.load_state_dict(cleaned, strict=False)
    print(
        f"[liveTryHeliumFM][FM] base loaded missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    lora_path = str(getattr(args, "lora_generator_path", "") or "")
    if lora_path:
        apply_lora_to_model(
            fm,
            rank=int(getattr(args, "lora_rank", 64) or 64),
            alpha=float(getattr(args, "lora_alpha", 128) or 128),
            dropout=float(getattr(args, "lora_dropout", 0.05)),
            include_pose_lora=not bool(getattr(args, "no_lora_pose_projection", False)),
            include_audio_lora=not bool(getattr(args, "no_lora_audio_projection", False)),
            only_pose_lora=bool(getattr(args, "only_lora_pose_projection", False)),
        )
        lora_ckpt = torch.load(lora_path, map_location="cpu")
        lora_cleaned = _clean_generator_state(lora_ckpt)
        missing_lora, unexpected_lora = fm.load_state_dict(lora_cleaned, strict=False)
        lora_keys = sum(1 for key in lora_cleaned if "lora_" in key)
        print(
            f"[liveTryHeliumFM][FM] lora loaded path={lora_path} "
            f"lora_keys={lora_keys} missing={len(missing_lora)} unexpected={len(unexpected_lora)}",
            flush=True,
        )
    fm.to(device).eval()
    _sync_cuda()
    print(f"[liveTryHeliumFM][FM] loaded in {_ms(t_total):.0f}ms", flush=True)
    return fm


def _load_renderer(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> IMTRenderer:
    t_total = time.perf_counter()
    renderer = IMTRenderer(args).to(device).eval()
    ckpt = torch.load(args.renderer_path, map_location="cpu")
    raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {k.replace("gen.", "", 1).replace("model.", "", 1): v for k, v in raw.items()}
    missing, unexpected = renderer.load_state_dict(cleaned, strict=False)
    renderer = renderer.to(dtype=dtype)
    _sync_cuda()
    if getattr(args, "compile_renderer", False):
        @torch.no_grad()
        def _fused_render(motion_latent, g_r, m_r, f_r):
            ta_c = renderer.adapt(motion_latent, g_r)
            m_c = renderer.latent_token_decoder(ta_c)
            frames = renderer.decode(m_c, m_r, f_r)
            return frames
        @torch.no_grad()
        def _fused_render_blink(motion_latent, g_r, m_r, f_r, blink_maps, eye_masks):
            ta_c = renderer.adapt(motion_latent, g_r)
            current_maps = renderer.latent_token_decoder(ta_c)
            composited_maps = tuple(
                blink * mask + current * (1.0 - mask)
                for current, blink, mask in zip(current_maps, blink_maps, eye_masks)
            )
            return renderer.decode(composited_maps, m_r, f_r)

        renderer._fused_render = torch.compile(_fused_render)
        renderer._fused_render_blink = torch.compile(_fused_render_blink)
        # Keep independently compiled batch-1 graphs for the first
        # interruption frame. The normal graphs stay specialized to the
        # throughput-oriented render_sub_batch shape.
        renderer._fused_render_first = torch.compile(_fused_render)
        renderer._fused_render_blink_first = torch.compile(_fused_render_blink)
    print(
        f"[liveTryHeliumFM][renderer] loaded in {_ms(t_total):.0f}ms "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return renderer


# ---------------------------------------------------------------------------
# Helium extractor (chunk-local, batch LM)
# ---------------------------------------------------------------------------

class HeliumExtractor:
    """Prefix-growing raw Helium + global interpolation.

    This matches the best Stage 3 diagnostic:
      audio prefix -> raw Helium -> append only new raw steps
      -> one global interpolation over accumulated raw Helium
      -> emit last target_frames
    """

    def __init__(
        self,
        helium_mimi: "MimiModel",
        helium_lm: "LMModel",
        device: torch.device,
    ) -> None:
        self.helium_mimi = helium_mimi
        self.helium_lm = helium_lm
        self.device = device
        self._lock = threading.Lock()
        self._prefix_pcm = np.empty(0, dtype=np.float32)
        self._raw_parts: list[torch.Tensor] = []
        self._prev_raw_len = 0
        self._emitted_frames = 0

    def reset(self) -> None:
        with self._lock:
            self._prefix_pcm = np.empty(0, dtype=np.float32)
            self._raw_parts = []
            self._prev_raw_len = 0
            self._emitted_frames = 0

    def _extract_raw(self, pcm_np: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(np.asarray(pcm_np, dtype=np.float32)).to(self.device, dtype=torch.float32)[None, None]

        codes = self.helium_mimi.encode(wav)
        codes = codes[:, :MAIN_CODEBOOKS, :].detach()
        batch_size, n_q, total_steps = codes.shape

        dtype = next(self.helium_lm.parameters()).dtype
        input_emb = torch.zeros(
            batch_size, total_steps, self.helium_lm.dim, device=self.device, dtype=dtype
        )
        for q in range(n_q):
            input_emb = input_emb + self.helium_lm.emb[q](codes[:, q].long())

        padding_ids = torch.full(
            (batch_size, total_steps),
            self.helium_lm.existing_text_padding_id,
            dtype=torch.long,
            device=self.device,
        )
        input_emb = input_emb + self.helium_lm.text_emb(padding_ids)

        if getattr(self.helium_lm.transformer, "_streaming_state", None) is not None:
            raise RuntimeError("helium_lm must stay in batch mode (non-streaming)")

        captured: list[torch.Tensor] = []

        def _hook(_mod, _inp, out):
            captured.append(out.detach())

        handle = self.helium_lm.transformer.layers[-2].register_forward_hook(_hook)
        try:
            self.helium_lm.transformer(input_emb)
        finally:
            if reply_engine is not None:
                pipeline_stop_event.set()
                session_started.clear()
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(f"Helium hook captured {len(captured)} tensors; expected 1")
        return captured[0].squeeze(0).float().contiguous()  # [T_raw, 4096]

    @torch.no_grad()
    def extract_raw_chunk(self, pcm_np: np.ndarray) -> torch.Tensor:
        """Return only the new raw 12.5Hz Helium steps for one new audio chunk."""
        pcm = np.asarray(pcm_np, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_raw_chunk expects non-empty 1D PCM")

        with self._lock:
            self._prefix_pcm = np.concatenate([self._prefix_pcm, pcm], axis=0)
            raw_prefix = self._extract_raw(self._prefix_pcm)
            new_raw = raw_prefix[self._prev_raw_len:]
            if int(new_raw.shape[0]) == 0 and int(raw_prefix.shape[0]) > 0:
                new_raw = raw_prefix[-1:]
            self._raw_parts.append(new_raw.cpu())
            self._prev_raw_len = int(raw_prefix.shape[0])
            return new_raw.contiguous()

    @torch.no_grad()
    def extract_exact_chunk_from_prefix(
        self,
        pcm_prefix: np.ndarray,
        chunk_start_frame: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Return the raw 12.5Hz Helium slice for a video-frame window.

        This path is used by file-mode lookahead. It must return raw Helium
        steps so the studio adapter remains the only temporal upsampler.
        """
        pcm = np.asarray(pcm_prefix, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_exact_chunk_from_prefix expects non-empty 1D PCM")
        with self._lock:
            raw_prefix = self._extract_raw(pcm)
        start_frame = int(chunk_start_frame)
        end_frame = start_frame + int(target_frames)
        start_raw = int(round(start_frame * 0.5))
        end_raw = int(round(end_frame * 0.5))
        start_raw = max(0, min(start_raw, int(raw_prefix.shape[0])))
        end_raw = max(start_raw + 1, min(end_raw, int(raw_prefix.shape[0])))
        if end_raw > int(raw_prefix.shape[0]):
            raise RuntimeError(
                f"Requested raw slice [{start_raw}, {end_raw}) exceeds prefix Helium length {raw_prefix.shape[0]}"
            )
        return raw_prefix[start_raw:end_raw].contiguous()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class LiveHeliumFMEngine:
    """Helium extraction + FM + renderer, session-stateful."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.args = args
        self.device = torch.device(args.device)
        renderer_precision = str(
            getattr(args, "renderer_precision", "fp32")
        ).lower()
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[renderer_precision]
        self.fps = float(args.fps)
        self.audio_chunk_sec = float(getattr(args, "audio_chunk_sec", 0.96))
        self.audio_chunk_samples = int(round(self.audio_chunk_sec * TARGET_SR))
        self.fm_chunk_frames = max(1, int(getattr(args, "fm_chunk_frames", 24)))
        self.live_sliding_window = bool(getattr(args, "enable_live_sliding_window", False))
        self.slide_past_frames = max(0, int(getattr(args, "slide_past_frames", 10)))
        self.slide_future_frames = max(0, int(getattr(args, "slide_future_frames", 3)))
        self.render_sub_batch = max(1, int(args.render_sub_batch))
        self.jpeg_quality = int(args.jpeg_quality)
        trained_window = int(round(float(args.wav2vec_sec) * self.fps))
        if self.fm_chunk_frames != trained_window:
            print(
                f"[liveTryHeliumFM] WARNING fm_chunk_frames={self.fm_chunk_frames} "
                f"but wav2vec_sec*fps={trained_window}",
                flush=True,
            )
        if self.live_sliding_window:
            print(
                f"[liveTryHeliumFM][typeAC] IMTalker lookahead enabled "
                f"past={self.slide_past_frames}f current={self.fm_chunk_frames}f "
                f"future={self.slide_future_frames}f",
                flush=True,
            )

        t_total = time.perf_counter()

        # FM + renderer
        # The motion student replaces _sample_motion_from_helium, the ONLY consumer
        # of self.fm (lines ~1288-1330). In student mode the FM generator loads
        # 1.3 GB in ~1000 ms and never runs a forward pass.
        if os.environ.get("TRY19_NO_FM", "1") == "1":
            # The student replaces FM *sampling*, but the speculative-rollout /
            # barge-in path still reads fm.num_prev_frames (lines ~3299/3512/3664).
            # That is a scalar from args (FM.py:73 -> opt.num_prev_frames, default
            # 10), not a weight. Expose exactly it; ANY other attribute raises
            # loudly -- a bare None here silently killed the gpu-producer thread
            # and froze video on one frame.
            class _FMStub:
                def __init__(self, n):
                    self.num_prev_frames = int(n)

                def __getattr__(self, name):
                    raise AttributeError(
                        f"FM is skipped (TRY19_NO_FM=1) but fm.{name} was accessed. "
                        "Set TRY19_NO_FM=0 to load the real generator.")

            self.fm = _FMStub(getattr(args, "num_prev_frames", 10))
            print(f"[liveTryHeliumFM] FM generator SKIPPED "
                  f"(stub: num_prev_frames={self.fm.num_prev_frames})", flush=True)
        else:
            self.fm = _load_fm(args, self.device)
        # try19.patch_render swaps _render_motion for the distilled single-identity
        # decoder BEFORE live.main() constructs this engine. That decoder takes
        # motion only -- D(mo, phase) -- so the full renderer's reference features
        # (f_r/g_r/ref_x/m_r) have no consumer left: every reader of them lives in
        # _sample_motion_from_helium or _render_motion, both replaced.
        self._no_full_renderer = os.environ.get("TRY19_NO_FULL_RENDERER", "1") == "1"
        if self._no_full_renderer:
            self.renderer = None
            print("[liveTryHeliumFM] full renderer SKIPPED (distilled decoder in use)",
                  flush=True)
        else:
            self.renderer = _load_renderer(args, self.device, self.dtype)

        # TRY19_NO_FRONTEND: the motion student replaces _sample_motion_from_helium,
        # which is the ONLY consumer of studio_adapter / wav2vec. In student mode these
        # load ~716 MB and never run a forward pass. Left as a switch, not a deletion,
        # so the legacy (non-student) path still works.
        if os.environ.get("TRY19_NO_FRONTEND", "1") == "1":
            self.studio_adapter = None
            self.wav2vec_feature_extractor = None
            self.wav2vec_model = None
            print("[liveTryHeliumFM] frontend SKIPPED (TRY19_NO_FRONTEND=1): "
                  "adapter + wav2vec NOT loaded", flush=True)
        else:
            # Adapter: either the six-layer projected-frontend model or UniTalk's
            # 12-layer model with only its final layer passed to IMTalker.
            t_adapter = time.perf_counter()
            if args.adapter_type == "unitalk_last_layer":
                self.studio_adapter = UniTalkLastLayerLiveAdapter(
                    args.wav2vec_model_path,
                    args.adapter_dropout,
                ).to(self.device).float().eval()
            else:
                self.studio_adapter = StudioNativeLiveAdapter(
                    args.wav2vec_model_path,
                    args.adapter_num_layers,
                    args.adapter_dropout,
                ).to(self.device).float().eval()
            payload = torch.load(args.adapter_path, map_location="cpu")
            if isinstance(payload, dict) and args.adapter_type == "frontend":
                saved_args = payload.get("args", {})
                if saved_args and int(saved_args.get("num_layers", args.adapter_num_layers)) != int(args.adapter_num_layers):
                    print(
                        f"[liveTryHeliumFrontendFM] WARNING checkpoint num_layers={saved_args.get('num_layers')} "
                        f"but CLI adapter_num_layers={args.adapter_num_layers}",
                        flush=True,
                    )
                state = payload.get("adapter", payload.get("model", payload))
            else:
                state = payload
            self.studio_adapter.load_state_dict(state, strict=True)
            _sync_cuda()
            print(
                f"[liveTryHeliumFrontendFM][adapter] type={args.adapter_type} loaded in {_ms(t_adapter):.0f}ms "
                f"path={args.adapter_path}",
                flush=True,
            )

            # Raw HF Wav2Vec2 target path used during Helium adapter training.
            t_w2v = time.perf_counter()
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                args.wav2vec_model_path,
                local_files_only=True,
            )
            self.wav2vec_model = self.studio_adapter.wav2vec
            _sync_cuda()
            print(
                f"[liveTryHeliumStudioFM][wav2vec] loaded in {_ms(t_w2v):.0f}ms "
                f"path={args.wav2vec_model_path}",
                flush=True,
            )

        # Reference image: pre-compute identity + motion-ref features once
        if self._no_full_renderer:
            self.f_r = self.g_r = self.ref_x = self.m_r = None
        else:
            ref_tensor = load_ref_image(args.ref_path, self.device, self.dtype)
            with torch.no_grad():
                self.f_r, self.g_r = self.renderer.dense_feature_encoder(ref_tensor)
                self.ref_x = self.renderer.latent_token_encoder(ref_tensor).to(dtype=torch.float32)
                ta_r = self.renderer.adapt(self.ref_x.to(dtype=self.dtype), self.g_r)
                self.m_r = self.renderer.latent_token_decoder(ta_r)
        _sync_cuda()
        self.eye_blink_enabled = bool(getattr(args, "enable_eye_blink_composite", False))
        self._blink_maps: tuple[torch.Tensor, ...] | None = None
        self._eye_masks: tuple[torch.Tensor, ...] | None = None
        self._render_frame_cursor: int = 0
        if self.eye_blink_enabled:
            self._init_eye_blink_composite()

        # Moshi models for Helium extraction
        self._init_moshi(args)

        # Optional local audio file (simulate-live mode)
        self.audio_pcm: np.ndarray | None = None
        if getattr(args, "audio_path", "") and Path(args.audio_path).is_file():
            self.audio_pcm = load_audio_24k(args.audio_path)
            print(
                f"[liveTryHeliumFM] audio_path loaded: {self.audio_pcm.shape[0]/TARGET_SR:.2f}s "
                f"chunk={self.audio_chunk_sec:.3f}s/{self.audio_chunk_samples} samples "
                f"fm_chunk={self.fm_chunk_frames}f",
                flush=True,
            )

        # Shared noise tensor (pre-generated, indexed by absolute frame position)
        self.noise_buf: torch.Tensor | None = None
        if getattr(args, "shared_noise", False):
            max_frames = int(getattr(args, "noise_max_frames", 5000))
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(getattr(args, "noise_seed", 1234)))
            self.noise_buf = torch.randn(
                1, max_frames, int(args.dim_w), device=self.device, generator=gen
            )
            print(f"[liveTryHeliumFM] shared noise buf: {tuple(self.noise_buf.shape)}", flush=True)

        # Per-session state (reset on each new client)
        self.stream_state: dict | None = None
        self.abs_frame: int = 0
        self.helium_context_tail: torch.Tensor | None = None
        self.helium_deque_size = max(
            1, int(getattr(args, "helium_deque_size", 100))
        )
        self.helium_deque: torch.Tensor | None = None
        self.helium_deque_filled: int = 0
        self.silence_helium_seed: torch.Tensor | None = None
        silence_helium_path = str(
            getattr(args, "silence_helium_path", "") or ""
        )
        if silence_helium_path:
            payload = torch.load(silence_helium_path, map_location="cpu")
            seed = (
                payload.get("silence_helium_mean")
                if isinstance(payload, dict)
                else payload
            )
            if not isinstance(seed, torch.Tensor) or seed.numel() != 4096:
                raise RuntimeError(
                    f"Invalid silence Helium seed: {silence_helium_path}"
                )
            self.silence_helium_seed = seed.reshape(1, 4096).to(
                device=self.device, dtype=torch.float32
            )
            print(
                f"[liveTryHeliumFM] silence Helium seed loaded: "
                f"{silence_helium_path}",
                flush=True,
            )
        self._pcm_accum: np.ndarray = np.empty(0, dtype=np.float32)
        self.dump_motion = bool(getattr(args, "dump_motion", False))
        self.dump_dir = Path(getattr(args, "dump_dir", ROOT / "live_try_dumps"))
        self._session_motion_parts: list[torch.Tensor] = []
        self._session_helium_parts: list[torch.Tensor] = []
        self._session_adapter_50_parts: list[torch.Tensor] = []
        self._session_adapter_25_parts: list[torch.Tensor] = []
        self._session_projected_audio_parts: list[torch.Tensor] = []
        self._session_audio_parts: list[np.ndarray] = []
        self._session_live_token_parts: list[torch.Tensor] = []
        self._session_chunk_rows: list[dict] = []
        self._session_reply_events: list[dict] = []
        self._session_started_wall: float = time.time()

        # JPEG encoding thread pool (CPU-only work, parallelizable)
        self._jpeg_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="jpeg"
        )

        # Warmup
        self._warmup()

        print(
            f"[liveTryHeliumFM] ready — total startup {_ms(t_total):.0f}ms "
            f"fm_chunk={self.fm_chunk_frames} render_sub={self.render_sub_batch} "
            f"dtype={self.dtype}",
            flush=True,
        )

    def _init_moshi(self, args: argparse.Namespace) -> None:
        if bool(getattr(args, "direct_reply_hidden", False)) and bool(getattr(args, "enable_moshi_reply", False)):
            self.extractor = None
            print("[liveTryHeliumStudioFM] using direct Moshi reply hidden; batch Helium extractor skipped", flush=True)
            return

        from generate_helium import load_mimi_and_lm

        t0 = time.perf_counter()
        helium_mimi, helium_lm, _ = load_mimi_and_lm(args)
        helium_mimi.eval()
        helium_lm.eval()
        print(
            f"[liveTryHeliumFM] Moshi loaded in {_ms(t0):.0f}ms "
            f"dtype={next(helium_lm.parameters()).dtype}",
            flush=True,
        )

        self.extractor = HeliumExtractor(helium_mimi, helium_lm, self.device)

    def reset_session(self) -> None:
        """Call when a new WebSocket client connects or sends 'start'."""
        self.stream_state = None
        self.abs_frame = 0
        self._render_frame_cursor = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0
        self._pcm_accum = np.empty(0, dtype=np.float32)
        if self.extractor is not None:
            self.extractor.reset()
        self._session_motion_parts = []
        self._session_helium_parts = []
        self._session_adapter_50_parts = []
        self._session_adapter_25_parts = []
        self._session_projected_audio_parts = []
        self._session_audio_parts = []
        self._session_live_token_parts = []
        self._session_chunk_rows = []
        self._session_reply_events = []
        self._session_started_wall = time.time()

    @torch.no_grad()
    def _warmup(self) -> None:
        dummy_pcm = np.zeros(self.audio_chunk_samples, dtype=np.float32)

        t0 = time.perf_counter()
        if self.extractor is None:
            raw_steps = max(1, int(round(self.fm_chunk_frames * 12.5 / float(self.fps))))
            if self.silence_helium_seed is not None:
                dummy_helium = self.silence_helium_seed.expand(
                    raw_steps, -1
                ).contiguous()
            else:
                dummy_helium = torch.zeros(
                    raw_steps, 4096, device=self.device, dtype=torch.float32
                )
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium=skipped direct_hidden raw_steps={raw_steps}", flush=True)
        else:
            dummy_helium = self.extractor.extract_raw_chunk(dummy_pcm)
            _sync_cuda()
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium={_ms(t0):.0f}ms", flush=True)
            self.extractor.reset()

        t0 = time.perf_counter()
        motion, _info = self._sample_motion_from_helium(dummy_helium, self.fm_chunk_frames)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] fm={_ms(t0):.0f}ms motion={tuple(motion.shape)}", flush=True)
        self.stream_state = None
        self.abs_frame = 0
        self._render_frame_cursor = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0

        t0 = time.perf_counter()
        dummy_motion = torch.zeros(self.render_sub_batch, 32, device=self.device, dtype=self.dtype)
        _frames, _timings = self._render_motion(dummy_motion)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] renderer={_ms(t0):.0f}ms", flush=True)
        self._render_frame_cursor = 0

        t0 = time.perf_counter()
        _first_frames, _first_timings = self._render_motion_first_frame(
            dummy_motion[:1]
        )
        _sync_cuda()
        print(
            f"[TRY13][warmup] renderer_batch1={_ms(t0):.0f}ms",
            flush=True,
        )
        self._render_frame_cursor = 0

        # Warmup JPEG pool
        t0 = time.perf_counter()
        dummy_np = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = encode_jpeg_b64(dummy_np, self.jpeg_quality)
        print(f"[liveTryHeliumFM][warmup] jpeg={_ms(t0):.0f}ms", flush=True)

        self.stream_state = None

    def feed_pcm(self, pcm_s16le_bytes: bytes) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.frombuffer(pcm_s16le_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.feed_pcm_f32(pcm)

    def feed_pcm_f32(self, pcm_f32: np.ndarray) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.asarray(pcm_f32, dtype=np.float32)
        self._pcm_accum = np.concatenate([self._pcm_accum, pcm])
        if self._pcm_accum.shape[0] < self.audio_chunk_samples:
            return None
        chunk = self._pcm_accum[:self.audio_chunk_samples].copy()
        self._pcm_accum = self._pcm_accum[self.audio_chunk_samples:]
        motion, info = self._process_pcm_chunk(chunk, self.fm_chunk_frames)
        self._record_session_chunk(chunk, motion, info)
        return motion, info, chunk

    @torch.no_grad()
    def _process_pcm_chunk(self, pcm_chunk: np.ndarray, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        target_frames = max(1, min(int(target_frames), self.fm_chunk_frames))
        if self.extractor is None:
            raise RuntimeError("direct_reply_hidden mode cannot process raw browser audio directly")

        t0 = time.perf_counter()
        helium = self.extractor.extract_raw_chunk(pcm_chunk)
        timings["helium_ms"] = _ms(t0)
        motion, fm_info = self._sample_motion_from_helium(helium, target_frames)
        timings.update(fm_info)
        return motion, timings

    @torch.no_grad()
    def _sample_motion_from_helium(self, helium: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        t_adapter = time.perf_counter()
        helium = helium.to(self.device, dtype=torch.float32).contiguous()
        target_frames = int(target_frames)
        current_steps = int(helium.shape[0])
        deque_size = int(getattr(self, "helium_deque_size", 100))
        if self.helium_deque is None:
            if self.silence_helium_seed is not None:
                self.helium_deque = self.silence_helium_seed.expand(
                    deque_size, -1
                ).clone()
            else:
                self.helium_deque = torch.zeros(
                    deque_size,
                    helium.shape[1],
                    device=self.device,
                    dtype=torch.float32,
                )
            self.helium_deque_filled = 0
        if current_steps >= deque_size:
            self.helium_deque = helium[-deque_size:].detach().clone()
            self.helium_deque_filled = deque_size
        else:
            self.helium_deque = torch.cat([self.helium_deque[current_steps:], helium], dim=0).contiguous()
            self.helium_deque_filled = min(deque_size, int(self.helium_deque_filled) + current_steps)

        adapter_window_mode = str(
            getattr(self.args, "adapter_window_mode", "tail")
        )
        if adapter_window_mode == "lookahead":
            # Match training: process the full 8-second window at 50 Hz,
            # emit .96 seconds, and retain .48 seconds as future context.
            future_steps = int(getattr(self.args, "adapter_future_steps", 6))
            target_len_50_full = deque_size * 4
            _baseline, _cnn, feat_50_full = self.studio_adapter.forward_single(
                self.helium_deque, target_len_50_full
            )
            if self.live_sliding_window:
                # Type AC: keep the Type A Helium deque + adapter path, but
                # give IMTalker/FM a small [past + current + future] feature
                # window and emit only current frames. This places lookahead in
                # IMTalker instead of discarding future context inside adapter.
                past_25 = int(self.slide_past_frames)
                future_25 = int(self.slide_future_frames)
                current_25 = int(target_frames)
                past_50 = past_25 * 2
                future_50 = future_25 * 2
                current_50 = current_25 * 2
                full_len_50 = int(feat_50_full.shape[0])
                current_end_50 = full_len_50 - future_50
                current_start_50 = current_end_50 - current_50
                window_start_50 = current_start_50 - past_50
                window_end_50 = current_end_50 + future_50
                if window_start_50 < 0 or current_start_50 < 0 or window_end_50 > full_len_50:
                    raise RuntimeError(
                        "Sliding adapter window is out of range: "
                        f"full={full_len_50} start={window_start_50} "
                        f"current_start={current_start_50} end={window_end_50}"
                    )
                feat_50 = feat_50_full[window_start_50:window_end_50].contiguous()
                feat_25 = F.interpolate(
                    feat_50.T.unsqueeze(0),
                    size=past_25 + current_25 + future_25,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T.contiguous()
            else:
                emitted_frames_50 = max(1, current_steps * 4)
                future_frames_50 = max(0, future_steps * 4)
                segment_end = int(feat_50_full.shape[0]) - future_frames_50
                segment_start = segment_end - emitted_frames_50
                if segment_start < 0:
                    raise RuntimeError(
                        "Look-ahead adapter output is shorter than its emit/future region"
                    )
                feat_50 = feat_50_full[segment_start:segment_end].contiguous()
                feat_25 = F.interpolate(
                    feat_50.T.unsqueeze(0),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T.contiguous()
        else:
            # Legacy behavior: predict at 25 Hz and emit the newest tail.
            target_len_25_full = deque_size * 2
            _baseline, _cnn, feat_25_full = self.studio_adapter.forward_single(
                self.helium_deque, target_len_25_full
            )
            fresh_frames = max(1, current_steps * 2)
            if int(feat_25_full.shape[0]) < fresh_frames:
                raise RuntimeError(
                    f"Deque adapter output too short: got "
                    f"{feat_25_full.shape[0]}, need {fresh_frames}"
                )
            feat_25 = feat_25_full[-fresh_frames:].contiguous()
            feat_50 = feat_25
        if (not self.live_sliding_window) and int(feat_25.shape[0]) != target_frames:
            feat_25 = F.interpolate(
                feat_25.T.unsqueeze(0),
                size=target_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(0).T.contiguous()
        projected_a = self.fm._project_audio(feat_25.unsqueeze(0).float())
        timings["adapter_ms"] = _ms(t_adapter)
        timings["helium_ms"] = timings["adapter_ms"]
        timings["helium_deque_filled"] = int(self.helium_deque_filled)

        data: dict = {"a_feat": feat_25.unsqueeze(0).float(), "ref_x": self.ref_x}
        if self.noise_buf is not None:
            if self.live_sliding_window:
                noise_start = max(0, self.abs_frame - int(self.slide_past_frames))
                noise_end = noise_start + int(feat_25.shape[0])
            else:
                noise_start = self.abs_frame
                noise_end = self.abs_frame + target_frames
            data["noise_init"] = self.noise_buf[:, noise_start:noise_end]
        t_fm = time.perf_counter()
        if self.live_sliding_window:
            motion = self.fm.sample(
                data,
                a_cfg_scale=float(self.args.a_cfg_scale),
                nfe=int(self.args.nfe),
            )
        else:
            motion, self.stream_state = self.fm.sample(
                data,
                a_cfg_scale=float(self.args.a_cfg_scale),
                nfe=int(self.args.nfe),
                stream_state=self.stream_state,
                return_state=True,
            )
        timings["fm_ms"] = _ms(t_fm)

        motion = motion.squeeze(0).detach()
        if self.live_sliding_window:
            start = int(self.slide_past_frames)
            motion = motion[start:start + target_frames]
        else:
            motion = motion[:target_frames]
        ref_blend = float(getattr(self.args, "motion_ref_blend", 0.0) or 0.0)
        if ref_blend > 0.0:
            ref_motion = self.ref_x.detach().float()
            if ref_motion.ndim == 3:
                ref_motion = ref_motion[0]
            if ref_motion.ndim == 1:
                ref_motion = ref_motion.unsqueeze(0)
            if int(ref_motion.shape[0]) == 1:
                ref_motion = ref_motion.expand(int(motion.shape[0]), -1)
            elif int(ref_motion.shape[0]) != int(motion.shape[0]):
                ref_motion = F.interpolate(
                    ref_motion.T.unsqueeze(0),
                    size=int(motion.shape[0]),
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T
            ref_motion = ref_motion.to(device=motion.device, dtype=motion.dtype)
            blend = max(0.0, min(1.0, ref_blend))
            motion = motion.mul(1.0 - blend).add(ref_motion, alpha=blend)
        timings["helium_feat"] = helium.detach().cpu()
        timings["adapter_feat_50"] = feat_50.detach().cpu()
        timings["adapter_feat_25"] = feat_25.detach().cpu()
        timings["projected_audio"] = projected_a.squeeze(0).detach().cpu()
        timings["frames"] = int(motion.shape[0])
        timings["abs_start"] = self.abs_frame
        self.abs_frame += timings["frames"]
        return motion, timings

    def _record_session_chunk(self, pcm_chunk: np.ndarray, motion: torch.Tensor, info: dict) -> None:
        self._session_audio_parts.append(np.asarray(pcm_chunk, dtype=np.float32).copy())
        self._session_motion_parts.append(motion.detach().float().cpu().clone())
        helium_feat = info.get("helium_feat")
        if isinstance(helium_feat, torch.Tensor):
            self._session_helium_parts.append(helium_feat.float().cpu().clone())
        adapter_feat_50 = info.get("adapter_feat_50")
        if isinstance(adapter_feat_50, torch.Tensor):
            self._session_adapter_50_parts.append(adapter_feat_50.float().cpu().clone())
        adapter_feat_25 = info.get("adapter_feat_25")
        if isinstance(adapter_feat_25, torch.Tensor):
            self._session_adapter_25_parts.append(adapter_feat_25.float().cpu().clone())
        projected_audio = info.get("projected_audio")
        if isinstance(projected_audio, torch.Tensor):
            self._session_projected_audio_parts.append(projected_audio.float().cpu().clone())
        self._session_chunk_rows.append({
            "chunk": len(self._session_chunk_rows) + 1,
            "abs_start": int(info.get("abs_start", 0)),
            "frames": int(info.get("frames", int(motion.shape[0]))),
            "samples": int(len(pcm_chunk)),
            "helium_ms": float(info.get("helium_ms", 0.0)),
            "fm_ms": float(info.get("fm_ms", 0.0)),
        })

    @torch.no_grad()
    def _extract_wav2vec_raw_50hz(self, audio_24k: np.ndarray) -> torch.Tensor:
        arr = np.asarray(audio_24k, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return torch.empty((0, 768), dtype=torch.float32)
        wav = torch.from_numpy(arr).view(1, -1)
        wav16 = torchaudio.functional.resample(wav, TARGET_SR, WAV2VEC_SR).squeeze(0).contiguous().numpy()
        inputs = self.wav2vec_feature_extractor(
            wav16,
            sampling_rate=WAV2VEC_SR,
            return_tensors="pt",
            padding=True,
        )
        kwargs = {
            "input_values": inputs.input_values.to(self.device),
        }
        if getattr(inputs, "attention_mask", None) is not None:
            kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        frontend = self.wav2vec_model.extract_projected_frontend(**kwargs)
        feat = self.wav2vec_model.encode_from_projected_frontend(
            frontend
        ).last_hidden_state.detach().float().cpu()[0].contiguous()
        return feat

    def dump_last_session(self, *, source: str = "") -> Optional[Path]:
        if not self.dump_motion or not self._session_motion_parts:
            return None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self.dump_dir / "last_session"
        session_dir.mkdir(parents=True, exist_ok=True)

        motion = torch.cat(self._session_motion_parts, dim=0).contiguous()
        audio = np.concatenate(self._session_audio_parts, axis=0) if self._session_audio_parts else np.empty(0, dtype=np.float32)

        motion_path = session_dir / "full_motion.pt"
        helium_path = session_dir / "full_helium_raw.pt"
        adapter_50_path = session_dir / "full_adapter_w2v_50hz.pt"
        adapter_25_path = session_dir / "full_adapter_w2v_25fps.pt"
        wav2vec_50_path = session_dir / "full_wav2vec_50hz.pt"
        projected_audio_path = session_dir / "full_projected_audio_32.pt"
        audio_path = session_dir / "full_moshi_reply_24k.wav"
        live_tokens_path = session_dir / "live_mimi_tokens.pt"
        reply_events_path = session_dir / "reply_events.jsonl"
        reply_text_path = session_dir / "reply_text.txt"
        meta_path = session_dir / "meta.json"
        helium = None
        adapter_50 = None
        adapter_25 = None
        wav2vec_50 = None
        projected_audio = None
        live_tokens = None
        if self._session_helium_parts:
            helium = torch.cat(self._session_helium_parts, dim=0).contiguous()
        if self._session_adapter_50_parts:
            adapter_50 = torch.cat(self._session_adapter_50_parts, dim=0).contiguous()
        if self._session_adapter_25_parts:
            adapter_25 = torch.cat(self._session_adapter_25_parts, dim=0).contiguous()
        if self._session_projected_audio_parts:
            projected_audio = torch.cat(self._session_projected_audio_parts, dim=0).contiguous()
        if self._session_live_token_parts:
            live_tokens = torch.cat(self._session_live_token_parts, dim=2).contiguous()
        if audio.size > 0:
            wav2vec_50 = self._extract_wav2vec_raw_50hz(audio)

        torch.save({
            "motion": motion,
            "chunks": self._session_chunk_rows,
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 768)),
            "audio_adapter_dim": int(getattr(self.args, "audio_adapter_dim", 512)),
            "wav2vec_sec": float(self.args.wav2vec_sec),
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
            "source": source,
        }, motion_path)
        if helium is not None:
            torch.save({
                "helium": helium,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "audio_chunk_sec": float(self.audio_chunk_sec),
                "fm_chunk_frames": int(self.fm_chunk_frames),
                "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 4096)),
                "source": source,
            }, helium_path)
        if adapter_50 is not None:
            torch.save({
                "adapter_feat_50": adapter_50,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, adapter_50_path)
        if adapter_25 is not None:
            torch.save({
                "adapter_feat_25": adapter_25,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, adapter_25_path)
        if wav2vec_50 is not None:
            torch.save({
                "wav2vec_50hz": wav2vec_50,
                "chunks": self._session_chunk_rows,
                "sample_rate": int(WAV2VEC_SR),
                "source": source,
            }, wav2vec_50_path)
        if projected_audio is not None:
            torch.save({
                "projected_audio": projected_audio,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, projected_audio_path)
        if live_tokens is not None:
            torch.save({
                "live_mimi_tokens": live_tokens,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, live_tokens_path)
        if audio.size > 0:
            torchaudio.save(str(audio_path), torch.from_numpy(audio).view(1, -1), TARGET_SR)
        if self._session_reply_events:
            with reply_events_path.open("w", encoding="utf-8") as f:
                for row in self._session_reply_events:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            reply_text = "".join(str(row.get("piece", "")) for row in self._session_reply_events)
            reply_text_path.write_text(reply_text, encoding="utf-8")
        meta_path.write_text(json.dumps({
            "source": source,
            "session_started_wall": float(self._session_started_wall),
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "motion_frames": int(motion.shape[0]),
            "helium_frames": int(helium.shape[0]) if helium is not None else 0,
            "adapter_50_frames": int(adapter_50.shape[0]) if adapter_50 is not None else 0,
            "adapter_25_frames": int(adapter_25.shape[0]) if adapter_25 is not None else 0,
            "wav2vec_50_frames": int(wav2vec_50.shape[0]) if wav2vec_50 is not None else 0,
            "projected_audio_frames": int(projected_audio.shape[0]) if projected_audio is not None else 0,
            "audio_samples": int(audio.shape[0]),
            "audio_seconds": float(audio.shape[0] / TARGET_SR) if audio.size > 0 else 0.0,
            "reply_text_chars": int(sum(len(str(row.get("piece", ""))) for row in self._session_reply_events)),
            "reply_events": int(len(self._session_reply_events)),
            "chunks": self._session_chunk_rows,
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
        }, indent=2), encoding="utf-8")
        print(f"[liveTryHeliumFM] dumped last session -> {session_dir}", flush=True)
        return session_dir

    def _extract_motion_tensor_from_payload(self, payload, path: str) -> torch.Tensor:
        if isinstance(payload, torch.Tensor):
            motion = payload
        elif isinstance(payload, dict):
            candidates = ["motion", "motion_latents", "latents", "full_motion", "pred_motion", "x"]
            motion = None
            for key in candidates:
                value = payload.get(key)
                if isinstance(value, torch.Tensor):
                    motion = value
                    break
            if motion is None:
                tensor_items = [
                    value
                    for value in payload.values()
                    if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == int(self.args.dim_w)
                ]
                if tensor_items:
                    motion = tensor_items[0]
            if motion is None:
                raise ValueError(
                    f"No motion tensor found in blink_motion_path={path}. "
                    f"Available keys={list(payload.keys())[:20]}"
                )
        else:
            raise TypeError(f"Unsupported blink motion payload type {type(payload)} from {path}")

        motion = motion.detach().float().cpu()
        if motion.ndim == 3 and motion.shape[0] == 1:
            motion = motion[0]
        if motion.ndim != 2 or motion.shape[-1] != int(self.args.dim_w):
            raise ValueError(
                f"Blink motion must have shape (T,{int(self.args.dim_w)}) or (1,T,{int(self.args.dim_w)}); "
                f"got {tuple(motion.shape)} from {path}"
            )
        return motion.contiguous()

    def _make_eye_mask(self, height: int, width: int) -> torch.Tensor:
        yy = torch.linspace(0.0, 1.0, int(height), device=self.device, dtype=self.dtype).view(1, 1, height, 1)
        xx = torch.linspace(0.0, 1.0, int(width), device=self.device, dtype=self.dtype).view(1, 1, 1, width)
        center_y = float(getattr(self.args, "eye_center_y", 0.405))
        radius_x = max(float(getattr(self.args, "eye_radius_x", 0.145)), 1e-6)
        radius_y = max(float(getattr(self.args, "eye_radius_y", 0.070)), 1e-6)
        feather = max(float(getattr(self.args, "eye_feather", 0.10)), 1e-6)

        mask = torch.zeros(1, 1, height, width, device=self.device, dtype=self.dtype)
        for center_x in (
            float(getattr(self.args, "eye_left_x", 0.36)),
            float(getattr(self.args, "eye_right_x", 0.64)),
        ):
            dist = ((xx - center_x) / radius_x).square() + ((yy - center_y) / radius_y).square()
            ellipse = ((1.0 + feather) - dist).clamp(0.0, feather) / feather
            mask = torch.maximum(mask, ellipse)
        return mask.clamp(0.0, 1.0).contiguous()

    @torch.no_grad()
    def _init_eye_blink_composite(self) -> None:
        blink_path = str(getattr(self.args, "blink_motion_path", "") or "")
        if not blink_path:
            raise ValueError("--enable_eye_blink_composite requires --blink_motion_path")
        if not Path(blink_path).is_file():
            raise FileNotFoundError(f"blink_motion_path does not exist: {blink_path}")

        payload = torch.load(blink_path, map_location="cpu")
        blink_motion = self._extract_motion_tensor_from_payload(payload, blink_path)
        blink_maps_parts: list[list[torch.Tensor]] = []
        chunk = max(1, int(getattr(self.args, "render_sub_batch", 8)))
        for start in range(0, int(blink_motion.shape[0]), chunk):
            sub = blink_motion[start:start + chunk].to(self.device, dtype=self.dtype)
            g_sub = self.g_r.expand(int(sub.shape[0]), -1)
            ta_b = self.renderer.adapt(sub, g_sub)
            maps_b = self.renderer.latent_token_decoder(ta_b)
            if not blink_maps_parts:
                blink_maps_parts = [[] for _ in range(len(maps_b))]
            for idx, map_b in enumerate(maps_b):
                blink_maps_parts[idx].append(map_b.detach())

        self._blink_maps = tuple(torch.cat(parts, dim=0).contiguous() for parts in blink_maps_parts)
        self._eye_masks = tuple(self._make_eye_mask(m.shape[-2], m.shape[-1]) for m in self._blink_maps)
        _sync_cuda()
        shapes = [tuple(m.shape) for m in self._blink_maps]
        print(
            f"[liveTryHeliumFM][blink] cached blink maps from {blink_path}: "
            f"frames={int(blink_motion.shape[0])} shapes={shapes}",
            flush=True,
        )

    def _composite_eye_blink_maps(
        self,
        current_maps: tuple[torch.Tensor, ...] | list[torch.Tensor],
        start_frame: int,
        num_frames: int,
    ) -> tuple[torch.Tensor, ...]:
        if self._blink_maps is None or self._eye_masks is None:
            return tuple(current_maps)
        blink_len = int(self._blink_maps[0].shape[0])
        if blink_len <= 0:
            return tuple(current_maps)
        indices = (torch.arange(int(num_frames), device=self.device) + int(start_frame)) % blink_len
        composited: list[torch.Tensor] = []
        for cur, blink_all, mask in zip(current_maps, self._blink_maps, self._eye_masks):
            blink = blink_all.index_select(0, indices).to(device=cur.device, dtype=cur.dtype)
            mask = mask.to(device=cur.device, dtype=cur.dtype)
            composited.append(blink * mask + cur * (1.0 - mask))
        return tuple(composited)

    @torch.no_grad()
    def _render_motion(self, motion: torch.Tensor) -> tuple[np.ndarray, dict]:
        timings: dict = {}
        t_total = time.perf_counter()
        motion = motion.to(self.device, dtype=self.dtype)
        n = int(motion.shape[0])

        render_start = self._render_frame_cursor
        fused = getattr(self.renderer, '_fused_render', None)
        fused_blink = getattr(self.renderer, '_fused_render_blink', None)
        active_fused = fused_blink if self.eye_blink_enabled else fused

        # torch.compile specializes the renderer to its warmup batch shape.
        # Pad the final short sub-batch so a 50-frame chunk does not compile a
        # second graph for the 2-frame tail during the first live response.
        render_motion = motion
        render_n = n
        if active_fused is not None and n < self.render_sub_batch:
            pad_n = self.render_sub_batch - n
            render_motion = torch.cat(
                [motion, motion[-1:].expand(pad_n, -1)],
                dim=0,
            ).contiguous()
            render_n = self.render_sub_batch

        g_r_sub = self.g_r.expand(render_n, -1)
        m_r_sub = tuple(m.expand(render_n, -1, -1, -1) for m in self.m_r)
        f_r_sub = [f.expand(render_n, -1, -1, -1) for f in self.f_r]
        if self.eye_blink_enabled and fused_blink is not None:
            blink_len = int(self._blink_maps[0].shape[0])
            indices = (
                torch.arange(render_n, device=self.device) + int(render_start)
            ) % blink_len
            blink_maps = tuple(
                blink_all.index_select(0, indices).to(dtype=self.dtype)
                for blink_all in self._blink_maps
            )
            frames = fused_blink(
                render_motion,
                g_r_sub,
                m_r_sub,
                f_r_sub,
                blink_maps,
                self._eye_masks,
            )[:n]
        elif fused is not None:
            frames = fused(render_motion, g_r_sub, m_r_sub, f_r_sub)[:n]
        else:
            ta_c = self.renderer.adapt(motion, g_r_sub)
            m_c = self.renderer.latent_token_decoder(ta_c)
            if self.eye_blink_enabled:
                m_c = self._composite_eye_blink_maps(m_c, render_start, n)
            frames = self.renderer.decode(m_c, m_r_sub, f_r_sub)
        self._render_frame_cursor += n

        frames_np = frames.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
        frames_np = frames_np.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        timings["total_ms"] = _ms(t_total)
        return frames_np, timings

    @torch.no_grad()
    def _render_motion_first_frame(self, motion: torch.Tensor) -> tuple[np.ndarray, dict]:
        """Render exactly one frame through a warmed batch-1 compiled graph."""
        timings: dict = {}
        t_total = time.perf_counter()
        motion = motion[:1].to(self.device, dtype=self.dtype).contiguous()
        render_start = self._render_frame_cursor
        g_r_sub = self.g_r.expand(1, -1)
        m_r_sub = tuple(m.expand(1, -1, -1, -1) for m in self.m_r)
        f_r_sub = [f.expand(1, -1, -1, -1) for f in self.f_r]
        fused_first = getattr(self.renderer, "_fused_render_first", None)
        fused_blink_first = getattr(
            self.renderer, "_fused_render_blink_first", None
        )

        if self.eye_blink_enabled and fused_blink_first is not None:
            blink_len = int(self._blink_maps[0].shape[0])
            indices = torch.tensor(
                [int(render_start) % blink_len],
                device=self.device,
                dtype=torch.long,
            )
            blink_maps = tuple(
                blink_all.index_select(0, indices).to(dtype=self.dtype)
                for blink_all in self._blink_maps
            )
            frames = fused_blink_first(
                motion,
                g_r_sub,
                m_r_sub,
                f_r_sub,
                blink_maps,
                self._eye_masks,
            )
        elif fused_first is not None:
            frames = fused_first(motion, g_r_sub, m_r_sub, f_r_sub)
        else:
            ta_c = self.renderer.adapt(motion, g_r_sub)
            current_maps = self.renderer.latent_token_decoder(ta_c)
            if self.eye_blink_enabled:
                current_maps = self._composite_eye_blink_maps(
                    current_maps, render_start, 1
                )
            frames = self.renderer.decode(current_maps, m_r_sub, f_r_sub)

        self._render_frame_cursor += 1
        frames_np = frames.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
        frames_np = frames_np.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        timings["total_ms"] = _ms(t_total)
        return frames_np, timings

    def render_and_encode_subbatch(
        self,
        motion_sub: torch.Tensor,
        audio_slices: list[np.ndarray],
        abs_start: int,
        text_payload: str,
        avatar_chunk_id: int,
        total_gen_ms: float,
    ) -> list[dict]:
        """Render a sub-batch of frames, JPEG-encode in parallel, return packet dicts."""
        frames_np, render_info = self._render_motion(motion_sub)

        jpeg_start = time.perf_counter()
        jpeg_futures = []
        for frame_rgb in frames_np:
            jpeg_futures.append(
                self._jpeg_pool.submit(encode_jpeg_bytes, frame_rgb, self.jpeg_quality)
            )

        packets = []
        gen_ms_i = int(round(float(total_gen_ms)))
        sr_i = int(round(float(TARGET_SR)))
        for j, fut in enumerate(jpeg_futures):
            idx = abs_start + j
            audio_slice = audio_slices[j] if j < len(audio_slices) else np.zeros(
                int(round(TARGET_SR / self.fps)), dtype=np.float32
            )
            jpeg_bytes = fut.result()
            output_audio_codec = str(
                getattr(self.args, "output_audio_codec", "pcm")
            ).lower()
            pcm_b = (
                b""
                if output_audio_codec == "opus"
                else _pcm_f32_to_i16_bytes(audio_slice)
            )
            blob = _wsbin.pack_av_frame(
                idx,
                idx + 1,
                gen_ms_i,
                sr_i,
                jpeg_bytes,
                pcm_b,
                text_payload,
                int(avatar_chunk_id),
            )
            packets.append(
                {
                    "frame_number": idx,
                    "ws_kind": "bytes",
                    "data": blob,
                    "audio_pcm": np.asarray(audio_slice, dtype=np.float32).copy(),
                    "t_ready": time.perf_counter(),
                }
            )
        self._last_render_encode_info = {
            "render_ms": float(render_info["total_ms"]),
            "jpeg_ms": _ms(jpeg_start),
        }
        return packets

    def audio_slice(self, frame_idx: int) -> np.ndarray:
        if self.audio_pcm is None:
            frame_samples = int(round(TARGET_SR / self.fps))
            return np.zeros(frame_samples, dtype=np.float32)
        frame_samples = int(round(TARGET_SR / self.fps))
        start = frame_idx * frame_samples
        chunk = self.audio_pcm[start:start + frame_samples]
        if chunk.shape[0] < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - chunk.shape[0]))
        return chunk


# ---------------------------------------------------------------------------
# WebSocket streaming coroutine (file-driven mode, unchanged)
# ---------------------------------------------------------------------------

async def stream_from_file(ws: WebSocket, engine: LiveHeliumFMEngine) -> None:
    """Simulate live streaming using --audio_path, sending frames back over WS."""
    if engine.audio_pcm is None:
        await ws.send_json({"type": "error", "msg": "No --audio_path given for file-streaming mode"})
        return

    audio = engine.audio_pcm
    lookahead_chunks = max(0, int(getattr(engine.args, "file_chunk_lookahead", 0)))
    total_chunks = int(np.ceil(len(audio) / engine.audio_chunk_samples))
    start_wall = time.perf_counter()
    emitted = 0

    print(
        f"[liveTryHeliumFM] stream_from_file: {total_chunks} chunks "
        f"lookahead={lookahead_chunks}",
        flush=True,
    )

    async def _emit_motion_chunk(
        motion: torch.Tensor,
        fm_info: dict,
        chunk_label: int,
        emitted_so_far: int,
    ) -> int:
        helium_ms = float(fm_info["helium_ms"])
        fm_ms = float(fm_info["fm_ms"])
        n_frames = int(motion.shape[0])
        all_frames_np: list[np.ndarray] = []
        render_ms = 0.0
        for sb_start in range(0, n_frames, engine.render_sub_batch):
            sub = motion[sb_start:sb_start + engine.render_sub_batch].to(
                engine.device, dtype=engine.dtype
            )
            frames_np, render_info = engine._render_motion(sub)
            render_ms += float(render_info["total_ms"])
            all_frames_np.extend(frames_np)

        print(
            f"[liveTryHeliumFM][chunk#{chunk_label}] "
            f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
            f"render={render_ms:.0f}ms frames={n_frames} "
            f"abs_start={fm_info['abs_start']}",
            flush=True,
        )

        for j, frame_rgb in enumerate(all_frames_np):
            idx = emitted_so_far + j
            chunk_id = idx + 1
            audio_b64 = _pcm_f32_to_i16_b64(engine.audio_slice(idx))
            jpeg_b64 = encode_jpeg_b64(frame_rgb, engine.jpeg_quality)

            await ws.send_json({
                "type": "chunk_audio",
                "chunk_id": chunk_id,
                "sample_rate": TARGET_SR,
                "pcm_s16le_b64": audio_b64,
            })
            await ws.send_json({
                "type": "chunk_frame",
                "chunk_id": chunk_id,
                "frame_idx": 0,
                "jpeg_b64": jpeg_b64,
                "moshi_text": (
                    f"Helium+FM | chunk#{chunk_label} "
                    f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
                    f"render={render_ms:.0f}ms"
                ),
                "server_fps": round(float(engine.fps), 1),
                "chunks_done": chunk_label,
            })
            target_t = start_wall + (idx + 1) / engine.fps
            await asyncio.sleep(max(0.0, target_t - time.perf_counter()))
        return emitted_so_far + len(all_frames_np)

    if lookahead_chunks <= 0:
        for chunk_idx in range(total_chunks):
            pcm_chunk = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ]
            pcm_real = pcm_chunk.copy()
            target_frames = int(round(pcm_chunk.shape[0] * engine.fps / TARGET_SR))
            if pcm_chunk.shape[0] < engine.audio_chunk_samples:
                pcm_chunk = np.pad(pcm_chunk, (0, engine.audio_chunk_samples - pcm_chunk.shape[0]))

            motion, fm_info = engine._process_pcm_chunk(pcm_chunk, target_frames)
            engine._record_session_chunk(pcm_real, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_idx + 1, emitted)
    else:
        pending_real: list[np.ndarray] = []
        prefix_audio = np.empty(0, dtype=np.float32)
        chunk_counter = 0

        def _process_exact_pending(pcm_real: np.ndarray) -> tuple[torch.Tensor, dict]:
            nonlocal prefix_audio
            target_frames = int(round(pcm_real.shape[0] * engine.fps / TARGET_SR))
            t0 = time.perf_counter()
            helium = engine.extractor.extract_exact_chunk_from_prefix(
                prefix_audio,
                engine.abs_frame,
                target_frames,
            )
            _sync_cuda()
            fm_info: dict = {"helium_ms": _ms(t0)}
            motion, sample_info = engine._sample_motion_from_helium(helium, target_frames)
            fm_info.update(sample_info)
            return motion, fm_info

        for chunk_idx in range(total_chunks):
            pcm_real = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ].copy()
            prefix_audio = np.concatenate([prefix_audio, pcm_real], axis=0)
            pending_real.append(pcm_real)

            while len(pending_real) > lookahead_chunks:
                chunk_counter += 1
                oldest = pending_real.pop(0)
                motion, fm_info = _process_exact_pending(oldest)
                engine._record_session_chunk(oldest, motion, fm_info)
                emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

        while pending_real:
            chunk_counter += 1
            oldest = pending_real.pop(0)
            motion, fm_info = _process_exact_pending(oldest)
            engine._record_session_chunk(oldest, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

    await ws.send_json({"type": "stream_end", "total_frames": emitted})
    engine.dump_last_session(source=str(engine.args.audio_path))
    print(f"[liveTryHeliumFM] stream done: {emitted} frames", flush=True)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class LiveHeliumFMOptions(BaseOptions):
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.set_defaults(wav2vec_sec=0.96)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8998)
        parser.add_argument("--html_path", default=str(ROOT / "static" / "index_v3_binary_fullscreen.html"))
        parser.add_argument("--generator_path", required=True)
        parser.add_argument("--lora_generator_path", default="", help="Optional LoRA generator checkpoint to apply on top of --generator_path")
        parser.add_argument("--lora_rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=float, default=128.0)
        parser.add_argument("--lora_dropout", type=float, default=0.05)
        parser.add_argument("--no_lora_pose_projection", action="store_true")
        parser.add_argument("--no_lora_audio_projection", action="store_true")
        parser.add_argument("--only_lora_pose_projection", action="store_true")
        parser.add_argument("--renderer_path", required=True)
        parser.add_argument("--adapter_path", required=True, help="Frontend fp32 Helium->Wav2Vec2 projected-frontend adapter checkpoint")
        parser.add_argument(
            "--adapter_type",
            choices=("frontend", "unitalk_last_layer"),
            default="frontend",
            help="Adapter architecture. UniTalk passes only layer 12 to IMTalker.",
        )
        parser.add_argument(
            "--adapter_window_mode",
            choices=("tail", "lookahead"),
            default="tail",
        )
        parser.add_argument(
            "--adapter_future_steps",
            type=int,
            default=6,
            help="12.5 Hz future-context steps retained in lookahead mode.",
        )
        parser.add_argument("--adapter_num_layers", type=int, default=6, help="Transformer layers in the frontend adapter checkpoint")
        parser.add_argument("--adapter_dropout", type=float, default=0.1, help="Dropout value used when constructing the frontend adapter")
        parser.add_argument("--stats_path", default="", help="Unused for frontend adapter mode; accepted for compatibility")
        parser.add_argument(
            "--silence_helium_path",
            default="",
            help="Optional real-silence Helium mean used to initialize the deque.",
        )
        parser.add_argument("--ref_path", required=True)
        parser.add_argument("--audio_path", default="", help="WAV to stream in fixed chunks (simulate-live mode)")
        # Moshi
        parser.add_argument("--moshi_root", default="/workspace/moshi")
        parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
        parser.add_argument("--moshi_weight", default="", help="Optional local PersonaPlex/Moshi LM checkpoint")
        parser.add_argument("--mimi_weight", default="", help="Optional local Mimi checkpoint")
        parser.add_argument("--tokenizer", default="", help="Optional local sentencepiece tokenizer")
        parser.add_argument("--quantize_4bit", action="store_true", help="Load PersonaPlex/Moshi LM with bnb 4-bit quantization")
        parser.add_argument("--num_codebooks", type=int, default=8, help="PersonaPlex/Moshi audio codebooks")
        parser.add_argument("--moshi_context", type=int, default=0, help="Optional PersonaPlex/Moshi KV context length")
        parser.add_argument("--voice_prompt", default="", help="PersonaPlex voice prompt filename, e.g. NATM0.pt")
        parser.add_argument("--voice_prompt_dir", default="", help="Optional PersonaPlex voice prompt directory")
        parser.add_argument("--text_prompt", default="", help="Optional PersonaPlex system text prompt")
        parser.add_argument("--moshi_reply_device", default=None, help="Optional separate CUDA device for Moshi reply generation")
        parser.add_argument("--enable_moshi_reply", action="store_true", help="Mic -> Moshi reply audio -> Helium/FM avatar")
        parser.add_argument("--moshi_cfg_coef", type=float, default=1.0)
        parser.add_argument("--direct_reply_hidden", action="store_true", default=True, help="Use Moshi generation hidden directly instead of re-encoding reply audio")
        parser.add_argument("--no_direct_reply_hidden", dest="direct_reply_hidden", action="store_false")
        parser.add_argument(
            "--disable_assistant_output_gate",
            action="store_true",
            help="Disable assistant-output RMS gating and always drive IMTalker from live PersonaPlex hidden states",
        )
        parser.add_argument(
            "--assistant_speech_rms_threshold",
            type=float,
            default=0.006,
            help="RMS threshold on PersonaPlex assistant reply audio before hidden states are allowed to drive avatar motion",
        )
        parser.add_argument(
            "--assistant_speech_hold_chunks",
            type=int,
            default=1,
            help="Keep assistant gate open for this many avatar chunks after reply audio drops below threshold",
        )
        # FM
        parser.add_argument("--audio_chunk_sec", type=float, default=0.96)
        parser.add_argument("--fm_chunk_frames", type=int, default=24, help="Must match wav2vec_sec×fps")
        parser.add_argument(
            "--helium_deque_size",
            type=int,
            default=100,
            help="Number of 12.5 Hz PersonaPlex hidden steps exposed to the adapter.",
        )
        parser.add_argument(
            "--enable_live_sliding_window",
            action="store_true",
            help="Run IMTalker/FM on a past/current/future feature window and emit only the current slice.",
        )
        parser.add_argument(
            "--slide_past_frames",
            type=int,
            default=10,
            help="Past 25Hz frames used by --enable_live_sliding_window.",
        )
        parser.add_argument(
            "--slide_future_frames",
            type=int,
            default=3,
            help="Future 25Hz frames used by --enable_live_sliding_window; 3 frames = 120ms.",
        )
        parser.add_argument("--skip_fm_audio_encoder", action="store_true", help="Skip FMGenerator's raw-audio Wav2Vec encoder; live PersonaPlex passes precomputed adapter features")
        parser.add_argument("--reply_hidden_steps_per_chunk", type=int, default=0, help="Raw Moshi 12.5Hz hidden steps per avatar chunk; 0 derives from fm_chunk_frames/fps")
        parser.add_argument("--prebuffer_chunks", type=int, default=3, help="Avatar chunks queued before sender starts pacing")
        parser.add_argument("--frame_q_backpressure", type=int, default=160)
        parser.add_argument(
            "--file_chunk_lookahead",
            type=int,
            default=0,
            help="For --audio_path mode, wait this many future chunks before emitting the oldest chunk",
        )
        parser.add_argument("--render_sub_batch", type=int, default=8)
        parser.add_argument("--jpeg_quality", type=int, default=86)
        parser.add_argument("--reply_audio_gain", type=float, default=1.0, help="Accepted for launch-script compatibility")
        parser.add_argument("--enable_seedvc", action="store_true")
        parser.add_argument("--seedvc_repo", default="/home/ubuntu/project/seed-vc")
        parser.add_argument("--seedvc_reference_audio", default="")
        parser.add_argument("--seedvc_steps", type=int, default=6)
        parser.add_argument("--seedvc_cfg", type=float, default=0.7)
        parser.add_argument("--seedvc_prompt_seconds", type=float, default=5.0)
        parser.add_argument(
            "--motion_ref_blend",
            type=float,
            default=0.0,
            help="Blend generated motion latent toward the reference image latent to reduce head tilt.",
        )
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--buffer_ms", type=int, default=80)
        parser.add_argument(
            "--renderer_precision",
            choices=("fp32", "fp16", "bf16"),
            default="fp32",
        )
        parser.add_argument(
            "--output_audio_codec",
            choices=("pcm", "opus"),
            default="pcm",
            help="Assistant audio transport; Opus is one persistent stream per session.",
        )
        parser.add_argument("--dump_motion", action="store_true", help="Dump last session motion/audio to disk")
        parser.add_argument("--dump_dir", default=str(ROOT / "live_try_dumps"))
        parser.add_argument(
            "--capture_dir",
            default="",
            help="Optional directory for exact mic/reply WAVs and replay metadata",
        )
        # Shared noise
        parser.add_argument("--shared_noise", action="store_true")
        parser.add_argument("--noise_seed", type=int, default=1234)
        parser.add_argument("--noise_max_frames", type=int, default=5000)
        parser.add_argument(
            "--noise_temporal_corr",
            type=float,
            default=0.0,
            help="AR(1)-style temporal correlation applied to FM initial noise.",
        )
        parser.add_argument(
            "--motion_prior_noise_blend",
            type=float,
            default=0.0,
            help="Small blend from previous/generated motion into FM initial noise.",
        )
        # Precision
        parser.add_argument("--fp32", action="store_true")
        parser.add_argument("--tf32", action="store_true")
        parser.add_argument("--compile_renderer", action="store_true")
        # Eye blink motion-map compositing
        parser.add_argument("--enable_eye_blink_composite", action="store_true")
        parser.add_argument("--blink_motion_path", default="", help="Cached blink motion latent .pt file")
        parser.add_argument("--eye_left_x", type=float, default=0.36)
        parser.add_argument("--eye_right_x", type=float, default=0.64)
        parser.add_argument("--eye_center_y", type=float, default=0.405)
        parser.add_argument("--eye_radius_x", type=float, default=0.145)
        parser.add_argument("--eye_radius_y", type=float, default=0.070)
        parser.add_argument("--eye_feather", type=float, default=0.10)
        return parser


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="IMTalker Helium MotionField Deque FM liveTry")
    assets_dir = ROOT / "static" / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)
    engine: LiveHeliumFMEngine | None = LiveHeliumFMEngine(args)
    moshi_engine: MoshiOnlyEngine | None = None
    seedvc: SeedVCStreamingConverter | None = None
    if bool(getattr(args, "enable_seedvc", False)):
        if not args.seedvc_reference_audio:
            raise ValueError("--seedvc_reference_audio is required with --enable_seedvc")
        seedvc = SeedVCStreamingConverter(
            repo=args.seedvc_repo,
            reference_audio=args.seedvc_reference_audio,
            input_sample_rate=TARGET_SR,
            block_seconds=float(args.audio_chunk_sec),
            diffusion_steps=int(args.seedvc_steps),
            cfg_rate=float(args.seedvc_cfg),
            prompt_seconds=float(args.seedvc_prompt_seconds),
            device=args.device,
        )
        print(
            f"[SeedVC] ready load={seedvc.load_seconds:.1f}s "
            f"reference={args.seedvc_reference_audio} steps={args.seedvc_steps} "
            f"cfg={args.seedvc_cfg}",
            flush=True,
        )

    def get_engine() -> LiveHeliumFMEngine:
        nonlocal engine
        if engine is None:
            engine = LiveHeliumFMEngine(args)
        return engine

    def get_moshi_engine() -> MoshiOnlyEngine:
        nonlocal moshi_engine
        if moshi_engine is None:
            moshi_engine = MoshiOnlyEngineWithHidden(
                moshi_root=args.moshi_root,
                mimi_hf_repo=args.mimi_hf_repo,
                device=getattr(args, "moshi_reply_device", None) or args.device,
                cfg_coef=float(args.moshi_cfg_coef),
                placeholder_jpeg_b64="",
                moshi_weight=getattr(args, "moshi_weight", ""),
                mimi_weight=getattr(args, "mimi_weight", ""),
                tokenizer=getattr(args, "tokenizer", ""),
                quantize_4bit=bool(getattr(args, "quantize_4bit", False)),
                num_codebooks=int(getattr(args, "num_codebooks", 8)),
                context=(int(args.moshi_context) if int(getattr(args, "moshi_context", 0)) > 0 else None),
                voice_prompt=getattr(args, "voice_prompt", ""),
                voice_prompt_dir=getattr(args, "voice_prompt_dir", ""),
                text_prompt=getattr(args, "text_prompt", ""),
            )
        return moshi_engine

    split_sessions: dict[str, dict] = {}
    # PersonaPlex/Mimi streaming state is process-global. Never let a new
    # browser session reset it while the previous session is still tearing down.
    conversation_lock = asyncio.Lock()

    async def _get_split_media_epoch(session: dict) -> float:
        while not session["prebuffer_ready"].is_set() and not session.get("closed", False):
            await asyncio.sleep(0.01)
        async with session["media_epoch_lock"]:
            if session.get("media_epoch") is None:
                session["media_epoch"] = time.perf_counter() + 0.08
            return float(session["media_epoch"])

    def _media_generation(session: dict) -> int:
        with session["media_generation_lock"]:
            return int(session["media_generation"])

    if bool(getattr(args, "enable_moshi_reply", False)):
        print("[liveTryHeliumFM_ws_binary] eager-loading Moshi/PersonaPlex reply engine", flush=True)
        get_moshi_engine()

    @app.get("/")
    async def index():
        if html_path.is_file():
            return FileResponse(
                html_path,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return HTMLResponse(f"<h1>Missing HTML</h1><p>Expected: {html_path}</p>", status_code=500)

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "live_moshi_reply_helium_fm" if args.enable_moshi_reply else "live_helium_fm",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "loaded": engine is not None,
        })

    @app.websocket("/ws/video")
    async def video_stream(ws: WebSocket):
        await ws.accept()
        session_id = str(ws.query_params.get("session_id", ""))
        session = split_sessions.get(session_id)
        if session is None:
            await ws.send_json({
                "type": "error",
                "message": "unknown or expired video session",
            })
            await ws.close()
            return

        frame_q: asyncio.Queue = session["frame_q"]
        send_start_wall = await _get_split_media_epoch(session)
        active_generation = _media_generation(session)
        base_frame_idx: int | None = None
        frames_sent = 0
        starvation_events = 0
        starve_start: float | None = None
        print(
            f"[AJ][VIDEO] connected session={session_id[:8]} "
            f"queued={frame_q.qsize()}",
            flush=True,
        )
        try:
            while True:
                try:
                    packet = frame_q.get_nowait()
                except asyncio.QueueEmpty:
                    if starve_start is None:
                        starve_start = time.perf_counter()
                        starvation_events += 1
                    await asyncio.sleep(0.004)
                    if session.get("closed", False) and frame_q.empty():
                        break
                    continue

                if packet is None:
                    break

                packet_generation = int(packet.get("media_generation", 0))
                current_generation = _media_generation(session)
                if packet_generation != current_generation:
                    continue
                if packet_generation != active_generation or base_frame_idx is None:
                    active_generation = packet_generation
                    base_frame_idx = int(packet["frame_number"])
                    send_start_wall = await _get_split_media_epoch(session)
                    await ws.send_json({
                        "type": "video_epoch",
                        "generation": active_generation,
                    })
                    print(f"[MEDIA-EPOCH][VIDEO] generation={active_generation} base_frame={base_frame_idx}", flush=True)

                if starve_start is not None:
                    gap_ms = 1000.0 * (time.perf_counter() - starve_start)
                    if gap_ms > 100:
                        print(
                            f"[AJ][VIDEO] STARVED {gap_ms:.0f}ms "
                            f"(event #{starvation_events}) "
                            f"frame_q={frame_q.qsize()} sent={frames_sent}",
                            flush=True,
                        )
                    starve_start = None

                idx = int(packet["frame_number"])
                target_t = send_start_wall + (idx - base_frame_idx) / float(args.fps)
                sleep_s = target_t - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

                # Interruption can advance the generation while this packet is
                # sleeping on its presentation timestamp. Revalidate at the
                # actual send boundary so one stale frame cannot enter the new
                # browser epoch and strand its A/V barrier at 1/6.
                session = split_sessions.get(session_id)
                if (
                    session is None
                    or packet_generation != _media_generation(session)
                ):
                    continue

                if packet.get("ws_kind") == "bytes":
                    await ws.send_bytes(packet["data"])
                else:
                    await ws.send_json(packet["msg"])
                frames_sent += 1

                late_s = time.perf_counter() - target_t
                if late_s > 0.5:
                    print(
                        f"[AJ][VIDEO] frame {idx} is {late_s*1000:.0f}ms late",
                        flush=True,
                    )
                if frames_sent % 50 == 0:
                    elapsed = time.perf_counter() - send_start_wall
                    print(
                        f"[AJ][VIDEO] sent={frames_sent} frame={idx} "
                        f"frame_q={frame_q.qsize()} elapsed={elapsed:.1f}s "
                        f"starve_events={starvation_events}",
                        flush=True,
                    )
        except (WebSocketDisconnect, RuntimeError, Exception) as exc:
            print(f"[AJ][VIDEO] closed session={session_id[:8]} {exc!r}", flush=True)
        finally:
            print(f"[AJ][VIDEO] done session={session_id[:8]} sent={frames_sent}", flush=True)

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        if conversation_lock.locked():
            print("[SESSION-STRICT] waiting for previous teardown", flush=True)
        await conversation_lock.acquire()
        print("[SESSION-STRICT] acquired PersonaPlex session lock", flush=True)
        fm_engine = get_engine()
        fm_engine.reset_session()
        if seedvc is not None:
            seedvc.reset()
        reply_engine = get_moshi_engine() if args.enable_moshi_reply else None
        # PersonaPlex is reset exactly once at the Start boundary below. Doing
        # it here as well replays uncached prompts twice for every connection.
        browser_input_sr = 48000
        opus_reader = sphn.OpusStreamReader(TARGET_SR)
        audio_packets_seen = 0

        # -- Queues for the reply pipeline --
        # mic_q: (raw_bytes, input_sr) or None to stop
        # frame_q: per-frame packet dicts or None to stop
        mic_q: queue.Queue[tuple[bytes, int, int, float] | None] | None = None
        frame_q: asyncio.Queue[dict | None] | None = None
        audio_q: asyncio.Queue[dict | None] | None = None
        gpu_thread: threading.Thread | None = None
        mic_ingest_thread: threading.Thread | None = None
        persona_thread: threading.Thread | None = None
        sender_task: asyncio.Task | None = None
        audio_sender_task: asyncio.Task | None = None
        event_loop: asyncio.AbstractEventLoop | None = None
        ws_send_lock = asyncio.Lock()
        media_epoch_lock = asyncio.Lock()
        media_epoch: float | None = None
        session_id = uuid.uuid4().hex
        prebuffer_ready = threading.Event()
        media_generation_lock = threading.Lock()
        media_generation = 0
        displayed_frame_lock = threading.Lock()
        displayed_frame_state = {"frame_number": None}
        playback_state = {"active": False, "hold": 0}
        session_started = threading.Event()
        last_mic_level_log_wall = 0.0
        profile_stats: dict[str, float | int | None] = {
            "connected_at": time.perf_counter(),
            "start_ready_at": None,
            "mic_packets": 0,
            "user_turns": 0,
            "user_turn_started_at": None,
            "user_turn_peak_rms": 0.0,
            "quiet_packets": 0,
            "media_resets": 0,
            "candidate_expirations": 0,
            "max_mic_queue": 0,
            "max_event_queue": 0,
            "max_frame_queue": 0,
            "max_audio_queue": 0,
            "max_event_age_ms": 0.0,
        }

        capture_root_text = str(getattr(args, "capture_dir", "") or "").strip()
        capture_enabled = bool(capture_root_text)
        capture_root = Path(capture_root_text).expanduser() if capture_enabled else None
        capture_lock = threading.Lock()
        capture_state: dict[str, object] = {
            "started_perf": None,
            "started_wall": None,
            "capture_id": "",
            "mic_parts": [],
            "reply_parts": [],
            "mic_packets": [],
            "persona_events": [],
            "timeline": [],
            "mic_samples": 0,
            "reply_samples": 0,
            "saved": False,
        }

        def _capture_reset() -> None:
            if not capture_enabled:
                return
            now_perf = time.perf_counter()
            now_wall = time.time()
            capture_id = (
                time.strftime("%Y%m%d_%H%M%S", time.localtime(now_wall))
                + f"_{session_id[:8]}"
            )
            with capture_lock:
                capture_state.update({
                    "started_perf": now_perf,
                    "started_wall": now_wall,
                    "capture_id": capture_id,
                    "mic_parts": [],
                    "reply_parts": [],
                    "mic_packets": [],
                    "persona_events": [],
                    "timeline": [{"offset_ms": 0.0, "kind": "start_received"}],
                    "mic_samples": 0,
                    "reply_samples": 0,
                    "saved": False,
                })
            print(f"[TRY13][CAPTURE-START] id={capture_id}", flush=True)

        def _capture_offset_ms(now: float | None = None) -> float | None:
            started_perf = capture_state.get("started_perf")
            if not isinstance(started_perf, float):
                return None
            return 1000.0 * ((time.perf_counter() if now is None else now) - started_perf)

        def _capture_marker(kind: str, **fields: object) -> None:
            if not capture_enabled:
                return
            with capture_lock:
                offset_ms = _capture_offset_ms()
                if offset_ms is None:
                    return
                row = {"offset_ms": round(offset_ms, 3), "kind": str(kind)}
                row.update(fields)
                timeline = capture_state["timeline"]
                assert isinstance(timeline, list)
                timeline.append(row)

        def _capture_mic_packet(
            pcm_i16: np.ndarray,
            *,
            packet_seq: int,
            sample_rate: int,
            mic_rms: float,
            mic_peak: float,
            media_generation_value: int,
        ) -> None:
            if not capture_enabled:
                return
            arr = np.asarray(pcm_i16, dtype=np.int16).reshape(-1).copy()
            with capture_lock:
                offset_ms = _capture_offset_ms()
                if offset_ms is None:
                    return
                sample_offset = int(capture_state["mic_samples"])
                parts = capture_state["mic_parts"]
                rows = capture_state["mic_packets"]
                assert isinstance(parts, list) and isinstance(rows, list)
                parts.append(arr)
                rows.append({
                    "seq": int(packet_seq),
                    "offset_ms": round(offset_ms, 3),
                    "sample_offset": sample_offset,
                    "samples": int(arr.size),
                    "sample_rate": int(sample_rate),
                    "rms": round(float(mic_rms), 8),
                    "peak": round(float(mic_peak), 8),
                    "assistant_playback_active": bool(playback_state["active"]),
                    "media_generation": int(media_generation_value),
                })
                capture_state["mic_samples"] = sample_offset + int(arr.size)

        def _capture_persona_event(event: dict) -> None:
            if not capture_enabled:
                return
            reply_part = np.zeros(0, dtype=np.int16)
            reply_b64 = event.get("reply_i16_b64")
            if isinstance(reply_b64, str) and reply_b64:
                with contextlib.suppress(Exception):
                    reply_part = np.frombuffer(
                        base64.b64decode(reply_b64), dtype=np.int16
                    ).copy()
            with media_generation_lock:
                generation = int(
                    split_sessions.get(session_id, {}).get("media_generation", 0)
                )
            with capture_lock:
                offset_ms = _capture_offset_ms()
                if offset_ms is None:
                    return
                sample_offset = int(capture_state["reply_samples"])
                parts = capture_state["reply_parts"]
                rows = capture_state["persona_events"]
                assert isinstance(parts, list) and isinstance(rows, list)
                if reply_part.size:
                    parts.append(reply_part)
                rows.append({
                    "step": int(event.get("step", -1)),
                    "offset_ms": round(offset_ms, 3),
                    "reply_sample_offset": sample_offset,
                    "reply_samples": int(reply_part.size),
                    "sample_rate": int(event.get("sample_rate", TARGET_SR)),
                    "input_rms": round(float(event.get("input_rms", 0.0) or 0.0), 8),
                    "reply_rms": round(float(event.get("reply_rms", 0.0) or 0.0), 8),
                    "reply_peak": round(float(event.get("reply_peak", 0.0) or 0.0), 8),
                    "token": int(event.get("token", -1)),
                    "piece": str(event.get("piece", "") or ""),
                    "media_generation": generation,
                    "encode_ms": round(float(event.get("encode_ms", 0.0) or 0.0), 3),
                    "lm_ms": round(float(event.get("lm_ms", 0.0) or 0.0), 3),
                    "decode_ms": round(float(event.get("decode_ms", 0.0) or 0.0), 3),
                    "total_ms": round(float(event.get("total_ms", 0.0) or 0.0), 3),
                })
                capture_state["reply_samples"] = sample_offset + int(reply_part.size)

        def _write_capture_wav(path: Path, parts: list[np.ndarray]) -> int:
            pcm = (
                np.concatenate(parts).astype(np.int16, copy=False)
                if parts
                else np.zeros(0, dtype=np.int16)
            )
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(TARGET_SR)
                wav_file.writeframes(pcm.tobytes())
            return int(pcm.size)

        def _save_capture(reason: str) -> Path | None:
            if not capture_enabled or capture_root is None:
                return None
            with capture_lock:
                if bool(capture_state.get("saved")):
                    return None
                started_perf = capture_state.get("started_perf")
                capture_id = str(capture_state.get("capture_id") or "")
                if not isinstance(started_perf, float) or not capture_id:
                    return None
                capture_state["saved"] = True
                mic_parts = list(capture_state["mic_parts"])
                reply_parts = list(capture_state["reply_parts"])
                mic_rows = list(capture_state["mic_packets"])
                persona_rows = list(capture_state["persona_events"])
                timeline_rows = list(capture_state["timeline"])
                started_wall = float(capture_state["started_wall"])

            capture_dir = capture_root / capture_id
            capture_dir.mkdir(parents=True, exist_ok=True)
            mic_samples = _write_capture_wav(capture_dir / "mic_24k.wav", mic_parts)
            reply_samples = _write_capture_wav(capture_dir / "reply_24k.wav", reply_parts)

            for name, rows in (
                ("mic_packets.jsonl", mic_rows),
                ("persona_events.jsonl", persona_rows),
                ("timeline.jsonl", timeline_rows),
            ):
                with (capture_dir / name).open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            manifest = {
                "schema_version": 1,
                "capture_id": capture_id,
                "session_id": session_id,
                "reason": str(reason),
                "started_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(started_wall)
                ),
                "duration_s": round(time.perf_counter() - started_perf, 3),
                "input_wav": "mic_24k.wav",
                "reply_wav": "reply_24k.wav",
                "sample_rate": TARGET_SR,
                "mic_samples": mic_samples,
                "mic_duration_s": round(mic_samples / TARGET_SR, 3),
                "reply_samples": reply_samples,
                "reply_duration_s": round(reply_samples / TARGET_SR, 3),
                "mic_packet_count": len(mic_rows),
                "persona_event_count": len(persona_rows),
                "persona_audio_text": str(
                    getattr(reply_engine, "audio_text", "") if reply_engine is not None else ""
                ),
                "persona_sampled_text": str(
                    getattr(reply_engine, "sampled_text", "") if reply_engine is not None else ""
                ),
                "runtime": {
                    "variant": "TRY13-HANDSHAKE-FULL-BRIDGE-BNB4-FP32",
                    "moshi_weight": str(getattr(args, "moshi_weight", "")),
                    "mimi_weight": str(getattr(args, "mimi_weight", "")),
                    "tokenizer": str(getattr(args, "tokenizer", "")),
                    "quantize_4bit": bool(getattr(args, "quantize_4bit", False)),
                    "voice_prompt": str(getattr(args, "voice_prompt", "")),
                    "voice_prompt_dir": str(getattr(args, "voice_prompt_dir", "")),
                    "text_prompt": str(getattr(args, "text_prompt", "")),
                    "moshi_cfg_coef": float(getattr(args, "moshi_cfg_coef", 1.0)),
                    "audio_chunk_sec": float(getattr(args, "audio_chunk_sec", 0.0)),
                    "fm_chunk_frames": int(getattr(args, "fm_chunk_frames", 0)),
                    "render_sub_batch": int(getattr(args, "render_sub_batch", 0)),
                    "jpeg_quality": int(getattr(args, "jpeg_quality", 0)),
                    "nfe": int(getattr(args, "nfe", 0)),
                    "a_cfg_scale": float(getattr(args, "a_cfg_scale", 0.0)),
                    "seed": int(getattr(args, "seed", 0)),
                },
                "profile": {
                    key: (round(value, 3) if isinstance(value, float) else value)
                    for key, value in profile_stats.items()
                },
            }
            with (capture_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(
                f"[TRY13][CAPTURE-SAVED] dir={capture_dir} "
                f"mic_s={mic_samples / TARGET_SR:.2f} "
                f"reply_s={reply_samples / TARGET_SR:.2f} reason={reason}",
                flush=True,
            )
            return capture_dir

        stream_task: asyncio.Task | None = None

        if reply_engine is not None:
            # Match native PersonaPlex: microphone ingress must never be dropped
            # because video delivery is temporarily slower than generation.
            mic_q = queue.Queue()
            # Try-7 keeps only a small, bounded media reserve. The senders
            # cannot accumulate minutes of stale audio/video behind the user.
            frame_q = asyncio.Queue(maxsize=24)
            audio_q = asyncio.Queue(maxsize=12)
            event_loop = asyncio.get_running_loop()
            reply_input_lock = threading.Lock()
            persona_event_q: queue.Queue[dict] = queue.Queue()
            pipeline_stop_event = threading.Event()
            split_sessions[session_id] = {
                "frame_q": frame_q,
                "prebuffer_ready": prebuffer_ready,
                "media_epoch_lock": media_epoch_lock,
                "media_epoch": media_epoch,
                "media_generation_lock": media_generation_lock,
                "media_generation": media_generation,
                "closed": False,
                "created_at": time.perf_counter(),
            }

        try:
            await ws.send_json({
                "type": "server_ready",
                "variant": "TRY13-HANDSHAKE-FULL-BRIDGE-BNB4-FP32",
                "session_id": session_id,
                "video_ws_path": f"/ws/video?session_id={session_id}",
                "sample_rate": TARGET_SR,
                "output_audio_codec": str(args.output_audio_codec),
                "model_type": "moshi_reply+helium_fm+renderer" if args.enable_moshi_reply else "helium_fm+renderer",
                "tokens_per_chunk": int(args.fm_chunk_frames),
                "has_audio_file": fm_engine.audio_pcm is not None,
                "buffer_ms": int(args.buffer_ms),
                "av_transport": "binary",
                "target_fps": round(float(args.fps), 2),
                "session_capture": capture_enabled,
            })
        except (WebSocketDisconnect, RuntimeError) as exc:
            # A browser can disconnect while queued behind the previous strict
            # session. Never let that stale socket retain the PersonaPlex lock.
            session = split_sessions.pop(session_id, None)
            if session is not None:
                session["closed"] = True
            if conversation_lock.locked():
                conversation_lock.release()
                print(
                    "[SESSION-STRICT] released stale pre-ready session lock",
                    flush=True,
                )
            print(
                f"[liveTryHeliumFM] client disconnected before server_ready: {exc!r}",
                flush=True,
            )
            return
        print("[liveTryHeliumFM] websocket connected; sent server_ready", flush=True)

        if reply_engine is not None:
            def _mic_ingest_worker() -> None:
                """Drain browser audio independently from video/render progress.

                Every wake-up takes ALL queued packets and appends them in ONE locked call.
                The PersonaPlex thread holds reply_input_lock for a whole ~67 ms step and
                re-takes it at once when it has a backlog; appending one packet per lock let
                the mic queue grow without bound (seen: 9 s, 460 packets). Batching makes the
                ingest catch up in a single lock hold, so that feedback loop cannot start.
                """
                assert mic_q is not None
                packets = 0
                stop = False
                while not stop:
                    item = mic_q.get()
                    if item is None:
                        break
                    batch = [item]
                    while True:                       # everything already waiting
                        try:
                            nxt = mic_q.get_nowait()
                        except queue.Empty:
                            break
                        if nxt is None:
                            stop = True
                            break
                        batch.append(nxt)
                    # group consecutive packets with the same sample rate (always 24 kHz in practice)
                    groups: list[tuple[int, list]] = []
                    for raw_bytes, input_sr, packet_seq, received_at in batch:
                        if not raw_bytes:
                            continue
                        if groups and groups[-1][0] == input_sr:
                            groups[-1][1].append((raw_bytes, packet_seq, received_at))
                        else:
                            groups.append((input_sr, [(raw_bytes, packet_seq, received_at)]))
                    if not groups:
                        continue
                    with reply_input_lock:
                        for input_sr, pk in groups:
                            reply_engine.append_browser_pcm(
                                np.frombuffer(b"".join(p[0] for p in pk), dtype=np.int16), input_sr
                            )
                    n_new = sum(len(pk) for _, pk in groups)
                    oldest_seq, oldest_at = groups[0][1][0][1], groups[0][1][0][2]
                    prev = packets
                    packets += n_new
                    queue_age_ms = 1000.0 * (time.perf_counter() - oldest_at)
                    if queue_age_ms > 100.0:
                        print(
                            f"[OPUS-TRANSPORT] seq={oldest_seq} "
                            f"ingest_age={queue_age_ms:.1f}ms batch={n_new} pending_q={mic_q.qsize()}",
                            flush=True,
                        )
                    if packets // 250 != prev // 250:
                        print(
                            f"[MIC-ISOLATED] ingested={packets} pending_q={mic_q.qsize()}",
                            flush=True,
                        )
                print(f"[MIC-ISOLATED] stopped ingested={packets}", flush=True)

            def _persona_buffered_samples() -> int:
                with reply_input_lock:
                    return int(reply_engine.input_buffer.shape[0])

            def _persona_process_ready(max_steps: int) -> list[dict]:
                with reply_input_lock:
                    return reply_engine.process_ready_steps_limited(max_steps)

            def _persona_priority_worker() -> None:
                """Run one PersonaPlex step whenever one complete Mimi frame is ready.

                Keeping the critical section to one 80 ms step lets microphone
                ingestion append new PCM between steps instead of waiting behind a
                multi-step catch-up burst.
                """
                priority_stream = None
                if torch.cuda.is_available():
                    try:
                        # PyTorch accepts -1 as a high-priority stream even in builds
                        # that do not expose get_stream_priority_range().
                        priority_stream = torch.cuda.Stream(priority=int(os.environ.get("TRY19_PERSONA_PRIO", "-1")))
                        print("[PERSONA-CONTINUOUS] high-priority CUDA stream enabled", flush=True)
                    except Exception as exc:
                        print(f"[PERSONA-CONTINUOUS] stream fallback: {exc!r}", flush=True)
                total_events = 0
                native_audio_seq = 0
                last_event_alarm_wall = 0.0

                while not pipeline_stop_event.is_set():
                    if not session_started.is_set() or _persona_buffered_samples() < MIMI_FRAME_SIZE:
                        time.sleep(0.002)
                        continue
                    buffered_before = _persona_buffered_samples()
                    ready_before = buffered_before // MIMI_FRAME_SIZE
                    if priority_stream is None:
                        events = _persona_process_ready(1)
                    else:
                        with torch.cuda.stream(priority_stream):
                            events = _persona_process_ready(1)
                        # Hidden/audio copies produced by _step synchronize the
                        # generated event before it is published to IMTalker.
                        priority_stream.synchronize()
                    for event in events:
                        _capture_persona_event(event)
                        with media_generation_lock:
                            event["media_generation"] = int(split_sessions[session_id]["media_generation"])
                        # Timestamp each native event so Try-7 can measure the
                        # real PersonaPlex-to-publication delay.
                        event["persona_enqueued_at"] = time.perf_counter()
                        persona_event_q.put(event)
                        event_depth = persona_event_q.qsize()
                        profile_stats["max_event_queue"] = max(
                            int(profile_stats["max_event_queue"] or 0),
                            event_depth,
                        )
                        now_wall = time.perf_counter()
                        if event_depth >= 8 and now_wall - last_event_alarm_wall >= 1.0:
                            print(
                                f"[TRY13][EVENT-BACKLOG] depth={event_depth} "
                                f"oldest_step={event.get('step', -1)}",
                                flush=True,
                            )
                            last_event_alarm_wall = now_wall
                    total_events += len(events)
                    if ready_before > 2:
                        print(
                            f"[PERSONA-CONTINUOUS] backlog={ready_before} "
                            f"drained={len(events)} event_q={persona_event_q.qsize()}",
                            flush=True,
                        )
                print(f"[PERSONA-CONTINUOUS] stopped events={total_events}", flush=True)

            def _gpu_producer_thread() -> None:
                """Convert real, mic-paced PersonaPlex events into matched A/V slices."""
                assert (
                    mic_q is not None
                    and frame_q is not None
                    and audio_q is not None
                    and event_loop is not None
                )
                pending_reply_steps: list[dict] = []
                pending_reply_hidden: list[torch.Tensor] = []
                pending_reply_audio: list[np.ndarray] = []
                reply_step_history: list[dict] = []
                reply_audio_history: list[np.ndarray] = []
                reply_avatar_chunk_idx = 0
                chunk_produce_count = 0
                active_generation = 0
                staged_audio_seq = 0
                # Pause producer when queue is this deep (qsize() is heuristic across threads).
                _bp_floor = 2 if NATIVE_PASSTHROUGH else 6
                FRAME_Q_BACKPRESS = max(
                    _bp_floor,
                    min(24, int(getattr(args, "frame_q_backpressure", 12))),
                )
                AUDIO_Q_BACKPRESS = max(3, FRAME_Q_BACKPRESS // 2)
                FRAME_Q_PUT_TIMEOUT_S = 120.0
                prebuffer_chunks = max(0, int(getattr(args, "prebuffer_chunks", PREBUFFER_CHUNKS)))
                hidden_steps_per_chunk = int(getattr(args, "reply_hidden_steps_per_chunk", 0))
                if hidden_steps_per_chunk <= 0:
                    hidden_steps_per_chunk = int(round(float(args.fm_chunk_frames) * 12.5 / float(args.fps)))
                hidden_steps_per_chunk = max(1, hidden_steps_per_chunk)
                # One producer pass may form at most one 2-second chunk. This
                # prevents a single stale event burst from publishing 2-3 more
                # chunks before backpressure is checked again.
                max_moshi_steps_per_loop = hidden_steps_per_chunk
                was_silent = True
                assistant_gate_hold = 0
                last_motion_frame: torch.Tensor | None = None
                neutral_motion_frame: torch.Tensor | None = None
                seedvc_was_active = False

                def _enqueue_frame(pkt: dict) -> None:
                    """Block until frame_q accepts pkt (real backpressure). Must run from GPU thread."""
                    pkt["media_generation"] = active_generation
                    fut = asyncio.run_coroutine_threadsafe(frame_q.put(pkt), event_loop)
                    try:
                        fut.result(timeout=FRAME_Q_PUT_TIMEOUT_S)
                    except TimeoutError:
                        print(
                            f"[GPU] WARNING frame_q.put timeout frame={pkt.get('frame_number')}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GPU] WARNING frame_q.put failed: {e!r}", flush=True)

                def _enqueue_audio(pkt: dict) -> None:
                    pkt["media_generation"] = active_generation
                    fut = asyncio.run_coroutine_threadsafe(audio_q.put(pkt), event_loop)
                    try:
                        fut.result(timeout=FRAME_Q_PUT_TIMEOUT_S)
                    except TimeoutError:
                        print(
                            f"[GPU] WARNING audio_q.put timeout frame={pkt.get('frame_number')}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GPU] WARNING audio_q.put failed: {e!r}", flush=True)

                if prebuffer_chunks <= 0 and not prebuffer_ready.is_set():
                    prebuffer_ready.set()
                    print("[GPU] prebuffer=0, sender starts immediately", flush=True)

                while not pipeline_stop_event.is_set():
                    # Mic ingestion runs in its own worker and remains independent
                    # from all video backpressure below.
                    while frame_q.qsize() >= FRAME_Q_BACKPRESS:
                        time.sleep(0.004)

                    if not session_started.is_set():
                        time.sleep(0.003)
                        continue

                    try:
                        first_event = persona_event_q.get(timeout=0.003)
                    except queue.Empty:
                        time.sleep(0.003)
                        continue

                    # PersonaPlex runs independently; this worker forms at
                    # most one avatar chunk per pass.
                    t_recv = time.perf_counter()
                    oldest_event_age_ms = 1000.0 * max(
                        0.0,
                        t_recv - float(first_event.get("persona_enqueued_at", t_recv)),
                    )
                    profile_stats["max_event_age_ms"] = max(
                        float(profile_stats["max_event_age_ms"] or 0.0),
                        oldest_event_age_ms,
                    )
                    events = [first_event]
                    while len(events) < max_moshi_steps_per_loop:
                        try:
                            events.append(persona_event_q.get_nowait())
                        except queue.Empty:
                            break
                    for ev in events:
                        event_generation = int(ev.get("media_generation", 0))
                        if event_generation != active_generation:
                            active_generation = event_generation
                            pending_reply_steps.clear(); pending_reply_hidden.clear(); pending_reply_audio.clear()
                            reply_step_history.clear(); reply_audio_history.clear()
                            chunk_produce_count = 0; was_silent = True; assistant_gate_hold = 0
                            fm_engine.helium_deque = None
                            fm_engine.helium_deque_filled = 0
                            print(
                                f"[MEDIA-EPOCH][GPU] generation={active_generation} "
                                "cleared adapter batching and Helium window",
                                flush=True,
                            )
                        pending_reply_steps.append(ev)
                        fm_engine._session_reply_events.append({
                            "step": int(ev.get("step", -1)),
                            "token": int(ev.get("token", -1)),
                            "piece": str(ev.get("piece", "")),
                            "audio_text": str(ev.get("audio_text", "")),
                            "reply_rms": float(ev.get("reply_rms", 0.0)),
                            "reply_peak": float(ev.get("reply_peak", 0.0)),
                            "input_rms": float(ev.get("input_rms", 0.0)),
                            "hidden": bool(isinstance(ev.get("helium_hidden"), torch.Tensor)),
                            "total_ms": float(ev.get("total_ms", 0.0)),
                        })
                        reply_pcm = (
                            np.frombuffer(base64.b64decode(ev["reply_i16_b64"]), dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        pending_commit_count = 0
                        next_reply_audio_history = None
                        next_reply_step_history = None
                        if bool(getattr(args, "direct_reply_hidden", False)):
                            hidden = ev.get("helium_hidden")
                            if not isinstance(hidden, torch.Tensor):
                                continue
                            pending_reply_hidden.append(hidden.squeeze(0).contiguous())
                            pending_reply_audio.append(reply_pcm)
                            if len(pending_reply_hidden) < hidden_steps_per_chunk:
                                continue

                            # Peek at a complete chunk. Commit it only after
                            # every matched A/V slice has been published.
                            used_hidden = list(
                                pending_reply_hidden[:hidden_steps_per_chunk]
                            )
                            used_audio = list(
                                pending_reply_audio[:hidden_steps_per_chunk]
                            )
                            used_steps = list(
                                pending_reply_steps[:hidden_steps_per_chunk]
                            )
                            pending_commit_count = hidden_steps_per_chunk

                            if str(getattr(args, "adapter_window_mode", "tail")) == "lookahead":
                                history_size = int(fm_engine.helium_deque_size)
                                future_steps = int(
                                    getattr(args, "adapter_future_steps", 6)
                                )
                                next_reply_audio_history = (
                                    reply_audio_history + used_audio
                                )[-history_size:]
                                next_reply_step_history = (
                                    reply_step_history + used_steps
                                )[-history_size:]

                                zero_audio = np.zeros_like(used_audio[0])
                                audio_window = (
                                    [zero_audio]
                                    * (history_size - len(next_reply_audio_history))
                                    + next_reply_audio_history
                                )
                                silent_step = {
                                    "token": 0,
                                    "reply_rms": 0.0,
                                    "total_ms": 0.0,
                                    "audio_text": "",
                                }
                                step_window = (
                                    [silent_step]
                                    * (history_size - len(next_reply_step_history))
                                    + next_reply_step_history
                                )
                                output_end = history_size - future_steps
                                output_start = (
                                    output_end - hidden_steps_per_chunk
                                )
                                used_audio = audio_window[output_start:output_end]
                                used_steps = step_window[output_start:output_end]

                            pcm_chunk = np.concatenate(used_audio, axis=0).astype(np.float32, copy=False)
                            reply_rms = float(np.sqrt(np.mean(np.square(pcm_chunk)))) if pcm_chunk.size else 0.0
                            step_rms = max(
                                (
                                    float(s.get("reply_rms", 0.0) or 0.0)
                                    for s in used_steps
                                ),
                                default=0.0,
                            )
                            speech_threshold = float(
                                getattr(args, "assistant_speech_rms_threshold", 0.006)
                            )
                            assistant_active_now = max(reply_rms, step_rms) > speech_threshold
                            if assistant_active_now:
                                assistant_gate_hold = max(
                                    0,
                                    int(getattr(args, "assistant_speech_hold_chunks", 1)),
                                )
                            elif assistant_gate_hold > 0:
                                assistant_gate_hold -= 1
                            assistant_active = assistant_active_now or assistant_gate_hold > 0
                            if bool(getattr(args, "disable_assistant_output_gate", False)):
                                assistant_active = True

                            if NATIVE_PASSTHROUGH:
                                # The original server never gates output by RMS.
                                # PersonaPlex already emits silence when it stops;
                                # detecting that and forcing digital zero is
                                # redundant. Motion follows the same Helium stream,
                                # so audio and video go quiet together by design.
                                assistant_active = True
                                assistant_active_now = True
                            previous_active = not was_silent
                            transition = assistant_active != previous_active
                            if transition:
                                print(
                                    f"[GPU] Avatar gate transition "
                                    f"{'speech' if assistant_active else 'idle'} "
                                    f"reply_rms={reply_rms:.5f} step_rms={step_rms:.5f}",
                                    flush=True,
                                )
                            was_silent = not assistant_active

                            # Keep the actual PersonaPlex state stream continuous
                            # through PAD/silent reply steps. Only audible output is
                            # gated; the adapter and FMT still receive native hidden.
                            helium_chunk = torch.cat(used_hidden, dim=0)
                            if not assistant_active:
                                used_audio = [np.zeros_like(audio) for audio in used_audio]
                                pcm_chunk = np.zeros_like(pcm_chunk)
                            target_frames = max(1, int(round(len(pcm_chunk) * float(args.fps) / TARGET_SR)))

                            motion, fm_info = fm_engine._sample_motion_from_helium(helium_chunk, target_frames)
                            if (
                                transition
                                and last_motion_frame is not None
                                and TRANSITION_BLEND_FRAMES > 0
                            ):
                                blend_frames = min(
                                    TRANSITION_BLEND_FRAMES,
                                    int(motion.shape[0]),
                                )
                                alpha = torch.linspace(
                                    0.0,
                                    1.0,
                                    blend_frames + 2,
                                    device=motion.device,
                                    dtype=motion.dtype,
                                )[1:-1].unsqueeze(-1)
                                previous = last_motion_frame.to(
                                    device=motion.device,
                                    dtype=motion.dtype,
                                ).unsqueeze(0)
                                motion[:blend_frames] = (
                                    previous * (1.0 - alpha)
                                    + motion[:blend_frames] * alpha
                                )
                            last_motion_frame = motion[-1].detach().clone()
                            if not assistant_active:
                                neutral_motion_frame = motion[-1].detach().clone()
                            fm_info["assistant_active"] = bool(assistant_active)
                            fm_info["assistant_reply_rms"] = float(reply_rms)
                            fm_info["assistant_step_rms"] = float(step_rms)
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)
                            fm_engine._record_session_chunk(pcm_chunk, motion, fm_info)
                        else:
                            result = fm_engine.feed_pcm_f32(reply_pcm)
                            if result is None:
                                continue

                            motion, fm_info, pcm_chunk = result
                            steps_per_avatar_chunk = max(1, int(round(len(pcm_chunk) / MIMI_FRAME_SIZE)))
                            used_steps = pending_reply_steps[:steps_per_avatar_chunk]
                            pending_reply_steps = pending_reply_steps[steps_per_avatar_chunk:]
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)

                        reply_avatar_chunk_idx += 1
                        chunk_produce_count += 1

                        moshi_total_ms = sum(float(s.get("total_ms", 0.0)) for s in used_steps)

                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        if seedvc is not None:
                            seedvc_active = bool(fm_info.get("assistant_active", True))
                            if seedvc_active:
                                if not seedvc_was_active:
                                    try:
                                        seedvc.reset()
                                    except Exception as exc:
                                        print(f"[SeedVC] reset failed; continuing: {exc!r}", flush=True)
                                try:
                                    pcm_chunk, seedvc_info = seedvc.convert(pcm_chunk)
                                    fm_info.update(seedvc_info)
                                    print(
                                        f"[SeedVC] chunk={reply_avatar_chunk_idx} "
                                        f"ms={seedvc_info['seedvc_ms']:.1f} "
                                        f"rtf={seedvc_info['seedvc_rtf']:.3f} "
                                        f"sola={int(seedvc_info['sola_offset'])}",
                                        flush=True,
                                    )
                                except Exception as exc:
                                    print(f"[SeedVC] conversion failed, using raw PCM: {exc!r}", flush=True)
                            else:
                                pcm_chunk = np.zeros_like(pcm_chunk)
                            seedvc_was_active = seedvc_active
                        frame_audio = split_audio_into_frame_slices(pcm_chunk, args.fps)
                        n_frames = int(motion.shape[0])
                        emitted = int(fm_info["abs_start"])
                        output_step = used_steps[-1] if used_steps else ev
                        text_payload = (
                            output_step.get("audio_text")
                            or output_step.get("sampled_text")
                            or ""
                        )
                        total_gen_ms = (
                            moshi_total_ms + float(fm_info["helium_ms"]) + float(fm_info["fm_ms"])
                        )


                        t_chunk_start = time.perf_counter()
                        render_ms = 0.0
                        jpeg_ms = 0.0
                        render_preempted = False
                        published_frames = 0
                        published_audio = 0
                        audio_cursor = 0

                        # Publish each renderer sub-batch with the matching native
                        # audio steps. This removes the old all-50-frame atomic
                        # barrier while keeping the two streams on one generation.
                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sb_end = min(
                                sb_start + fm_engine.render_sub_batch,
                                n_frames,
                            )
                            slice_frames = sb_end - sb_start
                            audio_end = min(
                                len(used_audio),
                                int(round(sb_end * len(used_audio) / n_frames)),
                            )
                            slice_audio = max(0, audio_end - audio_cursor)

                            while (
                                frame_q.qsize() + slice_frames > FRAME_Q_BACKPRESS
                                or audio_q.qsize() + slice_audio > AUDIO_Q_BACKPRESS
                            ):
                                if pipeline_stop_event.is_set():
                                    render_preempted = True
                                    break
                                with media_generation_lock:
                                    session_state = split_sessions.get(session_id)
                                    current_generation = (
                                        int(session_state["media_generation"])
                                        if session_state is not None
                                        else -1
                                    )
                                if (
                                    session_state is None
                                    or current_generation != active_generation
                                ):
                                    render_preempted = True
                                    break
                                time.sleep(0.004)
                            with media_generation_lock:
                                session_state = split_sessions.get(session_id)
                                current_generation = (
                                    int(session_state["media_generation"])
                                    if session_state is not None
                                    else -1
                                )
                            if (
                                render_preempted
                                or pipeline_stop_event.is_set()
                                or session_state is None
                                or current_generation != active_generation
                            ):
                                render_preempted = True
                                break

                            packets = fm_engine.render_and_encode_subbatch(
                                motion[sb_start:sb_end],
                                frame_audio[sb_start:sb_end],
                                abs_start=emitted + sb_start,
                                text_payload=text_payload,
                                avatar_chunk_id=avatar_chunk_id,
                                total_gen_ms=total_gen_ms,
                            )
                            timing = getattr(
                                fm_engine,
                                "_last_render_encode_info",
                                {},
                            )
                            render_ms += float(timing.get("render_ms", 0.0))
                            jpeg_ms += float(timing.get("jpeg_ms", 0.0))

                            for pkt in packets:
                                _enqueue_frame(pkt)
                            for audio_index in range(audio_cursor, audio_end):
                                _enqueue_audio({
                                    "audio_packet_index": staged_audio_seq,
                                    "audio_pcm": np.asarray(
                                        used_audio[audio_index],
                                        dtype=np.float32,
                                    ).copy(),
                                    "created_at": time.perf_counter(),
                                })
                                staged_audio_seq += 1
                            audio_cursor = audio_end
                            published_frames += len(packets)
                            published_audio += slice_audio

                            if not prebuffer_ready.is_set():
                                prebuffer_ready.set()
                                print(
                                    f"[TRY13][PROGRESSIVE-START] "
                                    f"audio={published_audio} "
                                    f"frames={published_frames} "
                                    f"generation={active_generation}",
                                    flush=True,
                                )

                        if render_preempted:
                            print(
                                f"[TRY13][PREEMPT] generation={active_generation} "
                                f"published_frames={published_frames} "
                                f"published_audio={published_audio} "
                                f"at_subbatch={sb_start}/{n_frames}; "
                                "stale generation will be cleared",
                                flush=True,
                            )
                            continue

                        if pending_commit_count:
                            del pending_reply_hidden[:pending_commit_count]
                            del pending_reply_audio[:pending_commit_count]
                            del pending_reply_steps[:pending_commit_count]
                            if next_reply_audio_history is not None:
                                reply_audio_history = next_reply_audio_history
                            if next_reply_step_history is not None:
                                reply_step_history = next_reply_step_history

                        chunk_wall_ms = _ms(t_chunk_start)
                        produce_latency_ms = _ms(t_recv)
                        q_depth = frame_q.qsize() if frame_q is not None else -1
                        audio_depth = audio_q.qsize() if audio_q is not None else -1
                        event_depth = persona_event_q.qsize()
                        profile_stats["max_frame_queue"] = max(
                            int(profile_stats["max_frame_queue"] or 0), q_depth
                        )
                        profile_stats["max_audio_queue"] = max(
                            int(profile_stats["max_audio_queue"] or 0), audio_depth
                        )

                        print(
                            f"[TRY13][chunk#{reply_avatar_chunk_idx}] "
                            f"moshi={moshi_total_ms:.0f}ms "
                            f"helium={float(fm_info['helium_ms']):.0f}ms "
                            f"fm={float(fm_info['fm_ms']):.0f}ms "
                            f"render={render_ms:.0f}ms "
                            f"jpeg={jpeg_ms:.0f}ms "
                            f"publish_wall={chunk_wall_ms:.0f}ms "
                            f"frames={n_frames} "
                            f"event_age={oldest_event_age_ms:.0f}ms "
                            f"produce_latency={produce_latency_ms:.0f}ms "
                            f"frame_q={q_depth} audio_q={audio_depth} "
                            f"event_q={event_depth} "
                            f"native_idle={int(fm_info.get('native_idle_steps', 0))} "
                            f"abs={emitted}",
                            flush=True,
                        )

                fut_done = asyncio.run_coroutine_threadsafe(frame_q.put(None), event_loop)
                try:
                    fut_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING sentinel put: {e!r}", flush=True)
                fut_audio_done = asyncio.run_coroutine_threadsafe(
                    audio_q.put(None),
                    event_loop,
                )
                try:
                    fut_audio_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING audio sentinel put: {e!r}", flush=True)

            async def _get_media_epoch() -> float:
                nonlocal media_epoch
                while not prebuffer_ready.is_set():
                    await asyncio.sleep(0.01)
                async with media_epoch_lock:
                    session = split_sessions.get(session_id)
                    if media_epoch is None and session is not None and session.get("media_epoch") is not None:
                        media_epoch = float(session["media_epoch"])
                    if media_epoch is None:
                        media_epoch = time.perf_counter() + 0.08
                        if session is not None:
                            session["media_epoch"] = media_epoch
                    return media_epoch

            async def _reply_sender() -> None:
                """Wait for prebuffer, then drain frame_q at 25fps."""
                assert frame_q is not None
                send_start_wall = await _get_media_epoch()
                frames_sent = 0
                starvation_events = 0
                starve_start: float | None = None
                ws_closed = False

                q_depth_at_start = frame_q.qsize()
                print(
                    f"[SENDER] prebuffer filled, starting pacing with "
                    f"{q_depth_at_start} frames queued",
                    flush=True,
                )

                while True:
                    if ws_closed:
                        break

                    try:
                        packet = frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        if send_start_wall is not None and starve_start is None:
                            starve_start = time.perf_counter()
                            starvation_events += 1
                        await asyncio.sleep(0.004)
                        continue

                    if packet is None:
                        break

                    if starve_start is not None:
                        gap_ms = 1000.0 * (time.perf_counter() - starve_start)
                        if gap_ms > 100:
                            print(
                                f"[SENDER] STARVED {gap_ms:.0f}ms "
                                f"(event #{starvation_events}) "
                                f"frame_q={frame_q.qsize()} sent={frames_sent}",
                                flush=True,
                            )
                        starve_start = None

                    idx = int(packet["frame_number"])

                    target_t = send_start_wall + idx / float(args.fps)
                    sleep_s = target_t - time.perf_counter()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                    try:
                        async with ws_send_lock:
                            if packet.get("ws_kind") == "bytes":
                                await ws.send_bytes(packet["data"])
                            else:
                                await ws.send_json(packet["msg"])
                    except (WebSocketDisconnect, RuntimeError, Exception):
                        ws_closed = True
                        break
                    frames_sent += 1

                    late_s = time.perf_counter() - target_t
                    if late_s > 0.5:
                        print(
                            f"[SENDER] video frame {idx} is {late_s*1000:.0f}ms late",
                            flush=True,
                        )

                    if frames_sent % 50 == 0:
                        q_depth = frame_q.qsize()
                        elapsed = time.perf_counter() - send_start_wall
                        print(
                            f"[SENDER] sent={frames_sent} frame={idx} "
                            f"frame_q={q_depth} "
                            f"elapsed={elapsed:.1f}s "
                            f"starve_events={starvation_events}",
                            flush=True,
                        )

            async def _audio_sender() -> None:
                assert audio_q is not None
                output_audio_codec = str(args.output_audio_codec).lower()
                if output_audio_codec != "opus":
                    raise RuntimeError("AJ requires --output_audio_codec opus")
                opus_writer = sphn.OpusStreamWriter(TARGET_SR)
                send_start_wall = await _get_media_epoch()
                packets_sent = 0
                bytes_sent = 0
                max_queue_depth = 0
                max_lateness_ms = 0.0
                last_send_wall: float | None = None
                # AG: prevent catch-up bursts after GPU/render stalls.
                # Even if target_t is already late, keep websocket audio writes
                # spaced near the media frame cadence instead of flushing a
                # backlog back-to-back into the browser decoder/worklet.
                # PersonaPlex emits one native Mimi audio step every 80 ms.
                native_packet_sec = float(MIMI_FRAME_SIZE) / float(TARGET_SR)
                min_send_interval_s = max(0.001, native_packet_sec - 0.005)
                next_send_wall = send_start_wall
                active_generation = 0
                base_audio_idx: int | None = None

                while True:
                    packet = await audio_q.get()
                    if packet is None:
                        break
                    session = split_sessions.get(session_id)
                    if session is None: break
                    packet_generation = int(packet.get("media_generation", 0))
                    current_generation = _media_generation(session)
                    if packet_generation != current_generation: continue
                    idx = int(packet["audio_packet_index"])
                    if packet_generation != active_generation or base_audio_idx is None:
                        active_generation = packet_generation
                        base_audio_idx = idx
                        send_start_wall = await _get_media_epoch()
                        next_send_wall = send_start_wall
                        last_send_wall = None
                        opus_writer = sphn.OpusStreamWriter(TARGET_SR)
                        print(f"[MEDIA-EPOCH][AUDIO] generation={active_generation} base_audio={base_audio_idx}", flush=True)
                    target_t = send_start_wall + (idx - base_audio_idx) * native_packet_sec
                    target_t = max(target_t, next_send_wall)
                    sleep_s = target_t - time.perf_counter()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    session = split_sessions.get(session_id)
                    if session is None or packet_generation != _media_generation(session):
                        continue
                    send_wall = time.perf_counter()
                    lateness_ms = max(0.0, (send_wall - target_t) * 1000.0)
                    max_lateness_ms = max(max_lateness_ms, lateness_ms)
                    interval_ms = (
                        0.0 if last_send_wall is None
                        else (send_wall - last_send_wall) * 1000.0
                    )
                    last_send_wall = send_wall
                    next_send_wall = max(target_t, send_wall) + min_send_interval_s
                    max_queue_depth = max(max_queue_depth, audio_q.qsize())

                    audio_pcm = np.asarray(
                        packet["audio_pcm"],
                        dtype=np.float32,
                    ).reshape(-1)
                    packet_rms = (
                        float(np.sqrt(np.mean(np.square(audio_pcm, dtype=np.float32))))
                        if audio_pcm.size else 0.0
                    )
                    if packet_rms >= 0.006:
                        playback_state["active"] = True
                        playback_state["hold"] = 4
                    elif int(playback_state["hold"]) > 0:
                        playback_state["hold"] = int(playback_state["hold"]) - 1
                        playback_state["active"] = True
                    else:
                        playback_state["active"] = False
                    if _REPLY_GAIN != 1.0:
                        audio_pcm = np.clip(audio_pcm * _REPLY_GAIN, -1.0, 1.0)
                    opus_payload = opus_writer.append_pcm(audio_pcm)
                    if opus_payload is None and hasattr(opus_writer, "read_bytes"):
                        opus_payload = opus_writer.read_bytes()
                    if opus_payload:
                        try:
                            async with ws_send_lock:
                                await ws.send_bytes(b"\x01" + opus_payload)
                        except (WebSocketDisconnect, RuntimeError, Exception):
                            break
                        packets_sent += 1
                        bytes_sent += len(opus_payload)

                    if packets_sent and packets_sent % 50 == 0:
                        print(
                            f"[NATIVE-AUDIO] sent={packets_sent} seq={idx} "
                            f"audio_q={audio_q.qsize()} max_q={max_queue_depth} "
                            f"interval={interval_ms:.1f}ms "
                            f"late={lateness_ms:.1f}ms max_late={max_lateness_ms:.1f}ms "
                            f"bytes={bytes_sent}",
                            flush=True,
                        )

            mic_ingest_thread = threading.Thread(
                target=_mic_ingest_worker,
                daemon=True,
                name="mic-ingest",
            )
            mic_ingest_thread.start()
            persona_thread = threading.Thread(
                target=_persona_priority_worker,
                daemon=True,
                name="persona-priority",
            )
            persona_thread.start()
            gpu_thread = threading.Thread(target=_gpu_producer_thread, daemon=True, name="gpu-producer")
            gpu_thread.start()
            print(
                f"[AJ][AUDIO] session={session_id[:8]} "
                f"video_path=/ws/video?session_id={session_id}",
                flush=True,
            )
            audio_sender_task = asyncio.create_task(_audio_sender())

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is not None:
                    if reply_engine is None:
                        continue
                    assert mic_q is not None
                    packet_sr = int(browser_input_sr)
                    if len(data) > 0 and data[0] == 3:
                        # Offline A/B harness: exact decoded 24 kHz PCM from a
                        # prior live session. Browser traffic never uses kind 3.
                        data = bytes(data[1:])
                        packet_sr = TARGET_SR
                    elif len(data) > 0 and data[0] == 1:
                        # sphn API differs by build: newer returns the PCM from
                        # append_bytes; older (Jetson) returns None and requires a
                        # separate read_pcm(). Support both.
                        decoded_pcm = opus_reader.append_bytes(bytes(data[1:]))
                        if decoded_pcm is None:
                            decoded_pcm = opus_reader.read_pcm()
                        if decoded_pcm is None or decoded_pcm.shape[-1] == 0:
                            continue
                        decoded_pcm = np.asarray(decoded_pcm, dtype=np.float32).reshape(-1)
                        decoded_i16 = (
                            np.clip(decoded_pcm, -1.0, 1.0) * 32767.0
                        ).astype(np.int16)
                        data = decoded_i16.tobytes()
                        packet_sr = TARGET_SR
                    audio_packets_seen += 1
                    profile_stats["mic_packets"] = audio_packets_seen
                    pcm_i16 = np.frombuffer(data, dtype=np.int16)
                    mic_rms = float(np.sqrt(np.mean((pcm_i16.astype(np.float32) / 32768.0) ** 2))) if pcm_i16.size else 0.0
                    mic_peak = float(np.max(np.abs(pcm_i16.astype(np.float32) / 32768.0))) if pcm_i16.size else 0.0
                    with media_generation_lock:
                        capture_generation = int(
                            split_sessions.get(session_id, {}).get("media_generation", 0)
                        )
                    _capture_mic_packet(
                        pcm_i16,
                        packet_seq=audio_packets_seen,
                        sample_rate=packet_sr,
                        mic_rms=mic_rms,
                        mic_peak=mic_peak,
                        media_generation_value=capture_generation,
                    )
                    if mic_rms >= 0.020:
                        profile_stats["quiet_packets"] = 0
                        profile_stats["user_turn_peak_rms"] = max(
                            float(profile_stats["user_turn_peak_rms"] or 0.0),
                            mic_rms,
                        )
                        if profile_stats["user_turn_started_at"] is None:
                            turn_id = int(profile_stats["user_turns"] or 0) + 1
                            profile_stats["user_turns"] = turn_id
                            profile_stats["user_turn_started_at"] = time.perf_counter()
                            profile_stats["user_turn_peak_rms"] = mic_rms
                            print(
                                f"[TRY13][USER-TURN-START] turn={turn_id} "
                                f"packet={audio_packets_seen} rms={mic_rms:.5f}",
                                flush=True,
                            )
                    elif profile_stats["user_turn_started_at"] is not None:
                        quiet_packets = int(profile_stats["quiet_packets"] or 0) + 1
                        profile_stats["quiet_packets"] = quiet_packets
                        if quiet_packets >= 8:
                            turn_id = int(profile_stats["user_turns"] or 0)
                            turn_started_at = float(profile_stats["user_turn_started_at"])
                            print(
                                f"[TRY13][USER-TURN-END] turn={turn_id} "
                                f"duration_ms={(time.perf_counter() - turn_started_at) * 1000.0:.1f} "
                                f"peak_rms={float(profile_stats['user_turn_peak_rms'] or 0.0):.5f}",
                                flush=True,
                            )
                            profile_stats["user_turn_started_at"] = None
                            profile_stats["user_turn_peak_rms"] = 0.0
                            profile_stats["quiet_packets"] = 0
                    now_wall = time.perf_counter()
                    if audio_packets_seen <= 3 or now_wall - last_mic_level_log_wall >= 1.0:
                        voice = "VOICE" if mic_rms >= 0.02 else "quiet"
                        print(
                            f"[MIC] packet={audio_packets_seen} rms={mic_rms:.5f} "
                            f"peak={mic_peak:.3f} sr={packet_sr} {voice}",
                            flush=True,
                        )
                        last_mic_level_log_wall = now_wall
                    if audio_packets_seen <= 3:
                        print(
                            f"[liveTryHeliumFM] rx binary mic packet#{audio_packets_seen} "
                            f"bytes={len(data)} sr={packet_sr}",
                            flush=True,
                        )
                    if not session_started.is_set():
                        session_started.set()
                        print(
                            "[liveTryHeliumFM] auto-started session from first binary mic packet",
                            flush=True,
                        )
                    try:
                        mic_q.put_nowait(
                            (bytes(data), int(packet_sr), audio_packets_seen, time.perf_counter())
                        )
                        profile_stats["max_mic_queue"] = max(
                            int(profile_stats["max_mic_queue"] or 0),
                            mic_q.qsize(),
                        )
                    except queue.Full:
                        # The queue is intentionally unbounded. If this ever
                        # triggers, fail loudly rather than losing speech invisibly.
                        print(
                            f"[MIC-NODROP] ERROR unexpected queue.Full "
                            f"packet={audio_packets_seen}",
                            flush=True,
                        )
                    continue
                text = msg.get("text")
                if text is None:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                msg_type = str(payload.get("type", "")).lower()
                if msg_type != "display_progress":
                    print(f"[liveTryHeliumFM] rx text type={msg_type or '<empty>'}", flush=True)

                if msg_type == "start":
                    _save_capture("replaced_by_new_start")
                    _capture_reset()
                    reset_started_at = time.perf_counter()
                    _capture_marker("model_reset_begin")
                    session_started.clear()
                    playback_state["active"] = False
                    playback_state["hold"] = 0
                    browser_input_sr = int(payload.get("sample_rate", payload.get("sampleRate", browser_input_sr)))
                    # A Start message is a hard session boundary. Remove any
                    # transport data that arrived before Start and recreate the
                    # stateful Opus decoder before resetting model/avatar state.
                    cleared_mic_packets = 0
                    if mic_q is not None:
                        while True:
                            try:
                                stale_item = mic_q.get_nowait()
                            except queue.Empty:
                                break
                            if stale_item is not None:
                                cleared_mic_packets += 1
                    cleared_persona_events = 0
                    while True:
                        try:
                            persona_event_q.get_nowait()
                            cleared_persona_events += 1
                        except queue.Empty:
                            break
                    opus_reader = sphn.OpusStreamReader(TARGET_SR)
                    fm_engine.reset_session()
                    if seedvc is not None:
                        seedvc.reset()
                    if fm_engine.audio_pcm is not None and (stream_task is None or stream_task.done()):
                        stream_task = asyncio.create_task(stream_from_file(ws, fm_engine))
                    if reply_engine is not None:
                        with reply_input_lock:
                            reply_engine.reset_session()
                        session_started.set()
                    reset_done_at = time.perf_counter()
                    profile_stats["start_ready_at"] = reset_done_at
                    await ws.send_json({
                        "type": "session_started",
                        "session_id": session_id,
                        "capture_id": str(capture_state.get("capture_id") or ""),
                        "reset_ms": round(
                            (reset_done_at - reset_started_at) * 1000.0,
                            1,
                        ),
                    })
                    _capture_marker(
                        "session_started",
                        reset_ms=round((reset_done_at - reset_started_at) * 1000.0, 1),
                    )
                    print(
                        f"[SESSION-FULL-RESET] session={session_id[:8]} "
                        f"cleared_mic_packets={cleared_mic_packets} "
                        f"cleared_persona_events={cleared_persona_events} "
                        "opus=reset persona=reset avatar=reset",
                        flush=True,
                    )
                    print(
                        "[liveTryHeliumFM] start → "
                        + ("streaming from file" if fm_engine.audio_pcm is not None else "live Moshi reply mode"),
                        flush=True,
                    )

                elif msg_type == "display_progress":
                    frame_number = payload.get("frame_number")
                    if frame_number is not None:
                        with displayed_frame_lock:
                            displayed_frame_state["frame_number"] = int(frame_number)

                elif msg_type == "chunk_audio":
                    pcm_b64 = payload.get("pcm_s16le_b64", "")
                    if not pcm_b64:
                        continue
                    pcm_bytes = base64.b64decode(pcm_b64)
                    result = fm_engine.feed_pcm(pcm_bytes)
                    if result is not None:
                        motion, fm_info, _pcm_chunk = result
                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        n_frames = int(motion.shape[0])
                        emitted = fm_info["abs_start"]
                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sub = motion[sb_start:sb_start + fm_engine.render_sub_batch].to(
                                fm_engine.device, dtype=fm_engine.dtype
                            )
                            frames_np, _ = fm_engine._render_motion(sub)
                            for j, frame_rgb in enumerate(frames_np):
                                idx = emitted + sb_start + j
                                await ws.send_json({
                                    "type": "chunk_frame",
                                    "chunk_id": idx + 1,
                                    "frame_idx": 0,
                                    "jpeg_b64": encode_jpeg_b64(frame_rgb, fm_engine.jpeg_quality),
                                    "moshi_text": (
                                        f"live Helium+FM "
                                        f"helium={fm_info['helium_ms']:.0f}ms "
                                        f"fm={fm_info['fm_ms']:.0f}ms"
                                    ),
                                    "server_fps": round(float(args.fps), 1),
                                    "chunks_done": avatar_chunk_id,
                                })

                elif msg_type == "stop":
                    _capture_marker("stop_requested")
                    print("[liveTryHeliumFM] stop requested", flush=True)
                    break

        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            session = split_sessions.get(session_id)
            if session is not None:
                session["closed"] = True
            # Signal every worker before joining it. Previously teardown joined
            # workers whose loops were still active, so a rapid second Start
            # waited forever and stale PersonaPlex output survived the boundary.
            if reply_engine is not None:
                pipeline_stop_event.set()
                session_started.clear()
            if stream_task is not None:
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            if mic_q is not None:
                with contextlib.suppress(queue.Full):
                    mic_q.put_nowait(None)
            if mic_ingest_thread is not None:
                await asyncio.to_thread(mic_ingest_thread.join, 5.0)
            if persona_thread is not None:
                await asyncio.to_thread(persona_thread.join, 10.0)
            if gpu_thread is not None:
                await asyncio.to_thread(gpu_thread.join, 30.0)
            if sender_task is not None:
                sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await sender_task
            if audio_sender_task is not None:
                audio_sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await audio_sender_task
            _capture_marker("workers_stopped")
            _save_capture("stop_or_disconnect")
            fm_engine.dump_last_session(source="websocket_live")
            session_duration = max(
                0.0,
                time.perf_counter() - float(profile_stats["connected_at"] or time.perf_counter()),
            )
            print(
                "[TRY13][SESSION-SUMMARY] "
                f"duration_s={session_duration:.1f} "
                f"mic_packets={int(profile_stats['mic_packets'] or 0)} "
                f"user_turns={int(profile_stats['user_turns'] or 0)} "
                f"media_resets={int(profile_stats['media_resets'] or 0)} "
                f"candidate_expirations={int(profile_stats['candidate_expirations'] or 0)} "
                f"max_mic_q={int(profile_stats['max_mic_queue'] or 0)} "
                f"max_event_q={int(profile_stats['max_event_queue'] or 0)} "
                f"max_frame_q={int(profile_stats['max_frame_queue'] or 0)} "
                f"max_audio_q={int(profile_stats['max_audio_queue'] or 0)} "
                f"max_event_age_ms={float(profile_stats['max_event_age_ms'] or 0.0):.1f}",
                flush=True,
            )
            split_sessions.pop(session_id, None)
            if conversation_lock.locked():
                conversation_lock.release()
                print("[SESSION-STRICT] released PersonaPlex session lock", flush=True)
            print("[liveTryHeliumFM] websocket closed", flush=True)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = LiveHeliumFMOptions()
    args = parser.parse()
    args.rank = args.device
    parser.print_options()

    app = build_app(args)

    import uvicorn

    print(f"[liveTryHeliumFM_ws_binary] serving {args.html_path} (binary av_transport)")
    print(f"[liveTryHeliumFM] open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
