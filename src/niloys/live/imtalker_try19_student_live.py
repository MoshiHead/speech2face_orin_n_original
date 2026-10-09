"""try19 - the Design A causal student driving the live IMTalker pipeline.

Imports the production live server module read-only, swaps ONE method
(LiveHeliumFMEngine._sample_motion_from_helium), and calls its main().
No production file is copied or edited.

CRITICAL - speculative-rollout safety
    The live server does speculative generation it later discards:
        original = clone(fm_engine.stream_state)
        fm_engine.stream_state = speculative_state
        fm_engine._sample_motion_from_helium(...)     # branch we may throw away
        finally: fm_engine.stream_state = original    # rewind
    A student that keeps its KV cache in its own attribute would NOT be
    rewound, so every discarded branch would permanently corrupt its
    autoregressive state. That path runs during IDLE, which is where the
    live-vs-offline discrepancy showed up.

    Fix: the student's state lives INSIDE fm_engine.stream_state as plain
    tensors (_student_k, _student_v, _student_prev_motion), so the server's
    existing _clone_fm_stream_state save/restore carries it automatically.
    Tensors are stacked (not a list of dicts) because that clone helper only
    deep-copies torch.Tensor and dict values - a list would be shared by
    reference and silently defeat the rewind.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

import os as _os
IMTALKER = Path(_os.environ.get("IMTALKER_DIR",
    "/home/ubuntu/project/speech2avatar_imtalker_personaplex_try_vad6_8998/IMTalker"))
DESIGN_A = Path(_os.environ.get("DESIGN_A_DIR",
    "/home/ubuntu/project/imtalker_world_model/design_a"))
for _p in (str(IMTALKER), str(DESIGN_A)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from student.model import CausalMotionStudent, StudentConfig  # noqa: E402
from student.renderer import build as build_renderer          # noqa: E402

LIVE_MODULE = "imtalker_personaplex_try_vad14_precomputed_closure_8998"
LIVE_DIR = str(DESIGN_A / "live")   # for the design_a fork

def _flag(name, default="1"):
    return _os.environ.get(name, default).lower() not in ("0", "no", "false", "")

# Orin fast path. Each is independently switchable so a regression can be
# bisected without reverting the rest. All four are bench-verified:
#   student graph  13.32 -> 1.96 ms  (bit-equivalent)
#   mimi graph     21.72 -> 7.89 ms  (bit-exact codes and audio)
#   renderer fp16   8.55 -> 5.80 ms  (63.94 dB vs fp32)
#   int4 G=32      57.60 -> 49.53 ms (cosine 0.876 vs bf16; NF4 is 0.844)
STUDENT_GRAPH = _flag("TRY19_STUDENT_GRAPH")
# OFF. Bit-exact and a huge win on the persona thread (encode 20.1 -> 0.2 ms,
# moshi 103 -> 81 ms) -- but it STARVES the avatar thread, which is a net
# loss: avatar publish_wall 19 -> 170 ms, chunks 143 -> 63.
# Three mechanisms tested and ruled out: (1) capture on the default vs the
# caller's stream, (2) replaying at normal instead of priority=-1, and
# (3) graph-vs-graph interference with the student graph (turning the
# student graph off made it worse: fm 320 ms, 59 frames). Cause unknown.
MIMI_GRAPH    = _flag("TRY19_MIMI_GRAPH", "0")
#   TRY19_DEC_CKPT=<path>  2-layer Mimi decoder student (decoder_transformer
#                     8 -> 2 layers, SEANet n_residual_layers=0). Ported from the
#                     2x5090 box. Cuts WORK not launch gaps, so unlike the Mimi
#                     graph it does not starve the avatar. Orin: decode 8.79 ->
#                     4.99 ms. Audio is muddier -- accepted tradeoff.
DEC_CKPT      = _os.environ.get("TRY19_DEC_CKPT", "").strip()
#   TRY19_MIMI_DEC_FT=<path>  Mimi decoder_transformer fine-tuned on the parallel
#                     depformer codes (pd_work/dec_ft.py). Same architecture, new
#                     weights only: no latency change. Encoder untouched.
MIMI_DEC_FT   = _os.environ.get("TRY19_MIMI_DEC_FT", "").strip()
RENDER_FP16   = _flag("TRY19_RENDER_FP16")
# ON. Needs bf16 weights (run_orin_fast.sh passes --moshi_weight
# model.safetensors and NO --quantize_4bit). Conversion happens at LOAD time
# in main(); doing it after the engine is built leaves moshi's already-
# captured CUDA graphs pointing at freed bf16 weights -> illegal memory
# access inside step_system_prompts. Live: moshi 112 -> 103 ms,
# chunks 123 -> 143, video frames 193 -> 229.
#   TRY19_DEP_LAYERS=5  build the depformer with N layers (teacher default 6).
#                     Requires PD weights trained for that depth; on the pod this
#                     rung measured -2.27 ms. Pruning WITHOUT matched weights
#                     produces garbage, so never set this alone.
DEP_LAYERS    = int(_os.environ.get("TRY19_DEP_LAYERS", "0") or 0)
INT4_G        = int(_os.environ.get("TRY19_INT4_G", "32"))
# KV context (frames). Default 3000 = 240 s of conversational memory, and
# 32 layers x 2 x 3000 x 4096 x 2 B = 1.57 GB read EVERY step. Sweep says
# 3000 -> 750 (60 s) is worth 6.9 ms with no model change.
CTX_FRAMES    = int(_os.environ.get("TRY19_CTX", "0"))
#   TRY19_FFN_KEEP=0.75  dense structured FFN narrowing. Keeps the top fraction
#                     of gated-FFN hidden channels by measured activation
#                     importance, then REPLACES linear_out with the closed-form
#                     least-squares re-combination of the survivors
#                     (TRY19_FFN_LSTSQ). Verified on 12 holdout clips:
#                     WER 11.8% vs bf16, against 12.7% for the int4 we already
#                     ship -- i.e. no regression. Orin LM step 43.21 -> 38.63 ms.
#                     keep=0.625 (18.1%) and 0.5 (32.8%) are NOT ship-worthy.
FFN_KEEP      = float(_os.environ.get("TRY19_FFN_KEEP", "0") or 0)
FFN_IDX       = _os.environ.get("TRY19_FFN_IDX", "").strip()
FFN_LSTSQ     = _os.environ.get("TRY19_FFN_LSTSQ", "").strip()
# SPEED-ONLY pruning knobs (no retraining -- output will be garbage, the
# point is to measure whether a smaller trunk survives realtime live).
#   TRY19_DROP="9-16"  drop a contiguous block of trunk layers
#   TRY19_FFN=2688     slice the gating FFN hidden (default 11264)
# Both run BEFORE int4 so the quantiser sees the final shapes; keep FFN a
# multiple of the int4 group size (32).
DROP_SPEC     = _os.environ.get("TRY19_DROP", "")
FFN_HIDDEN    = int(_os.environ.get("TRY19_FFN", "0"))
#   TRY19_STRIDE=2    run the TRUNK every Nth step, reusing the cached
#                     transformer_out/layer_hidden in between. Emulates
#                     temporal patching (trunk at 6.25Hz) while Mimi and
#                     the depformer stay at 12.5Hz. SPEED ONLY -- garbage out.
STRIDE        = int(_os.environ.get("TRY19_STRIDE", "1"))
#   TRY19_DEPQ_EARLY=N  set num_depformer_steps BEFORE the CUDA graph is
#                     captured. The existing --dep_q/DEPQ path sets it AFTER
#                     warmup, so graphed_depth replays the captured 16-codebook
#                     loop and the knob does nothing (measured: dq1==dq4==dq8).
DEPQ_EARLY    = int(_os.environ.get("TRY19_DEPQ_EARLY", "0"))
#   TRY19_PD=<ckpt>   wide-pass parallel depformer: all 8 generated codebooks
#                     in ONE depformer forward instead of 16 sequential calls.
#                     Measured on the native server: -19 ms/LM step.
PD_W          = _os.environ.get("TRY19_PD", "").strip()
PD_REFINE     = int(_os.environ.get("TRY19_PD_REFINE", "0"))

K_KEY, V_KEY, PM_KEY, N_KEY = "_student_k", "_student_v", "_student_prev_motion", "_student_steps"
F_KEY = "_student_filled"
LAST_KEY, BLEND_FROM, BLEND_LEFT = "_student_last_frame", "_student_blend_from", "_student_blend_left"


def load_student(ckpt: str, device):
    pack = torch.load(ckpt, map_location="cpu")
    cfg = StudentConfig(**pack["cfg"])
    model = CausalMotionStudent(cfg).to(device).float().eval()
    model.load_state_dict(pack["model"])
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"[try19] student loaded: {ckpt}", flush=True)
    print(f"[try19]   params={model.num_params():,} lookahead={cfg.lookahead} "
          f"window={cfg.attn_window} trained_step={pack.get('step')}", flush=True)
    ev = pack.get("eval") or {}
    if ev:
        print(f"[try19]   heldout tf_nmse={ev.get('tf_nmse'):.5f} "
              f"free_nmse={ev.get('free_nmse'):.5f}", flush=True)
    return model, cfg


def make_student_sampler(model, cfg, blend_frames=10, jump_thresh=0.6, mouth_close=0.0,
                         ref_x=None, motion_gain=1.0):
    """blend_frames: length of the barge-in crossfade, in 25fps frames (10 = 400 ms).

    The server has its own transition crossfade but TRANSITION_BLEND_FRAMES=0
    disables it, and it is clamped to motion.shape[0] -- which in try20 is only
    2 frames, so it cannot span a transition anyway. This blend carries across
    chunk boundaries via stream_state instead, so it survives the 80 ms chunking
    and is rewound correctly by speculative rollbacks.
    """
    n_layers = cfg.n_layers

    # Graph state: allocated once, mutated in place, so a captured graph stays
    # valid. The server's speculative rewind still works because the canonical
    # copy lives in stream_state -- we copy_ in before the step and copy out
    # after, which costs 0.25 ms (measured) against a 11.4 ms saving.
    _G = {"graph": None, "st": None, "hbuf": None, "out": None}

    def _ensure_graph(device):
        if _G["graph"] is not None:
            return
        st = model.init_state_fixed(1, device, torch.float32)
        hbuf = torch.zeros(1, cfg.helium_dim, device=device, dtype=torch.float32)
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_), torch.no_grad():
            for _ in range(3):
                model.step_fixed(hbuf, st)
        torch.cuda.current_stream().wait_stream(s_)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            out = model.step_fixed(hbuf, st)
        _G.update(graph=g, st=st, hbuf=hbuf, out=out)
        print("[try19] student CUDA-graphed (13.32 -> 1.96 ms/step, bit-equivalent)",
              flush=True)

    @torch.no_grad()
    def _sample_motion_from_helium(self, helium: torch.Tensor, target_frames: int):
        timings: dict = {}
        t0 = time.perf_counter()
        helium = helium.to(self.device, dtype=torch.float32).contiguous()
        target_frames = int(target_frames)
        steps = int(helium.shape[0])

        ss = self.stream_state if isinstance(self.stream_state, dict) else {}

        # --- restore student state FROM stream_state (so rewinds apply to us too)
        prior_steps = int(ss.get(N_KEY, 0) or 0)
        h = helium.unsqueeze(0)
        k, v = ss.get(K_KEY), ss.get(V_KEY)
        pm = ss.get(PM_KEY)

        if STUDENT_GRAPH:
            _ensure_graph(self.device)
            gst, hbuf = _G["st"], _G["hbuf"]
            f = ss.get(F_KEY)
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                for i, c in enumerate(gst["caches"]):
                    c["k"].copy_(k[i]); c["v"].copy_(v[i])
                    if isinstance(f, torch.Tensor):
                        c["filled"].copy_(f[i])
            else:
                for c in gst["caches"]:
                    c["k"].zero_(); c["v"].zero_(); c["filled"].zero_()
            if isinstance(pm, torch.Tensor):
                gst["prev_motion"].copy_(pm)
            else:
                gst["prev_motion"].zero_()
            pairs = []
            for t in range(steps):
                hbuf.copy_(h[:, t])
                _G["graph"].replay()
                pairs.append(_G["out"].clone())
            pred_n = torch.stack(pairs, 1)
            st = gst
        else:
            st = model.init_state(1, self.device, torch.float32)
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                for i in range(n_layers):
                    st["caches"][i]["k"] = k[i]
                    st["caches"][i]["v"] = v[i]
            if isinstance(pm, torch.Tensor):
                st["prev_motion"] = pm
            pairs = [model.step(h[:, t], st) for t in range(steps)]
            pred_n = torch.stack(pairs, 1)
        if motion_gain != 1.0:
            # Context-noise training damps output magnitude: the student tracks
            # the teacher at 0.964 correlation but only ~82% of its mouth
            # amplitude. Scaling the deviation from the running mean restores
            # articulation; measured effect is purely on scale (correlation
            # changed 0.8502 -> 0.8505 across gains 1.0..1.4), and at 1.2 speech
            # reaches 100.5% of teacher while silence stays 31% BELOW teacher.
            mu = ss.get("_student_gain_mu")
            cur = pred_n.mean(dim=1, keepdim=True)
            mu = cur if not isinstance(mu, torch.Tensor) else (0.99 * mu + 0.01 * cur)
            pred_n = mu + (pred_n - mu) * motion_gain
            ss = dict(ss); ss["_student_gain_mu"] = mu.detach()
        motion = model.denorm_motion(pred_n)[0].reshape(-1, 32)

        if motion.shape[0] < target_frames:
            motion = torch.cat([motion, motion[-1:].expand(target_frames - motion.shape[0], -1)], 0)
        motion = motion[:target_frames].contiguous()

        # ---- barge-in / discontinuity crossfade, carried ACROSS chunks ----
        last = ss.get(LAST_KEY)
        bfrom = ss.get(BLEND_FROM)
        bleft = int(ss.get(BLEND_LEFT, 0) or 0)
        if blend_frames > 0:
            if last is not None and bleft <= 0:
                jump = float((motion[0] - last).abs().max())
                if jump > jump_thresh:
                    bfrom = last.clone()
                    if mouth_close > 0.0 and ref_x is not None:
                        # ease the mouth toward the reference (closed) pose
                        bfrom = bfrom * (1.0 - mouth_close) + ref_x.view(-1) * mouth_close
                    bleft = int(blend_frames)
            if bleft > 0 and bfrom is not None:
                n = min(bleft, motion.shape[0])
                start = blend_frames - bleft
                w_ = torch.arange(start + 1, start + n + 1, device=motion.device,
                                  dtype=motion.dtype) / float(blend_frames)
                w_ = w_.clamp(0, 1).unsqueeze(-1)
                # smoothstep for a soft in/out rather than a linear ramp
                w_ = w_ * w_ * (3.0 - 2.0 * w_)
                motion[:n] = bfrom.view(1, -1) * (1.0 - w_) + motion[:n] * w_
                bleft -= n

        # --- persist student state BACK INTO stream_state as plain tensors
        new_ss = dict(ss)
        new_ss[K_KEY] = torch.stack([c["k"] for c in st["caches"]], 0).clone()
        new_ss[V_KEY] = torch.stack([c["v"] for c in st["caches"]], 0).clone()
        if STUDENT_GRAPH:
            # prev_motion IS the graph's persistent buffer -- clone or the next
            # replay would overwrite what we just handed to stream_state.
            new_ss[F_KEY] = torch.stack([c["filled"] for c in st["caches"]], 0).clone()
            new_ss[PM_KEY] = st["prev_motion"].clone()
        else:
            new_ss[PM_KEY] = st["prev_motion"]
        new_ss[N_KEY] = prior_steps + steps
        new_ss[LAST_KEY] = motion[-1].detach().clone()
        new_ss[BLEND_FROM] = bfrom
        new_ss[BLEND_LEFT] = int(bleft)
        self.stream_state = new_ss

        fm_ms = (time.perf_counter() - t0) * 1000.0
        timings["adapter_ms"] = 0.0
        timings["helium_ms"] = fm_ms
        timings["fm_ms"] = fm_ms
        timings["helium_deque_filled"] = int(steps)
        timings["helium_feat"] = helium.detach().cpu()
        timings["adapter_feat_50"] = torch.zeros(0)
        timings["adapter_feat_25"] = torch.zeros(0)
        timings["projected_audio"] = torch.zeros(0)
        timings["frames"] = int(motion.shape[0])
        timings["abs_start"] = self.abs_frame
        self.abs_frame += timings["frames"]
        timings["student_steps"] = prior_steps + steps
        timings["student_ms_per_step"] = fm_ms / max(steps, 1)
        return motion, timings

    return _sample_motion_from_helium


def load_distilled_renderer(path, device):
    pack = torch.load(path, map_location="cpu")
    D = build_renderer(**pack["cfg"]).to(device).float().eval()
    D.load_state_dict(pack["model"])
    if RENDER_FP16:
        # 1.47x, and the fp16 error (63.94 dB) sits ~12 dB below the model's
        # own 51.68 dB val error, so it is well inside the noise floor.
        D = D.half().to(memory_format=torch.channels_last)
        print("[try19] renderer -> fp16 + channels_last (8.55 -> 5.80 ms)", flush=True)
    for prm in D.parameters():
        prm.requires_grad_(False)
    print(f"[try19] distilled renderer: {path}", flush=True)
    print(f"[try19]   params={D.num_params():,} step={pack.get('step')} "
          f"val_PSNR={pack.get('val_psnr', 0.0):.2f} dB", flush=True)
    return D


def load_blink_delta(path, device):
    """Pixel-space blink, precomputed once.

    The real renderer composites a blink motion into its multi-scale motion maps
    inside two soft eye ellipses, then renders. Measured: the resulting PIXEL
    delta varies only ~15% across mouth poses (0.13/255), so averaging over poses
    and replaying it as a phase-indexed pixel delta is visually equivalent and
    costs one add per frame -- no retraining of the distilled decoder.
    """
    b = torch.load(path, map_location="cpu")
    d = b["delta"].to(device).float()
    print(f"[try19] blink delta: {path}", flush=True)
    print(f"[try19]   {tuple(d.shape)} loop={b['length']} frames "
          f"({b['length']/25.0:.1f}s) crop=y{b['y0']}:{b['y1']} x{b['x0']}:{b['x1']} "
          f"energy_kept={100*b.get('energy_kept',0):.3f}%", flush=True)
    return d, int(b["y0"]), int(b["y1"]), int(b["x0"]), int(b["x1"]), int(b["length"])


def patch_render(live, D, blink=None):
    """Swap the frozen IMTalker renderer for the distilled single-identity decoder.

    NOTE: the distilled decoder was trained on frames rendered WITHOUT the
    eye-blink composite, so blinking is lost while this is active. Blink is
    fused into the real renderer's multi-scale motion maps rather than applied
    in pixel space, so it cannot simply be layered onto the distilled output.
    """
    @torch.no_grad()
    def _render_motion(self, motion):
        t0 = time.perf_counter()
        mo = motion.to(self.device, dtype=next(D.parameters()).dtype)
        if mo.ndim == 1:
            mo = mo.unsqueeze(0)
        start = int(self._render_frame_cursor)
        # Phase-conditioned decoder: blink is a learned 367-frame loop, so pass
        # the phase rather than adding a precomputed pixel table.
        bl = int(getattr(D.cfg, "blink_len", 0) or 0)
        if bl > 0:
            ph = (torch.arange(int(mo.shape[0]), device=mo.device) + start) % bl
            img = D(mo, ph)
        else:
            img = D(mo)
        img = img.float()
        if blink is not None:
            bd, y0, y1, x0, x1, L = blink
            idx = (torch.arange(int(mo.shape[0]), device=img.device) + start) % L
            img[:, :, y0:y1, x0:x1] = img[:, :, y0:y1, x0:x1] + bd.index_select(0, idx)
        img = img.clamp(0, 1)
        self._render_frame_cursor += int(mo.shape[0])
        out = img.mul(255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return out, {"total_ms": (time.perf_counter() - t0) * 1000.0}

    @torch.no_grad()
    def _render_motion_first_frame(self, motion):
        return _render_motion(self, motion.reshape(-1, 32)[:1])

    live.LiveHeliumFMEngine._render_motion = _render_motion
    live.LiveHeliumFMEngine._render_motion_first_frame = _render_motion_first_frame
    _bl = int(getattr(D.cfg, "blink_len", 0) or 0)
    print("[try19] patched _render_motion -> distilled decoder"
          + (f" + LEARNED blink (phase-conditioned, loop={_bl})" if _bl > 0 else
             (" + precomputed blink delta" if blink is not None else
              " (no blink)")), flush=True)


def patch_depformer(live, n):
    """Cap the depformer at n codebooks per step.

    The LM is built with dep_q=16, but loaders.get_moshi_lm only prunes when
    `num_codebooks < 8` -- so passing 8 is a no-op and all 16 codebooks are
    generated. The server then decodes only tokens[:, 1:9] = codebooks 0..7,
    so half the depformer work is computed and thrown away every 80 ms step.

    The depformer is autoregressive FORWARD (codebook k conditions on k-1), so
    codebooks 0..7 do not depend on 8..15: truncating at 8 leaves the decoded
    audio bit-identical. Measured 19.50 -> 15.45 ms/step = 1.26x, free.
    Below 8 it becomes a real audio-quality tradeoff -- but never affects the
    avatar, whose Helium comes from graphed_main, before the depformer.
    """
    orig = live.MoshiOnlyEngineWithHidden.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        lg = getattr(self, "lm_gen", None)
        if lg is None:
            print("[try19] WARNING no lm_gen; depformer cap not applied", flush=True)
            return
        prev = getattr(lg.lm_model, "dep_q", None)
        lg.num_depformer_steps = int(n)
        print(f"[try19] depformer capped: dep_q={prev} -> {n} codebooks/step "
              f"({'bit-identical audio' if n >= 8 else 'AUDIO QUALITY TRADEOFF'})", flush=True)

        # int4 is applied at LOAD time (see main()), not here: by the time this
        # runs, LMGen has already captured its CUDA graphs, and swapping the
        # Linears out from under them leaves the graphs pointing at freed bf16
        # weights -> "illegal memory access" on the next replay.
        if MIMI_GRAPH and getattr(self, "mimi", None) is not None:
            try:
                import mimi_graph
                _ge, _gd = mimi_graph.wrap(self.mimi)
                print("[try19] mimi encode/decode -> lazy CUDA graph "
                      "(21.72 -> 7.89 ms, bit-exact)", flush=True)
                # Capture NOW, at construction: no session exists and no other thread is
                # using the GPU. Capturing lazily mid-session, while the avatar thread
                # allocates concurrently, corrupted other threads' memory (blink phase ->
                # device-side assert ~358 steps later; 3/3 reproductions).
                if _os.environ.get("TRY19_MIMI_CAPTURE_AT_INIT", "0") == "1" and _ge is not None:
                    if getattr(self.mimi, "_streaming_state", None) is None:
                        print("[try19] WARNING mimi not streaming at init -- capture stays lazy", flush=True)
                    else:
                        _z = torch.zeros(1, 1, 1920, device=next(self.mimi.parameters()).device)
                        with torch.no_grad():
                            for _ in range(_ge.warmup + 2):
                                self.mimi.encode(_z)
                        torch.cuda.synchronize()
                        self.mimi.reset_streaming()
                        print(f"[try19] mimi encode graph captured AT INIT: {_ge.g is not None}", flush=True)
            except Exception as e:
                print(f"[try19] WARNING mimi graph wrap failed: {e!r}", flush=True)

        # TRY19_TRT_SEANET=<engine>: Mimi's SEANet decoder -> TensorRT sliding-window engine
        # (trt_seanet.py; exact to rounding vs streaming, 6.0 -> 1.3 ms/frame on Orin).
        _trt_eng = _os.environ.get("TRY19_TRT_SEANET", "").strip()
        if _trt_eng and getattr(self, "mimi", None) is not None:
            try:
                import sys as _sys2
                _nil = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                for _pp in (_os.path.join(_nil, "pylibs_trt"), _nil):
                    if _pp not in _sys2.path:
                        _sys2.path.insert(0, _pp)
                import trt_seanet
                trt_seanet.install(self.mimi, _trt_eng, int(_os.environ.get("TRY19_TRT_SEANET_WIN", "12")))
            except Exception as e:
                print(f"[try19] WARNING TensorRT SEANet install failed: {e!r}", flush=True)

    live.MoshiOnlyEngineWithHidden.__init__ = __init__
    print(f"[try19] patched MoshiOnlyEngineWithHidden -> num_depformer_steps={n}", flush=True)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--student_ckpt",
                     default=str(DESIGN_A / "checkpoints/design_a_L0_idle_v5/best.pt"))
    pre.add_argument("--live_module", default=LIVE_MODULE,
                     help="module to wrap; use imtalker_native_passthrough for the fork")
    pre.add_argument("--dep_q", type=int, default=0,
                     help="cap depformer codebooks per step (8 = free, <8 trades audio quality)")
    pre.add_argument("--blink_delta", default="",
                     help="precomputed pixel-space blink delta (configs/blink_delta.pt)")
    pre.add_argument("--distilled_renderer", default="",
                     help="path to the distilled single-identity renderer checkpoint")
    pre.add_argument("--motion_gain", type=float, default=1.0,
                     help="scale motion deviation to restore articulation damped by ctx_noise")
    pre.add_argument("--blend_frames", type=int, default=10,
                     help="barge-in crossfade length in 25fps frames (10 = 400 ms)")
    pre.add_argument("--jump_thresh", type=float, default=0.15,
                     help="max-abs motion jump that counts as a discontinuity")
    pre.add_argument("--mouth_close", type=float, default=0.5,
                     help="pull the crossfade start toward the reference (closed) pose, 0..1")
    mine, remaining = pre.parse_known_args()

    if any(a == "8998" for a in remaining):
        raise SystemExit("[try19] refusing to bind port 8998 - that is the production server")

    import importlib
    if mine.live_module != LIVE_MODULE and LIVE_DIR not in sys.path:
        sys.path.insert(0, LIVE_DIR)
    if INT4_G:
        # Must run BEFORE LMGen is constructed and _warmup_runtime() captures the
        # CUDA graphs, so the graphs are captured over the int4 modules.
        from moshi.models import loaders as _loaders
        from int4_linear import convert_bf16 as _conv_int4
        if DEP_LAYERS:
            _loaders._lm_kwargs["depformer_num_layers"] = DEP_LAYERS
            print(f"[try19] depformer_num_layers -> {DEP_LAYERS} "
                  "(needs PD weights trained at this depth)", flush=True)
        _orig_get_lm = _loaders.get_moshi_lm

        def _get_moshi_lm_int4(*a, **k):
            lm = _orig_get_lm(*a, **k)
            import time as _t
            if DROP_SPEC:
                _a, _b = DROP_SPEC.split("-")
                _drop = set(range(int(_a), int(_b) + 1))
                _n0 = len(lm.transformer.layers)
                lm.transformer.layers = torch.nn.ModuleList(
                    [l for i, l in enumerate(lm.transformer.layers) if i not in _drop])
                print(f"[try19] trunk {_n0} -> {len(lm.transformer.layers)} layers "
                      f"(dropped {DROP_SPEC})", flush=True)
            if FFN_HIDDEN:
                _c = 0
                for _l in lm.transformer.layers:
                    _g = getattr(_l, "gating", None)
                    if _g is None or not hasattr(_g, "linear_in"):
                        continue
                    _w = _g.linear_in.weight.data          # [2*hidden, dim]
                    _h = _w.shape[0] // 2
                    _kp = min(FFN_HIDDEN, _h)
                    # linear_in emits [gate | value]; keep the first _kp of EACH half
                    _new_in = torch.cat([_w[:_kp], _w[_h:_h + _kp]], 0).clone()
                    _g.linear_in = torch.nn.Linear(_w.shape[1], 2 * _kp, bias=False,
                                                   device=_w.device, dtype=_w.dtype)
                    _g.linear_in.weight.data.copy_(_new_in)
                    _wo = _g.linear_out.weight.data        # [dim, hidden]
                    _new_out = _wo[:, :_kp].clone()
                    _g.linear_out = torch.nn.Linear(_kp, _wo.shape[0], bias=False,
                                                    device=_wo.device, dtype=_wo.dtype)
                    _g.linear_out.weight.data.copy_(_new_out)
                    _c += 1
                print(f"[try19] FFN hidden -> {FFN_HIDDEN} on {_c} layers "
                      "(SPEED TEST ONLY, output is garbage)", flush=True)
            if FFN_KEEP and FFN_IDX:
                _pk = torch.load(FFN_IDX, map_location="cpu")["keep_idx"]
                _key = min(_pk.keys(), key=lambda x: abs(float(x) - FFN_KEEP))
                _idxs = _pk[_key]
                _ls = None
                if FFN_LSTSQ:
                    _ls = torch.load(FFN_LSTSQ, map_location="cpu")["linear_out"]
                _c = 0
                for _i, _L in enumerate(lm.transformer.layers):
                    _g = _L.gating
                    _H = _g.linear_out.weight.shape[1]
                    _ii = _idxs[_i].to(_g.linear_out.weight.device)
                    _k = _ii.numel()
                    _wi = _g.linear_in.weight.data.view(2, _H, _g.linear_in.weight.shape[1])
                    _ln = torch.nn.Linear(_wi.shape[2], 2 * _k, bias=False,
                                          device=_wi.device, dtype=_wi.dtype)
                    _ln.weight.data.copy_(_wi[:, _ii].reshape(2 * _k, _wi.shape[2]))
                    _g.linear_in = _ln
                    _wo = _g.linear_out.weight.data
                    _lo = torch.nn.Linear(_k, _wo.shape[0], bias=False,
                                          device=_wo.device, dtype=_wo.dtype)
                    # solved re-combination of survivors, not a raw slice
                    _lo.weight.data.copy_((_ls[_i].to(_wo.device) if _ls is not None
                                           else _wo[:, _ii]).to(_wo.dtype))
                    _g.linear_out = _lo
                    _c += 1
                torch.cuda.empty_cache()
                print(f"[try19] FFN narrow keep={_key} -> hidden {_k} on {_c} "
                      f"layers, linear_out={'least-squares' if _ls is not None else 'sliced'}"
                      " (BEFORE int4)", flush=True)
            _t0 = _t.time()
            _n = _conv_int4(lm.transformer, INT4_G)
            if _os.environ.get("TRY19_INT4_DEPFORMER", "0") == "1":
                # The depformer was NEVER quantized: 825 M params x 2 B = 1.65 GB
                # read every step = 8.1 ms of pure bandwidth at 204 GB/s. The
                # CUPTI profile shows it as 10.3 ms of bf16 GEMMs.
                try:
                    _nd = _conv_int4(lm.depformer, INT4_G)
                    print(f"[try19] int4 depformer: {_nd} Linears", flush=True)
                except Exception as _e:
                    print(f"[try19] WARNING depformer int4 failed: {_e!r}", flush=True)
            if _os.environ.get("TRY19_INT4_DEPFORMER", "0") == "1":
                # The depformer was NEVER quantized: 825 M params x 2 B = 1.65 GB
                # read every step = 8.1 ms of pure bandwidth at 204 GB/s. The
                # CUPTI profile shows it as 10.3 ms of bf16 GEMMs.
                try:
                    _nd = _conv_int4(lm.depformer, INT4_G)
                    print(f"[try19] int4 depformer: {_nd} Linears", flush=True)
                except Exception as _e:
                    print(f"[try19] WARNING depformer int4 failed: {_e!r}", flush=True)
            if _os.environ.get("TRY19_INT4_DEPFORMER", "0") == "1":
                # The depformer was NEVER quantized: 825 M params x 2 B = 1.65 GB
                # read every step = 8.1 ms of pure bandwidth at 204 GB/s. The
                # CUPTI profile shows it as 10.3 ms of bf16 GEMMs.
                try:
                    _nd = _conv_int4(lm.depformer, INT4_G)
                    print(f"[try19] int4 depformer: {_nd} Linears", flush=True)
                except Exception as _e:
                    print(f"[try19] WARNING depformer int4 failed: {_e!r}", flush=True)
            torch.cuda.empty_cache()
            print(f"[try19] int4 G={INT4_G}: {_n} Linears at load in "
                  f"{_t.time()-_t0:.0f}s (before graph capture)", flush=True)
            if CTX_FRAMES:
                _c = 0
                for _m in lm.transformer.modules():
                    if hasattr(_m, "context") and isinstance(getattr(_m, "context"), int):
                        _m.context = CTX_FRAMES; _c += 1
                print(f"[try19] KV context -> {CTX_FRAMES} frames "
                      f"({CTX_FRAMES/12.5:.0f}s) on {_c} attn modules", flush=True)
            return lm
        _loaders.get_moshi_lm = _get_moshi_lm_int4
        print("[try19] int4 loader hook installed", flush=True)

    if DEC_CKPT:
        # BEFORE streaming is entered: swapping mimi.decoder for a fresh module
        # after the engine has started streaming leaves it without a
        # _streaming_state, and reset_streaming() then kills the persona thread.
        from moshi.models import loaders as _ldm
        _orig_get_mimi = _ldm.get_mimi

        def _get_mimi_dec(*a, **k):
            _m = _orig_get_mimi(*a, **k)
            try:
                import mimi_dec_serve
                mimi_dec_serve.install(_m, DEC_CKPT)
            except Exception as _e:
                print(f"[try19] WARNING decoder student failed: {_e!r}", flush=True)
            return _m

        _ldm.get_mimi = _get_mimi_dec
        print(f"[try19] decoder student hook installed -> {DEC_CKPT}", flush=True)

    if MIMI_DEC_FT:
        # Load at get_mimi time: BEFORE streaming state / CUDA graphs exist. Strict
        # load -- a silent partial load would keep the stock decoder unnoticed.
        from moshi.models import loaders as _ldf
        _orig_get_mimi_ft = _ldf.get_mimi

        def _get_mimi_ft(*a, **k):
            _m = _orig_get_mimi_ft(*a, **k)
            _sd = torch.load(MIMI_DEC_FT, map_location="cpu")["decoder_transformer"]
            _p = next(_m.decoder_transformer.parameters())
            _m.decoder_transformer.load_state_dict(
                {kk: vv.to(device=_p.device, dtype=_p.dtype) for kk, vv in _sd.items()})
            print(f"[try19] Mimi decoder_transformer fine-tune loaded: {len(_sd)} tensors "
                  f"<- {MIMI_DEC_FT}", flush=True)
            return _m

        _ldf.get_mimi = _get_mimi_ft
        print(f"[try19] Mimi decoder fine-tune hook installed -> {MIMI_DEC_FT}", flush=True)

    live = importlib.import_module(mine.live_module)

    if PD_W:
        # Install BEFORE LMGen builds its CUDA graphs. Use pd_parallel (not the
        # pod's pd_serve): this fork calls graphed_depth(text_token,
        # transformer_out) with only 2 args, and pd_serve requires 4.
        import pd_parallel
        from moshi.models.lm import LMGen as _LGP

        _orig_pd_init = _LGP._init_streaming_state
        _pdbox = {}

        def _init_pd(self, batch_size):
            if "done" not in _pdbox:
                dev = next(self.lm_model.parameters()).device
                pd_parallel.load_weights(self.lm_model, PD_W, dev)
                pd_parallel.install(self, refine=PD_REFINE)
                if _os.environ.get("TRY19_DEP_GATING_INT4", "0") == "1":
                    # multi_linear (the depformer ATTENTION) slices weights per
                    # step and cannot be quantised, but the gating modules are
                    # plain nn.Linear and are ~67% of depformer params. Measured
                    # -2.74 ms. Must run after load_weights (above) and before
                    # graph capture (below).
                    try:
                        from int4_linear import group_quantize as _gq, Int4Linear as _I4
                        _nd = 0
                        for _mod in self.lm_model.depformer.modules():
                            _gate = getattr(_mod, "gating", None)
                            if _gate is None:
                                continue
                            _subs = list(_gate) if isinstance(
                                _gate, (torch.nn.ModuleList, torch.nn.Sequential)) else [_gate]
                            for _sub in _subs:
                                for _nm in ("linear_in", "linear_out"):
                                    _lin = getattr(_sub, _nm, None)
                                    if isinstance(_lin, torch.nn.Linear) and \
                                            _lin.in_features % INT4_G == 0:
                                        _pk2, _sz = _gq(_lin.weight.data, INT4_G)
                                        setattr(_sub, _nm, _I4(_pk2, _sz, _lin.bias,
                                                               _lin.out_features, INT4_G))
                                        _nd += 1
                        torch.cuda.empty_cache()
                        print(f"[try19] depformer GATING int4: {_nd} Linears "
                              "(-2.74 ms)", flush=True)
                    except Exception as _e:
                        print(f"[try19] WARNING dep gating int4 failed: {_e!r}",
                              flush=True)
                _pdbox["done"] = True
            return _orig_pd_init(self, batch_size)

        _LGP._init_streaming_state = _init_pd
        print(f"[try19] TRY19_PD hook installed -> {PD_W} (refine={PD_REFINE})",
              flush=True)

    if DEPQ_EARLY:
        from moshi.models.lm import LMGen as _LMGenD
        _orig_lmgen_init = _LMGenD.__init__

        def _lmgen_init_depq(self, *a, **k):
            _orig_lmgen_init(self, *a, **k)
            self.num_depformer_steps = DEPQ_EARLY
            print(f"[try19] DEPQ_EARLY={DEPQ_EARLY} set in LMGen.__init__ "
                  "(BEFORE graph capture, so the graph actually shrinks)", flush=True)

        _LMGenD.__init__ = _lmgen_init_depq
        print(f"[try19] DEPQ_EARLY={DEPQ_EARLY} hook installed", flush=True)

    if STRIDE > 1:
        # Gate OUTSIDE the CUDA graph: state.graphed_main is a CUDAGraphed
        # object, so replay happens inside its __call__. Wrapping the object
        # (not the captured fn) means this conditional actually fires.
        from moshi.models.lm import LMGen as _LMGen
        _orig_init_state = _LMGen._init_streaming_state

        def _init_state_strided(self, batch_size):
            st = _orig_init_state(self, batch_size)
            inner = st.graphed_main
            box = {"n": 0, "cached": None}

            def _strided(*a, **k):
                n = box["n"]
                box["n"] = n + 1
                if box["cached"] is None or (n % STRIDE) == 0:
                    box["cached"] = inner(*a, **k)
                return box["cached"]

            try:
                st.graphed_main = _strided
            except Exception as e:
                print(f"[try19] STRIDE: could not wrap graphed_main: {e}", flush=True)
                return st
            print(f"[try19] STRIDE={STRIDE} ACTIVE: trunk runs every {STRIDE} "
                  f"steps ({12.5/STRIDE:.2f}Hz), depformer+Mimi stay 12.5Hz "
                  "(SPEED TEST ONLY, output is garbage)", flush=True)
            return st

        _LMGen._init_streaming_state = _init_state_strided
        print(f"[try19] STRIDE={STRIDE} hook installed", flush=True)
    if _flag("TRY19_DEBUG_RESET", "0"):
        # The server does `except (WebSocketDisconnect, RuntimeError): pass`
        # around the whole message loop, so a failure inside reset_session()
        # tears the session down silently. Surface it.
        import traceback as _tb
        _orig_reset = live.MoshiOnlyEngineWithHidden.reset_session

        def _reset_dbg(self, *a, **k):
            try:
                return _orig_reset(self, *a, **k)
            except BaseException:
                print("[try19][DEBUG] reply_engine.reset_session RAISED:", flush=True)
                _tb.print_exc()
                raise
        live.MoshiOnlyEngineWithHidden.reset_session = _reset_dbg
        print("[try19] DEBUG: reset_session traceback hook installed", flush=True)
    if MIMI_GRAPH:
        # Must precede Mimi construction: moshi rebinds streaming state tensors
        # each call, which a CUDA graph would read stale (27% of codes corrupted).
        import graph_safe_mimi
        graph_safe_mimi.enable()
        print("[try19] mimi streaming state -> graph-safe in-place buffers", flush=True)
    print(f"[try19] wrapped live module: {live.__file__}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_student(mine.student_ckpt, device)

    refx = None
    try:
        refx = torch.load(DESIGN_A / "configs" / "ref_x_3robert.pt",
                          map_location=device)["ref_x"].float().to(device)
    except Exception as e:
        print(f"[try19] ref_x unavailable ({e}); mouth_close disabled", flush=True)
    live.LiveHeliumFMEngine._sample_motion_from_helium = make_student_sampler(
        model, cfg, blend_frames=mine.blend_frames, jump_thresh=mine.jump_thresh,
        mouth_close=(mine.mouth_close if refx is not None else 0.0), ref_x=refx,
        motion_gain=mine.motion_gain)
    if mine.dep_q > 0:
        patch_depformer(live, mine.dep_q)
    if mine.distilled_renderer:
        _bl = load_blink_delta(mine.blink_delta, device) if mine.blink_delta else None
        patch_render(live, load_distilled_renderer(mine.distilled_renderer, device), _bl)
    print(f"[try19] motion_gain={mine.motion_gain}", flush=True)
    print(f"[try19] barge-in crossfade: {mine.blend_frames} frames "
          f"({mine.blend_frames*40} ms), jump_thresh={mine.jump_thresh}, "
          f"mouth_close={mine.mouth_close}", flush=True)
    print("[try19] patched _sample_motion_from_helium -> Design A causal student", flush=True)
    print("[try19] student KV state stored INSIDE stream_state -> speculative "
          "rollouts are rewound correctly", flush=True)
    if cfg.lookahead:
        print(f"[try19] NOTE lookahead={cfg.lookahead} -> video lags audio by "
              f"{cfg.lookahead*80} ms", flush=True)

    sys.argv = [sys.argv[0]] + remaining
    live.main()


if __name__ == "__main__":
    main()
