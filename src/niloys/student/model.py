"""Design A: causal Transformer motion student.

At PersonaPlex Helium step t (80 ms) the student consumes
    h_t            [4096]  current Helium hidden state
    prev_motion    [64]    the previous two 32-D motion frames
    causal history via a bounded KV cache
and emits [2,32] -- the two 25 fps motion frames for that step.

Positional encoding is RoPE applied over *window-relative* offsets, so a
session of unbounded length never exceeds the trained offset range and no
learned absolute position table can run out.  Attention uses a sliding causal
window of `attn_window` steps in both training and streaming, which makes the
full-sequence forward and the cached step-by-step forward numerically agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    helium_dim: int = 4096
    motion_dim: int = 32
    frames_per_step: int = 2
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 2048
    attn_window: int = 128     # bounded KV cache, in Helium steps
    rope_base: float = 10000.0
    dropout: float = 0.0
    lookahead: int = 1         # Helium steps of future context (label shift)

    @property
    def out_dim(self) -> int:
        return self.frames_per_step * self.motion_dim


# ---------------------------------------------------------------- RoPE

def build_rope(max_offset: int, head_dim: int, base: float, device, dtype):
    """cos/sin tables for offsets 0..max_offset-1."""
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(max_offset, device=device, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]                      # [P, half]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def apply_rope(x, cos, sin):
    """x: [B, H, T, D]; cos/sin: [T, D/2]."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


# ---------------------------------------------------------------- blocks

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.window = cfg.attn_window
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def forward(self, x, cos, sin):
        """Full-sequence windowed-causal forward. x: [B,T,C]."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        idx = torch.arange(T, device=x.device)
        rel = idx[:, None] - idx[None, :]
        mask = (rel >= 0) & (rel < self.window)            # windowed causal
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.masked_fill(~mask[None, None], float("-inf"))
        y = torch.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)

    def step(self, x, cache, cos, sin):
        """Single-step cached forward. x: [B,1,C]. cache holds UNROTATED k,v."""
        B, _, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, 1, self.h, self.dh).transpose(1, 2)
        k = k.view(B, 1, self.h, self.dh).transpose(1, 2)
        v = v.view(B, 1, self.h, self.dh).transpose(1, 2)

        if cache["k"] is None:
            kk, vv = k, v
        else:
            kk = torch.cat([cache["k"], k], dim=2)
            vv = torch.cat([cache["v"], v], dim=2)
        if kk.shape[2] > self.window:                       # bounded cache
            kk = kk[:, :, -self.window:]
            vv = vv[:, :, -self.window:]
        cache["k"], cache["v"] = kk, vv

        n = kk.shape[2]
        # window-relative positions: current token sits at n-1, key j at j.
        kk_rot = apply_rope(kk, cos[:n], sin[:n])
        q_rot = apply_rope(q, cos[n - 1:n], sin[n - 1:n])
        att = (q_rot @ kk_rot.transpose(-1, -2)) / math.sqrt(self.dh)
        y = torch.softmax(att, dim=-1) @ vv
        y = y.transpose(1, 2).reshape(B, 1, C)
        return self.proj(y)

    def step_fixed(self, x, cache, cos, sin):
        """CUDA-graph-safe variant of step().

        step() does cache["k"] = torch.cat([...]), allocating a NEW tensor and
        rebinding the dict every call. A captured graph freezes the addresses it
        saw, so on replay it reads the stale buffer. Here k/v live in tensors
        allocated once and mutated in place, so replay is correct.

        Equivalence to step(): RoPE is relative, so placing the newest token at
        a fixed slot (window-1) instead of a moving slot (n-1) shifts every
        position by a constant and leaves the attention scores unchanged. Slots
        not yet written are masked out, matching step()'s shorter cache exactly.
        """
        B, _, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, 1, self.h, self.dh).transpose(1, 2)
        k = k.view(B, 1, self.h, self.dh).transpose(1, 2)
        v = v.view(B, 1, self.h, self.dh).transpose(1, 2)

        kb, vb, filled = cache["k"], cache["v"], cache["filled"]
        kb.copy_(torch.roll(kb, -1, dims=2)); kb[:, :, -1:].copy_(k)
        vb.copy_(torch.roll(vb, -1, dims=2)); vb[:, :, -1:].copy_(v)
        filled.add_(1).clamp_(max=self.window)

        n = self.window
        kk_rot = apply_rope(kb, cos[:n], sin[:n])
        q_rot = apply_rope(q, cos[n - 1:n], sin[n - 1:n])
        att = (q_rot @ kk_rot.transpose(-1, -2)) / math.sqrt(self.dh)
        idx = torch.arange(n, device=x.device)
        valid = idx >= (n - filled)
        att = att.masked_fill(~valid[None, None, None, :], float("-inf"))
        y = torch.softmax(att, dim=-1) @ vb
        y = y.transpose(1, 2).reshape(B, 1, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_dim, cfg.d_model),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))

    def step(self, x, cache, cos, sin):
        x = x + self.attn.step(self.ln1(x), cache, cos, sin)
        return x + self.mlp(self.ln2(x))

    def step_fixed(self, x, cache, cos, sin):
        x = x + self.attn.step_fixed(self.ln1(x), cache, cos, sin)
        return x + self.mlp(self.ln2(x))


# ---------------------------------------------------------------- student

class CausalMotionStudent(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        D, C = cfg.helium_dim, cfg.d_model
        self.helium_ln = nn.LayerNorm(D)
        self.helium_proj = nn.Linear(D, C)
        self.prev_enc = nn.Linear(cfg.out_dim, C)
        self.fuse_gate = nn.Linear(2 * C, C)
        self.fuse_lin = nn.Linear(2 * C, C)
        self.fuse_ln = nn.LayerNorm(C)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.out_ln = nn.LayerNorm(C)
        self.head_abs = nn.Linear(C, cfg.out_dim)
        self.head_res = nn.Linear(C, cfg.out_dim)
        self.head_gate = nn.Linear(C, cfg.out_dim)

        # normalization stats (buffers -> travel with the checkpoint)
        self.register_buffer("helium_mean", torch.zeros(D))
        self.register_buffer("helium_std", torch.ones(D))
        self.register_buffer("motion_mean", torch.zeros(cfg.motion_dim))
        self.register_buffer("motion_std", torch.ones(cfg.motion_dim))

        self.apply(self._init)
        nn.init.zeros_(self.head_res.weight); nn.init.zeros_(self.head_res.bias)
        nn.init.zeros_(self.head_gate.weight); nn.init.zeros_(self.head_gate.bias)

        # RoPE tables. Streaming only ever needs `attn_window` offsets; the
        # full-sequence forward indexes absolute positions 0..T-1, so the table
        # grows on demand. Relative offsets stay <= attn_window either way, so
        # the two paths produce identical attention scores.
        self._rope_cache: dict = {}

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---- normalization helpers
    def set_stats(self, hm, hs, mm, ms):
        self.helium_mean.copy_(hm); self.helium_std.copy_(hs.clamp_min(1e-5))
        self.motion_mean.copy_(mm); self.motion_std.copy_(ms.clamp_min(1e-5))

    def norm_motion(self, m):
        return (m - self.motion_mean) / self.motion_std

    def denorm_motion(self, m):
        return m * self.motion_std + self.motion_mean

    def _embed(self, helium, prev_motion):
        h = (helium - self.helium_mean) / self.helium_std
        he = self.helium_proj(self.helium_ln(h))
        me = self.prev_enc(prev_motion)
        cat = torch.cat([he, me], dim=-1)
        g = torch.sigmoid(self.fuse_gate(cat))
        return self.fuse_ln(g * he + (1.0 - g) * me + self.fuse_lin(cat))

    def _heads(self, y, prev_motion):
        y = self.out_ln(y)
        a = self.head_abs(y)
        r = self.head_res(y)
        g = torch.sigmoid(self.head_gate(y))
        # residual head predicts a delta from the most recent known frame
        last = prev_motion[..., self.cfg.motion_dim:]
        base = last.repeat(1, 1, self.cfg.frames_per_step)
        res = base + r
        return g * a + (1.0 - g) * res, a, res, g

    def _rope(self, n, device, dtype):
        key = (int(n), str(device), str(dtype))
        hit = self._rope_cache.get(key)
        if hit is None:
            hit = build_rope(int(n), self.cfg.d_model // self.cfg.n_heads,
                             self.cfg.rope_base, device, dtype)
            self._rope_cache[key] = hit
        return hit

    def forward(self, helium, prev_motion):
        """helium [B,T,4096], prev_motion [B,T,64] (normalized) -> [B,T,2,32]."""
        B, T, _ = helium.shape
        cos, sin = self._rope(T, helium.device, helium.dtype)
        x = self._embed(helium, prev_motion)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        out, a, r, g = self._heads(x, prev_motion)
        shape = (B, T, self.cfg.frames_per_step, self.cfg.motion_dim)
        return out.view(shape), a.view(shape), r.view(shape), g.view(shape)

    # ---- streaming
    def init_state(self, batch_size=1, device="cpu", dtype=torch.float32):
        return {
            "caches": [{"k": None, "v": None} for _ in range(self.cfg.n_layers)],
            "prev_motion": torch.zeros(batch_size, 1, self.cfg.out_dim,
                                       device=device, dtype=dtype),
            "steps": 0,
        }

    @torch.no_grad()
    def step(self, helium_t, state, prev_motion=None):
        """helium_t [B,4096] (or [B,1,4096]) -> [B,2,32] normalized motion."""
        if helium_t.ndim == 2:
            helium_t = helium_t.unsqueeze(1)
        pm = state["prev_motion"] if prev_motion is None else prev_motion
        if pm.ndim == 2:
            pm = pm.unsqueeze(1)
        cos, sin = self._rope(self.cfg.attn_window, helium_t.device, helium_t.dtype)
        x = self._embed(helium_t, pm)
        for blk, cache in zip(self.blocks, state["caches"]):
            x = blk.step(x, cache, cos, sin)
        out, _a, _r, _g = self._heads(x, pm)
        state["prev_motion"] = out.detach()
        state["steps"] += 1
        B = helium_t.shape[0]
        return out.view(B, self.cfg.frames_per_step, self.cfg.motion_dim)

    def init_state_fixed(self, batch_size=1, device="cpu", dtype=torch.float32):
        """State whose KV buffers are allocated once and mutated in place, so a
        CUDA graph captured over step_fixed() stays valid across replays."""
        h = self.cfg.n_heads
        dh = self.cfg.d_model // self.cfg.n_heads
        w = self.cfg.attn_window
        return {
            "caches": [
                {"k": torch.zeros(batch_size, h, w, dh, device=device, dtype=dtype),
                 "v": torch.zeros(batch_size, h, w, dh, device=device, dtype=dtype),
                 "filled": torch.zeros((), device=device, dtype=torch.long)}
                for _ in range(self.cfg.n_layers)
            ],
            "prev_motion": torch.zeros(batch_size, 1, self.cfg.out_dim,
                                       device=device, dtype=dtype),
            "steps": torch.zeros((), device=device, dtype=torch.long),
        }

    @torch.no_grad()
    def step_fixed(self, helium_t, state, prev_motion=None):
        """Graph-safe twin of step(). prev_motion is written in place so the
        graph's view of it stays valid; callers read state["prev_motion"]."""
        if helium_t.ndim == 2:
            helium_t = helium_t.unsqueeze(1)
        pm = state["prev_motion"] if prev_motion is None else prev_motion
        if pm.ndim == 2:
            pm = pm.unsqueeze(1)
        cos, sin = self._rope(self.cfg.attn_window, helium_t.device, helium_t.dtype)
        x = self._embed(helium_t, pm)
        for blk, cache in zip(self.blocks, state["caches"]):
            x = blk.step_fixed(x, cache, cos, sin)
        out, _a, _r, _g = self._heads(x, pm)
        state["prev_motion"].copy_(out.detach())
        state["steps"].add_(1)
        B = helium_t.shape[0]
        return out.view(B, self.cfg.frames_per_step, self.cfg.motion_dim)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build_student(**kw):
    return CausalMotionStudent(StudentConfig(**kw))
