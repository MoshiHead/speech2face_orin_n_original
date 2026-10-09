"""Distilled single-identity renderer: 32-D motion -> 512x512 frame.

Design is driven by measurement, not guesswork:
  * 90% of temporal variance lives in 3.8% of pixels -- the frame is nearly
    static, so synthesising 262144 pixels per frame is wasted work.
  * A constant base frame alone already scores 37.2 dB.
  * A PERFECT residual at 256x256 scores 50.7 dB; at 128x128, 45.4 dB.
  * Identity is fixed, so the reference features and the renderer's 6-scale
    cross-attention (27% of its cost) collapse into constants and disappear.

So this predicts a LOW-RES RESIDUAL on a constant base frame and upsamples,
instead of running a full-resolution synthesis network (67% of renderer cost).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RendererConfig:
    motion_dim: int = 32
    res: int = 256          # residual synthesis resolution
    out_res: int = 512
    base_ch: int = 256
    start: int = 8
    widths: tuple = (256, 256, 128, 64, 32, 16)   # 8,16,32,64,128,256
    blink_len: int = 0      # >0 conditions the decoder on blink phase


class UpBlock(nn.Module):
    def __init__(self, cin, cout, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, cout), cout)
        self.act = nn.SiLU()
        # FiLM modulation from the motion code: cheap conditioning at every scale
        self.mod = nn.Linear(style_dim, cout * 2)

    def forward(self, x, s, up=True):
        if up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.norm(self.conv(x))
        g, b = self.mod(s).chunk(2, dim=-1)
        x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return self.act(x)


class DistilledRenderer(nn.Module):
    def __init__(self, cfg: RendererConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.widths[0]
        self.style = nn.Sequential(
            nn.Linear(cfg.motion_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
        )
        # Blink is a fixed 367-frame loop, so a lookup table is the natural
        # parameterisation: exact per-phase capacity, no frequency-encoding
        # approximation, and only blink_len x 512 parameters.
        self.blink_emb = nn.Embedding(cfg.blink_len, 512) if cfg.blink_len > 0 else None
        if self.blink_emb is not None:
            nn.init.zeros_(self.blink_emb.weight)
        self.const = nn.Parameter(torch.randn(1, C, cfg.start, cfg.start) * 0.02)
        self.inject = nn.Linear(512, C * cfg.start * cfg.start)
        blocks, cin = [], C
        for i, cout in enumerate(cfg.widths[1:]):
            blocks.append(UpBlock(cin, cout, 512))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Conv2d(cin, 3, 3, padding=1)
        nn.init.zeros_(self.to_rgb.weight); nn.init.zeros_(self.to_rgb.bias)

        self.register_buffer("base_frame", torch.zeros(3, cfg.out_res, cfg.out_res))
        self.register_buffer("motion_mean", torch.zeros(cfg.motion_dim))
        self.register_buffer("motion_std", torch.ones(cfg.motion_dim))

    def set_base(self, base, mean, std):
        self.base_frame.copy_(base)
        self.motion_mean.copy_(mean)
        self.motion_std.copy_(std.clamp_min(1e-5))

    def forward(self, motion, phase=None, return_residual=False):
        """motion [B,32] raw (+ blink phase [B]) -> frame [B,3,512,512] in [0,1]."""
        B = motion.shape[0]
        s = self.style((motion - self.motion_mean) / self.motion_std)
        if self.blink_emb is not None and phase is not None:
            s = s + self.blink_emb(phase)
        x = self.const.expand(B, -1, -1, -1) + \
            self.inject(s).view(B, -1, self.cfg.start, self.cfg.start)
        for i, blk in enumerate(self.blocks):
            x = blk(x, s, up=True)
        res = self.to_rgb(x)                                     # [B,3,res,res]
        if res.shape[-1] != self.cfg.out_res:
            res = F.interpolate(res, size=(self.cfg.out_res, self.cfg.out_res),
                                mode="bilinear", align_corners=False)
        out = (self.base_frame.unsqueeze(0) + res).clamp(0, 1)
        return (out, res) if return_residual else out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def build(**kw):
    return DistilledRenderer(RendererConfig(**kw))
