"""UniTalk 12-layer adapter wrapper for IMTalker's last-layer audio path."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from generator.wav2vec2 import Wav2VecModel


class UniTalkWav2VecAdapter(nn.Module):
    """Map 12.5 Hz PersonaPlex/Moshi hidden states to 12 Wav2Vec-like layers."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.0,
        conv_pos_kernel: int = 128,
        conv_pos_groups: int = 16,
        right_context_frames: int = -1,
        recurrent_layers: int = 0,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.conv_pos_kernel = int(conv_pos_kernel)
        self.right_context_frames = int(right_context_frames)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.conv_pos = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_pos_kernel,
            padding=conv_pos_kernel // 2 if right_context_frames < 0 else 0,
            groups=conv_pos_groups,
        )
        self.conv_pos_gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.recurrent = (
            nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=int(recurrent_layers),
                batch_first=True,
            )
            if recurrent_layers > 0
            else None
        )
        self.recurrent_norm = (
            nn.LayerNorm(hidden_dim) if recurrent_layers > 0 else None
        )
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )

    def _attention_mask(self, steps: int, device: torch.device) -> torch.Tensor | None:
        if self.right_context_frames < 0:
            return None
        diagonal = self.right_context_frames + 1
        return torch.ones(steps, steps, dtype=torch.bool, device=device).triu(
            diagonal=diagonal
        )

    def _position_features(self, x: torch.Tensor) -> torch.Tensor:
        source = x.transpose(1, 2)
        if self.right_context_frames < 0:
            return self.conv_pos(source)[:, :, : x.shape[1]]

        right = min(self.right_context_frames, self.conv_pos_kernel - 1)
        left = self.conv_pos_kernel - 1 - right
        source = F.pad(source, (left, right))
        return self.conv_pos(source)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        steps = x.shape[1]
        x = self.feature_projection(x)
        positional = self._position_features(x)
        x = x + self.conv_pos_gelu(positional).transpose(1, 2)
        x = self.input_dropout(self.layer_norm(x))
        if self.recurrent is not None:
            recurrent, _ = self.recurrent(x)
            x = self.recurrent_norm(x + recurrent)
        attention_mask = self._attention_mask(steps, x.device)
        layers = []
        for layer in self.transformer_layers:
            x = layer(x, src_mask=attention_mask)
            layers.append(x)
        output = torch.stack(layers, dim=2)
        return output.squeeze(0) if squeeze else output


class UniTalkLastLayerLiveAdapter(nn.Module):
    """Expose UniTalk layer 12 as IMTalker's `[T, 768]` audio feature."""

    def __init__(
        self,
        wav2vec_model_path: str,
        dropout: float = 0.0,
        right_context_frames: int = -1,
        recurrent_layers: int = 0,
    ) -> None:
        super().__init__()
        self.model = UniTalkWav2VecAdapter(
            dropout=dropout,
            right_context_frames=right_context_frames,
            recurrent_layers=recurrent_layers,
        )

        # Retained for existing optional raw-audio diagnostics. The live
        # UniTalk path itself sends layer 12 directly to IMTalker.
        self.wav2vec = Wav2VecModel.from_pretrained(
            wav2vec_model_path, local_files_only=True
        ).eval().float()
        for parameter in self.wav2vec.parameters():
            parameter.requires_grad_(False)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        if isinstance(state_dict, dict):
            for key in ("adapter", "model", "state_dict"):
                nested = state_dict.get(key)
                if isinstance(nested, dict):
                    state_dict = nested
                    break

        normalized = {}
        for key, value in state_dict.items():
            for prefix in ("module.", "adapter.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            normalized[key] = value
        return self.model.load_state_dict(normalized, strict=strict)

    @torch.no_grad()
    def forward_single(
        self, source: torch.Tensor, target_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = source.unsqueeze(0).float().contiguous()
        target_len = max(1, int(target_len))
        source = F.interpolate(
            source.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).contiguous()
        all_layers = self.model(source)
        last_layer = all_layers[:, :, -1, :].float().contiguous()
        features = last_layer[0]
        return features, features, features
