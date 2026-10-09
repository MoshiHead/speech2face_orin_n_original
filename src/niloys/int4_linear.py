"""Fused int4 Linear for the PersonaPlex trunk.

Measured on AGX Orin against the ORIGINAL bf16 weights (not the NF4 copy):
    bf16        98.64 ms/step   cosine 1.000 (reference)
    NF4 (prod)  57.60 ms/step   cosine 0.844
    int4 G=128  51.24 ms/step   cosine 0.816
    int4 G=32   49.53 ms/step   cosine 0.876   <-- faster AND more faithful than NF4

Group size 32 matters: this trunk is 32 residual layers deep, so per-layer error
compounds (0.995^32 ~ 0.85). Quantise from bf16, never from the NF4 copy.
"""
import torch
import torch.nn as nn

DEFAULT_G = 32


def group_quantize(w, groupsize=DEFAULT_G):
    out, inn = w.shape
    wf = w.to(torch.float32).reshape(out, inn // groupsize, groupsize)
    mx, mn = wf.amax(-1, keepdim=True), wf.amin(-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-6) / 15.0
    q = ((wf - mn) / scale).round().clamp_(0, 15).reshape(out, inn).to(torch.uint8)
    packed_u8 = (q[:, 0::2] << 4 | q[:, 1::2]).contiguous()   # even index -> HIGH nibble
    packed = torch.ops.aten._convert_weight_to_int4pack(packed_u8, 2)
    s = scale.squeeze(-1).transpose(0, 1).contiguous()
    z = (mn.squeeze(-1) + 8 * scale.squeeze(-1)).transpose(0, 1).contiguous()
    return packed, torch.stack([s, z], -1).to(torch.bfloat16).contiguous()


class Int4Linear(nn.Module):
    def __init__(self, packed, sz, bias, out_features, groupsize=DEFAULT_G):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("sz", sz)
        self.bias = bias
        self.out_features = out_features
        self.groupsize = int(groupsize)
        # moshi's streaming init reads .weight.device/.dtype; a zero-size buffer
        # satisfies those lookups without holding weights.
        _w = torch.empty(0, dtype=torch.bfloat16, device=packed.device)
        # gating.py dispatches on hasattr(weight, "quant_type") to call the module
        # instead of F.linear on raw weights -- take that same path.
        _w.quant_type = "int4pack"
        self.register_buffer("weight", _w)

    def forward(self, x):
        shp = x.shape
        y = torch.ops.aten._weight_int4pack_mm(
            x.reshape(-1, shp[-1]).to(torch.bfloat16),
            self.packed, self.groupsize, self.sz)
        y = y.reshape(*shp[:-1], self.out_features)
        return y + self.bias if self.bias is not None else y


def convert_bf16(module, groupsize=DEFAULT_G):
    """Replace nn.Linear with Int4Linear in place. Returns count converted."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.in_features % groupsize == 0 \
                and child.out_features % 8 == 0:
            try:
                packed, sz = group_quantize(child.weight.data, groupsize)
                setattr(module, name, Int4Linear(packed, sz, child.bias,
                                                 child.out_features, groupsize))
                n += 1
            except Exception:
                pass
        else:
            n += convert_bf16(child, groupsize)
    return n
