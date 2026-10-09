"""Wide-pass parallel depformer: all generated codebooks in ONE forward.

Keeps the REAL depformer (1.398 B) and its full capacity. The depformer's
self-attention runs over the CODEBOOK axis, not over time, so the 8 generated
positions are a sequence that can be batched into a single pass -- exactly the
form `LMModel.forward_depformer_training` already uses. Given identical inputs
the wide pass computes the same function as the sequential chain.

The ONLY approximation: position cb>=1 needs the CURRENT frame's codebook cb-1,
which has not been sampled yet, so the PREVIOUS frame's token is used as a warm
start. `pd_train.py` fine-tunes the depformer to expect that.

This is not the same idea as a small from-scratch head: that removes the residual
conditioning entirely and produces mutually inconsistent codebooks (measured:
cb1 0.887 but cb2..8 ~0.45, audio intermittently unintelligible). Here the
conditioning survives, just one frame stale.
"""
import os as _os
import torch
import torch.nn as nn

# Independent sampling across codebooks compounds error: no position can see
# any other position's DRAW, so 8 marginal samples are not a coherent RVQ
# chain. Greedy removes that randomness. Measured on the 54.9 M parallel
# head: 19.9% -> 9.8% WER, 7/8 identical transcripts.
GREEDY = _os.environ.get('PD_GREEDY', '0') == '1'


@torch.no_grad()
def wide_logits(lm, text_token, transformer_out, est, n):
    """One depformer pass over n codebook positions.

    text_token      [B]        text token for this frame
    transformer_out [B,1,D]    trunk hidden
    est             [B,n]      token feeding position cb is est[:, cb-1] (cb>=1)
    returns logits  [B,n,card]
    """
    card = lm.card
    est = est.clamp(0, card - 1)          # ungenerated_token_id is out of range
    text_token = text_token.clamp(0, lm.text_card - 1)
    ins = []
    for cb in range(n):
        ti = lm.depformer_in[cb](transformer_out)                  # [B,1,dep_dim]
        if cb == 0:
            tok = lm.depformer_text_emb(text_token[:, None])       # [B,1,dep_dim]
        else:
            tok = lm.depformer_emb[cb - 1](est[:, cb - 1:cb])      # [B,1,dep_dim]
        ins.append(ti + tok)
    # depformer.set_streaming_propagate(False) -> outside depformer.streaming()
    # this is a full causal pass over the codebook axis, which is what we want.
    dep_out = lm.depformer(torch.cat(ins, 1))                      # [B,n,dep_dim]
    return torch.stack([lm.linears[cb](dep_out[:, cb]) for cb in range(n)], 1)


class UEmb(nn.Module):
    """Quantile-noise input (pd_work/arch_test.py, arm "quant"). Each generated codebook
    gets its own u ~ U[0,1], embedded with Fourier features. The student was trained so
    that argmax(logits | h, text, u) = the teacher's inverse-CDF sample at u, given the
    student's OWN earlier codebooks (on-policy). So the 8 codebooks agree on one joint
    draw in a single pass -- no stale previous-frame token, no independent sampling.
    Frequencies are computed here in fp32, never stored: bf16 cannot hold k*pi."""
    def __init__(self, dim, n, nfreq=32):
        super().__init__()
        self.nfreq = nfreq
        self.proj = nn.ModuleList([nn.Linear(2 * nfreq + 1, dim) for _ in range(n)])

    def forward(self, u, cb):
        f = torch.arange(1, self.nfreq + 1, device=u.device, dtype=torch.float32) * torch.pi
        x = u[:, None].float() * f[None]
        feat = torch.cat([u[:, None].float(), x.sin(), x.cos()], -1)
        return self.proj[cb](feat.to(self.proj[cb].weight.dtype))[:, None, :]


@torch.no_grad()
def quant_logits(lm, text_token, transformer_out, u, n):
    """One depformer pass, quantile-noise arm: position cb sees (h, text if cb==0, u_cb)."""
    text_token = text_token.clamp(0, lm.text_card - 1)
    ue = lm._pd_uemb
    ins = []
    for cb in range(n):
        x = lm.depformer_in[cb](transformer_out)
        if cb == 0:
            x = x + lm.depformer_text_emb(text_token[:, None])
        ins.append(x + ue(u[:, cb], cb).to(x.dtype))
    dep_out = lm.depformer(torch.cat(ins, 1))
    return torch.stack([lm.linears[cb](dep_out[:, cb]) for cb in range(n)], 1)


def build_inputs_for_training(lm, text_token, transformer_out, est, n):
    """Same as wide_logits but differentiable (no no_grad)."""
    card = lm.card
    est = est.clamp(0, card - 1)
    text_token = text_token.clamp(0, lm.text_card - 1)
    ins = []
    for cb in range(n):
        ti = lm.depformer_in[cb](transformer_out)
        if cb == 0:
            tok = lm.depformer_text_emb(text_token[:, None])
        else:
            tok = lm.depformer_emb[cb - 1](est[:, cb - 1:cb])
        ins.append(ti + tok)
    dep_out = lm.depformer(torch.cat(ins, 1))
    return torch.stack([lm.linears[cb](dep_out[:, cb]) for cb in range(n)], 1)


def install(lm_gen, refine=0, verbose=True):
    """Replace LMGen.depformer_step with the wide pass.

    Must run BEFORE the CUDA graph is captured (LMGen wraps depformer_step in
    CUDAGraphed at streaming-state init), or the replay keeps the old path.
    """
    import types
    lm = lm_gen.lm_model
    dep_q = int(lm.dep_q)
    n = dep_q // 2                      # generated half; the rest is the user stream
    QUANT = getattr(lm, "_pd_uemb", None) is not None
    if QUANT and refine:
        raise RuntimeError("[pd] refine has no meaning for the quantile-noise arm")
    # GRAPH-SAFE warm start. LMGen wraps depformer_step in CUDAGraphed, so after
    # capture the python body never runs again (measured: 2 python calls per 20
    # frames). Rebinding `prev` to a NEW tensor means the replayed graph keeps
    # reading the capture-time buffer forever -- a frozen warm start, which is
    # heard as dropped/garbled words. Allocate ONE buffer and copy_ into it:
    # the graph then records a read and an in-place write on the same memory,
    # and the read precedes the write, so frame t sees frame t-1.
    state = {"prev": None}

    def _ensure(B, dev):
        if state["prev"] is None or state["prev"].shape[0] != B \
                or state["prev"].device != dev:
            state["prev"] = torch.zeros(B, dep_q, dtype=torch.long, device=dev)
        return state["prev"]

    @torch.no_grad()
    def _pd_step(self_gen, text_token, transformer_out, audio_tokens=None,
                 audio_provided=None):
        from moshi.models.lm import sample_token
        B = text_token.shape[0]
        prev = _ensure(B, text_token.device)

        est = prev
        if audio_tokens is not None and audio_provided is not None:
            prov = audio_provided[:, :dep_q].clone()
            prov[:, :n] = False          # the first n are generated, never provided
            est = torch.where(prov, audio_tokens[:, :dep_q], prev)

        if QUANT:
            # torch.rand inside the graphed step is fine: CUDA-graph replay advances
            # the philox offset, so every frame gets fresh u (sample_token relies on
            # the same thing for multinomial).
            u = torch.rand(B, n, device=text_token.device)
            toks = quant_logits(lm, text_token, transformer_out, u, n).float().argmax(-1)
            out = torch.cat([toks, est[:, n:]], 1)
            if audio_tokens is not None and audio_provided is not None:
                out = torch.where(audio_provided[:, :dep_q],
                                  audio_tokens[:, :dep_q], out)
            prev.copy_(out)
            return out

        logits = wide_logits(lm, text_token, transformer_out, est, n)
        _us = False if GREEDY else self_gen.use_sampling
        toks = sample_token(logits[:, :, None].float(), _us,
                            self_gen.temp, self_gen.top_k)[:, :, 0]      # [B,n]

        for _ in range(int(refine)):
            est2 = torch.cat([toks, est[:, n:]], 1)
            logits = wide_logits(lm, text_token, transformer_out, est2, n)
            toks = sample_token(logits[:, :, None].float(), _us,
                                self_gen.temp, self_gen.top_k)[:, :, 0]

        out = torch.cat([toks, est[:, n:]], 1)
        if audio_tokens is not None and audio_provided is not None:
            out = torch.where(audio_provided[:, :dep_q],
                              audio_tokens[:, :dep_q], out)
        prev.copy_(out)                  # IN-PLACE: survives graph replay
        return out

    lm_gen.depformer_step = types.MethodType(_pd_step, lm_gen)

    # reset the warm start between conversations -- otherwise frame 0 of a new
    # session is conditioned on the last frame of the previous one.
    orig_reset = getattr(lm_gen, "reset_streaming", None)
    if orig_reset is not None:
        def _reset(self_gen, *a, **k):
            state["prev"] = None
            return orig_reset(*a, **k)
        lm_gen.reset_streaming = types.MethodType(_reset, lm_gen)

    if verbose:
        print(f"[pd] wide-pass depformer installed: {n} codebooks in ONE pass, "
              f"mode={'quantile-noise' if QUANT else 'stale warm start'}, refine={refine}, greedy={GREEDY}, full {sum(p.numel() for p in lm.depformer.parameters())/1e6:.0f} M "
              f"depformer retained", flush=True)
    return state


def load_weights(lm, path, device):
    """Load a fine-tuned depformer state dict into the five module groups."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for _k in ("state", "model", "state_dict"):
        if isinstance(sd, dict) and _k in sd and isinstance(sd[_k], dict):
            sd = sd[_k]; break
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    ue = {k[len("uemb."):]: v for k, v in sd.items() if k.startswith("uemb.")}
    sd = {k: v for k, v in sd.items() if not k.startswith("uemb.")}
    if ue:
        n = len({k.split(".")[1] for k in ue})
        dim = lm.depformer_in[0].out_features
        m = UEmb(dim, n)
        m.load_state_dict(ue)                              # strict: every tensor must land
        lm._pd_uemb = m.to(device=device, dtype=next(lm.depformer.parameters()).dtype).eval()
        print(f"[pd] quantile-noise checkpoint: u-embedding for {n} codebooks loaded", flush=True)
    else:
        lm._pd_uemb = None
    # Restrict to the depformer path. pd_kl5.py builds the student with LMModel(...)
    # (random init) and saves the WHOLE non-transformer state dict, so the file also
    # carries untrained emb/text_emb/text_linear/out_norm -- N(0,1) noise, cosine
    # ~0.0003 against the real weights. A `k in tgt` match would happily overwrite
    # the trunk's correct embeddings with that noise.
    _GROUPS = ("depformer.", "depformer_in.", "depformer_emb.",
               "depformer_text_emb.", "linears.")
    _before = len(sd)
    sd = {k: v for k, v in sd.items() if k.startswith(_GROUPS)}
    if _before != len(sd):
        print(f"[pd] dropped {_before - len(sd)} non-depformer tensor(s) "
              "(trunk emb/head -- would corrupt the trunk)", flush=True)
    if not sd:
        raise RuntimeError(f"[pd] {path}: 0 tensors after unwrapping")
    tgt = lm.state_dict()
    # Trees disagree: `self_attn.in_proj_weight` (raw Parameter) vs
    # `self_attn.in_proj.weight` (Linear). Same shape, so an exact-key filter
    # silently drops EVERY layer's Q/K/V and you get base-model attention.
    DRIFT = ((".in_proj_weight", ".in_proj.weight"), (".in_proj.weight", ".in_proj_weight"))
    keep, remapped = {}, 0
    for k, v in sd.items():
        tk = k
        if tk not in tgt:
            for a, b in DRIFT:
                if k.endswith(a) and (k[: -len(a)] + b) in tgt:
                    tk = k[: -len(a)] + b; break
        if tk in tgt and tgt[tk].shape == v.shape:
            keep[tk] = v
            if tk != k: remapped += 1
    if remapped:
        print(f"[pd] remapped {remapped} in_proj key(s) for this moshi version", flush=True)
    want = [k for k in sd if k.startswith("depformer") and "in_proj" in k]
    got  = [k for k in keep if k.startswith("depformer") and "in_proj" in k]
    if len(got) != len(want):
        raise RuntimeError(f"[pd] only {len(got)}/{len(want)} depformer in_proj tensors matched")
    miss = lm.load_state_dict(
        {k: v.to(device=device, dtype=tgt[k].dtype) for k, v in keep.items()},
        strict=False)
    print(f"[pd] loaded {len(keep)}/{len(sd)} depformer tensors from {path}", flush=True)
    return len(keep), miss
