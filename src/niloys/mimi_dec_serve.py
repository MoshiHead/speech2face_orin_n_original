"""Serve-time swap of the Mimi decoder for a trained student decoder.

The student keeps the quantizer, the upsample, the SEANet widths and the frame rate, and only drops
operations (decoder_transformer layers and the SEANet residual blocks). The codes therefore mean
exactly what they meant before, and only the PCM reconstruction changes.

Enable with MOSHI_DEC_CKPT=/path/to/mimi_decoder_student.pt
"""
import torch


def _seanet_kwargs_no_res():
    from moshi.models import loaders as _ld
    kwargs = dict(_ld._seanet_kwargs)
    for k in ("input_dimension", "output_dimension"):
        kwargs.pop(k, None)
    kwargs["n_residual_layers"] = 0
    return kwargs


def install(mimi, ckpt_path, verbose=True):
    """Modify `mimi` in place to use the student decoder. Returns True on success."""
    ck = torch.load(ckpt_path, map_location="cpu")
    tr_layers = int(ck.get("tr_layers", 2))
    res = int(ck.get("res", 0))
    device = next(mimi.parameters()).device
    dtype = next(mimi.parameters()).dtype

    trunk = mimi.decoder_transformer.transformer
    layers = list(trunk.layers)
    if tr_layers < len(layers):
        trunk.layers = torch.nn.ModuleList(layers[:tr_layers])
        if verbose:
            print(f"[mimi_dec_student] decoder_transformer {len(layers)} -> {tr_layers} layers",
                  flush=True)

    if res == 0:
        from moshi.modules import SEANetDecoder
        mimi.decoder = SEANetDecoder(**_seanet_kwargs_no_res()).to(device=device, dtype=dtype)
        if verbose:
            print("[mimi_dec_student] SEANet decoder rebuilt with n_residual_layers=0", flush=True)

    # Key-name drift between moshi versions: the checkpoint was written by a fork
    # where attention input projections are a bare Parameter
    # (`self_attn.in_proj_weight`), while our vendored moshi uses an nn.Linear
    # (`self_attn.in_proj.weight`). Without this remap BOTH student layers'
    # attention weights load as "unexpected" and silently keep the TEACHER
    # weights -- a half-installed student that looks fine in the log.
    def _remap(sd, model):
        have = set(model.state_dict().keys())
        out, fixed = {}, 0
        for k, v in sd.items():
            if k not in have:
                for a, b in ((".in_proj_weight", ".in_proj.weight"),
                             (".in_proj.weight", ".in_proj_weight")):
                    if k.endswith(a) and k[: -len(a)] + b in have:
                        k = k[: -len(a)] + b
                        fixed += 1
                        break
            out[k] = v
        return out, fixed

    dec_sd, f1 = _remap(ck["decoder"], mimi.decoder)
    tr_sd, f2 = _remap(ck["decoder_transformer"], mimi.decoder_transformer)
    if verbose and (f1 or f2):
        print(f"[mimi_dec_student] remapped {f1 + f2} in_proj key(s) for this "
              "moshi version", flush=True)
    r1 = mimi.decoder.load_state_dict(dec_sd, strict=True)
    r2 = mimi.decoder_transformer.load_state_dict(tr_sd, strict=False)
    if r2.unexpected_keys or [k for k in r2.missing_keys if "in_proj" in k]:
        print(f"[mimi_dec_student] WARNING still unmatched: "
              f"unexpected={list(r2.unexpected_keys)[:4]} "
              f"missing={[k for k in r2.missing_keys if 'in_proj' in k][:4]}", flush=True)
    if verbose:
        n = sum(p.numel() for p in mimi.decoder.parameters())
        nt = sum(p.numel() for p in mimi.decoder_transformer.parameters())
        print(f"[mimi_dec_student] loaded decoder ({n:,} params) + decoder_transformer "
              f"({nt:,} params); unexpected={len(r2.unexpected_keys)}", flush=True)
    mimi.eval()
    return True
