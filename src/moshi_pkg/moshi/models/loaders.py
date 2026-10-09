# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Moshi and Mimi."""
from pathlib import Path
import logging

from safetensors.torch import load_model, load_file
import torch

logger = logging.getLogger(__name__)

from .compression import MimiModel
from .lm import LMModel
from ..modules import SEANetEncoder, SEANetDecoder, transformer
from ..quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

TEXT_TOKENIZER_NAME = 'tokenizer_spm_32k_3.model'
MOSHI_NAME = 'model.safetensors'
MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'
DEFAULT_REPO = 'nvidia/personaplex-7b-v1'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    # We train using weight_norm but then the weights are pre-processed for inference so
    # that we can use a normal convolution.
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}
_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}
_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}

_lm_kwargs = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,
    "card": _quantizer_kwargs["bins"],
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_causal": True,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


def prune_for_codebook_count(model: LMModel, n: int) -> None:
    """Prune depformer structures for reduced codebook inference.

    When using fewer than dep_q codebooks, the depformer only runs steps 0..n-1.
    This function removes the unused weights for steps n..dep_q-1, saving memory.

    Pruned structures:
    - depformer_in (ModuleList): projection from main transformer to depformer per CB
    - depformer_emb (ModuleList): input embeddings for depformer per CB
    - linears (ModuleList): output projection per CB
    - depformer transformer: multi_linear weights and per-step gating modules

    Args:
        model: The LMModel to prune in-place.
        n: Number of codebooks to keep (1 to dep_q).
    """
    old_dep_q = model.dep_q
    if n >= old_dep_q:
        return

    logger.info(f"Pruning depformer from {old_dep_q} to {n} codebooks")

    # 1. Prune depformer_in (ModuleList, old_dep_q entries) → keep 0..n-1
    model.depformer_in = torch.nn.ModuleList(list(model.depformer_in)[:n])

    # 2. Prune depformer_emb (ModuleList, old_dep_q-1 entries) → keep 0..n-2
    #    (depformer_emb[cb_index-1] used for cb_index 1..n-1)
    keep_emb = max(n - 1, 0)
    model.depformer_emb = torch.nn.ModuleList(list(model.depformer_emb)[:keep_emb])

    # 3. Prune linears (ModuleList, old_dep_q entries) → keep 0..n-1
    model.linears = torch.nn.ModuleList(list(model.linears)[:n])

    # 4. Prune depformer transformer multi_linear weights and gating
    for layer in model.depformer.layers:
        attn = layer.self_attn
        if attn.weights_per_step and attn.weights_per_step > n:
            old_wps = attn.weights_per_step
            embed_dim = attn.embed_dim

            # in_proj: weight shape [old_wps * 3 * embed_dim, embed_dim]
            # Keep first n slices
            in_w = attn.in_proj.weight.data
            in_w = in_w.view(old_wps, 3 * embed_dim, embed_dim)[:n]
            new_in = torch.nn.Linear(embed_dim, n * 3 * embed_dim, bias=False,
                                     device=in_w.device, dtype=in_w.dtype)
            new_in.weight.data.copy_(in_w.reshape(n * 3 * embed_dim, embed_dim))
            attn.in_proj = new_in

            # out_proj: weight shape [old_wps * embed_dim, embed_dim]
            out_w = attn.out_proj.weight.data
            out_w = out_w.view(old_wps, embed_dim, embed_dim)[:n]
            new_out = torch.nn.Linear(embed_dim, n * embed_dim, bias=False,
                                      device=out_w.device, dtype=out_w.dtype)
            new_out.weight.data.copy_(out_w.reshape(n * embed_dim, embed_dim))
            attn.out_proj = new_out

            attn.weights_per_step = n

        # Prune gating ModuleList (one per step when weights_per_step is set)
        if layer.weights_per_step and layer.weights_per_step > n:
            if isinstance(layer.gating, torch.nn.ModuleList) and len(layer.gating) > n:
                layer.gating = torch.nn.ModuleList(list(layer.gating)[:n])
            layer.weights_per_step = n

    # 5. Update dep_q on the model
    model.dep_q = n

    # Count savings
    saved_params = 0
    # depformer_in: (old_dep_q - n) linears
    saved_params += (old_dep_q - n) * model.dim * _lm_kwargs["depformer_dim"]
    # depformer_emb: (old_dep_q - 1 - keep_emb) embeddings
    saved_params += (old_dep_q - 1 - keep_emb) * (model.card + 1) * _lm_kwargs["depformer_dim"]
    # linears: (old_dep_q - n) linears
    saved_params += (old_dep_q - n) * _lm_kwargs["depformer_dim"] * model.card

    saved_mb = saved_params * 2 / (1024 ** 2)  # bf16
    logger.info(f"Pruned ~{saved_mb:.0f} MiB of depformer weights "
                f"(depformer_in, depformer_emb, linears, transformer multi_linear/gating)")


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def _remap_state_dict_keys(state_dict: dict) -> dict:
    """Remap legacy state_dict keys to match current model architecture.

    The attention module now stores in_proj as a proper nn.Linear child module,
    so 'self_attn.in_proj_weight' becomes 'self_attn.in_proj.weight'.
    """
    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace("self_attn.in_proj_weight", "self_attn.in_proj.weight")
        remapped[new_key] = value
    return remapped


def get_mimi(filename: str | Path,
             device: torch.device | str = 'cpu') -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        state_dict = load_file(filename, device=str(device))
        state_dict = _remap_state_dict_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:
        pkg = torch.load(filename, "cpu")
        state_dict = _remap_state_dict_keys(pkg["model"])
        model.load_state_dict(state_dict, strict=False)
    model.set_num_codebooks(8)
    return model


def get_moshi_lm(
    filename: str | Path | None,
    copy_missing_weights: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    delays=None,
    cpu_offload: bool = False,
    quantize_4bit: bool = False,
    num_codebooks: int = 8,
    context: int | None = None,
) -> LMModel:
    """Return a pretrained Moshi LM model.

    Args:
        filename: Path to model weights.
        copy_missing_weights: Whether to copy missing weights from existing layers.
        device: Target device for the model.
        dtype: Data type for model weights.
        delays: Optional custom delays configuration.
        cpu_offload: If True, offload model layers to CPU when GPU memory is
                     insufficient. Uses accelerate's device_map="auto".
        quantize_4bit: If True, quantize main transformer to 4-bit NF4.
        num_codebooks: Number of audio codebooks (1-8). Prunes depformer if < 8.
        context: KV cache context length. None uses default (3000).
    """
    # Copy to avoid mutating a shared/global dict
    lm_kwargs = dict(_lm_kwargs)
    lm_kwargs["dep_q"] = 16
    if delays is not None:
        lm_kwargs["delays"] = delays
    if context is not None:
        lm_kwargs["context"] = context

    if quantize_4bit and filename is not None:
        return _get_moshi_lm_quantized_4bit(
            filename, copy_missing_weights, device, dtype, lm_kwargs,
            num_codebooks=num_codebooks,
        )

    if cpu_offload and filename is not None:
        return _get_moshi_lm_with_offload(
            filename, copy_missing_weights, device, dtype, lm_kwargs
        )

    # Init with meta device to avoid init dummy memory
    init_device = "meta" if filename is not None else device
    model = LMModel(device=init_device, dtype=dtype, **lm_kwargs)
    if filename is None:
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    filename = str(filename)

    # Load state_dict
    if filename.endswith(".safetensors"):
        # safetensors does not support mps directly
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type == "mps":
            state_dict = load_file(filename, device="cpu")
        else:
            state_dict = load_file(filename, device=dev.type)
    else:
        # torch checkpoint
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
    state_dict = _remap_state_dict_keys(state_dict)
    # Patch 1: expand depformer self_attn weights if needed
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                print("Expanding %s", name)
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0] :]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    # Patch 2: fill missing keys by copying 0..7 -> 8..15 for certain groups
    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            print("Replacing %s <- %s", name, src)
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                print("Missing %s", name)

    # Assign weights to target device
    dev = torch.device(device) if isinstance(device, str) else device
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device=dev, dtype=dtype)
    
    model.load_state_dict(state_dict, strict=False, assign=True)
    if num_codebooks < 8:
        prune_for_codebook_count(model, num_codebooks)
    model.eval()
    return model.to(device=device, dtype=dtype)


def _is_prequantized_state_dict(state_dict: dict) -> bool:
    """Check if a state_dict contains pre-quantized bitsandbytes 4-bit weights."""
    return any("quant_state.bitsandbytes__" in k for k in state_dict)


def _load_prequantized_weights(
    model: torch.nn.Module,
    state_dict: dict,
    device: torch.device | str,
    dtype: torch.dtype,
) -> None:
    """Load pre-quantized bitsandbytes 4-bit weights into a model.

    Expects that Linear4bit modules have already been created via
    _replace_linears_with_4bit. Reconstructs Params4bit with QuantState
    for each quantized layer, and loads remaining weights normally.
    """
    import bitsandbytes as bnb

    dev = torch.device(device) if isinstance(device, str) else device

    # Find all quantized weight prefixes by looking for quant_state metadata keys.
    # Key format: "{module_path}.weight.quant_state.bitsandbytes__nf4"
    quant_prefixes = set()
    for key in state_dict:
        if "quant_state.bitsandbytes__" in key:
            prefix = key.split(".weight.quant_state.")[0]
            quant_prefixes.add(prefix)

    logger.info(f"Found {len(quant_prefixes)} pre-quantized layers")

    # For each quantized layer, collect weight data and metadata, then
    # reconstruct Params4bit using the from_prequantized API which sets
    # bnb_quantized=True (preventing re-quantization on device move).
    keys_consumed = set()
    for prefix in quant_prefixes:
        weight_key = prefix + ".weight"

        # Collect quantization metadata (everything under {prefix}.weight.*)
        metadata = {}
        for key in state_dict:
            if key.startswith(weight_key + "."):
                meta_key = key[len(weight_key) + 1:]
                metadata[meta_key] = state_dict[key]
                keys_consumed.add(key)
        keys_consumed.add(weight_key)

        # Navigate to the module
        module = model
        for part in prefix.split("."):
            module = getattr(module, part)

        module.weight = bnb.nn.Params4bit.from_prequantized(
            data=state_dict[weight_key],
            quantized_stats=metadata,
            requires_grad=False,
            device=dev,
            module=module,
        )

    # Load remaining (non-quantized) weights normally
    remaining = {k: v.to(device=dev, dtype=dtype)
                 for k, v in state_dict.items()
                 if k not in keys_consumed}
    model.load_state_dict(remaining, strict=False, assign=True)


def _replace_linears_with_4bit(
    model: torch.nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    min_size: int = 4096,
) -> None:
    """Replace nn.Linear modules with bitsandbytes Linear4bit (NF4) in-place.

    Skips layers where both dimensions are below min_size to avoid quantizing
    small projection heads and depformer layers where the quality loss isn't
    worth the memory savings.
    """
    import bitsandbytes as bnb

    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            if max(child.in_features, child.out_features) < min_size:
                continue
            on_meta = child.weight.device.type == "meta"
            bnb_linear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                compress_statistics=True,
                device="meta" if on_meta else None,
            )
            if not on_meta:
                bnb_linear.weight = bnb.nn.Params4bit(
                    child.weight.data,
                    requires_grad=False,
                    quant_type="nf4",
                    compress_statistics=True,
                )
                if child.bias is not None:
                    bnb_linear.bias = child.bias
            setattr(model, name, bnb_linear)
        else:
            _replace_linears_with_4bit(child, compute_dtype, min_size)


def _get_moshi_lm_quantized_4bit(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
    num_codebooks: int = 8,
) -> LMModel:
    """Load Moshi LM with 4-bit NF4 quantization via bitsandbytes.

    Supports two checkpoint formats:
    - Standard (safetensors/torch): Loads bf16 weights on CPU, replaces large
      Linear layers with Linear4bit, then moves to GPU where bitsandbytes
      quantizes on the fly.
    - Pre-quantized (torch checkpoint with bitsandbytes metadata): Detects
      quantization metadata keys, creates Linear4bit modules first, then
      reconstructs Params4bit via from_prequantized() to avoid re-quantization.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401
    except ImportError:
        raise ImportError(
            "4-bit quantization requires the 'bitsandbytes' package. "
            "Install it with: pip install bitsandbytes"
        )

    filename = str(filename)
    logger.info("Loading model with 4-bit NF4 quantization")

    # Load state_dict to CPU
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device="cpu")
    else:
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
    state_dict = _remap_state_dict_keys(state_dict)

    dev = torch.device(device) if isinstance(device, str) else device

    if _is_prequantized_state_dict(state_dict):
        # Pre-quantized checkpoint: weights are already packed uint8 with
        # quantization metadata. All weight patches (depformer expansion,
        # missing key copying) were applied before the checkpoint was saved.
        logger.info("Detected pre-quantized 4-bit checkpoint")

        model = LMModel(device="meta", dtype=dtype, **lm_kwargs)
        _replace_linears_with_4bit(model.transformer, compute_dtype=dtype)
        _load_prequantized_weights(model, state_dict, device=dev, dtype=dtype)
        del state_dict

        if num_codebooks < 8:
            prune_for_codebook_count(model, num_codebooks)

        model.eval()

        if dev.type == "cuda":
            allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
            logger.info(f"GPU memory after pre-quantized load: {allocated:.1f} GiB")

        return model

    # Standard checkpoint: load bf16 weights, patch, quantize on the fly.
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    # Apply weight patches (same as standard path)
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                logger.warning(f"Missing {name}")

    # Load weights on CPU in bf16
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device="cpu", dtype=dtype)
    model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    # Prune depformer before quantization (saves memory on GPU)
    if num_codebooks < 8:
        prune_for_codebook_count(model, num_codebooks)

    # Only quantize the main transformer — it holds the vast majority of
    # parameters.  The depformer uses multi_linear (raw weight reshaping)
    # which is incompatible with compressed 4-bit formats, and the embedding/
    # output-head layers are small enough to keep in bf16.
    _replace_linears_with_4bit(model.transformer, compute_dtype=dtype)

    n_4bit = sum(
        1 for m in model.modules()
        if hasattr(m, 'weight') and hasattr(m.weight, 'quant_type')
    )
    logger.info(f"Quantized {n_4bit} linear layers to 4-bit NF4")

    # Move to GPU — bitsandbytes quantizes weights on the fly during .cuda()
    model = model.to(dev)
    model.eval()

    if dev.type == "cuda":
        allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
        logger.info(f"GPU memory after quantized load: {allocated:.1f} GiB")

    return model


def _get_moshi_lm_with_offload(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    lm_kwargs: dict,
) -> LMModel:
    """Load Moshi LM with CPU offloading using accelerate.

    This function distributes model layers across GPU and CPU based on
    available GPU memory. Layers that don't fit on GPU are kept on CPU
    and moved to GPU only during forward pass.
    """
    try:
        from accelerate import infer_auto_device_map, dispatch_model
    except ImportError:
        raise ImportError(
            "CPU offloading requires the 'accelerate' package. "
            "Install it with: pip install accelerate"
        )

    filename = str(filename)
    logger.info("Loading model with CPU offloading enabled")

    # First, create model on CPU to get the architecture
    model = LMModel(device="cpu", dtype=dtype, **lm_kwargs)

    # Load state_dict to CPU
    if filename.endswith(".safetensors"):
        state_dict = load_file(filename, device="cpu")
    else:
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
    state_dict = _remap_state_dict_keys(state_dict)

    # Apply weight patches (same as non-offload path)
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.info(f"Expanding {name}")
                missing = (
                    tensor
                    if copy_missing_weights
                    else model_sd[name][tensor.shape[0]:]
                )
                state_dict[name] = torch.concat([tensor, missing], dim=0)

    if copy_missing_weights:
        to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
        for name in model_sd.keys():
            if name in state_dict:
                continue
            replaced = False
            for old, new in zip(range(8), range(8, 16)):
                for rep in to_replace:
                    needle = f"{rep}.{new}."
                    if needle in name:
                        src = name.replace(needle, f"{rep}.{old}.")
                        if src in state_dict:
                            logger.info(f"Replacing {name} <- {src}")
                            state_dict[name] = state_dict[src]
                            replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                logger.warning(f"Missing {name}")

    model.load_state_dict(state_dict, strict=False, assign=True)

    # Determine target device
    dev = torch.device(device) if isinstance(device, str) else device

    if dev.type != "cuda":
        # If not using CUDA, just move to the target device without offloading
        logger.info(f"CPU offload requested but device is {dev}, skipping offload")
        model.to(dev)
        model.eval()
        return model

    # Reserve GPU memory for KV cache that gets allocated after model loading.
    # Each transformer layer's KV cache is:
    #   2 * batch_size * num_heads * context * dim_per_head * bytes_per_element
    # For the default config (32 heads, context=3000, dim_per_head=128, bf16):
    #   ~48 MiB per layer, ~1.5 GiB total for 32 layers.
    # Reserve 2 GiB to cover KV cache + runtime overhead.
    free_mem, _ = torch.cuda.mem_get_info(dev.index or 0)
    kv_cache_reserve = 2 * (1024 ** 3)  # 2 GiB
    usable_gpu = max(free_mem - kv_cache_reserve, 0)

    # Infer device map based on available GPU memory
    device_map = infer_auto_device_map(
        model,
        max_memory={0: usable_gpu, "cpu": "32GiB"},
        no_split_module_classes=["StreamingTransformerLayer"],
        dtype=dtype,
    )

    # Log the device distribution
    gpu_layers = sum(1 for v in device_map.values() if v == 0 or v == "cuda:0")
    cpu_layers = sum(1 for v in device_map.values() if v == "cpu")
    logger.info(f"Device map: {gpu_layers} modules on GPU, {cpu_layers} modules on CPU")

    # Dispatch model across devices
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir="offload_weights",  # Directory for disk offload if needed
    )

    model.eval()
    return model
