#!/usr/bin/env bash
# prepare_x86.sh -- weights + TensorRT engine for the PersonaPlex + IMTalker avatar on an
# x86_64 NVIDIA GPU (RunPod, Lambda, a local workstation). The Jetson counterpart is ./prepare.sh.
#
#   ./setup_env_x86.sh                             (once: Python env from scratch in ./env)
#   ./prepare_x86.sh hf_yourtokenhere              (or `hf auth login` first and omit the token)
#   PY=/path/to/python ./prepare_x86.sh hf_...     (to use your own Python 3.10 instead of ./env)
#
# Checks the environment, downloads ~19 GB of weights, probes which fast paths this GPU supports
# and builds the audio-decoder TensorRT engine for THIS GPU. Installs nothing. Then: ./run_x86.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
TOKEN="${1:-${HF_TOKEN:-}}"
[ -n "$TOKEN" ] && export HF_TOKEN="$TOKEN" HUGGING_FACE_HUB_TOKEN="$TOKEN"
export HF_HOME="${HF_HOME:-$HERE/hf}"
# Python: $PY if given, else ./env from setup_env_x86.sh, else python3 on PATH
[ -z "${PY:-}" ] && [ -x "$HERE/env/bin/python" ] && PY="$HERE/env/bin/python"
PY="$(command -v "${PY:-python3}")" || { echo "python3 not found (run ./setup_env_x86.sh, or set PY=...)"; exit 1; }
W="$HERE/weights"
BASE_REPO=nvidia/personaplex-7b-v1
BASE_REV=fdaf4090a61cb315c138a1faee287ffd6c716309
OURS_REPO=niloy629/personaplex-parallel-depformer
log(){ printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn(){ printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die(){ printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

log "1/6  environment check ($PY)"
[ "$(uname -m)" = x86_64 ] || die "this script is for x86_64; on a Jetson use ./prepare.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /' || die "nvidia-smi failed"
"$PY" - <<'T' || die "environment incomplete -- run ./setup_env_x86.sh (see README_RUNPOD.md)"
import sys
assert sys.version_info[:2] == (3, 10), f"need Python 3.10, got {sys.version.split()[0]}"
import torch, torchaudio, torchvision, triton, sphn, fastapi, uvicorn, timm, transformers, sentencepiece, cv2, librosa, torchdiffeq, huggingface_hub
assert torch.cuda.is_available(), "torch has no CUDA"
assert hasattr(sphn, "OpusStreamReader"), "sphn too old -- need 0.2.x"
f = torch.compile(lambda x: torch.sin(x) * 2 + 1)
x = torch.randn(1024, device="cuda"); assert torch.allclose(f(x), torch.sin(x) * 2 + 1, atol=1e-5)
print(f"    python {sys.version.split()[0]} | torch {torch.__version__} (CUDA {torch.version.cuda}) | triton {triton.__version__} | torch.compile OK")
T
echo "$PY" > "$HERE/.python_path"

log "2/6  base model from $BASE_REPO (~16 GB, gated: needs your token + accepted licence)"
"$PY" - <<PYEOF || die "base download failed: is the token valid and the model licence accepted on HF?"
import os
from huggingface_hub import snapshot_download
p = snapshot_download("$BASE_REPO", revision="$BASE_REV", token=os.environ.get("HF_TOKEN"),
                      allow_patterns=["model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors",
                                      "tokenizer_spm_32k_3.model", "voices.tgz"])
open(os.path.join("$HERE", ".snap_path"), "w").write(p); print("    ->", p)
PYEOF
SNAP="$(cat "$HERE/.snap_path")"
mkdir -p "$W"
ln -sfn "$SNAP/tokenizer-e351c8d8-checkpoint125.safetensors" "$W/mimi.safetensors"
ln -sfn "$SNAP/tokenizer_spm_32k_3.model" "$W/tokenizer_spm_32k_3.model"
[ -d "$W/voices" ] || tar xzf "$SNAP/voices.tgz" -C "$W"
[ -s "$W/voices/NATM0.pt" ] || die "voices.tgz did not contain voices/NATM0.pt"

fetch() {  # fetch <hf folder> <dest dir> <files...> : download from our public repo, sha256-check, link
  "$PY" - "$@" <<PYEOF || die "download from $OURS_REPO/\$1 failed"
import os, sys, hashlib
from huggingface_hub import hf_hub_download
folder, dest, files = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(dest, exist_ok=True)
want = {}
for line in open(hf_hub_download("$OURS_REPO", f"{folder}/SHA256SUMS")):
    h, f = line.split(); want[os.path.basename(f)] = h
for f in files:
    src = hf_hub_download("$OURS_REPO", f"{folder}/{f}")
    h = hashlib.sha256()
    with open(src, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
    if h.hexdigest() != want[f]:
        raise SystemExit(f"hash mismatch for {f}")
    dst = os.path.join(dest, f)
    if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
    os.symlink(src, dst)
print(f"    {folder}: {len(files)} files, sha256 OK")
PYEOF
}

log "3/6  Robert_5 one-pass depformer + GAN audio decoder from $OURS_REPO (~2.9 GB, public)"
fetch robert5_gan "$W" pd_quant_onpol_robert5_plaus2.pt mimi_dec_gan_robert5.pt

log "4/6  avatar weights + Robert_5 voice + decoder ONNX from $OURS_REPO (~250 MB, public)"
fetch avatar_assets "$HERE/assets" motion_student_38M.pt renderer_narrow.pt silence_helium_mean.pt \
  blink_motion.pt ref.jpeg Robert_5.pt seanet_w12.onnx
for f in motion_student_38M.pt renderer_narrow.pt silence_helium_mean.pt blink_motion.pt ref.jpeg; do
  ln -sfn "$HERE/assets/$f" "$W/$f"
done
ln -sfn "$HERE/assets/Robert_5.pt" "$W/voices/Robert_5.pt"

log "5/6  GPU capability probe (int4 trunk kernels, VRAM, TensorRT)"
"$PY" "$HERE/tools/gpu_probe.py" "$HERE/.gpu_probe.env" || die "GPU probe failed"

log "6/6  TensorRT engine for the audio decoder (built for THIS GPU, ~1-5 min)"
# Optional: if it cannot be built the server keeps Mimi's PyTorch SEANet decoder (a few ms
# slower per step, identical audio), so a failure here is a warning, not an error.
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [ ! -s "$W/seanet_w12_fp32.engine" ]; then
  if [ -x "$TRTEXEC" ]; then
    "$TRTEXEC" --onnx="$HERE/assets/seanet_w12.onnx" --saveEngine="$W/seanet_w12_fp32.engine" \
      > "$W/trtexec_build.log" 2>&1 || { tail -20 "$W/trtexec_build.log"; rm -f "$W/seanet_w12_fp32.engine"; }
  else
    "$PY" "$HERE/tools/build_trt_engine.py" "$HERE/assets/seanet_w12.onnx" \
      "$W/seanet_w12_fp32.engine" > "$W/trtexec_build.log" 2>&1 \
      || { tail -20 "$W/trtexec_build.log"; rm -f "$W/seanet_w12_fp32.engine"; }
  fi
fi
if [ -s "$W/seanet_w12_fp32.engine" ]; then
  "$PY" - <<PYEOF || { warn "engine does not load -- removing it, the PyTorch decoder will be used"; rm -f "$W/seanet_w12_fp32.engine"; }
import tensorrt as trt
e = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open("$W/seanet_w12_fp32.engine", "rb").read())
assert e is not None
print("    engine OK:", [(e.get_tensor_name(i), tuple(e.get_tensor_shape(e.get_tensor_name(i)))) for i in range(e.num_io_tensors)])
PYEOF
else
  warn "no TensorRT engine (log: weights/trtexec_build.log) -- the server will use Mimi's PyTorch SEANet decoder"
fi
log "READY  --  ./run_x86.sh   then open the HTTPS URL for port \${PORT:-8991}"
