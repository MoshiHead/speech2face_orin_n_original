#!/usr/bin/env bash
# prepare.sh -- weights + TensorRT engine for the PersonaPlex + IMTalker avatar on a Jetson AGX Orin.
#
#   ./setup_env.sh                             (once: Python env from scratch in ./env; or your own env, see README)
#   ./prepare.sh hf_yourtokenhere              (or `hf auth login` first and omit the token)
#   PY=/path/to/python ./prepare.sh hf_...     (if that Python is not `python3` on PATH)
#
# Checks the environment, downloads ~19 GB of weights, builds the audio-decoder TensorRT engine
# for THIS device. Installs nothing. Then: ./run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
TOKEN="${1:-${HF_TOKEN:-}}"
[ -n "$TOKEN" ] && export HF_TOKEN="$TOKEN" HUGGING_FACE_HUB_TOKEN="$TOKEN"
export HF_HOME="${HF_HOME:-$HERE/hf}"
# Python: $PY if given, else ./env from setup_env.sh, else python3 on PATH
[ -z "${PY:-}" ] && [ -x "$HERE/env/bin/python" ] && PY="$HERE/env/bin/python"
PY="$(command -v "${PY:-python3}")" || { echo "python3 not found (run ./setup_env.sh, or set PY=...)"; exit 1; }
W="$HERE/weights"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
BASE_REPO=nvidia/personaplex-7b-v1
BASE_REV=fdaf4090a61cb315c138a1faee287ffd6c716309
OURS_REPO=niloy629/personaplex-parallel-depformer
log(){ printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die(){ printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

log "1/5  environment check ($PY)"
[ "$(uname -m)" = aarch64 ] || die "this package is for a Jetson (aarch64)"
head -1 /etc/nv_tegra_release 2>/dev/null | sed 's/^/    /' || true
[ -x "$TRTEXEC" ] || die "trtexec not found at $TRTEXEC (JetPack TensorRT; or set TRTEXEC=...)"
PYTHONPATH="$HERE/src/pylibs:${PYTHONPATH:-}" "$PY" - <<'T' || die "environment incomplete -- run ./setup_env.sh (or pip install -r requirements.txt into your Python; see README)"
import sys
assert sys.version_info[:2] == (3, 10), f"need Python 3.10, got {sys.version.split()[0]}"
import torch, torchaudio, torchvision, triton, tensorrt, sphn, fastapi, uvicorn, timm, transformers, sentencepiece, cv2, librosa, torchdiffeq, huggingface_hub
assert torch.cuda.is_available(), "torch has no CUDA (not the Jetson build?)"
assert hasattr(sphn, "OpusStreamReader"), "sphn too old (bundled 0.2.1 not picked up)"
f = torch.compile(lambda x: torch.sin(x) * 2 + 1)
x = torch.randn(1024, device="cuda"); assert torch.allclose(f(x), torch.sin(x) * 2 + 1, atol=1e-5)
print(f"    python {sys.version.split()[0]} | torch {torch.__version__} (CUDA {torch.version.cuda}) | triton {triton.__version__} | TensorRT {tensorrt.__version__} | torch.compile OK")
T
echo "$PY" > "$HERE/.python_path"

log "2/5  base model from $BASE_REPO (~16 GB, gated: needs your token + accepted licence)"
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

log "3/5  Robert_5 one-pass depformer + GAN audio decoder from $OURS_REPO (~2.9 GB, public)"
fetch robert5_gan "$W" pd_quant_onpol_robert5_plaus2.pt mimi_dec_gan_robert5.pt

log "4/5  avatar weights + Robert_5 voice + decoder ONNX from $OURS_REPO (~250 MB, public)"
fetch avatar_assets "$HERE/assets" motion_student_38M.pt renderer_narrow.pt silence_helium_mean.pt \
  blink_motion.pt ref.jpeg Robert_5.pt seanet_w12.onnx
for f in motion_student_38M.pt renderer_narrow.pt silence_helium_mean.pt blink_motion.pt ref.jpeg; do
  ln -sfn "$HERE/assets/$f" "$W/$f"
done
ln -sfn "$HERE/assets/Robert_5.pt" "$W/voices/Robert_5.pt"

log "5/5  TensorRT engine for the audio decoder (built for THIS device, ~2-5 min)"
if [ ! -s "$W/seanet_w12_fp32.engine" ]; then
  "$TRTEXEC" --onnx="$HERE/assets/seanet_w12.onnx" --saveEngine="$W/seanet_w12_fp32.engine" > "$W/trtexec_build.log" 2>&1 \
    || { tail -20 "$W/trtexec_build.log"; die "TensorRT engine build failed (log: weights/trtexec_build.log)"; }
fi
"$PY" - <<PYEOF || die "engine does not load"
import tensorrt as trt
e = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open("$W/seanet_w12_fp32.engine", "rb").read())
assert e is not None
print("    engine OK:", [(e.get_tensor_name(i), tuple(e.get_tensor_shape(e.get_tensor_name(i)))) for i in range(e.num_io_tensors)])
PYEOF
log "READY  --  ./run.sh   (or ./run_watchdog.sh)   then open http://<orin>:8991/"
