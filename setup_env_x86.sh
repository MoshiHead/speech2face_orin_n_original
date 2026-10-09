#!/usr/bin/env bash
# setup_env_x86.sh -- create the Python environment for this package on an x86_64 NVIDIA GPU box
# (RunPod, Lambda, a local workstation). The Jetson counterpart is ./setup_env.sh.
#
#   ./setup_env_x86.sh
#
# Installs Miniforge (conda-forge, free) into ./.miniforge, creates a Python 3.10 env in ./env,
# installs torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 / triton 3.4.0 built for CUDA 12.8
# (requirements-x86-torch.txt), then requirements.txt, then sphn + TensorRT from pip
# (requirements-x86-extra.txt). Nothing is installed outside this folder (no apt, no sudo).
# Takes ~8-20 min (downloads ~5 GB). prepare_x86.sh and run_x86.sh use ./env automatically.
# Safe to re-run: finished steps are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
MF="$HERE/.miniforge"; ENV="$HERE/env"
MF_URL="${MF_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"
# keep every cache next to the package so it lands on the persistent volume, not the container disk
export CONDA_PKGS_DIRS="$HERE/.cache/conda-pkgs" PIP_CACHE_DIR="$HERE/.cache/pip" TMPDIR="$HERE/.cache/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
log(){ printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die(){ printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

log "1/5  checks"
[ "$(uname -m)" = x86_64 ] || die "this script is for x86_64; on a Jetson use ./setup_env.sh"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found -- no NVIDIA driver visible in this container"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
[ -n "$(nvidia-smi --query-gpu=name --format=csv,noheader)" ] || die "no GPU visible"

log "2/5  Miniforge -> $MF"
if [ ! -x "$MF/bin/conda" ]; then
  curl -fL --retry 3 -o "$TMPDIR/miniforge.sh" "$MF_URL" || die "could not download Miniforge from $MF_URL"
  bash "$TMPDIR/miniforge.sh" -b -p "$MF" > "$TMPDIR/miniforge_install.log" 2>&1 \
    || { tail -20 "$TMPDIR/miniforge_install.log"; die "Miniforge install failed"; }
  rm -f "$TMPDIR/miniforge.sh"
fi
"$MF/bin/conda" --version | sed 's/^/    /'

log "3/5  Python 3.10 env -> $ENV (3.10: tokenizers 0.13.3 / transformers 4.30.2 have no newer wheels)"
if [ ! -x "$ENV/bin/python" ]; then
  "$MF/bin/conda" create -y -q -p "$ENV" python=3.10 pip > "$TMPDIR/conda_create.log" 2>&1 \
    || { tail -20 "$TMPDIR/conda_create.log"; die "conda create failed"; }
fi
PY="$ENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || die "env python is not 3.10"
"$PY" -m pip install -q --upgrade pip

log "4/5  torch/torchaudio/torchvision/triton for CUDA 12.8, then requirements.txt (~5 GB)"
if ! "$PY" -c 'import torch, sys; sys.exit(0 if torch.version.cuda and torch.cuda.is_available() else 1)' 2>/dev/null; then
  "$PY" -m pip install -r "$HERE/requirements-x86-torch.txt" \
    || die "torch install failed (index: https://download.pytorch.org/whl/cu128)"
fi
"$PY" -c 'import torch, sys; print("    torch", torch.__version__, "CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0)); sys.exit(0 if torch.cuda.is_available() else 1)' \
  || die "torch cannot see the GPU"
"$PY" -m pip install -r "$HERE/requirements.txt" || die "pip install -r requirements.txt failed"

log "5/5  sphn + TensorRT from pip (on a Jetson these come from src/pylibs and JetPack)"
"$PY" -m pip install -r "$HERE/requirements-x86-extra.txt" || die "sphn/TensorRT install failed"
# uvloop (pulled in by uvicorn[standard]) made every 80 ms step ~13 ms slower on the reference box
"$PY" -m pip uninstall -y -q uvloop 2>/dev/null || true
"$PY" - <<'T' || die "installed, but the environment check failed"
import importlib.util, torch, triton, torchaudio, torchvision, sphn, fastapi, timm, transformers, cv2
import tensorrt
assert importlib.util.find_spec("uvloop") is None, "uvloop installed: remove it (pip uninstall uvloop)"
assert torch.cuda.is_available(), "torch has no CUDA"
assert hasattr(sphn, "OpusStreamReader"), "sphn too old -- need 0.2.x"
cc = torch.cuda.get_device_capability(0)
print(f"    torch {torch.__version__} (CUDA {torch.version.cuda}) | triton {triton.__version__} | "
      f"TensorRT {tensorrt.__version__} | {torch.cuda.get_device_name(0)} sm_{cc[0]}{cc[1]} "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")
T
rm -rf "$TMPDIR"
log "ENV READY: $ENV   --  next: ./prepare_x86.sh hf_YOUR_TOKEN"
