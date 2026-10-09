#!/usr/bin/env bash
# setup_env.sh -- create the Python environment for this package FROM SCRATCH, inside this folder.
#
#   ./setup_env.sh
#
# Installs Miniforge (conda-forge, free) into ./.miniforge, creates a Python 3.10 env in ./env,
# makes JetPack's TensorRT importable in it, installs NVIDIA's Jetson torch 2.8.0 / torchaudio /
# torchvision / triton 3.4.0 (requirements-jetson-torch.txt) and then requirements.txt.
# Nothing is installed outside this folder (no apt, no sudo). Takes ~20-40 min (downloads ~3 GB).
# prepare.sh and run.sh use ./env automatically. Safe to re-run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
MF="$HERE/.miniforge"; ENV="$HERE/env"
MF_URL="${MF_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh}"
JP_TRT="${JP_TRT:-/usr/lib/python3.10/dist-packages}"       # JetPack's TensorRT python bindings
# every cache/temp next to the package (the Orin system eMMC is often nearly full)
export CONDA_PKGS_DIRS="$HERE/.cache/conda-pkgs" PIP_CACHE_DIR="$HERE/.cache/pip" TMPDIR="$HERE/.cache/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
log(){ printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die(){ printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

log "1/4  checks"
[ "$(uname -m)" = aarch64 ] || die "this package is for a Jetson (aarch64)"
head -1 /etc/nv_tegra_release 2>/dev/null | sed 's/^/    /' || true
[ -d "$JP_TRT/tensorrt" ] || die "JetPack TensorRT python bindings not found in $JP_TRT (JetPack 6 needed; or set JP_TRT=...)"
echo "    TensorRT bindings: $JP_TRT/tensorrt"

log "2/4  Miniforge -> $MF"
if [ ! -x "$MF/bin/conda" ]; then
  curl -fL --retry 3 -o "$TMPDIR/miniforge.sh" "$MF_URL" || die "could not download Miniforge from $MF_URL"
  bash "$TMPDIR/miniforge.sh" -b -p "$MF" > "$TMPDIR/miniforge_install.log" 2>&1 \
    || { tail -20 "$TMPDIR/miniforge_install.log"; die "Miniforge install failed"; }
  rm -f "$TMPDIR/miniforge.sh"
fi
"$MF/bin/conda" --version | sed 's/^/    /'

log "3/4  Python 3.10 env -> $ENV (3.10 because JetPack's TensorRT is built for it)"
if [ ! -x "$ENV/bin/python" ]; then
  "$MF/bin/conda" create -y -q -p "$ENV" python=3.10 pip > "$TMPDIR/conda_create.log" 2>&1 \
    || { tail -20 "$TMPDIR/conda_create.log"; die "conda create failed"; }
fi
PY="$ENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || die "env python is not 3.10"
# TensorRT only (not the whole system site-packages): link it into a folder and add that via a .pth
TRTDIR="$HERE/.jetpack_trt"; mkdir -p "$TRTDIR"
for p in "$JP_TRT"/tensorrt "$JP_TRT"/tensorrt-*.dist-info "$JP_TRT"/tensorrt_lean* "$JP_TRT"/tensorrt_dispatch*; do
  [ -e "$p" ] && ln -sfn "$p" "$TRTDIR/$(basename "$p")"
done
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
echo "$TRTDIR" > "$SITE/jetpack_tensorrt.pth"
"$PY" -c 'import tensorrt; print("    TensorRT", tensorrt.__version__, "importable")' || die "TensorRT does not import in the env"

log "4/4  Jetson torch/torchaudio/torchvision/triton (NVIDIA index only), then requirements.txt (~3 GB)"
"$PY" -m pip install -q --upgrade pip
# torch's own dependencies FIRST: the Jetson wheels below are installed with --no-deps (so pip cannot
# swap in PyPI's CPU-only build), and torch cannot even be imported without these.
"$PY" -m pip install -q "typing-extensions==4.16.0" "filelock==3.31.0" "sympy==1.14.0" "networkx==3.4.2" \
  "jinja2==3.1.6" "fsspec==2026.6.0" "numpy==1.26.4" || die "torch dependency install failed"
# torch etc. ONLY from NVIDIA's Jetson index: PyPI's torch 2.8.0 for aarch64 is a CPU-only build.
# If a CPU-only torch is already there (e.g. from an older setup_env.sh), replace it.
if ! "$PY" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  "$PY" -m pip install --force-reinstall --no-deps -r "$HERE/requirements-jetson-torch.txt" \
    || die "Jetson torch install failed (index: https://pypi.jetson-ai-lab.io/jp6/cu126)"
fi
"$PY" -c 'import torch, sys; print("    torch", torch.__version__, "CUDA", torch.version.cuda); sys.exit(0 if torch.cuda.is_available() else 1)' \
  || die "torch still has no CUDA -- not the Jetson build"
"$PY" -m pip install -r "$HERE/requirements.txt" || die "pip install -r requirements.txt failed"
# uvloop (pulled in by uvicorn[standard], e.g. from an older requirements.txt) slows every step ~13 ms here
"$PY" -m pip uninstall -y -q uvloop 2>/dev/null || true
PYTHONPATH="$HERE/src/pylibs" "$PY" - <<'T' || die "installed, but the environment check failed"
import importlib.util, torch, triton, tensorrt, torchaudio, torchvision, sphn, fastapi, timm, transformers, cv2
assert importlib.util.find_spec("uvloop") is None, "uvloop installed: remove it (pip uninstall uvloop)"
assert torch.cuda.is_available(), "torch has no CUDA"
assert hasattr(sphn, "OpusStreamReader")
print(f"    torch {torch.__version__} (CUDA {torch.version.cuda}) | triton {triton.__version__} | TensorRT {tensorrt.__version__} | {torch.cuda.get_device_name(0)}")
T
rm -rf "$TMPDIR"
log "ENV READY: $ENV   --  next: ./prepare.sh hf_YOUR_TOKEN"
