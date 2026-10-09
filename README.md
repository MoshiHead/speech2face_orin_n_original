# speech2face_orin: PersonaPlex + IMTalker real-time talking avatar on a Jetson AGX Orin

Full-duplex speech-to-speech (NVIDIA PersonaPlex 7B) driving a talking-head avatar in the
browser, running fully on a Jetson AGX Orin in real time. You talk; it answers in the Robert_5
voice while the face lip-syncs. You can talk over it (barge-in).

Tested on a Jetson AGX Orin 64 GB (JetPack 6.2, L4T R36.5) with `./setup_env.sh` -> `./prepare.sh` ->
`./run.sh` -> `tools/test_session.py`: PASS, ~67 ms per 80 ms audio step (0 % over budget), 25 fps video,
audio packets at most ~4 ms late. A 9-minute, 51-turn live conversation on the same code ran at 67.5 ms/step.

This is the **best-audio Orin configuration** (the one in production on our Orin):

| part | what |
|---|---|
| trunk | PersonaPlex 7B, weights quantized to int4 at load (group 32) |
| depformer | **one-pass Robert_5 depformer** (all 8 audio codebooks in one pass, plausibility-trained), int4 gating |
| audio decoder | **Robert_5 GAN fine-tuned** Mimi decoder; its SEANet runs as a **TensorRT** engine built on the device |
| audio encoder | Mimi encoder CUDA graph, captured at start-up (fixes the old mid-session crash) |
| avatar | causal motion student + distilled single-identity renderer |
| speed | ~67 ms of the 80 ms budget per audio step (p90 ~70, none over 80); 0 crashes in a 15 000-step soak |

---

## Quick start (exact commands)

```bash
unzip speech2face_orin.zip
cd speech2face_orin
chmod +x setup_env.sh prepare.sh run.sh run_watchdog.sh   # only if your unzip dropped permissions
./setup_env.sh                                # once, ~20-40 min: Python env from scratch in ./env (Miniforge)
./prepare.sh hf_YOUR_TOKEN                    # once: checks the environment, ~19 GB of weights, TensorRT engine
./run_watchdog.sh &                           # serves on port 8991, auto-restarts on a crash
```

Then open **http://<orin-ip>:8991/** in Chrome, click **Start conversation**, allow the
microphone and speak. From another machine over plain HTTP the browser blocks the microphone:
use an HTTPS tunnel (`ngrok http 8991`), an SSH tunnel (`ssh -L 8991:localhost:8991 user@orin`,
then `http://localhost:8991/`), or open it on the Orin's own desktop.

`./run.sh` runs the server in the foreground instead (no auto-restart).
`prepare.sh` is safe to re-run; it skips what is already done.

## Python environment

**Default: `./setup_env.sh`** builds it from scratch inside this folder, nothing outside is touched
(no apt, no sudo):

1. installs Miniforge (conda-forge, free) into `./.miniforge`
2. creates a **Python 3.10** env in `./env` (3.10 because JetPack 6's TensorRT bindings are built for it)
3. makes JetPack's TensorRT importable in that env (links only `tensorrt*` from `/usr/lib/python3.10/dist-packages`)
4. installs NVIDIA's Jetson builds of torch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 / triton 3.4.0
   **only** from `https://pypi.jetson-ai-lab.io/jp6/cu126` (`requirements-jetson-torch.txt`; PyPI's torch for
   aarch64 is CPU-only), then the pinned packages in `requirements.txt`
5. checks torch sees the GPU and everything imports

Downloads ~3 GB; all caches go into `./.cache` (the Orin's system eMMC is often nearly full).
`prepare.sh` and `run.sh` then use `./env` automatically. Re-running is safe (it skips finished steps).
To remove the environment: `rm -rf env .miniforge .jetpack_trt .cache`.

**Alternative: your own Python 3.10** (system python3, a venv with `--system-site-packages`, or
an existing conda env) where `import tensorrt` works:
```bash
pip install --no-deps -r requirements-jetson-torch.txt   # FIRST, Jetson index only (PyPI's torch is CPU-only here)
pip install -r requirements.txt
PY=/path/to/python ./prepare.sh hf_...
```

`sphn` 0.2.1 has no aarch64 wheel on PyPI, so a prebuilt copy is bundled in `src/pylibs/` and used
automatically.

## Requirements

| what | needed |
|---|---|
| device | **Jetson AGX Orin 64 GB** (uses ~26 GB of unified memory). Smaller Orins (AGX 32 GB, Orin NX/Nano) are not supported: too little memory and too slow for real time |
| software | **JetPack 6.1 or 6.2** (L4T R36.4+/R36.5, **CUDA 12.6**; JetPack 6.0 has CUDA 12.2 and does not work with the torch builds used). Tested on JetPack 6.2.2. The standard full JetPack install (`nvidia-jetpack`) provides what is needed: TensorRT 10.3 with its python3.10 bindings (`python3-libnvinfer`) and `trtexec` (`libnvinfer-bin`, at `/usr/src/tensorrt/bin/trtexec`). The Python environment is created by `./setup_env.sh`. |
| power mode | MAXN (`sudo nvpmodel -m 0; sudo jetson_clocks`) for real-time speed |
| disk | ~30 GB free on the drive you unzip to (weights ~19 GB, env ~6 GB; the zip itself is only code) |
| internet | github.com (Miniforge), conda-forge, PyPI, pypi.jetson-ai-lab.io, huggingface.co |
| Hugging Face | a token whose account has **accepted the licence of `nvidia/personaplex-7b-v1`** (gated; accept on the model page first). Or run `hf auth login` beforehand and call `./prepare.sh` with no argument. |

## What `prepare.sh` does (5 steps)

1. checks the environment: aarch64, Python 3.10, every import (torch with CUDA, triton,
   TensorRT, sphn 0.2.x, ...), and that `torch.compile` works on the GPU. Installs nothing; on a
   failure it tells you to install `requirements.txt`
2. downloads `nvidia/personaplex-7b-v1` (bf16 model, Mimi codec, tokenizer, voices) into `hf/`
3. downloads our Robert_5 depformer + GAN decoder from `niloy629/personaplex-parallel-depformer`
   (`robert5_gan/`, public, sha256-checked)
4. downloads the avatar weights, the Robert_5 voice and the decoder ONNX from the same repo
   (`avatar_assets/`, public, ~250 MB, sha256-checked)
5. builds the TensorRT engine for the audio decoder from that `seanet_w12.onnx` **on this
   device** (engines are not portable between TensorRT versions) and checks it loads

Last line on success: `==> READY ...`. Any failure prints `ERROR: <reason>` and stops.

## Verify it works (headless, no browser needed)

Start the server, wait until it has loaded (~1 min on the Orin), then run the test client:

```bash
cd speech2face_orin
./run_watchdog.sh &                              # or: ./run.sh > run.log 2>&1 &
until curl -s http://localhost:8991/health | grep -q '"loaded":true'; do sleep 10; done
$(cat .python_path) tools/test_session.py 8991 40
```

It plays a 20 s spoken question (then silence) into the server and prints, e.g.:

```
sent 40s of audio | received 471 audio msgs, 940 video frames (21.9 fps)
heard 17.2s of speech | said 17.2s | step time median 66.7 ms (budget 80)
model said: ...
PASS
```

The answer differs every run. PASS = it heard the question, answered with speech and streamed
video at > 15 fps; the step time must stay below 80 ms for real time.

The server log (`run.log`) must contain these lines (if one is missing, that feature is off):

```bash
grep -aE "orin:|int4 G=32|GATING|TRY19_PD hook|fine-tune hook|captured AT INIT|trt_seanet|Uvicorn" run.log
```
```
orin: int4 trunk + robert5 one-pass depformer + GAN decoder (TensorRT) voice=Robert_5.pt port=8991
[try19] int4 G=32: 128 Linears at load ...
[try19] depformer GATING int4: 192 Linears ...
[try19] TRY19_PD hook installed -> .../weights/pd_quant_onpol_robert5_plaus2.pt ...
[try19] Mimi decoder fine-tune hook installed -> .../weights/mimi_dec_gan_robert5.pt
[try19] mimi encode graph captured AT INIT: True
[trt_seanet] mimi.decoder -> TensorRT .../weights/seanet_w12_fp32.engine (window 12 steps = 480 ms)
INFO:     Uvicorn running on http://0.0.0.0:8991
```

## Options (environment variables for `run.sh` / `run_watchdog.sh`)

| variable | default | effect |
|---|---|---|
| `PORT` | 8991 | server port (8998 is refused by the server: reserved for a legacy production server) |
| `VOICE` | `Robert_5.pt` | voice prompt from `weights/voices/` (the depformer and decoder are trained for Robert_5; other voices sound worse) |
| `TEXT_PROMPT` | Robert, RB Labs assistant | persona / system prompt |
| `PREBUFFER` | 3 | pre-roll chunks of 80 ms |
| `HF_HOME` | `./hf` | Hugging Face cache (must match what `prepare.sh` used) |

Stop everything: `pkill -f run_watchdog.sh; pkill -f speech2face_orin/src`

## Layout

```
prepare.sh, run.sh, run_watchdog.sh   setup, launch, launch with auto-restart
src/niloys/        our live server: one-pass depformer, encoder graph, TensorRT decoder, int4, student/renderer
src/imtalker/      IMTalker live-pipeline code the server imports
src/moshi_pkg/     the PersonaPlex moshi package
assets/            avatar weights, Robert_5 voice, seanet_w12.onnx (downloaded by prepare.sh)
tools/             test_session.py + test_question.wav (headless end-to-end check)
setup_env.sh       creates the Python env in ./env (Miniforge, Python 3.10, Jetson torch, requirements)
requirements-jetson-torch.txt   torch/torchaudio/torchvision/triton -- NVIDIA Jetson index only
requirements.txt   the other Python packages
env/, .miniforge/  created by setup_env.sh
hf/, weights/      created by prepare.sh (weights/ also holds the device-built TensorRT engine)
captures/, crashlogs/, run.log         created at run time
```

## Troubleshooting

| symptom | fix |
|---|---|
| `this package is for a Jetson` | wrong machine; use the desktop package on x86 GPUs |
| `trtexec not found` / `No module named tensorrt` | JetPack TensorRT not fully installed (`sudo apt install tensorrt`) |
| `environment incomplete` at step 1 | run `./setup_env.sh` (or install `requirements.txt` into the Python you pass as `PY`; `import tensorrt` must work in it) |
| `torch has no CUDA` | torch came from PyPI (CPU-only build). Re-run `./setup_env.sh`: it replaces it with the Jetson build |
| `setup_env.sh` fails at step 4 | network to `pypi.jetson-ai-lab.io`; re-run (finished steps are skipped). Log lines above show the failing package |
| step 2 fails | token wrong, or the HF account has not accepted the `nvidia/personaplex-7b-v1` licence |
| step 5 fails | see `weights/trtexec_build.log`; delete `weights/seanet_w12_fp32.engine` and re-run `./prepare.sh` |
| choppy/crackly audio, step time > 80 ms although all hooks loaded | `uvloop` is installed in the env (e.g. via `uvicorn[standard]`): it makes every step ~13 ms slower here. `env/bin/pip uninstall uvloop` (setup_env.sh removes it and refuses to finish while it is present) |
| `CUDA driver error: out of memory` with lots of memory free | `PYTORCH_CUDA_ALLOC_CONF` must not be set on Jetson (`run.sh` unsets it) |
| slow (step time > 80 ms, choppy audio) | set MAXN + `jetson_clocks`; stop other GPU jobs; run only one server |
| browser connects then disconnects with no error | `sphn` must be 0.2.x: `run.sh` puts the bundled one in `src/pylibs` first on the path; do not remove it |

## Known behaviour

* With very long continuous speech, the avatar video (and its audio) can fall a few seconds
  behind; normal back-and-forth conversation stays ~0.1-0.15 s behind.
* The model's context window is 4 minutes; very long sessions can drift after ~4-6 minutes.
  Reconnecting starts a fresh session.
