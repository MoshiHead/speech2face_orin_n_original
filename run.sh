#!/usr/bin/env bash
# run.sh -- PersonaPlex + IMTalker real-time avatar on a Jetson AGX Orin (best-audio configuration):
#   7B trunk int4 (G=32) + Robert_5 one-pass depformer (plausibility-trained, int4 gating)
#   + Robert_5 GAN Mimi decoder with a TensorRT SEANet, Mimi encoder CUDA graph captured at
#   start-up, causal motion student + distilled renderer. ~66 ms per 80 ms audio step. Port 8991.
#
#   ./run.sh                 defaults
#   PORT=8992 ./run.sh       another port (not 8998: refused)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"; W="$HERE/weights"
[ -s "$HERE/.snap_path" ] && [ -s "$HERE/.python_path" ] && [ -s "$W/seanet_w12_fp32.engine" ] || { echo "run ./prepare.sh <hf_token> first"; exit 1; }
PY="${PY:-$(cat "$HERE/.python_path")}"   # the Python prepare.sh checked
SNAP="$(cat "$HERE/.snap_path")"
PORT="${PORT:-8991}"   # 8998 is refused by the server (reserved for a legacy production server)
export HF_HOME="${HF_HOME:-$HERE/hf}"
# compiled-kernel caches next to the package (the Orin system eMMC is often nearly full)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HERE/.cache/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HERE/.cache/inductor}"
export DESIGN_A_DIR="$SRC/niloys" IMTALKER_DIR="$SRC/imtalker"
export PYTHONPATH="$SRC/pylibs:$SRC/niloys:$SRC/moshi_pkg:${PYTHONPATH:-}"   # pylibs: bundled sphn 0.2.1 (aarch64)
# expandable_segments uses CUDA VMM APIs unsupported on Tegra -> spurious OOM. Keep it unset.
unset PYTORCH_CUDA_ALLOC_CONF TRY19_DEC_CKPT
export NATIVE_PASSTHROUGH=1
export TRY19_STUDENT_GRAPH=1 TRY19_RENDER_FP16=1 TRY19_INT4_G=32 TRY19_PD_REFINE=0
export TRY19_PD="$W/pd_quant_onpol_robert5_plaus2.pt"
export TRY19_MIMI_DEC_FT="$W/mimi_dec_gan_robert5.pt"
export TRY19_TRT_SEANET="$W/seanet_w12_fp32.engine"
export TRY19_MIMI_GRAPH=1 TRY19_MIMI_PARTS=seanet_quant TRY19_MIMI_CAPTURE_AT_INIT=1
export TRY19_DEP_GATING_INT4=1 TRY19_CODE_GUARD=1
mkdir -p "$SRC/niloys/live/static" "$HERE/captures"
ln -sfn "$SRC/imtalker/static/assets" "$SRC/niloys/live/static/assets"
ln -sfn "$SRC/niloys/static/index_try21_no_idle_dissolve.html" "$SRC/niloys/live/static/index.html"
echo "orin: int4 trunk + robert5 one-pass depformer + GAN decoder (TensorRT) voice=${VOICE:-Robert_5.pt} port=$PORT"
exec "$PY" -u "$SRC/niloys/live/imtalker_try19_student_live.py" \
  --live_module imtalker_native_passthrough \
  --student_ckpt "$W/motion_student_38M.pt" --distilled_renderer "$W/renderer_narrow.pt" \
  --motion_gain "${GAIN:-1.2}" --dep_q 8 \
  --blend_frames 0 --jump_thresh 0.15 --mouth_close 0.5 \
  --host 0.0.0.0 --port "$PORT" \
  --html_path "$SRC/niloys/static/index_try21_no_idle_dissolve.html" \
  --generator_path "$W/generator_last.ckpt" --renderer_path "$W/renderer.ckpt" \
  --adapter_path "$W/adapter_last.pt" \
  --adapter_type unitalk_last_layer --adapter_num_layers 12 --adapter_dropout 0.0 \
  --adapter_window_mode lookahead --adapter_future_steps 0 \
  --ref_path "$W/ref.jpeg" --wav2vec_model_path "$W/wav2vec2-base-960h" \
  --moshi_root "$W" --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$SNAP/model.safetensors" --mimi_weight "$W/mimi.safetensors" \
  --tokenizer "$W/tokenizer_spm_32k_3.model" \
  --text_prompt "${TEXT_PROMPT:-You are Robert, an assistant from RB Labs. You have knowledge about crypto, finance, investment, technology, and general topics. Answer every question completely, descriptively, and with useful detail.}" \
  --voice_prompt "${VOICE:-Robert_5.pt}" --voice_prompt_dir "$W/voices" \
  --enable_moshi_reply --direct_reply_hidden --reply_hidden_steps_per_chunk 1 \
  --audio_chunk_sec 2.0 --wav2vec_sec 2.0 --fm_chunk_frames 2 --helium_deque_size 25 \
  --prebuffer_chunks "${PREBUFFER:-3}" --render_sub_batch 2 --renderer_precision fp32 \
  --frame_q_backpressure 3 --buffer_ms 80 --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold 0.006 --assistant_speech_hold_chunks 1 \
  --motion_ref_blend 0.0 --motion_prior_noise_blend 0.0 \
  --a_cfg_scale 1.24 --nfe 3 --seed 42 --noise_seed 42 --shared_noise \
  --fp32 --tf32 \
  --silence_helium_path "$W/silence_helium_mean.pt" \
  --jpeg_quality 90 --device cuda --reply_audio_gain 1.0 --output_audio_codec opus \
  --capture_dir "$HERE/captures" \
  --blink_motion_path "$W/blink_motion.pt"
