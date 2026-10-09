import argparse
import os
import subprocess

import torch

from generator.generate import InferenceAgent
from generator.train_lora import apply_lora_to_model


def clean_generator_state(checkpoint):
    state = (
        checkpoint.get("ema_state_dict")
        or checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    )
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    return {
        key.removeprefix("model."): value
        for key, value in state.items()
    }


def load_lora(agent, options):
    apply_lora_to_model(
        agent.fm,
        rank=options.lora_rank,
        alpha=options.lora_alpha,
        dropout=0.0,
        include_pose_lora=not options.no_lora_pose_projection,
        include_audio_lora=not options.no_lora_audio_projection,
        only_pose_lora=options.only_lora_pose_projection,
    )
    checkpoint = torch.load(options.lora_path, map_location="cpu")
    state = clean_generator_state(checkpoint)
    missing, unexpected = agent.fm.load_state_dict(state, strict=False)
    lora_keys = sum("lora_" in key for key in state)
    print(
        f"[preview] loaded lora={options.lora_path} "
        f"lora_keys={lora_keys} missing={len(missing)} "
        f"unexpected={len(unexpected)}",
        flush=True,
    )
    agent.fm.to(options.rank).eval()


def render(options, output_path, use_lora):
    agent = InferenceAgent(options)
    if use_lora:
        load_lora(agent, options)
    agent.run_inference(
        output_path,
        options.ref_path,
        options.aud_path,
        pose_path=None,
        gaze_path=None,
        a_cfg_scale=options.a_cfg_scale,
        nfe=options.nfe,
        crop=options.crop,
        seed=options.seed,
    )


def write_comparison(input_paths, labels, output_path):
    inputs = []
    filters = []
    for index, (path, label) in enumerate(zip(input_paths, labels)):
        inputs.extend(["-i", path])
        safe_label = label.replace("'", "")
        filters.append(
            f"[{index}:v]scale=512:512,setsar=1,"
            f"drawtext=text='{safe_label}':x=12:y=12:fontsize=22:"
            "fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=6"
            f"[v{index}]"
        )
    filter_complex = (
        ";".join(filters)
        + ";"
        + "".join(f"[v{index}]" for index in range(len(input_paths)))
        + f"hstack=inputs={len(input_paths)}[v]"
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-shortest",
            output_path,
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", required=True)
    parser.add_argument("--aud_path", required=True)
    parser.add_argument("--renderer_path", required=True)
    parser.add_argument("--generator_path", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="preview")
    parser.add_argument("--wav2vec_model_path", required=True)
    parser.add_argument("--rank", default="cuda")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--dim_w", type=int, default=32)
    parser.add_argument("--dim_c", type=int, default=32)
    parser.add_argument("--input_nc", type=int, default=3)
    parser.add_argument("--audio_marcing", type=int, default=2)
    parser.add_argument("--attention_window", type=int, default=5)
    parser.add_argument("--audio_dropout_prob", type=float, default=0.1)
    parser.add_argument("--ref_dropout_prob", type=float, default=0.1)
    parser.add_argument("--emotion_dropout_prob", type=float, default=0.1)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--dim_a", type=int, default=512)
    parser.add_argument("--dim_h", type=int, default=512)
    parser.add_argument("--dim_e", type=int, default=7)
    parser.add_argument("--dim_motion", type=int, default=32)
    parser.add_argument("--fmt_depth", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--no_learned_pe", action="store_true")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--swin_res_threshold", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--low_res_depth", type=int, default=2)
    parser.add_argument("--num_prev_frames", type=int, default=10)
    parser.add_argument("--wav2vec_sec", type=float, default=2.0)
    parser.add_argument("--only_last_features", action="store_true", default=True)
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--torchdiffeq_ode_method", default="euler")
    parser.add_argument("--a_cfg_scale", type=float, default=1.0)
    parser.add_argument("--nfe", type=int, default=5)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--fix_noise_seed", action="store_true", default=True)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=128.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_lora_pose_projection", action="store_true")
    parser.add_argument("--no_lora_audio_projection", action="store_true")
    parser.add_argument("--only_lora_pose_projection", action="store_true")
    parser.add_argument("--enable_eye_blink_composite", action="store_true")
    parser.add_argument("--blink_motion_path", default="", help="Cached blink motion latent .pt file")
    parser.add_argument("--render_sub_batch", type=int, default=8, help="Blink motion map decode batch size")
    parser.add_argument("--eye_left_x", type=float, default=0.36)
    parser.add_argument("--eye_right_x", type=float, default=0.64)
    parser.add_argument("--eye_center_y", type=float, default=0.405)
    parser.add_argument("--eye_radius_x", type=float, default=0.145)
    parser.add_argument("--eye_radius_y", type=float, default=0.070)
    parser.add_argument("--eye_feather", type=float, default=0.10)
    options = parser.parse_args()
    options.ngpus = 1
    return options


def main():
    options = parse_args()
    os.makedirs(options.out_dir, exist_ok=True)
    original_path = os.path.join(options.out_dir, f"{options.name}_original.mp4")
    lora_path = os.path.join(options.out_dir, f"{options.name}_withaudio_lora.mp4")
    comparison_path = os.path.join(
        options.out_dir,
        f"{options.name}_original_vs_withaudio_lora.mp4",
    )

    print("[preview] rendering original", flush=True)
    render(options, original_path, use_lora=False)
    torch.cuda.empty_cache()
    print("[preview] rendering with-audio LoRA", flush=True)
    render(options, lora_path, use_lora=True)
    torch.cuda.empty_cache()
    write_comparison(
        [original_path, lora_path],
        ["Original", "With-audio LoRA"],
        comparison_path,
    )
    print(f"[preview] wrote {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
