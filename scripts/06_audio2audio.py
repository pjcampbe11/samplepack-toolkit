#!/usr/bin/env python3
"""
06_audio2audio.py
Audio-to-audio: feed an existing sound file to your fine-tuned model and get a
NEW sound derived from it, steered by a text prompt. Your file is used as the
diffusion init; `--strength` controls how far the output strays from it.

  --strength 0.2  -> subtle re-texture (same groove/structure, new character)
  --strength 0.5  -> clearly derived but transformed (the sweet spot for flips)
  --strength 0.8  -> loose inspiration; mostly the prompt's sound

Usage:
    python 06_audio2audio.py --model-config model_config.json --ckpt hiphop_v1.ckpt \
        --input my_break.wav \
        --prompt "hip hop, drums loops, boom bap, 90 BPM, dusty drum break, vinyl texture" \
        --strength 0.5 --variations 4 --out flipped/

Works with the base model too: --pretrained stabilityai/stable-audio-open-1.0
"""
import argparse
import json
from pathlib import Path

import torch
import torchaudio
from einops import rearrange


def load_model(args, device):
    if args.pretrained:
        from stable_audio_tools import get_pretrained_model
        model, model_config = get_pretrained_model(args.pretrained)
    else:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict
        model_config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
        model = create_model_from_config(model_config)
        model.load_state_dict(load_ckpt_state_dict(args.ckpt))
    return model.to(device).eval().requires_grad_(False), model_config


def load_audio(path, target_sr, device):
    audio, sr = torchaudio.load(str(path))
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    peak = audio.abs().max().clamp(min=1e-8)
    return (audio / peak).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config")
    ap.add_argument("--ckpt")
    ap.add_argument("--pretrained", help="HF model id instead of a local checkpoint")
    ap.add_argument("--input", required=True, help="Source WAV to derive from")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--strength", type=float, default=0.5,
                    help="0..1, how much to transform away from the source (default 0.5)")
    ap.add_argument("--variations", type=int, default=4)
    ap.add_argument("--out", default="audio2audio_out")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()
    if not args.pretrained and not (args.model_config and args.ckpt):
        ap.error("Provide either --pretrained, or both --model-config and --ckpt.")
    if not 0.0 < args.strength <= 1.0:
        ap.error("--strength must be in (0, 1].")

    from stable_audio_tools.inference.generation import generate_diffusion_cond

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, model_config = load_model(args, device)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    init_audio = load_audio(args.input, sample_rate, device)
    src_len = init_audio.shape[1]
    seconds = min(src_len / sample_rate, sample_size / sample_rate)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem

    for v in range(args.variations):
        seed = args.seed if args.seed >= 0 else torch.randint(0, 2**31 - 1, (1,)).item()
        conditioning = [{
            "prompt": args.prompt,
            "seconds_start": 0,
            "seconds_total": seconds,
        }]
        audio = generate_diffusion_cond(
            model,
            steps=args.steps,
            cfg_scale=args.cfg,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            seed=seed,
            device=device,
            init_audio=(sample_rate, init_audio),
            init_noise_level=args.strength * 10.0,  # maps 0..1 to the sampler's noise scale
        )
        audio = rearrange(audio, "b d n -> d (b n)")[:, :src_len]
        audio = audio.to(torch.float32)
        peak = audio.abs().max().clamp(min=1e-8)
        audio = (audio / peak).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        name = f"{stem}_var{v+1:02d}_s{args.strength:.2f}_seed{seed}.wav"
        torchaudio.save(str(out_dir / name), audio, sample_rate)
        print(f"{v+1}/{args.variations} -> {name}")

    print(f"\nDone. Variations in {out_dir}/ - run 04_postprocess.py on them before release.")


if __name__ == "__main__":
    main()
