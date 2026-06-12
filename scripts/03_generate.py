#!/usr/bin/env python3
"""
03_generate.py
Batch-generate raw audio from your fine-tuned model according to a pack plan
(prompts/pack_plan.example.json). Requires stable-audio-tools installed and a
GPU (~8 GB VRAM is enough for inference).

Two ways to load a model:
  A) Fine-tuned (yours):  --model-config model_config.json --ckpt unwrapped.ckpt
     (produce unwrapped.ckpt with stable-audio-tools' unwrap_model.py)
  B) Base model sanity check:  --pretrained stabilityai/stable-audio-open-1.0

Usage:
    python 03_generate.py --model-config model_config.json --ckpt unwrapped.ckpt \
        --plan prompts/pack_plan.example.json --out generated/ --steps 100 --cfg 7
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config")
    ap.add_argument("--ckpt")
    ap.add_argument("--pretrained", help="HF model id (e.g. stabilityai/stable-audio-open-1.0)")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=-1, help="-1 = random per item")
    args = ap.parse_args()
    if not args.pretrained and not (args.model_config and args.ckpt):
        ap.error("Provide either --pretrained, or both --model-config and --ckpt.")

    from stable_audio_tools.inference.generation import generate_diffusion_cond

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, model_config = load_model(args, device)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    out_root = Path(args.out)

    for category in plan["categories"]:
        cat_dir = out_root / category["name"]
        cat_dir.mkdir(parents=True, exist_ok=True)
        seconds = float(category.get("seconds", 4.0))
        for i in range(int(category["count"])):
            prompt = category["prompt"]
            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": seconds,
            }]
            seed = args.seed if args.seed >= 0 else torch.randint(0, 2**31 - 1, (1,)).item()
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
            )
            audio = rearrange(audio, "b d n -> d (b n)")
            # trim to requested duration (model generates the full window)
            audio = audio[:, : int(seconds * sample_rate)]
            audio = audio.to(torch.float32)
            peak = audio.abs().max().clamp(min=1e-8)
            audio = (audio / peak).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            name = f"{category['name']}_{i+1:03d}_seed{seed}.wav"
            torchaudio.save(str(cat_dir / name), audio, sample_rate)
            print(f"[{category['name']}] {i+1}/{category['count']} -> {name}")
    print(f"\nDone. Raw generations in {out_root}/ - run 04_postprocess.py next.")


if __name__ == "__main__":
    main()
