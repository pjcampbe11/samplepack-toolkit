#!/usr/bin/env python3
"""
22_sa3_workflow.py  -  Stable Audio 3 workflow (June 2026 generation core)
SA3 (released 2026-05-20) upgrades the toolkit's generation: open weights
(Small 433M / Medium 1.4B), trained on licensed data, LoRA fine-tuning
(~16 GB VRAM, ~1000 steps), variable-length up to 380 s, and two NEW modes:
inpainting (regenerate a region in place) and continuation (extend a clip).

Run INSIDE the stable-audio-3 repo environment:
    git clone https://github.com/Stability-AI/stable-audio-3 && cd stable-audio-3
    uv sync --extra lora        # see cloud/sa3_setup.sh
    uv run python /path/to/22_sa3_workflow.py <subcommand> ...

Subcommands:
  prepare  toolkit dataset (.json sidecars) -> SA3 LoRA data_dir (.txt captions)
  plan     batch text-to-audio from a pack plan (03_generate.py equivalent)
  flip     audio-to-audio (06_audio2audio.py equivalent)
  fill     inpaint a time region ("replace bars 2-3 with a punchy kick fill")
  extend   continuation (stretch a loop past its original end)

All generation subcommands accept --lora my.safetensors --lora-strength 0.8.
After LoRA training (scripts/train_lora.py in the SA3 repo), point --lora at
the produced .safetensors. Outputs still flow into 04_postprocess.py -> 05.
"""
import argparse
import json
import shutil
from pathlib import Path


def get_model(args):
    from stable_audio_3 import StableAudioModel
    model = StableAudioModel.from_pretrained(args.model)
    if args.lora:
        from stable_audio_3.models.lora import set_lora_strength
        model.load_lora(args.lora)          # API per SA3 docs; adjust if renamed
        set_lora_strength(model, args.lora_strength)
    return model


def save(audio, path, sr=44100):
    import torch
    import torchaudio
    if isinstance(audio, tuple):
        audio, sr = audio[0], audio[1]
    audio = audio.to(torch.float32).cpu()
    if audio.dim() == 3:
        audio = audio[0]
    peak = audio.abs().max().clamp(min=1e-8)
    audio = (audio / peak).clamp(-1, 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio, sr)


def add_common(p):
    p.add_argument("--model", default="medium", help="small-music | medium | medium-base")
    p.add_argument("--lora", help=".safetensors LoRA checkpoint")
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--out", required=True)


def cmd_prepare(args):
    """Toolkit dataset (01_prepare_dataset.py output) -> SA3 train_lora data_dir."""
    src, dst = Path(args.dataset), Path(args.data_dir)
    n = 0
    for wav in sorted(src.rglob("*.wav")):
        sidecar = wav.with_suffix(".json")
        if not sidecar.exists():
            continue
        prompt = (json.loads(sidecar.read_text(encoding="utf-8")).get("prompt") or "").strip()
        if not prompt:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        flat = "__".join(wav.relative_to(src).parts)
        shutil.copy2(wav, dst / flat)
        (dst / flat).with_suffix(".txt").write_text(prompt, encoding="utf-8")
        n += 1
    print(f"{n} clips staged in {dst} (audio + .txt captions).")
    print("NOTE: confirm the caption format your SA3 version expects with")
    print("  uv run python scripts/train_lora.py --help   (run inside the SA3 repo)")
    print("Then train:  uv run python scripts/train_lora.py --model medium-base \\")
    print(f"    --data_dir {dst} --rank 16 --adapter_type dora-rows --steps 1000 \\")
    print("    --exclude seconds_total   # prevents conditioner hijack on small datasets")


def cmd_plan(args):
    model = get_model(args)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    out_root = Path(args.out)
    for cat in plan["categories"]:
        seconds = float(cat.get("seconds", 4.0))
        for i in range(int(cat["count"])):
            audio = model.generate(prompt=cat["prompt"], duration=seconds)
            save(audio, out_root / cat["name"] / f"{cat['name']}_{i+1:03d}.wav")
            print(f"[{cat['name']}] {i+1}/{cat['count']}")
    print(f"Done -> {out_root}. Next: 04_postprocess.py")


def cmd_flip(args):
    import torchaudio
    model = get_model(args)
    init = torchaudio.load(args.input)
    audio = model.generate(init_audio=init, init_noise_level=args.strength,
                           prompt=args.prompt, duration=args.duration)
    save(audio, args.out)
    print(f"Flipped -> {args.out}")


def cmd_fill(args):
    import torchaudio
    model = get_model(args)
    inp = torchaudio.load(args.input)
    audio = model.generate(inpaint_audio=inp,
                           inpaint_mask_start_seconds=args.start,
                           inpaint_mask_end_seconds=args.end,
                           prompt=args.prompt, duration=args.duration)
    save(audio, args.out)
    print(f"Inpainted {args.start}-{args.end}s -> {args.out}")


def cmd_extend(args):
    import torchaudio
    model = get_model(args)
    inp = torchaudio.load(args.input)
    src_len = inp[0].shape[-1] / inp[1]
    audio = model.generate(inpaint_audio=inp,
                           inpaint_mask_start_seconds=src_len,
                           inpaint_mask_end_seconds=args.duration,
                           prompt=args.prompt, duration=args.duration)
    save(audio, args.out)
    print(f"Extended {src_len:.1f}s -> {args.duration}s -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-dir", required=True)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("plan")
    add_common(p)
    p.add_argument("--plan", required=True)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("flip")
    add_common(p)
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--strength", type=float, default=0.7,
                   help="SA3 init_noise_level 0..1 (0.9 = heavy transform)")
    p.add_argument("--duration", type=float, default=30)
    p.set_defaults(fn=cmd_flip)

    p = sub.add_parser("fill")
    add_common(p)
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--duration", type=float, default=30)
    p.set_defaults(fn=cmd_fill)

    p = sub.add_parser("extend")
    add_common(p)
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=float, required=True, help="New total length (s)")
    p.set_defaults(fn=cmd_extend)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
