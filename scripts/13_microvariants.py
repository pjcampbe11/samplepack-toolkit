#!/usr/bin/env python3
"""
13_microvariants.py  -  Timbre-level humanization ("no two hits the same")
Generate N subtle micro-variants of every one-shot in a folder via low-strength
audio-to-audio. Feed the output to 08_beat_builder.py --rotate so every hit in
a beat is a slightly different take of the same drum - like a human drummer.

Usage (GPU + stable-audio-tools):
    python 13_microvariants.py --model-config model_config.json --ckpt hiphop_v1.ckpt \
        --input organized/drums_oneshots/kicks --variants 8 --strength 0.15 \
        --prompt "hip hop, drums oneshots, kicks, one shot" --out variants/kicks
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sat_common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sat_common.add_model_args(ap)
    ap.add_argument("--input", required=True, help="Folder of one-shot WAVs")
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.15,
                    help="Keep low (0.1-0.25) for same-drum-different-take feel")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sat_common.validate_model_args(ap, args)

    model, cfg, device = sat_common.load_model(args.model_config, args.ckpt, args.pretrained)
    sr = cfg["sample_rate"]
    wavs = sorted(Path(args.input).glob("*.wav"))
    if not wavs:
        sys.exit("No WAVs in --input.")

    for wav in wavs:
        init = sat_common.load_audio_file(wav, sr, device)
        seconds = max(init.shape[1] / sr, 0.5)
        out_dir = Path(args.out) / wav.stem
        # the original is variant 0 - rotation pools should include it
        sat_common.save_wav(init, out_dir / f"{wav.stem}_var00_original.wav", sr)
        for v in range(1, args.variants + 1):
            audio, seed = sat_common.generate(
                model, cfg, args.prompt, seconds, device, steps=args.steps,
                init_audio=init, strength=args.strength)
            audio = audio[:, : init.shape[1]]
            sat_common.save_wav(audio, out_dir / f"{wav.stem}_var{v:02d}_seed{seed}.wav", sr)
        print(f"{wav.name}: {args.variants} variants -> {out_dir}/")
    print("\nUse with:  08_beat_builder.py --library ... --rotate  (point the library's")
    print("instrument folders at these variant folders, or copy variants in).")


if __name__ == "__main__":
    main()
