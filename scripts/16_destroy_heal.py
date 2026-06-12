#!/usr/bin/env python3
"""
16_destroy_heal.py  -  The destruction-and-heal sandwich
1. DESTROY: batch-process audio through a deliberately extreme VST/builtin chain
   (09_vst_chain.py) - bitcrush, broken tape, GSM codec, whatever.
2. HEAL: low-strength audio-to-audio pulls the wreckage back toward musicality
   using your fine-tuned model. The model acts as restoration glue; the scars
   that survive ARE the texture.

Usage:
    python 16_destroy_heal.py --model-config model_config.json --ckpt hiphop_v1.ckpt \
        --input loops/ --chain configs/vst_chain.destroy.example.json \
        --prompt "hip hop, melodic loops, dusty vinyl, warm analog texture" \
        --heal-strength 0.25 --out healed/
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sat_common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sat_common.add_model_args(ap)
    ap.add_argument("--input", required=True)
    ap.add_argument("--chain", required=True, help="Destroy-chain JSON (see configs/)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--heal-strength", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--keep-destroyed", action="store_true",
                    help="Also save the destroyed intermediates")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sat_common.validate_model_args(ap, args)

    out_dir = Path(args.out)
    vst_script = Path(__file__).parent / "09_vst_chain.py"

    with tempfile.TemporaryDirectory() as tmp:
        destroyed = Path(tmp) / "destroyed"
        print("=== DESTROY (09_vst_chain.py) ===")
        subprocess.run([sys.executable, str(vst_script), "--input", args.input,
                        "--output", str(destroyed), "--chain", args.chain], check=True)

        print("\n=== HEAL (audio-to-audio) ===")
        model, cfg, device = sat_common.load_model(args.model_config, args.ckpt, args.pretrained)
        sr = cfg["sample_rate"]
        wavs = sorted(destroyed.rglob("*.wav"))
        for wav in wavs:
            init = sat_common.load_audio_file(wav, sr, device)
            n = init.shape[1]
            seconds = min(n / sr, cfg["sample_size"] / sr)
            audio, seed = sat_common.generate(
                model, cfg, args.prompt, seconds, device, steps=args.steps,
                init_audio=init, strength=args.heal_strength)
            rel = wav.relative_to(destroyed)
            sat_common.save_wav(audio[:, :n], out_dir / rel.parent / f"{rel.stem}_healed.wav", sr)
            if args.keep_destroyed:
                import shutil
                dd = out_dir / rel.parent / f"{rel.stem}_destroyed.wav"
                dd.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wav, dd)
            print(f"healed: {rel} (seed {seed})")
    print(f"\nDone -> {out_dir}/")


if __name__ == "__main__":
    main()
