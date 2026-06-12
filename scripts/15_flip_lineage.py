#!/usr/bin/env python3
"""
15_flip_lineage.py  -  Flip lineages (telephone-game morphing)
Chain audio-to-audio on its own output: source -> stage 1 -> stage 2 -> ...
with a different prompt (and optional strength) per stage. Every generation is
saved, plus lineage.json documenting the full evolution (prompts, strengths,
seeds, file hashes). The lineage itself is content.

Usage:
    python 15_flip_lineage.py --model-config model_config.json --ckpt hiphop_v1.ckpt \
        --input source_loop.wav --out lineages/loop1 \
        --stage "0.3:hip hop, melodic loops, soul keys, dusty vinyl" \
        --stage "0.3:hip hop, melodic loops, dark strings, tape warble" \
        --stage "0.35:hip hop, melodic loops, eerie synth, lofi texture"
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sat_common  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    sat_common.add_model_args(ap)
    ap.add_argument("--input", required=True)
    ap.add_argument("--stage", action="append", required=True,
                    help='Repeatable: "STRENGTH:PROMPT", e.g. "0.3:hip hop, dusty keys"')
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sat_common.validate_model_args(ap, args)

    stages = []
    for s in args.stage:
        strength, _, prompt = s.partition(":")
        try:
            strength = float(strength)
        except ValueError:
            ap.error(f"Bad --stage '{s}' - format is STRENGTH:PROMPT")
        if not prompt.strip():
            ap.error(f"Bad --stage '{s}' - empty prompt")
        stages.append((strength, prompt.strip()))

    model, cfg, device = sat_common.load_model(args.model_config, args.ckpt, args.pretrained)
    sr = cfg["sample_rate"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    current = sat_common.load_audio_file(args.input, sr, device)
    n_samples = current.shape[1]
    seconds = min(n_samples / sr, cfg["sample_size"] / sr)
    sat_common.save_wav(current, out_dir / "gen00_source.wav", sr)

    lineage = {"source": str(args.input), "source_sha256": sha256(args.input), "stages": []}
    for i, (strength, prompt) in enumerate(stages, 1):
        audio, seed = sat_common.generate(
            model, cfg, prompt, seconds, device, steps=args.steps,
            cfg_scale=args.cfg, init_audio=current, strength=strength)
        audio = audio[:, :n_samples]
        fname = f"gen{i:02d}_s{strength:.2f}.wav"
        sat_common.save_wav(audio, out_dir / fname, sr)
        lineage["stages"].append({
            "stage": i, "file": fname, "prompt": prompt, "strength": strength,
            "seed": seed, "sha256": sha256(out_dir / fname)})
        print(f"stage {i}/{len(stages)}: s={strength} '{prompt[:60]}' -> {fname}")
        current = sat_common.load_audio_file(out_dir / fname, sr, device)

    (out_dir / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    print(f"\nLineage complete: {out_dir}/lineage.json")
    print("Rights note: outputs are derivative of the source - only use cleared material.")


if __name__ == "__main__":
    main()
