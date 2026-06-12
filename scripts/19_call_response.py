#!/usr/bin/env python3
"""
19_call_response.py  -  AI session musician (call-and-response)
Watches a folder. Every time a new WAV appears (export a clip from Live:
right-click clip -> Export Audio, or freeze+flatten then drag to the folder),
the model answers with N variations, written to a response folder you've added
to Live's browser. Trade bars with a model trained on your own catalog.

Usage:
    python 19_call_response.py --model-config model_config.json --ckpt hiphop_v1.ckpt \
        --watch "C:/Users/you/Ableton/CallFolder" \
        --respond "C:/Users/you/Ableton/ResponseFolder" \
        --prompt "hip hop, melodic loops, soul keys, dusty response phrase" \
        --strength 0.45 --variations 2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sat_common  # noqa: E402


def stable_size(path, wait=1.0):
    """True when the file has stopped growing (export finished)."""
    s1 = path.stat().st_size
    time.sleep(wait)
    return path.stat().st_size == s1 and s1 > 0


def main():
    ap = argparse.ArgumentParser()
    sat_common.add_model_args(ap)
    ap.add_argument("--watch", required=True)
    ap.add_argument("--respond", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--variations", type=int, default=2)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()
    sat_common.validate_model_args(ap, args)

    watch, respond = Path(args.watch), Path(args.respond)
    watch.mkdir(parents=True, exist_ok=True)
    respond.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model, cfg, device = sat_common.load_model(args.model_config, args.ckpt, args.pretrained)
    sr = cfg["sample_rate"]

    seen = {p.name for p in watch.glob("*.wav")}
    print(f"Watching {watch} (already present: {len(seen)} ignored). Ctrl+C to stop.")
    while True:
        try:
            for wav in sorted(watch.glob("*.wav")):
                if wav.name in seen:
                    continue
                if not stable_size(wav):
                    continue  # still being written; retry next poll
                seen.add(wav.name)
                print(f"\nCALL: {wav.name}")
                init = sat_common.load_audio_file(wav, sr, device)
                n = init.shape[1]
                seconds = min(n / sr, cfg["sample_size"] / sr)
                for v in range(1, args.variations + 1):
                    audio, seed = sat_common.generate(
                        model, cfg, args.prompt, seconds, device, steps=args.steps,
                        init_audio=init, strength=args.strength)
                    name = f"{wav.stem}_response{v:02d}_seed{seed}.wav"
                    sat_common.save_wav(audio[:, :n], respond / name, sr)
                    print(f"  RESPONSE {v}: {name}")
            time.sleep(args.poll)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
