#!/usr/bin/env python3
"""
02_validate_dataset.py
Pre-flight checks before paying for GPU time. Verifies every WAV in the prepared
dataset is readable, 44.1 kHz stereo, non-clipping, within duration bounds, and
has a JSON sidecar with a non-empty prompt. Prints a summary report.

Usage:
    python 02_validate_dataset.py --dataset dataset
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--min-seconds", type=float, default=0.05)
    ap.add_argument("--max-seconds", type=float, default=47.0)
    args = ap.parse_args()

    root = Path(args.dataset)
    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        sys.exit("No WAVs found.")

    errors, warnings = [], []
    kinds, durations = Counter(), []
    with_bpm = with_key = 0

    for wav in wavs:
        try:
            info = sf.info(str(wav))
        except Exception as e:
            errors.append(f"{wav}: unreadable ({e})")
            continue
        if info.samplerate != TARGET_SR:
            errors.append(f"{wav}: sample rate {info.samplerate}, expected {TARGET_SR}")
        if info.channels != 2:
            errors.append(f"{wav}: {info.channels} channels, expected 2")
        dur = info.frames / info.samplerate
        durations.append(dur)
        if dur < args.min_seconds:
            errors.append(f"{wav}: too short ({dur:.2f}s)")
        if dur > args.max_seconds:
            errors.append(f"{wav}: too long ({dur:.2f}s) - exceeds model window")

        y, _ = sf.read(str(wav))
        peak = float(np.abs(y).max()) if y.size else 0.0
        if peak >= 0.999:
            warnings.append(f"{wav}: possible clipping (peak {peak:.3f})")
        if peak < 1e-4:
            errors.append(f"{wav}: silent")

        sidecar = wav.with_suffix(".json")
        if not sidecar.exists():
            errors.append(f"{wav}: missing JSON sidecar")
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"{sidecar}: invalid JSON ({e})")
            continue
        if not meta.get("prompt", "").strip():
            errors.append(f"{sidecar}: empty prompt")
        kinds[meta.get("kind", "unknown")] += 1
        with_bpm += 1 if meta.get("bpm") else 0
        with_key += 1 if meta.get("key") else 0

    total_h = sum(durations) / 3600
    print(f"\n=== Dataset report: {root} ===")
    print(f"Files: {len(wavs)}   Total audio: {total_h:.2f} h")
    print(f"Kinds: {dict(kinds)}")
    print(f"With BPM: {with_bpm}   With key: {with_key}")
    if durations:
        print(f"Duration: min {min(durations):.2f}s / median {sorted(durations)[len(durations)//2]:.2f}s / max {max(durations):.2f}s")
    print(f"\nWarnings: {len(warnings)}")
    for w in warnings[:20]:
        print("  " + w)
    print(f"\nErrors: {len(errors)}")
    for e in errors[:50]:
        print("  " + e)
    if errors:
        sys.exit("\nFIX ERRORS BEFORE TRAINING.")
    print("\nDataset is ready for training.")


if __name__ == "__main__":
    main()
