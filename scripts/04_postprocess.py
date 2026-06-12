#!/usr/bin/env python3
"""
04_postprocess.py
Clean up raw generations into release-quality samples:
  - reject silent / near-silent duds
  - trim leading & trailing silence (one-shots: tight head, natural tail)
  - micro fade-in/out to kill clicks
  - loudness-normalize: one-shots peak-normalized to -0.3 dBFS,
    loops/stems normalized to a target LUFS (default -14) with -0.3 dBFS ceiling
  - re-analyze BPM/key on loops and write a sidecar JSON used by 05_build_pack.py

Usage:
    python 04_postprocess.py --input generated --output processed --lufs -14
Category folders containing 'kick', 'snare', 'hat', 'perc' or 'oneshot' in their
name are treated as one-shots; everything else as loops/stems.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

SR = 44100
ONESHOT_KEYWORDS = ("kick", "snare", "hat", "perc", "oneshot", "one_shot", "808")

# Reuse analysis from the prep script
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(mono):
    chroma = librosa.feature.chroma_cqt(y=mono, sr=SR).mean(axis=1)
    if chroma.sum() <= 0:
        return None
    best = (-2.0, None)
    for shift in range(12):
        rolled = np.roll(chroma, -shift)
        for profile, mode in ((MAJOR, "maj"), (MINOR, "min")):
            r = np.corrcoef(rolled, profile)[0, 1]
            if r > best[0]:
                best = (r, f"{PITCHES[shift]}{mode}")
    return best[1] if best[0] > 0.5 else None


def estimate_bpm(mono, bpm_min=60, bpm_max=180):
    tempo, _ = librosa.beat.beat_track(y=mono, sr=SR)
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo <= 0:
        return None
    while tempo > bpm_max:
        tempo /= 2
    while tempo < bpm_min:
        tempo *= 2
    return round(tempo)


def trim_silence(y, is_oneshot):
    """y: (channels, samples). Trim using max-abs envelope."""
    env = np.abs(y).max(axis=0)
    thresh = max(env.max() * 0.001, 1e-5)  # -60 dB relative
    idx = np.where(env > thresh)[0]
    if idx.size == 0:
        return None
    start = max(idx[0] - int(0.005 * SR), 0)          # keep 5 ms pre-roll
    tail = int((0.05 if is_oneshot else 0.2) * SR)    # natural decay room
    end = min(idx[-1] + tail, y.shape[1])
    return y[:, start:end]


def apply_fades(y, fade_ms=3.0):
    n = y.shape[1]
    f = min(int(fade_ms / 1000 * SR), n // 4)
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f)
        y[:, :f] *= ramp
        y[:, -f:] *= ramp[::-1]
    return y


def normalize(y, is_oneshot, target_lufs, meter):
    ceiling = 10 ** (-0.3 / 20)
    if is_oneshot:
        peak = np.abs(y).max()
        return y * (ceiling / peak) if peak > 0 else y
    loudness = meter.integrated_loudness(y.T)
    if not np.isfinite(loudness):
        peak = np.abs(y).max()
        return y * (ceiling / peak) if peak > 0 else y
    gain = 10 ** ((target_lufs - loudness) / 20)
    y = y * gain
    peak = np.abs(y).max()
    if peak > ceiling:
        y *= ceiling / peak
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lufs", type=float, default=-14.0)
    ap.add_argument("--bpm-min", type=int, default=60)
    ap.add_argument("--bpm-max", type=int, default=180)
    args = ap.parse_args()

    in_root, out_root = Path(args.input), Path(args.output)
    meter = pyln.Meter(SR)
    wavs = sorted(in_root.rglob("*.wav"))
    rejected = 0

    for wav in tqdm(wavs, desc="Post-processing"):
        category = wav.parent.name
        is_oneshot = any(k in category.lower() for k in ONESHOT_KEYWORDS)
        y, sr = sf.read(str(wav), always_2d=True)
        y = y.T  # (channels, samples)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        if np.abs(y).max() < 1e-3:
            rejected += 1
            continue
        y = trim_silence(y, is_oneshot)
        if y is None or y.shape[1] < int(0.05 * SR):
            rejected += 1
            continue
        y = apply_fades(y.astype(np.float64))
        y = normalize(y, is_oneshot, args.lufs, meter)

        meta = {"category": category, "kind": "oneshot" if is_oneshot else "loop"}
        if not is_oneshot:
            mono = y.mean(axis=0).astype(np.float32)
            meta["bpm"] = estimate_bpm(mono, args.bpm_min, args.bpm_max)
            meta["key"] = estimate_key(mono)

        out_dir = out_root / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_wav = out_dir / wav.name
        sf.write(str(out_wav), y.T, SR, subtype="PCM_24")
        out_wav.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    kept = len(wavs) - rejected
    print(f"\nDone. Kept {kept}, auto-rejected {rejected}. Output: {out_root}/")
    print("NOW LISTEN TO EVERYTHING. Human QA is the step that makes a pack sellable.")


if __name__ == "__main__":
    main()
