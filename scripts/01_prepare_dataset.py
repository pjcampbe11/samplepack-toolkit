#!/usr/bin/env python3
"""
01_prepare_dataset.py
Prepare a hip-hop WAV library for fine-tuning Stable Audio Open.

Input layout (you organize your library into tag folders; folder names become prompt tags):
    raw_library/
      drums_oneshots/kicks/*.wav
      drums_oneshots/snares/*.wav
      drums_loops/boom_bap/*.wav
      melodic_loops/soul_keys/*.wav
      stems/*.wav

Optionally drop a `tags.txt` in any folder: one comma-separated line of extra
descriptors applied to every file in that folder (e.g. "dusty, vinyl, swung, 1990s boom bap").

Output: 44.1 kHz stereo WAVs (long files sliced to <= MAX_SECONDS) + one JSON
sidecar per WAV containing the training prompt and analysis metadata.

Usage:
    python 01_prepare_dataset.py --input raw_library --output dataset --max-seconds 40
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

TARGET_SR = 44100
ONESHOT_MAX_SECONDS = 2.5  # files shorter than this are treated as one-shots

# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(y_mono: np.ndarray, sr: int):
    """Return (key_name, mode) e.g. ('F', 'minor'), or (None, None) if ambiguous."""
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr).mean(axis=1)
    if chroma.sum() <= 0:
        return None, None
    best = (-2.0, None, None)
    for shift in range(12):
        rolled = np.roll(chroma, -shift)
        for profile, mode in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
            r = np.corrcoef(rolled, profile)[0, 1]
            if r > best[0]:
                best = (r, PITCHES[shift], mode)
    return (best[1], best[2]) if best[0] > 0.5 else (None, None)


def estimate_bpm(y_mono: np.ndarray, sr: int, bpm_min=60, bpm_max=180):
    tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    if tempo <= 0:
        return None
    # fold octave errors into the genre's plausible range
    # hip hop 60-180 (default) | dnb/dubstep --bpm-min 100 --bpm-max 200 | metal 80-220
    while tempo > bpm_max:
        tempo /= 2
    while tempo < bpm_min:
        tempo *= 2
    return round(tempo)


def folder_tags(wav_path: Path, input_root: Path):
    """Folder names (underscores -> spaces) plus any tags.txt contents along the path."""
    tags, extra = [], []
    rel = wav_path.relative_to(input_root)
    current = input_root
    for part in rel.parts[:-1]:
        tags.append(part.replace("_", " ").replace("-", " ").strip())
        current = current / part
        tf = current / "tags.txt"
        if tf.exists():
            extra += [t.strip() for t in tf.read_text(encoding="utf-8").split(",") if t.strip()]
    return tags, extra


def build_prompt(tags, extra_tags, kind, bpm, key, mode):
    parts = ["hip hop"] + tags
    if kind == "oneshot":
        parts.append("one shot")
    if bpm and kind != "oneshot":
        parts.append(f"{bpm} BPM")
    if key and kind != "oneshot":
        parts.append(f"key of {key} {mode}")
    parts += extra_tags
    # dedupe, preserve order
    seen, out = set(), []
    for p in parts:
        pl = p.lower()
        if pl and pl not in seen:
            seen.add(pl)
            out.append(p)
    return ", ".join(out)


def process_file(wav_path, input_root, output_root, max_seconds, bpm_min=60, bpm_max=180):
    try:
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=False)
    except Exception as e:
        return [f"SKIP (unreadable): {wav_path} -> {e}"]
    if y.ndim == 1:
        y = np.stack([y, y])  # mono -> stereo
    n = y.shape[1]
    dur = n / TARGET_SR
    if dur < 0.05:
        return [f"SKIP (too short): {wav_path}"]
    peak = float(np.abs(y).max())
    if peak < 1e-4:
        return [f"SKIP (silent): {wav_path}"]

    mono = y.mean(axis=0)
    kind = "oneshot" if dur <= ONESHOT_MAX_SECONDS else "loop"
    bpm = estimate_bpm(mono, TARGET_SR, bpm_min, bpm_max) if kind == "loop" else None
    key, mode = estimate_key(mono, TARGET_SR) if kind == "loop" else (None, None)
    tags, extra = folder_tags(wav_path, input_root)
    prompt = build_prompt(tags, extra, kind, bpm, key, mode)

    rel = wav_path.relative_to(input_root)
    out_dir = output_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slice long files into max_seconds chunks (drop tiny remainders)
    chunk = int(max_seconds * TARGET_SR)
    segments = [(0, n)] if n <= chunk else [
        (s, min(s + chunk, n)) for s in range(0, n, chunk)
        if (min(s + chunk, n) - s) / TARGET_SR >= 4.0 or n <= chunk
    ]
    logs = []
    for i, (s, e) in enumerate(segments):
        stem = rel.stem if len(segments) == 1 else f"{rel.stem}_part{i+1:02d}"
        out_wav = out_dir / f"{stem}.wav"
        sf.write(str(out_wav), y[:, s:e].T, TARGET_SR, subtype="PCM_16")
        meta = {
            "prompt": prompt,
            "kind": kind,
            "bpm": bpm,
            "key": f"{key} {mode}" if key else None,
            "seconds_total": round((e - s) / TARGET_SR, 3),
            "source": str(rel),
        }
        out_wav.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logs.append(f"OK: {out_wav.relative_to(output_root)} | {prompt}")
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-seconds", type=float, default=40.0,
                    help="Max segment length; keep below the model window (47 s for SAO 1.0).")
    ap.add_argument("--bpm-min", type=int, default=60, help="BPM fold range (genre-aware: dnb 100)")
    ap.add_argument("--bpm-max", type=int, default=180, help="BPM fold range (genre-aware: dnb/metal 200-220)")
    args = ap.parse_args()

    input_root, output_root = Path(args.input), Path(args.output)
    if not input_root.is_dir():
        sys.exit(f"Input directory not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    wavs = sorted(p for p in input_root.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".aif", ".aiff"})
    if not wavs:
        sys.exit("No audio files found.")
    log_path = output_root / "prepare_log.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        for wav in tqdm(wavs, desc="Preparing"):
            for line in process_file(wav, input_root, output_root, args.max_seconds, args.bpm_min, args.bpm_max):
                log.write(line + "\n")
    print(f"Done. {len(wavs)} source files processed. Log: {log_path}")


if __name__ == "__main__":
    main()
