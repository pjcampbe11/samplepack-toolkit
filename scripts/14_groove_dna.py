#!/usr/bin/env python3
"""
14_groove_dna.py  -  Groove transplants ("quantize to Dilla")
Extract the micro-timing + velocity fingerprint of a reference drum break and
save it as a groove template. 08_beat_builder.py --groove applies it to YOUR
samples - groove and sound fully decoupled.

How: detect onsets, fit them to a 16th-note grid derived from beat tracking,
then record the median timing deviation (as a fraction of a 16th step) and the
mean onset strength for each of the 16 step positions.

Usage:
    python 14_groove_dna.py --input classic_break.wav --name dilla_a --out grooves/
    python 08_beat_builder.py --library organized --groove grooves/dilla_a.groove.json ...
Rights note: a groove template stores timing numbers, not audio.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np

STEPS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Reference drum loop/break (WAV/MP3)")
    ap.add_argument("--name", help="Template name (default: input filename)")
    ap.add_argument("--out", default="grooves")
    ap.add_argument("--engine", choices=["auto", "beat_this", "librosa"], default="auto")
    args = ap.parse_args()

    y, sr = librosa.load(args.input, sr=22050, mono=True)
    if y.size == 0:
        raise SystemExit("Empty audio.")
    # Beat tracking: beat_this (CPJKU, 2024 SOTA - pip install beat-this) when
    # available, librosa fallback. beat_this is markedly more accurate on real
    # mixes, which directly improves groove extraction.
    beats = None
    if args.engine in ("auto", "beat_this"):
        try:
            from beat_this.inference import File2Beats
            f2b = File2Beats(checkpoint_path="final0", dbn=False)
            b, _downbeats = f2b(args.input)
            if len(b) >= 4:
                beats = np.asarray(b, dtype=float)
                tempo = 60.0 / float(np.median(np.diff(beats)))
                print("(beat tracker: beat_this)")
        except ImportError:
            if args.engine == "beat_this":
                raise SystemExit("beat_this not installed:  pip install beat-this")
    if beats is None:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        tempo = float(np.atleast_1d(tempo)[0])
        print("(beat tracker: librosa)")
    if len(beats) < 4:
        raise SystemExit("Couldn't track beats - use a cleaner drum loop.")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    raw_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False)
    bt_frames = librosa.onset.onset_backtrack(raw_frames, onset_env)
    onset_times = librosa.frames_to_time(bt_frames, sr=sr)
    # strength read at the detected peak, not the backtracked start
    onset_strengths = onset_env[np.clip(raw_frames, 0, len(onset_env) - 1)]
    if onset_strengths.max() > 0:
        onset_strengths = onset_strengths / onset_strengths.max()

    # 16th grid interpolated between tracked beats
    grid = []
    for i in range(len(beats) - 1):
        for k in range(4):
            grid.append(beats[i] + (beats[i + 1] - beats[i]) * k / 4)
    grid.append(beats[-1])
    grid = np.array(grid)
    step_sec = float(np.median(np.diff(grid)))

    dev_by_step = [[] for _ in range(STEPS)]
    vel_by_step = [[] for _ in range(STEPS)]
    all_devs = []
    pairs = []
    for t, s in zip(onset_times, onset_strengths):
        idx = int(np.argmin(np.abs(grid - t)))
        dev = (t - grid[idx]) / step_sec  # fraction of a 16th
        if abs(dev) > 0.45:  # not actually on this grid point
            continue
        pairs.append((idx % STEPS, dev, float(s)))
        all_devs.append(dev)
    # remove global grid phase: only relative micro-timing is the groove
    phase = float(np.median(all_devs)) if all_devs else 0.0
    for step_idx, dev, s in pairs:
        dev_by_step[step_idx].append(dev - phase)
        vel_by_step[step_idx].append(s)

    offsets = [round(float(np.median(d)), 4) if d else 0.0 for d in dev_by_step]
    velocities = [round(float(np.mean(v)), 4) if v else 0.8 for v in vel_by_step]

    name = args.name or Path(args.input).stem
    template = {
        "name": name,
        "source_bpm": round(tempo, 1),
        "steps": STEPS,
        "offsets": offsets,        # per-step timing deviation, fraction of a 16th
        "velocities": velocities,  # per-step accent profile, 0..1
        "onsets_used": int(sum(len(d) for d in dev_by_step)),
    }
    out = Path(args.out) / f"{name}.groove.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(f"Groove DNA -> {out}")
    print(f"  source tempo {tempo:.1f} BPM, {template['onsets_used']} onsets analyzed")
    print(f"  offsets   : {offsets}")
    print(f"  velocities: {velocities}")


if __name__ == "__main__":
    main()
