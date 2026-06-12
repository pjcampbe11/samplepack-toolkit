#!/usr/bin/env python3
"""
08_beat_builder.py
Build hip-hop beats from YOUR samples (the organized library from
07_organize_soundbank.py). For each beat it:
  - randomly picks a kick/snare/hat/perc (and optional 808) from your folders
  - sequences them on a 16-step grid using style pattern templates with
    probability variation, swing, and velocity humanization
  - renders per-instrument WAV stems + a summed master (44.1 kHz / 24-bit)
  - writes a standard MIDI file (.mid) of the pattern so you can drop it on an
    Ableton Drum Rack (GM mapping: kick=36 snare=38 closed hat=42 perc=47 808=35)
  - writes a manifest.json recording exactly which of your samples were used

Layer melodic loops from 03_generate.py on top in your DAW, or point --melodic
at a folder of melodic loops to mix one in automatically.

Usage:
    python 08_beat_builder.py --library organized --style boom_bap --bpm 92 \
        --bars 4 --count 8 --out beats/
Styles: boom_bap, trap, drill, lofi

Creative options:
  --rotate            every hit picks a different sample from the folder pool
                      (pair with 13_microvariants.py: no two hits the same)
  --groove FILE.json  apply a groove template from 14_groove_dna.py - replaces
                      the style's swing with the reference break's micro-timing
                      and accent profile (groove transplant)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
STEPS_PER_BAR = 16  # 16th-note grid

# Pattern templates: per instrument, 16 steps, value = hit probability (0..1).
STYLES = {
    "boom_bap": {
        "swing": 0.58, "hat_choke": 0.9,
        "kick":  [1, 0, 0, 0,  0, 0, 0, .3,  0, 0, 1, 0,  0, .2, 0, 0],
        "snare": [0, 0, 0, 0,  1, 0, 0, 0,   0, 0, 0, 0,  1, 0, 0, .15],
        "hat":   [.9, 0, .9, 0, .9, 0, .9, .3, .9, 0, .9, 0, .9, 0, .9, .3],
        "perc":  [0, 0, .15, 0, 0, 0, 0, .2,  0, .15, 0, 0, 0, 0, .2, 0],
        "e808":  [0]*16,
    },
    "trap": {
        "swing": 0.5, "hat_choke": 1.0,
        "kick":  [1, 0, 0, 0,  0, 0, .4, 0,  0, .5, 0, 0,  0, 0, 0, 0],
        "snare": [0, 0, 0, 0,  0, 0, 0, 0,   1, 0, 0, 0,   0, 0, 0, 0],
        "hat":   [1, .3, 1, .3, 1, .3, 1, .6, 1, .3, 1, .9, 1, .3, 1, .6],
        "perc":  [0, 0, 0, .1,  0, 0, 0, 0,   0, 0, .15, 0, 0, 0, 0, .1],
        "e808":  [1, 0, 0, 0,  0, 0, .4, 0,  0, .5, 0, 0,  0, 0, 0, 0],
    },
    "drill": {
        "swing": 0.54, "hat_choke": 1.0,
        "kick":  [1, 0, 0, 0,  0, 0, 0, .6,  0, 0, .6, 0,  0, 0, 0, 0],
        "snare": [0, 0, 0, 0,  0, 0, 0, 0,   1, 0, 0, .3,  0, 0, 0, 0],
        "hat":   [1, 0, .7, .4, 1, 0, .7, 0, 1, .4, .7, 0, 1, 0, .7, .4],
        "perc":  [0, .1, 0, 0,  0, 0, .2, 0,  0, 0, 0, .15, 0, .1, 0, 0],
        "e808":  [1, 0, 0, .3,  0, 0, 0, .5,  0, 0, .6, 0,  0, .3, 0, 0],
    },
    "lofi": {
        "swing": 0.62, "hat_choke": 0.85,
        "kick":  [1, 0, 0, .2,  0, 0, .5, 0,  0, .3, 1, 0,  0, 0, 0, 0],
        "snare": [0, 0, 0, 0,  1, 0, 0, 0,   0, 0, 0, 0,   1, 0, 0, 0],
        "hat":   [.8, 0, .8, 0, .8, 0, .8, 0, .8, 0, .8, 0, .8, 0, .8, .4],
        "perc":  [0, 0, 0, .25, 0, 0, 0, 0,  0, .2, 0, 0,  0, 0, .25, 0],
        "e808":  [0]*16,
    },
    # --- rock / metal (typical BPM: rock 120, metal 160+) ---
    "rock": {  # straight 8ths backbeat
        "swing": 0.5, "hat_choke": 0.9,
        "kick":  [1, 0, 0, 0,  0, 0, .6, 0,  1, 0, .3, 0,  0, 0, 0, 0],
        "snare": [0, 0, 0, 0,  1, 0, 0, 0,   0, 0, 0, 0,   1, 0, 0, .2],
        "hat":   [1, 0, .8, 0, 1, 0, .8, 0,  1, 0, .8, 0,  1, 0, .8, 0],
        "perc":  [0, 0, 0, 0,  0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, .15],
        "e808":  [0]*16,
    },
    "metal": {  # double-kick 16ths under a halftime backbeat
        "swing": 0.5, "hat_choke": 1.0,
        "kick":  [1, .9, .9, .9, .9, .9, .9, .9, 1, .9, .9, .9, .9, .9, .9, .9],
        "snare": [0, 0, 0, 0,  0, 0, 0, 0,   1, 0, 0, 0,   0, 0, 0, .25],
        "hat":   [1, 0, 1, 0,  1, 0, 1, 0,   1, 0, 1, 0,   1, 0, 1, 0],
        "perc":  [0]*16,
        "e808":  [0]*16,
    },
    "dbeat": {  # d-beat / punk drive
        "swing": 0.5, "hat_choke": 0.9,
        "kick":  [1, 0, 0, .8,  0, 0, 1, 0,  0, .8, 0, 0,  1, 0, 0, 0],
        "snare": [0, 0, 1, 0,  0, 1, 0, 0,   1, 0, 0, 1,   0, 0, 1, 0],
        "hat":   [1, 0, 1, 0,  1, 0, 1, 0,   1, 0, 1, 0,   1, 0, 1, 0],
        "perc":  [0]*16,
        "e808":  [0]*16,
    },
    # --- dubstep / drum & bass (dubstep 140 halftime, dnb 172-176) ---
    "dubstep": {  # halftime: snare on 3, space for wobble
        "swing": 0.5, "hat_choke": 0.9,
        "kick":  [1, 0, 0, 0,  0, 0, 0, .3,  0, 0, .4, 0,  0, 0, 0, 0],
        "snare": [0, 0, 0, 0,  0, 0, 0, 0,   1, 0, 0, 0,   0, 0, 0, .2],
        "hat":   [.7, 0, .4, .6, .7, 0, .4, 0, .7, 0, .4, .6, .7, .3, .4, 0],
        "perc":  [0, .15, 0, 0, 0, 0, .2, 0,  0, 0, 0, .15, 0, 0, 0, .2],
        "e808":  [1, 0, 0, 0,  0, 0, 0, 0,   0, 0, .5, 0,  0, 0, 0, 0],
    },
    "dnb": {  # two-step
        "swing": 0.5, "hat_choke": 1.0,
        "kick":  [1, 0, 0, 0,  0, 0, 0, 0,   0, 0, 1, 0,   0, .3, 0, 0],
        "snare": [0, 0, 0, 0,  1, 0, 0, .2,  0, 0, 0, 0,   1, 0, 0, .3],
        "hat":   [1, .4, .8, .4, 1, .4, .8, .5, 1, .4, .8, .4, 1, .4, .8, .5],
        "perc":  [0, 0, .2, 0,  0, 0, 0, .25, 0, .2, 0, 0,  0, 0, .25, 0],
        "e808":  [1, 0, 0, 0,  0, 0, .4, 0,  0, 0, 1, 0,   0, 0, 0, 0],
    },
    "amen": {  # chopped-break feel: ghost-heavy snare placement
        "swing": 0.5, "hat_choke": 1.0,
        "kick":  [1, 0, .9, 0,  0, 0, 0, .4,  0, .5, 1, 0,  0, 0, 0, 0],
        "snare": [0, 0, 0, 0,  1, 0, 0, .5,  0, .4, 0, 0,  1, 0, .5, .3],
        "hat":   [.9, .3, .9, .3, .9, .3, .9, .3, .9, .3, .9, .3, .9, .3, .9, .3],
        "perc":  [0, 0, 0, .2,  0, 0, .15, 0, 0, 0, 0, .2,  0, .15, 0, 0],
        "e808":  [0]*16,
    },
}

# instrument -> (library subfolder candidates, GM MIDI note)
INSTRUMENTS = {
    "kick":  (["drums_oneshots/kicks", "Kicks", "kicks"], 36),
    "snare": (["drums_oneshots/snares", "Snares", "snares"], 38),
    "hat":   (["drums_oneshots/hats", "Hats", "hats"], 42),
    "perc":  (["drums_oneshots/percs", "Percs", "percs"], 47),
    "e808":  (["drums_oneshots/808s", "808s"], 35),
}


def find_samples(library: Path, candidates):
    for c in candidates:
        d = library / c
        if d.is_dir():
            wavs = sorted(d.rglob("*.wav")) + sorted(d.rglob("*.flac"))
            if wavs:
                return wavs
    return []


def load_sample(path: Path):
    y, sr = sf.read(str(path), always_2d=True)
    y = y.T.astype(np.float64)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y


def render(events, length_samples):
    """events: list of (sample_pos, gain, sample_array). Returns stereo stem buffer."""
    buf = np.zeros((2, length_samples))
    for pos, gain, sample in events:
        seg = sample[:, : max(length_samples - pos, 0)]
        if seg.shape[1] > 0:
            buf[:, pos: pos + seg.shape[1]] += seg * gain
    return buf


def write_midi(path, pattern_events, bpm, bars):
    import mido
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    ticks_per_step = 480 // 4
    msgs = []
    for inst, steps in pattern_events.items():
        note = INSTRUMENTS[inst][1]
        for step, vel in steps:
            t_on = step * ticks_per_step
            msgs.append((t_on, mido.Message("note_on", note=note, velocity=vel, channel=9)))
            msgs.append((t_on + ticks_per_step // 2,
                         mido.Message("note_off", note=note, velocity=0, channel=9)))
    msgs.sort(key=lambda m: m[0])
    last = 0
    for t, msg in msgs:
        msg.time = t - last
        track.append(msg)
        last = t
    mid.save(str(path))


def build_beat(library, style_name, bpm, bars, out_dir, rng, melodic_dir=None,
               rotate=False, groove=None):
    style = STYLES[style_name]
    step_sec = 60.0 / bpm / 4.0
    total_steps = bars * STEPS_PER_BAR
    length = int(total_steps * step_sec * SR) + SR  # +1 s tail
    g_off = groove["offsets"] if groove else None
    g_vel = groove["velocities"] if groove else None

    chosen, stems, midi_events = {}, {}, {}
    cache = {}
    for inst, (folders, _note) in INSTRUMENTS.items():
        probs = style[inst]
        if not any(probs):
            continue
        pool = find_samples(library, folders)
        if not pool:
            continue
        if rotate:
            pool_used = pool if len(pool) <= 16 else rng.sample(pool, 16)
            chosen[inst] = [str(p) for p in pool_used]
        else:
            pool_used = [rng.choice(pool)]
            chosen[inst] = str(pool_used[0])
        events, mevents = [], []
        for step in range(total_steps):
            p = probs[step % STEPS_PER_BAR]
            if rng.random() < p:
                t = step * step_sec
                if g_off is not None:
                    t += g_off[step % STEPS_PER_BAR] * step_sec
                elif step % 2 == 1:  # style swing on offbeat 16ths
                    t += (style["swing"] - 0.5) * 2 * step_sec
                gain = rng.uniform(0.75, 1.0) if p < 1 else rng.uniform(0.9, 1.0)
                if g_vel is not None:
                    gain *= 0.6 + 0.4 * g_vel[step % STEPS_PER_BAR]
                sp = rng.choice(pool_used)
                if sp not in cache:
                    cache[sp] = load_sample(sp)
                events.append((max(int(t * SR), 0), gain, cache[sp]))
                mevents.append((step, max(1, min(int(gain * 127), 127))))
        if events:
            stems[inst] = render(events, length)
            midi_events[inst] = mevents

    if "kick" not in stems or "snare" not in stems:
        sys.exit("Library must contain at least kicks and snares (run 07_organize_soundbank.py first).")

    # mix levels
    levels = {"kick": 0.95, "snare": 0.9, "hat": 0.5, "perc": 0.45, "e808": 0.85, "melodic": 0.55}
    master = np.zeros((2, length))
    for inst, stem in stems.items():
        master += stem * levels[inst]

    if melodic_dir:
        loops = sorted(Path(melodic_dir).rglob("*.wav"))
        if loops:
            loop_path = rng.choice(loops)
            loop = load_sample(loop_path)
            reps = int(np.ceil(length / loop.shape[1]))
            tiled = np.tile(loop, (1, reps))[:, :length]
            master += tiled * levels["melodic"]
            stems["melodic"] = tiled
            chosen["melodic"] = str(loop_path)

    peak = np.abs(master).max()
    if peak > 0:
        master *= (10 ** (-1.0 / 20)) / peak  # -1 dBFS headroom

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "master.wav"), master.T, SR, subtype="PCM_24")
    for inst, stem in stems.items():
        sp = np.abs(stem).max()
        sf.write(str(out_dir / f"stem_{inst}.wav"), (stem / sp if sp > 0 else stem).T, SR, subtype="PCM_24")
    write_midi(out_dir / "pattern.mid", midi_events, bpm, bars)
    (out_dir / "manifest.json").write_text(json.dumps(
        {"style": style_name, "bpm": bpm, "bars": bars, "rotate": rotate,
         "groove": groove["name"] if groove else None, "samples": chosen},
        indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True, help="Organized sample library root")
    ap.add_argument("--style", choices=sorted(STYLES), default="boom_bap")
    ap.add_argument("--bpm", type=int, default=92,
                    help="Typical: boom_bap/lofi 85-95, trap 130-150, rock 120, "
                         "metal 160-200, dbeat 180, dubstep 140 (halftime feel), dnb/amen 172-176")
    ap.add_argument("--bars", type=int, default=4)
    ap.add_argument("--count", type=int, default=4, help="How many beats to build")
    ap.add_argument("--melodic", help="Optional folder of melodic loops to layer in")
    ap.add_argument("--rotate", action="store_true",
                    help="Different sample per hit (use with 13_microvariants.py output)")
    ap.add_argument("--groove", help="Groove template from 14_groove_dna.py")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--out", default="beats")
    args = ap.parse_args()

    groove = None
    if args.groove:
        groove = json.loads(Path(args.groove).read_text(encoding="utf-8"))
        assert len(groove["offsets"]) == STEPS_PER_BAR, "groove template must have 16 steps"

    library = Path(args.library)
    rng = random.Random(args.seed if args.seed >= 0 else None)
    for i in range(args.count):
        tag = f"_{groove['name']}" if groove else ""
        out_dir = Path(args.out) / f"{args.style}_{args.bpm}bpm{tag}_{i+1:02d}"
        build_beat(library, args.style, args.bpm, args.bars, out_dir, rng,
                   args.melodic, rotate=args.rotate, groove=groove)
        print(f"Built {out_dir}")
    print("\nEach beat folder: master.wav, per-instrument stems, pattern.mid, manifest.json")
    print("Ableton: drop pattern.mid on a Drum Rack holding the samples from manifest.json,")
    print("then your VST chain on that track shapes the sound (see 09/10 + README).")


if __name__ == "__main__":
    main()
