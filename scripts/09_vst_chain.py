#!/usr/bin/env python3
"""
09_vst_chain.py
Batch-process audio through YOUR VST3 plugins from Python - no DAW needed.
Uses Spotify's pedalboard library, which hosts real VST3 (and AU) plugins.
Typical use: run generated samples/beats through your character chain
(RC-20 style saturation, tape, compressor) before packaging.

Chain config (JSON), e.g. configs/vst_chain.example.json:
{
  "chain": [
    {"vst3": "C:/Program Files/Common Files/VST3/MyTape.vst3",
     "params": {"mix": 0.4, "drive": 0.6}},
    {"builtin": "Compressor", "params": {"threshold_db": -18, "ratio": 3}},
    {"builtin": "Limiter", "params": {"threshold_db": -1.0}}
  ]
}
- "vst3": absolute path to the .vst3; "params" set by parameter name
  (run with --list-params <path-to.vst3> to print a plugin's parameter names)
- "preset": optional path to a .vstpreset to load instead of/before params
- "builtin": any pedalboard effect (Compressor, Distortion, Reverb, Chorus,
  Delay, Limiter, LowpassFilter, HighpassFilter, Bitcrush, ...)
- "--edit N" opens the native GUI of chain item N so you can dial it in by ear;
  the settings are then used for the whole batch.

Usage:
    pip install pedalboard
    python 09_vst_chain.py --input processed --output processed_vst --chain configs/vst_chain.example.json
    python 09_vst_chain.py --list-params "C:/.../MyPlugin.vst3"
"""
import argparse
import json
import sys
from pathlib import Path


def build_chain(cfg, edit_index=None):
    import pedalboard
    from pedalboard import Pedalboard, load_plugin

    plugins = []
    for i, item in enumerate(cfg["chain"]):
        if "vst3" in item:
            p = load_plugin(item["vst3"])
            if item.get("preset"):
                p.load_preset(item["preset"])
            for name, val in item.get("params", {}).items():
                setattr(p, name, val)
            if edit_index == i:
                print(f"Opening editor for {item['vst3']} - close the window to continue...")
                p.show_editor()
        elif "builtin" in item:
            cls = getattr(pedalboard, item["builtin"])
            p = cls(**item.get("params", {}))
        else:
            sys.exit(f"Chain item {i}: needs 'vst3' or 'builtin'.")
        plugins.append(p)
    return Pedalboard(plugins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--chain")
    ap.add_argument("--edit", type=int, help="Open GUI editor for chain item N before batch")
    ap.add_argument("--list-params", metavar="VST3_PATH",
                    help="Print a plugin's automatable parameter names and exit")
    args = ap.parse_args()

    if args.list_params:
        from pedalboard import load_plugin
        p = load_plugin(args.list_params)
        print(f"Parameters for {args.list_params}:")
        for name, param in p.parameters.items():
            print(f"  {name:40s} = {param}")
        return

    if not (args.input and args.output and args.chain):
        ap.error("--input, --output and --chain are required (or use --list-params).")

    from pedalboard.io import AudioFile
    from tqdm import tqdm

    cfg = json.loads(Path(args.chain).read_text(encoding="utf-8"))
    board = build_chain(cfg, args.edit)

    in_root, out_root = Path(args.input), Path(args.output)
    files = sorted(p for p in in_root.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".aif", ".aiff"})
    if not files:
        sys.exit("No audio files found.")
    for f in tqdm(files, desc="Processing through chain"):
        rel = f.relative_to(in_root)
        dest = (out_root / rel).with_suffix(".wav")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with AudioFile(str(f)) as src:
            audio = src.read(src.frames)
            sr = src.samplerate
        processed = board(audio, sr)
        with AudioFile(str(dest), "w", samplerate=sr,
                       num_channels=processed.shape[0], bit_depth=24) as dst:
            dst.write(processed)
    print(f"Done. {len(files)} files -> {out_root}/")


if __name__ == "__main__":
    main()
