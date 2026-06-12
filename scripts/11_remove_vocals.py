#!/usr/bin/env python3
"""
11_remove_vocals.py
One job: strip vocals from a large set of MP3/WAV files.

Engines (June 2026):
  roformer (default) - BS-RoFormer via the audio-separator package. Current
                       SOTA (~12.9 dB vocals SDR vs ~9 for htdemucs).
                       pip install "audio-separator[gpu]"   (or [cpu])
  demucs             - htdemucs fallback, also gives you 4-stem separation.
                       pip install demucs

See README_vocal_removal.md for setup and details.

Usage:
    python 11_remove_vocals.py --input songs/ --output instrumentals/
    python 11_remove_vocals.py --input songs/ --output out/ --engine demucs --keep-vocals
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aif", ".aiff"}
ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def collect(in_path, out_root, ext, overwrite):
    files = ([in_path] if in_path.is_file()
             else sorted(p for p in in_path.rglob("*") if p.suffix.lower() in AUDIO_EXTS))
    if not files:
        sys.exit("No audio files found.")
    todo = [f for f in files
            if overwrite or not (out_root / f"{f.stem}_instrumental{ext}").exists()]
    print(f"{len(files)} file(s) found, {len(todo)} to process.")
    return todo


def run_roformer(args, todo, out_root, ext):
    try:
        from audio_separator.separator import Separator
    except ImportError:
        sys.exit('audio-separator not installed:  pip install "audio-separator[gpu]"')
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        sep = Separator(output_dir=tmp, output_format=ext.lstrip(".").upper())
        sep.load_model(model_filename=args.model)
        for i, f in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {f.name}")
            try:
                outputs = sep.separate(str(f))
            except Exception as e:
                print(f"  FAILED: {e}")
                failed.append(str(f))
                continue
            got = False
            for o in outputs:
                op = Path(tmp) / Path(o).name
                if not op.exists():
                    op = Path(o)
                low = op.name.lower()
                if "instrumental" in low:
                    shutil.move(str(op), str(out_root / f"{f.stem}_instrumental{ext}"))
                    got = True
                elif "vocal" in low and args.keep_vocals:
                    shutil.move(str(op), str(out_root / f"{f.stem}_vocals{ext}"))
            if not got:
                failed.append(str(f))
    return failed


def run_demucs(args, todo, out_root, ext):
    if shutil.which("demucs") is None:
        sys.exit("Demucs not found:  pip install demucs")
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {f.name}")
            cmd = ["demucs", "--two-stems", "vocals", "-n", args.model
                   if args.model != ROFORMER_MODEL else "htdemucs",
                   "-o", tmp, "-j", str(args.jobs)]
            if ext == ".mp3":
                cmd += ["--mp3", "--mp3-bitrate", "320"]
            cmd.append(str(f))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                failed.append(str(f))
                continue
            model_name = args.model if args.model != ROFORMER_MODEL else "htdemucs"
            stem_dir = Path(tmp) / model_name / f.stem
            no_vox = stem_dir / f"no_vocals{ext}"
            vox = stem_dir / f"vocals{ext}"
            if no_vox.exists():
                shutil.move(str(no_vox), str(out_root / f"{f.stem}_instrumental{ext}"))
            else:
                failed.append(str(f))
                continue
            if args.keep_vocals and vox.exists():
                shutil.move(str(vox), str(out_root / f"{f.stem}_vocals{ext}"))
            shutil.rmtree(stem_dir, ignore_errors=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="File or folder (recursive)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--engine", choices=["roformer", "demucs"], default="roformer")
    ap.add_argument("--model", default=ROFORMER_MODEL,
                    help=f"roformer: separator model filename (default {ROFORMER_MODEL}); "
                         "demucs: htdemucs | htdemucs_ft")
    ap.add_argument("--keep-vocals", action="store_true")
    ap.add_argument("--mp3", action="store_true", help="Output MP3 320k instead of WAV")
    ap.add_argument("--jobs", type=int, default=1, help="demucs CPU parallel jobs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_path, out_root = Path(args.input), Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    ext = ".mp3" if args.mp3 else ".wav"
    todo = collect(in_path, out_root, ext, args.overwrite)
    if not todo:
        return
    failed = (run_roformer if args.engine == "roformer" else run_demucs)(args, todo, out_root, ext)
    print(f"\nDone. Instrumentals in {out_root}/")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for f in failed:
            print("  " + f)
        sys.exit(1)


if __name__ == "__main__":
    main()
