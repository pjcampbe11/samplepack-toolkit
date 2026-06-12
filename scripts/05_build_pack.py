#!/usr/bin/env python3
"""
05_build_pack.py
Assemble QA'd samples into a distributable sample pack:
  - standard folder structure (One Shots / Loops / Stems)
  - producer-style file naming: PackName_Kick_01.wav, PackName_DrumLoop_03_92BPM.wav,
    PackName_MelodicLoop_05_Fmin_90BPM.wav
  - pack README + license txt
  - zip archive

Run AFTER you've manually deleted the duds from the processed folder.

Usage:
    python 05_build_pack.py --input processed --pack-name "Dusty Crates Vol 1" \
        --out packs --license-file my_license.txt
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

ONESHOT_KEYWORDS = ("kick", "snare", "hat", "perc", "oneshot", "one_shot", "808")

DEFAULT_LICENSE = """SAMPLE PACK LICENSE

All sounds in this pack are licensed royalty-free for use in your own musical
compositions, beats, and productions, commercial or non-commercial.

You may NOT redistribute, resell, or repackage these sounds as a sample pack,
sound library, or in any other isolated form.

All sounds were produced with a generative model trained exclusively on audio
owned by or licensed to the publisher.
"""


def slug(s):
    return "".join(c for c in s.title().replace(" ", "") if c.isalnum())


def singular(category):
    c = category.rstrip("s")
    return slug(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pack-name", required=True)
    ap.add_argument("--out", default="packs")
    ap.add_argument("--license-file")
    args = ap.parse_args()

    in_root = Path(args.input)
    pack_slug = slug(args.pack_name)
    pack_dir = Path(args.out) / pack_slug
    if pack_dir.exists():
        shutil.rmtree(pack_dir)

    manifest = []
    for cat_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        category = cat_dir.name
        is_oneshot = any(k in category.lower() for k in ONESHOT_KEYWORDS)
        section = "One Shots" if is_oneshot else ("Stems" if "stem" in category.lower() else "Loops")
        dest = pack_dir / section / category
        dest.mkdir(parents=True, exist_ok=True)

        wavs = sorted(cat_dir.glob("*.wav"))
        for i, wav in enumerate(wavs, 1):
            meta = {}
            sidecar = wav.with_suffix(".json")
            if sidecar.exists():
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            parts = [pack_slug, singular(category), f"{i:02d}"]
            if meta.get("key"):
                parts.append(meta["key"].replace(" ", ""))
            if meta.get("bpm"):
                parts.append(f"{meta['bpm']}BPM")
            new_name = "_".join(parts) + ".wav"
            shutil.copy2(wav, dest / new_name)
            manifest.append(f"{section}/{category}/{new_name}")

    if not manifest:
        raise SystemExit("No WAVs found - did you run 04_postprocess.py and QA?")

    license_text = (Path(args.license_file).read_text(encoding="utf-8")
                    if args.license_file else DEFAULT_LICENSE)
    (pack_dir / "LICENSE.txt").write_text(license_text, encoding="utf-8")
    readme = [f"{args.pack_name}", "=" * len(args.pack_name), "",
              f"{len(manifest)} samples - 24-bit / 44.1 kHz WAV", "", "Contents:"]
    readme += [f"  {m}" for m in manifest]
    (pack_dir / "README.txt").write_text("\n".join(readme), encoding="utf-8")

    zip_path = pack_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in pack_dir.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(pack_dir.parent))
    print(f"Pack built: {pack_dir}  ({len(manifest)} samples)")
    print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()
