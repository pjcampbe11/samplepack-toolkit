#!/usr/bin/env python3
"""
12_curation_loop.py  -  Taste distillation (closed-loop curation)
Score generated candidates against a reference folder of YOUR best sounds using
CLAP audio embeddings, keep the closest matches, and stage them as the training
set for the next fine-tune round. Each cycle converges the model on your ear.

    pip install laion-clap

score:   rank candidates by similarity to your reference sounds
    python 12_curation_loop.py score --candidates generated/ --reference my_best/ \
        --keep-top 0.1 --keep-dir round2_keepers/
promote: stage keepers (plus prompt sidecars) as a training dataset for the next round
    python 12_curation_loop.py promote --keep-dir round2_keepers/ --dataset-dir dataset_round2/ \
        --base-prompt "hip hop, drums oneshots"
Then: 02_validate_dataset.py on dataset_round2 and fine-tune FROM YOUR LAST
checkpoint (train.py --pretrained_ckpt_path your_last_unwrapped.ckpt).
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".wav", ".flac", ".mp3"}


def embed(paths, batch=16):
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # downloads default 630k checkpoint on first run
    embs = []
    files = [str(p) for p in paths]
    for i in range(0, len(files), batch):
        e = model.get_audio_embedding_from_filelist(x=files[i:i + batch], use_tensor=False)
        embs.append(e)
    e = np.concatenate(embs, axis=0)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def cmd_score(args):
    cands = sorted(p for p in Path(args.candidates).rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    refs = sorted(p for p in Path(args.reference).rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not cands or not refs:
        raise SystemExit("Need audio in both --candidates and --reference.")
    print(f"Embedding {len(refs)} reference + {len(cands)} candidate files (CLAP)...")
    ref_emb, cand_emb = embed(refs), embed(cands)
    centroid = ref_emb.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    scores = cand_emb @ centroid
    order = np.argsort(-scores)

    out_csv = Path(args.candidates) / "curation_scores.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "file", "similarity"])
        for rank, idx in enumerate(order, 1):
            w.writerow([rank, str(cands[idx]), f"{scores[idx]:.4f}"])
    print(f"Scores: {out_csv}")

    if args.keep_dir:
        n_keep = max(1, int(len(cands) * args.keep_top)) if args.keep_top < 1 else int(args.keep_top)
        keep_dir = Path(args.keep_dir)
        for idx in order[:n_keep]:
            src = cands[idx]
            rel = src.relative_to(args.candidates)
            dest = keep_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            sidecar = src.with_suffix(".json")
            if sidecar.exists():
                shutil.copy2(sidecar, dest.with_suffix(".json"))
        print(f"Kept top {n_keep} -> {keep_dir}/  (listen before promoting - CLAP ranks, your ear decides)")


def cmd_promote(args):
    keep, dataset = Path(args.keep_dir), Path(args.dataset_dir)
    wavs = sorted(p for p in keep.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not wavs:
        raise SystemExit("No audio in --keep-dir.")
    for wav in wavs:
        rel = wav.relative_to(keep)
        dest = dataset / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, dest)
        sidecar = wav.with_suffix(".json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            prompt = meta.get("prompt") or ""
            if not prompt:
                bits = [args.base_prompt] + [p.replace("_", " ") for p in rel.parts[:-1]]
                if meta.get("bpm"):
                    bits.append(f"{meta['bpm']} BPM")
                if meta.get("key"):
                    bits.append(f"key of {meta['key']}")
                meta["prompt"] = ", ".join(b for b in bits if b)
        else:
            bits = [args.base_prompt] + [p.replace("_", " ") for p in rel.parts[:-1]]
            meta = {"prompt": ", ".join(b for b in bits if b)}
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Staged {len(wavs)} keepers in {dataset}/ with prompt sidecars.")
    print("Next: 02_validate_dataset.py, then fine-tune FROM your last checkpoint.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--candidates", required=True)
    s.add_argument("--reference", required=True)
    s.add_argument("--keep-top", type=float, default=0.1,
                   help="<1 = fraction, >=1 = absolute count (default 0.1)")
    s.add_argument("--keep-dir")
    s.set_defaults(fn=cmd_score)
    p = sub.add_parser("promote")
    p.add_argument("--keep-dir", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--base-prompt", default="hip hop")
    p.set_defaults(fn=cmd_promote)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
