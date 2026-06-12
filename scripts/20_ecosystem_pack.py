#!/usr/bin/env python3
"""
20_ecosystem_pack.py  -  Ecosystem packs (everything inter-compatible)
Lock an entire pack series to one key + BPM so every loop in every volume
combines with every other. Two subcommands:

plan:   rewrite a base pack plan so all prompts carry the same key/BPM
    python 20_ecosystem_pack.py plan --base prompts/pack_plan.example.json \
        --key "F minor" --bpm 90 --name "Crate Ecosystem Vol 2" \
        --out prompts/eco_fmin_90_v2.json

verify: after 04_postprocess.py, check detected key/BPM in the sidecars against
        the lock and quarantine mismatches before packaging
    python 20_ecosystem_pack.py verify --dir processed --key "F minor" --bpm 90 \
        --bpm-tolerance 3
"""
import argparse
import json
import re
import shutil
from pathlib import Path

BPM_RE = re.compile(r"\b\d{2,3}\s*BPM\b", re.IGNORECASE)
KEY_RE = re.compile(r"\bkey of [A-G][#b]? ?(major|minor|maj|min)\b", re.IGNORECASE)
ONESHOT_HINTS = ("kick", "snare", "hat", "perc", "oneshot", "one_shot", "808")


def short_key(key):  # "F minor" -> "Fmin"
    note, _, mode = key.partition(" ")
    return note + ("min" if mode.lower().startswith("min") else "maj")


def cmd_plan(args):
    plan = json.loads(Path(args.base).read_text(encoding="utf-8"))
    plan["pack_name"] = args.name or plan.get("pack_name", "Ecosystem Pack")
    plan["bpm"] = args.bpm
    plan["key"] = args.key
    for cat in plan["categories"]:
        prompt = KEY_RE.sub("", BPM_RE.sub("", cat["prompt"]))
        prompt = re.sub(r"(,\s*)+", ", ", prompt).strip(" ,")
        is_oneshot = any(h in cat["name"].lower() for h in ONESHOT_HINTS)
        if not is_oneshot:
            prompt += f", {args.bpm} BPM"
            if not cat["name"].lower().startswith("drum"):
                prompt += f", key of {args.key}"
        cat["prompt"] = prompt
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Locked plan ({args.key} / {args.bpm} BPM) -> {out}")
    for c in plan["categories"]:
        print(f"  {c['name']:14s} {c['prompt']}")


def cmd_verify(args):
    root = Path(args.dir)
    want_key = short_key(args.key)
    quarantine = root / "_keybpm_mismatch"
    checked = moved = nometa = 0
    for sidecar in sorted(root.rglob("*.json")):
        if sidecar.parent == quarantine:
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("kind") == "oneshot":
            continue
        wav = sidecar.with_suffix(".wav")
        if not wav.exists():
            continue
        checked += 1
        problems = []
        bpm = meta.get("bpm")
        if bpm is None:
            nometa += 1
        else:
            # accept half/double time detections
            if not any(abs(bpm * m - args.bpm) <= args.bpm_tolerance for m in (0.5, 1, 2)):
                problems.append(f"bpm {bpm} != {args.bpm}")
        key = (meta.get("key") or "").replace(" ", "")
        if key and key.lower() != want_key.lower():
            problems.append(f"key {key} != {want_key}")
        if problems:
            quarantine.mkdir(parents=True, exist_ok=True)
            if not args.report_only:
                shutil.move(str(wav), str(quarantine / wav.name))
                shutil.move(str(sidecar), str(quarantine / sidecar.name))
                moved += 1
            print(f"MISMATCH {wav.relative_to(root)}: {'; '.join(problems)}")
    print(f"\nChecked {checked} loops/stems. "
          f"{'Would quarantine' if args.report_only else 'Quarantined'}: {moved if not args.report_only else 'see above'}. "
          f"No BPM metadata: {nometa}.")
    print("Verify quarantined files by ear - detection has errors both ways.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--base", required=True)
    p.add_argument("--key", required=True, help='e.g. "F minor"')
    p.add_argument("--bpm", type=int, required=True)
    p.add_argument("--name")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_plan)
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    v.add_argument("--key", required=True)
    v.add_argument("--bpm", type=int, required=True)
    v.add_argument("--bpm-tolerance", type=int, default=3)
    v.add_argument("--report-only", action="store_true")
    v.set_defaults(fn=cmd_verify)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
