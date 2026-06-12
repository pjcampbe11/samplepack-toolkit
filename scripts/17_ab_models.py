#!/usr/bin/env python3
"""
17_ab_models.py  -  Two-producer packs (same idea, two sonic personalities)
Run the SAME pack plan with the SAME seeds through two different fine-tuned
checkpoints (e.g. a 70s-soul model vs a Memphis-90s model). Outputs paired
A/ and B/ folders where item N in each was generated from an identical seed
and prompt - only the model differs.

Usage:
    python 17_ab_models.py --plan prompts/pack_plan.example.json \
        --model-a-config cfgA.json --model-a-ckpt soulA.ckpt \
        --model-b-config cfgB.json --model-b-ckpt memphisB.ckpt \
        --out ab_packs/ --base-seed 1234
Models are loaded one at a time (VRAM-friendly).
"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sat_common  # noqa: E402


def run_model(tag, model_config, ckpt, pretrained, plan, out_root, base_seed, steps, cfg_scale):
    model, cfg, device = sat_common.load_model(model_config, ckpt, pretrained)
    sr = cfg["sample_rate"]
    item = 0
    for category in plan["categories"]:
        cat_dir = out_root / tag / category["name"]
        seconds = float(category.get("seconds", 4.0))
        for i in range(int(category["count"])):
            seed = base_seed + item
            item += 1
            audio, _ = sat_common.generate(
                model, cfg, category["prompt"], seconds, device,
                steps=steps, cfg_scale=cfg_scale, seed=seed)
            sat_common.save_wav(audio, cat_dir / f"{category['name']}_{i+1:03d}_seed{seed}.wav", sr)
            print(f"[{tag}] {category['name']} {i+1}/{category['count']} (seed {seed})")
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model-a-config")
    ap.add_argument("--model-a-ckpt")
    ap.add_argument("--model-a-pretrained")
    ap.add_argument("--model-b-config")
    ap.add_argument("--model-b-ckpt")
    ap.add_argument("--model-b-pretrained")
    ap.add_argument("--base-seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for m in ("a", "b"):
        pre = getattr(args, f"model_{m}_pretrained")
        cfg_ = getattr(args, f"model_{m}_config")
        ck = getattr(args, f"model_{m}_ckpt")
        if not pre and not (cfg_ and ck):
            ap.error(f"Model {m.upper()}: provide --model-{m}-pretrained or both --model-{m}-config and --model-{m}-ckpt")

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    out_root = Path(args.out)
    run_model("A", args.model_a_config, args.model_a_ckpt, args.model_a_pretrained,
              plan, out_root, args.base_seed, args.steps, args.cfg)
    run_model("B", args.model_b_config, args.model_b_ckpt, args.model_b_pretrained,
              plan, out_root, args.base_seed, args.steps, args.cfg)
    print(f"\nPaired outputs in {out_root}/A and {out_root}/B - item N matches item N.")


if __name__ == "__main__":
    main()
