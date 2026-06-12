#!/usr/bin/env python3
"""
21_provenance.py  -  Provenance as product
Aggregate the audit trail the toolkit already produces into a single
PROVENANCE.json + human-readable certificate for a pack:
  - training sources (the 'source' fields from dataset sidecars)
  - training run identifier you supply
  - generation seeds (parsed from generated/ filenames)
  - file inventory + SHA-256 of the pack zip

"100% rights-cleared, provenance-verified AI" is a differentiator - this makes
the claim checkable.

Usage:
    python 21_provenance.py --pack packs/DustyCratesVol1 --dataset dataset \
        --generated generated --run-name hiphop-finetune-v1 \
        --statement "All training audio owned by Patrick Campbell Productions."
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

SEED_RE = re.compile(r"seed(\d+)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="Built pack folder from 05_build_pack.py")
    ap.add_argument("--dataset", help="Training dataset dir (for source inventory)")
    ap.add_argument("--generated", help="Raw generation dir (for seed inventory)")
    ap.add_argument("--run-name", default="", help="Training run identifier")
    ap.add_argument("--statement", default="", help="Rights statement to embed")
    args = ap.parse_args()

    pack = Path(args.pack)
    if not pack.is_dir():
        raise SystemExit(f"Pack folder not found: {pack}")

    prov = {
        "pack": pack.name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": "stabilityai/stable-audio-open-1.0 (Stability AI Community License)",
        "training_run": args.run_name,
        "rights_statement": args.statement,
    }

    if args.dataset:
        sources = set()
        for sc in Path(args.dataset).rglob("*.json"):
            try:
                src = json.loads(sc.read_text(encoding="utf-8")).get("source")
                if src:
                    sources.add(src)
            except Exception:
                pass
        prov["training_sources"] = {"count": len(sources), "files": sorted(sources)}

    if args.generated:
        seeds = sorted({int(m.group(1)) for p in Path(args.generated).rglob("*.wav")
                        for m in [SEED_RE.search(p.stem)] if m})
        prov["generation_seeds"] = {"count": len(seeds), "seeds": seeds}

    files = sorted(p for p in pack.rglob("*.wav"))
    prov["pack_files"] = {"count": len(files),
                          "sha256": {str(p.relative_to(pack)): sha256(p) for p in files}}
    zip_path = pack.with_suffix(".zip")
    if zip_path.exists():
        prov["zip_sha256"] = sha256(zip_path)

    out_json = pack / "PROVENANCE.json"
    out_json.write_text(json.dumps(prov, indent=2), encoding="utf-8")

    cert = [
        f"PROVENANCE CERTIFICATE - {pack.name}",
        "=" * 50,
        f"Issued: {prov['generated_at']}",
        f"Base model: {prov['base_model']}",
        f"Training run: {args.run_name or '(unspecified)'}",
        f"Training sources on file: {prov.get('training_sources', {}).get('count', 'n/a')}",
        f"Generation seeds on file: {prov.get('generation_seeds', {}).get('count', 'n/a')}",
        f"Samples in pack: {prov['pack_files']['count']}",
        f"Pack zip SHA-256: {prov.get('zip_sha256', 'n/a')}",
        "",
        args.statement or "(no rights statement supplied)",
        "",
        "Full machine-readable record: PROVENANCE.json",
    ]
    (pack / "PROVENANCE_CERTIFICATE.txt").write_text("\n".join(cert), encoding="utf-8")
    print(f"Wrote {out_json} and PROVENANCE_CERTIFICATE.txt")
    print("Tip: rebuild the zip after adding these (rerun 05_build_pack.py or re-zip).")


if __name__ == "__main__":
    main()
