"""
custom_metadata.py
Custom metadata module for stable-audio-tools. Reads the JSON sidecar written by
01_prepare_dataset.py (same filename as the audio, .json extension) and returns
its prompt for text conditioning. Samples without a valid prompt are rejected.

Referenced from configs/dataset_config.json via "custom_metadata_module".
"""
import json
from pathlib import Path


def get_custom_metadata(info, audio):
    sidecar = Path(info["path"]).with_suffix(".json")
    if not sidecar.exists():
        return {"__reject__": True}
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {"__reject__": True}
    prompt = (meta.get("prompt") or "").strip()
    if not prompt:
        return {"__reject__": True}
    return {"prompt": prompt}
