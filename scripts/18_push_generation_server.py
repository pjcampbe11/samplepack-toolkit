#!/usr/bin/env python3
"""
18_push_generation_server.py  -  Push/Live as a generation instrument
An OSC server that holds your fine-tuned model in memory and fires generation
jobs when it receives messages - map Push pads/knobs in Live to OSC and
generation becomes performance, not a render queue.

OSC API (default listen 127.0.0.1:11001):
  /gen/preset  <int>           select a prompt preset (from --presets JSON)
  /gen/strength <float 0..1>   set a2a strength (used when a source is loaded)
  /gen/source  <path>          set source WAV for audio-to-audio ("" = text mode)
  /gen/fire                    generate now -> WAV lands in --out
Generated files land in --out; point a Live browser location there, or set
--out to your Live Project's Samples folder. New files appear on Push's browser.

Sending OSC from Live: Ableton's free "Connection Kit" M4L pack has an OSC Send
device - map its buttons/dials (and therefore Push controls) to the addresses
above. Anything that speaks OSC works (TouchOSC, a phone, etc).

Presets JSON: {"presets": [{"name": "kick", "prompt": "...", "seconds": 1.5}, ...]}

Usage:
    pip install python-osc
    python 18_push_generation_server.py --model-config model_config.json \
        --ckpt hiphop_v1.ckpt --presets prompts/push_presets.example.json \
        --out "C:/Users/you/Documents/Ableton/GenSamples"
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sat_common  # noqa: E402


class State:
    preset_idx = 0
    strength = 0.4
    source = None
    busy = False


def main():
    ap = argparse.ArgumentParser()
    sat_common.add_model_args(ap)
    ap.add_argument("--presets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11001)
    ap.add_argument("--steps", type=int, default=60,
                    help="Lower = faster pad-to-sound (60 is a good live setting)")
    args = ap.parse_args()
    sat_common.validate_model_args(ap, args)

    import json
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer

    presets = json.loads(Path(args.presets).read_text(encoding="utf-8"))["presets"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model (stays in memory)...")
    model, cfg, device = sat_common.load_model(args.model_config, args.ckpt, args.pretrained)
    sr = cfg["sample_rate"]
    st = State()

    def on_preset(addr, idx):
        st.preset_idx = int(idx) % len(presets)
        print(f"preset -> [{st.preset_idx}] {presets[st.preset_idx]['name']}")

    def on_strength(addr, v):
        st.strength = min(max(float(v), 0.05), 1.0)
        print(f"strength -> {st.strength:.2f}")

    def on_source(addr, path):
        st.source = path if path and Path(path).exists() else None
        print(f"source -> {st.source or 'text mode'}")

    def on_fire(addr, *a):
        if st.busy:
            print("busy - ignored")
            return
        st.busy = True
        try:
            p = presets[st.preset_idx]
            seconds = float(p.get("seconds", 2.0))
            init = None
            if st.source:
                init = sat_common.load_audio_file(st.source, sr, device)
                seconds = min(init.shape[1] / sr, cfg["sample_size"] / sr)
            t0 = time.time()
            audio, seed = sat_common.generate(
                model, cfg, p["prompt"], seconds, device, steps=args.steps,
                init_audio=init, strength=st.strength if init is not None else None)
            name = f"{p['name']}_{time.strftime('%H%M%S')}_seed{seed}.wav"
            sat_common.save_wav(audio, out_dir / name, sr)
            print(f"FIRED [{p['name']}] {time.time()-t0:.1f}s -> {name}")
        finally:
            st.busy = False

    disp = Dispatcher()
    disp.map("/gen/preset", on_preset)
    disp.map("/gen/strength", on_strength)
    disp.map("/gen/source", on_source)
    disp.map("/gen/fire", on_fire)

    print(f"Presets: {[p['name'] for p in presets]}")
    print(f"Listening on {args.host}:{args.port} - output -> {out_dir}")
    BlockingOSCUDPServer((args.host, args.port), disp).serve_forever()


if __name__ == "__main__":
    main()
