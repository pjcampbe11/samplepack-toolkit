#!/usr/bin/env python3
"""
10_ableton_bridge.py
Push beat patterns from 08_beat_builder.py straight into a running Ableton Live
set over OSC - tempo set, MIDI clips created, ready to play from Live or Push.

One-time setup (Live 11/12, any OS):
  1. Install AbletonOSC: https://github.com/ideoforms/AbletonOSC
     (copy the AbletonOSC folder into Live's "Remote Scripts" dir, then in
     Live: Preferences > Link/MIDI > Control Surface = AbletonOSC)
  2. pip install python-osc mido
  3. In your Live set: put a Drum Rack on a MIDI track and drag in the samples
     listed in the beat's manifest.json (kick->C1/36, 808->B0/35, snare->D1/38,
     closed hat->F#1/42, perc->B1/47). Add your VST/VST3 chain after the rack.
  4. Your Push plays/edits these clips natively once they exist in the set.

Usage:
    python 10_ableton_bridge.py --beat beats/boom_bap_92bpm_01 --track 0 --scene 0
    python 10_ableton_bridge.py --beat beats/... --track 0 --scene 0 --fire
"""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beat", required=True, help="Beat folder from 08_beat_builder.py")
    ap.add_argument("--track", type=int, default=0, help="Live track index (0-based)")
    ap.add_argument("--scene", type=int, default=0, help="Live scene/clip-slot index (0-based)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11000, help="AbletonOSC listen port")
    ap.add_argument("--fire", action="store_true", help="Launch the clip after creating it")
    args = ap.parse_args()

    import mido
    from pythonosc.udp_client import SimpleUDPClient

    beat_dir = Path(args.beat)
    manifest = json.loads((beat_dir / "manifest.json").read_text(encoding="utf-8"))
    mid = mido.MidiFile(str(beat_dir / "pattern.mid"))

    # Convert MIDI to (pitch, start_beats, duration_beats, velocity) tuples
    tpb = mid.ticks_per_beat
    notes, active, t = [], {}, 0
    for msg in mido.merge_tracks(mid.tracks):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (t, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            if msg.note in active:
                start, vel = active.pop(msg.note)
                notes.append((msg.note, start / tpb, max((t - start) / tpb, 0.05), vel))

    bpm = manifest["bpm"]
    length_beats = manifest["bars"] * 4
    client = SimpleUDPClient(args.host, args.port)

    print(f"Setting tempo to {bpm} BPM")
    client.send_message("/live/song/set/tempo", [float(bpm)])
    time.sleep(0.1)

    print(f"Creating {length_beats}-beat clip at track {args.track}, scene {args.scene}")
    client.send_message("/live/clip_slot/delete_clip", [args.track, args.scene])
    time.sleep(0.1)
    client.send_message("/live/clip_slot/create_clip", [args.track, args.scene, float(length_beats)])
    time.sleep(0.2)

    for pitch, start, dur, vel in notes:
        client.send_message("/live/clip/add/notes",
                            [args.track, args.scene, pitch, start, dur, vel, False])
    client.send_message("/live/clip/set/name", [args.track, args.scene, beat_dir.name])
    print(f"Wrote {len(notes)} notes")

    if args.fire:
        client.send_message("/live/clip/fire", [args.track, args.scene])
        print("Clip launched")

    print("\nSamples this pattern expects in your Drum Rack (from manifest.json):")
    for inst, path in manifest["samples"].items():
        print(f"  {inst:8s} {path}")


if __name__ == "__main__":
    main()
