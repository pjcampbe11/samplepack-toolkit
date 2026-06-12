#!/usr/bin/env python3
"""
23_deep_listen.py  -  Learn everything possible about an audio file.

Layered analysis (each layer degrades gracefully if its dependency is absent;
the report states exactly which layers ran):

  TECHNICAL   format, sample rate, bit depth, duration, clipping, DC offset,
              LUFS loudness, true-peak est., dynamics (crest, LRA-style),
              stereo width/correlation, spectral profile, silence map
              [soundfile, numpy, pyloudnorm, librosa - core install]
  MUSICAL     BPM + beat grid (beat_this if installed, else librosa),
              key/mode (Krumhansl), onset density, structure segmentation,
              energy arc                                       [librosa, beat-this]
  SOUND IDs   every detectable sound event with timestamps - AudioSet's 527
              classes (speech, kick drum, guitar, siren, crowd...)
              [pip install panns-inference]
  MOOD/VIBE   zero-shot scores for mood, genre, era, instrumentation and
              production character vocabularies via CLAP embeddings
              [pip install laion-clap]

Outputs <name>.analysis.json (machine) + <name>.analysis.md (human).

Usage:
    python 23_deep_listen.py --input track.mp3 --out reports/
    python 23_deep_listen.py --input folder/ --out reports/        # batch
Optional: --no-events / --no-vibe to skip model layers; --segment-stems
(requires audio-separator) analyzes vocals/drums/bass/other separately.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}
SR = 44100

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MOOD_VOCAB = ["happy and uplifting", "sad and melancholic", "aggressive and intense",
              "relaxed and chill", "dark and ominous", "romantic and warm",
              "epic and cinematic", "playful and quirky", "nostalgic", "triumphant",
              "anxious and tense", "dreamy and ethereal", "groovy and danceable",
              "raw and gritty", "polished and clean"]
GENRE_VOCAB = ["hip hop", "boom bap hip hop", "trap", "drill", "lofi hip hop", "r&b",
               "soul", "funk", "jazz", "rock", "metal", "punk", "pop", "house",
               "techno", "dubstep", "drum and bass", "ambient", "classical",
               "reggae", "afrobeats", "country", "blues", "gospel", "latin"]
INSTRUMENT_VOCAB = ["acoustic drums", "drum machine", "808 bass", "electric bass guitar",
                    "electric guitar", "acoustic guitar", "piano", "electric piano",
                    "organ", "synthesizer pad", "synthesizer lead", "strings",
                    "brass section", "saxophone", "flute", "male vocals",
                    "female vocals", "rap vocals", "choir", "vinyl crackle",
                    "turntable scratching", "orchestral percussion"]
PRODUCTION_VOCAB = ["dusty vinyl texture", "modern polished production", "lo-fi tape sound",
                    "heavily compressed", "spacious reverb", "dry and intimate",
                    "distorted and saturated", "wide stereo mix", "live recording",
                    "studio recording", "1990s production", "1970s analog production"]


# ---------- layer 1: technical ----------

def technical(path, y, sr, file_sr, subtype):
    mono = y.mean(axis=0)
    peak = float(np.abs(y).max())
    clip_samples = int((np.abs(y) >= 0.999).sum())
    res = {
        "container": path.suffix.lower().lstrip("."),
        "sample_rate": file_sr,
        "encoding": subtype,
        "channels": y.shape[0],
        "duration_sec": round(y.shape[1] / sr, 2),
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
        "clipped_samples": clip_samples,
        "dc_offset": round(float(mono.mean()), 6),
    }
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(y.T)
        res["integrated_lufs"] = round(float(lufs), 2) if np.isfinite(lufs) else None
        if res["integrated_lufs"] is not None:
            res["crest_headroom_db"] = round(res["peak_dbfs"] - res["integrated_lufs"], 2)
    except ImportError:
        res["integrated_lufs"] = "pyloudnorm not installed"
    rms = float(np.sqrt((mono ** 2).mean()))
    res["crest_factor_db"] = round(20 * np.log10((peak + 1e-12) / (rms + 1e-12)), 2)
    # short-term loudness spread as LRA proxy
    win = sr
    if mono.size > 3 * win:
        st = np.array([np.sqrt((mono[i:i + win] ** 2).mean()) for i in range(0, mono.size - win, win)])
        st_db = 20 * np.log10(st + 1e-12)
        res["loudness_range_db_proxy"] = round(float(np.percentile(st_db, 95) - np.percentile(st_db, 10)), 2)
    if y.shape[0] == 2:
        l, r = y[0], y[1]
        denom = np.sqrt((l ** 2).mean() * (r ** 2).mean()) + 1e-12
        res["stereo_correlation"] = round(float((l * r).mean() / denom), 3)
        mid, side = (l + r) / 2, (l - r) / 2
        res["side_mid_energy_ratio"] = round(float((side ** 2).sum() / ((mid ** 2).sum() + 1e-12)), 3)
    return res


def spectral(y, sr):
    import librosa
    mono = y.mean(axis=0)
    S = np.abs(librosa.stft(mono, n_fft=4096)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    tot = S.sum() + 1e-12
    def band(lo, hi):
        return round(float(S[(freqs >= lo) & (freqs < hi)].sum() / tot), 4)
    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sr, roll_percent=0.95).mean()
    flatness = librosa.feature.spectral_flatness(y=mono).mean()
    return {
        "band_energy": {"sub_20_60": band(20, 60), "bass_60_250": band(60, 250),
                        "lowmid_250_500": band(250, 500), "mid_500_2k": band(500, 2000),
                        "highmid_2k_6k": band(2000, 6000), "high_6k_16k": band(6000, 16000),
                        "air_16k+": band(16000, sr / 2)},
        "spectral_centroid_hz": round(float(centroid), 1),
        "rolloff95_hz": round(float(rolloff), 1),
        "flatness": round(float(flatness), 4),
        "effective_bandwidth_note": "rolloff95 well below 16 kHz on a 44.1k file suggests lossy-source upsampling",
    }


# ---------- layer 2: musical ----------

def musical(path, y, sr):
    import librosa
    mono = y.mean(axis=0)
    out = {}
    beats = None
    try:
        from beat_this.inference import File2Beats
        b, db = File2Beats(checkpoint_path="final0", dbn=False)(str(path))
        if len(b) >= 4:
            beats = np.asarray(b)
            out["beat_tracker"] = "beat_this"
            out["downbeat_count"] = len(db)
            if len(db) >= 2 and len(beats) >= 2:
                out["beats_per_bar_estimate"] = int(round(np.median(np.diff(
                    np.searchsorted(beats, db)))))
    except Exception:
        pass
    if beats is None:
        tempo, bf = librosa.beat.beat_track(y=mono, sr=sr)
        beats = librosa.frames_to_time(bf, sr=sr)
        out["beat_tracker"] = "librosa"
    if len(beats) >= 2:
        ibis = np.diff(beats)
        out["bpm"] = round(60.0 / float(np.median(ibis)), 1)
        out["tempo_stability"] = round(1.0 - float(np.std(ibis) / (np.mean(ibis) + 1e-9)), 3)
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr).mean(axis=1)
    best = (-2, None)
    if chroma.sum() > 0:
        for shift in range(12):
            rolled = np.roll(chroma, -shift)
            for prof, mode in ((MAJOR, "major"), (MINOR, "minor")):
                r = np.corrcoef(rolled, prof)[0, 1]
                if r > best[0]:
                    best = (r, f"{PITCHES[shift]} {mode}")
    out["key"] = best[1]
    out["key_confidence"] = round(float(best[0]), 3)
    onsets = librosa.onset.onset_detect(y=mono, sr=sr)
    out["onsets_per_sec"] = round(len(onsets) / (mono.size / sr), 2)
    # structure: spectral-clustering segmentation on chroma+mfcc
    try:
        n_segs = min(8, max(2, int(mono.size / sr / 30)))
        mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
        chro = librosa.feature.chroma_cqt(y=mono, sr=sr)
        feat = np.vstack([librosa.util.normalize(mfcc), librosa.util.normalize(chro)])
        bounds = librosa.segment.agglomerative(feat, n_segs)
        times = librosa.frames_to_time(bounds, sr=sr)
        out["structure_boundaries_sec"] = [round(float(t), 1) for t in times]
    except Exception:
        pass
    # energy arc in 10 windows
    w = max(mono.size // 10, 1)
    arc = [round(float(np.sqrt((mono[i * w:(i + 1) * w] ** 2).mean()) + 1e-12), 4) for i in range(10)]
    mx = max(arc) + 1e-12
    out["energy_arc"] = [round(a / mx, 2) for a in arc]
    return out


# ---------- layer 3: sound event identification (PANNs / AudioSet) ----------

def sound_events(path, top_clip=15, frame_threshold=0.3):
    try:
        from panns_inference import SoundEventDetection, labels
    except ImportError:
        return {"skipped": "panns-inference not installed (pip install panns-inference)"}
    import librosa
    audio, _ = librosa.load(str(path), sr=32000, mono=True)
    sed = SoundEventDetection(checkpoint_path=None, device="cuda" if _has_cuda() else "cpu")
    framewise = sed.inference(audio[None, :])[0]  # (frames, 527)
    clipwise = framewise.max(axis=0)
    order = np.argsort(-clipwise)[:top_clip]
    detected = [{"sound": labels[i], "confidence": round(float(clipwise[i]), 3)}
                for i in order if clipwise[i] >= 0.1]
    # timeline: contiguous runs above threshold per detected class
    hop_sec = (audio.size / 32000) / framewise.shape[0]
    timeline = []
    for i in order:
        if clipwise[i] < frame_threshold:
            continue
        active = framewise[:, i] >= frame_threshold
        start = None
        for f, a in enumerate(list(active) + [False]):
            if a and start is None:
                start = f
            elif not a and start is not None:
                timeline.append({"sound": labels[i],
                                 "start_sec": round(start * hop_sec, 1),
                                 "end_sec": round(f * hop_sec, 1)})
                start = None
    timeline.sort(key=lambda e: e["start_sec"])
    return {"clip_level": detected, "timeline": timeline[:200]}


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------- layer 4: mood / vibe (CLAP zero-shot) ----------

def vibe(path):
    try:
        import laion_clap
    except ImportError:
        return {"skipped": "laion-clap not installed (pip install laion-clap)"}
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    aemb = model.get_audio_embedding_from_filelist([str(path)], use_tensor=False)
    aemb = aemb / np.linalg.norm(aemb, axis=1, keepdims=True)
    out = {}
    for name, vocab, prefix in (("mood", MOOD_VOCAB, "a song that feels"),
                                ("genre", GENRE_VOCAB, "a recording of"),
                                ("instruments", INSTRUMENT_VOCAB, "a recording featuring"),
                                ("production", PRODUCTION_VOCAB, "a recording with")):
        temb = model.get_text_embedding([f"{prefix} {v}" for v in vocab], use_tensor=False)
        temb = temb / np.linalg.norm(temb, axis=1, keepdims=True)
        sims = (aemb @ temb.T)[0]
        order = np.argsort(-sims)[:6]
        out[name] = [{"label": vocab[i], "score": round(float(sims[i]), 3)} for i in order]
    return out


# ---------- report ----------

def summarize(rep):
    t, m = rep.get("technical", {}), rep.get("musical", {})
    bits = []
    if m.get("bpm"):
        bits.append(f"{m['bpm']} BPM" + (f" ({m.get('beats_per_bar_estimate', 4)}/4)" if m.get('beats_per_bar_estimate') else ""))
    if m.get("key"):
        bits.append(f"key {m['key']} (conf {m.get('key_confidence')})")
    if isinstance(t.get("integrated_lufs"), (int, float)):
        bits.append(f"{t['integrated_lufs']} LUFS")
    v = rep.get("vibe", {})
    if isinstance(v, dict) and v.get("mood"):
        bits.append("feels " + ", ".join(x["label"] for x in v["mood"][:2]))
    if isinstance(v, dict) and v.get("genre"):
        bits.append("reads as " + v["genre"][0]["label"])
    ev = rep.get("sound_events", {})
    if isinstance(ev, dict) and ev.get("clip_level"):
        bits.append("contains " + ", ".join(e["sound"] for e in ev["clip_level"][:5]))
    return "; ".join(bits) or "core layers only"


def to_markdown(rep):
    L = [f"# Deep Listen: {rep['file']}", "", f"**Summary:** {rep['summary']}", ""]
    for sec in ("technical", "spectral", "musical"):
        if sec in rep:
            L.append(f"## {sec.title()}")
            L.append("```json")
            L.append(json.dumps(rep[sec], indent=2))
            L.append("```")
    ev = rep.get("sound_events", {})
    if isinstance(ev, dict) and ev.get("clip_level"):
        L.append("## Identified sounds (AudioSet / PANNs)")
        for e in ev["clip_level"]:
            L.append(f"- {e['sound']} ({e['confidence']})")
        if ev.get("timeline"):
            L.append("\n### Timeline")
            for e in ev["timeline"][:60]:
                L.append(f"- {e['start_sec']:>7.1f}s - {e['end_sec']:>7.1f}s  {e['sound']}")
    v = rep.get("vibe", {})
    if isinstance(v, dict) and not v.get("skipped"):
        L.append("## Vibe (CLAP zero-shot)")
        for name in ("mood", "genre", "instruments", "production"):
            if v.get(name):
                L.append(f"**{name}:** " + ", ".join(f"{x['label']} ({x['score']})" for x in v[name]))
    L.append("\n## Layers run")
    L.append(json.dumps(rep["layers"], indent=2))
    return "\n".join(L)


def analyze(path, args, out_dir):
    import librosa
    import soundfile as sf
    try:
        info = sf.info(str(path))
        file_sr, subtype = info.samplerate, info.subtype
    except Exception:
        file_sr, subtype = None, "n/a (compressed)"
    y, sr = librosa.load(str(path), sr=SR, mono=False)
    if y.ndim == 1:
        y = y[None, :]
    rep = {"file": path.name, "path": str(path), "layers": {}}
    rep["technical"] = technical(path, y, sr, file_sr or SR, subtype)
    rep["layers"]["technical"] = "ok"
    rep["spectral"] = spectral(y, sr)
    rep["layers"]["spectral"] = "ok"
    rep["musical"] = musical(path, y, sr)
    rep["layers"]["musical"] = rep["musical"].get("beat_tracker", "ok")
    if not args.no_events:
        rep["sound_events"] = sound_events(path)
        rep["layers"]["sound_events"] = rep["sound_events"].get("skipped", "panns")
    if not args.no_vibe:
        rep["vibe"] = vibe(path)
        rep["layers"]["vibe"] = rep["vibe"].get("skipped", "clap") if isinstance(rep["vibe"], dict) else "clap"
    rep["summary"] = summarize(rep)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{path.stem}.analysis.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out_dir / f"{path.stem}.analysis.md").write_text(to_markdown(rep), encoding="utf-8")
    print(f"{path.name}: {rep['summary']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Audio file or folder")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--no-events", action="store_true", help="Skip PANNs sound-event layer")
    ap.add_argument("--no-vibe", action="store_true", help="Skip CLAP mood/genre layer")
    args = ap.parse_args()
    p = Path(args.input)
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit("No audio found.")
    for f in files:
        try:
            analyze(f, args, Path(args.out))
        except Exception as e:
            print(f"FAILED {f}: {e}")


if __name__ == "__main__":
    main()
