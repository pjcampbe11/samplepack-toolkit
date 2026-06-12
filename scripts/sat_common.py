"""
sat_common.py
Shared model-inference helpers for the creative-technique scripts (12-19).
Requires stable-audio-tools + a GPU for anything that generates audio.
"""
import json
from pathlib import Path


def add_model_args(ap):
    ap.add_argument("--model-config")
    ap.add_argument("--ckpt")
    ap.add_argument("--pretrained", help="HF model id instead of local checkpoint")


def validate_model_args(ap, args):
    if not args.pretrained and not (args.model_config and args.ckpt):
        ap.error("Provide either --pretrained, or both --model-config and --ckpt.")


def load_model(model_config=None, ckpt=None, pretrained=None, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if pretrained:
        from stable_audio_tools import get_pretrained_model
        model, cfg = get_pretrained_model(pretrained)
    else:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict
        cfg = json.loads(Path(model_config).read_text(encoding="utf-8"))
        model = create_model_from_config(cfg)
        model.load_state_dict(load_ckpt_state_dict(ckpt))
    return model.to(device).eval().requires_grad_(False), cfg, device


def load_audio_file(path, target_sr, device):
    """Load any audio file -> normalized stereo tensor on device."""
    import torch
    import torchaudio
    audio, sr = torchaudio.load(str(path))
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    peak = audio.abs().max().clamp(min=1e-8)
    return (audio / peak).to(device)


def generate(model, cfg, prompt, seconds, device, steps=100, cfg_scale=7.0,
             seed=None, init_audio=None, strength=None):
    """Text-to-audio, or audio-to-audio when init_audio (+strength 0..1) given.
    Returns (audio_tensor [2, n], seed)."""
    import torch
    from einops import rearrange
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    if seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
    kwargs = {}
    if init_audio is not None:
        kwargs["init_audio"] = (cfg["sample_rate"], init_audio)
        kwargs["init_noise_level"] = float(strength) * 10.0
    audio = generate_diffusion_cond(
        model, steps=steps, cfg_scale=cfg_scale,
        conditioning=[{"prompt": prompt, "seconds_start": 0, "seconds_total": seconds}],
        sample_size=cfg["sample_size"], sigma_min=0.3, sigma_max=500,
        sampler_type="dpmpp-3m-sde", seed=seed, device=device, **kwargs)
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio[:, : int(seconds * cfg["sample_rate"])]
    return audio, seed


def save_wav(audio, path, sample_rate):
    import torch
    import torchaudio
    audio = audio.to(torch.float32)
    peak = audio.abs().max().clamp(min=1e-8)
    audio = (audio / peak).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio, sample_rate)
