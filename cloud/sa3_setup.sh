#!/usr/bin/env bash
# Stable Audio 3 setup (June 2026 path). LoRA fine-tunes run on a single
# 16-24 GB GPU (RTX 4090 / A10 works) - far cheaper than the SAO full fine-tune.
set -euo pipefail
cd /workspace

# uv + repo
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/Stability-AI/stable-audio-3
cd stable-audio-3
uv sync --extra lora

# Flash Attention 2 (required for Medium) - use a prebuilt wheel matching your
# CUDA/torch/python; browse: github.com/mjun0812/flash-attention-prebuild-wheels
# Example (CUDA 12.6, torch 2.7, py3.10):
# uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu126torch2.7-cp310-cp310-linux_x86_64.whl
# Then keep it across syncs:  uv sync --inexact

# Accept model licenses on HF first: huggingface.co/stabilityai/stable-audio-3-medium
uv run hf auth login

echo "=== LoRA training (from toolkit dataset) ==="
cat << 'CMD'
# 1. stage data (toolkit dataset -> SA3 format):
uv run python /workspace/toolkit/scripts/22_sa3_workflow.py prepare \
    --dataset /workspace/dataset --data-dir /workspace/sa3_data
# 2. confirm caption format:  uv run python scripts/train_lora.py --help
# 3. train (default recipe; ~6.5GB VRAM, or add --base_precision bf16 --adapter_type lora-xs for ~5.5GB):
uv run python scripts/train_lora.py --model medium-base \
    --data_dir /workspace/sa3_data --rank 16 --adapter_type dora-rows \
    --steps 1000 --exclude seconds_total
# 4. generate with it:
uv run python /workspace/toolkit/scripts/22_sa3_workflow.py plan \
    --model medium-base --lora lora_out/lora_step1000.safetensors \
    --plan /workspace/toolkit/prompts/pack_plan.example.json --out /workspace/generated
CMD
