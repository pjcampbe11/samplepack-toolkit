#!/usr/bin/env bash
# Run this ON the cloud GPU instance (RunPod / Lambda / Vast.ai, Ubuntu + CUDA image).
# Tested target: 1x A100 80GB or A6000 48GB. PyTorch >= 2.3, Python 3.10+.
set -euo pipefail

cd /workspace

# 1. stable-audio-tools
git clone https://github.com/Stability-AI/stable-audio-tools.git
cd stable-audio-tools
pip install -e .
pip install wandb
cd /workspace

# 2. Base model weights + config (requires HF login; accept the license on the
#    model page first: https://huggingface.co/stabilityai/stable-audio-open-1.0)
pip install -U "huggingface_hub[cli]"
hf auth login
hf download stabilityai/stable-audio-open-1.0 model.ckpt model_config.json --local-dir /workspace/base_model

# 3. Your data (upload from your machine first, e.g.:
#    runpodctl send / scp -r dataset/ root@<pod-ip>:/workspace/dataset/
#    and the toolkit:        scp -r hiphop-samplepack-toolkit/ root@<pod-ip>:/workspace/toolkit/ )
echo "Expect dataset at /workspace/dataset and toolkit at /workspace/toolkit"

# 4. Launch training (see README step 5 for flag explanations)
echo "Ready. Start training with:"
cat << 'CMD'
cd /workspace/stable-audio-tools
python train.py \
  --dataset_config /workspace/toolkit/configs/dataset_config.json \
  --model_config /workspace/base_model/model_config.json \
  --pretrained_ckpt_path /workspace/base_model/model.ckpt \
  --name hiphop-finetune \
  --save_dir /workspace/checkpoints \
  --checkpoint_every 1000 \
  --batch_size 8 \
  --accum_batches 2 \
  --num_gpus 1 \
  --precision 16-mixed \
  --seed 42
CMD
# NOTE: depending on your stable-audio-tools version, flags use hyphens instead
# of underscores (--dataset-config, --pretrained-ckpt-path, ...). Run
# `python train.py --help` and match what it prints.
