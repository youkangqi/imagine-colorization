#!/usr/bin/env bash
set -euo pipefail

/homeB/youkangqi/miniconda3/envs/sa_sam/bin/python scripts/refine_reference.py \
  --gray graypic.jpg \
  --candidates "outputs/candidates/candidate_*.png" \
  --out outputs/refined.png \
  --semantic-sam-ckpt model_Seg/swint_only_sam_many2many.pth \
  --semantic-sam-model-type T \
  --semantic-sam-config Semantic-SAM/configs/semantic_sam_only_sa-1b_swinT.yaml \
  --semantic-sam-resize 640 \
  --dino-ckpt model_Seg/dinov2_vitl14_pretrain.pth \
  --dino-repo dinov2 \
  --dino-arch dinov2_vitl14 \
  --dino-device cpu
