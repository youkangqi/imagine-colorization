#!/usr/bin/env bash
set -euo pipefail

GRAY_IMAGE="data/gray_apple.jpeg"
OUTDIR="outputs/pipeline"

BLIP2_MODEL="/homeB/youkangqi/.cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/59a1ef6c1e5117b3f65523d1c6066825bcf315e3"
CONTROLNET_WEIGHTS="ControlNet/models/control_sd15_canny.pth"

python scripts/run_pipeline.py \
  --gray "${GRAY_IMAGE}" \
  --outdir "${OUTDIR}" \
  --num-candidates 8 \
  --batch-size 3 \
  --controlnet-weights "${CONTROLNET_WEIGHTS}" \
  --controlnet-device cuda:0 \
  --blip2-model "${BLIP2_MODEL}" \
  --blip2-device cuda:1 \
  --semantic-sam-ckpt model_Seg/swint_only_sam_many2many.pth \
  --semantic-sam-config Semantic-SAM/configs/semantic_sam_only_sa-1b_swinT.yaml \
  --semantic-sam-model-type T \
  --semantic-sam-resize 640 \
  --max-segments 10 \
  --hint-max-per-segment 10 \
  --semantic-sam-device cuda:0 \
  --dino-ckpt model_Seg/dinov2_vitl14_pretrain.pth \
  --dino-repo dinov2 \
  --dino-arch dinov2_vitl14 \
  --dino-device cuda:1 \
  --color-mode hint_propagation \
  --save-json
