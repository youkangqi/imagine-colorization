#!/usr/bin/env bash
set -euo pipefail

# Modes: hint_propagation | sd15_controlnet
MODE="hint_propagation"

PYTHON_BIN="/homeB/youkangqi/miniconda3/envs/sa_sam/bin/python"
GRAY_IMAGE="image.png"
REFINED_REF="outputs/refined.png"
CAPTION_FILE="outputs/candidates/caption.txt"
OUT_IMAGE="outputs/colorized.png"
HINT_MAP="outputs/hint_map.png"

CONTROLNET_WEIGHTS="ControlNet/models/control_sd15_canny.pth"
CONTROLNET_CONFIG="ControlNet/models/cldm_v15.yaml"

if [[ "${MODE}" == "sd15_controlnet" ]]; then
  ${PYTHON_BIN} scripts/colorize_reference.py \
    --mode sd15_controlnet \
    --gray "${GRAY_IMAGE}" \
    --reference "${REFINED_REF}" \
    --caption-file "${CAPTION_FILE}" \
    --controlnet-weights "${CONTROLNET_WEIGHTS}" \
    --controlnet-config "${CONTROLNET_CONFIG}" \
    --controlnet-repo ControlNet \
    --controlnet-device cuda \
    --controlnet-control-type canny \
    --image-resolution 512 \
    --steps 20 \
    --guidance-scale 9.0 \
    --strength 1.0 \
    --prompt-template "{caption}, realistic, natural lighting, translucent colors" \
    --negative-prompt "monochrome, grayscale, black and white, desaturated, oil painting, cartoon" \
    --control-source grayscale \
    --out "${OUT_IMAGE}"
else
  ${PYTHON_BIN} scripts/colorize_reference.py \
    --mode hint_propagation \
    --gray "${GRAY_IMAGE}" \
    --reference "${REFINED_REF}" \
    --caption-file "${CAPTION_FILE}" \
    --hint-coarse-size 64 \
    --hint-refine-strength 0.7 \
    --hint-refine-blur 9 \
    --propagation-radius 8 \
    --propagation-eps 1e-3 \
    --propagation-sigma-color 12 \
    --propagation-sigma-space 12 \
    --propagation-strength 1.0 \
    --save-hint-map "${HINT_MAP}" \
    --out "${OUT_IMAGE}"
fi
