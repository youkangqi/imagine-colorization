"""SAM2 adapter wrapping the submodule implementation."""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

from imagine_colorization.config import Sam2Config


class Sam2Adapter:
    """Thin wrapper around SAM2 automatic mask generation."""

    def __init__(self, config: Sam2Config) -> None:
        self.config = config
        self._loaded = False
        self._mask_generator = None

    def _ensure_repo_on_path(self) -> None:
        repo_path = os.path.abspath(self.config.repo_path)
        package_path = os.path.join(repo_path, "sam2")
        if package_path not in sys.path:
            sys.path.insert(0, package_path)

    def load(self) -> None:
        """Load SAM2 model and automatic mask generator."""

        if self._loaded:
            return
        self._ensure_repo_on_path()
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except Exception as exc:  # pragma: no cover - depends on external libs
            raise RuntimeError("Failed to import SAM2 dependencies.") from exc

        model = build_sam2(
            config_file=self.config.model_cfg_path,
            ckpt_path=self.config.checkpoint_path,
            device=self.config.device,
        )
        self._mask_generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            min_mask_region_area=self.config.min_mask_region_area,
            output_mode="binary_mask",
        )
        self._loaded = True

    def generate_masks(self, image: np.ndarray) -> List[dict]:
        """Generate raw mask dicts using SAM2 AMG."""

        self.load()
        return self._mask_generator.generate(image)

    def postprocess(self, masks: List[dict]) -> List[np.ndarray]:
        """Extract binary masks from SAM2 AMG output."""

        binary_masks: List[np.ndarray] = []
        for mask in masks:
            if "segmentation" in mask:
                binary_masks.append(mask["segmentation"].astype("uint8"))
        return binary_masks
