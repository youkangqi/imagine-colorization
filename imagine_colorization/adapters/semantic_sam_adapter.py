"""Semantic-SAM adapter wrapping the submodule implementation."""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from imagine_colorization.config import SemanticSamConfig


class SemanticSamAdapter:
    """Thin wrapper around Semantic-SAM automatic mask generation."""

    def __init__(self, config: SemanticSamConfig) -> None:
        self.config = config
        self._loaded = False
        self._mask_generator = None
        self._torch = None

    def _ensure_repo_on_path(self) -> None:
        repo_path = os.path.abspath(self.config.repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)
        repo_path = os.path.abspath(self.config.repo_path)
        return os.path.abspath(os.path.join(repo_path, path))

    def _resize_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        if not self.config.resize_short_edge:
            return image, None
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PIL 未安装，无法进行图像缩放。") from exc
        height, width = image.shape[:2]
        short_edge = min(height, width)
        target = self.config.resize_short_edge
        if short_edge == target:
            return image, None
        scale = float(target) / float(short_edge)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        resized = Image.fromarray(image).resize(new_size, resample=Image.BICUBIC)
        return np.asarray(resized), (height, width)

    def _resize_mask(self, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PIL 未安装，无法进行 mask 缩放。") from exc
        target_h, target_w = size
        mask_img = Image.fromarray(mask.astype("uint8") * 255)
        resized = mask_img.resize((target_w, target_h), resample=Image.NEAREST)
        return (np.asarray(resized) > 0).astype("uint8")

    def load(self) -> None:
        if self._loaded:
            return
        self._ensure_repo_on_path()
        try:
            import torch
            from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator
            from semantic_sam.BaseModel import BaseModel
            from semantic_sam import build_model
            from utils.arguments import load_opt_from_config_file
        except Exception as exc:  # pragma: no cover - depends on external libs
            message = str(exc)
            if "MultiScaleDeformableAttention" in message:
                message += (
                    "\nPlease compile Semantic-SAM CUDA ops: "
                    "`cd Semantic-SAM/semantic_sam/body/encoder/ops && sh make.sh`"
                )
            raise RuntimeError(f"Failed to import Semantic-SAM dependencies: {message}") from exc

        checkpoint_path = self._resolve_path(self.config.checkpoint_path)
        if self.config.model_type:
            repo_path = os.path.abspath(self.config.repo_path)
            current_dir = os.getcwd()
            try:
                os.chdir(repo_path)
                model = build_semantic_sam(model_type=self.config.model_type, ckpt=checkpoint_path)
            finally:
                os.chdir(current_dir)
        else:
            config_path = self._resolve_path(self.config.config_path)
            opt = load_opt_from_config_file(config_path)
            model = BaseModel(opt, build_model(opt)).from_pretrained(checkpoint_path)
            model.eval()
            model.to(self.config.device)

        self._mask_generator = SemanticSamAutomaticMaskGenerator(
            model,
            points_per_side=self.config.points_per_side,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            min_mask_region_area=self.config.min_mask_region_area,
            level=self.config.levels,
        )
        self._torch = torch
        self._loaded = True

    def generate_masks(self, image: np.ndarray) -> List[dict]:
        """Generate mask records for a given image."""

        self.load()
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        resized, original_size = self._resize_image(image)
        tensor = self._torch.from_numpy(resized.copy()).permute(2, 0, 1).to(self.config.device)
        with self._torch.no_grad():
            masks = self._mask_generator.generate(tensor)
        if original_size is None:
            return masks
        for mask in masks:
            segmentation = mask.get("segmentation")
            if segmentation is None:
                continue
            mask["segmentation"] = self._resize_mask(segmentation, original_size)
        return masks
