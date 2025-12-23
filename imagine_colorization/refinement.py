"""Reference refinement module skeleton.

The paper proposes selecting, aligning, and composing multiple reference
segments to form an optimal guidance image. This module captures the control
points for those steps while keeping computation light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from imagine_colorization.adapters.dinov2_adapter import DinoV2Adapter
from imagine_colorization.adapters.semantic_sam_adapter import SemanticSamAdapter
from imagine_colorization.config import RefinementConfig
from imagine_colorization.types import ColorizationSample, ReferenceCandidate, ReferenceComposition
from imagine_colorization.utils.vision import resize


@dataclass
class RefinementOutputs:
    """Outputs from the refinement stage."""

    composition: ReferenceComposition
    selected_indices: List[int]


class ReferenceRefinementModule:
    """Align and compose the best parts of multiple references."""

    def __init__(self, config: RefinementConfig) -> None:
        self.config = config
        self.semantic_sam = SemanticSamAdapter(config.semantic_sam)
        self.feature_extractor = DinoV2Adapter(config.dino)

    def estimate_alignment(self, candidate: ReferenceCandidate, sample: ColorizationSample) -> np.ndarray:
        """Estimate structural alignment between candidate and grayscale input."""

        # Placeholder: identity alignment (no warping).
        return candidate.image

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 1:
            return np.repeat(image, 3, axis=-1)
        return image

    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if image.shape[:2] == size:
            return image
        return resize(image, size)

    def _resize_mask(self, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if mask.shape[:2] == size:
            return mask.astype(bool)
        resized = resize(mask.astype("float32"), size)
        return resized > 0.5

    def _blur_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.config.mask_blur <= 0:
            return mask.astype("float32")
        try:
            import cv2
        except Exception:
            # Fall back to a sharp mask if cv2 isn't available.
            return mask.astype("float32")
        sigma = float(self.config.mask_blur)
        mask_u8 = mask.astype("uint8")
        dist_in = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        dist_out = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 3)
        signed = dist_in - dist_out
        scale = signed / max(sigma, 1e-6)
        scale = np.clip(scale, -50.0, 50.0)
        weight = 1.0 / (1.0 + np.exp(-scale))
        max_radius = sigma * 3.0
        if max_radius > 0:
            weight = np.where(signed > max_radius, 1.0, weight)
            weight = np.where(signed < -max_radius, 0.0, weight)
        return weight.astype("float32")

    def _background_weight(self, mask_union: np.ndarray) -> np.ndarray:
        if mask_union.sum() == 0:
            return np.ones_like(mask_union, dtype="float32")
        if mask_union.all():
            return np.zeros_like(mask_union, dtype="float32")
        return self._blur_mask(~mask_union)

    def _labels_to_masks(self, labels: np.ndarray) -> List[np.ndarray]:
        masks: List[np.ndarray] = []
        for label in np.unique(labels):
            if label == 0:
                continue
            masks.append((labels == label).astype("uint8"))
        return masks

    def _filter_masks(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        if not masks:
            return []
        min_area = self.config.segment_min_area
        overlap_thresh = self.config.overlap_thresh
        max_segments = self.config.max_segments if self.config.max_segments else None
        sorted_masks = sorted(masks, key=lambda m: int(m.sum()), reverse=True)
        selected: List[np.ndarray] = []
        occupied = np.zeros_like(sorted_masks[0], dtype=bool)
        for mask in sorted_masks:
            area = int(mask.sum())
            if area < min_area:
                continue
            overlap = (mask.astype(bool) & occupied).sum()
            if area > 0 and overlap / float(area) > overlap_thresh:
                continue
            selected.append(mask.astype(bool))
            occupied |= mask.astype(bool)
            if max_segments and len(selected) >= max_segments:
                break
        return selected

    def compute_masks(self, sample: ColorizationSample) -> List[np.ndarray]:
        """Compute semantic-aware and instance-aware segments."""

        if sample.segmentation is not None:
            segmentation = sample.segmentation
            if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
                return self._filter_masks([segmentation[..., 0]])
            if segmentation.ndim == 2:
                return self._filter_masks(self._labels_to_masks(segmentation))
            return self._filter_masks([segmentation])

        base_image = self._ensure_rgb(sample.grayscale)
        mask_records = self.semantic_sam.generate_masks(base_image)
        masks = [record["segmentation"] for record in mask_records if "segmentation" in record]
        return self._filter_masks(masks)

    def _masked_mean(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        if mask.sum() == 0:
            return None
        return image[mask].mean(axis=0)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        metric = self.config.distance_metric
        if metric == "l1":
            return float(np.mean(np.abs(a - b)))
        if metric == "l2":
            return float(np.linalg.norm(a - b))
        if metric == "cosine":
            denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            return float(1.0 - (np.dot(a, b) / denom))
        raise ValueError(f"Unknown distance metric: {metric}")

    def _segment_feature(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        try:
            return self.feature_extractor.extract(image, mask)
        except Exception:
            fallback = self._masked_mean(image, mask)
            if fallback is None:
                raise
            return fallback.astype("float32")

    def _select_best_candidate(
        self,
        base_image: np.ndarray,
        aligned_images: List[np.ndarray],
        mask: np.ndarray,
    ) -> Tuple[int, float]:
        base_feat = self._segment_feature(base_image, mask)
        best_idx = 0
        best_score = float("inf")
        for idx, image in enumerate(aligned_images):
            cand_feat = self._segment_feature(image, mask)
            score = self._distance(base_feat, cand_feat)
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def compose(self, candidates: List[ReferenceCandidate], sample: ColorizationSample) -> ReferenceComposition:
        """Compose the top-ranked segments into a single reference image."""

        if not candidates:
            raise ValueError("No reference candidates provided.")
        aligned_images = [self.estimate_alignment(c, sample) for c in candidates]
        base_image = self._ensure_rgb(sample.grayscale)
        base_size = base_image.shape[:2]
        aligned_images = [self._resize_image(img, base_size) for img in aligned_images]
        masks = [self._resize_mask(mask, base_size) for mask in self.compute_masks(sample)]
        if not masks:
            composed = aligned_images[0]
            mask = np.ones(composed.shape[:2], dtype="float32")[..., None]
            return ReferenceComposition(image=composed, mask=mask, provenance={"fallback": 0})

        composed = np.zeros_like(aligned_images[0], dtype="float32")
        mask_union = np.zeros(aligned_images[0].shape[:2], dtype=bool)
        weight_sum = np.zeros(aligned_images[0].shape[:2], dtype="float32")
        provenance: Dict[str, int] = {}
        for idx, mask in enumerate(masks):
            best_idx, _ = self._select_best_candidate(base_image, aligned_images, mask)
            soft_mask = self._blur_mask(mask)
            if soft_mask.ndim == 2:
                soft_mask = soft_mask[..., None]
            composed += aligned_images[best_idx].astype("float32") * soft_mask
            weight_sum += soft_mask[..., 0]
            mask_union |= mask.astype(bool)
            provenance[f"segment_{idx}"] = best_idx

        if not mask_union.all():
            background_mask = ~mask_union
            background_idx, _ = self._select_best_candidate(base_image, aligned_images, background_mask)
            background = self._background_weight(mask_union)
            if background.ndim == 2:
                background = background[..., None]
            fallback = aligned_images[background_idx].astype("float32")
            composed += fallback * background
            weight_sum += background[..., 0]
            provenance["background"] = background_idx

        weight_sum = np.clip(weight_sum, 1e-6, None)
        composed = composed / weight_sum[..., None]

        composed_mask = mask_union.astype("float32")[..., None]
        return ReferenceComposition(
            image=composed.astype("uint8"),
            mask=composed_mask,
            provenance=provenance,
        )

    def __call__(self, candidates: List[ReferenceCandidate], sample: ColorizationSample) -> RefinementOutputs:
        composition = self.compose(candidates, sample)
        selected_indices = list(composition.provenance.values())
        return RefinementOutputs(composition=composition, selected_indices=selected_indices)
