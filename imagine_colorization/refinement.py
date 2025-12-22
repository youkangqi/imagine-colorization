"""Reference refinement module skeleton.

The paper proposes selecting, aligning, and composing multiple reference
segments to form an optimal guidance image. This module captures the control
points for those steps while keeping computation light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from imagine_colorization.config import RefinementConfig
from imagine_colorization.types import ColorizationSample, ReferenceCandidate, ReferenceComposition


@dataclass
class RefinementOutputs:
    """Outputs from the refinement stage."""

    composition: ReferenceComposition
    selected_indices: List[int]


class ReferenceRefinementModule:
    """Align and compose the best parts of multiple references."""

    def __init__(self, config: RefinementConfig) -> None:
        self.config = config

    def estimate_alignment(self, candidate: ReferenceCandidate, sample: ColorizationSample) -> np.ndarray:
        """Estimate structural alignment between candidate and grayscale input."""

        # Placeholder: identity alignment (no warping).
        return candidate.image

    def compute_masks(self, aligned: np.ndarray, sample: ColorizationSample) -> np.ndarray:
        """Compute instance-aware masks to pick salient regions."""

        if sample.segmentation is not None:
            return sample.segmentation
        height, width = aligned.shape[:2]
        return np.ones((height, width, 1), dtype="float32")

    def rank_segments(self, aligned: np.ndarray, mask: np.ndarray, sample: ColorizationSample) -> float:
        """Rank aligned segments using semantic and structural cues."""

        # Placeholder: treat every candidate equally.
        return 1.0

    def compose(self, candidates: List[ReferenceCandidate], sample: ColorizationSample) -> ReferenceComposition:
        """Compose the top-ranked segments into a single reference image."""

        aligned_images = [self.estimate_alignment(c, sample) for c in candidates]
        masks = [self.compute_masks(img, sample) for img in aligned_images]
        scores = [self.rank_segments(img, mask, sample) for img, mask in zip(aligned_images, masks)]

        # Select top-k candidates by score.
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected_indices = sorted_indices[: self.config.max_segments]

        # Naive composition: average the selected aligned images.
        stacked = np.stack([aligned_images[i] for i in selected_indices], axis=0)
        composed = stacked.mean(axis=0)
        composed_mask = masks[selected_indices[0]]

        provenance = {f"candidate_{i}": i for i in selected_indices}
        return ReferenceComposition(image=composed, mask=composed_mask, provenance=provenance)

    def __call__(self, candidates: List[ReferenceCandidate], sample: ColorizationSample) -> RefinementOutputs:
        composition = self.compose(candidates, sample)
        selected_indices = list(composition.provenance.values())
        return RefinementOutputs(composition=composition, selected_indices=selected_indices)
