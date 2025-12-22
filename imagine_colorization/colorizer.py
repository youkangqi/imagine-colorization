"""Reference-guided colorization module skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from imagine_colorization.config import ColorizationConfig
from imagine_colorization.types import ColorizationSample, ReferenceComposition
from imagine_colorization.utils.vision import normalize_uint8, to_uint8


@dataclass
class ColorizationOutputs:
    """Outputs from the colorization stage."""

    colorized: np.ndarray
    low_level_features: np.ndarray
    reference_features: np.ndarray


class ColorizationModel:
    """Decode grayscale inputs into color using a refined reference."""

    def __init__(self, config: ColorizationConfig) -> None:
        self.config = config

    def encode_grayscale(self, grayscale: np.ndarray) -> np.ndarray:
        """Encode grayscale input to multi-scale features."""

        gray = normalize_uint8(grayscale)
        # Placeholder: single-scale features.
        return gray[..., None]

    def encode_reference(self, reference: ReferenceComposition) -> np.ndarray:
        """Encode the refined reference for cross-attention or fusion."""

        ref = normalize_uint8(reference.image)
        return ref

    def fuse(self, gray_features: np.ndarray, reference_features: np.ndarray) -> np.ndarray:
        """Fuse grayscale and reference features."""

        if self.config.use_cross_attention:
            # Placeholder: concatenate along channel axis.
            return np.concatenate([gray_features, reference_features], axis=-1)
        return gray_features

    def decode(self, fused_features: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
        """Decode fused features back to an RGB image."""

        # Placeholder: tile the luminance and blend simple chroma hints.
        height, width = output_shape
        luminance = fused_features[..., 0]
        rgb = np.repeat(luminance[..., None], 3, axis=-1)
        return to_uint8(rgb)

    def __call__(self, sample: ColorizationSample, composition: ReferenceComposition) -> ColorizationOutputs:
        gray_features = self.encode_grayscale(sample.grayscale)
        reference_features = self.encode_reference(composition)
        fused = self.fuse(gray_features, reference_features)
        colorized = self.decode(fused, output_shape=sample.grayscale.shape[:2])
        return ColorizationOutputs(
            colorized=colorized,
            low_level_features=gray_features,
            reference_features=reference_features,
        )
