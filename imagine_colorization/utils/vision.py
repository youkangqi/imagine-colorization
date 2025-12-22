"""Utility helpers for image normalization and resizing."""

from typing import Tuple

import numpy as np


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Convert uint8 images to float32 in [0, 1].

    Args:
        image: HxWxC array, typically uint8.

    Returns:
        Float32 array with values scaled to [0, 1].
    """

    if image.dtype == np.uint8:
        return image.astype("float32") / 255.0
    return image.astype("float32")


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float images in [0, 1] to uint8."""

    image = np.clip(image, 0.0, 1.0)
    return (image * 255).round().astype("uint8")


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image using a simple nearest-neighbor fallback.

    This placeholder keeps the skeleton lightweight. Substitute with a
    high-quality resize (e.g., OpenCV, PIL) in production code.
    """

    height, width = size
    y_indices = (np.linspace(0, image.shape[0] - 1, height)).astype(int)
    x_indices = (np.linspace(0, image.shape[1] - 1, width)).astype(int)
    return image[np.ix_(y_indices, x_indices)]
