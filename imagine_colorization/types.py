"""Shared type definitions for the Imagine-Colorization pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ColorizationSample:
    """Container for grayscale inputs and derived metadata."""

    grayscale: np.ndarray
    caption: Optional[str] = None
    depth_map: Optional[np.ndarray] = None
    segmentation: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceCandidate:
    """A single colorful reference synthesized by the imagination module."""

    image: np.ndarray
    caption: str
    latent: Optional[Any] = None
    score: Optional[float] = None
    control_image: Optional[np.ndarray] = None
    seed: Optional[int] = None


@dataclass
class ReferenceComposition:
    """Composite reference assembled by the refinement module."""

    image: np.ndarray
    mask: np.ndarray
    provenance: Dict[str, int]
