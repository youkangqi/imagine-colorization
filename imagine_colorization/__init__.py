"""Package skeleton for the Imagine-Colorization pipeline.

This package mirrors the high-level components described in the paper
“Automatic Controllable Colorization via Imagination”. The modules expose
clean, testable interfaces without binding to specific deep learning
frameworks so that concrete implementations can be slotted in later.
"""

from imagine_colorization.pipeline import ImagineColorizationPipeline
from imagine_colorization.config import (
    ColorizationConfig,
    ControlNetConfig,
    ImaginationConfig,
    PipelineConfig,
    RefinementConfig,
    Sam2Config,
)
from imagine_colorization.types import (
    ColorizationSample,
    ReferenceCandidate,
    ReferenceComposition,
)

__all__ = [
    "ImagineColorizationPipeline",
    "ColorizationConfig",
    "ControlNetConfig",
    "ImaginationConfig",
    "PipelineConfig",
    "RefinementConfig",
    "Sam2Config",
    "ColorizationSample",
    "ReferenceCandidate",
    "ReferenceComposition",
]
