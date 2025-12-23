"""Package skeleton for the Imagine-Colorization pipeline.

This package mirrors the high-level components described in the paper
“Automatic Controllable Colorization via Imagination”. The modules expose
clean, testable interfaces without binding to specific deep learning
frameworks so that concrete implementations can be slotted in later.
"""

from imagine_colorization.pipeline import ImagineColorizationPipeline
from imagine_colorization.config import (
    Blip2Config,
    ColorizationConfig,
    ControlNetConfig,
    DinoV2Config,
    ImaginationConfig,
    PipelineConfig,
    RefinementConfig,
    Sam2Config,
    SemanticSamConfig,
)
from imagine_colorization.types import (
    ColorizationSample,
    ReferenceCandidate,
    ReferenceComposition,
)

__all__ = [
    "ImagineColorizationPipeline",
    "Blip2Config",
    "ColorizationConfig",
    "ControlNetConfig",
    "DinoV2Config",
    "ImaginationConfig",
    "PipelineConfig",
    "RefinementConfig",
    "Sam2Config",
    "SemanticSamConfig",
    "ColorizationSample",
    "ReferenceCandidate",
    "ReferenceComposition",
]
