"""Configuration dataclasses for the Imagine-Colorization pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ImaginationConfig:
    """Settings for the imagination module.

    Attributes:
        captioner_name: Identifier of the text captioning model used to
            describe the grayscale input.
        generator_name: Identifier of the diffusion or GAN backbone used to
            synthesize colorful references.
        num_candidates: Number of colorful reference candidates to generate for
            each grayscale input.
        guidance_scale: Strength of classifier-free guidance (or analogous
            control parameter) applied during sampling.
        seed: Optional seed for deterministic sampling.
    """

    captioner_name: str = "blip-base"
    generator_name: str = "stable-diffusion-xl"
    num_candidates: int = 8
    guidance_scale: float = 7.5
    seed: Optional[int] = None


@dataclass
class RefinementConfig:
    """Settings for reference refinement and composition.

    Attributes:
        mask_blur: Gaussian blur radius applied to soft masks to reduce seams.
        alignment_reg_weight: Regularization weight when estimating structural
            alignment (e.g., optical flow or deformable attention).
        clip_weight: Weight of semantic similarity when ranking segments.
        depth_weight: Weight of structural consistency (depth/edges) when
            ranking segments.
        max_segments: Maximum number of segments to compose into the final
            reference.
    """

    mask_blur: float = 1.5
    alignment_reg_weight: float = 0.1
    clip_weight: float = 0.5
    depth_weight: float = 0.5
    max_segments: int = 5


@dataclass
class ColorizationConfig:
    """Settings for the reference-guided colorization network.

    Attributes:
        backbone: Identifier for the base UNet/Transformer backbone.
        feature_channels: Channel dimensions for multi-scale fusion.
        use_cross_attention: Whether to inject reference features via
            cross-attention (recommended by the paper for controllability).
        use_segmentation_prior: Whether to use instance-aware priors during
            decoding to preserve object boundaries.
    """

    backbone: str = "unet-base"
    feature_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_cross_attention: bool = True
    use_segmentation_prior: bool = True


@dataclass
class PipelineConfig:
    """Top-level configuration for the Imagine-Colorization pipeline."""

    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    colorization: ColorizationConfig = field(default_factory=ColorizationConfig)
